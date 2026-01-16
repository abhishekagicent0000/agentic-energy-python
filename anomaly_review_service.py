import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from snowflake.connector import connect as snowflake_connect
import psycopg2 
import openai
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnomalyReviewService')

# Constants
OIL_PRICE = 75.00
WATER_DISPOSAL_COST = 1.50
GAS_PRICE = 2.50

DETECTION_WINDOW_DAYS = 7
TRAINING_CUTOFF_DAYS = 7
ROLLING_BASELINE_DAYS = 90
STABILITY_THRESHOLD_DAYS = 30

ALL_SENSORS = [
    'strokes_per_minute', 
    'torque', 
    'pump_fillage', 
    'tubing_pressure', 
    'pump_intake_pressure', 
    'casing_pressure', 
    'wellhead_temp',
    'motor_current', 
    'discharge_pressure', 
    'intake_fluid_temp', 
    'injection_rate', 
    'injection_pressure',
    'surface_pressure'
]

def get_snowflake_conn():
    return snowflake_connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )

def get_db_url():
    url = os.getenv("DATABASE_URL")
    if url and "schema=" in url:
        return url.replace("?schema=public", "").replace("&schema=public", "")
    return url

def get_openai_client():
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

def fetch_well_data(well_id: str, days: int = None) -> pd.DataFrame:
    sensor_selects = ", ".join([f"s.{col}" for col in ALL_SENSORS])
    
    if days:
        where_clause = f"AND s.timestamp >= DATEADD(day, -{days}, CURRENT_DATE())"
    else:
        where_clause = ""
    
    query = f"""
    SELECT 
        s.well_id, s.timestamp, s.lift_type,
        {sensor_selects},
        p.oil_volume, p.water_volume, p.gas_volume
    FROM well_sensor_readings s
    LEFT JOIN well_daily_production p 
        ON s.well_id = p.well_id AND DATE(s.timestamp) = p.date
    WHERE s.well_id = %s
      {where_clause}
    ORDER BY s.timestamp ASC
    """
    
    try:
        with get_snowflake_conn() as conn:
            df = pd.read_sql(query, conn, params=(well_id,))
            
        df.columns = [c.lower() for c in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        sensor_cols = [c.lower() for c in ALL_SENSORS]
        df[sensor_cols] = df[sensor_cols].ffill().fillna(0)
        
        prod_cols = ['oil_volume', 'water_volume', 'gas_volume']
        df[prod_cols] = df[prod_cols].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Fetch failed for {well_id}: {e}")
        return pd.DataFrame()

def calculate_rolling_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['oil_baseline_90d'] = df['oil_volume'].rolling(ROLLING_BASELINE_DAYS, min_periods=30).mean()
    df['water_baseline_90d'] = df['water_volume'].rolling(ROLLING_BASELINE_DAYS, min_periods=30).mean()
    df['gas_baseline_90d'] = df['gas_volume'].rolling(ROLLING_BASELINE_DAYS, min_periods=30).mean()
    
    df['oil_std_90d'] = df['oil_volume'].rolling(ROLLING_BASELINE_DAYS, min_periods=30).std()
    df['water_std_90d'] = df['water_volume'].rolling(ROLLING_BASELINE_DAYS, min_periods=30).std()
    
    df['oil_avg_30d'] = df['oil_volume'].rolling(30, min_periods=5).mean()
    df['water_avg_30d'] = df['water_volume'].rolling(30, min_periods=5).mean()
    df['gas_avg_30d'] = df['gas_volume'].rolling(30, min_periods=5).mean()
    
    df['oil_avg_7d'] = df['oil_volume'].rolling(7, min_periods=3).mean()
    
    return df

def detect_new_normal(df: pd.DataFrame, metric: str = 'oil_volume') -> pd.DataFrame:
    df = df.copy()
    
    recent_30d = df[metric].tail(STABILITY_THRESHOLD_DAYS)
    
    if len(recent_30d) >= STABILITY_THRESHOLD_DAYS:
        recent_mean = recent_30d.mean()
        recent_std = recent_30d.std()
        
        cv = (recent_std / recent_mean) if recent_mean > 0 else 1.0
        
        df['is_stable_production'] = cv < 0.15
        df['stable_production_level'] = recent_mean if cv < 0.15 else None
    else:
        df['is_stable_production'] = False
        df['stable_production_level'] = None
    
    return df

def get_or_train_model(well_id: str, df: pd.DataFrame) -> Dict:
    available_features = [f for f in ALL_SENSORS if f in df.columns and df[f].std() > 0]
    
    if len(df) < 30 or not available_features: 
        return {}
    
    cutoff_date = df['timestamp'].max() - timedelta(days=TRAINING_CUTOFF_DAYS)
    train_df = df[df['timestamp'] <= cutoff_date]
    train_df = train_df[train_df['oil_volume'] > 0]
    
    if len(train_df) < 30: 
        return {}

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    
    try:
        X_train = imputer.fit_transform(train_df[available_features])
        X_train = scaler.fit_transform(X_train)
        model.fit(X_train, train_df['oil_volume'])
        
        return {
            'model': model, 
            'scaler': scaler, 
            'imputer': imputer, 
            'features': available_features
        }
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return {}

# =============================================================================
# ANOMALY CHECKERS (With Enhanced Numerical Context)
# =============================================================================

def check_operational_pattern_shift(ctx: Dict) -> Optional[Dict]:
    if ctx['run_pct_30d'] > 0.1:
        diff = abs(ctx['run_pct_7d'] - ctx['run_pct_30d'])
        if diff > 0.25:
            direction = "Decreased" if ctx['run_pct_7d'] < ctx['run_pct_30d'] else "Increased"
            
            data_summary = f"""
            Well Runtime Pattern Change Detected:
            - Historical 30-day Runtime: {ctx['run_pct_30d']*100:.1f}%
            - Recent 7-day Runtime: {ctx['run_pct_7d']*100:.1f}%
            - Change: {direction} by {diff*100:.1f} percentage points
            """
            
            return {
                "code": "PATTERN_SHIFT",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity": "Low",
                "impact_value": ctx['actual_oil'] * OIL_PRICE * 0.1,
                "chart_metric": "strokes_per_minute"
            }
    return None

def check_production_volatility(ctx: Dict) -> Optional[Dict]:
    if ctx['oil_std_7d'] > 0 and ctx['oil_avg_7d'] > 5:
        cv = ctx['oil_std_7d'] / ctx['oil_avg_7d']
        
        if cv > 0.30:
            data_summary = f"""
            Erratic Production Pattern Detected:
            - 7-day Average Production: {ctx['oil_avg_7d']:.1f} bbl/day
            - 7-day Volatility (StdDev): {ctx['oil_std_7d']:.1f} bbl/day
            - Coefficient of Variation: {cv*100:.1f}% (Threshold is 30%)
            - Range: {ctx['oil_min_7d']:.1f} to {ctx['oil_max_7d']:.1f} bbl/day
            """
            
            return {
                "code": "PRODUCTION_VOLATILITY",
                "category": "PRODUCTION",
                "raw_data": data_summary,
                "severity": "Moderate",
                "impact_value": ctx['actual_oil'] * OIL_PRICE * 0.1,
                "chart_metric": "oil_volatility"
            }
    return None

def check_week_over_week_variance(ctx: Dict) -> Optional[Dict]:
    if ctx['oil_avg_7d'] > 0 and ctx['oil_avg_14d'] > 0:
        wow_change = ((ctx['oil_avg_7d'] - ctx['oil_avg_14d']) / ctx['oil_avg_14d']) * 100
        
        if abs(wow_change) > 20:
            direction = "increased" if wow_change > 0 else "decreased"
            
            data_summary = f"""
            Significant Week-over-Week Production Change:
            - Last Week (Days 8-14) Avg: {ctx['oil_avg_14d']:.1f} bbl/day
            - This Week (Days 1-7) Avg: {ctx['oil_avg_7d']:.1f} bbl/day
            - Week-over-Week Change: {wow_change:+.1f}% ({abs(ctx['oil_avg_7d'] - ctx['oil_avg_14d']):.1f} bbl/day difference)
            """
            
            return {
                "code": "WEEK_OVER_WEEK",
                "category": "PRODUCTION",
                "raw_data": data_summary,
                "severity": "Moderate",
                "impact_value": abs(ctx['oil_avg_7d'] - ctx['oil_avg_14d']) * OIL_PRICE,
                "chart_metric": "weekly_comparison"
            }
    return None

def check_sustained_production_change(ctx: Dict) -> Optional[Dict]:
    if not ctx.get('is_stable_production', False):
        return None
    
    stable_level = ctx.get('stable_production_level', 0)
    baseline_90d = ctx.get('oil_baseline_90d', 0)
    
    if stable_level > 0 and baseline_90d > 0:
        change_pct = ((stable_level - baseline_90d) / baseline_90d) * 100
        
        if abs(change_pct) > 20:
            direction = "increased to" if change_pct > 0 else "decreased to"
            impact = (baseline_90d - stable_level) * OIL_PRICE if change_pct < 0 else 0
            
            data_summary = f"""
            Production Stabilized at New Level:
            - Historical 90-day Baseline: {baseline_90d:.1f} bbl/day
            - New Stable Production Level: {stable_level:.1f} bbl/day
            - Change: {direction} by {abs(change_pct):.1f}%
            """
            
            return {
                "code": "NEW_PRODUCTION_BASELINE",
                "category": "PRODUCTION",
                "raw_data": data_summary,
                "severity": "Low",
                "impact_value": impact,
                "chart_metric": "baseline_shift"
            }
    return None

def check_gor_shift(ctx: Dict) -> Optional[Dict]:
    if ctx['actual_oil'] > 0 and ctx['actual_gas'] > 0:
        current_gor = ctx['actual_gas'] / ctx['actual_oil']
        avg_gor = ctx.get('gor_avg_30d', 0)
        
        if avg_gor > 0:
            gor_change_pct = ((current_gor - avg_gor) / avg_gor) * 100
            
            if abs(gor_change_pct) > 30:
                interpretation = "gas breakthrough" if gor_change_pct > 0 else "reduced gas production"
                
                data_summary = f"""
                Gas-Oil Ratio Shift Detected:
                - Current GOR: {current_gor:.0f} MCF/barrel
                - Historical Average GOR: {avg_gor:.0f} MCF/barrel
                - GOR Change: {gor_change_pct:+.1f}%
                """
                
                return {
                    "code": "GOR_SHIFT",
                    "category": "RESERVOIR",
                    "raw_data": data_summary,
                    "severity": "Moderate",
                    "impact_value": ctx['actual_oil'] * OIL_PRICE * 0.1,
                    "chart_metric": "gor_trend"
                }
    return None

def check_efficiency_degradation(ctx: Dict) -> Optional[Dict]:
    if ctx['pred_oil'] > 5 and ctx['actual_oil'] > 0:
        current_efficiency = (ctx['actual_oil'] / ctx['pred_oil']) * 100
        avg_efficiency = ctx.get('efficiency_avg_30d', 100)
        
        if current_efficiency < (avg_efficiency - 15):
            impact = (ctx['pred_oil'] - ctx['actual_oil']) * OIL_PRICE
            
            # Explicit logic explanation for the user
            data_summary = f"""
            Production Efficiency Degradation Detected:
            - Current Efficiency: {current_efficiency:.1f}% (Historical Avg: {avg_efficiency:.1f}%)
            - Model Prediction: {ctx['pred_oil']:.1f} bbl/day
            - Actual Production: {ctx['actual_oil']:.1f} bbl/day
            - Efficiency Drop: {avg_efficiency - current_efficiency:.1f} percentage points
            - Interpretation: Well is underperforming by {ctx['pred_oil'] - ctx['actual_oil']:.1f} bbl/day relative to energy inputs.
            """
            
            return {
                "code": "EFFICIENCY_DEGRADATION",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity": "Moderate",
                "impact_value": impact,
                "chart_metric": "efficiency_trend"
            }
    return None

def check_pressure_instability(ctx: Dict) -> Optional[Dict]:
    if ctx['tp_avg'] > 50:
        if ctx['tp_std'] > (0.2 * ctx['tp_avg']):
            volatility_pct = (ctx['tp_std'] / ctx['tp_avg']) * 100
            
            data_summary = f"""
            Pressure Instability Detected:
            - Current Tubing Pressure: {ctx['tubing_pressure']:.1f} psi
            - 30-Day Average Pressure: {ctx['tp_avg']:.1f} psi
            - Pressure Volatility (StdDev): {ctx['tp_std']:.1f} psi
            - Instability Level: {volatility_pct:.1f}% (Threshold is 20%)
            """
            
            return {
                "code": "PRESSURE_INSTABILITY",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity": "Moderate",
                "impact_value": ctx['actual_oil'] * OIL_PRICE * 0.1,
                "chart_metric": "tubing_pressure"
            }
    return None

def check_financial_gap(ctx: Dict) -> Optional[Dict]:
    if ctx.get('is_stable_production', False):
        return None
    
    if ctx['is_active'] and (ctx['pred_oil'] > 5) and (ctx['actual_oil'] < (0.75 * ctx['pred_oil'])):
        gap = ctx['pred_oil'] - ctx['actual_oil']
        impact = gap * OIL_PRICE
        
        data_summary = f"""
        Production Efficiency Gap Detected:
        - Model Predicted: {ctx['pred_oil']:.1f} bbl/day
        - Actual: {ctx['actual_oil']:.1f} bbl/day
        - Gap: {gap:.1f} bbl/day ({((1 - ctx['actual_oil']/ctx['pred_oil'])*100):.1f}% below potential)
        - Daily Revenue Loss: ${impact:.2f}
        """
        
        return {
            "code": "FINANCIAL_EFFICIENCY",
            "category": "FINANCIAL",
            "raw_data": data_summary,
            "severity": "High" if impact > 500 else "Moderate",
            "impact_value": impact,
            "chart_metric": "oil_comparison"
        }
    return None

def check_ghost_production(ctx: Dict) -> Optional[Dict]:
    """Detect missing production reports with Data Lag protection."""
    if 'timestamp' in ctx:
        row_date = ctx['timestamp'].date()
        current_date = datetime.now().date()
        days_old = (current_date - row_date).days
        if days_old < 2: return None # Skip recent data due to lag
            
    total = ctx['actual_oil'] + ctx['actual_water'] + ctx['actual_gas']
    if ctx['is_active'] and (total < 0.1):
        impact = ctx['avg_oil'] * OIL_PRICE if ctx['avg_oil'] > 0 else 100.0
        
        data_summary = f"""
        Missing Production Report (Ghost Production):
        - Sensors indicate well is ACTIVE (Amps/SPM present)
        - Reported Production is 0.
        - Historical Average: {ctx['avg_oil']:.1f} bbl/day
        """
        
        return {
            "code": "GHOST_PROD",
            "category": "PROCESS",
            "raw_data": data_summary,
            "severity": "Low",
            "impact_value": impact,
            "chart_metric": "production_bar"
        }
    return None

def check_cost_creep(ctx: Dict) -> Optional[Dict]:
    if (ctx['avg_water'] > 10) and (ctx['actual_water'] > (1.25 * ctx['avg_water'])) and (ctx['actual_oil'] <= ctx['avg_oil']):
        excess_water = ctx['actual_water'] - ctx['avg_water']
        cost = excess_water * WATER_DISPOSAL_COST
        increase_pct = ((ctx['actual_water']/ctx['avg_water'])-1)*100
        
        data_summary = f"""
        Rising Water Disposal Costs Detected:
        - Current Water: {ctx['actual_water']:.1f} bbl/day
        - Average Water: {ctx['avg_water']:.1f} bbl/day
        - Increase: {excess_water:.1f} bbl/day ({increase_pct:.1f}%)
        - Daily Disposal Cost Increase: ${cost:.2f}
        """
        
        return {
            "code": "COST_CREEP",
            "category": "FINANCIAL",
            "raw_data": data_summary,
            "severity": "Moderate",
            "impact_value": cost,
            "chart_metric": "water_trend"
        }
    return None

def check_bsw_spike(ctx: Dict) -> Optional[Dict]:
    if (ctx['bsw'] > 10) and (ctx['bsw'] > ctx['bsw_avg'] + 20):
        bsw_increase = ctx['bsw'] - ctx['bsw_avg']
        
        data_summary = f"""
        Water Cut (BSW) Spike Detected:
        - Current BSW: {ctx['bsw']:.1f}%
        - Historical Average: {ctx['bsw_avg']:.1f}%
        - Increase: {bsw_increase:.1f} percentage points
        """
        
        return {
            "code": "BSW_SPIKE",
            "category": "OPERATIONAL",
            "raw_data": data_summary,
            "severity": "Moderate",
            "impact_value": ctx['actual_oil'] * OIL_PRICE * 0.1,
            "chart_metric": "water_trend"
        }
    return None

def check_monthly_production_decline(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    recent_df = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=90))].copy()
    recent_df['month'] = recent_df['timestamp'].dt.to_period('M')
    
    monthly = recent_df.groupby('month').agg({
        'oil_volume': ['sum', 'count']
    }).reset_index()
    monthly.columns = ['month', 'oil_volume', 'days_count']
    monthly['oil_rate'] = monthly['oil_volume'] / monthly['days_count']
    
    if len(monthly) >= 2:
        latest = monthly.iloc[-1]
        previous = monthly.iloc[-2]
        
        if latest['days_count'] < 3: return None

        if previous['oil_rate'] > 5:
            decline_pct = ((previous['oil_rate'] - latest['oil_rate']) / previous['oil_rate']) * 100
            
            if decline_pct > 15:
                daily_loss = previous['oil_rate'] - latest['oil_rate']
                revenue_impact = daily_loss * OIL_PRICE
                
                data_summary = f"""
                Significant Monthly Production Rate Decline:
                - Previous Month Avg Rate: {previous['oil_rate']:.1f} bbl/day
                - Current Month Avg Rate: {latest['oil_rate']:.1f} bbl/day
                - Decline: -{decline_pct:.1f}%
                - Daily Oil Loss Rate: {daily_loss:.1f} bbl/day
                """
                
                return {
                    "code": "MONTHLY_DECLINE",
                    "category": "PRODUCTION",
                    "raw_data": data_summary,
                    "severity": "High" if decline_pct > 25 else "Moderate",
                    "impact_value": revenue_impact,
                    "chart_metric": "monthly_production"
                }
    return None

def generate_llm_content(anomaly_data: str, lift_type: str, chart_metric: str) -> Dict:
    try:
        client = get_openai_client()
        if not client:
            raise Exception("No OpenAI Client")
        
        # KEY CHANGE: STRICTER PROMPT INSTRUCTIONS FOR DATA
        prompt = f"""
Analyze this oil & gas well anomaly for a {lift_type} well.

ANOMALY DATA:
{anomaly_data}

Generate a comprehensive JSON response with these exact keys:

1. "title": A specific title (8-12 words) containing key metrics(like money, production, efficiency etc) if possible.
2. "description": A detailed 2-3 sentence explanation. 
   - **MANDATORY**: You MUST explicitly quote the numeric values, dates, and percentages provided in the ANOMALY DATA.
   - Example: "On Jan 15, efficiency dropped to 74% against a historical average of 90%."
   - Do NOT use generic phrases like "significant drop" without backing it up with the specific number found in the data.
3. "why_is_this_an_anomaly": Explain the deviation using the numbers (e.g., "Current pressure of 600 psi is 50% below the historical average").
4. "suspected_root_cause": Provide 2-3 technical causes.
5. "economic_impact": A clear statement of financial impact.

Return ONLY valid JSON.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM content generation failed: {e}")
        return {
            "title": "Well Performance Anomaly Detected",
            "description": "An unusual pattern has been detected in well operations.",
            "why_is_this_an_anomaly": "The detected pattern deviates from normal parameters.",
            "suspected_root_cause": "Multiple factors could be contributing.",
            "economic_impact": "Review required."
        }

def generate_chart_config(anomaly_data: str, chart_metric: str, df: pd.DataFrame) -> Dict:
    try:
        client = get_openai_client()
        if not client:
            raise Exception("No OpenAI Client")
        
        # 1. Aggregate Data by Day (Clean inputs)
        recent_df = df.tail(60).copy()
        recent_df['date_only'] = recent_df['timestamp'].dt.date
        
        daily_df = recent_df.groupby('date_only').agg({
            'oil_volume': 'mean',
            'water_volume': 'mean',
            'gas_volume': 'mean',
            'tubing_pressure': 'mean',
            'strokes_per_minute': 'mean',
            'predicted_oil': 'mean' if 'predicted_oil' in df.columns else 'mean'
        }).reset_index()
        
        # 2. Define High-Quality Visual Strategy
        chart_strategy = ""
        chart_data_context = ""
        
        if chart_metric == "oil_comparison":
            # STRATEGY: Grouped Bar Chart (Actual vs Predicted)
            # Perfect for showing the "Gap"
            chart_df = daily_df.tail(10) # Last 10 days
            dates = [d.strftime('%b %d') for d in chart_df['date_only']]
            chart_strategy = "Create a GROUPED BAR CHART. Dataset 1: Actual Oil (Blue #3b82f6). Dataset 2: Predicted Oil (Purple #8b5cf6). This visually proves the efficiency gap."
            chart_data_context = f"""
            Labels: {dates}
            Actual Oil: {chart_df['oil_volume'].round(1).tolist()}
            Predicted Oil: {chart_df['predicted_oil'].round(1).tolist()}
            """
            
        elif chart_metric == "tubing_pressure":
            # STRATEGY: Dual-Axis Line Chart (Pressure vs Oil)
            # Solves the "1500 vs 200" scale problem perfectly
            chart_df = daily_df.tail(14)
            dates = [d.strftime('%b %d') for d in chart_df['date_only']]
            chart_strategy = """
            Create a DUAL Y-AXIS LINE CHART. 
            - Dataset 1: Tubing Pressure (Line, Purple #8b5cf6, yAxisID='y').
            - Dataset 2: Oil Production (Line, Blue #3b82f6, yAxisID='y1').
            - CRITICAL: You MUST define both 'y' (left) and 'y1' (right) scales in the options.
            """
            chart_data_context = f"""
            Labels: {dates}
            Tubing Pressure (psi): {chart_df['tubing_pressure'].round(1).tolist()}
            Oil Production (bbl/day): {chart_df['oil_volume'].round(1).tolist()}
            """

        elif chart_metric == "monthly_production":
            # STRATEGY: Simple Bar Chart (Monthly Totals)
            # Best for showing "Step Down" decline
            monthly_df = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=120))].copy()
            monthly_df['month'] = monthly_df['timestamp'].dt.to_period('M')
            monthly = monthly_df.groupby('month').agg({'oil_volume': ['sum', 'count']}).reset_index()
            monthly.columns = ['month', 'oil_total', 'days']
            monthly['oil_rate'] = (monthly['oil_total'] / monthly['days']).round(1)
            
            chart_strategy = "Create a SIMPLE BAR CHART showing Avg Daily Production per Month. Color: Blue #3b82f6."
            chart_data_context = f"""
            Labels: {[str(m) for m in monthly['month'].tolist()]}
            Avg Oil Rate (bbl/day): {monthly['oil_rate'].tolist()}
            """

        elif chart_metric == "efficiency_trend":
             # STRATEGY: Area Chart (Efficiency %)
            chart_df = daily_df.tail(14)
            dates = [d.strftime('%b %d') for d in chart_df['date_only']]
            eff = ((chart_df['oil_volume'] / chart_df['predicted_oil']) * 100).fillna(0).round(1).tolist()
            
            chart_strategy = "Create a LINE CHART with fill (Area Chart) showing Efficiency %. Color: Purple #8b5cf6."
            chart_data_context = f"""
            Labels: {dates}
            Efficiency %: {eff}
            """
            
        else:
            # Fallback: Simple Line
            chart_df = daily_df.tail(14)
            dates = [d.strftime('%b %d') for d in chart_df['date_only']]
            chart_strategy = "Create a Simple Line Chart showing Oil Production."
            chart_data_context = f"""
            Labels: {dates}
            Oil: {chart_df['oil_volume'].round(1).tolist()}
            """

        # 3. Enhanced Prompt to prevent UI Breakage
        prompt = f"""
You are a Data Visualization Expert. Generate a Chart.js (v4) JSON config.

GOAL: {chart_strategy}

DATA:
{chart_data_context}

CRITICAL RULES FOR VALID JSON:
1. Output ONLY valid JSON. No markdown formatting.
2. Use this color palette: ["#8b5cf6", "#6366f1", "#3b82f6", "#06b6d4"].
3. FOR DUAL AXIS CHARTS (tubing_pressure), you MUST use this structure for options:
   "options": {{
     "responsive": true,
     "scales": {{
       "y": {{ "type": "linear", "display": true, "position": "left", "title": {{ "display": true, "text": "Pressure (psi)" }} }},
       "y1": {{ "type": "linear", "display": true, "position": "right", "grid": {{ "drawOnChartArea": false }}, "title": {{ "display": true, "text": "Oil (bbl)" }} }}
     }}
   }}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"LLM chart generation failed: {e}")
        return create_fallback_chart(df.tail(14), chart_metric)

def create_fallback_chart(df: pd.DataFrame, chart_metric: str) -> Dict:
    if df.empty: return {"error": "No data"}
    chart_df = df[df['oil_volume'] > 0] if len(df[df['oil_volume'] > 0]) > 0 else df
    labels = chart_df['timestamp'].dt.strftime('%b %d').tolist()
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Oil Production",
                "data": chart_df['oil_volume'].round(1).tolist(),
                "borderColor": "#8b5cf6",
                "fill": False
            }]
        },
        "options": {"responsive": True, "plugins": {"legend": {"display": True}}}
    }

def detect_anomalies(well_id: str, df: Optional[pd.DataFrame] = None, lookback_days: int = DETECTION_WINDOW_DAYS) -> List[Dict]:
    logger.info(f"Fetching all historical data for {well_id} to train model...")
    full_df = fetch_well_data(well_id, days=None)
    
    if full_df.empty or len(full_df) < 30:
        logger.warning(f"Insufficient data for {well_id}")
        return []

    full_df['lift_type'] = full_df['lift_type'].ffill().fillna('Rod Pump')
    
    logger.info("Calculating rolling baselines...")
    full_df = calculate_rolling_baseline(full_df)
    full_df = detect_new_normal(full_df, 'oil_volume')
    
    full_df['oil_avg'] = full_df['oil_avg_30d']
    full_df['oil_std_7d'] = full_df['oil_volume'].rolling(7, min_periods=3).std()
    full_df['oil_avg_14d'] = full_df['oil_volume'].rolling(14, min_periods=7).mean()
    full_df['oil_min_7d'] = full_df['oil_volume'].rolling(7, min_periods=3).min()
    full_df['oil_max_7d'] = full_df['oil_volume'].rolling(7, min_periods=3).max()
    full_df['gor'] = full_df['gas_volume'] / full_df['oil_volume'].replace(0, np.nan)
    full_df['gor_avg_30d'] = full_df['gor'].rolling(30, min_periods=10).mean()
    
    if 'tubing_pressure' in full_df: 
        full_df['tp_avg'] = full_df['tubing_pressure'].rolling(30).mean()
        full_df['tp_std'] = full_df['tubing_pressure'].rolling(7).std()
    
    if 'motor_current' in full_df:
        full_df['amps_avg'] = full_df['motor_current'].rolling(30).mean()
        
    if 'strokes_per_minute' in full_df: 
        full_df['spm_avg'] = full_df['strokes_per_minute'].rolling(30).mean()
        full_df['is_running'] = (full_df['strokes_per_minute'] > 0.1).astype(int)
        full_df['run_pct_7d'] = full_df['is_running'].rolling(7).mean()
        full_df['run_pct_30d'] = full_df['is_running'].rolling(30).mean()

    total_fluid = full_df['oil_volume'] + full_df['water_volume']
    full_df['bsw'] = (full_df['water_volume'] / total_fluid.replace(0, np.nan)) * 100
    full_df['bsw'] = full_df['bsw'].fillna(0)
    full_df['bsw_avg'] = full_df['bsw'].rolling(30).mean()

    logger.info(f"Training predictive model on historical records (excluding last {TRAINING_CUTOFF_DAYS} days)...")
    model_art = get_or_train_model(well_id, full_df)
    full_df['predicted_oil'] = np.nan
    
    if model_art:
        try:
            X = model_art['scaler'].transform(
                model_art['imputer'].transform(full_df[model_art['features']])
            )
            full_df['predicted_oil'] = model_art['model'].predict(X)
            full_df['efficiency'] = (full_df['oil_volume'] / full_df['predicted_oil'].replace(0, np.nan)) * 100
            full_df['efficiency_avg_30d'] = full_df['efficiency'].rolling(30, min_periods=10).mean()
            logger.info("Predictions generated successfully")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")

    all_checkers = [
        check_operational_pattern_shift,
        check_production_volatility,
        check_week_over_week_variance,
        check_sustained_production_change,
        check_gor_shift,
        check_efficiency_degradation,
        check_pressure_instability,
        check_financial_gap,
        check_ghost_production,
        check_cost_creep,
        check_bsw_spike
    ]
    
    logger.info(f"Detecting anomalies in last {lookback_days} days...")
    window = full_df[full_df['timestamp'] > (full_df['timestamp'].max() - pd.Timedelta(days=lookback_days))]
    raw_findings = []
    
    for _, row in window.iterrows():
        ctx = row.to_dict()
        ctx['is_active'] = (ctx.get('strokes_per_minute',0) > 0.1) or (ctx.get('motor_current',0) > 5)
        ctx['activity_source'] = "Sensors"
        ctx['actual_oil'] = ctx.get('oil_volume',0)
        ctx['actual_water'] = ctx.get('water_volume',0)
        ctx['actual_gas'] = ctx.get('gas_volume',0)
        ctx['avg_oil'] = ctx.get('oil_avg_30d', 0)
        ctx['avg_water'] = ctx.get('water_avg_30d', 0)
        ctx['pred_oil'] = ctx.get('predicted_oil', ctx['actual_oil'])
        if pd.isna(ctx['pred_oil']): ctx['pred_oil'] = ctx['actual_oil']
        
        for field in ['tubing_pressure', 'tp_avg', 'tp_std', 'spm', 'spm_avg', 'run_pct_7d', 'run_pct_30d', 'bsw', 'bsw_avg', 'oil_min_7d', 'oil_max_7d', 'oil_avg_7d', 'oil_avg_14d', 'oil_std_7d', 'gor_avg_30d', 'efficiency_avg_30d']:
            ctx[field] = ctx.get(field, 0)
            if pd.isna(ctx[field]): ctx[field] = 0

        for check in all_checkers:
            res = check(ctx)
            if res:
                raw_findings.append({
                    "well_id": well_id,
                    "event_date": str(row['timestamp'].date()),
                    "timestamp": row['timestamp'],
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "anomaly_code": res['code'],
                    "category": res['category'],
                    "severity": res['severity'],
                    "raw_data": res['raw_data'],
                    "chart_metric": res['chart_metric'],
                    "impact_value": res['impact_value'],
                    "lift_type": str(row['lift_type']) if pd.notnull(row['lift_type']) else 'Rod Pump'
                })

    anomalies = []
    findings_by_code = defaultdict(list)
    for f in raw_findings:
        findings_by_code[f['anomaly_code']].append(f)
        
    for code, findings in findings_by_code.items():
        sorted_findings = sorted(findings, key=lambda x: x['timestamp'])
        
        should_aggregate = len(sorted_findings) > 1 and code in [
            "FINANCIAL_EFFICIENCY", "GHOST_PROD", "COST_CREEP", 
            "PRESSURE_INSTABILITY", "PATTERN_SHIFT", "EFFICIENCY_DEGRADATION"
        ]
        
        if should_aggregate:
            latest_finding = sorted_findings[-1]
            first_date = datetime.strptime(sorted_findings[0]['event_date'], '%Y-%m-%d')
            last_date_obj = datetime.strptime(latest_finding['event_date'], '%Y-%m-%d')
            
            duration_days = (last_date_obj - first_date).days + 1
            avg_daily_impact = sum(f['impact_value'] for f in sorted_findings) / len(sorted_findings)
            total_impact = avg_daily_impact * duration_days
            
            aggregated_raw_data = f"""
            PERSISTENT ANOMALY DETECTED ({duration_days} days in window):
            - First Detection: {sorted_findings[0]['event_date']}
            - Last Detection: {latest_finding['event_date']}
            - Est. Cumulative Impact: ${total_impact:.2f} (Daily Avg: ${avg_daily_impact:.2f})
            - Latest Status ({latest_finding['event_date']}):
            {latest_finding['raw_data']}
            """
            
            logger.info(f"Aggregating {len(sorted_findings)} rows into {duration_days} days for {code}...")
            llm_content = generate_llm_content(aggregated_raw_data, latest_finding['lift_type'], latest_finding['chart_metric'])
            chart_config = generate_chart_config(aggregated_raw_data, latest_finding['chart_metric'], full_df)
            
            unit = "USD (Cumulative)"
            if code in ["PATTERN_SHIFT", "PRODUCTION_VOLATILITY", "GOR_SHIFT", "PRESSURE_INSTABILITY", "BSW_SPIKE"]:
                unit = "Revenue at Risk (Cumulative)"
            elif total_impact <= 0:
                unit = "Review Required"

            anomalies.append({
                "well_id": well_id,
                "event_date": latest_finding['event_date'],
                "timestamp": latest_finding['timestamp'],
                "detected_at": latest_finding['detected_at'],
                "anomaly_code": code,
                "category": latest_finding['category'],
                "severity": latest_finding['severity'],
                "title": llm_content.get('title', 'Persistent Anomaly Detected'),
                "status": "ACTIVE",
                "ui_text": {
                    "description": llm_content.get('description', ''),
                    "why_is_this_an_anomaly": llm_content.get('why_is_this_an_anomaly', ''),
                    "suspected_root_cause": llm_content.get('suspected_root_cause', ''),
                    "economic_impact": llm_content.get('economic_impact', '')
                },
                "impact_metrics": {
                    "value": float(total_impact), 
                    "unit": unit
                },
                "chart_data": chart_config
            })
            
        else:
            for finding in sorted_findings:
                logger.info(f"Generating content for {finding['anomaly_code']} on {finding['event_date']}...")
                llm_content = generate_llm_content(finding['raw_data'], finding['lift_type'], finding['chart_metric'])
                chart_config = generate_chart_config(finding['raw_data'], finding['chart_metric'], full_df)
                
                unit = "USD/day"
                if finding['anomaly_code'] in ["PATTERN_SHIFT", "PRODUCTION_VOLATILITY", "GOR_SHIFT", "PRESSURE_INSTABILITY", "BSW_SPIKE"]:
                    unit = "Daily Revenue at Risk"
                elif finding['impact_value'] <= 0:
                    unit = "Review Required"

                anomalies.append({
                    "well_id": well_id,
                    "event_date": finding['event_date'],
                    "timestamp": finding['timestamp'],
                    "detected_at": finding['detected_at'],
                    "anomaly_code": finding['anomaly_code'],
                    "category": finding['category'],
                    "severity": finding['severity'],
                    "title": llm_content.get('title', 'Anomaly Detected'),
                    "status": "ACTIVE",
                    "ui_text": {
                        "description": llm_content.get('description', ''),
                        "why_is_this_an_anomaly": llm_content.get('why_is_this_an_anomaly', ''),
                        "suspected_root_cause": llm_content.get('suspected_root_cause', ''),
                        "economic_impact": llm_content.get('economic_impact', '')
                    },
                    "impact_metrics": {
                        "value": float(finding['impact_value']), 
                        "unit": unit
                    },
                    "chart_data": chart_config
                })

    if not window.empty:
        last_row = window.iloc[-1]
        ctx = last_row.to_dict()
        ctx['lift_type'] = str(last_row['lift_type']) if pd.notnull(last_row['lift_type']) else 'Rod Pump'
        monthly_check = check_monthly_production_decline(ctx, full_df)
        
        if monthly_check:
            if not any(x['anomaly_code'] == monthly_check['code'] for x in anomalies):
                llm_content = generate_llm_content(monthly_check['raw_data'], ctx['lift_type'], monthly_check['chart_metric'])
                chart_config = generate_chart_config(monthly_check['raw_data'], monthly_check['chart_metric'], full_df)
                
                anomalies.append({
                    "well_id": well_id,
                    "event_date": str(last_row['timestamp'].date()),
                    "timestamp": last_row['timestamp'],
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "anomaly_code": monthly_check['code'],
                    "category": monthly_check['category'],
                    "severity": monthly_check['severity'],
                    "title": llm_content.get('title', 'Monthly Production Rate Decline'),
                    "status": "ACTIVE",
                    "ui_text": {
                        "description": llm_content.get('description', ''),
                        "why_is_this_an_anomaly": llm_content.get('why_is_this_an_anomaly', ''),
                        "suspected_root_cause": llm_content.get('suspected_root_cause', ''),
                        "economic_impact": llm_content.get('economic_impact', '')
                    },
                    "impact_metrics": {
                        "value": float(monthly_check['impact_value']),
                        "unit": "USD/day"
                    },
                    "chart_data": chart_config
                })
    
    logger.info(f"Detected {len(anomalies)} anomalies for {well_id}")
    save_reviews(anomalies)
    return anomalies

def save_reviews(reviews):
    if not reviews: return
    
    try:
        url = get_db_url()
        if url:
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    q = """
                        INSERT INTO anomaly_review
                        (well_id, event_date, detected_at, anomaly_code, category, severity, 
                         title, ui_text, impact_value, impact_unit, chart_data, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (well_id, anomaly_code, event_date) DO NOTHING
                    """
                    for r in reviews:
                        cur.execute(q, (
                            r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                            r['category'], r['severity'], r['title'],
                            json.dumps(r['ui_text']),
                            r['impact_metrics']['value'],
                            r['impact_metrics']['unit'],
                            json.dumps(r['chart_data']),
                            r['status']
                        ))
                conn.commit()
                logger.info(f"Saved {len(reviews)} anomalies to PostgreSQL")
    except Exception as e:
        logger.error(f"Postgres Save failed: {e}")

    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            check_q = "SELECT count(*) FROM anomaly_review WHERE well_id=%s AND anomaly_code=%s AND event_date=%s"
            
            insert_q = """
                INSERT INTO anomaly_review
                (well_id, event_date, detected_at, anomaly_code, category, severity, 
                 title, ui_text, impact_value, impact_unit, chart_data, status)
                SELECT %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, %s, PARSE_JSON(%s), %s
            """
            
            count = 0
            for r in reviews:
                cur.execute(check_q, (r['well_id'], r['anomaly_code'], r['event_date']))
                if cur.fetchone()[0] == 0:
                    cur.execute(insert_q, (
                        r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                        r['category'], r['severity'], r['title'],
                        json.dumps(r['ui_text']),
                        r['impact_metrics']['value'],
                        r['impact_metrics']['unit'],
                        json.dumps(r['chart_data']),
                        r['status']
                    ))
                    count += 1
            conn.commit()
            if count > 0:
                logger.info(f"Saved {count} anomalies to Snowflake")
    except Exception as e:
        logger.error(f"Snowflake Save failed: {e}")

if __name__ == "__main__":
    import sys
    wid = sys.argv[1] if len(sys.argv) > 1 else "Well_001"
    print(json.dumps(detect_anomalies(wid), indent=2, default=str))