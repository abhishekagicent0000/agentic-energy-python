import os
import json
import logging
import re # Added for regex operations
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

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================
OIL_PRICE = 75.00
WATER_DISPOSAL_COST = 1.50
GAS_PRICE = 2.50  # Added for BOE/Revenue calculation

DETECTION_WINDOW_DAYS = 5
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

SEVERITY_THRESHOLDS = {
    "EFFICIENCY_DEGRADATION": {"high": 30, "moderate": 15},
    "NEW_PRODUCTION_BASELINE": {"high": 40, "moderate": 25},
    "MONTHLY_DECLINE": {"high": 30, "moderate": 20},
    "PRESSURE_INSTABILITY": {"high": 40, "moderate": 22},  # volatility %
    "PATTERN_SHIFT": {"high": 200},
    "GHOST_PROD": {"default": "Moderate"},
}

SENSOR_DISPLAY_NAMES = {
    'oil_volume': 'Oil Volume',
    'water_volume': 'Water Volume',
    'gas_volume': 'Gas Volume',
    'strokes_per_minute': 'Strokes Per Minute (SPM)',
    'torque': 'Torque',
    'pump_fillage': 'Pump Fillage',
    'tubing_pressure': 'Tubing Pressure',
    'pump_intake_pressure': 'Pump Intake Pressure',
    'casing_pressure': 'Casing Pressure',
    'wellhead_temp': 'Wellhead Temperature',
    'motor_current': 'Motor Current',
    'discharge_pressure': 'Discharge Pressure',
    'intake_fluid_temp': 'Intake Fluid Temperature',
    'injection_rate': 'Injection Rate',
    'injection_pressure': 'Injection Pressure',
    'surface_pressure': 'Surface Pressure'
}

# =============================================================================
# DATABASE & API CLIENTS
# =============================================================================

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
    if not url: return None
    if "schema=" in url:
        return url.replace("?schema=public", "").replace("&schema=public", "")
    return url

def get_openai_client():
    return openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )

# =============================================================================
# DATA FETCHING & PROCESSING
# =============================================================================

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
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        sensor_cols = [c.lower() for c in ALL_SENSORS]
        df[sensor_cols] = df[sensor_cols].ffill().fillna(0)
        
        prod_cols = ['oil_volume', 'water_volume', 'gas_volume']
        df[prod_cols] = df[prod_cols].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Fetch failed for {well_id}: {e}")
        return pd.DataFrame()

def get_recent_anomaly_history(well_id: str, anomaly_code: str, exclude_date: str = None) -> str:
    # UPDATED: Added exclude_date logic to filter out the current event
    query = """
    SELECT event_date, severity, title 
    FROM anomaly_review 
    WHERE well_id = %s 
    AND anomaly_code = %s 
    ORDER BY event_date DESC
    LIMIT 5
    """
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            rows = cur.execute(query, (well_id, anomaly_code)).fetchall()
            
        if not rows:
            return "No prior history found."
            
        history_list = []
        for r in rows:
            # Skip if this record matches the current detection date
            if exclude_date and str(r[0]) == str(exclude_date):
                continue
            history_list.append(f"- Date: {r[0]}, Severity: {r[1]}, Title: {r[2]}")
        
        if not history_list:
            return "No prior history found."
            
        return "\n".join(history_list)
    except Exception as e:
        logger.warning(f"History fetch failed: {e}")
        return "History check unavailable."

def calculate_rolling_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Set Index to Timestamp for Time-Based Rolling
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp').sort_index()
    
    # 2. Use Time Strings ('90D', '30D') instead of Integers
    df['oil_baseline_90d'] = df['oil_volume'].rolling('90D').mean()
    df['water_baseline_90d'] = df['water_volume'].rolling('90D').mean()
    df['gas_baseline_90d'] = df['gas_volume'].rolling('90D').mean()
    
    df['oil_std_90d'] = df['oil_volume'].rolling('90D').std()
    
    df['oil_avg_30d'] = df['oil_volume'].rolling('30D').mean()
    df['water_avg_30d'] = df['water_volume'].rolling('30D').mean()
    df['gas_avg_30d'] = df['gas_volume'].rolling('30D').mean()
    
    df['oil_avg_7d'] = df['oil_volume'].rolling('7D').mean()
    df['gas_avg_7d'] = df['gas_volume'].rolling('7D').mean()
    
    # 3. Reset index
    df = df.reset_index()
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
# PHYSICS ENGINE
# =============================================================================

def diagnose_mechanics(ctx: Dict) -> Dict:
    amps = ctx.get('motor_current', 0)
    amps_avg = ctx.get('amps_avg', 0)
    pressure = ctx.get('tubing_pressure', 0)
    pressure_avg = ctx.get('tp_avg', 0)
    spm = ctx.get('strokes_per_minute', 0)
    spm_avg = ctx.get('spm_avg', 0)
    oil = ctx.get('actual_oil', 0)
    pred_oil = ctx.get('pred_oil', 1)
    
    amps_ratio = amps / amps_avg if amps_avg > 0 else 1.0
    pressure_ratio = pressure / pressure_avg if pressure_avg > 0 else 1.0
    spm_ratio = spm / spm_avg if spm_avg > 0 else 1.0
    efficiency = (oil / pred_oil) * 100 if pred_oil > 0 else 100
    
    if amps_ratio < 0.7 and efficiency < 50:
        return {
            'type': 'Parted Rods or Belt Break',
            'confidence': 'High',
            'indicators': f'Motor current dropped to {amps:.1f}A ({amps_ratio*100:.0f}% of normal), production at {efficiency:.0f}% efficiency'
        }
    elif amps_ratio > 1.3 and pressure_ratio > 1.2:
        return {
            'type': 'Stuck Pump or Severe Friction',
            'confidence': 'High',
            'indicators': f'Motor current elevated to {amps:.1f}A ({amps_ratio*100:.0f}% of normal), pressure at {pressure:.1f} psi'
        }
    elif pressure_ratio > 1.4 and efficiency < 60:
        return {
            'type': 'Flowline Blockage',
            'confidence': 'High',
            'indicators': f'Tubing pressure elevated to {pressure:.1f} psi ({pressure_ratio*100:.0f}% of normal)'
        }
    elif amps_ratio > 0.9 and amps_ratio < 1.1 and efficiency < 40:
        return {
            'type': 'Tubing Leak or Pump Wear',
            'confidence': 'Moderate',
            'indicators': f'Normal pump operation (Amps normal) but production collapsed ({efficiency:.0f}% eff)'
        }
    elif spm_ratio < 0.8:
        return {
            'type': 'Controller Fault / Speed Drop',
            'confidence': 'Moderate',
            'indicators': f'Pump speed reduced to {spm:.1f} SPM ({spm_ratio*100:.0f}% of normal)'
        }
    else:
        return {
            'type': 'Operational Volatility / Data Variance',
            'confidence': 'Low',
            'indicators': f'Inconsistent readings: Efficiency {efficiency:.0f}%, Amps Ratio {amps_ratio:.2f}'
        }

def determine_severity(anomaly_code: str, metrics: Dict) -> str:
    impact = metrics.get('impact_value', 0)
    thresholds = SEVERITY_THRESHOLDS.get(anomaly_code, {"high": 999, "moderate": 999})
    
    if anomaly_code == "EFFICIENCY_DEGRADATION":
        efficiency_drop = metrics.get('efficiency_drop', 0)
        production_loss_pct = metrics.get('production_loss_pct', 0)
        if efficiency_drop > thresholds["high"] or production_loss_pct > 50: return "High"
        elif efficiency_drop > thresholds["moderate"] or production_loss_pct > 30: return "Moderate"
        else: return "Low"
    
    elif anomaly_code == "NEW_PRODUCTION_BASELINE":
        decline_pct = abs(metrics.get('decline_pct', 0))
        if decline_pct > thresholds["high"]: return "High"
        elif decline_pct > thresholds["moderate"]: return "Moderate"
        else: return "Low"
    
    elif anomaly_code == "MONTHLY_DECLINE":
        decline_pct = metrics.get('decline_pct', 0)
        if decline_pct > thresholds["high"]: return "High"
        elif decline_pct > thresholds["moderate"]: return "Moderate"
        else: return "Low"
    
    elif anomaly_code == "PRESSURE_INSTABILITY":
        volatility_pct = metrics.get('volatility_pct', 0)
        if volatility_pct > thresholds["high"]: return "High" 
        elif volatility_pct > thresholds["moderate"]: return "Moderate"
        else: return "Low"
    
    elif anomaly_code == "PATTERN_SHIFT":
        frequency_increase = metrics.get('frequency_increase_pct', 0)
        if frequency_increase > thresholds["high"]: return "High"
        return "Moderate"
    
    elif anomaly_code == "GHOST_PROD":
        return SEVERITY_THRESHOLDS["GHOST_PROD"]["default"]
    
    else:
        if impact > 5000: return "High"
        elif impact > 1000: return "Moderate"
        else: return "Low"

# =============================================================================
# ANOMALY CHECKERS
# =============================================================================

def check_sustained_production_change(ctx: Dict) -> Optional[Dict]:
    current_prod = ctx.get('actual_oil', 0)
    baseline_90d = ctx.get('oil_baseline_90d', 0)
    
    if baseline_90d > 5:
        current_level = ctx.get('oil_avg_7d', current_prod)
        change_pct = ((baseline_90d - current_level) / baseline_90d) * 100
        
        if change_pct > 25:
            diagnosis = diagnose_mechanics(ctx)
            
            # CALCULATION: Real Loss (Baseline - Current)
            oil_loss = max(0, baseline_90d - current_level)
            # Estimate gas loss based on baseline gas vs current gas 7-day avg
            gas_loss = max(0, ctx.get('gas_baseline_90d', 0) - ctx.get('gas_avg_7d', 0))
            
            # 2. COMPUTE METRICS
            boe_loss = oil_loss + (gas_loss / 6)
            revenue_loss = (oil_loss * OIL_PRICE) + (gas_loss * GAS_PRICE)
            
            data_summary = f"""
            DETECTION TYPE: Production Slow Down (Baseline Shift)
            SENSOR: {SENSOR_DISPLAY_NAMES.get('oil_volume', 'Oil Volume')}
            
            --- PRODUCTION METRICS ---
            90-DAY HISTORICAL AVG: {baseline_90d:.1f} bbl/day
            CURRENT AVG (7-Day): {current_level:.1f} bbl/day
            PERCENTAGE DROP: {change_pct:.1f}%
            
            --- IMPACT ANALYSIS ---
            CALCULATED_REVENUE_LOSS: ${revenue_loss:.2f}
            CALCULATED_BOE_LOSS: {boe_loss:.1f} BOE
            
            --- DIAGNOSIS ---
            ROOT CAUSE DIAGNOSIS: {diagnosis['type']}
            CONFIDENCE: {diagnosis['confidence']}
            TECHNICAL INDICATORS: {diagnosis['indicators']}
            IMPLICATION: Significant shift in production baseline.
            """
            
            return {
                "code": "NEW_PRODUCTION_BASELINE",
                "category": "PRODUCTION",
                "raw_data": data_summary,
                "severity_metrics": {
                    'decline_pct': change_pct,
                    'impact_value': revenue_loss
                },
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${revenue_loss:,.0f}/day", "subtext": "potential daily loss"},
                    "production": {"value": boe_loss, "label": f"{boe_loss:.1f} BOE/day", "subtext": "production at risk"}
                },
                "chart_metric": "baseline_shift",
                "diagnosis": diagnosis
            }
    return None

def check_efficiency_degradation(ctx: Dict) -> Optional[Dict]:
    pred_oil = ctx.get('pred_oil', 0)
    actual_oil = ctx.get('actual_oil', 0)
    
    if pred_oil > 5 and actual_oil >= 0:
        current_efficiency = (actual_oil / pred_oil) * 100
        avg_efficiency = ctx.get('efficiency_avg_30d', 100)
        efficiency_drop = avg_efficiency - current_efficiency
        
        if efficiency_drop > 15:
            diagnosis = diagnose_mechanics(ctx)
            
            # 1. CALCULATE LOSS
            oil_loss = max(0, pred_oil - actual_oil)
            # Estimate gas loss using Gas/Oil Ratio (GOR)
            gor = ctx.get('actual_gas', 0) / (actual_oil if actual_oil > 0 else 1)
            gas_loss = oil_loss * gor
            
            # 2. COMPUTE METRICS
            boe_loss = oil_loss + (gas_loss / 6)
            revenue_loss = (oil_loss * OIL_PRICE) + (gas_loss * GAS_PRICE)
            
            production_loss_pct = (oil_loss / pred_oil) * 100
            
            data_summary = f"""
            DETECTION TYPE: Efficiency Degradation
            
            --- EFFICIENCY DATA ---
            CURRENT EFFICIENCY: {current_efficiency:.1f}% 
            HISTORICAL EFFICIENCY (30d avg): {avg_efficiency:.1f}%
            EXPECTED PRODUCTION (Based on Pump Parameters): {pred_oil:.1f} bbl
            ACTUAL PRODUCTION: {actual_oil:.1f} bbl
            EFFICIENCY DROP: {efficiency_drop:.1f} percentage points
            
            --- IMPACT ANALYSIS ---
            CALCULATED_REVENUE_LOSS: ${revenue_loss:.2f}
            CALCULATED_BOE_LOSS: {boe_loss:.1f} BOE
            
            --- DIAGNOSIS ---
            ROOT CAUSE DIAGNOSIS: {diagnosis['type']}
            TECHNICAL INDICATORS: {diagnosis['indicators']}
            IMPLICATION: Pump efficiency has degraded.
            """
            
            return {
                "code": "EFFICIENCY_DEGRADATION",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity_metrics": {
                    'efficiency_drop': efficiency_drop,
                    'production_loss_pct': production_loss_pct,
                    'impact_value': revenue_loss
                },
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${revenue_loss:,.0f}/day", "subtext": "potential daily loss"},
                    "production": {"value": boe_loss, "label": f"{boe_loss:.1f} BOE/day", "subtext": "production at risk"}
                },
                "chart_metric": "efficiency_trend",
                "diagnosis": diagnosis
            }
    return None

def check_pressure_instability(ctx: Dict) -> Optional[Dict]:
    tp_avg = ctx.get('tp_avg', 0)
    tp_std = ctx.get('tp_std', 0)
    current_tp = ctx.get('tubing_pressure', 0)
    
    # --- LOGIC FIX: DEAD WELL CHECK ---
    # If production has crashed (>50% drop), ignore pressure instability.
    current_oil = ctx.get('actual_oil', 0)
    avg_oil = ctx.get('oil_avg_30d', 0)
    
    if avg_oil > 5 and (current_oil / avg_oil) < 0.5:
        return None  # Suppress this alert, let "Baseline Shift" handle it.

    if tp_avg > 50:
        volatility_pct = (tp_std / tp_avg) * 100
        
        if volatility_pct > 22 and tp_std > 40:
            primary_indicator = "Flow Instability"
            if ctx.get('actual_gas', 0) / (ctx.get('actual_oil', 1) or 1) > 10:
                primary_indicator = "Gas Interference"
            
            # Risk calculation (10% of current production)
            risk_factor = 0.10
            oil_at_risk = ctx.get('actual_oil', 0) * risk_factor
            gas_at_risk = ctx.get('actual_gas', 0) * risk_factor
            
            boe_loss = oil_at_risk + (gas_at_risk / 6)
            revenue_loss = (oil_at_risk * OIL_PRICE) + (gas_at_risk * GAS_PRICE)
            
            data_summary = f"""
            DETECTION TYPE: Tubing Pressure Instability
            SENSOR: {SENSOR_DISPLAY_NAMES.get('tubing_pressure', 'Tubing Pressure')}
            VOLATILITY: {volatility_pct:.1f}%
            
            --- IMPACT ANALYSIS ---
            CALCULATED_REVENUE_LOSS: ${revenue_loss:.2f}
            CALCULATED_BOE_LOSS: {boe_loss:.1f} BOE
            """
            
            return {
                "code": "PRESSURE_INSTABILITY",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity_metrics": {
                    'volatility_pct': volatility_pct,
                    'impact_value': revenue_loss
                },
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${revenue_loss:,.0f}/day", "subtext": "potential daily loss"},
                    "production": {"value": boe_loss, "label": f"{boe_loss:.1f} BOE/day", "subtext": "production at risk"}
                },
                "chart_metric": "tubing_pressure"
            }
    return None

def check_ghost_production(ctx: Dict) -> Optional[Dict]:
    # Skip if data is very old (2+ days)
    if 'timestamp' in ctx:
        row_date = ctx['timestamp'].date()
        current_date = datetime.now(timezone.utc).date()
        days_old = (current_date - row_date).days
        if days_old < 1: return None
            
    total = ctx['actual_oil'] + ctx['actual_water'] + ctx['actual_gas']
    is_active = ctx.get('is_active', False)
    
    if is_active and (total < 0.1):
        # 1. CALCULATE LOSS (Entire average production is lost)
        avg_oil = ctx.get('avg_oil', 0)
        avg_gas = ctx.get('gas_avg_30d', 0)
        
        # 2. COMPUTE METRICS
        boe_loss = avg_oil + (avg_gas / 6)
        revenue_loss = (avg_oil * OIL_PRICE) + (avg_gas * GAS_PRICE)
        
        if revenue_loss == 0: revenue_loss = 100.0 
        
        data_summary = f"""
        DETECTION TYPE: Ghost Production
        STATUS: Pump is physically running (Active Amps/SPM), but sensors report ZERO volume.
        AVG PRODUCTION: {avg_oil:.1f} bbl/day
        
        --- IMPACT ANALYSIS ---
        CALCULATED_REVENUE_LOSS: ${revenue_loss:.2f}
        CALCULATED_BOE_LOSS: {boe_loss:.1f} BOE
        
        DIAGNOSTIC HINT: Downhole Pump Failure (Parted Rods) OR Surface Meter/Sensor Failure.
        """
        
        return {
            "code": "GHOST_PROD",
            "category": "PROCESS",
            "raw_data": data_summary,
            "severity_metrics": {
                'impact_value': revenue_loss
            },
            "impact_value": revenue_loss,
            "impact_payload": {
                "financial": {"value": revenue_loss, "label": f"${revenue_loss:,.0f}/day", "subtext": "potential daily loss"},
                "production": {"value": boe_loss, "label": f"{boe_loss:.1f} BOE/day", "subtext": "production at risk"}
            },
            "chart_metric": "production_bar"
        }
    return None

def check_monthly_production_decline(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    recent_df = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=120))].copy()
    recent_df['month'] = recent_df['timestamp'].dt.to_period('M')
    
    monthly = recent_df.groupby('month').agg({
        'oil_volume': ['sum', 'count'],
        'gas_volume': ['sum', 'count']
    }).reset_index()
    monthly.columns = ['month', 'oil_total', 'oil_days', 'gas_total', 'gas_days']
    monthly['oil_rate'] = monthly['oil_total'] / monthly['oil_days']
    monthly['gas_rate'] = monthly['gas_total'] / monthly['gas_days']
    
    if len(monthly) >= 2:
        latest = monthly.iloc[-1]
        previous = monthly.iloc[-2]
        
        if latest['oil_days'] < 3: return None

        if previous['oil_rate'] > 5:
            decline_pct = ((previous['oil_rate'] - latest['oil_rate']) / previous['oil_rate']) * 100
            
            if decline_pct > 15:
                # 1. CALCULATE LOSS (Rate Difference)
                oil_loss = max(0, previous['oil_rate'] - latest['oil_rate'])
                gas_loss = max(0, previous['gas_rate'] - latest['gas_rate'])
                
                # 2. COMPUTE METRICS
                boe_loss = oil_loss + (gas_loss / 6)
                revenue_loss = (oil_loss * OIL_PRICE) + (gas_loss * GAS_PRICE)
                
                data_summary = f"""
                DETECTION TYPE: Monthly Production Decline
                PREV MONTH RATE: {previous['oil_rate']:.1f} bbl/day
                CURR MONTH RATE: {latest['oil_rate']:.1f} bbl/day
                DECLINE: {decline_pct:.1f}%
                
                --- IMPACT ANALYSIS ---
                CALCULATED_REVENUE_LOSS: ${revenue_loss:.2f}
                CALCULATED_BOE_LOSS: {boe_loss:.1f} BOE
                """
                
                return {
                    "code": "MONTHLY_DECLINE",
                    "category": "PRODUCTION",
                    "raw_data": data_summary,
                    "severity_metrics": {
                        'decline_pct': decline_pct,
                        'impact_value': revenue_loss
                    },
                    "impact_value": revenue_loss,
                    "impact_payload": {
                        "financial": {"value": revenue_loss, "label": f"${revenue_loss:,.0f}/day", "subtext": "potential daily loss"},
                        "production": {"value": boe_loss, "label": f"{boe_loss:.1f} BOE/day", "subtext": "production at risk"}
                    },
                    "chart_metric": "monthly_production"
                }
    return None

def check_operational_pattern_shift(ctx: Dict, df: pd.DataFrame) -> Optional[Dict]:
    if 'timestamp' not in ctx: return None
    current_timestamp = ctx['timestamp']
    max_timestamp = df['timestamp'].max()
    if current_timestamp != max_timestamp: return None
    
    violation_rules = {
        'tubing_pressure': {'min': 50, 'max': 2000, 'label': 'Tubing Pressure Spikes'},
        'motor_current': {'min': 5, 'max': 200, 'label': 'Motor Current Anomalies'}
    }
    
    for sensor, rules in violation_rules.items():
        if sensor not in df.columns: continue
        
        sensor_data = df[[sensor, 'timestamp']].copy()
        sensor_data = sensor_data[sensor_data[sensor] > 0]
        if len(sensor_data) < 60: continue
        
        sensor_data['violation'] = ((sensor_data[sensor] < rules['min']) | (sensor_data[sensor] > rules['max'])).astype(int)
        
        historical_cutoff = df['timestamp'].max() - pd.Timedelta(days=90)
        recent_cutoff = df['timestamp'].max() - pd.Timedelta(days=30)
        
        historical_rate = sensor_data[sensor_data['timestamp'] < historical_cutoff]['violation'].mean() * 100
        recent_rate = sensor_data[sensor_data['timestamp'] >= recent_cutoff]['violation'].mean() * 100
        
        if recent_rate > 10 and recent_rate > (historical_rate * 2.5):
            frequency_increase_pct = ((recent_rate - historical_rate) / max(historical_rate, 1)) * 100
            
            # 1. CALCULATE RISK (15% of production at risk)
            risk_factor = 0.15
            oil_at_risk = ctx.get('actual_oil', 0) * risk_factor
            gas_at_risk = ctx.get('actual_gas', 0) * risk_factor
            
            # 2. COMPUTE METRICS
            boe_loss = oil_at_risk + (gas_at_risk / 6)
            revenue_loss = (oil_at_risk * OIL_PRICE) + (gas_at_risk * GAS_PRICE)
            
            data_summary = f"""
            DETECTION TYPE: Operational Pattern Shift - {rules['label']}
            FREQUENCY INCREASE: {frequency_increase_pct:.0f}%
            
            --- IMPACT ANALYSIS ---
            CALCULATED_REVENUE_LOSS: ${revenue_loss:.2f}
            CALCULATED_BOE_LOSS: {boe_loss:.1f} BOE
            """
            
            return {
                "code": "PATTERN_SHIFT",
                "category": "OPERATIONAL",
                "raw_data": data_summary,
                "severity_metrics": {
                    'frequency_increase_pct': frequency_increase_pct,
                    'impact_value': revenue_loss
                },
                "impact_value": revenue_loss,
                "impact_payload": {
                    "financial": {"value": revenue_loss, "label": f"${revenue_loss:,.0f}/day", "subtext": "potential daily loss"},
                    "production": {"value": boe_loss, "label": f"{boe_loss:.1f} BOE/day", "subtext": "production at risk"}
                },
                "chart_metric": "sensor_violations"
            }
    return None

# =============================================================================
# LLM GENERATION
# =============================================================================

def generate_llm_content(anomaly_data: str, lift_type: str, chart_metric: str, historical_context: str, detected_at_str: str, diagnosis: Dict = None) -> Dict:
    try:
        client = get_openai_client()
        if not client:
            raise Exception("No OpenAI Client")
        
        try:
            if 'T' in str(detected_at_str):
                dt_obj = datetime.fromisoformat(str(detected_at_str).replace('Z', '+00:00'))
            else:
                dt_obj = datetime.strptime(str(detected_at_str), '%Y-%m-%d %H:%M:%S.%f')
            formatted_date = dt_obj.strftime('%m/%d/%Y %H:%M:%S')
        except:
            formatted_date = str(detected_at_str)

        # Handling "Low Confidence" diagnoses in the prompt
        confidence_instruction = ""
        diagnosis_context = ""
        
        if diagnosis:
            if diagnosis.get('confidence') == 'Low':
                diagnosis_context = f"DIAGNOSIS (LOW CONFIDENCE): {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                NOTE: The root cause confidence is LOW. 
                - In 'suspected_root_cause', state "Possible causes include [Diagnosis] depending on validation."
                - DO NOT use the word "Unconfirmed" in the title. Treat the anomaly as valid but needing review.
                - Emphasize the need for further field validation in the description.
                """
            else:
                diagnosis_context = f"CONFIRMED DIAGNOSIS: {diagnosis['type']} (Indicators: {diagnosis['indicators']})"
                confidence_instruction = """
                - Use ONLY declarative statements ("The root cause is...").
                - NO hedging ("might", "could").
                - Be confident and direct in all fields.
                """

        prompt = f"""
ROLE: Senior Production Engineer with 20+ years in oil & gas operations, specializing in rod pump systems and anomaly detection.
TASK: Generate a detailed, professional report on a detected well anomaly. The report must be factual, concise, and actionable for field operators and engineers. Use technical terminology appropriately but explain implications clearly.

{confidence_instruction}

ANOMALY DATA (raw metrics and indicators - use these exactly in your report):
{anomaly_data}

DIAGNOSIS (root cause assessment - integrate this into the report):
{diagnosis_context}

HISTORICAL CONTEXT (past similar events - reference in historical_context field):
{historical_context}

LIFT TYPE: {lift_type} (tailor explanations to this, e.g., rod pump specifics like SPM, amps, tubing pressure correlations)

CHART METRIC: {chart_metric} (mention relevant visualization in risk_analysis if applicable)

DETECTED AT: {formatted_date}

IMPORTANT GUIDELINES FOR ALL FIELDS:
- ALWAYS explicitly name the primary sensor(s) involved (e.g., "Tubing Pressure", "Motor Current", "Strokes Per Minute") in the description, why_is_this_an_anomaly, and suspected_root_cause.
- SENSOR NAMING: Never use snake_case (e.g. oil_volume) in the output text. Always convert to Title Case (e.g. Oil Volume).
- If volatility, deviation, drop, or other percentages are in ANOMALY DATA, PROMINENTLY INCLUDE THE EXACT NUMBERS.
- Prioritize key metrics: volatility % over small mean deviations for instability; efficiency drop % for degradation; decline % for production changes.
- Use industry-specific language: reference rod pump dynamics, gas interference, pump off, parted rods, etc., where relevant.
- Ensure the report is actionable.
- Financial impacts: You MUST use the 'CALCULATED_REVENUE_LOSS' provided in the data. Do NOT calculate it yourself.

Generate a JSON response with these EXACT keys and STRICT formats:

1. "title": 
   - If High Confidence Diagnosis: Use diagnosis type (e.g., "Tubing Leak Detected").
   - If Low Confidence: Use the observed symptom (e.g., "Efficiency Drop Detected").
   - Max 6 words. No $ symbols. Make it attention-grabbing and specific.

2. "description": 
   - Write EXACTLY 3-4 sentences.
   - FIRST SENTENCE: State current value vs baseline with specific numbers and sensor name.
     Example: "Tubing pressure is currently at 1166.0 psi, a 2.9% deviation below the 30-day average of 1200.1 psi, with volatility at 24.5%."
   - SECOND SENTENCE: Explain the physical/mechanical implication, naming sensors.
     Example: "This high tubing pressure volatility indicates gas interference disrupting pump fillage and efficiency."
   - THIRD SENTENCE: State the financial impact.
     * CRITICAL: Use the 'CALCULATED_REVENUE_LOSS' value from ANOMALY DATA.
     Example: "This instability is costing an estimated $[VALUE] per day in lost oil production."
   - FOURTH SENTENCE (optional): Add operational context or recommendation.
     Example: "Immediate review of downhole conditions recommended to prevent escalation."

3. "why_is_this_an_anomaly": 
   - ONE clear sentence stating the deviation math, including sensor and key percentages.
   - Use format: "The [sensor/metric] shows [X value] with [Y% volatility/deviation/drop], compared to the historical average of [Z]."
   - Example: "The tubing pressure volatility of 24.5% is a 150% increase over normal (<10%), with current value 2.9% below the historical average of 1200.1 psi."

4. "suspected_root_cause": 
   - Return a SINGLE, DETAILED sentence (30-60 words).
   - If ROOT CAUSE DIAGNOSIS is provided, use this format EXACTLY:
     "The root cause is {{diagnosis.get('type') if diagnosis else 'mechanical failure'}}, confirmed by {{diagnosis.get('indicators') if diagnosis else 'operational data'}}, likely involving [expand with sensor correlations from ANOMALY DATA]."
   - Otherwise, infer most probable physical cause based on sensor correlations and lift type.
   - Include sensor names and numbers.
   - DO NOT provide a list.
   - Example: "The root cause is gas interference in the rod pump system, evidenced by tubing pressure volatility at 24.5% and elevated gas-to-oil ratio exceeding 10:1, leading to incomplete pump fillage and reduced stroke efficiency."

5. "economic_impact": 
   - ONE sentence in this format: "This [problem type, e.g., pressure instability] is resulting in an estimated daily revenue loss of $[X], based on current rates."
   - CRITICAL: You MUST extract $[X] from 'CALCULATED_REVENUE_LOSS' in ANOMALY DATA. Do not recalculate.
   - Example: "This pressure instability is resulting in an estimated daily revenue loss of $1,200, based on production impact at current rates."

6. "risk_analysis": 
   - You MUST follow this EXACT two-paragraph structure, no objects just string:
   
   PARAGRAPH 1 (Immediate Risk - Detection Statement):
   "The production anomaly [Insert specific anomaly type from title] was detected through continuous monitoring of [list key sensors, e.g., tubing pressure and motor current]. [Insert ONE sentence with the specific percentage drop or deviation, including numbers]. [Insert ONE sentence about immediate operational stability or safety concern - focus on equipment damage risk (e.g., pump seizure) or safety hazard (e.g., pressure buildup) if applicable, tailored to lift type]."
   
   PARAGRAPH 2 (Long-term Consequence):
   "Detected at: {formatted_date}
   Root cause: [Insert the EXACT root cause from field 4 above]
   
   [Insert 2-3 sentences detailing potential long-term consequences if left unaddressed. MUST MENTION the 'CALCULATED_BOE_LOSS' volume as production at risk. Focus on: permanent reservoir damage, equipment failure cascade (e.g., rod failure leading to well shutdown), compounded financial loss, regulatory non-compliance (e.g., emissions from gas venting), or safety escalation (e.g., blowout risk). Be specific about timelines and dollar amounts where possible, using provided impact_value]. This analysis is based on historical equipment performance data and industry failure progression patterns."

7. "historical_context": 
   - Return a JSON object with:
     {{
       "recurrence_status": "[First occurrence|Recurring issue - X previous events|Chronic problem - Y events in Z days]" (analyze from HISTORICAL CONTEXT),
       "history_timeline": [
         {{"date": "MM/DD/YYYY", "severity": "High|Moderate|Low", "title": "Previous event title"}},
         ...
       ] (limit to 5, format dates properly)
     }}

Return ONLY valid JSON. Ensure all fields are populated based on data provided.
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
            "title": "Anomaly Detected",
            "description": "An operational deviation was detected. Review sensor data manually. Impact unknown.",
            "why_is_this_an_anomaly": "Metrics deviate from baseline.",
            "suspected_root_cause": "Unknown - data analysis required.",
            "economic_impact": "Impact calculation unavailable.",
            "risk_analysis": "Review required for immediate and long-term risks.",
            "historical_context": {"recurrence_status": "Unknown", "history_timeline": []}
        }

# =============================================================================
# CHART GENERATION SYSTEM
# =============================================================================

def generate_chart_config(anomaly_data: str, chart_metric: str, df: pd.DataFrame, anomaly_code: str = None, severity_metrics: Dict = None) -> Dict:
    """
    Enhanced chart generation with anomaly-specific visualizations.
    Now creates multi-layer, contextual charts with clear anomaly highlighting.
    """
    if df.empty:
        return create_fallback_chart(df, "No Data")

    try:
        # Universal data preparation
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date_only'] = df['timestamp'].dt.date
        
        # Get detection point from anomaly_data
        detection_date = extract_detection_date(anomaly_data)
        
        # Route to anomaly-specific chart generators
        if chart_metric == "monthly_production":
            return generate_monthly_decline_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "baseline_shift":
            return generate_baseline_shift_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "efficiency_trend":
            return generate_efficiency_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "tubing_pressure":
            return generate_pressure_instability_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "sensor_violations":
            return generate_pattern_shift_chart(df, anomaly_data, detection_date, severity_metrics)
        elif chart_metric == "production_bar":
            return generate_ghost_production_chart(df, anomaly_data, detection_date, severity_metrics)
        else:
            return generate_default_enhanced_chart(df, anomaly_data, detection_date)
            
    except Exception as e:
        logger.error(f"Enhanced chart generation failed: {e}")
        return create_fallback_chart(df, str(e))


def extract_detection_date(anomaly_data: str) -> Optional[str]:
    """Extract detection date from anomaly data text"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', anomaly_data)
    return match.group(1) if match else None


def prepare_daily_aggregation(df: pd.DataFrame, days: int = 60) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregates high-frequency data to daily points and returns clean labels.
    Reusable across all chart types.
    """
    start_date = df['timestamp'].max() - pd.Timedelta(days=days)
    recent_df = df[df['timestamp'] >= start_date].copy()
    
    # Aggregate by day
    daily_df = recent_df.groupby('date_only').agg({
        'oil_volume': 'mean',
        'water_volume': 'mean',
        'gas_volume': 'mean',
        'tubing_pressure': ['mean', 'std', 'min', 'max'],
        'motor_current': ['mean', 'std'],
        'strokes_per_minute': 'mean',
        'predicted_oil': 'mean' if 'predicted_oil' in df.columns else 'mean',
        'efficiency': 'mean' if 'efficiency' in df.columns else 'mean',
        'oil_baseline_90d': 'mean' if 'oil_baseline_90d' in df.columns else 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    daily_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in daily_df.columns]
    
    # Create clean labels
    labels = [d.strftime('%Y-%m-%d') for d in daily_df['date_only']]
    
    return daily_df, labels


def create_anomaly_zone_annotation(detection_date: str, labels: List[str]) -> Dict:
    """Creates a vertical line annotation at the detection point"""
    if not detection_date or detection_date not in labels:
        return {}
    
    detection_index = labels.index(detection_date)
    
    return {
        'type': 'line',
        'xMin': detection_index,
        'xMax': detection_index,
        'borderColor': '#ef4444',
        'borderWidth': 2,
        'borderDash': [5, 5],
        'label': {
            'display': True,
            'content': 'Anomaly Detected',
            'position': 'start',
            'backgroundColor': '#ef4444',
            'color': '#ffffff',
            'font': {'size': 11, 'weight': 'bold'}
        }
    }


def create_threshold_band(labels: List[str], threshold_value: float, label: str, color: str = '#10b981') -> Dict:
    """Creates a horizontal threshold line"""
    return {
        'type': 'line',
        'yMin': threshold_value,
        'yMax': threshold_value,
        'borderColor': color,
        'borderWidth': 2,
        'borderDash': [8, 4],
        'label': {
            'display': True,
            'content': label,
            'position': 'end',
            'backgroundColor': color,
            'color': '#ffffff',
            'font': {'size': 10}
        }
    }


# =============================================================================
# ANOMALY-SPECIFIC CHART GENERATORS
# =============================================================================

def generate_efficiency_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    
    if 'efficiency_mean' not in daily_df.columns and 'predicted_oil_mean' in daily_df.columns:
        daily_df['efficiency_mean'] = (daily_df['oil_volume_mean'] / daily_df['predicted_oil_mean'].replace(0, 1)) * 100
        daily_df['efficiency_mean'] = daily_df['efficiency_mean'].clip(0, 120)
    
    actual_oil = daily_df['oil_volume_mean'].fillna(0).round(1).tolist()
    predicted_oil = daily_df.get('predicted_oil_mean', daily_df['oil_volume_mean']).fillna(0).round(1).tolist()
    efficiency = daily_df.get('efficiency_mean', [100] * len(labels)).fillna(100).round(1).tolist()
    
    baseline_efficiency = 100
    match = re.search(r"HISTORICAL EFFICIENCY.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try: baseline_efficiency = float(match.group(1))
        except: pass
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    if baseline_efficiency > 0:
        annotations.append(create_threshold_band(labels, baseline_efficiency, f'Normal Efficiency ({baseline_efficiency:.0f}%)', '#10b981'))
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Pump Efficiency Degradation Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    
    options['scales']['y'] = {'type': 'linear', 'position': 'left', 'title': {'display': True, 'text': 'Efficiency (%)'}, 'min': 0, 'max': 120, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    options['scales']['y1'] = {'type': 'linear', 'position': 'right', 'title': {'display': True, 'text': 'Production (bbl/day)'}, 'grid': {'drawOnChartArea': False}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Actual Production", "data": actual_oil, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y1"},
                {"label": "Predicted Production", "data": predicted_oil, "borderColor": "#9ca3af", "borderDash": [5, 5], "yAxisID": "y1"},
                {"label": "Pump Efficiency", "data": efficiency, "borderColor": "#8b5cf6", "borderWidth": 3, "yAxisID": "y"}
            ]
        },
        "options": options
    }


def generate_baseline_shift_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=90)
    actual_data = daily_df['oil_volume_mean'].round(1).fillna(0).tolist()
    baseline_90d = daily_df.get('oil_baseline_90d_mean', [0] * len(labels)).fillna(0).tolist()
    
    baseline_val = None
    match = re.search(r"90-DAY HISTORICAL AVG.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try: baseline_val = float(match.group(1))
        except: pass
    if baseline_val is None or baseline_val == 0:
        baseline_val = daily_df['oil_volume_mean'].quantile(0.75)
    
    baseline_line = [baseline_val] * len(labels)
    if len(daily_df) >= 7:
        rolling_avg = daily_df['oil_volume_mean'].rolling(7, min_periods=3).mean().fillna(method='bfill').round(1).tolist()
    else:
        rolling_avg = actual_data
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    annotations.append(create_threshold_band(labels, baseline_val, f'90-Day Baseline ({baseline_val:.0f} bbl)', '#10b981'))
    
    if severity_metrics and severity_metrics.get('decline_pct', 0) > 25:
        current_level = severity_metrics.get('current_level', baseline_val * 0.7)
        annotations.append(create_threshold_band(labels, current_level, 'New Baseline', '#ef4444'))
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Production Baseline Shift Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Decline: {severity_metrics.get('decline_pct', 0):.1f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Daily Production", "data": actual_data, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True},
                {"label": "7-Day Trend", "data": rolling_avg, "borderColor": "#0ea5e9", "borderWidth": 3, "pointRadius": 0},
                {"label": "Historical Baseline", "data": baseline_line, "borderColor": "#10b981", "borderWidth": 2, "borderDash": [8, 4], "pointRadius": 0}
            ]
        },
        "options": options
    }


def generate_pressure_instability_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    pressure_avg = daily_df.get('tubing_pressure_mean', [0] * len(labels)).fillna(0).round(1).tolist()
    pressure_min = daily_df.get('tubing_pressure_min', pressure_avg).fillna(0).round(1).tolist()
    pressure_max = daily_df.get('tubing_pressure_max', pressure_avg).fillna(0).round(1).tolist()
    oil_production = daily_df['oil_volume_mean'].fillna(0).round(1).tolist()
    
    avg_pressure = None
    match = re.search(r"AVERAGE TUBING PRESSURE.*?:.*?([\d\.]+)", anomaly_data)
    if match:
        try: avg_pressure = float(match.group(1))
        except: pass
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    if avg_pressure:
        annotations.append(create_threshold_band(labels, avg_pressure, f'Normal Pressure ({avg_pressure:.0f} psi)', '#10b981'))
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Tubing Pressure Instability & Production Correlation', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Volatility: {severity_metrics.get('volatility_pct', 0):.1f}% (Normal <10%)", 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {'type': 'linear', 'position': 'left', 'title': {'display': True, 'text': 'Pressure (PSI)'}, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    options['scales']['y1'] = {'type': 'linear', 'position': 'right', 'title': {'display': True, 'text': 'Oil Production (bbl/day)'}, 'grid': {'drawOnChartArea': False}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Pressure Range", "data": pressure_max, "borderColor": "rgba(239, 68, 68, 0.3)", "backgroundColor": "rgba(239, 68, 68, 0.1)", "borderWidth": 0, "fill": "+1", "pointRadius": 0, "yAxisID": "y"},
                {"label": "Pressure Min", "data": pressure_min, "borderColor": "rgba(239, 68, 68, 0.3)", "borderWidth": 0, "pointRadius": 0, "yAxisID": "y"},
                {"label": "Avg Pressure", "data": pressure_avg, "borderColor": "#ef4444", "borderWidth": 3, "yAxisID": "y"},
                {"label": "Oil Production", "data": oil_production, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y1"}
            ]
        },
        "options": options
    }


def generate_monthly_decline_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    start_date = df['timestamp'].max() - pd.Timedelta(days=180)
    monthly_raw = df[df['timestamp'] >= start_date].copy()
    monthly_raw['month_str'] = monthly_raw['timestamp'].dt.to_period('M').astype(str)
    
    monthly_df = monthly_raw.groupby('month_str').agg({
        'oil_volume': 'sum',
        'timestamp': 'count'
    }).reset_index()
    
    monthly_df['days_count'] = (monthly_df['timestamp'] / 24).round(0).clip(lower=1)
    
    if monthly_df['days_count'].mean() < 32:
        monthly_df['days_count'] = monthly_df['timestamp']
    
    monthly_df['rate'] = (monthly_df['oil_volume'] / monthly_df['days_count']).round(1)
    monthly_df['change_pct'] = monthly_df['rate'].pct_change() * 100
    
    labels = monthly_df['month_str'].tolist()
    rates = monthly_df['rate'].tolist()
    changes = monthly_df['change_pct'].fillna(0).round(1).tolist()
    
    bar_colors = []
    for change in changes:
        if change < -15:
            bar_colors.append('#ef4444')
        elif change < 0:
            bar_colors.append('#f59e0b')
        else:
            bar_colors.append('#10b981')
    
    options = get_common_options()
    options['plugins']['title'] = {'display': True, 'text': 'Monthly Production Decline Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Overall Decline: {severity_metrics.get('decline_pct', 0):.1f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    options['plugins']['datalabels'] = {'display': True, 'anchor': 'end', 'align': 'top', 'font': {'size': 10, 'weight': 'bold'}}
    
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {
                    "label": "Avg Daily Rate (bbl/d)",
                    "data": rates,
                    "backgroundColor": bar_colors,
                    "borderRadius": 6,
                    "borderWidth": 2,
                    "borderColor": "#ffffff"
                }
            ]
        },
        "options": options
    }


def generate_pattern_shift_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=90)
    
    sensor_col = 'motor_current_mean'
    sensor_label = 'Motor Current (Amps)'
    violation_label = 'Current Anomalies'
    
    if 'tubing' in anomaly_data.lower() or 'pressure' in anomaly_data.lower():
        sensor_col = 'tubing_pressure_mean'
        sensor_label = 'Tubing Pressure (PSI)'
        violation_label = 'Pressure Spikes'
    
    sensor_data = daily_df.get(sensor_col, [0] * len(labels)).fillna(0).round(1).tolist()
    sensor_std = daily_df.get(sensor_col.replace('_mean', '_std'), [0] * len(labels)).fillna(0).round(1).tolist()
    
    violation_threshold = np.percentile([s for s in sensor_std if s > 0], 75) if any(sensor_std) else 0
    violation_days = [1 if std > violation_threshold else 0 for std in sensor_std]
    
    if len(violation_days) >= 7:
        violation_rate = pd.Series(violation_days).rolling(7).mean().fillna(0).tolist()
        violation_rate = [v * 100 for v in violation_rate]
    else:
        violation_rate = [0] * len(labels)
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    annotations.append({
        'type': 'line',
        'yMin': 10,
        'yMax': 10,
        'borderColor': '#f59e0b',
        'borderWidth': 2,
        'borderDash': [5, 5],
        'label': {
            'display': True,
            'content': 'Normal Threshold (10%)',
            'position': 'end',
            'backgroundColor': '#f59e0b',
            'color': '#ffffff'
        }
    })
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': f'Operational Pattern Shift - {violation_label}', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': f"Violation Frequency Increase: {severity_metrics.get('frequency_increase_pct', 0):.0f}%", 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {'type': 'linear', 'position': 'left', 'title': {'display': True, 'text': sensor_label}, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    options['scales']['y1'] = {'type': 'linear', 'position': 'right', 'title': {'display': True, 'text': 'Violation Rate (%)'}, 'min': 0, 'max': 100, 'grid': {'drawOnChartArea': False}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": sensor_label, "data": sensor_data, "borderColor": "#64748b", "backgroundColor": "rgba(100, 116, 139, 0.1)", "borderWidth": 2, "fill": True, "yAxisID": "y"},
                {"label": "Violation Rate (7-day %)", "data": violation_rate, "borderColor": "#ef4444", "backgroundColor": "rgba(239, 68, 68, 0.2)", "borderWidth": 3, "fill": True, "yAxisID": "y1"}
            ]
        },
        "options": options
    }


def generate_ghost_production_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str, severity_metrics: Dict) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=30)
    
    oil_data = daily_df['oil_volume_mean'].fillna(0).round(1).tolist()
    water_data = daily_df.get('water_volume_mean', [0] * len(labels)).fillna(0).round(1).tolist()
    gas_data = daily_df.get('gas_volume_mean', [0] * len(labels)).fillna(0).round(1).tolist()
    
    motor_current = daily_df.get('motor_current_mean', [0] * len(labels)).fillna(0).round(1).tolist()
    spm = daily_df.get('strokes_per_minute_mean', [0] * len(labels)).fillna(0).round(1).tolist()
    
    pump_running = [1 if (mc > 5 or s > 0.1) else 0 for mc, s in zip(motor_current, spm)]
    pump_running_scaled = [pr * max(oil_data + water_data + gas_data + [10]) * 0.1 for pr in pump_running]
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Ghost Production Analysis - Equipment vs Output', 'font': {'size': 16, 'weight': 'bold'}}
    options['plugins']['subtitle'] = {'display': True, 'text': 'Pump running but zero production - potential SCADA/meter failure', 'font': {'size': 12}, 'color': '#ef4444'}
    
    options['scales']['y'] = {'stacked': True, 'title': {'display': True, 'text': 'Production Volume'}, 'grid': {'color': 'rgba(200, 200, 200, 0.1)'}}
    
    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Oil", "data": oil_data, "backgroundColor": "#10b981", "borderRadius": 4, "stack": "production"},
                {"label": "Water", "data": water_data, "backgroundColor": "#3b82f6", "borderRadius": 4, "stack": "production"},
                {"label": "Gas", "data": gas_data, "backgroundColor": "#f59e0b", "borderRadius": 4, "stack": "production"},
                {"label": "Pump Active (Indicator)", "data": pump_running_scaled, "type": "line", "borderColor": "#ef4444", "borderWidth": 3, "borderDash": [5, 5], "fill": False, "pointRadius": 0, "yAxisID": "y"}
            ]
        },
        "options": options
    }


def generate_default_enhanced_chart(df: pd.DataFrame, anomaly_data: str, detection_date: str) -> Dict:
    daily_df, labels = prepare_daily_aggregation(df, days=45)
    oil_data = daily_df['oil_volume_mean'].fillna(0).round(1).tolist()
    
    annotations = [create_anomaly_zone_annotation(detection_date, labels)]
    
    options = get_common_options()
    options['plugins']['annotation'] = {'annotations': annotations}
    options['plugins']['title'] = {'display': True, 'text': 'Production Trend Analysis', 'font': {'size': 16, 'weight': 'bold'}}
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{"label": "Oil Production", "data": oil_data, "borderColor": "#3b82f6", "backgroundColor": "rgba(59, 130, 246, 0.1)", "borderWidth": 2, "fill": True, "tension": 0.3}]
        },
        "options": options
    }


def get_common_options():
    return {
        "responsive": True,
        "maintainAspectRatio": False,
        "interaction": {
            "mode": "index",
            "intersect": False
        },
        "plugins": {
            "legend": {
                "position": "top",
                "labels": {
                    "usePointStyle": True,
                    "padding": 15,
                    "font": {"size": 11}
                }
            },
            "tooltip": {
                "usePointStyle": True,
                "backgroundColor": "rgba(15, 23, 42, 0.95)",
                "titleColor": "#f8fafc",
                "bodyColor": "#f1f5f9",
                "borderColor": "#334155",
                "borderWidth": 1,
                "padding": 12,
                "titleFont": {"size": 13, "weight": "bold"},
                "bodyFont": {"size": 12},
                "displayColors": True
            }
        },
        "scales": {
            "x": {
                "grid": {"display": False, "drawBorder": True},
                "ticks": {
                    "maxTicksLimit": 10,
                    "maxRotation": 45,
                    "minRotation": 0,
                    "font": {"size": 10}
                }
            },
            "y": {
                "grid": {"color": "rgba(200, 200, 200, 0.15)"},
                "beginAtZero": False,
                "ticks": {"font": {"size": 10}}
            }
        },
        "elements": {
            "point": {
                "radius": 0,
                "hitRadius": 20,
                "hoverRadius": 6,
                "hoverBorderWidth": 2
            },
            "line": {
                "borderJoinStyle": "round"
            }
        }
    }


def create_fallback_chart(df: pd.DataFrame, error_msg: str) -> Dict:
    if df.empty:
        return {
            "type": "line",
            "data": {"labels": [], "datasets": []},
            "options": {
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "No Data Available",
                        "font": {"size": 14, "weight": "bold"},
                        "color": "#ef4444"
                    },
                    "subtitle": {
                        "display": True,
                        "text": "Insufficient data to generate chart",
                        "font": {"size": 11},
                        "color": "#64748b"
                    }
                }
            }
        }
    
    daily_df, labels = prepare_daily_aggregation(df, days=30)
    data = daily_df['oil_volume_mean'].fillna(0).round(1).tolist()
    
    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [{
                "label": "Oil Production (Fallback View)",
                "data": data,
                "borderColor": "#64748b",
                "backgroundColor": "rgba(100, 116, 139, 0.1)",
                "borderWidth": 2,
                "fill": True,
                "tension": 0.3
            }]
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": "Production Data (Limited View)",
                    "font": {"size": 14}
                },
                "subtitle": {
                    "display": True,
                    "text": f"Chart generation issue: {error_msg[:50]}",
                    "font": {"size": 10},
                    "color": "#f59e0b"
                }
            }
        }
    }
    
# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def prioritize_anomalies(anomalies: List[Dict]) -> List[Dict]:
    """
    Filters out redundant anomalies.
    Logic: If 'GHOST_PROD' exists on a specific date, remove 'EFFICIENCY', 'BASELINE', 
    and 'MONTHLY' alerts for THAT SAME DATE, as they are side effects of 0 production.
    """
    if not anomalies: return []
    
    # 1. Identify which dates have Ghost Production
    ghost_dates = set()
    for a in anomalies:
        if a['anomaly_code'] == 'GHOST_PROD':
            ghost_dates.add(a['event_date'])
            
    cleaned_list = []
    
    for a in anomalies:
        code = a['anomaly_code']
        date = a['event_date']
        
        # 2. Suppression Rule:
        if date in ghost_dates and code in ['EFFICIENCY_DEGRADATION', 'NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE']:
            continue
            
        # 3. Monthly Rule (Legacy): 
        has_baseline_crash = any(
            x['anomaly_code'] == 'NEW_PRODUCTION_BASELINE' and x['event_date'] == date 
            for x in anomalies
        )
        if has_baseline_crash and code == 'MONTHLY_DECLINE':
            continue

        cleaned_list.append(a)
        
    return cleaned_list

def detect_anomalies(well_id: str, df: Optional[pd.DataFrame] = None, lookback_days: int = DETECTION_WINDOW_DAYS) -> List[Dict]:
    logger.info(f"Fetching all historical data for {well_id} to train model...")
    full_df = fetch_well_data(well_id, days=None)
    
    if full_df.empty or len(full_df) < 30:
        logger.warning(f"Insufficient data for {well_id}")
        return []

    full_df['lift_type'] = full_df['lift_type'].ffill().fillna('Rod Pump')
    
    logger.info("Calculating rolling baselines (Explicit Time-Based)...")
    
    # 1. Create Temporary Time-Indexed DataFrame for Calculation
    temp_df = full_df.set_index('timestamp').sort_index()
    
    # 2. Production Baselines (90 Days)
    full_df['oil_baseline_90d'] = temp_df['oil_volume'].rolling('90D').mean().values
    full_df['gas_baseline_90d'] = temp_df['gas_volume'].rolling('90D').mean().values
    full_df['oil_std_90d'] = temp_df['oil_volume'].rolling('90D').std().values
    
    # 3. Recent Averages (30 Days)
    full_df['oil_avg_30d'] = temp_df['oil_volume'].rolling('30D').mean().values
    full_df['gas_avg_30d'] = temp_df['gas_volume'].rolling('30D').mean().values
    full_df['water_avg_30d'] = temp_df['water_volume'].rolling('30D').mean().values
    
    # 4. Short Term Averages (7 Days)
    full_df['oil_avg_7d'] = temp_df['oil_volume'].rolling('7D').mean().values
    full_df['gas_avg_7d'] = temp_df['gas_volume'].rolling('7D').mean().values
    
    # 5. Sensor Rolling Metrics (CRITICAL FIXES)
    full_df['tp_avg'] = temp_df['tubing_pressure'].rolling('30D').mean().values
    full_df['tp_std'] = temp_df['tubing_pressure'].rolling('3D').std().values # 3 Days for volatility
    full_df['amps_avg'] = temp_df['motor_current'].rolling('30D').mean().values
    full_df['spm_avg'] = temp_df['strokes_per_minute'].rolling('30D').mean().values
    
    full_df['is_active'] = (full_df['strokes_per_minute'] > 0.1) | (full_df['motor_current'] > 5)

    logger.info(f"Training predictive model...")
    model_art = get_or_train_model(well_id, full_df)
    full_df['predicted_oil'] = np.nan
    
    if model_art:
        try:
            X = model_art['scaler'].transform(
                model_art['imputer'].transform(full_df[model_art['features']])
            )
            full_df['predicted_oil'] = model_art['model'].predict(X)
            # Safe division
            full_df['efficiency'] = (full_df['oil_volume'] / full_df['predicted_oil'].replace(0, np.nan)) * 100
            
            # 6. Efficiency Rolling (Fixed)
            temp_df['efficiency'] = full_df['efficiency'].values
            full_df['efficiency_avg_30d'] = temp_df['efficiency'].rolling('30D').mean().values
            
            full_df['efficiency'] = full_df['efficiency'].replace([np.inf, -np.inf], np.nan).fillna(100)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")

    # Registered Checkers
    all_checkers = [
        check_efficiency_degradation,
        check_sustained_production_change, 
        check_pressure_instability,
        lambda ctx: check_monthly_production_decline(ctx, full_df) if ctx.get('timestamp') == full_df['timestamp'].max() else None,
        check_ghost_production,
        lambda ctx: check_operational_pattern_shift(ctx, full_df)
    ]
    
    logger.info(f"Detecting anomalies in last {lookback_days} days...")
    window = full_df[full_df['timestamp'] > (full_df['timestamp'].max() - pd.Timedelta(days=lookback_days))]
    raw_findings = []
    
    for _, row in window.iterrows():
        ctx = row.to_dict()
        ctx['is_active'] = (ctx.get('strokes_per_minute',0) > 0.1) or (ctx.get('motor_current',0) > 5)
        
        # Populate context with correct values
        ctx['actual_oil'] = ctx.get('oil_volume', 0)
        ctx['actual_gas'] = ctx.get('gas_volume', 0) 
        ctx['actual_water'] = ctx.get('water_volume', 0)
        ctx['avg_oil'] = ctx.get('oil_avg_30d', 0)
        ctx['avg_water'] = ctx.get('water_avg_30d', 0)
        ctx['pred_oil'] = ctx.get('predicted_oil', ctx['actual_oil'])
        if pd.isna(ctx['pred_oil']): ctx['pred_oil'] = ctx['actual_oil']
        
        # Ensure context has rolling metrics
        for field in ['tubing_pressure', 'tp_avg', 'tp_std', 'spm', 'spm_avg', 'motor_current', 'amps_avg', 'oil_avg_7d', 'oil_baseline_90d', 'efficiency_avg_30d', 'gas_baseline_90d', 'gas_avg_30d', 'gas_avg_7d']:
            ctx[field] = ctx.get(field, 0)
            if pd.isna(ctx[field]): ctx[field] = 0

        for check in all_checkers:
            res = check(ctx)
            if res:
                severity = determine_severity(res['code'], res.get('severity_metrics', {}))
                raw_findings.append({
                    "well_id": well_id,
                    "event_date": str(row['timestamp'].date()),
                    "timestamp": row['timestamp'],
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "anomaly_code": res['code'],
                    "category": res['category'],
                    "severity": severity,
                    "raw_data": res['raw_data'],
                    "chart_metric": res['chart_metric'],
                    "impact_value": res['impact_value'],
                    "impact_payload": res['impact_payload'],
                    "lift_type": str(row['lift_type']),
                    "diagnosis": res.get('diagnosis'),
                    "severity_metrics": res.get('severity_metrics', {})
                })

    raw_findings = prioritize_anomalies(raw_findings)
    final_anomalies = []
    
    findings_by_code = defaultdict(list)
    for f in raw_findings:
        findings_by_code[f['anomaly_code']].append(f)
        
    for code, findings in findings_by_code.items():
        sorted_findings = sorted(findings, key=lambda x: x['timestamp'])
        latest_finding = sorted_findings[-1]

        history_str = get_recent_anomaly_history(well_id, code, exclude_date=latest_finding['event_date'])
        
        llm_content = generate_llm_content(
            latest_finding['raw_data'], 
            latest_finding['lift_type'], 
            latest_finding['chart_metric'], 
            history_str, 
            latest_finding['detected_at'],
            diagnosis=latest_finding.get('diagnosis')
        )
        
        chart_config = generate_chart_config(
            latest_finding['raw_data'], 
            latest_finding['chart_metric'], 
            full_df,
            anomaly_code=code,
            severity_metrics=latest_finding.get('severity_metrics', {})
        )
        
        final_anomalies.append({
            "well_id": well_id,
            "event_date": latest_finding['event_date'],
            "detected_at": latest_finding['detected_at'],
            "anomaly_code": code,
            "category": latest_finding['category'],
            "severity": latest_finding['severity'],
            "title": llm_content.get('title', 'Anomaly Detected'),
            "status": "ACTIVE",
            "ui_text": llm_content,
            "impact_metrics": latest_finding['impact_payload'],
            "impact_value": latest_finding['impact_value'],
            "chart_data": chart_config
        })

    logger.info(f"Detected {len(final_anomalies)} actionable anomalies for {well_id}")
    save_reviews(final_anomalies)
    return final_anomalies
    
def save_reviews(reviews):
    if not reviews: return

    # --- HELPER: Recursively convert Numpy types to Python native types ---
    def clean_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: clean_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_numpy(v) for v in obj]
        return obj

    # Clean all data before sending to DB
    cleaned_reviews = [clean_numpy(r) for r in reviews]
    
    # POSTGRES SAVE
    try:
        url = get_db_url()
        if url:
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    q = """
                        INSERT INTO anomaly_review
                        (well_id, event_date, detected_at, anomaly_code, category, severity, 
                         title, ui_text, impact_value, impact_metrics, chart_data, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (well_id, anomaly_code, event_date) DO NOTHING
                    """
                    for r in cleaned_reviews:
                        cur.execute(q, (
                            r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                            r['category'], r['severity'], r['title'],
                            json.dumps(r['ui_text']),
                            r['impact_value'],               # Now a native float
                            json.dumps(r['impact_metrics']), # Now safe for JSON
                            json.dumps(r['chart_data']),     # Now safe for JSON
                            r['status']
                        ))
                conn.commit()
    except Exception as e:
        logger.error(f"Postgres Save failed: {e}")

    # SNOWFLAKE SAVE
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            check_q = "SELECT count(*) FROM AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW WHERE well_id=%s AND anomaly_code=%s AND event_date=%s"
            insert_q = """
                INSERT INTO AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW
                (id, well_id, event_date, detected_at, anomaly_code, category, severity, 
                 title, ui_text, impact_value, impact_metrics, chart_data, status)
                SELECT UUID_STRING(), %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, PARSE_JSON(%s), PARSE_JSON(%s), %s
            """
            for r in cleaned_reviews:
                cur.execute(check_q, (r['well_id'], r['anomaly_code'], r['event_date']))
                if cur.fetchone()[0] == 0:
                    cur.execute(insert_q, (
                        r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                        r['category'], r['severity'], r['title'],
                        json.dumps(r['ui_text']),
                        r['impact_value'],               # Now a native float
                        json.dumps(r['impact_metrics']), # Now safe for JSON
                        json.dumps(r['chart_data']),     # Now safe for JSON
                        r['status']
                    ))
            conn.commit()
    except Exception as e:
        logger.error(f"Snowflake Save failed: {e}")

if __name__ == "__main__":
    import sys
    wid = sys.argv[1] if len(sys.argv) > 1 else "Well_001"
    print(json.dumps(detect_anomalies(wid), indent=2, default=str))