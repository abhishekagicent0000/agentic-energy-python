import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from snowflake.connector import connect as snowflake_connect
import psycopg2 
import openai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnomalyReviewService')

OIL_PRICE = 75.00
WATER_DISPOSAL_COST = 1.50
GAS_PRICE = 2.50

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

def fetch_well_data(well_id: str, days: int = 120) -> pd.DataFrame:
    sensor_selects = ", ".join([f"s.{col}" for col in ALL_SENSORS])
    
    query = f"""
    SELECT 
        s.well_id, s.timestamp, s.lift_type,
        {sensor_selects},
        p.oil_volume, p.water_volume, p.gas_volume
    FROM well_sensor_readings s
    LEFT JOIN well_daily_production p 
        ON s.well_id = p.well_id AND DATE(s.timestamp) = p.date
    WHERE s.well_id = %s
      AND s.timestamp >= DATEADD(day, -{days}, CURRENT_DATE())
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

def get_or_train_model(well_id: str, df: pd.DataFrame) -> Dict:
    available_features = [f for f in ALL_SENSORS if f in df.columns and df[f].std() > 0]
    
    if len(df) < 14 or not available_features: 
        return {}
    
    cutoff_date = df['timestamp'].max() - timedelta(days=14)
    train_df = df[df['timestamp'] <= cutoff_date]
    train_df = train_df[train_df['oil_volume'] > 0]
    
    if len(train_df) < 10: 
        return {}

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    
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
    except Exception:
        return {}

def check_operational_pattern_shift(ctx: Dict) -> Optional[Dict]:
    if ctx['run_pct_30d'] > 0.1:
        diff = abs(ctx['run_pct_7d'] - ctx['run_pct_30d'])
        if diff > 0.25:
            direction = "Running Less" if ctx['run_pct_7d'] < ctx['run_pct_30d'] else "Running More"
            return {
                "code": "PATTERN_SHIFT",
                "category": "OPERATIONAL",
                "type": f"Operational Pattern Change ({direction})",
                "severity": "Low",
                "impact_value": 0,
                "impact_unit": "Review Required",
                "context": f"Runtime changed from {ctx['run_pct_30d']*100:.0f}% (Hist) to {ctx['run_pct_7d']*100:.0f}% (Recent).",
                "drivers": "Controller change, power issues, or pump sticking.",
                "chart_type": "sensor_trend",
                "sensor_metric": "strokes_per_minute"
            }
    return None

def check_pressure_instability(ctx: Dict) -> Optional[Dict]:
    if ctx['tp_avg'] > 50:
        if ctx['tp_std'] > (0.2 * ctx['tp_avg']):
            return {
                "code": "PRESSURE_INSTABILITY",
                "category": "OPERATIONAL",
                "type": "Flow Instability / Slugging",
                "severity": "Moderate",
                "impact_value": 0,
                "impact_unit": "Efficiency Risk",
                "context": f"Tubing Pressure is highly volatile (StdDev: {ctx['tp_std']:.1f} psi).",
                "drivers": "Gas slugging, heading, or valve hunting.",
                "chart_type": "sensor_trend",
                "sensor_metric": "tubing_pressure"
            }
    return None

def check_financial_gap(ctx: Dict) -> Optional[Dict]:
    if ctx['is_active'] and (ctx['pred_oil'] > 5) and (ctx['actual_oil'] < (0.75 * ctx['pred_oil'])):
        gap = ctx['pred_oil'] - ctx['actual_oil']
        impact = gap * OIL_PRICE
        return {
            "code": "FINANCIAL_EFFICIENCY",
            "category": "FINANCIAL",
            "type": "Production Efficiency Gap",
            "severity": "High" if impact > 500 else "Moderate",
            "impact_value": impact,
            "impact_unit": "USD/day",
            "context": f"Actual: {ctx['actual_oil']:.1f} bbl vs Model: {ctx['pred_oil']:.1f} bbl.",
            "drivers": "Underperformance vs Potential.",
            "chart_type": "oil_comparison",
            "sensor_metric": None
        }
    return None

def check_ghost_production(ctx: Dict) -> Optional[Dict]:
    total = ctx['actual_oil'] + ctx['actual_water'] + ctx['actual_gas']
    if ctx['is_active'] and (total < 0.1):
        impact = ctx['avg_oil'] * OIL_PRICE
        if impact == 0: impact = 100.0
        return {
            "code": "GHOST_PROD",
            "category": "PROCESS",
            "type": "Missing Production Report",
            "severity": "Low",
            "impact_value": impact,
            "impact_unit": "USD/day",
            "context": f"Well ACTIVE ({ctx['activity_source']}) but Zero Production reported.",
            "drivers": "Missing Daily Report or Meter Failure.",
            "chart_type": "production_bar",
            "sensor_metric": None
        }
    return None

def check_cost_creep(ctx: Dict) -> Optional[Dict]:
    if (ctx['avg_water'] > 10) and (ctx['actual_water'] > (1.25 * ctx['avg_water'])) and (ctx['actual_oil'] <= ctx['avg_oil']):
        excess_water = ctx['actual_water'] - ctx['avg_water']
        cost = excess_water * WATER_DISPOSAL_COST
        return {
            "code": "COST_CREEP",
            "category": "FINANCIAL",
            "type": "Rising Disposal Costs",
            "severity": "Moderate",
            "impact_value": cost,
            "impact_unit": "USD/day",
            "context": f"Water up {((ctx['actual_water']/ctx['avg_water'])-1)*100:.0f}% vs Avg.",
            "drivers": "Water Breakthrough.",
            "chart_type": "water_trend",
            "sensor_metric": None
        }
    return None

def check_flowline_blockage(ctx: Dict) -> Optional[Dict]:
    if (ctx['tp_avg'] > 0) and (ctx['tubing_pressure'] > 1.25 * ctx['tp_avg']) and (ctx['actual_oil'] < 0.85 * ctx['avg_oil']):
        impact = (ctx['avg_oil'] - ctx['actual_oil']) * OIL_PRICE
        return {
            "code": "FLOWLINE_BLOCK",
            "category": "OPERATIONAL",
            "type": "Flowline Blockage",
            "severity": "High",
            "impact_value": impact,
            "impact_unit": "USD/day Risk",
            "context": f"Pressure High (+{((ctx['tubing_pressure']/ctx['tp_avg'])-1)*100:.0f}%) & Oil Down.",
            "drivers": "Closed Choke/Valve or Paraffin.",
            "chart_type": "sensor_trend",
            "sensor_metric": "tubing_pressure"
        }
    return None

def check_pump_wear(ctx: Dict) -> Optional[Dict]:
    if (ctx['spm'] > ctx['spm_avg']) and (ctx['actual_oil'] < 0.8 * ctx['pred_oil']):
        impact = (ctx['pred_oil'] - ctx['actual_oil']) * OIL_PRICE
        return {
            "code": "PUMP_WEAR",
            "category": "OPERATIONAL",
            "type": "Pump Wear / Slippage",
            "severity": "Moderate",
            "impact_value": impact,
            "impact_unit": "USD/day Lost",
            "context": "Pump running faster than average, but production is low.",
            "drivers": "Worn Plunger/Barrel or Traveling Valve.",
            "chart_type": "oil_comparison",
            "sensor_metric": None
        }
    return None

def check_prod_instability(ctx: Dict) -> Optional[Dict]:
    if ctx['is_active'] and (ctx['avg_oil'] > 5) and (ctx['actual_oil'] < 0.7 * ctx['avg_oil']):
        if abs(ctx['tubing_pressure'] - ctx['tp_avg']) < (0.15 * ctx['tp_avg']):
            impact = (ctx['avg_oil'] - ctx['actual_oil']) * OIL_PRICE
            return {
                "code": "PROD_INSTABILITY",
                "category": "OPERATIONAL",
                "type": "Unexplained Production Drop",
                "severity": "Moderate",
                "impact_value": impact,
                "impact_unit": "USD/day Risk",
                "context": f"Oil dropped {(1-(ctx['actual_oil']/ctx['avg_oil']))*100:.0f}% with stable pressures.",
                "drivers": "Hole in Tubing or Check Valve Leak.",
                "chart_type": "oil_comparison",
                "sensor_metric": None
            }
    return None

def check_bsw_spike(ctx: Dict) -> Optional[Dict]:
    if (ctx['bsw'] > 10) and (ctx['bsw'] > ctx['bsw_avg'] + 20):
        return {
            "code": "BSW_SPIKE",
            "category": "OPERATIONAL",
            "type": "Water Cut Spike (BSW)",
            "severity": "Moderate",
            "impact_value": 0.0,
            "impact_unit": "Quality Risk",
            "context": f"Water Cut spiked to {ctx['bsw']:.1f}% (Avg {ctx['bsw_avg']:.1f}%).",
            "drivers": "Reservoir Watering Out.",
            "chart_type": "water_trend",
            "sensor_metric": None
        }
    return None

def build_chart_data(df: pd.DataFrame, chart_type: str, sensor_metric: Optional[str] = None) -> Dict:
    plot = df.tail(14).copy()
    if plot.empty: 
        return {"labels": [], "datasets": []}
    
    labels = plot['timestamp'].dt.strftime('%b %d').tolist()
    
    def clean(s): 
        return [None if pd.isna(x) else x for x in s]

    datasets = []
    
    if chart_type == "sensor_trend":
        if sensor_metric and sensor_metric in plot.columns and not plot[sensor_metric].isnull().all():
            datasets.append({
                "label": sensor_metric.replace('_', ' ').title(), 
                "data": clean(plot[sensor_metric].round(1)), 
                "borderColor": "#FEB019", 
                "type": "line"
            })
        else:
            datasets.append({
                "label": "Oil Volume (Context)", 
                "data": clean(plot['oil_volume'].round(1)), 
                "borderColor": "#008FFB", 
                "type": "bar"
            })
            
    elif chart_type == "oil_comparison":
        if 'predicted_oil' in plot and not plot['predicted_oil'].isnull().all():
            datasets.append({
                "label": "Model", 
                "data": clean(plot['predicted_oil'].round(1)), 
                "borderColor": "#00E396", 
                "type": "line"
            })
        datasets.append({
            "label": "Actual Oil", 
            "data": clean(plot['oil_volume'].round(1)), 
            "borderColor": "#FF4560", 
            "type": "line"
        })
        
    elif chart_type == "water_trend":
        datasets.append({
            "label": "Water", 
            "data": clean(plot['water_volume'].round(1)), 
            "borderColor": "#008FFB", 
            "type": "line"
        })
        
    elif chart_type == "production_bar":
        total = (plot['oil_volume'] + plot['water_volume'] + plot['gas_volume']).round(1)
        datasets.append({
            "label": "Total Fluids", 
            "data": clean(total), 
            "type": "bar", 
            "backgroundColor": "#008FFB"
        })
        
    return {"labels": labels, "datasets": datasets}

def generate_narrative(anom: Dict, lift_type: str) -> Dict:
    try:
        client = get_openai_client()
        if not client: raise Exception("No Client")
        
        prompt = f"""
        Analyze this {lift_type} anomaly.
        Type: {anom['type']} ({anom['category']})
        Details: {anom['context']}
        Drivers: {anom['drivers']}
        
        Return JSON with keys: description, why, root_cause.
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {
            "description": f"{anom['type']} detected. {anom['context']}",
            "why": "Significant deviation impacting production or safety.",
            "root_cause": anom['drivers']
        }

def detect_anomalies(well_id: str, df: Optional[pd.DataFrame] = None, lookback_days: int = 7) -> List[Dict]:
    if df is None: 
        df = fetch_well_data(well_id)
    if df.empty or len(df) < 14: 
        return []

    df['lift_type'] = df['lift_type'].ffill()
    
    df['oil_avg'] = df['oil_volume'].rolling(30, min_periods=5).mean()
    df['water_avg'] = df['water_volume'].rolling(30, min_periods=5).mean()
    df['gas_avg'] = df['gas_volume'].rolling(30, min_periods=5).mean()
    
    if 'tubing_pressure' in df: 
        df['tp_avg'] = df['tubing_pressure'].rolling(30).mean()
        df['tp_std'] = df['tubing_pressure'].rolling(7).std()
    
    if 'motor_current' in df: df['amps_avg'] = df['motor_current'].rolling(30).mean()
    if 'strokes_per_minute' in df: 
        df['spm_avg'] = df['strokes_per_minute'].rolling(30).mean()
        df['is_running'] = (df['strokes_per_minute'] > 0.1).astype(int)
        df['run_pct_7d'] = df['is_running'].rolling(7).mean()
        df['run_pct_30d'] = df['is_running'].rolling(30).mean()

    total_fluid = df['oil_volume'] + df['water_volume']
    df['bsw'] = (df['water_volume'] / total_fluid.replace(0, np.nan)) * 100
    df['bsw'] = df['bsw'].fillna(0)
    df['bsw_avg'] = df['bsw'].rolling(30).mean()

    model_art = get_or_train_model(well_id, df)
    df['predicted_oil'] = np.nan
    if model_art:
        try:
            X = model_art['scaler'].transform(model_art['imputer'].transform(df[model_art['features']]))
            df['predicted_oil'] = model_art['model'].predict(X)
        except Exception: 
            pass

    checkers = [
        check_operational_pattern_shift,
        check_pressure_instability,
        check_financial_gap, 
        check_ghost_production, 
        check_cost_creep,
        check_flowline_blockage, 
        check_pump_wear,
        check_prod_instability,
        check_bsw_spike
    ]
    
    window = df[df['timestamp'] > (df['timestamp'].max() - pd.Timedelta(days=lookback_days))]
    anomalies = []
    
    for _, row in window.iterrows():
        ctx = row.to_dict()
        ctx['is_active'] = (ctx.get('strokes_per_minute',0) > 0.1) or (ctx.get('motor_current',0) > 5)
        ctx['activity_source'] = "Sensors"
        
        ctx['actual_oil'] = ctx.get('oil_volume',0)
        ctx['actual_water'] = ctx.get('water_volume',0)
        ctx['actual_gas'] = ctx.get('gas_volume',0)
        
        ctx['avg_oil'] = ctx.get('oil_avg', 0) if pd.notnull(ctx.get('oil_avg')) else 0
        ctx['avg_water'] = ctx.get('water_avg', 0) if pd.notnull(ctx.get('water_avg')) else 0
        ctx['avg_gas'] = ctx.get('gas_avg', 0) if pd.notnull(ctx.get('gas_avg')) else 0
        
        ctx['pred_oil'] = ctx.get('predicted_oil', ctx['actual_oil'])
        if pd.isna(ctx['pred_oil']): ctx['pred_oil'] = ctx['actual_oil']
        
        ctx['tubing_pressure'] = ctx.get('tubing_pressure',0)
        ctx['tp_avg'] = ctx.get('tp_avg',0)
        ctx['tp_std'] = ctx.get('tp_std',0)
        
        ctx['spm'] = ctx.get('strokes_per_minute',0)
        ctx['spm_avg'] = ctx.get('spm_avg',0)
        
        ctx['run_pct_7d'] = ctx.get('run_pct_7d', 0)
        ctx['run_pct_30d'] = ctx.get('run_pct_30d', 0)
        
        ctx['amps'] = ctx.get('motor_current',0)
        ctx['amps_avg'] = ctx.get('amps_avg',0)
        
        ctx['bsw'] = ctx.get('bsw', 0)
        ctx['bsw_avg'] = ctx.get('bsw_avg', 0)
        
        current_lift = str(row['lift_type']) if pd.notnull(row['lift_type']) else 'Rod Pump'

        for check in checkers:
            res = check(ctx)
            if res:
                if any(x['anomaly_code'] == res['code'] and x['event_date'] == str(row['timestamp'].date()) for x in anomalies):
                    continue
                
                narrative = generate_narrative(res, current_lift)
                chart = build_chart_data(df, res['chart_type'], res['sensor_metric'])
                
                anomalies.append({
                    "well_id": well_id,
                    "event_date": str(row['timestamp'].date()),
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "anomaly_code": res['code'],
                    "category": res['category'],
                    "severity": res['severity'],
                    "title": res['type'],
                    "status": "ACTIVE",
                    "ui_text": {
                        "description": narrative['description'],
                        "why_is_this_an_anomaly": narrative['why'],
                        "suspected_root_cause": narrative['root_cause'],
                        "economic_impact": f"{res['impact_unit']} {res['impact_value']:.0f}"
                    },
                    "impact_metrics": {"value": res['impact_value'], "unit": res['impact_unit']},
                    "chart_data": chart
                })
                
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
                logger.info(f"Saved {count} anomalies to Snowflake.")
    except Exception as e:
        logger.error(f"Snowflake Save failed: {e}")

if __name__ == "__main__":
    import sys
    wid = sys.argv[1] if len(sys.argv) > 1 else "Well_001"
    print(json.dumps(detect_anomalies(wid), indent=2, default=str))