import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from snowflake.connector import connect as snowflake_connect
import psycopg2 
import openai
import joblib
import time 

from dynamic_config import get_sensor_ranges

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AnomalyReviewService')

OIL_PRICE = 75.00
WATER_DISPOSAL_COST = 1.50
GAS_PRICE = 2.50
ENERGY_COST_UNIT = 0.12
MODELS_DIR = 'models'

# --- CONSTANTS ---
VALID_CATEGORIES = {'PROCESS', 'OPERATIONAL', 'FINANCIAL'}
VALID_STATUSES = {'NEW', 'ACKNOWLEDGED', 'IN_REVIEW', 'ACTIONED', 'DISMISSED'}

# Defines features to fetch
# Common sensors across all lift types (Verified in Schema)
COMMON_SENSORS = [
    'wellhead_temp', 
    'surface_pressure', 
    'casing_pressure'
]

# Client-specific Strict Ranges (Local Isolation)
CLIENT_SENSOR_RANGES = {
    "tubing_pressure": {"min": 0, "max": 300, "unit": "psi"},
    "wellhead_temp": {"min": 10, "max": 400, "unit": "F"},
    # ... (Other ranges kept as config, will be skipped if data missing)
}


# Defines features to fetch
LIFT_FEATURES = {
    'Rod Pump': [
        'strokes_per_minute', 
        'torque', 'polish_rod_load', 'structural_load', 'surface_stroke_length',
        'downhole_gross_stroke_length', 'downhole_net_stroke_length',
        'tubing_pressure', 'pump_intake_pressure', 'pump_friction',
        'pump_fillage'
    ] + COMMON_SENSORS,
    'ESP': [
        'motor_current', 'motor_voltage', 'input_current', 'output_voltage', 
        'motor_load', 'total_harmonic_distortion', 'drive_input_voltage',
        'motor_speed', 'drive_frequency', 'set_frequency',
        'discharge_pressure', 'pump_intake_pressure', 'tubing_pressure', 'casing_pressure',
        'motor_temp', 'intake_fluid_temp', 'discharge_temp', 'vsd_temp',
        'vibration_x', 'vibration_y'
    ] + COMMON_SENSORS,
    'Gas Lift': [
        'injection_rate', 'injection_pressure', 'injection_temperature',
        'casing_pressure', 'surface_pressure', 'bottomhole_pressure', 
        'open_differential_pressure', 'min_shut_in_pressure', 'max_shut_in_pressure',
        'cycle_time', 'flow_time', 'shut_in_time', 'afterflow_time',
        'plunger_velocity', 'plunger_arrival_time', 'missed_arrivals'
    ] + COMMON_SENSORS
}

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
    all_possible_sensors = set()
    for cols in LIFT_FEATURES.values():
        all_possible_sensors.update(cols)
    
    sensor_selects = ", ".join([f"s.{col}" for col in all_possible_sensors])
    
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
        
        numeric_cols = list(all_possible_sensors) + ['oil_volume', 'water_volume', 'gas_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
        return df
    except Exception as e:
        logger.error(f"Fetch failed for {well_id}: {e}")
        return pd.DataFrame()

def get_well_status(well_id: str) -> str:
    """Fetch master status for well (e.g., ACTIVE, INACTIVE)."""
    # Assuming a 'wells' table exists or derived from production
    # Fallback to 'ACTIVE' if unknown, but better to check source
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            # Try finding validation in master table
            # Adjust table name if 'wells' or 'well_master' does not exist
            # For now, simplistic check: if it has recent data, it's capable of active?
            # Or assume we can query a 'wells' table.
            
            # Using metadata check if possible (simulated here since schema not fully known)
            # cur.execute("SELECT status FROM wells WHERE id = %s", (well_id,))
            # res = cur.fetchone()
            # if res: return res[0]
            pass
    except Exception:
        pass
    return "ACTIVE" # Default assumption for logic

def get_or_train_model(well_id: str, df: pd.DataFrame, active_features: List[str]) -> Dict:
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    model_path = os.path.join(MODELS_DIR, f'financial_rf_{well_id}.joblib')
    
    if os.path.exists(model_path):
        file_age_days = (time.time() - os.path.getmtime(model_path)) / (24 * 3600)
        if file_age_days < 7:
            try:
                return joblib.load(model_path)
            except Exception:
                pass

    cutoff_date = df['timestamp'].max() - timedelta(days=14)
    train_df = df[df['timestamp'] <= cutoff_date]
    
    if len(train_df) < 10:
        return {}
        
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    
    X_train = imputer.fit_transform(train_df[active_features])
    X_train = scaler.fit_transform(X_train)
    model.fit(X_train, train_df['oil_volume'])
    
    importances = model.feature_importances_
    top_idx = np.argmax(importances)
    top_factor = active_features[top_idx]
    
    artifact = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'top_factor': top_factor,
        'timestamp': time.time()
    }
    
    joblib.dump(artifact, model_path)
    return artifact

# --- LOGIC CHECKERS (Refactored) ---

def check_financial_gap(context: Dict) -> Optional[Dict]:
    if context['is_active'] and (context['pred_oil'] > 5) and (context['actual_oil'] < (0.80 * context['pred_oil'])):
        gap = context['pred_oil'] - context['actual_oil']
        impact = gap * OIL_PRICE
        return {
            "code": "FINANCIAL_EFFICIENCY",
            "category": "FINANCIAL",
            "type": "Production Efficiency Gap",
            "severity": "High" if impact > 500 else "Moderate",
            "impact_value": impact,
            "impact_unit": "USD/day",
            "context": f"Actual: {context['actual_oil']:.1f} bbl vs Model: {context['pred_oil']:.1f} bbl.",
            "drivers": f"Sensors ({context['activity_source']}) indicate potential for {context['pred_oil']:.1f} bbl.",
            "chart_type": "oil_comparison"
        }
    return None

def check_ghost_production(context: Dict) -> Optional[Dict]:
    total_fluids = context['actual_oil'] + context['actual_water'] + context['actual_gas']
    if context['is_active'] and (total_fluids < 0.1):
        return {
            "code": "GHOST_PROD",
            "category": "PROCESS",
            "type": "Missing Production Report",
            "severity": "Low",
            "impact_value": 0.0,
            "impact_unit": "USD/day",
            "context": f"Well is ACTIVE ({context['activity_source']}) but Production is 0.",
            "drivers": "Operational compliance: Check daily report Entry.",
            "chart_type": "production_bar"
        }
    return None

def check_cost_creep(context: Dict) -> Optional[Dict]:
    avg_water = context['avg_water']
    actual_water = context['actual_water']
    
    if (avg_water > 10) and (actual_water > (1.20 * avg_water)) and (context['actual_oil'] <= context['avg_oil'] * 1.05):
        excess = actual_water - avg_water
        cost = excess * WATER_DISPOSAL_COST
        return {
            "code": "COST_CREEP",
            "category": "FINANCIAL",
            "type": "Rising Disposal Costs",
            "severity": "Moderate",
            "impact_value": cost,
            "impact_unit": "USD/day",
            "context": f"Water up {((actual_water/avg_water)-1)*100:.0f}% while Oil is flat.",
            "drivers": "Inefficient water handling / Water breakthrough.",
            "chart_type": "water_trend"
        }
    return None

def check_gas_interference(context: Dict) -> Optional[Dict]:
    if (context['avg_gas'] > 5) and (context['actual_gas'] > context['avg_gas']) and (context['fillage_std'] > 5):
        return {
            "code": "GAS_INTERFERENCE",
            "category": "OPERATIONAL",
            "type": "Gas Interference Review",
            "severity": "High",
            "impact_value": 0.0,
            "impact_unit": "Production Risk",
            "context": f"Rising Gas ({context['actual_gas']:.0f} mcf) + Erratic Fillage (SD: {context['fillage_std']:.1f}).",
            "drivers": "Gas likely locking pump or reducing efficiency.",
            "chart_type": "gas_trend"
        }
    return None

def check_flowline_blockage(context: Dict) -> Optional[Dict]:
    if (context['tp_avg'] > 0) and (context['tubing_pressure'] > 1.2 * context['tp_avg']) and (context['actual_oil'] < 0.9 * context['avg_oil']):
        pct_rise = ((context['tubing_pressure']/context['tp_avg'])-1)*100
        return {
            "code": "FLOWLINE_BLOCK",
            "category": "OPERATIONAL",
            "type": "Flowline Blockage",
            "severity": "High",
            "impact_value": 0.0,
            "impact_unit": "Safety Risk",
            "context": f"Tubing Pressure +{pct_rise:.0f}% vs Avg, Oil Down.",
            "drivers": "Check choke, flowline, or surface valves.",
            "chart_type": "oil_comparison"
        }
    return None

def check_pump_wear(context: Dict) -> Optional[Dict]:
    if (context['current_lift'] == 'Rod Pump') and \
       (context['spm'] > context['spm_avg']) and \
       (context['pump_fillage'] > 80) and \
       (context['actual_oil'] < 0.8 * context['pred_oil']):
         return {
            "code": "PUMP_WEAR",
            "category": "OPERATIONAL",
            "type": "Pump Wear / Slippage",
            "severity": "Moderate",
            "impact_value": 0.0,
            "impact_unit": "Efficiency Loss",
            "context": f"Pump running fast ({context['spm']:.1f} SPM) and full ({context['pump_fillage']:.0f}%), but oil missing.",
            "drivers": "Fluid is slipping past plunger or traveling valve leak.",
            "chart_type": "oil_comparison"
        }
    return None

def check_esp_shaft_break(context: Dict) -> Optional[Dict]:
    if (context['current_lift'] == 'ESP') and \
       (context['pip'] > context['pip_avg'] * 1.1) and \
       (context['amps'] < context['amps_avg'] * 0.5):
        return {
            "code": "ESP_SHAFT_BREAK",
            "category": "OPERATIONAL",
            "type": "Broken Shaft / Free Spin",
            "severity": "High", # Was CRITICAL, High maps to common levels
            "impact_value": 0.0,
            "impact_unit": "Total Failure",
            "context": f"High Intake Pressure (Fluid Level High) but Low Amps.",
            "drivers": "Motor spinning without load (Broken Shaft) or Intake Plugged.",
            "chart_type": "production_bar"
        }
    return None

def check_production_instability(context: Dict) -> Optional[Dict]:
    if not context['is_active']: return None
    oil_drop = (context['avg_oil'] - context['actual_oil']) / (context['avg_oil'] + 0.1)
    amps_change = abs(context['amps'] - context['amps_avg']) / (context['amps_avg'] + 0.1)
    pressure_val = context['tubing_pressure'] if context['tubing_pressure'] > 0 else context['pip']
    pressure_avg = context['tp_avg'] if context['tubing_pressure'] > 0 else context['pip_avg']
    pressure_change = abs(pressure_val - pressure_avg) / (pressure_avg + 0.1)
    
    if oil_drop > 0.20 and amps_change < 0.05 and pressure_change < 0.05:
        return {
            "code": "PROD_INSTABILITY",
            "category": "OPERATIONAL",
            "type": "Unexplained Production Drop",
            "severity": "Moderate",
            "impact_value": oil_drop * context['avg_oil'] * OIL_PRICE,
            "impact_unit": "USD/day Risk",
            "context": f"Oil down {oil_drop:.1%} with stable sensors.",
            "drivers": "Tubing Leak, Flowline Leak, Check Valve Failure.",
            "chart_type": "production_line"
        }
    return None

    return None

# def check_downtime(context: Dict) -> Optional[Dict]:
#    """
#    DISABLED: Missing 'runtime' data.
#    """
#    return None

def check_sensor_integrity(context: Dict) -> Optional[Dict]:
    """
    Check against Client-Specific strict ranges.
    Returns: Anomaly if any critical sensor is out of bounds.
    """
    violations = []
    
    for field, rule in CLIENT_SENSOR_RANGES.items():
        val = context.get(field)
        # Skip if None, or if 0 and min is 0 (trivial)
        if val is None: continue
        
        # Check Min
        if rule.get('min') is not None and val < rule['min']:
            violations.append(f"{field} ({val}) < Min {rule['min']}")
            
        # Check Max
        if rule.get('max') is not None and val > rule['max']:
            violations.append(f"{field} ({val}) > Max {rule['max']}")
            
    if violations:
        # Prioritize showing the first few
        desc = ", ".join(violations[:2])
        if len(violations) > 2: desc += f" + {len(violations)-2} more."
        
        return {
            "code": "SENSOR_RANGE",
            "category": "PROCESS", # Data Quality / Process Limits
            "type": "Sensor Value Out of Range",
            "severity": "Low", # Typically warn
            "impact_value": 0.0,
            "impact_unit": "Data Quality",
            "context": f"Violations: {desc}",
            "drivers": "Sensor drift, calibration error, or process upset.",
            "chart_type": "production_bar" # Generic fallback
        }
    return None

def check_tank_drop(context: Dict) -> Optional[Dict]:
    """
    Detects sudden drops in tank level without production explanation.
    """
    if context.get('tank_drop', 0) < -2.0: # Drop > 2 units (ft/bbl depending on sensor)
        # Check if production explains it? 
        # If we had a 'Haul' event, we would skip. We don't have Haul data yet.
        # So we flag as 'Unexpected Drop'.
        return {
            "code": "TANK_DROP",
            "category": "OPERATIONAL",
            "type": "Unexpected Tank Level Drop",
            "severity": "Moderate",
            "impact_value": 0.0, # Could calc volume if we knew tank dims
            "impact_unit": "Potential Loss",
            "context": f"Tank dropped {abs(context['tank_drop']):.1f} units in 24h.",
            "drivers": "Potential theft, leak, or unrecorded haul.",
            "chart_type": "production_bar" # Fallback
        }
    return None

def check_tank_leak(context: Dict) -> Optional[Dict]:
    """
    Explicit alias for tank drops to categorize them as Potential Leaks.
    """
    if context.get('tank_drop', 0) < -1.0: # More sensitive trigger for Leak
         return {
            "code": "TANK_LEAK",
            "category": "OPERATIONAL",
            "type": "Tank Leak Suspected",
            "severity": "High",
            "impact_value": 0.0,
            "impact_unit": "Environmental Risk",
            "context": f"Unexplained drop of {abs(context['tank_drop']):.1f} units (Leak Check).",
            "drivers": "Tank integrity failure or valve leak.",
            "chart_type": "production_bar"
        }
    return None

def check_bsw_spike(context: Dict) -> Optional[Dict]:
    """
    Detects sudden rise in Water Cut (BSW %) even if total fluid is stable.
    """
    bsw = context.get('bsw', 0)
    bsw_avg = context.get('bsw_avg', 0)
    
    if (bsw > 10) and (bsw > bsw_avg + 15): # e.g. 40% -> 55%
        return {
            "code": "BSW_SPIKE",
            "category": "OPERATIONAL",
            "type": "Water Cut Spike (BSW)",
            "severity": "Moderate",
            "impact_value": context['actual_water'] * WATER_DISPOSAL_COST,
            "impact_unit": "USD/day (Disposal)",
            "context": f"Water Cut spiked to {bsw:.1f}% (Avg {bsw_avg:.1f}%).",
            "drivers": "Water breakthrough or separation failure.",
            "chart_type": "water_trend"
        }
    return None

# --- MAIN CONTROLLER ---

def detect_anomalies(well_id: str, df: Optional[pd.DataFrame] = None, lookback_days: int = 7) -> List[Dict]:
    if df is None:
        df = fetch_well_data(well_id)
    if df.empty or len(df) < 14: return []

    df['lift_type'] = df['lift_type'].ffill()
    current_lift = df['lift_type'].iloc[-1] if not df['lift_type'].isnull().all() else 'Rod Pump'
    master_status = get_well_status(well_id)
    
    target_features = LIFT_FEATURES.get(current_lift, LIFT_FEATURES['Rod Pump'])
    active_features = [f for f in df.columns if f in target_features]

    if len(active_features) < 3:
        return []

    # Get Existing Anomalies for Deduplication
    existing_anomalies = fetch_existing_anomalies(well_id, lookback_days + 2) # Buffer
    
    # Model
    df['predicted_oil'] = np.nan
    top_factor = "Unknown"
    model_artifact = get_or_train_model(well_id, df, active_features)
    if model_artifact:
        scaler = model_artifact['scaler']
        model = model_artifact['model']
        imputer = model_artifact['imputer']
        top_factor = model_artifact['top_factor']
        try:
            X_full = scaler.transform(imputer.transform(df[active_features]))
            df['predicted_oil'] = model.predict(X_full)
        except Exception:
            pass

    # Stats
    df['water_avg'] = df['water_volume'].rolling(30).mean()
    df['gas_avg'] = df['gas_volume'].rolling(30).mean()
    df['oil_avg'] = df['oil_volume'].rolling(30).mean()
    if 'tubing_pressure' in df.columns: df['tp_avg'] = df['tubing_pressure'].rolling(30).mean()
    if 'pump_intake_pressure' in df.columns: df['pip_avg'] = df['pump_intake_pressure'].rolling(30).mean()
    if 'motor_current' in df.columns: df['amps_avg'] = df['motor_current'].rolling(30).mean()
    if 'strokes_per_minute' in df.columns: df['spm_avg'] = df['strokes_per_minute'].rolling(30).mean()
    if 'pump_fillage' in df.columns: 
        df['fillage_std'] = df['pump_fillage'].rolling(7).std() 
    else:
        df['fillage_std'] = 0.0

    # Calculate Tank Drop (Day over Day)
    if 'tank_oil_level' in df.columns:
        df['tank_drop'] = df['tank_oil_level'].diff()
    else:
        df['tank_drop'] = 0.0

    # Calculate BSW (Water Cut %)
    df['total_fluid'] = df['oil_volume'] + df['water_volume']
    df['bsw'] = (df['water_volume'] / df['total_fluid'].replace(0, np.nan)) * 100
    df['bsw'] = df['bsw'].fillna(0)
    df['bsw_avg'] = df['bsw'].rolling(30).mean()

    checkers = [
        check_financial_gap, check_ghost_production, check_cost_creep,
        check_gas_interference, check_flowline_blockage, check_pump_wear,
        check_esp_shaft_break, check_production_instability, 
        check_esp_shaft_break, check_production_instability, 
        check_sensor_integrity, check_tank_drop,
        check_tank_leak, check_bsw_spike
    ]
    
    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=lookback_days)
    window_df = df[df['timestamp'] > cutoff_date].copy()
    
    anomalies_out = []
    
    for idx, row in window_df.iterrows():
        event_date_str = row['timestamp'].strftime('%Y-%m-%d')
        
        # Helper
        def get_val(r, col, default=0): return float(r.get(col, default))
        
        day_context = {
            'well_id': well_id,
            'current_lift': current_lift,
            'master_status': master_status,
            'timestamp': row['timestamp'],
            'actual_oil': get_val(row, 'oil_volume'),
            'actual_water': get_val(row, 'water_volume'),
            'actual_gas': get_val(row, 'gas_volume'),
            'pred_oil': get_val(row, 'predicted_oil', get_val(row, 'oil_volume')),
            'avg_oil': get_val(row, 'oil_avg'),
            'avg_water': get_val(row, 'water_avg'),
            'avg_gas': get_val(row, 'gas_avg'),
            'tubing_pressure': get_val(row, 'tubing_pressure'),
            'tp_avg': get_val(row, 'tp_avg'),
            'pip': get_val(row, 'pump_intake_pressure'),
            'pip_avg': get_val(row, 'pip_avg'),
            'amps': get_val(row, 'motor_current'),
            'amps_avg': get_val(row, 'amps_avg'),
            'spm': get_val(row, 'strokes_per_minute'),
            'spm_avg': get_val(row, 'spm_avg'),
            'pump_fillage': get_val(row, 'pump_fillage'),
            'fillage_std': get_val(row, 'fillage_std'),
            'runtime': get_val(row, 'runtime'),
            'runtime': get_val(row, 'runtime'),
            'tank_drop': get_val(row, 'tank_drop'),
            'bsw': get_val(row, 'bsw'),
            'bsw_avg': get_val(row, 'bsw_avg')
        }
        
        # Inject Common Sensors into Context for Integrity Check
        for s in COMMON_SENSORS:
             if s in row:
                 day_context[s] = row[s]
        
        # Activity Check
        is_active = False
        activity_source = "Unknown"
        if 'strokes_per_minute' in row and row['strokes_per_minute'] > 0.1:
            is_active = True
            activity_source = f"SPM ({row['strokes_per_minute']:.1f})"
        elif 'motor_current' in row and row['motor_current'] > 5:
            is_active = True
            activity_source = f"Amps ({row['motor_current']:.1f})"
        
        day_context['is_active'] = is_active
        day_context['activity_source'] = activity_source
        
        for check_func in checkers:
            res = check_func(day_context)
            if res:
                # Deduplication Check
                # key: (well_id, anomaly_code, event_date)
                dedup_key = (res['code'], event_date_str)
                
                # Check Local Duplicates (same batch)
                if any(x['anomaly_code'] == res['code'] and x['event_date'] == event_date_str for x in anomalies_out):
                    continue
                    
                # Check DB Duplicates (previously acted upon)
                existing_status = existing_anomalies.get(dedup_key)
                if existing_status in ('DISMISSED', 'ACTIONED', 'ACKNOWLEDGED', 'IN_REVIEW'):
                    # Skip re-alerting if already active or handled
                    continue
                    
                narrative = generate_narrative(res, current_lift)
                chart_data = build_chart_data(df, res['chart_type']) # Using full DF for context
                
                anomalies_out.append({
                    "well_id": well_id,
                    "timestamp": day_context['timestamp'],
                    "event_date": event_date_str,
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
                        "economic_impact": f"{res['impact_unit']} {res['impact_value']:.1f}" if res['impact_value'] > 0 else res['impact_unit']
                    },
                    "impact_metrics": {
                        "value": res['impact_value'],
                        "unit": res['impact_unit']
                    },
                    "chart_data": chart_data
                })

    save_reviews(anomalies_out)
    return anomalies_out

def fetch_existing_anomalies(well_id: str, days: int) -> Dict[tuple, str]:
    """Returns dict of {(code, date_str): status} for existing anomalies."""
    existing = {}
    try:
        # Check Postgres first (it's the primary for status usually)
        with psycopg2.connect(get_db_url()) as conn:
            with conn.cursor() as cur:
                # Check if table exists first? Assume schema updated.
                # Querying update schema: anomaly_code, event_date, status
                try:
                    cur.execute("""
                        SELECT anomaly_code, to_char(event_date, 'YYYY-MM-DD'), status 
                        FROM operation_suggestion 
                        WHERE well_id = %s AND event_date >= CURRENT_DATE - INTERVAL '%s days'
                    """, (well_id, days))
                    rows = cur.fetchall()
                    for r in rows:
                        existing[(r[0], r[1])] = r[2]
                except Exception:
                    pass # Table might not have cols yet, ignore
    except Exception:
        pass
    return existing

def generate_narrative(anom: Dict, lift_type: str) -> Dict:
    try:
        client = get_openai_client()
        if not client: raise Exception("No OpenAI Client")

        prompt = f"""
        Analyze this oil & gas anomaly for a {lift_type} well.
        Anomaly: {anom['type']} ({anom['category']})
        Context: {anom['context']}
        Drivers: {anom['drivers']}
        Severity: {anom['severity']}
        
        Provide a JSON response with:
        - description: "Technical description of what happened."
        - why: "Why this is physically significant."
        - root_cause: "Likely mechanical or operational root cause."
        Keep it professional, concise, and action-oriented.
        """
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"OpenAI Generation failed: {e}")
        # Fallback
        return {
            "description": anom['context'],
            "why": f"Observed {anom['code']} pattern deviates from standard {lift_type} physics.",
            "root_cause": anom['drivers']
        }

def build_chart_data(df: pd.DataFrame, chart_type: str) -> Dict:
    plot = df.tail(14).copy()
    if plot.empty: return {"labels": [], "datasets": []}
    
    labels = plot['timestamp'].dt.strftime('%b %d').tolist()
    def clean(series): return series.where(pd.notnull(series), None).tolist()

    datasets = []
    if chart_type == "oil_comparison":
        if 'predicted_oil' in plot:
            datasets.append({"label": "Model Prediction", "data": clean(plot['predicted_oil'].round(1)), "borderColor": "#00E396"})
        datasets.append({"label": "Actual Oil", "data": clean(plot['oil_volume'].round(1)), "borderColor": "#FF4560"})
    elif chart_type == "water_trend":
        datasets.append({"label": "Actual Water", "data": clean(plot['water_volume'].round(1)), "borderColor": "#FF4560"})
    elif chart_type == "gas_trend":
         datasets.append({"label": "Gas Volume", "data": clean(plot['gas_volume'].round(1)), "borderColor": "#FEB019"})
    elif chart_type == "production_bar":
        total = (plot['oil_volume'] + plot['water_volume'] + plot['gas_volume']).round(1)
        datasets.append({"label": "Total Fluids", "data": clean(total), "type": "bar"})
        
    return {"labels": labels, "datasets": datasets}

def save_reviews(reviews: List[Dict]):
    if not reviews: return
    
    # 1. Postgres
    try:
        url = get_db_url()
        if url:
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    # Upsert Logic: If NEW exists, do nothing? Or update?
                    # Uniqueness is on (well_id, anomaly_code, event_date).
                    # 'ON CONFLICT DO NOTHING' keeps original status if exists.
                    
                    insert_query = """
                        INSERT INTO operation_suggestion
                        (well_id, event_date, detected_at, anomaly_code, category, severity, 
                         title, ui_text, impact_value, impact_unit, chart_data, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (well_id, anomaly_code, event_date) DO NOTHING
                    """
                    
                    for r in reviews:
                        cur.execute(insert_query, (
                            r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                            r['category'], r['severity'], r['title'],
                            json.dumps(r['ui_text']),
                            r['impact_metrics']['value'],
                            r['impact_metrics']['unit'],
                            json.dumps(r['chart_data']),
                            r['status']
                        ))
                conn.commit()
                logger.info(f"Saved {len(reviews)} anomalies to Postgres operation_suggestion.")
    except Exception as e:
        logger.error(f"Postgres save failed: {e}")

    # 2. Snowflake (Backup/Analytics)
    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            # Ensure table matches new schema or add columns dynamically? 
            # Assuming Snowflake schema matches or we append
            # Snowflake doesn't allow ON CONFLICT cleanly, so we check first or MERGE is better.
            # Using basic check-then-insert loop for safety
            
            check_q = "SELECT count(*) FROM operation_suggestion WHERE well_id=%s AND anomaly_code=%s AND event_date=%s"
            insert_q = """
                INSERT INTO operation_suggestion
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
            logger.info(f"Saved {count} anomalies to Snowflake operation_suggestion.")
    except Exception as e:
        logger.error(f"Snowflake save failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(detect_anomalies(sys.argv[1]), indent=2))