import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
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

# Add 'runtime' to active features for Rod Pump if available to fix "Blind Spot"
LIFT_FEATURES = {
    'Rod Pump': [
        'strokes_per_minute', 'cycles_per_day', 'runtime', 
        'torque', 'polish_rod_load', 'structural_load', 'surface_stroke_length',
        'downhole_gross_stroke_length', 'downhole_net_stroke_length',
        'tubing_pressure', 'pump_intake_pressure', 'pump_friction',
        'pump_fillage', 'inferred_production'
    ],
    'ESP': [
        'motor_current', 'motor_voltage', 'input_current', 'output_voltage', 
        'motor_load', 'total_harmonic_distortion', 'drive_input_voltage',
        'motor_speed', 'drive_frequency', 'set_frequency',
        'discharge_pressure', 'pump_intake_pressure', 'tubing_pressure', 'casing_pressure',
        'motor_temp', 'intake_fluid_temp', 'discharge_temp', 'vsd_temp',
        'vibration_x', 'vibration_y', 'runtime'
    ],
    'Gas Lift': [
        'injection_rate', 'injection_pressure', 'injection_temperature',
        'casing_pressure', 'surface_pressure', 'bottomhole_pressure', 
        'open_differential_pressure', 'min_shut_in_pressure', 'max_shut_in_pressure',
        'cycle_time', 'flow_time', 'shut_in_time', 'afterflow_time',
        'plunger_velocity', 'plunger_arrival_time', 'missed_arrivals'
    ]
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

def get_postgres_context(well_id: str) -> str:
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url: return ""
        
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT action, status FROM operation_suggestion 
                    WHERE well_id = %s AND status IN ('New', 'In Progress')
                    ORDER BY created_at DESC LIMIT 1
                """, (well_id,))
                row = cur.fetchone()
                if row: return f"Active Ticket: {row[0]} ({row[1]})."
    except Exception:
        pass
    return "No active tickets."

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

def validate_sensor_data(latest_row: pd.Series, lift_type: str) -> List[str]:
    """Checks if sensors are within DB defined Min/Max ranges."""
    invalid_sensors = []
    try:
        ranges = get_sensor_ranges(lift_type)
        for sensor, limits in ranges.items():
            if sensor in latest_row:
                val = latest_row[sensor]
                if pd.notnull(val):
                    if val < limits.get('min', -999999) or val > limits.get('max', 999999):
                        invalid_sensors.append(sensor)
    except Exception as e:
        logger.error(f"Range check failed: {e}")
    return invalid_sensors

def get_or_train_model(well_id: str, df: pd.DataFrame, active_features: List[str]) -> Dict:
    """
    Retrieves a cached model or trains a new one if missing/stale.
    Returns dictionary with 'model', 'scaler', 'imputer', 'top_factor'.
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    model_path = os.path.join(MODELS_DIR, f'financial_rf_{well_id}.joblib')
    
    # Check cache (7 days validity)
    if os.path.exists(model_path):
        file_age_days = (time.time() - os.path.getmtime(model_path)) / (24 * 3600)
        if file_age_days < 7:
            try:
                # logger.info(f"Loading cached model for {well_id}")
                return joblib.load(model_path)
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")

    # Train New Model
    # logger.info(f"Training new model for {well_id}")
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

# --- LOGIC CHECKER FUNCTIONS ---

def check_financial_gap(context: Dict) -> Optional[Dict]:
    if context['is_active'] and (context['pred_oil'] > 5) and (context['actual_oil'] < (0.80 * context['pred_oil'])):
        gap = context['pred_oil'] - context['actual_oil']
        impact = gap * OIL_PRICE
        return {
            "category": "FINANCIAL",
            "type": "Production Efficiency Gap",
            "severity": "High" if impact > 500 else "Moderate",
            "impact_str": f"${impact:,.0f}/day Revenue Risk",
            "context": f"Actual: {context['actual_oil']:.1f} bbl vs Model: {context['pred_oil']:.1f} bbl.",
            "drivers": f"Sensors ({context['activity_source']}) indicate potential for {context['pred_oil']:.1f} bbl.",
            "chart_type": "oil_comparison"
        }
    return None

def check_ghost_production(context: Dict) -> Optional[Dict]:
    total_fluids = context['actual_oil'] + context['actual_water'] + context['actual_gas']
    if context['is_active'] and (total_fluids < 0.1):
        return {
            "category": "PROCESS",
            "type": "Missing Production Report",
            "severity": "Low",
            "impact_str": "$0 (Data Gap)",
            "context": f"Well is ACTIVE ({context['activity_source']}) but Production is 0.",
            "drivers": "Operational compliance: Check daily report Entry.",
            "chart_type": "production_bar"
        }
    return None

def check_cost_creep(context: Dict) -> Optional[Dict]:
    # Water up > 20% AND Oil flat/down
    avg_water = context['avg_water']
    actual_water = context['actual_water']
    
    if (avg_water > 10) and (actual_water > (1.20 * avg_water)) and (context['actual_oil'] <= context['avg_oil'] * 1.05):
        excess = actual_water - avg_water
        cost = excess * WATER_DISPOSAL_COST
        return {
            "category": "FINANCIAL",
            "type": "Rising Disposal Costs",
            "severity": "Moderate",
            "impact_str": f"${cost:,.0f}/day Excess Cost",
            "context": f"Water up {((actual_water/avg_water)-1)*100:.0f}% while Oil is flat.",
            "drivers": "Inefficient water handling / Water breakthrough.",
            "chart_type": "water_trend"
        }
    return None

def check_gas_interference(context: Dict) -> Optional[Dict]:
    # Gas trending up AND Erratic Fillage
    if (context['avg_gas'] > 5) and (context['actual_gas'] > context['avg_gas']) and (context['fillage_std'] > 5):
        return {
            "category": "OPERATIONAL",
            "type": "Gas Interference Review",
            "severity": "High",
            "impact_str": "Production Risk",
            "context": f"Rising Gas ({context['actual_gas']:.0f} mcf) + Erratic Fillage (SD: {context['fillage_std']:.1f}).",
            "drivers": "Gas likely locking pump or reducing efficiency.",
            "chart_type": "gas_trend"
        }
    return None

def check_flowline_blockage(context: Dict) -> Optional[Dict]:
    # Tubing Pressure Up, Oil Down
    if (context['tp_avg'] > 0) and (context['tubing_pressure'] > 1.2 * context['tp_avg']) and (context['actual_oil'] < 0.9 * context['avg_oil']):
        pct_rise = ((context['tubing_pressure']/context['tp_avg'])-1)*100
        return {
            "category": "OPERATIONAL",
            "type": "Flowline Blockage",
            "severity": "High",
            "impact_str": "Safety / Production Risk",
            "context": f"Tubing Pressure +{pct_rise:.0f}% vs Avg, Oil Down.",
            "drivers": "Check choke, flowline, or surface valves.",
            "chart_type": "oil_comparison"
        }
    return None

def check_pump_wear(context: Dict) -> Optional[Dict]:
    # Rod Pump specific
    if (context['current_lift'] == 'Rod Pump') and \
       (context['spm'] > context['spm_avg']) and \
       (context['pump_fillage'] > 80) and \
       (context['actual_oil'] < 0.8 * context['pred_oil']):
         return {
            "category": "EQUIPMENT",
            "type": "Pump Wear / Slippage",
            "severity": "Moderate",
            "impact_str": "Efficiency Loss",
            "context": f"Pump running fast ({context['spm']:.1f} SPM) and full ({context['pump_fillage']:.0f}%), but oil missing.",
            "drivers": "Fluid is slipping past plunger or traveling valve leak.",
            "chart_type": "oil_comparison"
        }
    return None

def check_esp_shaft_break(context: Dict) -> Optional[Dict]:
    # ESP specific: High PIP, Low Amps
    if (context['current_lift'] == 'ESP') and \
       (context['pip'] > context['pip_avg'] * 1.1) and \
       (context['amps'] < context['amps_avg'] * 0.5):
        return {
            "category": "EQUIPMENT",
            "type": "Broken Shaft / Free Spin",
            "severity": "CRITICAL",
            "impact_str": "Total Failure",
            "context": f"High Intake Pressure (Fluid Level High) but Low Amps.",
            "drivers": "Motor spinning without load (Broken Shaft) or Intake Plugged.",
            "chart_type": "production_bar"
        }
    return None

# --- MAIN CONTROLLER ---

def detect_anomalies(well_id: str, df: Optional[pd.DataFrame] = None, lookback_days: int = 7) -> List[Dict]:
    """
    Detect anomalies over the last `lookback_days`.
    Aggregates recurring anomalies into single review items.
    """
    if df is None:
        df = fetch_well_data(well_id)
    if df.empty or len(df) < 14: return []

    df['lift_type'] = df['lift_type'].ffill()
    current_lift = df['lift_type'].iloc[-1] if not df['lift_type'].isnull().all() else 'Rod Pump'
    
    target_features = LIFT_FEATURES.get(current_lift, LIFT_FEATURES['Rod Pump'])
    active_features = [f for f in df.columns if f in target_features]

    if len(active_features) < 3:
        logger.warning(f"Not enough sensor data for {well_id} model.")
        return []

    latest = df.iloc[-1]
    
    # --- 1. DATA QUALITY GATE ---
    invalid_sensors = validate_sensor_data(latest, current_lift)
    anomalies = []
    
    if invalid_sensors:
        anomalies.append({
            "category": "DATA_QUALITY",
            "type": "Sensor Range Violation",
            "severity": "Low",
            "impact_str": "Data Integrity Risk",
            "context": f"Sensors out of DB range: {', '.join(invalid_sensors)}.",
            "drivers": "Sensor Calibration / Failure.",
            "chart_type": "production_bar" 
        })
    
    # --- 2. PREDICTIVE MODEL (Get or Train) ---
    df['predicted_oil'] = np.nan
    top_factor = "Unknown"
    
    # Note: Logic depends on predicted_oil. If Model fails, we can't run Financial/Wear checks.
    # But other checks (Cost Creep, Gas) don't need it.
    
    model_artifact = get_or_train_model(well_id, df, active_features)
    if model_artifact:
        scaler = model_artifact['scaler']
        model = model_artifact['model']
        imputer = model_artifact['imputer']
        top_factor = model_artifact['top_factor']
        
        # Predict on latest (full DF)
        try:
            X_full = scaler.transform(imputer.transform(df[active_features]))
            df['predicted_oil'] = model.predict(X_full)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")

    # --- 3. CONTEXT PREPARATION ---
    # Prepare a clean context dictionary for logical checkers to use
    
    # Rolling Stats
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

    latest = df.iloc[-1]
    
    context = {
        'well_id': well_id,
        'current_lift': current_lift,
        'actual_oil': float(latest.get('oil_volume', 0)),
        'actual_water': float(latest.get('water_volume', 0)),
        'actual_gas': float(latest.get('gas_volume', 0)),
        'pred_oil': float(latest.get('predicted_oil', latest.get('oil_volume', 0))), # Default to actual if no pred
        
        'avg_oil': float(latest.get('oil_avg', 0)),
        'avg_water': float(latest.get('water_avg', 0)),
        'avg_gas': float(latest.get('gas_avg', 0)),
        
        'tubing_pressure': float(latest.get('tubing_pressure', 0)),
        'tp_avg': float(latest.get('tp_avg', 0)),
        
        'pip': float(latest.get('pump_intake_pressure', 0)),
        'pip_avg': float(latest.get('pip_avg', 0)),
        
        'amps': float(latest.get('motor_current', 0)),
        'amps_avg': float(latest.get('amps_avg', 0)),
        
        'spm': float(latest.get('strokes_per_minute', 0)),
        'spm_avg': float(latest.get('spm_avg', 0)),
        
        'pump_fillage': float(latest.get('pump_fillage', 0)),
        'fillage_std': float(latest.get('fillage_std', 0)),
    }
    
    # Activity Check
    is_active = False
    activity_source = "Unknown"
    
    if 'strokes_per_minute' in latest and latest['strokes_per_minute'] > 0.1:
        is_active = True
        activity_source = f"SPM ({latest['strokes_per_minute']:.1f})"
    elif 'motor_current' in latest and latest['motor_current'] > 5:
        is_active = True
        activity_source = f"Amps ({latest['motor_current']:.1f})"
    elif top_factor in latest and latest[top_factor] > 0.1:
        is_active = True
        activity_source = f"{top_factor}"
    
    context['is_active'] = is_active
    context['activity_source'] = activity_source

    # --- 4. EXECUTE CHECKERS ---
    
    # --- 4. EXECUTE CHECKERS OVER LOOKBACK WINDOW ---
    
    checkers = [
        check_financial_gap,
        check_ghost_production,
        check_cost_creep,
        check_gas_interference,
        check_flowline_blockage,
        check_pump_wear,
        check_esp_shaft_break
    ]
    
    # Filter to lookback window
    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=lookback_days)
    window_df = df[df['timestamp'] > cutoff_date].copy()
    
    if window_df.empty:
        return []

    aggregated_anomalies = {}

    # Iterate through each day in the window
    for idx, row in window_df.iterrows():
        # Build context for this specific row (day)
        # Note: We need to re-calculate context for each row because variables like 'actual_oil' change
        
        # Helper to safely get float
        def get_val(r, col, default=0): return float(r.get(col, default))
        
        day_context = {
            'well_id': well_id,
            'current_lift': current_lift,
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
        }

        # Activity Check for this day
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

        # Run Checkers
        for check_func in checkers:
            result = check_func(day_context)
            if result:
                # Aggregate Logic
                anom_type = result['type']
                if anom_type not in aggregated_anomalies:
                    aggregated_anomalies[anom_type] = {
                        "base_anomaly": result,
                        "occurrences": 1,
                        "dates": [row['timestamp']],
                        "max_severity": result['severity'] # Simple string comparison risk, but acceptable for MVP
                    }
                else:
                    agg = aggregated_anomalies[anom_type]
                    agg['occurrences'] += 1
                    agg['dates'].append(row['timestamp'])
                    # Keep latest context for narrative, maybe most severe is better but latest is actionable
                    agg['base_anomaly'] = result 

    # --- 5. FORMAT OUTPUT ---
    results_out = []
    postgres_context = get_postgres_context(well_id)
    
    for anom_type, agg_data in aggregated_anomalies.items():
        base = agg_data['base_anomaly']
        count = agg_data['occurrences']
        dates = sorted(agg_data['dates'])
        first_date = dates[0].strftime('%b %d')
        last_date = dates[-1].strftime('%b %d')
        
        # Modify title to reflect persistence if needed
        title = base['type']
        if count > 1:
            title = f"{title} (Detected {count} times: {first_date}-{last_date})"
        
        narrative = generate_narrative(base, current_lift, postgres_context)
        chart_data = build_chart_data(df, base['chart_type'])
        
        results_out.append({
            "well_id": well_id,
            "timestamp": dates[-1].isoformat(), # Use latest occurrence
            "category": base['category'],
            "severity": base['severity'],
            "title": title,
            "ui_text": {
                "description": narrative['description'],
                "why_is_this_an_anomaly": narrative['why'],
                "suspected_root_cause": narrative['root_cause'],
                "economic_impact": base['impact_str']
            },
            "chart_type": "line" if base['chart_type'] != "production_bar" else "bar",
            "chart_data": chart_data,
            "actions": ["Dismiss", "Acknowledge", "Request Review"]
        })

    # --- 6. PERSISTENCE ---
    save_reviews(results_out)

    return results_out

def generate_narrative(anom: Dict, lift_type: str, history: str) -> Dict:
    client = get_openai_client()
    if not client:
        return {"description": anom['context'], "why": "Deviation detected.", "root_cause": "Manual Review."}
    
    prompt = f"""
    Role: Senior Petroleum Engineer.
    Task: Write a professional, concise Incident Report for a well anomaly.
    Style: Formal, technical, minimal filler words. "Fact-based observation" followed by "Context".
    
    Context:
    - Well Type: {lift_type}
    - Anomaly Type: {anom['type']}
    - Sensor Data: {anom['context']}
    - Physics Driver: {anom['drivers']}
    - Maintenance: {history}
    
    Output JSON:
    1. "description": One sentence summarizing the EVENT. Format: "[Event] detected at [Time] (if avail), [Status]." (e.g., "Pump failure detected at 20:30, no replacement initiated" or "LOE increased 22% over budget").
    2. "why": Explain the ANOMALY logic. Contrast the observation with the expectation. (e.g., "Production has been consistent, yet expenses rose significantly, indicating potential allocation error" or "Chemical injection is critical for corrosion prevention; downtime risks equipment failure").
    3. "root_cause": A specific technical hypothesis based on the driver. (e.g., "Motor overheating due to blocked cooling system" or "Invoice misallocation").
    """
    try:
        res = client.chat.completions.create(
            model="gpt-5-mini", # corrected model name
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"description": anom['context'], "why": "AI Error", "root_cause": "Unknown"}

def build_chart_data(df: pd.DataFrame, chart_type: str) -> Dict:
    plot = df.tail(14).copy()
    if plot.empty: return {"labels": [], "datasets": []}
    
    labels = plot['timestamp'].dt.strftime('%b %d').tolist()
    
    def clean(series): return series.where(pd.notnull(series), None).tolist()

    datasets = []
    if chart_type == "oil_comparison":
        if 'predicted_oil' in plot:
            datasets.append({"label": "Model Prediction", "data": clean(plot['predicted_oil'].round(1)), "borderColor": "#00E396", "tension": 0.4})
        datasets.append({"label": "Actual Revenue (Oil)", "data": clean(plot['oil_volume'].round(1)), "borderColor": "#FF4560", "tension": 0.4})
    
    elif chart_type == "water_trend":
        datasets.append({"label": "30-Day Avg", "data": clean(plot['water_avg'].round(1)), "borderColor": "#775DD0", "borderDash": [5,5]})
        datasets.append({"label": "Actual Water", "data": clean(plot['water_volume'].round(1)), "borderColor": "#FF4560"})

    elif chart_type == "gas_trend":
        datasets.append({"label": "Gas Volume", "data": clean(plot['gas_volume'].round(1)), "borderColor": "#FEB019"})

    elif chart_type == "production_bar":
        total = (plot['oil_volume'] + plot['water_volume'] + plot['gas_volume']).round(1)
        datasets.append({"label": "Total Fluids", "data": clean(total), "backgroundColor": "#FF4560", "type": "bar"})

    return {"labels": labels, "datasets": datasets}

def save_reviews(reviews: List[Dict]):
    """Persist generated reviews to PostgreSQL (and Snowflake backup)."""
    if not reviews: return

    # 1. PostgreSQL
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    # Ensure table exists (simple check)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS anomaly_reviews (
                            id SERIAL PRIMARY KEY,
                            well_id VARCHAR(50) NOT NULL,
                            timestamp TIMESTAMP NOT NULL,
                            category VARCHAR(50),
                            severity VARCHAR(20),
                            title VARCHAR(255),
                            description TEXT,
                            why_anomaly TEXT,
                            root_cause TEXT,
                            economic_impact VARCHAR(100),
                            status VARCHAR(20) DEFAULT 'New',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    for r in reviews:
                        # Check duplicates (by title + timestamp) to prevent spam on re-runs
                        cur.execute("""
                            SELECT id FROM anomaly_reviews 
                            WHERE well_id = %s AND timestamp = %s AND title = %s
                        """, (r['well_id'], r['timestamp'], r['title']))
                        
                        if not cur.fetchone():
                            cur.execute("""
                                INSERT INTO anomaly_reviews 
                                (well_id, timestamp, category, severity, title, description, why_anomaly, root_cause, economic_impact)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                r['well_id'], 
                                r['timestamp'], 
                                r['category'], 
                                r['severity'], 
                                r['title'], 
                                r['ui_text']['description'], 
                                r['ui_text']['why_is_this_an_anomaly'], 
                                r['ui_text']['suspected_root_cause'], 
                                r['ui_text']['economic_impact']
                            ))
                conn.commit()
                # logger.info(f"Saved {len(reviews)} reviews to Postgres.")
        except Exception as e:
            logger.error(f"Postgres save failed: {e}")

    # 2. Snowflake (Optional / Async)
    # Implementing synchronous for now for simplicity
    try:
        with get_snowflake_conn() as conn:
             # Ensure table exists
            conn.cursor().execute("""
                CREATE TABLE IF NOT EXISTS anomaly_reviews (
                    well_id VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP_NTZ NOT NULL,
                    category VARCHAR(50),
                    severity VARCHAR(20),
                    title VARCHAR(255),
                    description TEXT,
                    why_anomaly TEXT,
                    root_cause TEXT,
                    economic_impact VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'New',
                    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
                )
            """)
            
            for r in reviews:
                 # Check duplicates
                chk = conn.cursor().execute("""
                    SELECT count(*) FROM anomaly_reviews 
                    WHERE well_id = %s AND timestamp = %s AND title = %s
                """, (r['well_id'], r['timestamp'], r['title'])).fetchone()
                
                if chk and chk[0] == 0:
                    conn.cursor().execute("""
                        INSERT INTO anomaly_reviews 
                        (well_id, timestamp, category, severity, title, description, why_anomaly, root_cause, economic_impact)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        r['well_id'], 
                        r['timestamp'], 
                        r['category'], 
                        r['severity'], 
                        r['title'], 
                        r['ui_text']['description'], 
                        r['ui_text']['why_is_this_an_anomaly'], 
                        r['ui_text']['suspected_root_cause'], 
                        r['ui_text']['economic_impact']
                    ))
            # logger.info(f"Saved {len(reviews)} reviews to Snowflake.")
    except Exception as e:
        logger.error(f"Snowflake save failed: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Check for optional days argument
        days = 7
        if len(sys.argv) > 2:
            try:
                days = int(sys.argv[2])
            except:
                pass
        print(json.dumps(detect_anomalies(sys.argv[1], lookback_days=days), indent=2))
    else:
        print("Usage: python anomaly_review_service.py <well_id>")