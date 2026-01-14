from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging
import os
from uuid import uuid4
import random
import openai

# Custom JSON Encoder to handle non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        else:
            return super().default(obj)

# Helper function to convert non-JSON-serializable types
def convert_to_serializable(obj):
    """Convert non-JSON-serializable objects (like bool, numpy types) to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, bool):
        return obj  # Python bools are JSON serializable (True/False -> true/false)
    elif isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()  # Convert numpy types to Python native types
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj
try:
    from snowflake.connector import connect as snowflake_connect
except Exception:
    snowflake_connect = None
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Import database operations from operation.py
from operation import (
    create_operation_suggestion_table,
    save_operation_suggestion,
    get_operation_suggestions,
    update_operation_suggestion_status,
    get_operation_suggestion_detail,
    get_pg_connection
)

# Try to import operational recommendations, handle if missing
try:
    from operational_recommendations import generate_recommendations_for_well
except ImportError:
    generate_recommendations_for_well = None

# Import unified anomaly detector (replaces hardcoded RULES)
try:
    from anomaly_detector import get_detector, check_anomaly as detector_check_anomaly
    ANOMALY_DETECTOR_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTOR_AVAILABLE = False
    logging.warning("anomaly_detector module not available; falling back to hardcoded rules")

# Import dynamic configuration
try:
    from dynamic_config import get_lift_types, get_form_sensors, get_sensor_ranges, get_form_fields
    DYNAMIC_CONFIG_AVAILABLE = True
except ImportError:
    DYNAMIC_CONFIG_AVAILABLE = False
    logging.warning("dynamic_config module not available")

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
APP_URL = os.getenv("APP_URL", "http://localhost:5000")

# OpenAI Config
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY is missing in .env")
else:
    logging.info(f"OPENAI_API_KEY loaded: {api_key[:8]}...{api_key[-4:]}")

client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_API_BASE"),
    default_headers={
        "HTTP-Referer": APP_URL,
        "X-Title": "Well Anomaly Detection",
    }
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Snowflake Config
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

REQUIRED_SNOWFLAKE_KEYS = ['user', 'password', 'account', 'warehouse', 'database', 'schema', 'role']
missing_sf = [k for k in REQUIRED_SNOWFLAKE_KEYS if not SNOWFLAKE_CONFIG.get(k)]
if missing_sf:
    logging.warning(f"Missing Snowflake configuration keys: {missing_sf}")

# Initialize anomaly detector (loads rules from Snowflake dynamic config)
if ANOMALY_DETECTOR_AVAILABLE:
    anomaly_detector = get_detector()
    logging.info("✓ Unified anomaly detector initialized with dynamic Snowflake config")
else:
    raise RuntimeError("Anomaly detector unavailable; dynamic Snowflake config required")

# --- Database Connections ---

def get_db_connection():
    missing = [k for k in REQUIRED_SNOWFLAKE_KEYS if not SNOWFLAKE_CONFIG.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake configuration: {missing}")
    try:
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to Snowflake: {e}")
        raise

def get_historical_anomalies(limit=100, filter_clauses=None, filter_params=None):
    """
    Fetch global anomalies from Snowflake with optional filtering.
    
    Args:
        limit (int): Max rows to fetch.
        filter_clauses (list): List of SQL WHERE fragments (e.g., "TRY_PARSE_JSON(raw_values):X > Y").
        filter_params (list): List of parameters for the SQL fragments. 
    """
    try:
        conn = get_db_connection()
        
        # Base query
        query = """
        SELECT 
            well_id, 
            timestamp, 
            category, 
            severity, 
            raw_values, 
            anomaly_score, 
            anomaly_type as violation_summary 
        FROM well_anomalies
        """
        
        # Add dynamic filters
        params = []
        if filter_clauses:
            query += " WHERE " + " OR ".join(filter_clauses)
            if filter_params:
                params.extend(filter_params)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        df = pd.read_sql(query, conn, params=params)
        df.columns = [c.lower() for c in df.columns]
        conn.close()
        return df
    except Exception as e:
        logging.warning(f"Could not fetch historical anomalies: {e}")
        return pd.DataFrame()

def find_similar_anomalies(current_readings, current_violations, historical_df, lift_type=None, similarity_threshold=0.8):
    """
    Find historical anomalies similar to the current one based on sensor values.
    
    Args:
        current_readings (dict): Current sensor readings.
        current_violations (list): List of dicts with 'field' and 'value' for current violations.
        historical_df (pd.DataFrame): DataFrame of historical anomalies.
        lift_type (str): Optional lift type (currently not used for filtering df, assuming caller handles context).
        similarity_threshold (float): Minimum similarity (0-1) to consider a match.
        
    Returns:
        list: Top 3 similar historical records.
    """
    if historical_df.empty or not current_violations:
        return []

    similar_records = []
    
    # Identify keys that are actually violated right now
    violated_keys = [v['field'] for v in current_violations if 'field' in v]
    
    for _, row in historical_df.iterrows():
        try:
            hist_values = row['raw_values']
            if isinstance(hist_values, str):
                hist_values = json.loads(hist_values)
            elif not isinstance(hist_values, dict):
                continue
                
            # IMPLICIT LIFT TYPE FILTERING:
            # If the historical record has NONE of the keys in the current violation,
            # it's likely from a different lift type (e.g. Rod Pump vs ESP).
            # We skip such records to avoid comparing apples to oranges.
            # Skip self-match logic moved to insert_reading() where we filter by timestamp.
            # We must NOT filter by values here, as identical values at DIFFERENT times are valid history.
            
            common_keys = set(hist_values.keys()) & set(current_readings.keys())
            if not common_keys:
                 continue 

            # Calculate match score
            matches = 0
            total_score = 0
            
            for key in violated_keys:
                if key in hist_values and hist_values[key] is not None and current_readings.get(key) is not None:
                    curr_val = float(current_readings[key])
                    # Handle historical value which might be a string in JSON
                    hist_val = float(hist_values[key])
                    
                    # Avoid division by zero
                    max_val = max(abs(curr_val), abs(hist_val))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        similarity = 1.0 - (abs(curr_val - hist_val) / max_val)
                    
                    if similarity >= similarity_threshold:
                        matches += 1
                        total_score += similarity
            
            if matches > 0:
                # Average similarity across matched items
                avg_similarity = total_score / matches
                
                # Calculate deviation and score
            if matches > 0:
                # Average similarity across matched items
                avg_similarity = total_score / matches
                
                # Create a record structure compatible with prompt generation
                record = {
                    "well_id": row.get('well_id', 'Unknown'),
                    "timestamp": row['timestamp'],
                    "alert_title": row['violation_summary'],
                    "severity": row['severity'],
                    "similarity_score": avg_similarity,
                    "match_count": matches,
                    "raw_anomaly_data": {
                        "violations": [{"field": k, "value": hist_values.get(k), "violation": str(hist_values.get(k))} for k in violated_keys],
                        "full_readings": hist_values  # Include full readings for context
                    }
                }
                similar_records.append(record)
                
        except Exception as e:
            # logging.debug(f"Skipping history row due to parse error: {e}")
            continue
            
    # Sort by match count (desc), then similarity score (desc)
    similar_records.sort(key=lambda x: (x['match_count'], x['similarity_score']), reverse=True)
    
    return similar_records[:3]

def init_postgres():
    """Initialize PostgreSQL with operation_suggestion table."""
    success = create_operation_suggestion_table()
    if success:
        logging.info("✓ PostgreSQL initialization complete.")
    else:
        logging.error("✗ Error initializing PostgreSQL")
    
    conn = get_pg_connection()
    if not conn:
        logging.warning("✗ Could not connect to PostgreSQL for anomaly_suggestions table check.")
        return

    try:
        cur = conn.cursor()
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'anomaly_suggestions');")
        table_exists = cur.fetchone()[0]

        if not table_exists:
            cur.execute("""
                CREATE TABLE anomaly_suggestions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    well_id VARCHAR(50) NOT NULL,
                    asset_id VARCHAR(50),
                    lift_type VARCHAR(50),
                    timestamp TIMESTAMP NOT NULL,
                    alert_title VARCHAR(255),
                    severity VARCHAR(50),
                    status VARCHAR(50),
                    confidence VARCHAR(20),
                    description TEXT,
                    suggested_actions JSONB,
                    explanation TEXT,
                    historical_context JSONB, 
                    risk_analysis JSONB,
                    raw_anomaly_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logging.info("✓ PostgreSQL table 'anomaly_suggestions' created.")
        else:
            logging.info("✓ PostgreSQL table 'anomaly_suggestions' already exists.")
            try:
                cur.execute("ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS asset_id VARCHAR(50);")
                cur.execute("ALTER TABLE anomaly_suggestions ADD COLUMN IF NOT EXISTS lift_type VARCHAR(50);")
            except Exception:
                pass

        conn.commit()
        logging.info("✓ PostgreSQL anomaly tables verified.")
    except Exception as e:
        logging.error(f"✗ Error initializing PostgreSQL: {e}")
        if conn: conn.rollback()
    finally:
        if conn:
             cur.close()
             conn.close()

init_postgres()

# --- Anomaly Detection with Dynamic Snowflake Config ---
def check_anomaly(readings, well_id=None, lift_type=None):
    """
    Check for anomalies using unified detector with Snowflake dynamic config.
    
    Uses dynamic configuration from Snowflake for all rules and sensor definitions.
    """
    # Use unified detector with well-type awareness (required)
    return anomaly_detector.check_anomaly(readings, well_id, lift_type)

def calculate_severity(anomaly_details, anomaly_score, readings):
    """
    Calculate severity based on violation count, anomaly score, and deviation percentage.
    """
    violations = anomaly_details.get('violations', [])
    num_violations = len(violations)
    
    # Calculate average deviation percentage
    deviations = []
    for v in violations:
        field = v.get('field')
        value = v.get('value')
        min_val = v.get('min')
        max_val = v.get('max')
        
        if value is not None and min_val is not None and max_val is not None:
            if value < min_val:
                deviation = abs(value - min_val) / abs(max_val - min_val) * 100
            else:
                deviation = abs(value - max_val) / abs(max_val - min_val) * 100
            deviations.append(deviation)
    
    avg_deviation = sum(deviations) / len(deviations) if deviations else 0
    
    # Apply severity matrix with increased frequency
    if num_violations >= 3 or anomaly_score >= 0.7 or avg_deviation > 50:
        return "CRITICAL"
    elif num_violations == 2 or anomaly_score >= 0.5 or (avg_deviation > 40 and avg_deviation <= 40):
        return "HIGH"
    elif num_violations == 1 or anomaly_score >= 0.15 or (avg_deviation > 20 and avg_deviation <= 30):
        return "MEDIUM"
    else:
        return "LOW"

# --- AI Logic ---
def generate_suggestion(anomaly_details, well_id, readings, timestamp_str, historical_anomalies=None, lift_type=None):
    try:
        # --- FIX: Always prepare a list, extracting ALL violations as an array ---
        formatted_history = []
        
        if historical_anomalies:
            for anom in historical_anomalies:
                raw_data = anom.get('raw_anomaly_data')
                
                if isinstance(raw_data, str):
                    try:
                        raw_data = json.loads(raw_data)
                    except:
                        raw_data = {}
                
                # Extract ALL violations for this incident
                past_violations_list = []
                if raw_data and 'violations' in raw_data and isinstance(raw_data['violations'], list):
                    for v in raw_data['violations']:
                        past_violations_list.append({
                            "field": v.get('field'),
                            "violation": v.get('violation')  # Grab the specific text string directly
                        })

                # Create the formatted object with the violations array
                formatted_history.append({
                    "well": anom.get('well_id'),          
                    "date": str(anom.get('timestamp')),   
                    "issue": anom.get('alert_title'),
                    "severity": anom.get('severity'),
                    "violations": past_violations_list,  # Sending the array of {field, violation}
                    "readings": raw_data.get('full_readings', {}), # Include full readings
                })
        
        # Always dump as JSON string
        history_context_str = json.dumps(formatted_history, indent=2, cls=CustomJSONEncoder, default=str)
    
        # calculated_severity is already computed in insert_reading or needs to be computed here if this function called standalone
        # We can re-calculate just to be safe as it's cheap
        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
 
        # Instruct the model to use the units provided in each violation and not to convert units.
        # Also include authoritative sensor ranges fetched from dynamic config for the well's lift type
        sensor_ranges_str = "{}"
        try:
            if lift_type and DYNAMIC_CONFIG_AVAILABLE:
                sensor_ranges = get_sensor_ranges(lift_type)
                sensor_ranges_str = json.dumps(sensor_ranges, indent=2, cls=CustomJSONEncoder, default=str)
        except Exception:
            sensor_ranges_str = "{}"
        prompt = f"""
        You are an expert Production Engineer AI assistant. Analyze the new anomaly.
        
        ---
        New Anomaly Details:
        Well ID: {well_id}
        Timestamp: {timestamp_str}
        Current Sensor Readings:
        {json.dumps(readings, indent=2, cls=CustomJSONEncoder, default=str)}
        Violations Detected:
        {json.dumps(anomaly_details['violations'], indent=2, cls=CustomJSONEncoder, default=str)}
        
        Authoritative Sensor Ranges for lift type ({lift_type}):
        {sensor_ranges_str}
        ---
        Real Database History for this well (for context):
        {history_context_str}
        ---
 
        IMPORTANT: Use the units provided in each violation entry exactly as given. Do NOT convert units (e.g., do not change °F to °C) or invent ranges. Use the 'unit', 'min', and 'max' fields from the Violations Detected objects when describing values and expected ranges. If a violation list is empty, fall back to the authoritative sensor ranges provided in the 'Authoritative Sensor Ranges' section above.

        Based on the information above, provide a structured JSON response.

        SEVERITY CLASSIFICATION GUIDE:
        - CRITICAL: >50% deviation, 3+ violations, or score >= 0.7
        - HIGH: 30-50% deviation, 2 violations, or score >= 0.5
        - MEDIUM: 20-30% deviation, 1 violation, or score >= 0.3
        - LOW: <10% deviation, 0 violations, or score < 0.20
        
        CONFIDENCE CALCULATION:
        Calculate confidence as a SINGLE percentage value based on the anomaly score.
        Do NOT return a range. Pick a specific number within the bucket.
        - If anomaly_score >= 0.8: Return a value between 90% and 99% 
        - If anomaly_score >= 0.6: Return a value between 70% and 85% 
        - If anomaly_score >= 0.4: Return a value between 50% and 70% 
        - If anomaly_score >= 0.2: Return a value between 30% and 50% 
        - If anomaly_score < 0.2: Return a value between 10% and 30% 
        
        PRE-CLASSIFIED SEVERITY: {calculated_severity}
        ANOMALY SCORE: {anomaly_details.get('anomaly_score', 0):.2f}
        Strictly assign severity level that matches or stays close to this pre-classification.

        Important:
        1. If 'Real Database History' above is not an empty list, populate 'similar_incidents' using that data, ensuring you transform the 'readings' into the 'readings_summary' format requested below.
        
        Required JSON Structure:
        1. "alert_title": Short title (e.g., "ESP Motor Current Spike").
        2. "severity": Assign based on severity matrix above (CRITICAL, HIGH, MEDIUM, or LOW).
        3. "status": "ACTIVE".
        4. "confidence": IMPORTANT - Single Percentage string (e.g., "78%") calculated from anomaly score. MUST be a single number. DO NOT return a range.
        5. "description": Write a concise, technical paragraph. Within this paragraph, you MUST clearly describe each parameter from 'Violations Detected' that is out of range.
            For each parameter, state the nature of the anomaly by including its observed value and the expected range, both with their units.
            For example: "The motor temperature reached 265 [unit], exceeding the expected operational range of 100–250 [unit]." Use the 'unit' field from each violation object.
        6. "suggested_actions": A list of at least three clear, actionable steps.
        7. "explanation": Detailed explanation of the new anomaly.
        8. "historical_context": Object containing:
            - "asset_history": Object with keys "commissioned" (date), "operating_hours" (int), "last_inspection" (date). Since this data is not provided, you MUST use the string "Data Not Available" for each value.
            - "similar_incidents": A list of objects based ONLY on the 'Real Database History' provided above. If the history is empty, this MUST be an empty list []. For each incident in the history, create an object with the following keys:
                - "well": Use the "well" value from the history.
                - "date": Use the "date" value from the history, use format "MM/DD/YYYY".
                - "issue": Summarize the technical issue briefly (e.g., "Motor Current High" or "Pressure Violation"). Do NOT copy long "Out of range" error strings.
                - "severity": Use the "severity" value from the history.
                 - "violations": Use the "violations" array from history (containing "field" and "violation" ,violation must be in simple value example 300 A).
            - "key_learnings": String field (NOT inside similar_incidents). Analyze the 'Real Database History' provided. If similar incidents exist, identify patterns in frequency, severity, or affected parameters (e.g., "Three pressure-related incidents occurred within 24 hours, suggesting potential equipment degradation")".
       
        9. "risk_analysis": Object containing:
            - "severity_breakdown": Object with keys "safety_risk", "production_impact", "financial_exposure".
            - "mitigation_strategies": Object with keys "immediate", "short_term", "long_term".
            - "decision_support": A final string recommendation.
            
        Ensure your entire output is a single, valid JSON object.
        """
 
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for oil and gas anomaly detection. Output valid JSON only, matching the required schema exactly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
 
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        suggestion_data = json.loads(content.strip())
        return suggestion_data
 
    except Exception as e:
        logging.error(f"Error generating suggestion: {e}")
        return {
            "alert_title": "Anomaly Detected",
            "severity": "UNKNOWN",
            "status": "ACTIVE",
            "confidence": "0%",
            "description": "Could not generate detailed analysis.",
            "suggested_actions": ["Investigate manually"],
            "explanation": f"AI Generation failed: {str(e)}",
            "historical_context": {},
            "risk_analysis": {}
        }

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/form-config', methods=['GET'])
def get_form_config():
    """Get dynamic form configuration for all lift types."""
    try:
        if not DYNAMIC_CONFIG_AVAILABLE:
            return jsonify({"error": "Dynamic config not available"}), 503
        
        lift_types = get_lift_types()
        config = {}
        
        for lift_type_name in lift_types.keys():
            sensors = get_form_sensors(lift_type_name)
            ranges = get_sensor_ranges(lift_type_name)
            
            config[lift_type_name] = {
                'sensors': sensors,
                'ranges': ranges
            }
        
        return jsonify(config), 200
    except Exception as e:
        logging.error(f"Error fetching form config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/insert-reading', methods=['POST'])
def insert_reading():
    try:
        data = request.get_json()
        required_fields = ['well_id', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        well_id = data.get('well_id')
        timestamp_str = data.get('timestamp')
        lift_type = data.get('lift_type')  # For well-type aware detection and storage
        
        readings = {}
        # Extract fields based on well type
        if lift_type == 'Rod Pump':
            field_list = ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure']
        elif lift_type == 'ESP':
            field_list = ['motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage']
        elif lift_type == 'Gas Lift':
            field_list = ['injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time']
        else:
            field_list = ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure']  # Default to Rod Pump
        
        for field in field_list:
            value = data.get(field)
            readings[field] = float(value) if value is not None else None

        # Use unified detector with optional lift_type awareness
        anomaly_details = check_anomaly(readings, well_id=well_id, lift_type=lift_type)

        # Calculate severity once to pass to prompt AND to database
        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
        
        # Safety post-check: if there are no rule violations and the anomaly_score
        # is below the ML-only threshold, treat as non-anomalous to avoid false positives.
        try:
            from anomaly_detector import ML_ONLY_ANOMALY_THRESHOLD
            if (not anomaly_details.get('violations')) and float(anomaly_details.get('anomaly_score', 0)) < ML_ONLY_ANOMALY_THRESHOLD:
                anomaly_details['is_anomaly'] = False
        except Exception:
            pass

        # 1. Insert Raw Reading into Snowflake
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Convert numpy types to native Python types for database insertion
            def convert_for_db(val):
                if isinstance(val, np.integer):
                    return int(val)
                elif isinstance(val, np.floating):
                    return float(val)
                elif isinstance(val, np.bool_):
                    return bool(val)
                elif isinstance(val, np.ndarray):
                    return val.tolist()
                return val
            
            # Build dynamic INSERT with well-type specific fields
            field_names = ['well_id', 'timestamp', 'lift_type'] + list(readings.keys())
            field_values = [well_id, timestamp_str, lift_type] + [convert_for_db(v) for v in readings.values()]
            
            placeholders = ', '.join(['%s'] * len(field_values))
            fields_str = ', '.join(field_names)
            
            query = f"INSERT INTO well_sensor_readings ({fields_str}) VALUES ({placeholders})"
            
            try:
                cursor.execute(query, field_values)
            except Exception as e:
                # Fallback: insert without lift_type if column doesn't exist yet
                if "lift_type" in str(e).lower() or "invalid identifier" in str(e).lower():
                    logging.warning(f"⚠ lift_type column not found. Run: python migrate_add_lift_type.py")
                    field_names = ['well_id', 'timestamp'] + list(readings.keys())
                    field_values = [well_id, timestamp_str] + [convert_for_db(v) for v in readings.values()]
                    placeholders = ', '.join(['%s'] * len(field_values))
                    fields_str = ', '.join(field_names)
                    query = f"INSERT INTO well_sensor_readings ({fields_str}) VALUES ({placeholders})"
                    cursor.execute(query, field_values)
                else:
                    raise
            conn.commit()
        finally:
            cursor.close()
        
        # 1B. Insert into lift-type specific table
        
        suggestion_result = None

        if anomaly_details['is_anomaly']:
            # 2. Insert Anomaly Record into Snowflake (Legacy/Backup)
            violation_summary = "; ".join([v['violation'] for v in anomaly_details['violations']])
            raw_values = json.dumps(readings, cls=CustomJSONEncoder, default=str)
            cursor = conn.cursor()
            try:
                anomaly_params = (
                    well_id,
                    timestamp_str,
                    violation_summary,
                    float(anomaly_details['anomaly_score']),  # Convert numpy float to Python float
                    raw_values,
                    'Rule_Based_Frontend',
                    'New',
                    calculated_severity,
                    lift_type # Use lift_type as category
                )
                cursor.execute(
                    """
                    INSERT INTO well_anomalies
                    (well_id, timestamp, anomaly_type, anomaly_score, raw_values, model_name, status, severity, category)
                    SELECT %s, %s, %s, %s, TRY_PARSE_JSON(%s), %s, %s, %s, %s
                    """,
                    anomaly_params
                )
                conn.commit()
            finally:
                cursor.close()

            ### START: SIMILARITY SEARCH INTEGRATION ###
            
            # 3. Intelligent Context Retrieval (Similarity Search Method) - PYTHON IMPLEMENTATION
            recent_anomalies_for_prompt = []
            try:
                # OPTIMIZATION: Construct dynamic SQL filters to push 80% similarity check to Snowflake
                # This prevents fetching 2000+ rows and only fetches highly relevant candidates.
                filter_clauses = []
                filter_params = []
                
                for v in anomaly_details.get('violations', []):
                    field = v.get('field')
                    val = readings.get(field)
                    # Ensure we have a valid numeric value to build a range
                    if field and val is not None and isinstance(val, (int, float)):
                        # 80% similarity window = +/- 20%
                        # Range: [val * 0.8, val * 1.2]
                        lower = float(val) * 0.8
                        upper = float(val) * 1.2
                        
                        # Snowflake SQL: Check if the JSON field is within range
                        # Note: We use TRY_PARSE_JSON just in case, though raw_values is likely variant/json
                        filter_clauses.append(f"(CAST(TRY_PARSE_JSON(raw_values):{field} AS FLOAT) BETWEEN %s AND %s)")
                        filter_params.extend([lower, upper])

                # Fetch global historical anomalies with filters (limit 100 is now sufficient due to filtering)
                
                # Fetch global historical anomalies with filters (limit 100 is now sufficient due to filtering)
                hist_anomalies_df = get_historical_anomalies(
                    limit=100, 
                    filter_clauses=filter_clauses, 
                    filter_params=filter_params
                )

                # --- FIX: Filter out the CURRENT anomaly we just inserted ---
                # The 'hist_anomalies_df' will contain the record from step 2 above.
                # We simply drop rows that match this well_id AND this specific timestamp.
                if not hist_anomalies_df.empty:
                    # Ensure timestamp column is datetime
                    hist_anomalies_df['timestamp'] = pd.to_datetime(hist_anomalies_df['timestamp'])
                    current_ts = pd.to_datetime(timestamp_str)
                    
                    # Filter: Keep rows where (well_id != current_well) OR (timestamp != current_ts)
                    # Note: Timestamp from DB might have slight microsecond diffs vs string, so we allow tiny buffer or exact string match if possible.
                    # Best approach: exclude if well_id matches AND abs(time_diff) < 1 second
                    
                    mask = (hist_anomalies_df['well_id'] == well_id) & \
                           ((hist_anomalies_df['timestamp'] - current_ts).abs() < pd.Timedelta(seconds=1))
                    
                    hist_anomalies_df = hist_anomalies_df[~mask]

                
                # Perform final precise ranking in-memory
                recent_anomalies_for_prompt = find_similar_anomalies(
                    current_readings=readings,
                    current_violations=anomaly_details.get('violations', []),
                    historical_df=hist_anomalies_df,
                    lift_type=lift_type,
                    similarity_threshold=0.8
                )
                
                if recent_anomalies_for_prompt:
                    logging.info(f"✓ Found {len(recent_anomalies_for_prompt)} similar historical anomalies (optimized cross-well) using Snowflake push-down.")
                else:
                    logging.info(f"No similar historical anomalies found for well {well_id} using Python similarity search.")
                    
            except Exception as e:
                logging.error(f"Could not fetch similarity-based historical anomalies: {e}")
            
            ### END: SIMILARITY SEARCH INTEGRATION ###

            # If rule-based violations are empty but anomaly score is high,
            # augment anomaly_details with synthetic violations using dynamic sensor ranges
            augmented_anomaly_details = anomaly_details.copy()
            try:
                if (not augmented_anomaly_details.get('violations')) and lift_type and DYNAMIC_CONFIG_AVAILABLE:
                    sensor_ranges = get_sensor_ranges(lift_type)
                    synth_violations = []
                    for field, val in readings.items():
                        if val is None:
                            continue
                        if field in sensor_ranges:
                            sr = sensor_ranges[field]
                            mn = sr.get('min')
                            mx = sr.get('max')
                            unit = sr.get('unit') or ""
                            if mn is not None and mx is not None and (val < mn or val > mx):
                                # compute deviation pct like detector
                                rng = mx - mn if (mx - mn) != 0 else 1
                                if val < mn:
                                    deviation = ((mn - val) / rng) * 100
                                else:
                                    deviation = ((val - mx) / rng) * 100
                                violation_text = f"Out of range. Expected {mn}-{mx} {unit}, got {val} {unit} ({deviation:.1f}% deviation)"
                                synth_violations.append({
                                    'field': field,
                                    'value': val,
                                    'min': mn,
                                    'max': mx,
                                    'unit': unit,
                                    'deviation_pct': round(deviation, 2),
                                    'violation': violation_text
                                })

                    if synth_violations:
                        augmented_anomaly_details['violations'] = synth_violations
                        augmented_anomaly_details['summary'] = f"{len(synth_violations)} synthetic violation(s) augmented from dynamic config"
            except Exception:
                # If augmentation fails, fall back to original anomaly_details
                augmented_anomaly_details = anomaly_details

            # If we augmented violations, make augmented_anomaly_details the canonical anomaly_details
            if augmented_anomaly_details is not anomaly_details:
                anomaly_details = augmented_anomaly_details

            suggestion_result = generate_suggestion(
                anomaly_details,
                well_id,
                readings,
                timestamp_str,
                historical_anomalies=recent_anomalies_for_prompt,
                lift_type=lift_type
            )

            asset_id = f"ALT-{random.randint(100000, 999999)}"
            if suggestion_result is None:
                suggestion_result = {}
            suggestion_result['asset_id'] = asset_id

            # 5. Store Suggestion in Postgres
            pg_conn_insert = get_pg_connection()
            if pg_conn_insert:
                try:
                    with pg_conn_insert.cursor() as cur:
                        insert_query = """
                            INSERT INTO anomaly_suggestions 
                            (id, well_id, asset_id, lift_type, timestamp, alert_title, severity, status, confidence, description, suggested_actions, explanation, historical_context, risk_analysis, raw_anomaly_data)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        insert_values = (
                            str(uuid4()),
                            well_id,
                            suggestion_result.get('asset_id'),
                            lift_type,
                            timestamp_str,
                            suggestion_result.get('alert_title'),
                            suggestion_result.get('severity'),
                            suggestion_result.get('status'),
                            suggestion_result.get('confidence'),
                            suggestion_result.get('description'),
                            json.dumps(suggestion_result.get('suggested_actions'), cls=CustomJSONEncoder, default=str),
                            suggestion_result.get('explanation'),
                            json.dumps(suggestion_result.get('historical_context', {}), cls=CustomJSONEncoder, default=str),
                            json.dumps(suggestion_result.get('risk_analysis', {}), cls=CustomJSONEncoder, default=str),
                            json.dumps(anomaly_details, cls=CustomJSONEncoder, default=str)
                        )
                        
                        cur.execute(insert_query, insert_values)
                    pg_conn_insert.commit()
                    logging.info(f"✓ Saved suggestion for {well_id} with lift_type={lift_type} to Postgres")
                except Exception as pg_e:
                    logging.error(f"✗ Error storing suggestion in Postgres: {pg_e}")
                    if pg_conn_insert:
                        pg_conn_insert.rollback()
                finally:
                    if pg_conn_insert:
                        pg_conn_insert.close()
            else:
                logging.warning(f"✗ PostgreSQL connection failed for well {well_id} - suggestion NOT stored")

        conn.close()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        return jsonify({
            "success": True,
            "message": f"Reading inserted successfully for well {well_id}",
            "anomaly_detected": convert_types(anomaly_details['is_anomaly']),
            "anomaly_score": convert_types(anomaly_details['anomaly_score']),
            "summary": anomaly_details['summary'],
            "violations": convert_types(anomaly_details['violations']),
            "suggestion": convert_types(suggestion_result)
        }), 200
    except Exception as e:
        logging.error(f"Error inserting reading: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/well-history/<well_id>', methods=['GET'])
def get_well_history(well_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get readings from main table (now contains all fields)
        try:
            cursor.execute(
                """
                SELECT * FROM well_sensor_readings
                WHERE well_id = %s
                ORDER BY timestamp DESC
                LIMIT 20
                """,
                (well_id,)
            )
            rows = cursor.fetchall()
            if rows and cursor.description:
                cols = [c[0].lower() for c in cursor.description]
                readings_df = pd.DataFrame(rows, columns=cols)
            else:
                readings_df = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching readings: {e}")
            readings_df = pd.DataFrame()
        finally:
            cursor.close()

        # Get anomaly suggestions from PostgreSQL
        suggestions_list = []
        pg_conn = get_pg_connection()
        if pg_conn:
            try:
                with pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM anomaly_suggestions 
                        WHERE well_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 20
                    """, (well_id,))
                    suggestions_list = cur.fetchall()
                    for item in suggestions_list:
                        item['timestamp'] = str(item.get('timestamp'))
                        item['created_at'] = str(item.get('created_at'))
            except Exception as e:
                logging.error(f"Error fetching Postgres history: {e}")
            finally:
                if pg_conn:
                    pg_conn.close()

        # Fallback to legacy anomalies if no suggestions
        if not suggestions_list:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT * FROM well_anomalies
                    WHERE well_id = %s
                    ORDER BY timestamp DESC
                    LIMIT 20
                    """,
                    (well_id,)
                )
                rows = cursor.fetchall()
                if rows and cursor.description:
                    cols = [c[0].lower() for c in cursor.description]
                    anomalies_df = pd.DataFrame(rows, columns=cols)
                else:
                    anomalies_df = pd.DataFrame()
            except Exception as e:
                logging.error(f"Error fetching legacy anomalies: {e}")
                anomalies_df = pd.DataFrame()
            finally:
                cursor.close()

            for _, row in anomalies_df.iterrows():
                record = row.to_dict()
                # Replace NaN and inf values with None for JSON serialization
                record = {k: (None if pd.isna(v) or (isinstance(v, float) and np.isinf(v)) else v) for k, v in record.items()}
                if 'timestamp' in record:
                    record['timestamp'] = str(record['timestamp'])
                suggestions_list.append({
                    "alert_title": "Anomaly Detected (Legacy)",
                    "severity": "UNKNOWN",
                    "status": record.get("status", "New"),
                    "confidence": "N/A",
                    "description": record.get("anomaly_type", "Unknown issue"),
                    "timestamp": record.get("timestamp"),
                    "raw_anomaly_data": record
                })

        conn.close()
        
        # Convert readings to list, handling NaN/inf values
        readings_list = []
        for _, row in readings_df.iterrows():
            record = row.to_dict()
            # Replace NaN and inf values with None for JSON serialization
            record = {k: (None if pd.isna(v) or (isinstance(v, float) and np.isinf(v)) else v) for k, v in record.items()}
            if 'timestamp' in record:
                record['timestamp'] = str(record['timestamp'])
            readings_list.append(record)
        
        return jsonify({
            "well_id": well_id,
            "readings": readings_list,
            "anomalies": suggestions_list 
        }), 200
    except Exception as e:
        logging.error(f"Error fetching history: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/api/wells', methods=['GET'])
def get_wells():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT DISTINCT well_id FROM well_sensor_readings
                ORDER BY well_id
                """
            )
            rows = cursor.fetchall()
            if rows and cursor.description:
                cols = [c[0].lower() for c in cursor.description]
                wells_df = pd.DataFrame(rows, columns=cols)
                wells = wells_df['well_id'].tolist() if 'well_id' in wells_df.columns else [r[0] for r in rows]
            else:
                wells = []
        finally:
            cursor.close()
            conn.close()
        return jsonify({"wells": wells}), 200
    except Exception as e:
        logging.error(f"Error fetching wells: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/api/operation-recommendations', methods=['GET'])
def get_operation_recommendations():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            ID,
            WELL_ID,
            ACTION,
            PRIORITY,
            STATUS,
            REASON,
            EXPECTED_IMPACT,
            CONFIDENCE,
            PREDICTION_SOURCE,
            PROBABILITY,
            DETAILS,
            CREATED_AT
        FROM OPERATION_RECOMMENDATION
        ORDER BY CREATED_AT DESC, WELL_ID
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        columns = [desc[0].lower() for desc in cursor.description] if cursor.description else []
        
        cursor.close()
        conn.close()
        
        recommendations = []
        if results:
            for row in results:
                rec = {}
                for i, col in enumerate(columns):
                    rec[col] = row[i] if i < len(row) else None
                recommendations.append(rec)
        
        logging.info(f"Retrieved {len(recommendations)} operation recommendations")
        return jsonify({
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations
        }), 200
        
    except Exception as e:
        logging.exception(f"Error fetching operation recommendations: {e}")
        import traceback
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route('/api/operation-suggestions', methods=['GET'])
def get_operation_suggestions_api():
    """Retrieve operation suggestions from PostgreSQL."""
    try:
        well_id = request.args.get('well_id')
        status = request.args.get('status')
        priority = request.args.get('priority')
        limit = int(request.args.get('limit', 100))
        
        suggestions = get_operation_suggestions(well_id=well_id, status=status, priority=priority, limit=limit)
        
        for sugg in suggestions:
            for key, value in sugg.items():
                if hasattr(value, 'isoformat'):
                    sugg[key] = value.isoformat()
        
        logging.info(f"Retrieved {len(suggestions)} operation suggestions")
        return jsonify({
            "status": "success",
            "count": len(suggestions),
            "suggestions": suggestions
        }), 200
        
    except Exception as e:
        logging.error(f"Error fetching operation suggestions: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/operation-suggestions/<suggestion_id>', methods=['GET'])
def get_operation_suggestion_api(suggestion_id):
    """Retrieve detailed info about a specific operation suggestion."""
    try:
        suggestion = get_operation_suggestion_detail(suggestion_id)
        
        if not suggestion:
            return jsonify({"status": "error", "message": "Suggestion not found"}), 404
        
        for key, value in suggestion.items():
            if hasattr(value, 'isoformat'):
                suggestion[key] = value.isoformat()
        
        return jsonify({
            "status": "success",
            "suggestion": suggestion
        }), 200
        
    except Exception as e:
        logging.error(f"Error fetching suggestion detail: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/operation-suggestions', methods=['POST'])
def create_operation_suggestion_api():
    """Create a new operation suggestion with OpenAI-generated detailed content."""
    try:
        data = request.get_json()
        
        well_id = data.get('well_id')
        action = data.get('action')
        status = data.get('status', 'New')
        priority = data.get('priority', 'HIGH')
        confidence = float(data.get('confidence', 0.0))
        production_data = data.get('production_data')
        sensor_metrics = data.get('sensor_metrics')
        reason = data.get('reason', '')
        expected_impact = data.get('expected_impact', '')
        
        if not well_id or not action:
            return jsonify({
                "status": "error",
                "message": "well_id and action are required"
            }), 400
        
        success = save_operation_suggestion(
            well_id=well_id,
            action=action,
            status=status,
            priority=priority,
            confidence=confidence,
            production_data=production_data,
            sensor_metrics=sensor_metrics,
            reason=reason,
            expected_impact=expected_impact
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Operation suggestion created for {well_id}",
                "well_id": well_id,
                "action": action
            }), 201
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to create operation suggestion"
            }), 500
        
    except Exception as e:
        logging.error(f"Error creating operation suggestion: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/operation-suggestions/<suggestion_id>/status', methods=['PUT'])
def update_operation_suggestion_status_api(suggestion_id):
    """Update the status of an operation suggestion."""
    try:
        data = request.get_json()
        new_status = data.get('status')
        
        if not new_status:
            return jsonify({
                "status": "error",
                "message": "status is required"
            }), 400
        
        success = update_operation_suggestion_status(suggestion_id, new_status)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Suggestion status updated to {new_status}"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update suggestion status"
            }), 500
        
    except Exception as e:
        logging.error(f"Error updating suggestion status: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Production configuration - set APP_DEBUG=False in production
    debug_mode = os.getenv("APP_DEBUG", "False").lower() == "true"
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "5000"))
    app.run(debug=debug_mode, host=host, port=port)