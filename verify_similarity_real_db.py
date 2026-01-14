import os
import pandas as pd
import json
import logging
from dotenv import load_dotenv
try:
    from snowflake.connector import connect as snowflake_connect
except ImportError:
    print("snowflake-connector-python not found")
    exit(1)

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)

# --- Config and Helpers Copied from app.py to avoid side effects ---

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
    try:
        conn = get_db_connection()
        
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
        
        params = []
        if filter_clauses:
            query += " WHERE " + " OR ".join(filter_clauses)
            if filter_params:
                params.extend(filter_params)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        print(f"Executing query...")
        df = pd.read_sql(query, conn, params=params)
        df.columns = [c.lower() for c in df.columns]
        conn.close()
        return df
    except Exception as e:
        logging.warning(f"Could not fetch historical anomalies: {e}")
        return pd.DataFrame()

def find_similar_anomalies(current_readings, current_violations, historical_df, lift_type=None, similarity_threshold=0.8):
    if historical_df.empty or not current_violations:
        return []

    similar_records = []
    
    violated_keys = [v['field'] for v in current_violations if 'field' in v]
    
    for _, row in historical_df.iterrows():
        try:
            hist_values = row['raw_values']
            if isinstance(hist_values, str):
                hist_values = json.loads(hist_values)
            elif not isinstance(hist_values, dict):
                continue
            
            # IMPLICIT LIFT TYPE FILTERING (Synced with app.py):
            common_keys = set(hist_values.keys()) & set(current_readings.keys())
            if not common_keys:
                 continue
                
            matches = 0
            total_score = 0
            
            for key in violated_keys:
                if key in hist_values and hist_values[key] is not None and current_readings.get(key) is not None:
                    curr_val = float(current_readings[key])
                    msg_val = hist_values[key]
                    
                    # Clean up value if needed (handle strings in JSON)
                    try:
                        hist_val = float(msg_val)
                    except (ValueError, TypeError):
                        continue
                    
                    max_val = max(abs(curr_val), abs(hist_val))
                    if max_val == 0:
                        similarity = 1.0
                    else:
                        similarity = 1.0 - (abs(curr_val - hist_val) / max_val)
                    
                    if similarity >= similarity_threshold:
                        matches += 1
                        total_score += similarity
            
            if matches > 0:
                avg_similarity = total_score / matches
                record = {
                    "well_id": row.get('well_id', 'Unknown'),
                    "timestamp": row['timestamp'],
                    "alert_title": row['violation_summary'],
                    "severity": row['severity'],
                    "similarity_score": avg_similarity,
                    "match_count": matches,
                    "raw_anomaly_data": {
                        "violations": [{"field": k, "value": hist_values.get(k), "violation": f"Historical Value: {hist_values.get(k)}"} for k in violated_keys],
                        "full_readings": hist_values
                    } 
                }
                similar_records.append(record)
                
        except Exception as e:
            continue
            
    similar_records.sort(key=lambda x: (x['match_count'], x['similarity_score']), reverse=True)
    return similar_records[:3]


# --- LLM Integration (Copied/Adapted from app.py to ensure identical logic) ---
import openai

# OpenAI Config
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_API_BASE"),
    default_headers={
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Well Anomaly Detection",
    }
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def calculate_severity(anomaly_details, anomaly_score, readings):
    # Simplified version matching app.py logic roughly for severity
    violations = anomaly_details.get('violations', [])
    num_violations = len(violations)
    if num_violations >= 4 or anomaly_score >= 0.75:
        return "CRITICAL"
    elif num_violations >= 3 or anomaly_score >= 0.5:
        return "HIGH"
    elif num_violations >= 2 or anomaly_score >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"

def generate_suggestion(anomaly_details, well_id, readings, timestamp_str, historical_anomalies=None):
    try:
        formatted_history = []
        if historical_anomalies:
            for anom in historical_anomalies:
                raw_data = anom.get('raw_anomaly_data')
                past_violations_list = []
                if raw_data and 'violations' in raw_data:
                    past_violations_list = raw_data['violations']

                formatted_history.append({
                    "well": anom.get('well_id'),          
                    "date": str(anom.get('timestamp')),   
                    "issue": anom.get('alert_title'),
                    "severity": anom.get('severity'),
                    "violations": past_violations_list,
                    "readings": raw_data.get('full_readings', {}),
                })
        
        history_context_str = json.dumps(formatted_history, indent=2)
        calculated_severity = calculate_severity(anomaly_details, anomaly_details.get('anomaly_score', 0), readings)
 
        prompt = f"""
        You are an expert Production Engineer AI assistant. Analyze the new anomaly.
        
        ---
        New Anomaly Details:
        Well ID: {well_id}
        Timestamp: {timestamp_str}
        Current Sensor Readings:
        {json.dumps(readings, indent=2)}
        Violations Detected:
        {json.dumps(anomaly_details['violations'], indent=2)}
        ---
        Real Database History for this well (for context):
        {history_context_str}
        ---
 
        Based on the information above, provide a structured JSON response.

        SEVERITY CLASSIFICATION GUIDE:
        - CRITICAL: >50% deviation, 4+ violations, or score >= 0.75
        - HIGH: 30-50% deviation, 3 violations, or score >= 0.5
        - MEDIUM: 10-30% deviation, 2 violations, or score >= 0.25
        - LOW: <10% deviation, 1 violation, or score < 0.25
        
        PRE-CLASSIFIED SEVERITY: {calculated_severity}
        ANOMALY SCORE: {anomaly_details.get('anomaly_score', 0):.2f}
        Strictly assign severity level that matches or stays close to this pre-classification.

        Important:
        1. If 'Real Database History' above is not an empty list, you MUST strictly copy that data into 'similar_incidents'.
        
        Required JSON Structure:
        1. "alert_title": Short title.
        2. "severity": Severity level.
        3. "status": "ACTIVE".
        4. "confidence": Percentage string.
        5. "description": Concise technical paragraph.
        6. "suggested_actions": List of steps.
        7. "explanation": Detailed explanation.
        8. "historical_context": Object containing:
            - "similar_incidents": List of objects based ONLY on the 'Real Database History' provided above.
            - "key_learnings": String field analysis.
            
        Ensure your entire output is a single, valid JSON object.
        """
 
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for oil and gas anomaly detection. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
 
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        return json.loads(content.strip())
 
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {}

# --- Main Test Execution ---
if __name__ == "__main__":
    print("--- Verifying with REAL Snowflake Data ---")
    
    # 1. Define Dummy Current Anomaly
    current_readings = {
        "motor_temp": 280,        
        "motor_current": 120,    
        "tubing_pressure": 450
    }
    current_violations = [
        {"field": "motor_temp", "value": 280, "violation": "High Temp"}
    ]
    anomaly_details = {
        "violations": current_violations,
        "anomaly_score": 0.8
    }
    
    print(f"Current Dummy Issue: {current_violations[0]['violation']} (Temp: {current_readings['motor_temp']})")

    # 2. Fetch Real History
    print("\nFetching last 100 anomalies from Snowflake...")
    df = get_historical_anomalies(limit=100)
    
    if df.empty:
        print("WARNING: No data returned from Snowflake. Cannot verify similarity logic.")
    else:
        print(f"Successfully fetched {len(df)} records.")
        
        # 3. Run Similarity Logic
        print("\nRunning find_similar_anomalies...")
        matches = find_similar_anomalies(current_readings, current_violations, df)
        
        print(f"\nFound {len(matches)} matches (Threshold > 0.8).")
        for i, m in enumerate(matches):
            print(f"\n--- Result {i+1} ---")
            print(f"Well: {m['well_id']}")
            print(f"Similarity Score: {m['similarity_score']:.2f}")
            print(json.dumps(m.get('raw_anomaly_data'), indent=2, default=str))

        # 4. Generate AI Suggestion
        print("\n--- Generating AI Suggestion (Real OpenAI Call) ---")
        
        # Inspect the context being built (for verification purposes)
        formatted_history = []
        if matches:
            for anom in matches:
                raw_data = anom.get('raw_anomaly_data')
                past_violations_list = []
                if raw_data and 'violations' in raw_data:
                    past_violations_list = raw_data['violations']

                formatted_history.append({
                    "well": anom.get('well_id'),          
                    "date": str(anom.get('timestamp')),   
                    "issue": anom.get('alert_title'),
                    "severity": anom.get('severity'),
                    "violations": past_violations_list,
                    "readings": raw_data.get('full_readings', {})
                })
        print("\n[VERIFICATION] Historical Context Structure Sent to LLM:")
        print(json.dumps(formatted_history, indent=2, default=str))
        
        suggestion = generate_suggestion(
            anomaly_details=anomaly_details,
            well_id="Dummy_Well_For_Test",
            readings=current_readings,
            timestamp_str="2026-01-14 12:00:00",
            historical_anomalies=matches
        )
        
        print("\n--- AI Response ---")
        print(json.dumps(suggestion, indent=2))
        
        # Verification check
        try:
            hist_ctx = suggestion.get("historical_context", {})
            incidents = hist_ctx.get("similar_incidents", [])
            if incidents:
                print(f"\n[VERIFICATION] Similar Incidents Found: {len(incidents)}")
                first_incident = incidents[0]
                print(f"Severity Present: {'severity' in first_incident} ({first_incident.get('severity')})")
                print(f"Readings Summary Present: {'readings_summary' in first_incident}")
                if 'readings_summary' in first_incident:
                    print(f"Readings Summary Content: {first_incident['readings_summary']}")
        except Exception as e:
            print(f"Verification Check Failed: {e}")


    print("\n--- Done ---")
