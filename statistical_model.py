import pandas as pd
import re
import json
import logging
import os
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXCEL_FILE = 'data types and ranges.xlsx'

# Snowflake Configuration
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

def parse_range(range_str):
    if not isinstance(range_str, str): return None
    range_str = range_str.strip()
    # Matches patterns like "100-200" or "100 - 200"
    pairs = re.findall(r"([\d,.]+)\s*-\s*([\d,.]+)", range_str)
    if not pairs: return None
    
    def to_float(s):
        try: return float(s.replace(',', ''))
        except: return 0.0
    
    vals = [(to_float(a), to_float(b)) for a, b in pairs]
    
    if not vals:
        return None

    # 1st pair = Absolute Limits, 2nd pair = Normal Limits (if present)
    abs_min, abs_max = vals[0]
    
    # Ensure Min < Max
    if abs_min > abs_max: abs_min, abs_max = abs_max, abs_min
    
    return {"min": abs_min, "max": abs_max}

def normalize_name(raw_name):
    clean_name = raw_name.strip()
    if "Tubing" in raw_name and "Pressure" in raw_name: return "tubing_pressure"
    elif "Casing" in raw_name and "Pressure" in raw_name: return "casing_pressure"
    elif "Surface" in raw_name and "Pressure" in raw_name: return "surface_pressure"
    elif "Pump Intake Pressure" in raw_name: return "surface_pressure"
    elif "Motor" in raw_name and "Temp" in raw_name: return "motor_temp"
    elif "Discharge" in raw_name and "Temp" in raw_name: return "wellhead_temp"
    elif "Intake/Fluid Temp" in raw_name: return "wellhead_temp"
    elif "Motor" in raw_name and "Current" in raw_name: return "motor_current"
    
    return None

def main():
    logging.info("Starting Statistical Rules Model...")
    
    # --- 1. LOAD PARAMETERS ---
    logging.info(f"Reading rules from {EXCEL_FILE}...")
    try:
        df = pd.read_excel(EXCEL_FILE, header=None, engine='openpyxl')
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        return

    # Helper to find relevant rows based on structure in generate.py
    categories = df.iloc[1].ffill()
    data_names = df.iloc[2].fillna("Unknown")
    
    try:
        range_row_idx = df[df[0].astype(str).str.lower().str.contains("range")].index[0]
        ranges = df.iloc[range_row_idx]
    except IndexError:
        logging.error("Could not find a row named 'range' in the input file.")
        return

    rules = []
    
    # Iterate and extract rules
    for i in range(1, len(data_names)):
        raw_name = str(data_names[i])
        rng_str = ranges[i] if i < len(ranges) else ""
        
        parsed = parse_range(str(rng_str))
        if parsed:
            db_col = normalize_name(raw_name)
            if db_col:
                rules.append({
                    "name": raw_name,
                    "db_col": db_col,
                    "min": parsed['min'],
                    "max": parsed['max']
                })

    logging.info(f"Loaded {len(rules)} rules based on DB mapping.")
    
    # --- 2. CONNECT TO DB ---
    conn = duckdb.connect(DB_FILE)
    
    # Clear old statistical model anomalies to avoid duplicates?
    # Or maybe we just append. For this task, assuming fresh run or append is fine.
    # Let's delete old runs for this model to keep it clean.
    conn.execute("DELETE FROM well_anomalies WHERE model_name = 'Statistical_Rules'")
    
    # --- 3. APPLY RULES ---
    total_anomalies = 0
    
    for rule in rules:
        col = rule['db_col']
        limit_min = rule['min']
        limit_max = rule['max']
        
        logging.info(f"Checking {col} (Rule: {rule['name']} [{limit_min}, {limit_max}])...")
        
        # Query for violations
        query = f"""
            SELECT well_id, timestamp, {col} 
            FROM well_sensor_readings 
            WHERE {col} < {limit_min} OR {col} > {limit_max}
        """
        
        violations_df = conn.execute(query).fetchdf()
        
        if not violations_df.empty:
            count = len(violations_df)
            logging.info(f"  -> Found {count} violations for {rule['name']}")
            total_anomalies += count
            
            # Prepare insertion
            # Schema: well_id, timestamp, anomaly_type, anomaly_score, raw_values, processed_features, model_name, status
            
            # We construct the DF for insertion
            violations_df['anomaly_type'] = f"Out of Bounds: {rule['name']}"
            violations_df['anomaly_score'] = 1.0 # Statistical hard limit is 100% anomaly
            violations_df['model_name'] = 'Statistical_Rules'
            violations_df['status'] = 'New'
            
            # Create JSON for raw_values: e.g. {"tubing_pressure": 123.4}
            # DuckDB can insert JSON types if mapped correctly, or we pass strings
            # violations_df['raw_values'] = violations_df.apply(lambda r: json.dumps({col: r[col]}), axis=1) # Slow in pandas
            
            # Faster SQL-based insertion might be better for large data, but let's try pandas first.
            # If 4M rows, this might be slow. DuckDB SQL is better.
            
            insert_query = f"""
                INSERT INTO well_anomalies (well_id, timestamp, anomaly_type, anomaly_score, raw_values, model_name, status)
                SELECT 
                    well_id, 
                    timestamp, 
                    'Out of Bounds: {col}' as anomaly_type, 
                    1.0 as anomaly_score, 
                    json_object('{col}', {col}) as raw_values,
                    'Statistical_Rules' as model_name,
                    'New' as status
                FROM well_sensor_readings
                WHERE {col} < {limit_min} OR {col} > {limit_max}
            """
            
            conn.execute(insert_query)
            
        else:
            logging.info(f"  -> No violations.")

    logging.info(f"Analysis Complete. Total anomalies detected: {total_anomalies}")
    conn.close()

if __name__ == "__main__":
    main()
