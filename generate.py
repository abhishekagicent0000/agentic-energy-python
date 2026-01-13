import pandas as pd
import numpy as np
import re
import os
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- 1. CONFIGURATION ---
# Set to True for quick testing, False for full 10-year generation
SAMPLE_MODE = True  

DURATION_YEARS = 10
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

# --- 2. LOAD & PARSE PARAMETERS ---
print(f"Reading parameters from {EXCEL_FILE}...")
try:
    df = pd.read_excel(EXCEL_FILE, header=None, engine='openpyxl')
except FileNotFoundError:
    print(f"ERROR: Could not find '{EXCEL_FILE}'. Make sure the file is in the same folder.")
    exit()

# Locate Header Rows and Parse
# Use forward fill for categories as merged cells often behave this way
categories = df.iloc[1].ffill()
data_names = df.iloc[2].fillna("Unknown")

try:
    # Find row with "range" in the first column
    range_row_idx = df[df[0].astype(str).str.lower().str.contains("range")].index[0]
    ranges = df.iloc[range_row_idx]
except IndexError:
    print("ERROR: Could not find a row named 'range' in the input file.")
    exit()

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
    norm_min, norm_max = vals[1] if len(vals) > 1 else (abs_min, abs_max)
    
    # Ensure Min < Max
    if abs_min > abs_max: abs_min, abs_max = abs_max, abs_min
    if norm_min > norm_max: norm_min, norm_max = norm_max, norm_min
    
    return {"min": abs_min, "max": abs_max, "norm_min": norm_min, "norm_max": norm_max}

# Map columns
columns_def = []
for i in range(1, len(data_names)):
    rng_str = ranges[i] if i < len(ranges) else ""
    parsed = parse_range(str(rng_str))
    if parsed:
        columns_def.append({
            "index": i, "category": categories[i], "name": data_names[i], "range": parsed
        })
cols_df = pd.DataFrame(columns_def)

# --- 3. DECLINE CURVE MODEL ---
def generate_decline(days, qi_oil, qi_gas, qi_water):
    Di_annual = 0.35
    b = 0.75
    D_term_annual = 0.06
    t_switch = 365
    
    Di = Di_annual / 365.0
    D_term = D_term_annual / 365.0
    t = np.arange(days)
    q_oil = np.zeros(days)
    
    mask_hyp = t < t_switch
    q_oil[mask_hyp] = qi_oil / ((1 + b * Di * t[mask_hyp])**(1/b))
    
    q_switch = qi_oil / ((1 + b * Di * t_switch)**(1/b))
    mask_exp = t >= t_switch
    q_oil[mask_exp] = q_switch * np.exp(-D_term * (t[mask_exp] - t_switch))
    
    q_gas = q_oil * (qi_gas / qi_oil)
    q_water = q_oil * (qi_water / qi_oil)
    
    return q_oil, q_gas, q_water

# --- 4. GENERATION LOOP ---
print("Starting Data Generation...")
dates_daily = pd.date_range(start='2024-01-01', periods=DURATION_YEARS*365, freq='D')

# Set duration for High Frequency data
# If SAMPLE_MODE, only 30 days. Else full duration.
hf_hours = (30 * 24) if SAMPLE_MODE else (DURATION_YEARS * 365 * 24)
dates_hf = pd.date_range(start='2024-01-01', periods=hf_hours, freq='H')

well_types = [
    (10, "Rod Pump", "Vertical", (250, 500, 350)),
    (10, "ESP", "Vertical", (250, 500, 350)),
    (10, "ESP", "Horizontal", (1500, 2500, 2500)),
    (10, "Gas Lift", "Vertical", (250, 500, 350)),
    (10, "Gas Lift", "Horizontal", (1500, 2500, 2500))
]

daily_data, hf_data = [], []

count = 1
for count_limit, w_type, orient, (qi_o, qi_g, qi_w) in well_types:
    if SAMPLE_MODE and count > 4: break # Limit wells in sample mode to allow Rod, ESP, Gas Lift

    for _ in range(count_limit if not SAMPLE_MODE else 1):
        w_id = f"Well_{count}_{w_type.replace(' ','')}_{orient[0]}"
        
        # A. Daily Production
        qo, qg, qw = generate_decline(len(dates_daily), qi_o, qi_g, qi_w)
        qo *= np.random.normal(1, 0.05, len(qo)) # Add Noise
        qg *= np.random.normal(1, 0.05, len(qg))
        qw *= np.random.normal(1, 0.05, len(qw))
        
        daily_data.append(pd.DataFrame({
            "WellID": w_id, "Date": dates_daily, 
            "Oil_Volume": qo, "Gas_Volume": qg, "Water_Volume": qw, "Lift_Type": w_type
        }))
        
        # B. High Frequency Sensors - Generate for specific well type
        w_hf = pd.DataFrame({"WellID": w_id, "Timestamp": dates_hf})
        
        if "Rod" in w_type:
            # Rod Pump sensors
            w_hf["Strokes_Per_Minute"] = np.clip(np.random.normal(15, 3, len(dates_hf)), 5, 25)
            w_hf["Torque"] = np.clip(np.random.normal(1000, 300, len(dates_hf)), 100, 5000)
            w_hf["Polish_Rod_Load"] = np.clip(np.random.normal(5000, 1500, len(dates_hf)), 500, 10000)
            w_hf["Pump_Fillage"] = np.clip(np.random.normal(75, 15, len(dates_hf)), 20, 100)
            w_hf["Tubing_Pressure"] = np.clip(np.random.normal(1200, 400, len(dates_hf)), 100, 3000)
            
        elif "ESP" in w_type:
            # ESP sensors
            w_hf["Motor_Temp"] = np.clip(np.random.normal(100, 20, len(dates_hf)), 50, 150)
            w_hf["Motor_Current"] = np.clip(np.random.normal(120, 30, len(dates_hf)), 50, 200)
            w_hf["Discharge_Pressure"] = np.clip(np.random.normal(2500, 700, len(dates_hf)), 1000, 4500)
            w_hf["Pump_Intake_Pressure"] = np.clip(np.random.normal(500, 200, len(dates_hf)), 100, 1000)
            w_hf["Motor_Voltage"] = np.clip(np.random.normal(440, 20, len(dates_hf)), 400, 480)
            
        elif "Gas" in w_type:
            # Gas Lift sensors
            w_hf["Injection_Rate"] = np.clip(np.random.normal(10, 4, len(dates_hf)), 2, 20)
            w_hf["Injection_Temp"] = np.clip(np.random.normal(50, 15, len(dates_hf)), 20, 80)
            w_hf["Bottomhole_Pressure"] = np.clip(np.random.normal(2000, 600, len(dates_hf)), 1000, 3500)
            w_hf["Injection_Pressure"] = np.clip(np.random.normal(1200, 400, len(dates_hf)), 500, 2000)
            w_hf["Cycle_Time"] = np.clip(np.random.normal(60, 20, len(dates_hf)), 10, 120)
        
        # Add generic pressure/temp columns for all (if applicable)
        if "Surface_Pressure" not in w_hf.columns:
            w_hf["Surface_Pressure"] = np.clip(np.random.normal(500, 150, len(dates_hf)), 0, 2000)
        if "Casing_Pressure" not in w_hf.columns:
            w_hf["Casing_Pressure"] = np.clip(np.random.normal(600, 200, len(dates_hf)), 0, 2500)
        if "Wellhead_Temp" not in w_hf.columns:
            w_hf["Wellhead_Temp"] = np.clip(np.random.normal(40, 10, len(dates_hf)), 20, 80)
            
        hf_data.append(w_hf)
        count += 1
        print(f"Generated {w_id}")

# --- 5. STORE IN SNOWFLAKE ---
print(f"Connecting to Snowflake...")
try:
    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()
    
    # Clean existing data
    print("Cleaning old data...")
    cursor.execute("DELETE FROM well_daily_production")
    cursor.execute("DELETE FROM well_sensor_readings")
    cursor.execute("DELETE FROM well_anomalies")
    
    # Insert Daily Data
    if daily_data:
        print("Inserting well_daily_production...")
        df_daily = pd.concat(daily_data)
        
        # Batch insert
        data_tuples = [(row['WellID'], str(row['Date'].date()), row['Oil_Volume'], row['Gas_Volume'], row['Water_Volume'], row['Lift_Type']) 
                       for _, row in df_daily.iterrows()]
        cursor.executemany("""
            INSERT INTO well_daily_production (well_id, date, oil_volume, gas_volume, water_volume, lift_type)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, data_tuples)
        
        print(f"Inserted {len(df_daily)} daily records.")
    
    # Insert High Frequency Data
    if hf_data:
        print("Inserting well_sensor_readings...")
        df_hf = pd.concat(hf_data, ignore_index=True)
        
        # Convert timestamp to string
        df_hf['Timestamp'] = df_hf['Timestamp'].astype(str)
        
        # Rename columns to match Snowflake schema
        column_mapping = {
            'WellID': 'well_id',
            'Timestamp': 'timestamp',
            'Strokes_Per_Minute': 'strokes_per_minute',
            'Torque': 'torque',
            'Polish_Rod_Load': 'polish_rod_load',
            'Pump_Fillage': 'pump_fillage',
            'Tubing_Pressure': 'tubing_pressure',
            'Motor_Temp': 'motor_temp',
            'Motor_Current': 'motor_current',
            'Discharge_Pressure': 'discharge_pressure',
            'Pump_Intake_Pressure': 'pump_intake_pressure',
            'Motor_Voltage': 'motor_voltage',
            'Injection_Rate': 'injection_rate',
            'Injection_Temp': 'injection_temperature',
            'Bottomhole_Pressure': 'bottomhole_pressure',
            'Injection_Pressure': 'injection_pressure',
            'Cycle_Time': 'cycle_time',
            'Surface_Pressure': 'surface_pressure',
            'Casing_Pressure': 'casing_pressure',
            'Wellhead_Temp': 'wellhead_temp'
        }
        
        # Rename existing columns
        df_hf = df_hf.rename(columns=column_mapping)
        
        # Add lift_type based on well_id pattern
        def get_lift_type(well_id):
            if 'RodPump' in well_id:
                return 'Rod Pump'
            elif 'ESP' in well_id:
                return 'ESP'
            elif 'GasLift' in well_id:
                return 'Gas Lift'
            return 'Rod Pump'
        
        df_hf['lift_type'] = df_hf['well_id'].apply(get_lift_type)
        
        # List of all 21 required columns
        all_columns = [
            'well_id', 'timestamp', 'lift_type',
            'strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure',
            'motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage',
            'injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time',
            'surface_pressure', 'casing_pressure', 'wellhead_temp'
        ]
        
        # Ensure all columns exist, fill missing with None
        for col in all_columns:
            if col not in df_hf.columns:
                df_hf[col] = None
        
        # Select only required columns and convert to list of tuples
        df_hf = df_hf[all_columns].copy()
        
        # Convert NaN/inf to None for Snowflake
        df_hf = df_hf.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Build insert statement
        column_list = ', '.join(all_columns)
        placeholders = ', '.join(['%s'] * len(all_columns))
        
        # Batch insert using qmark style
        batch_size = 1000
        for i in range(0, len(df_hf), batch_size):
            batch = df_hf.iloc[i:i+batch_size]
            data_tuples = [tuple(None if pd.isna(v) else v for v in row) for _, row in batch.iterrows()]
            
            try:
                cursor.executemany(f"""
                    INSERT INTO well_sensor_readings ({column_list})
                    VALUES ({placeholders})
                """, data_tuples)
            except Exception as e:
                print(f"Error inserting batch {i//batch_size}: {e}")
                raise
        
        print(f"Inserted {len(df_hf)} sensor records.")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("DONE: Data generated and stored in Snowflake.")
    
except Exception as e:
    print(f"ERROR: Failed to insert data into Snowflake: {str(e)}")
    raise
