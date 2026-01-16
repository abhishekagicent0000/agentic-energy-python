import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

# ───────────────────────────────────────────────────────────────
#                        CONFIGURATION
# ───────────────────────────────────────────────────────────────

# How many last years of data you want to generate
GENERATE_LAST_YEARS = 2          # ← Change this value

SAMPLE_MODE = False              # True = very small dataset for testing

if SAMPLE_MODE:
    GENERATE_LAST_YEARS = 0.08   # ~1 month for quick testing

EXCEL_FILE = 'data types and ranges.xlsx'

# Snowflake connection settings (loaded from .env)
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

# ───────────────────────────────────────────────────────────────
#                       DATE RANGE SETUP
# ───────────────────────────────────────────────────────────────

DAYS_PER_YEAR = 365.25
TOTAL_DAYS = int(GENERATE_LAST_YEARS * DAYS_PER_YEAR)

END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=TOTAL_DAYS)

dates_daily = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq='D',
    inclusive='both'
)[:TOTAL_DAYS]

TOTAL_HOURS = len(dates_daily) * 24
dates_hf = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq='H',
    inclusive='both'
)[:TOTAL_HOURS]

print(f"Data period: {dates_daily[0].date()}  →  {dates_daily[-1].date()}")
print(f"Daily records per well:   {len(dates_daily):,d}")
print(f"Hourly records per well:  {len(dates_hf):,d}\n")

# ───────────────────────────────────────────────────────────────
#                  LOAD PARAMETERS FROM EXCEL
# ───────────────────────────────────────────────────────────────

print(f"Reading parameters from {EXCEL_FILE}...")
try:
    df_excel = pd.read_excel(EXCEL_FILE, header=None, engine='openpyxl')
except FileNotFoundError:
    print(f"ERROR: File '{EXCEL_FILE}' not found!")
    exit()

categories = df_excel.iloc[1].ffill()
data_names = df_excel.iloc[2].fillna("Unknown")

try:
    range_row_idx = df_excel[df_excel[0].astype(str).str.lower().str.contains("range")].index[0]
    ranges = df_excel.iloc[range_row_idx]
except IndexError:
    print("ERROR: 'range' row not found in first column")
    exit()

def parse_range(range_str):
    if not isinstance(range_str, str):
        return None
    range_str = str(range_str).strip()
    pairs = re.findall(r"([\d,.]+)\s*-\s*([\d,.]+)", range_str)
    if not pairs:
        return None

    def to_float(s):
        try:
            return float(s.replace(',', ''))
        except:
            return 0.0

    vals = [(to_float(a), to_float(b)) for a, b in pairs]
    if not vals:
        return None

    abs_min, abs_max = vals[0]
    norm_min, norm_max = vals[1] if len(vals) > 1 else (abs_min, abs_max)

    if abs_min > abs_max: abs_min, abs_max = abs_max, abs_min
    if norm_min > norm_max: norm_min, norm_max = norm_max, norm_min

    return {"min": abs_min, "max": abs_max, "norm_min": norm_min, "norm_max": norm_max}

# ───────────────────────────────────────────────────────────────
#                    DECLINE CURVE FUNCTION
# ───────────────────────────────────────────────────────────────

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
    q_oil[mask_hyp] = qi_oil / ((1 + b * Di * t[mask_hyp]) ** (1 / b))

    q_switch = qi_oil / ((1 + b * Di * t_switch) ** (1 / b))
    mask_exp = t >= t_switch
    q_oil[mask_exp] = q_switch * np.exp(-D_term * (t[mask_exp] - t_switch))

    q_gas = q_oil * (qi_gas / qi_oil) if qi_oil > 0 else q_oil
    q_water = q_oil * (qi_water / qi_oil) if qi_oil > 0 else q_oil

    return q_oil, q_gas, q_water

# ───────────────────────────────────────────────────────────────
#                       WELL TYPES (ONLY 3)
# ───────────────────────────────────────────────────────────────

well_types = [
    (15, "Rod Pump",  (300,  600,   400)),
    (20, "ESP",       (1200, 2000,  1800)),
    (15, "Gas Lift",  (800,  1500,  1200))
]

# ───────────────────────────────────────────────────────────────
#                        DATA GENERATION
# ───────────────────────────────────────────────────────────────

daily_data = []
hf_data = []

count = 1
print("Generating wells...\n")

for count_limit, w_type, initial_rates in well_types:
    wells_to_create = count_limit if not SAMPLE_MODE else min(2, count_limit)

    for _ in range(wells_to_create):
        w_id = f"Well_{count:03d}_{w_type.replace(' ','')}"
        qi_o, qi_g, qi_w = initial_rates

        # Daily production
        qo, qg, qw = generate_decline(len(dates_daily), qi_o, qi_g, qi_w)
        qo *= np.random.normal(1, 0.05, len(qo))
        qg *= np.random.normal(1, 0.05, len(qg))
        qw *= np.random.normal(1, 0.05, len(qw))

        daily_df = pd.DataFrame({
            "WellID": w_id,
            "Date": dates_daily,
            "Oil_Volume": qo,
            "Gas_Volume": qg,
            "Water_Volume": qw,
            "Lift_Type": w_type
        })
        daily_data.append(daily_df)

        # High frequency sensors
        w_hf = pd.DataFrame({"WellID": w_id, "Timestamp": dates_hf})

        if w_type == "Rod Pump":
            w_hf["Strokes_Per_Minute"] = np.clip(np.random.normal(15, 3, len(dates_hf)), 5, 25)
            w_hf["Torque"] = np.clip(np.random.normal(1000, 300, len(dates_hf)), 100, 5000)
            w_hf["Polish_Rod_Load"] = np.clip(np.random.normal(5000, 1500, len(dates_hf)), 500, 10000)
            w_hf["Pump_Fillage"] = np.clip(np.random.normal(75, 15, len(dates_hf)), 20, 100)
            w_hf["Tubing_Pressure"] = np.clip(np.random.normal(1200, 400, len(dates_hf)), 100, 3000)

        elif w_type == "ESP":
            w_hf["Motor_Temp"] = np.clip(np.random.normal(100, 20, len(dates_hf)), 50, 150)
            w_hf["Motor_Current"] = np.clip(np.random.normal(120, 30, len(dates_hf)), 50, 200)
            w_hf["Discharge_Pressure"] = np.clip(np.random.normal(2500, 700, len(dates_hf)), 1000, 4500)
            w_hf["Pump_Intake_Pressure"] = np.clip(np.random.normal(500, 200, len(dates_hf)), 100, 1000)
            w_hf["Motor_Voltage"] = np.clip(np.random.normal(440, 20, len(dates_hf)), 400, 480)

        elif w_type == "Gas Lift":
            w_hf["Injection_Rate"] = np.clip(np.random.normal(10, 4, len(dates_hf)), 2, 20)
            w_hf["Injection_Temp"] = np.clip(np.random.normal(50, 15, len(dates_hf)), 20, 80)
            w_hf["Bottomhole_Pressure"] = np.clip(np.random.normal(2000, 600, len(dates_hf)), 1000, 3500)
            w_hf["Injection_Pressure"] = np.clip(np.random.normal(1200, 400, len(dates_hf)), 500, 2000)
            w_hf["Cycle_Time"] = np.clip(np.random.normal(60, 20, len(dates_hf)), 10, 120)

        # Common sensors for all wells
        common_sensors = [
            ("Surface_Pressure", 500, 150, 0, 2000),
            ("Casing_Pressure",  600, 200, 0, 2500),
            ("Wellhead_Temp",    40,  10, 20, 80)
        ]

        for col, mean, std, minv, maxv in common_sensors:
            if col not in w_hf.columns:
                w_hf[col] = np.clip(np.random.normal(mean, std, len(dates_hf)), minv, maxv)

        hf_data.append(w_hf)

        print(f"  Generated {w_id}   ({len(dates_daily):,} daily / {len(dates_hf):,} hourly)")
        count += 1

# ───────────────────────────────────────────────────────────────
#                        SAVE TO SNOWFLAKE
# ───────────────────────────────────────────────────────────────

if not daily_data or not hf_data:
    print("No data was generated. Exiting.")
    exit()

print("\nConnecting to Snowflake...")

try:
    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    print("Clearing previous data...")
    cursor.execute("DELETE FROM well_daily_production")
    cursor.execute("DELETE FROM well_sensor_readings")

    # Daily data insert
    print("Inserting daily production...")
    df_daily = pd.concat(daily_data, ignore_index=True)
    daily_tuples = [
        (row['WellID'], row['Date'].strftime('%Y-%m-%d'),
         float(row['Oil_Volume']), float(row['Gas_Volume']),
         float(row['Water_Volume']), row['Lift_Type'])
        for _, row in df_daily.iterrows()
    ]

    cursor.executemany("""
        INSERT INTO well_daily_production
        (well_id, date, oil_volume, gas_volume, water_volume, lift_type)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, daily_tuples)

    print(f"→ Inserted {len(daily_tuples):,} daily records")

    # High frequency data insert
    print("Inserting high-frequency sensor data...")
    df_hf = pd.concat(hf_data, ignore_index=True)
    df_hf['Timestamp'] = df_hf['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

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

    df_hf = df_hf.rename(columns=column_mapping)

    df_hf['lift_type'] = df_hf['well_id'].apply(
        lambda x: 'Rod Pump' if 'RodPump' in x else
                  'ESP' if 'ESP' in x else
                  'Gas Lift' if 'GasLift' in x else 'Unknown'
    )

    all_columns = [
        'well_id', 'timestamp', 'lift_type',
        'strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure',
        'motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage',
        'injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time',
        'surface_pressure', 'casing_pressure', 'wellhead_temp'
    ]

    for col in all_columns:
        if col not in df_hf.columns:
            df_hf[col] = None

    df_hf = df_hf[all_columns].replace({np.nan: None, np.inf: None, -np.inf: None})

    batch_size = 5000
    total = len(df_hf)
    placeholders = ', '.join(['%s'] * len(all_columns))

    for i in range(0, total, batch_size):
        batch = df_hf.iloc[i:i+batch_size]
        data_tuples = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        cursor.executemany(f"""
            INSERT INTO well_sensor_readings ({', '.join(all_columns)})
            VALUES ({placeholders})
        """, data_tuples)
        print(f"  → Batch {i//batch_size + 1}  ({i + len(batch):,} / {total:,})")

    conn.commit()
    print("\nSUCCESS: Data generation and upload completed!")

except Exception as e:
    print(f"\nERROR during Snowflake operation:\n{str(e)}")
    raise

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()