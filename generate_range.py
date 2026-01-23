import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────────────────────────────────────────
#                        CONFIGURATION
# ───────────────────────────────────────────────────────────────

TEST_MODE = True

# 🎯 Highest priority: inject data only for this well
TARGET_WELL_ID = "Well_003_RodPump"  # e.g. "Well_001_RodPump"

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
#                        INPUT HELPERS
# ───────────────────────────────────────────────────────────────

def get_date_input(prompt):
    while True:
        try:
            return datetime.strptime(input(prompt).strip(), '%Y-%m-%d').date()
        except ValueError:
            print("Invalid format. Use YYYY-MM-DD.")

def get_scenario_input():
    print("\n--- SELECT DATA MODE ---")
    print("1. GOOD DATA (Stable)")
    print("2. BAD DATA: Pressure Instability")
    print("3. BAD DATA: Efficiency Degradation")
    print("4. BAD DATA: Production Crash")
    print("5. BAD DATA: Ghost Production")
    return input("Select (1-5): ").strip()

# ───────────────────────────────────────────────────────────────
#                        DATA GENERATOR
# ───────────────────────────────────────────────────────────────

def generate_data(start_date, end_date, scenario):
    dates_daily = pd.date_range(start=start_date, end=end_date, freq='D')
    dates_hf = pd.date_range(start=start_date, end=end_date + timedelta(days=1), freq='h', inclusive='left')

    well_types = [
        (1 if TEST_MODE else 15, "Rod Pump", (300, 600, 400)),
        (1 if TEST_MODE else 20, "ESP", (1200, 2000, 1800)),
        (1 if TEST_MODE else 15, "Gas Lift", (800, 1500, 1200))
    ]

    daily_data, hf_data = [], []
    count = 1

    for count_limit, w_type, initial_rates in well_types:
        for _ in range(count_limit):
            w_id = TARGET_WELL_ID if TARGET_WELL_ID else f"Well_{count:03d}_{w_type.replace(' ', '')}"
            qi_o, qi_g, qi_w = initial_rates
            
            # --- BASELINE PRODUCTION ---
            oil = np.random.normal(qi_o, qi_o * 0.02, len(dates_daily))
            gas = np.random.normal(qi_g, qi_g * 0.05, len(dates_daily))
            water = np.random.normal(qi_w, qi_w * 0.05, len(dates_daily))

            # --- SENSOR BASELINES (HEALTHY) ---
            hf = pd.DataFrame({"WellID": w_id, "Timestamp": dates_hf})
            hf["Surface_Pressure"] = np.random.normal(500, 20, len(dates_hf))
            hf["Casing_Pressure"] = np.random.normal(600, 30, len(dates_hf))
            hf["Wellhead_Temp"] = np.random.normal(40, 2, len(dates_hf))

            if w_type == "Rod Pump":
                hf["Strokes_Per_Minute"] = np.random.normal(15, 1, len(dates_hf))
                hf["Pump_Fillage"] = np.random.normal(90, 5, len(dates_hf))
                hf["Tubing_Pressure"] = np.random.normal(1200, 50, len(dates_hf))
                hf["Motor_Current"] = np.random.normal(15, 2, len(dates_hf)) # Add a healthy motor current

            # --- INJECT ANOMALY LOGIC ---

            # Scenario 2: Pressure Instability
            if scenario == "2" and "Tubing_Pressure" in hf.columns:
                print("Injecting: Pressure Instability")
                # Make pressure bounce wildly
                hf["Tubing_Pressure"] += np.random.normal(0, 300, len(dates_hf))
                # Slightly reduce production due to instability
                oil *= 0.9

            # Scenario 3: Efficiency Degradation (e.g., Tubing Leak)
            if scenario == "3" and "Tubing_Pressure" in hf.columns:
                print("Injecting: Efficiency Degradation (Tubing Leak)")
                # Drop tubing pressure to simulate a leak
                hf["Tubing_Pressure"] *= 0.5 
                # Production drops, but motor keeps running normally
                oil *= 0.4

            # Scenario 4: Production Crash (e.g., Stuck Pump)
            if scenario == "4" and "Motor_Current" in hf.columns:
                print("Injecting: Production Crash (Stuck Pump)")
                # Spike motor current
                hf["Motor_Current"] *= 2.5
                # Production drops to almost zero
                oil *= 0.05

            # Scenario 5: Ghost Production (e.g., Parted Rods)
            if scenario == "5" and "Motor_Current" in hf.columns:
                print("Injecting: Ghost Production (Parted Rods)")
                # Drop motor current (no load)
                hf["Motor_Current"] *= 0.3
                # Production goes to zero
                oil[:] = 0
                gas[:] = 0
                water[:] = 0
            
            # --- FINALIZE DATA ---
            daily_data.append(pd.DataFrame({
                "WellID": w_id, "Date": dates_daily, "Oil_Volume": np.abs(oil),
                "Gas_Volume": np.abs(gas), "Water_Volume": np.abs(water), "Lift_Type": w_type
            }))
            
            hf_data.append(hf)
            count += 1
            if TARGET_WELL_ID: break
        if TARGET_WELL_ID: break

    return pd.concat(daily_data), pd.concat(hf_data)

# ───────────────────────────────────────────────────────────────
#                        MAIN EXECUTION
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    start_date = get_date_input("Start Date (YYYY-MM-DD): ")
    end_date = get_date_input("End Date (YYYY-MM-DD): ")
    scenario = get_scenario_input()

    df_daily, df_hf = generate_data(start_date, end_date, scenario)

    print("\nConnecting to Snowflake...")
    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()

    try:
        where_clause = ""
        if TARGET_WELL_ID:
            where_clause = f"AND well_id = '{TARGET_WELL_ID}'"

        print("Deleting old data...")
        cursor.execute(
            f"DELETE FROM well_daily_production WHERE date BETWEEN '{start_date}' AND '{end_date}' {where_clause}"
        )
        cursor.execute(
            f"""DELETE FROM well_sensor_readings
                WHERE TO_DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}' {where_clause}"""
        )

        print("Inserting daily production...")
        cursor.executemany("""
            INSERT INTO well_daily_production
            (well_id, date, oil_volume, gas_volume, water_volume, lift_type)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, [
            (r.WellID, r.Date.strftime('%Y-%m-%d'), r.Oil_Volume, r.Gas_Volume, r.Water_Volume, r.Lift_Type)
            for r in df_daily.itertuples()
        ])

        print("Inserting sensor data...")
        df_hf['Timestamp'] = df_hf['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_hf = df_hf.rename(columns={"WellID": "well_id", "Timestamp": "timestamp"})
        df_hf["lift_type"] = df_hf["well_id"].apply(
            lambda x: "Rod Pump" if "RodPump" in x else "ESP" if "ESP" in x else "Gas Lift"
        )

        cols = df_hf.columns.tolist()
        placeholders = ",".join(["%s"] * len(cols))
        cursor.executemany(
            f"INSERT INTO well_sensor_readings ({','.join(cols)}) VALUES ({placeholders})",
            [tuple(row) for row in df_hf.itertuples(index=False)]
        )

        conn.commit()
        print("\n✅ SUCCESS: Test data inserted")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    finally:
        conn.close()
