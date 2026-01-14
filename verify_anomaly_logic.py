
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from unittest.mock import patch

load_dotenv()
from anomaly_review_service import detect_anomalies, fetch_well_data
import argparse
import sys

# Mock Helper to create base DataFrame
def create_mock_df(days=60, lift_type='Rod Pump'):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    df = pd.DataFrame({
        'well_id': ['Well_Test'] * days,
        'timestamp': dates,
        'lift_type': [lift_type] * days,
        'strokes_per_minute': [8.0] * days,     # Constant high SPM
        'motor_current': [12.0] * days,
        'pump_fillage': [80.0] * days,
        'tubing_pressure': [100.0] * days,       # Added to satisfy feature count > 3
        'pump_intake_pressure': [200.0] * days,
        'oil_volume': [100.0] * days,           # Perfect correlation
        'water_volume': [50.0] * days,
        'gas_volume': [10.0] * days
    })
    return df

# Mock Ranges for testing
MOCK_RANGES = {
    'tubing_pressure': {'min': 0, 'max': 5000},
    'strokes_per_minute': {'min': 0, 'max': 20}
}

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_range_check(mock_get_ranges):
    print("\n--- Test 0: Data Quality (Range Check) ---")
    df = create_mock_df()
    # Set Invalid Value
    df.iloc[-1, df.columns.get_loc('tubing_pressure')] = -100 # Below Min 0
    
    results = detect_anomalies("Test_Range", df)
    
    found = False
    found = False
    for r in results:
        if r['category'] == 'DATA_QUALITY': # or check code if implemented
            print(f"PASS: Detected Range Violation. Context: {r['ui_text']['description']}")
            found = True
    if not found: 
        print("FAIL: Did not detect range violation.")
        print(results)

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_financial_gap(mock_get_ranges):
    print("\n--- Test 1: Financial Gap ---")
    df = create_mock_df()
    df['strokes_per_minute'] = 8.0 
    # Last day: High SPM (should produce 100), but Actual Oil is 50.
    df.loc[df.index[-1], 'oil_volume'] = 50 
    
    results = detect_anomalies("Test_Gap", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'FINANCIAL_EFFICIENCY':
            print(f"PASS: Detected Gap. Impact: {r['impact_metrics']['value']} {r['impact_metrics']['unit']}")
            found = True
    if not found: 
        print("FAIL: Did not detect gap.")
        print(results)

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_ghost_production(mock_get_ranges):
    print("\n--- Test 2: Ghost Production ---")
    df = create_mock_df()
    # Active
    df['strokes_per_minute'] = 8.0
    # Zero Production
    df.iloc[-1, df.columns.get_loc('oil_volume')] = 0
    df.iloc[-1, df.columns.get_loc('water_volume')] = 0
    df.iloc[-1, df.columns.get_loc('gas_volume')] = 0
    
    results = detect_anomalies("Test_Ghost", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'GHOST_PROD':
            print(f"PASS: Detected Ghost Production. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect ghost production.")

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_cost_creep(mock_get_ranges):
    print("\n--- Test 3: Cost Creep ---")
    df = create_mock_df()
    # Baseline Water 50.
    # Last day Water 100 (> 20% increase over avg ~50).
    df.iloc[-1, df.columns.get_loc('water_volume')] = 100
    # Oil flat (100).
    
    results = detect_anomalies("Test_Creep", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'COST_CREEP':
            print(f"PASS: Detected Cost Creep. Impact: {r['impact_metrics']['value']} {r['impact_metrics']['unit']}")
            found = True
    if not found: print("FAIL: Did not detect cost creep.")

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_gas_interference(mock_get_ranges):
    print("\n--- Test 4: Gas Interference ---")
    df = create_mock_df()
    # Gas Trending Up
    df.iloc[-1, df.columns.get_loc('gas_volume')] = 20
    
    # Erratic Fillage (High Std Dev)
    for i in range(7):
        val = 90 if i % 2 == 0 else 10
        df.iloc[-1-i, df.columns.get_loc('pump_fillage')] = val
        
    results = detect_anomalies("Test_Gas", df)
    
    found = False
    found = False
    for r in results:
        if r['anomaly_code'] == 'GAS_INTERFERENCE':
            print(f"PASS: Detected Gas Interference. Context: {r['ui_text']['description']}")
            found = True
    if not found: 
        print("FAIL: Did not detect gas interference. Results found:")
        print(results)

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_flowline_blockage(mock_get_ranges):
    print("\n--- Test 5: Flowline Blockage ---")
    df = create_mock_df()
    # Tubing Pressure Up (> 1.2 * Avg)
    df.iloc[-1, df.columns.get_loc('tubing_pressure')] = 150
    # Oil Down (< 0.9 * Avg)
    df.iloc[-1, df.columns.get_loc('oil_volume')] = 80
    
    results = detect_anomalies("Test_Blockage", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'FLOWLINE_BLOCK':
            print(f"PASS: Detected Flowline Blockage. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect blockage.")

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_pump_wear(mock_get_ranges):
    print("\n--- Test 6: Pump Wear ---")
    df = create_mock_df(lift_type='Rod Pump')
    # High SPM (> Avg)
    df.iloc[-1, df.columns.get_loc('strokes_per_minute')] = 10
    # High Fillage (> 80)
    df.iloc[-1, df.columns.get_loc('pump_fillage')] = 90
    # Low Oil (< 0.8 * Predicted)
    df.iloc[-1, df.columns.get_loc('oil_volume')] = 50
    
    results = detect_anomalies("Test_Wear", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'PUMP_WEAR':
            print(f"PASS: Detected Pump Wear. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect pump wear.")

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_esp_broken_shaft(mock_get_ranges):
    print("\n--- Test 7: ESP Broken Shaft ---")
    df = create_mock_df(lift_type='ESP')
    # High PIP (> 1.1 * Avg)
    df.iloc[-1, df.columns.get_loc('pump_intake_pressure')] = 250
    # Low Amps (< 0.5 * Avg)
    df.iloc[-1, df.columns.get_loc('motor_current')] = 5
    
    results = detect_anomalies("Test_ESP", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'ESP_SHAFT_BREAK':
            print(f"PASS: Detected ESP Shaft Break. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect ESP failure.")

@patch('anomaly_review_service.get_well_status', return_value='ACTIVE') # Mock Master Status
@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_unplanned_downtime(mock_get_ranges, mock_status):
    print("\n--- Test 8: Unplanned Downtime ---")
    df = create_mock_df()
    # Inactive Sensors
    df.iloc[-1, df.columns.get_loc('strokes_per_minute')] = 0
    df.iloc[-1, df.columns.get_loc('motor_current')] = 0
    
    # Logic requires master_status=ACTIVE (mocked) and is_active=False
    results = detect_anomalies("Test_Downtime", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'UNPLANNED_DOWNTIME':
            print(f"PASS: Detected Unplanned Downtime. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect downtime.")

@patch('anomaly_review_service.get_well_status', return_value='ACTIVE') # Mock Master Status
@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_sensor_integrity(mock_get_ranges, mock_status):
    print("\n--- Test 9: Data Integrity (Local Isolation) ---")
    df = create_mock_df()
    
    # Violation: Tank Oil Level > Max (30)
    # We need to manually inject this into the DF or Mock Row because create_mock_df doesn't have it by default
    # But detect_anomalies reads from valid columns.
    # We must add the column to the DF.
    df['tank_oil_level'] = 10.0 # Normal
    df.iloc[-1, df.columns.get_loc('tank_oil_level')] = 35.0 # Violation (Max 30)
    
    results = detect_anomalies("Test_Integrity", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'SENSOR_RANGE':
            print(f"PASS: Detected Sensor Integrity Violation. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect integrity violation.")

@patch('anomaly_review_service.generate_narrative')
def verify_real_well(mock_generate, well_id):
    # Mock return for AI Generation
    mock_generate.return_value = {
         "description": "Static Description (OpenAI Disabled for Test)",
         "why": "Testing Logic Engine w/o AI",
         "root_cause": "Static Fallback"
    }

    print(f"\n=== Verifying REAL WELL: {well_id} ===")
    
    # 1. Fetch Real Data
    print(f"Fetching data from Snowflake for {well_id}...")
    try:
        df = fetch_well_data(well_id, days=120)
        print(f"Fetched {len(df)} rows.")
        if df.empty:
            print("❌ No data found.")
            return

        print(f"Columns: {list(df.columns)}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # 2. Run Detection
    print("\nRunning Logic Engine...")
    start_time = datetime.now()
    results = detect_anomalies(well_id, df)
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"Completed in {duration:.2f} seconds.")
    print(f"Found {len(results)} anomalies.")
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Type:     {res['title']}")
        print(f"Severity: {res['severity']}")
        print(f"Context:  {res['ui_text']['description']}")
        print(f"Impact:   {res['ui_text']['economic_impact']}")
        print(f"Driver:   {res['ui_text']['suspected_root_cause']}")


# Persistence Test
def test_persistence():
    print("\n=== Testing Persistence (Postgres & Snowflake) ===")
    from anomaly_review_service import save_reviews
    
    # 1. Create Dummy Review
    dummy_review = {
        'well_id': 'TEST_PERSISTENCE_User_1',
        'timestamp': datetime.now(),
        'category': 'TEST',
        'severity': 'Info',
        'title': 'Persistence Check ' + datetime.now().strftime("%H:%M:%S"),
        'ui_text': {
            'description': 'This is a test record to verify DB storage.',
            'why_is_this_an_anomaly': 'Manual Test',
            'suspected_root_cause': 'User Verification',
            'economic_impact': 'None'
        }
    }
    
    print(f"Attempting to save review: {dummy_review['title']}")
    
    try:
        # Call the service function
        save_reviews([dummy_review])
        print("✅ save_reviews() function called successfully.")
    except Exception as e:
        print(f"❌ save_reviews() failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Verify Postgres
    print("\nVerifying Postgres...")
    try:
        import os, psycopg2
        db_url = os.getenv("DATABASE_URL")
        # Fix schema param if present
        if db_url and "schema=" in db_url:
            db_url = db_url.replace("?schema=public", "").replace("&schema=public", "")
            
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, title, content, description FROM anomaly_reviews WHERE well_id = 'TEST_PERSISTENCE_User_1' ORDER BY created_at DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    try:
                        content_prev = str(row[2])[:50] if row[2] else "None"
                    except:
                        content_prev = "Error printing content"
                        
                    print(f"✅ Postgres Found Record: ID={row[0]}, Title='{row[1]}'")
                    print(f"   Content: {content_prev}...")
                    print(f"   Description: {row[3]}")
                else:
                    print("❌ Postgres: Record NOT found.")
    except Exception as e:
        print(f"❌ Postgres Verification failed: {e}")

    # 3. Verify Snowflake
    print("\nVerifying Snowflake...")
    try:
        from anomaly_review_service import get_snowflake_conn
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT title, content, description FROM anomaly_reviews WHERE well_id = 'TEST_PERSISTENCE_User_1' ORDER BY created_at DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                print(f"✅ Snowflake Found Record: Title='{row[0]}'")
                print(f"   Content (Variant): {row[1]}")
                print(f"   Description: {row[2]}")
            else:
                print("❌ Snowflake: Record NOT found.")
    except Exception as e:
        print(f"❌ Snowflake Verification failed: {e}")




@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_esp_broken_shaft(mock_get_ranges):
    print("\n--- Test 7: ESP Broken Shaft ---")
    df = create_mock_df(lift_type='ESP')
    # High PIP (> 1.1 * Avg)
    df.iloc[-1, df.columns.get_loc('pump_intake_pressure')] = 250
    # Low Amps (< 0.5 * Avg)
    df.iloc[-1, df.columns.get_loc('motor_current')] = 5
    
    results = detect_anomalies("Test_ESP", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'ESP_SHAFT_BREAK':
            print(f"PASS: Detected ESP Shaft Break. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect ESP failure.")



@patch('anomaly_review_service.get_well_status', return_value='ACTIVE') # Mock Master Status
@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_sensor_integrity(mock_get_ranges, mock_status):
    print("\n--- Test 9: Data Integrity (Local Isolation) ---")
    df = create_mock_df()
    
    # Violation: Tank Oil Level > Max (30)
    # We need to manually inject this into the DF or Mock Row because create_mock_df doesn't have it by default
    # But detect_anomalies reads from valid columns.
    # We must add the column to the DF.
    df['tank_oil_level'] = 10.0 # Normal
    df.iloc[-1, df.columns.get_loc('tank_oil_level')] = 35.0 # Violation (Max 30)
    
    results = detect_anomalies("Test_Integrity", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'SENSOR_RANGE':
            print(f"PASS: Detected Sensor Integrity Violation. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect integrity violation.")

    if not found: print("FAIL: Did not detect integrity violation.")

@patch('anomaly_review_service.get_well_status', return_value='ACTIVE')
@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_tank_drop(mock_get_ranges, mock_status):
    print("\n--- Test 10: Unexpected Tank Drop ---")
    df = create_mock_df()
    
    # Simulate Tank Level Drop
    df['tank_oil_level'] = 20.0 # Stable
    df.iloc[-1, df.columns.get_loc('tank_oil_level')] = 15.0 # Drop of 5.0
    
    results = detect_anomalies("Test_TankDrop", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'TANK_DROP':
            print(f"PASS: Detected Tank Drop. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect tank drop.")

@patch('anomaly_review_service.get_well_status', return_value='ACTIVE')
@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_bsw_spike(mock_get_ranges, mock_status):
    print("\n--- Test 11: BSW Spike ---")
    df = create_mock_df()
    
    # Simulate BSW (Water Cut) Spike
    # Normal: Oil 100, Water 50 -> BSW ~33%
    # Spike: Oil 20, Water 130 -> Vol 150 (Same), BSW ~86%
    
    # Set Baseline (Days 0-29)
    # create_mock_df defaults are fine (Oil 100, Water 50)
    
    # Set Spike on Last Day
    last_idx = df.index[-1]
    df.at[last_idx, 'oil_volume'] = 20.0
    df.at[last_idx, 'water_volume'] = 130.0
    
    results = detect_anomalies("Test_BSW", df)
    
    found = False
    for r in results:
        if r['anomaly_code'] == 'BSW_SPIKE':
            print(f"PASS: Detected BSW Spike. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect BSW spike.")

def verify_real_well(well_id):
    print(f"\n=== Verifying REAL WELL: {well_id} ===")
    
    # 1. Fetch Real Data
    print(f"Fetching data from Snowflake for {well_id}...")
    try:
        df = fetch_well_data(well_id, days=120)
        print(f"Fetched {len(df)} rows.")
        if df.empty:
            print("❌ No data found.")
            return

        print(f"Columns: {list(df.columns)}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # 2. Run Detection
    print("\nRunning Logic Engine...")
    start_time = datetime.now()
    results = detect_anomalies(well_id, df)
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"Completed in {duration:.2f} seconds.")
    print(f"Found {len(results)} anomalies.")
    
    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Type:     {res['title']}")
        print(f"Severity: {res['severity']}")
        print(f"Context:  {res['ui_text']['description']}")
        print(f"Impact:   {res['ui_text']['economic_impact']}")
        print(f"Driver:   {res['ui_text']['suspected_root_cause']}")


# Persistence Test
def test_persistence():
    print("\n=== Testing Persistence (Postgres & Snowflake) ===")
    from anomaly_review_service import save_reviews
    
    # 1. Create Dummy Review
    dummy_review = {
        'well_id': 'TEST_PERSISTENCE_User_1',
        'event_date': datetime.now().strftime('%Y-%m-%d'),
        'detected_at': datetime.now().isoformat(),
        'anomaly_code': 'TEST_PERSISTENCE',
        'category': 'OPERATIONAL',
        'severity': 'Info',
        'title': 'Persistence Check ' + datetime.now().strftime("%H:%M:%S"),
        'status': 'NEW',
        'ui_text': {
            'description': 'This is a test record to verify DB storage.',
            'why_is_this_an_anomaly': 'Manual Test',
            'suspected_root_cause': 'User Verification',
            'economic_impact': 'None'
        },
        'impact_metrics': {
            'value': 0.0,
            'unit': 'None'
        },
        'chart_data': {}
    }
    
    print(f"Attempting to save review: {dummy_review['title']}")
    
    try:
        # Call the service function
        save_reviews([dummy_review])
        print("✅ save_reviews() function called successfully.")
    except Exception as e:
        print(f"❌ save_reviews() failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Verify Postgres
    print("\nVerifying Postgres...")
    try:
        import os, psycopg2
        db_url = os.getenv("DATABASE_URL")
        # Fix schema param if present
        if db_url and "schema=" in db_url:
            db_url = db_url.replace("?schema=public", "").replace("&schema=public", "")
            
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT anomaly_code, title, ui_text, status FROM operation_suggestion WHERE well_id = 'TEST_PERSISTENCE_User_1' ORDER BY detected_at DESC LIMIT 1")
                row = cur.fetchone()
                if row:     
                    print(f"✅ Postgres Found Record: Code='{row[0]}', Title='{row[1]}'")
                    print(f"   Status: {row[3]}")
                else:
                    print("❌ Postgres: Record NOT found.")
    except Exception as e:
        print(f"❌ Postgres Verification failed: {e}")

    # 3. Verify Snowflake
    print("\nVerifying Snowflake...")
    try:
        from anomaly_review_service import get_snowflake_conn
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT title, ui_text, status FROM operation_suggestion WHERE well_id = 'TEST_PERSISTENCE_User_1' ORDER BY detected_at DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                print(f"✅ Snowflake Found Record: Title='{row[0]}'")
                print(f"   UI Text (Variant): {row[1]}")
            else:
                print("❌ Snowflake: Record NOT found.")
    except Exception as e:
        print(f"❌ Snowflake Verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Anomaly Logic Engine")
    parser.add_argument("--real", help="Run on REAL data for specific WELL_ID", type=str)
    parser.add_argument("--persistence", help="Run DB Persistence Check", action="store_true") # Added flag
    parser.add_argument("--mock", help="Run MOCK unit tests (Default)", action="store_true")
    
    args = parser.parse_args()
    
    if args.persistence:
        test_persistence()
    elif args.real:
        verify_real_well(args.real)
    else:
        # Default to Mock Tests
        print("Running MOCK Tests (Pass --real <WELL_ID> to test real data, --persistence to test DB)")
        # test_range_check() # Deprecated in this service
        test_financial_gap()
        test_ghost_production()
        test_esp_broken_shaft()
        test_esp_broken_shaft()
 
        test_sensor_integrity()
        test_sensor_integrity()
        test_tank_drop()
        test_bsw_spike()
        test_flowline_blockage()
        test_pump_wear()
        test_esp_broken_shaft()
