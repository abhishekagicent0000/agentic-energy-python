
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
    for r in results:
        if r['category'] == 'DATA_QUALITY':
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
        if r['title'] == 'Production Efficiency Gap':
            print(f"PASS: Detected Gap. Impact: {r['ui_text']['economic_impact']}")
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
        if r['title'] == 'Missing Production Report':
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
        if r['title'] == 'Rising Disposal Costs':
            print(f"PASS: Detected Cost Creep. Impact: {r['ui_text']['economic_impact']}")
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
    for r in results:
        if r['title'] == 'Gas Interference Review':
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
    # Avg is 100. Rise to 150.
    df.iloc[-1, df.columns.get_loc('tubing_pressure')] = 150
    # Oil Down (< 0.9 * Avg)
    # Avg is 100. Drop to 80.
    df.iloc[-1, df.columns.get_loc('oil_volume')] = 80
    
    results = detect_anomalies("Test_Blockage", df)
    
    found = False
    for r in results:
        if r['title'] == 'Flowline Blockage':
            print(f"PASS: Detected Flowline Blockage. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect blockage.")

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_pump_wear(mock_get_ranges):
    print("\n--- Test 6: Pump Wear ---")
    df = create_mock_df(lift_type='Rod Pump')
    # High SPM (> Avg)
    # Avg 8. Set to 10.
    df.iloc[-1, df.columns.get_loc('strokes_per_minute')] = 10
    # High Fillage (> 80)
    df.iloc[-1, df.columns.get_loc('pump_fillage')] = 90
    # Low Oil (< 0.8 * Predicted)
    # Predicts 100 (based on correlation), Actual 50.
    df.iloc[-1, df.columns.get_loc('oil_volume')] = 50
    
    results = detect_anomalies("Test_Wear", df)
    
    found = False
    for r in results:
        if r['title'] == 'Pump Wear / Slippage':
            print(f"PASS: Detected Pump Wear. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect pump wear.")

@patch('anomaly_review_service.get_sensor_ranges', return_value=MOCK_RANGES)
def test_esp_broken_shaft(mock_get_ranges):
    print("\n--- Test 7: ESP Broken Shaft ---")
    df = create_mock_df(lift_type='ESP')
    # High PIP (> 1.1 * Avg)
    # Avg 200. Set to 250.
    df.iloc[-1, df.columns.get_loc('pump_intake_pressure')] = 250
    # Low Amps (< 0.5 * Avg)
    # Avg 12. Set to 5.
    df.iloc[-1, df.columns.get_loc('motor_current')] = 5
    
    # Needs valid ESP features to pass init checks
    # create_mock_df handles standard columns, but ESP needs 'motor_current' (which is there).
    
    results = detect_anomalies("Test_ESP", df)
    
    found = False
    for r in results:
        if r['title'] == 'Broken Shaft / Free Spin':
            print(f"PASS: Detected ESP Shaft Break. Context: {r['ui_text']['description']}")
            found = True
    if not found: print("FAIL: Did not detect ESP failure.")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Anomaly Logic Engine")
    parser.add_argument("--real", help="Run on REAL data for specific WELL_ID", type=str)
    parser.add_argument("--mock", help="Run MOCK unit tests (Default)", action="store_true")
    
    args = parser.parse_args()
    
    if args.real:
        verify_real_well(args.real)
    else:
        # Default to Mock Tests
        print("Running MOCK Tests (Pass --real <WELL_ID> to test real data)")
        test_range_check()
        test_financial_gap()
        test_ghost_production()
        test_cost_creep()
        test_gas_interference()
        test_flowline_blockage()
        test_pump_wear()
        test_esp_broken_shaft()
