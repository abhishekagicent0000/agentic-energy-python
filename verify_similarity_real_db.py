
import os
import pandas as pd
import json
import logging
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import app
    # Ensure app's Snowflake config is loaded
    if not app.SNOWFLAKE_CONFIG['user']:
        logger.error("Snowflake config missing in app.py. Check .env")
        exit(1)
except ImportError as e:
    logger.error(f"Failed to import app: {e}")
    exit(1)

def test_similarity_with_real_db():
    print("--- Fetching Real Historical Data from Snowflake ---")
    
    # 1. Fetch real history
    # We'll fetch a small batch of recent anomalies
    try:
        df = app.get_historical_anomalies(limit=20)
        print(f"Fetched {len(df)} historical records.")
    except Exception as e:
        print(f"Error fetching history: {e}")
        return

    if df.empty:
        print("No historical data found in DB to test against.")
        return

    # 2. Pick a target record from history to simulate a "match"
    # We will try to match the first record found.
    try:
        target_row = df.iloc[0]
        print(f"\nTargeting Record from {target_row['timestamp']} (Well: {target_row['well_id']})")
    except:
         print("DataFrame processing error")
         return
    
    # Parse its values
    try:
        raw_values = target_row['raw_values']
        if isinstance(raw_values, str):
            raw_values = json.loads(raw_values)
    except:
        print("Could not parse raw_values of target record.")
        return

    # 3. Construct a "Current Anomaly" that matches this target
    # We need to know what violations it currently has.
    # We'll use the check_anomaly function (re-evaluation logic) to see what it violates.
    print("Re-evaluating target record to find its active violations...")
    detection_result = app.check_anomaly(raw_values, well_id=target_row['well_id'], lift_type=target_row.get('lift_type') or 'Rod Pump') # Defaulting to RP if missing
    
    current_violations = detection_result.get('violations', [])
    if not current_violations:
        print("Target record has NO violations under current rules. Cannot test similarity matching (needs violations).")
        print("Trying next record...")
        return

    print(f"Target has {len(current_violations)} violations: {[v['field'] for v in current_violations]}")
    
    # 4. Test Case 1: Same Well, Exact Match (Should work)
    print("\n[Test 1] Same Well, Exact Match")
    fake_readings = raw_values.copy()
    # We pass the EXACT same violations
    fake_violations = current_violations 
    
    matches = app.find_similar_anomalies(
        current_readings=fake_readings,
        current_violations=fake_violations,
        historical_df=df,
        current_well_id=target_row['well_id'],
        lift_type='Rod Pump' 
    )
    
    print(f"Found {len(matches)} matches.")
    if len(matches) > 0:
        print("✓ Success! Found the record itself (or duplicate).")
        print(f"   Match Score: {matches[0]['similarity_score']}")
        print(f"   Violations Shown: {[v['field'] for v in matches[0]['raw_anomaly_data']['violations']]}")
    else:
        print("✗ Failed to find the target record.")

    # 5. Test Case 2: Different Well, Exact Match (Should work)
    print("\n[Test 2] Different Well, Exact Match")
    diff_well_matches = app.find_similar_anomalies(
        current_readings=fake_readings,
        current_violations=fake_violations,
        historical_df=df,
        current_well_id="DIFFERENT_WELL_ID_999",
        lift_type='Rod Pump'
    )
    
    print(f"Found {len(diff_well_matches)} matches.")
    found_target = any(m['well_id'] == target_row['well_id'] for m in diff_well_matches)
    if found_target:
        print("✓ Success! Found the target record even when searching as a different well (Strict Match Success).")
    else:
        print("✗ Failed to find target record for different well.")

    # 6. Test Case 3: Different Well, Partial Match (Should FAIL)
    print("\n[Test 3] Different Well, Partial Match (Should FAIL)")
    if len(current_violations) > 0:
        # Let's verify strictness by removing a violation or adding a fake one
        # If we have 1 violation, strict matching requires ONLY that 1.
        # If we query with 2 violations, it shouldn't match.
        
        modified_violations = current_violations + [{'field': 'FAKE_SENSOR', 'value': 999, 'violation': 'fake'}]
        
        fail_matches = app.find_similar_anomalies(
            current_readings=fake_readings,
            current_violations=modified_violations,
            historical_df=df,
            current_well_id="DIFFERENT_WELL_ID_999",
            lift_type='Rod Pump'
        )
        
        found_target_fail = any(m['well_id'] == target_row['well_id'] for m in fail_matches)
        if not found_target_fail:
             print("✓ Success! Did NOT find target record (Strict Match Correctly Excluded).")
        else:
             print("✗ Failed! Found target record but should have excluded it.")

if __name__ == "__main__":
    test_similarity_with_real_db()
