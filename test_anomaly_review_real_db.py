
import os
import logging
import json
from dotenv import load_dotenv
import pandas as pd
from anomaly_review_service import detect_anomalies, get_snowflake_conn

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_active_wells(limit=10):
    """Fetch a list of active well IDs from Snowflake."""
    query = """
    SELECT DISTINCT well_id 
    FROM well_sensor_readings 
    WHERE timestamp >= DATEADD(day, -7, CURRENT_DATE())
    LIMIT %s
    """
    try:
        with get_snowflake_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"Failed to fetch active wells: {e}")
        return []

def test_anomaly_review():
    print("--- Starting Anomaly Review Service Integration Test ---")
    
    # 1. Fetch Wells
    print("1. Fetching active wells...")
    wells = fetch_active_wells(limit=5) # Start with 5 for testing
    if not wells:
        print("No active wells found or connection failed.")
        return

    print(f"Found {len(wells)} wells: {wells}")
    
    total_anomalies = 0
    results_summary = {}

    # 2. Run Detection
    print("\n2. Running Detection for each well...")
    for well_id in wells:
        print(f"\nProcessing Well: {well_id}")
        try:
            # We let the service fetch its own data
            anomalies = detect_anomalies(well_id, lookback_days=30)
            
            count = len(anomalies)
            total_anomalies += count
            results_summary[well_id] = [a['title'] for a in anomalies]
            
            if count > 0:
                print(f"  -> Found {count} anomalies:")
                for a in anomalies:
                    print(f"     - [{a['severity']}] {a['title']} ({a['category']})")
                    print(f"       Context: {a['ui_text']['description'][:100]}...")
            else:
                print("  -> No anomalies detected.")
                
        except Exception as e:
            print(f"  -> Error processing well {well_id}: {e}")
            results_summary[well_id] = f"ERROR: {e}"

    # 3. Summary
    print("\n--- Test Summary ---")
    print(f"Total Wells Processed: {len(wells)}")
    print(f"Total Anomalies Found: {total_anomalies}")
    print("\nDetails:")
    for well, result in results_summary.items():
        if isinstance(result, list):
            if result:
                 print(f"  {well}: {len(result)} anomalies ({', '.join(result)})")
            else:
                 print(f"  {well}: 0 anomalies")
        else:
            print(f"  {well}: {result}")

if __name__ == "__main__":
    test_anomaly_review()
