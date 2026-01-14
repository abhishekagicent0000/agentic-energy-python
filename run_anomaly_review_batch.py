
import os
import sys
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from snowflake.connector import connect as snowflake_connect

# Import the service logic
from anomaly_review_service import detect_anomalies

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_execution.log")
    ]
)
logger = logging.getLogger("BatchRunner")

# Load Env
load_dotenv()

def get_snowflake_conn():
    return snowflake_connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )

def get_all_active_wells():
    """Fetches list of all distinct well IDs that have data in the last 30 days."""
    try:
        conn = get_snowflake_conn()
        cursor = conn.cursor()
        
        # We only care about wells that are actually reporting recently
        query = """
        SELECT DISTINCT well_id 
        FROM well_sensor_readings 
        WHERE timestamp >= DATEADD(day, -30, CURRENT_DATE())
        ORDER BY well_id
        """
        cursor.execute(query)
        wells = [row[0] for row in cursor.fetchall()]
        conn.close()
        return wells
    except Exception as e:
        logger.error(f"Failed to fetch well list: {e}")
        return []

def run_batch():
    logger.info("="*50)
    logger.info("STARTING ANOMALY REVIEW BATCH")
    logger.info("="*50)
    
    start_time = time.time()
    
    # 1. Get List of Wells
    wells = get_all_active_wells()
    if not wells:
        logger.error("No active wells found. Exiting.")
        return
        
    logger.info(f"Found {len(wells)} active wells to process.")
    
    stats = {
        "processed": 0,
        "failed": 0,
        "anomalies_found": 0,
        "start_time": datetime.now().isoformat()
    }
    
    # 2. Iterate
    for i, well_id in enumerate(wells):
        try:
            # Progress Log
            sys.stdout.write(f"\rProcessing {i+1}/{len(wells)}: {well_id}...")
            sys.stdout.flush()
            
            # --- EXECUTE LOGIC ENGINE ---
            results = detect_anomalies(well_id)
            # ----------------------------
            
            stats["anomalies_found"] += len(results)
            
            # Optional: If results found, maybe log them briefly
            if results:
                logger.info(f"\n[!] Anomaly in {well_id}: {[r['title'] for r in results]}")
                # We assume detect_anomalies internally might save to DB if configured, 
                # or we handle it here. 
                # For this implementation, detect_anomalies returns the list.
                # In a real pipeline, we would INSERT into `well_anomalies`.
                insert_anomalies_to_db(results)
                
            stats["processed"] += 1
            
        except Exception as e:
            logger.error(f"\nFailed to process {well_id}: {e}")
            stats["failed"] += 1

    total_time = time.time() - start_time
    logger.info(f"\n\n{'='*50}")
    logger.info("BATCH COMPLETE")
    logger.info(f"Time Taken: {total_time:.2f}s ({total_time/len(wells):.2f}s/well)")
    logger.info(f"Wells Processed: {stats['processed']}")
    logger.info(f"Failures: {stats['failed']}")
    logger.info(f"Total Anomalies Detected: {stats['anomalies_found']}")
    logger.info("="*50)

def insert_anomalies_to_db(results):
    """
    Helper to persist results to Snowflake 'well_anomalies' table.
    """
    if not results: return
    
    try:
        conn = get_snowflake_conn()
        cursor = conn.cursor()
        
        insert_data = []
        for r in results:
            # Map result dict to DB schema columns
            # Schema: well_id, timestamp, anomaly_type, anomaly_score, raw_values, model_name, status, severity, category
            # Note: detect_anomalies output structure might differ slightly, adapted here.
            
            # r has: well_id, timestamp, category, severity, title, ui_text, ...
            
            # Assuming 'anomaly_score' isn't explicitly in r, use 1.0 or derive
            # stored raw_values ?? -> r doesn't have raw values of all sensors, maybe skip or json dump the context
            
            insert_data.append((
                r['well_id'],
                r['timestamp'],
                r['title'],     # anomaly_type
                1.0,            # score (logic engine is binary trigger, implicitly high confidence)
                json.dumps(r['ui_text']), # raw_values (using narrative as context payload)
                'Logic_Engine_v2', # model_name
                'New',          # status
                r['severity'],
                r['category']
            ))
            
        if insert_data:
            cursor.executemany("""
                INSERT INTO well_anomalies 
                (well_id, timestamp, anomaly_type, anomaly_score, raw_values, model_name, status, severity, category)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, insert_data)
            
        conn.close()
    except Exception as e:
        logger.error(f"DB Insert Failed: {e}")
        import json # valid fallback

if __name__ == "__main__":
    run_batch()
