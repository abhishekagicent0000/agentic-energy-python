import os
import sys
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from snowflake.connector import connect as snowflake_connect

# Import the service logic
from anomaly_review_service import detect_anomalies

# =====================================================
# CONFIG
# =====================================================
TEST_MODE = "true"
TEST_WELL_LIMIT = 5
MAX_WORKERS = 10
# =====================================================

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_execution.log"),
    ],
)

logger = logging.getLogger("BatchRunner")

load_dotenv()


# =====================================================
# SNOWFLAKE CONNECTION
# =====================================================
def get_snowflake_conn():
    return snowflake_connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


# =====================================================
# FETCH WELLS
# =====================================================
def get_all_active_wells():
    """Fetch list of wells with data in last 7 days."""
    try:
        conn = get_snowflake_conn()
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT well_id
            FROM well_sensor_readings
            WHERE timestamp >= DATEADD(day, -7, CURRENT_DATE())
            ORDER BY well_id
        """

        cursor.execute(query)
        wells = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return wells

    except Exception as e:
        logger.error(f"Failed to fetch well list: {e}")
        return []


# =====================================================
# PROCESS SINGLE WELL
# =====================================================
def process_single_well(well_id):
    """Wrapper function for parallel execution."""
    try:
        results = detect_anomalies(well_id)

        return {
            "well_id": well_id,
            "count": len(results),
            "titles": [r.get("title") for r in results],
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Error processing {well_id}: {e}")
        return {
            "well_id": well_id,
            "count": 0,
            "status": "failed",
            "error": str(e),
        }


# =====================================================
# RUN BATCH
# =====================================================
def run_batch():
    logger.info("=" * 60)
    logger.info("STARTING ANOMALY REVIEW BATCH (PARALLEL)")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Fetch wells
    wells = get_all_active_wells()

    if not wells:
        logger.error("No active wells found. Exiting.")
        return

    # -------------------------------------------------
    # TEST MODE LOGIC
    # -------------------------------------------------
    if TEST_MODE:
        logger.warning("⚠️  TEST MODE ENABLED")
        logger.warning(f"Processing only first {TEST_WELL_LIMIT} wells")
        wells = wells[:TEST_WELL_LIMIT]

    logger.info(f"Total wells to process: {len(wells)}")

    logger.info("-" * 30)
    logger.info("Target Wells:")
    for w in wells:
        logger.info(f" -> {w}")
    logger.info("-" * 30)

    stats = {
        "processed": 0,
        "failed": 0,
        "anomalies_found": 0,
    }

    # 2. Parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_well = {
            executor.submit(process_single_well, well): well
            for well in wells
        }

        for i, future in enumerate(as_completed(future_to_well)):
            result = future.result()

            sys.stdout.write(f"\rProgress: {i + 1}/{len(wells)}")
            sys.stdout.flush()

            if result["status"] == "success":
                stats["processed"] += 1
                stats["anomalies_found"] += result["count"]

                if result["count"] > 0:
                    sys.stdout.write("\n")
                    logger.info(
                        f"[+] {result['well_id']}: "
                        f"{result['count']} anomalies -> {result['titles']}"
                    )
            else:
                stats["failed"] += 1
                sys.stdout.write("\n")
                logger.error(
                    f"[-] {result['well_id']} failed: {result.get('error')}"
                )

    total_time = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"Execution Time     : {total_time:.2f}s")
    logger.info(f"Wells Processed   : {stats['processed']}")
    logger.info(f"Wells Failed      : {stats['failed']}")
    logger.info(f"Anomalies Saved   : {stats['anomalies_found']}")
    logger.info("=" * 60)


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    run_batch()
