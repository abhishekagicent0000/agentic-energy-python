import os
import uuid
import random
from datetime import datetime
import sys
import time
from dotenv import load_dotenv
from snowflake.connector import connect as snowflake_connect
from operation import get_pg_connection
from psycopg2.extras import execute_values

load_dotenv()

# Snowflake connection settings (from env)
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}


def compute_boe(oil_bbl, gas_mcf):
    """Compute BOE = oil (bbl) + gas (mcf) / 6."""
    try:
        return float(oil_bbl) + float(gas_mcf) / 6.0
    except Exception:
        return None


def ensure_table(conn):
    cur = conn.cursor()
    create_sql = """
    CREATE TABLE IF NOT EXISTS production_forecasting (
        id VARCHAR(50) PRIMARY KEY,
        uid VARCHAR(20) UNIQUE,
        well_id VARCHAR(100),
        boe FLOAT,
        oil FLOAT,
        gas FLOAT,
        water FLOAT,
        timestamp TIMESTAMP,
        lift_type VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cur.execute(create_sql)
    # If uid column existed as INTEGER, convert it to text to allow 'alt-######' values
    try:
        cur.execute("ALTER TABLE production_forecasting ALTER COLUMN uid TYPE VARCHAR(20) USING uid::text;")
    except Exception:
        pass
    conn.commit()
    cur.close()


def generate_unique_uid(existing_uids):
    # Generate UID strings like 'alt-123456' (6 digit numeric part)
    for _ in range(10000):
        n = random.randint(0, 999999)
        val = f"alt-{n:06d}"
        if val not in existing_uids:
            existing_uids.add(val)
            return val
    # Fallback: compute max numeric suffix and increment
    maxn = 0
    for u in existing_uids:
        try:
            if isinstance(u, str) and u.startswith('alt-') and u[4:].isdigit():
                maxn = max(maxn, int(u[4:]))
        except Exception:
            continue
    candidate = maxn + 1
    val = f"alt-{candidate:06d}"
    while val in existing_uids:
        candidate += 1
        val = f"alt-{candidate:06d}"
    existing_uids.add(val)
    return val


def fetch_from_snowflake():
    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cur = conn.cursor()
    query = (
        "SELECT well_id, date, oil_volume, gas_volume, water_volume, lift_type "
        "FROM well_daily_production"
    )
    cur.execute(query)
    rows = cur.fetchall()
    # cursor.description for column order
    cols = [c[0].lower() for c in cur.description]
    cur.close()
    conn.close()
    return cols, rows


def insert_into_postgres(rows):
    pg = get_pg_connection()
    if not pg:
        raise RuntimeError('Postgres connection unavailable')

    ensure_table(pg)

    cur = pg.cursor()
    # load existing uids (as text) to avoid collisions
    try:
        cur.execute("SELECT uid::text FROM production_forecasting WHERE uid IS NOT NULL")
        existing = {r[0] for r in cur.fetchall()}
    except Exception:
        existing = set()

    insert_sql = (
        "INSERT INTO production_forecasting (id, uid, well_id, boe, oil, gas, water, timestamp, lift_type) "
        "VALUES %s"
    )

    # reduce batch size for more frequent commits and better visibility
    batch_size = 2000
    total = len(rows)
    inserted = 0
    batch = []
    batch_no = 0
    batch_times = []
    processed = 0

    def flush_batch(b):
        nonlocal inserted, batch_no
        if not b:
            return
        batch_no += 1
        start = inserted + 1
        end = inserted + len(b)
        print(f"  Inserting batch {batch_no}: rows {start:,} - {end:,} (count {len(b):,})")
        sys.stdout.flush()
        t0 = time.time()
        try:
            print(f"    → executing execute_values for {len(b):,} rows")
            sys.stdout.flush()
            execute_values(cur, insert_sql, b, page_size=1000)
            pg.commit()
        except Exception as e:
            pg.rollback()
            print(f"  ERROR inserting batch {batch_no}: {e}")
            raise
        dt = time.time() - t0
        batch_times.append(dt)
        inserted += len(b)
        # ETA calculation
        avg = sum(batch_times) / len(batch_times)
        remaining = total - inserted
        est_batches = (remaining / len(b)) if len(b) else 0
        eta_sec = est_batches * avg
        eta = f"{int(eta_sec//60)}m{int(eta_sec%60)}s" if eta_sec >= 60 else f"{int(eta_sec)}s"
        pct = inserted / total * 100 if total else 100
        print(f"  → Batch {batch_no} committed in {dt:.2f}s; total inserted: {inserted:,} ({pct:.1f}%), ETA: {eta}")
        sys.stdout.flush()

    try:
        for r in rows:
            # Expecting order: well_id, date, oil_volume, gas_volume, water_volume, lift_type
            well_id, date_val, oil_v, gas_v, water_v, lift_type = r
            # Normalize date -> timestamp
            if isinstance(date_val, str):
                try:
                    ts = datetime.fromisoformat(date_val)
                except Exception:
                    ts = datetime.strptime(date_val, '%Y-%m-%d')
            elif hasattr(date_val, 'isoformat'):
                # date or datetime
                try:
                    ts = datetime.combine(date_val, datetime.min.time()) if hasattr(date_val, 'day') and not hasattr(date_val, 'hour') else date_val
                except Exception:
                    ts = date_val
            else:
                ts = date_val

            boe = compute_boe(oil_v or 0, gas_v or 0)
            uid = generate_unique_uid(existing)
            row_id = str(uuid.uuid4())

            batch.append((row_id, uid, well_id, boe, float(oil_v or 0), float(gas_v or 0), float(water_v or 0), ts, lift_type))

            processed += 1
            # heartbeat every 1000 rows to show progress
            if processed % 1000 == 0 or processed == total:
                pct = processed / total * 100 if total else 100
                print(f"  Processed {processed:,}/{total:,} rows ({pct:.1f}%), current batch size: {len(batch):,}")
                sys.stdout.flush()

            if len(batch) >= batch_size:
                flush_batch(batch)
                batch = []

        # flush remainder
        if batch:
            flush_batch(batch)

    except KeyboardInterrupt:
        print("\nInsertion interrupted by user. Flushing current batch and exiting...")
        sys.stdout.flush()
        if batch:
            flush_batch(batch)
        cur.close()
        pg.close()
        raise

    # summary
    print(f"Finished inserting. Total inserted: {inserted:,} / {total:,}")
    sys.stdout.flush()

    cur.close()
    pg.close()


def main():
    print("Fetching data from Snowflake...")
    cols, rows = fetch_from_snowflake()
    print(f"Fetched {len(rows):,} rows")

    print("Inserting into Postgres...")
    insert_into_postgres(rows)
    print("Done.")


if __name__ == '__main__':
    main()
