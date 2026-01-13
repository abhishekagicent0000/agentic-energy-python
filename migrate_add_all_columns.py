#!/usr/bin/env python3
"""
Migration to add all lift-type columns to well_sensor_readings table
and drop the lift-type specific tables.
"""

import logging
import os
from dotenv import load_dotenv

try:
    from snowflake.connector import connect as snowflake_connect
except Exception:
    print("ERROR: snowflake-connector-python not installed")
    exit(1)

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

# All columns to add (combining all lift types)
ADD_COLUMNS = [
    # Rod Pump fields
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS strokes_per_minute FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS torque FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS polish_rod_load FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS pump_fillage FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS surface_stroke_length FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS downhole_gross_stroke_length FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS downhole_net_stroke_length FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS runtime FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS cycles_per_day FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS structural_load FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS inferred_production FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS pump_intake_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS pumping_unit_type VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS rod_string VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS dry_rod_weight FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS buoyant_rod_weight FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS pump_friction FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS pump_diameter FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS barrel_length FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS pump_depth FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS controller_mode VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS idle_time_setpoint FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS vfd_settings VARCHAR;",
    
    # ESP fields (already has motor_temp, motor_current)
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS discharge_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS motor_voltage FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS intake_fluid_temp FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS vibration_x FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS vibration_y FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS discharge_temp FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS downhole_flow_rate FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS drive_frequency FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS drive_input_voltage FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS output_voltage FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS input_current FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS motor_speed FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS vsd_temp FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS total_harmonic_distortion FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS motor_load FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS run_stop_status VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS set_frequency FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS acceleration_ramp_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS cycle VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS run_mode VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS alarm VARCHAR;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS flags VARCHAR;",
    
    # Gas Lift fields
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS injection_rate FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS injection_temperature FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS bottomhole_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS injection_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS cycle_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS bottomhole_temp FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS plunger_arrival_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS plunger_velocity FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS arrival_count FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS missed_arrivals FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS shut_in_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS afterflow_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS flow_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS plunger_drop_time FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS min_shut_in_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS max_shut_in_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS open_differential_pressure FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS well_open FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS well_close FLOAT;",
    "ALTER TABLE well_sensor_readings ADD COLUMN IF NOT EXISTS velocity_limit FLOAT;",
]

# Drop lift-type specific tables
DROP_TABLES = [
    "DROP TABLE IF EXISTS rod_pump_readings;",
    "DROP TABLE IF EXISTS esp_readings;",
    "DROP TABLE IF EXISTS gas_lift_readings;",
]

# Clear existing data from well_sensor_readings
CLEAR_DATA = "TRUNCATE TABLE well_sensor_readings;"

def migrate():
    """Add all columns to well_sensor_readings and drop lift-type tables."""
    missing = [k for k in SNOWFLAKE_CONFIG.keys() if not SNOWFLAKE_CONFIG.get(k)]
    if missing:
        logger.error(f"❌ Missing config: {missing}")
        return False
    
    try:
        logger.info("🔗 Connecting to Snowflake...")
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()
        
        logger.info("🗑️ Clearing existing data from well_sensor_readings...")
        cursor.execute(CLEAR_DATA)
        logger.info("✅ Data cleared")
        
        logger.info("➕ Adding all columns to well_sensor_readings...")
        for i, sql in enumerate(ADD_COLUMNS, 1):
            cursor.execute(sql)
            if i % 10 == 0:
                logger.info(f"  Added {i}/{len(ADD_COLUMNS)} columns...")
        logger.info(f"✅ Added {len(ADD_COLUMNS)} columns")
        
        logger.info("🗑️ Dropping lift-type specific tables...")
        for sql in DROP_TABLES:
            cursor.execute(sql)
            logger.info(f"  ✅ {sql.split()[2]}")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ ✅ ✅ Migration completed successfully!")
        logger.info("Now run: python generate.py")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate()
    exit(0 if success else 1)
