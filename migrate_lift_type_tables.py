#!/usr/bin/env python3
"""
Migration script to create lift-type specific tables in Snowflake.
Run this ONCE to create the tables: python migrate_lift_type_tables.py
"""

import logging
import os
from dotenv import load_dotenv

try:
    from snowflake.connector import connect as snowflake_connect
except Exception as e:
    print(f"ERROR: snowflake-connector-python not installed. Install with: pip install snowflake-connector-python")
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

# SQL for Rod Pump Table
ROD_PUMP_TABLE = """
CREATE TABLE IF NOT EXISTS rod_pump_readings (
    id INT AUTOINCREMENT PRIMARY KEY,
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP_NTZ NOT NULL,
    
    -- Form fields (5)
    strokes_per_minute FLOAT,
    torque FLOAT,
    polish_rod_load FLOAT,
    pump_fillage FLOAT,
    tubing_pressure FLOAT,
    
    -- Additional attributes (19)
    surface_stroke_length FLOAT,
    downhole_gross_stroke_length FLOAT,
    downhole_net_stroke_length FLOAT,
    runtime FLOAT,
    cycles_per_day FLOAT,
    structural_load FLOAT,
    inferred_production FLOAT,
    pump_intake_pressure FLOAT,
    pumping_unit_type VARCHAR(255),
    rod_string VARCHAR(255),
    dry_rod_weight FLOAT,
    buoyant_rod_weight FLOAT,
    pump_friction FLOAT,
    pump_diameter FLOAT,
    barrel_length FLOAT,
    pump_depth FLOAT,
    controller_mode VARCHAR(255),
    idle_time_setpoint FLOAT,
    vfd_settings VARCHAR(255),
    alarm VARCHAR(255)
);
"""

# SQL for ESP Table
ESP_TABLE = """
CREATE TABLE IF NOT EXISTS esp_readings (
    id INT AUTOINCREMENT PRIMARY KEY,
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP_NTZ NOT NULL,
    
    -- Form fields (5)
    motor_temp FLOAT,
    motor_current FLOAT,
    discharge_pressure FLOAT,
    pump_intake_pressure FLOAT,
    motor_voltage FLOAT,
    
    -- Additional attributes (21)
    intake_fluid_temp FLOAT,
    vibration_x FLOAT,
    vibration_y FLOAT,
    discharge_temp FLOAT,
    downhole_flow_rate FLOAT,
    drive_frequency FLOAT,
    tubing_pressure FLOAT,
    casing_pressure FLOAT,
    drive_input_voltage FLOAT,
    output_voltage FLOAT,
    input_current FLOAT,
    motor_speed FLOAT,
    vsd_temp FLOAT,
    total_harmonic_distortion FLOAT,
    motor_load FLOAT,
    run_stop_status VARCHAR(255),
    set_frequency FLOAT,
    acceleration_ramp_time FLOAT,
    cycle VARCHAR(255),
    run_mode VARCHAR(255),
    alarm VARCHAR(255),
    flags VARCHAR(255)
);
"""

# SQL for Gas Lift Table
GAS_LIFT_TABLE = """
CREATE TABLE IF NOT EXISTS gas_lift_readings (
    id INT AUTOINCREMENT PRIMARY KEY,
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP_NTZ NOT NULL,
    
    -- Form fields (5)
    injection_rate FLOAT,
    injection_temperature FLOAT,
    bottomhole_pressure FLOAT,
    injection_pressure FLOAT,
    cycle_time FLOAT,
    
    -- Additional attributes (17)
    bottomhole_temp FLOAT,
    plunger_arrival_time FLOAT,
    plunger_velocity FLOAT,
    arrival_count FLOAT,
    missed_arrivals FLOAT,
    shut_in_time FLOAT,
    afterflow_time FLOAT,
    flow_time FLOAT,
    plunger_drop_time FLOAT,
    min_shut_in_pressure FLOAT,
    max_shut_in_pressure FLOAT,
    open_differential_pressure FLOAT,
    well_open FLOAT,
    well_close FLOAT,
    velocity_limit FLOAT,
    alarm VARCHAR(255),
    flags VARCHAR(255)
);
"""

def migrate():
    """Create lift-type specific tables in Snowflake."""
    missing = [k for k in SNOWFLAKE_CONFIG.keys() if not SNOWFLAKE_CONFIG.get(k)]
    if missing:
        logger.error(f"❌ Missing Snowflake config: {missing}")
        logger.error("Ensure these environment variables are set: SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_ROLE")
        return False
    
    try:
        logger.info("🔗 Connecting to Snowflake...")
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()
        
        logger.info("📝 Creating Rod Pump readings table...")
        cursor.execute(ROD_PUMP_TABLE)
        logger.info("✅ Rod Pump table created/verified")
        
        logger.info("📝 Creating ESP readings table...")
        cursor.execute(ESP_TABLE)
        logger.info("✅ ESP table created/verified")
        
        logger.info("📝 Creating Gas Lift readings table...")
        cursor.execute(GAS_LIFT_TABLE)
        logger.info("✅ Gas Lift table created/verified")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ ✅ ✅ Migration completed successfully!")
        logger.info("📊 Tables created:")
        logger.info("   - rod_pump_readings (24 columns)")
        logger.info("   - esp_readings (26 columns)")
        logger.info("   - gas_lift_readings (22 columns)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = migrate()
    exit(0 if success else 1)
