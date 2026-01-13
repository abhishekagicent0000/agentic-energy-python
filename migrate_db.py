#!/usr/bin/env python3
"""
Database Migration Script
Adds well-type specific columns to well_sensor_readings table
Supports both PostgreSQL and Snowflake
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================================
# POSTGRESQL MIGRATION
# ============================================================================

def migrate_postgresql():
    """Add well-type specific columns to PostgreSQL well_sensor_readings table"""
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            database=os.getenv('POSTGRES_DB', 'at_risk_assets'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            port=os.getenv('POSTGRES_PORT', 5432)
        )
        
        cursor = conn.cursor()
        logger.info("✓ Connected to PostgreSQL")
        
        # List of columns to add
        columns_to_add = [
            ('lift_type', 'VARCHAR(50)'),
            ('strokes_per_minute', 'DOUBLE PRECISION'),
            ('torque', 'DOUBLE PRECISION'),
            ('polish_rod_load', 'DOUBLE PRECISION'),
            ('pump_fillage', 'DOUBLE PRECISION'),
            ('discharge_pressure', 'DOUBLE PRECISION'),
            ('pump_intake_pressure', 'DOUBLE PRECISION'),
            ('motor_voltage', 'DOUBLE PRECISION'),
            ('injection_rate', 'DOUBLE PRECISION'),
            ('injection_temperature', 'DOUBLE PRECISION'),
            ('bottomhole_pressure', 'DOUBLE PRECISION'),
            ('injection_pressure', 'DOUBLE PRECISION'),
            ('cycle_time', 'DOUBLE PRECISION'),
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"""
                    ALTER TABLE well_sensor_readings 
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                """)
                logger.info(f"  ✓ Added column: {col_name} ({col_type})")
            except Exception as e:
                logger.warning(f"  ⚠ Column {col_name} might already exist: {e}")
        
        # Create indexes
        indexes = [
            ('idx_lift_type', 'lift_type'),
            ('idx_well_lift_time', 'well_id, lift_type, timestamp DESC'),
        ]
        
        for idx_name, idx_cols in indexes:
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} 
                    ON well_sensor_readings({idx_cols})
                """)
                logger.info(f"  ✓ Created index: {idx_name}")
            except Exception as e:
                logger.warning(f"  ⚠ Index {idx_name} might already exist: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✓ PostgreSQL migration completed successfully\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ PostgreSQL migration failed: {e}")
        return False

# ============================================================================
# SNOWFLAKE MIGRATION
# ============================================================================

def migrate_snowflake():
    """Add well-type specific columns to Snowflake well_sensor_readings table"""
    try:
        import snowflake.connector
        
        conn = snowflake.connector.connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            database=os.getenv('SNOWFLAKE_DATABASE', 'AT_RISK_ASSETS'),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'SENSOR_DATA'),
        )
        
        cursor = conn.cursor()
        logger.info("✓ Connected to Snowflake")
        
        # List of columns to add
        columns_to_add = [
            ('lift_type', 'VARCHAR(50)'),
            ('strokes_per_minute', 'FLOAT'),
            ('torque', 'FLOAT'),
            ('polish_rod_load', 'FLOAT'),
            ('pump_fillage', 'FLOAT'),
            ('discharge_pressure', 'FLOAT'),
            ('pump_intake_pressure', 'FLOAT'),
            ('motor_voltage', 'FLOAT'),
            ('injection_rate', 'FLOAT'),
            ('injection_temperature', 'FLOAT'),
            ('bottomhole_pressure', 'FLOAT'),
            ('injection_pressure', 'FLOAT'),
            ('cycle_time', 'FLOAT'),
        ]
        
        table_name = f"{os.getenv('SNOWFLAKE_DATABASE', 'AT_RISK_ASSETS')}.{os.getenv('SNOWFLAKE_SCHEMA', 'SENSOR_DATA')}.WELL_SENSOR_READINGS"
        
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name}
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                """)
                logger.info(f"  ✓ Added column: {col_name} ({col_type})")
            except Exception as e:
                if 'already exists' in str(e).lower():
                    logger.info(f"  ✓ Column {col_name} already exists")
                else:
                    logger.warning(f"  ⚠ Error with {col_name}: {e}")
        
        conn.close()
        logger.info("✓ Snowflake migration completed successfully\n")
        return True
        
    except Exception as e:
        logger.error(f"✗ Snowflake migration failed: {e}")
        logger.error("Ensure SNOWFLAKE_* environment variables are set")
        return False

# ============================================================================
# VERIFICATION QUERIES
# ============================================================================

def verify_postgresql():
    """Verify PostgreSQL columns were added"""
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            database=os.getenv('POSTGRES_DB', 'at_risk_assets'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            port=os.getenv('POSTGRES_PORT', 5432)
        )
        
        cursor = conn.cursor()
        
        # Get all columns in the table
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'well_sensor_readings'
            ORDER BY column_name
        """)
        
        columns = cursor.fetchall()
        
        logger.info("PostgreSQL well_sensor_readings columns:")
        logger.info("-" * 50)
        for col_name, col_type in columns:
            logger.info(f"  {col_name:30} {col_type}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")

def verify_snowflake():
    """Verify Snowflake columns were added"""
    try:
        import snowflake.connector
        
        conn = snowflake.connector.connect(
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH'),
            database=os.getenv('SNOWFLAKE_DATABASE', 'AT_RISK_ASSETS'),
            schema=os.getenv('SNOWFLAKE_SCHEMA', 'SENSOR_DATA'),
        )
        
        cursor = conn.cursor()
        table_name = f"{os.getenv('SNOWFLAKE_DATABASE', 'AT_RISK_ASSETS')}.{os.getenv('SNOWFLAKE_SCHEMA', 'SENSOR_DATA')}.WELL_SENSOR_READINGS"
        
        cursor.execute(f"DESC TABLE {table_name}")
        
        columns = cursor.fetchall()
        
        logger.info("Snowflake WELL_SENSOR_READINGS columns:")
        logger.info("-" * 50)
        for col in columns:
            logger.info(f"  {col[0]:30} {col[1]:20} {col[2]}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run migrations"""
    logger.info("=" * 70)
    logger.info("DATABASE MIGRATION: Add Well-Type Specific Fields")
    logger.info("=" * 70)
    logger.info("")
    
    # PostgreSQL Migration
    logger.info("1. POSTGRESQL MIGRATION")
    logger.info("-" * 70)
    postgres_success = migrate_postgresql()
    
    # Snowflake Migration
    logger.info("2. SNOWFLAKE MIGRATION")
    logger.info("-" * 70)
    snowflake_success = migrate_snowflake()
    
    # Verification
    logger.info("3. VERIFICATION")
    logger.info("-" * 70)
    
    if postgres_success:
        logger.info("\nPostgreSQL Columns:")
        verify_postgresql()
    
    if snowflake_success:
        logger.info("\nSnowflake Columns:")
        verify_snowflake()
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"PostgreSQL: {'✓ SUCCESS' if postgres_success else '✗ FAILED'}")
    logger.info(f"Snowflake: {'✓ SUCCESS' if snowflake_success else '✗ FAILED'}")
    logger.info("")
    
    if postgres_success or snowflake_success:
        logger.info("✓ Migration completed")
        logger.info("\nColumns added:")
        logger.info("  ROD PUMP: strokes_per_minute, torque, polish_rod_load, pump_fillage, tubing_pressure")
        logger.info("  ESP: motor_temp, motor_current, discharge_pressure, pump_intake_pressure, motor_voltage")
        logger.info("  GAS LIFT: injection_rate, injection_temperature, bottomhole_pressure, injection_pressure, cycle_time")
        return 0
    else:
        logger.error("✗ Migration failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
