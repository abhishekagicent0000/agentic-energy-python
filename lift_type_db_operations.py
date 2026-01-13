"""
Lift-type specific database operations for Rod Pump, ESP, and Gas Lift readings.
Handles insertions and queries for lift-type specific tables.
"""

import logging
import os
import pandas as pd
from dotenv import load_dotenv

try:
    from snowflake.connector import connect as snowflake_connect
except Exception:
    snowflake_connect = None

load_dotenv()

logger = logging.getLogger(__name__)

# Snowflake Config
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

def get_db_connection():
    """Get Snowflake connection."""
    missing_sf = [k for k in SNOWFLAKE_CONFIG.keys() if not SNOWFLAKE_CONFIG.get(k)]
    if missing_sf:
        raise ValueError(f"Missing Snowflake configuration: {missing_sf}")
    try:
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise

# ==================== ROD PUMP ====================

def insert_rod_pump_reading(well_id, timestamp_str, readings):
    """Insert Rod Pump specific reading into rod_pump_readings table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # All Rod Pump attributes (5 from form + others as NULL)
        rod_pump_fields = {
            # Form fields
            'strokes_per_minute': readings.get('strokes_per_minute'),
            'torque': readings.get('torque'),
            'polish_rod_load': readings.get('polish_rod_load'),
            'pump_fillage': readings.get('pump_fillage'),
            'tubing_pressure': readings.get('tubing_pressure'),
            # Additional fields (NULL initially)
            'surface_stroke_length': readings.get('surface_stroke_length'),
            'downhole_gross_stroke_length': readings.get('downhole_gross_stroke_length'),
            'downhole_net_stroke_length': readings.get('downhole_net_stroke_length'),
            'runtime': readings.get('runtime'),
            'cycles_per_day': readings.get('cycles_per_day'),
            'structural_load': readings.get('structural_load'),
            'inferred_production': readings.get('inferred_production'),
            'pump_intake_pressure': readings.get('pump_intake_pressure'),
            'pumping_unit_type': readings.get('pumping_unit_type'),
            'rod_string': readings.get('rod_string'),
            'dry_rod_weight': readings.get('dry_rod_weight'),
            'buoyant_rod_weight': readings.get('buoyant_rod_weight'),
            'pump_friction': readings.get('pump_friction'),
            'pump_diameter': readings.get('pump_diameter'),
            'barrel_length': readings.get('barrel_length'),
            'pump_depth': readings.get('pump_depth'),
            'controller_mode': readings.get('controller_mode'),
            'idle_time_setpoint': readings.get('idle_time_setpoint'),
            'vfd_settings': readings.get('vfd_settings'),
            'alarm': readings.get('alarm'),
        }
        
        # Build INSERT query dynamically
        field_names = ['well_id', 'timestamp'] + list(rod_pump_fields.keys())
        placeholders = ', '.join(['%s'] * len(field_names))
        field_list = ', '.join(field_names)
        
        query = f"INSERT INTO rod_pump_readings ({field_list}) VALUES ({placeholders})"
        
        values = [well_id, timestamp_str] + [float(v) if v and isinstance(v, (int, float)) else v for v in rod_pump_fields.values()]
        
        cursor.execute(query, values)
        conn.commit()
        logger.info(f"✓ Inserted Rod Pump reading for {well_id}")
        
    except Exception as e:
        logger.error(f"Error inserting Rod Pump reading: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def get_rod_pump_history(well_id, limit=20):
    """Fetch Rod Pump readings for a well."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT * FROM rod_pump_readings
            WHERE well_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (well_id, limit)
        )
        rows = cursor.fetchall()
        if rows and cursor.description:
            cols = [c[0].lower() for c in cursor.description]
            df = pd.DataFrame(rows, columns=cols)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching Rod Pump history: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        conn.close()

# ==================== ESP ====================

def insert_esp_reading(well_id, timestamp_str, readings):
    """Insert ESP specific reading into esp_readings table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # All ESP attributes (5 from form + others as NULL)
        esp_fields = {
            # Form fields
            'motor_temp': readings.get('motor_temp'),
            'motor_current': readings.get('motor_current'),
            'discharge_pressure': readings.get('discharge_pressure'),
            'pump_intake_pressure': readings.get('pump_intake_pressure'),
            'motor_voltage': readings.get('motor_voltage'),
            # Additional fields (NULL initially)
            'intake_fluid_temp': readings.get('intake_fluid_temp'),
            'vibration_x': readings.get('vibration_x'),
            'vibration_y': readings.get('vibration_y'),
            'discharge_temp': readings.get('discharge_temp'),
            'downhole_flow_rate': readings.get('downhole_flow_rate'),
            'drive_frequency': readings.get('drive_frequency'),
            'tubing_pressure': readings.get('tubing_pressure'),
            'casing_pressure': readings.get('casing_pressure'),
            'drive_input_voltage': readings.get('drive_input_voltage'),
            'output_voltage': readings.get('output_voltage'),
            'input_current': readings.get('input_current'),
            'motor_speed': readings.get('motor_speed'),
            'vsd_temp': readings.get('vsd_temp'),
            'total_harmonic_distortion': readings.get('total_harmonic_distortion'),
            'motor_load': readings.get('motor_load'),
            'run_stop_status': readings.get('run_stop_status'),
            'set_frequency': readings.get('set_frequency'),
            'acceleration_ramp_time': readings.get('acceleration_ramp_time'),
            'cycle': readings.get('cycle'),
            'run_mode': readings.get('run_mode'),
            'alarm': readings.get('alarm'),
            'flags': readings.get('flags'),
        }
        
        # Build INSERT query dynamically
        field_names = ['well_id', 'timestamp'] + list(esp_fields.keys())
        placeholders = ', '.join(['%s'] * len(field_names))
        field_list = ', '.join(field_names)
        
        query = f"INSERT INTO esp_readings ({field_list}) VALUES ({placeholders})"
        
        values = [well_id, timestamp_str] + [float(v) if v and isinstance(v, (int, float)) else v for v in esp_fields.values()]
        
        cursor.execute(query, values)
        conn.commit()
        logger.info(f"✓ Inserted ESP reading for {well_id}")
        
    except Exception as e:
        logger.error(f"Error inserting ESP reading: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def get_esp_history(well_id, limit=20):
    """Fetch ESP readings for a well."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT * FROM esp_readings
            WHERE well_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (well_id, limit)
        )
        rows = cursor.fetchall()
        if rows and cursor.description:
            cols = [c[0].lower() for c in cursor.description]
            df = pd.DataFrame(rows, columns=cols)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching ESP history: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        conn.close()

# ==================== GAS LIFT ====================

def insert_gas_lift_reading(well_id, timestamp_str, readings):
    """Insert Gas Lift specific reading into gas_lift_readings table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # All Gas Lift attributes (5 from form + others as NULL)
        gas_lift_fields = {
            # Form fields
            'injection_rate': readings.get('injection_rate'),
            'injection_temperature': readings.get('injection_temperature'),
            'bottomhole_pressure': readings.get('bottomhole_pressure'),
            'injection_pressure': readings.get('injection_pressure'),
            'cycle_time': readings.get('cycle_time'),
            # Additional fields (NULL initially)
            'bottomhole_temp': readings.get('bottomhole_temp'),
            'plunger_arrival_time': readings.get('plunger_arrival_time'),
            'plunger_velocity': readings.get('plunger_velocity'),
            'arrival_count': readings.get('arrival_count'),
            'missed_arrivals': readings.get('missed_arrivals'),
            'shut_in_time': readings.get('shut_in_time'),
            'afterflow_time': readings.get('afterflow_time'),
            'flow_time': readings.get('flow_time'),
            'plunger_drop_time': readings.get('plunger_drop_time'),
            'min_shut_in_pressure': readings.get('min_shut_in_pressure'),
            'max_shut_in_pressure': readings.get('max_shut_in_pressure'),
            'open_differential_pressure': readings.get('open_differential_pressure'),
            'well_open': readings.get('well_open'),
            'well_close': readings.get('well_close'),
            'velocity_limit': readings.get('velocity_limit'),
            'alarm': readings.get('alarm'),
            'flags': readings.get('flags'),
        }
        
        # Build INSERT query dynamically
        field_names = ['well_id', 'timestamp'] + list(gas_lift_fields.keys())
        placeholders = ', '.join(['%s'] * len(field_names))
        field_list = ', '.join(field_names)
        
        query = f"INSERT INTO gas_lift_readings ({field_list}) VALUES ({placeholders})"
        
        values = [well_id, timestamp_str] + [float(v) if v and isinstance(v, (int, float)) else v for v in gas_lift_fields.values()]
        
        cursor.execute(query, values)
        conn.commit()
        logger.info(f"✓ Inserted Gas Lift reading for {well_id}")
        
    except Exception as e:
        logger.error(f"Error inserting Gas Lift reading: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def get_gas_lift_history(well_id, limit=20):
    """Fetch Gas Lift readings for a well."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT * FROM gas_lift_readings
            WHERE well_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (well_id, limit)
        )
        rows = cursor.fetchall()
        if rows and cursor.description:
            cols = [c[0].lower() for c in cursor.description]
            df = pd.DataFrame(rows, columns=cols)
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching Gas Lift history: {e}")
        return pd.DataFrame()
    finally:
        cursor.close()
        conn.close()

# ==================== DISPATCHER ====================

def insert_lift_type_specific_reading(well_id, timestamp_str, lift_type, readings):
    """Route to correct lift-type specific insert function."""
    if lift_type == "Rod Pump":
        insert_rod_pump_reading(well_id, timestamp_str, readings)
    elif lift_type == "ESP":
        insert_esp_reading(well_id, timestamp_str, readings)
    elif lift_type == "Gas Lift":
        insert_gas_lift_reading(well_id, timestamp_str, readings)
    else:
        logger.warning(f"Unknown lift_type: {lift_type}")

def get_lift_type_specific_history(well_id, lift_type, limit=20):
    """Fetch lift-type specific readings for a well."""
    if lift_type == "Rod Pump":
        return get_rod_pump_history(well_id, limit)
    elif lift_type == "ESP":
        return get_esp_history(well_id, limit)
    elif lift_type == "Gas Lift":
        return get_gas_lift_history(well_id, limit)
    else:
        logger.warning(f"Unknown lift_type: {lift_type}")
        return pd.DataFrame()
