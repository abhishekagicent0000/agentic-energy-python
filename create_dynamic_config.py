#!/usr/bin/env python3
"""
Create and seed dynamic configuration tables in Snowflake.
"""

import os
import logging
from dotenv import load_dotenv
from snowflake.connector import connect as snowflake_connect

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config from environment
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

def main():
    conn = snowflake_connect(**SNOWFLAKE_CONFIG)
    cursor = conn.cursor()
    
    # Create lift_types table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lift_types (
            id INT AUTOINCREMENT PRIMARY KEY,
            lift_type_name VARCHAR NOT NULL UNIQUE,
            description VARCHAR,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    logger.info("✅ Created lift_types table")
    
    # Create sensors table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_definitions (
            id INT AUTOINCREMENT PRIMARY KEY,
            lift_type_id INT NOT NULL,
            sensor_name VARCHAR NOT NULL,
            field_name VARCHAR NOT NULL,
            unit VARCHAR,
            min_value FLOAT,
            max_value FLOAT,
            is_form_field BOOLEAN DEFAULT FALSE,
            display_order INT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    logger.info("✅ Created sensor_definitions table")
    
    # Create anomaly_rules table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_rules (
            id INT AUTOINCREMENT PRIMARY KEY,
            lift_type_id INT NOT NULL,
            sensor_id INT NOT NULL,
            rule_type VARCHAR,
            lower_bound FLOAT,
            upper_bound FLOAT,
            severity VARCHAR,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)
    logger.info("✅ Created anomaly_rules table")
    
    conn.commit()
    logger.info("✅ Configuration tables created!")
    
    # Seed lift types
    lift_types_data = [
        ("Rod Pump", "Sucker rod pumping system"),
        ("ESP", "Electrical Submersible Pump"),
        ("Gas Lift", "Gas lift artificial lift system")
    ]
    
    for name, desc in lift_types_data:
        try:
            cursor.execute(
                "INSERT INTO lift_types (lift_type_name, description) VALUES (%s, %s)",
                (name, desc)
            )
        except:
            pass
    
    logger.info("✅ Inserted lift types")
    
    # Get lift_type IDs
    cursor.execute("SELECT id, lift_type_name FROM lift_types")
    lift_types_map = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Rod Pump sensors - ALL data from Excel
    rod_pump_sensors = [
        (lift_types_map["Rod Pump"], "Strokes Per Minute", "strokes_per_minute", "SPM", 0.5, 12.0, True, 1),
        (lift_types_map["Rod Pump"], "Torque", "torque", "in-lbs", 100000, 1824000, False, 2),
        (lift_types_map["Rod Pump"], "Polish Rod Load", "polish_rod_load", "lbs", 0, 125000, True, 3),
        (lift_types_map["Rod Pump"], "Pump Fillage", "pump_fillage", "%", 0, 100, True, 4),
        (lift_types_map["Rod Pump"], "Surface Stroke Length", "surface_stroke_length", "inches", 24, 500, False, 5),
        (lift_types_map["Rod Pump"], "Downhole Gross Stroke Length", "downhole_gross_stroke", "inches", 20, 600, False, 6),
        (lift_types_map["Rod Pump"], "Downhole Net Stroke Length", "downhole_net_stroke", "inches", 20, 600, False, 7),
        (lift_types_map["Rod Pump"], "Runtime", "runtime", "hours", 0, 24, False, 8),
        (lift_types_map["Rod Pump"], "Cycles", "cycles", "cycles/day", 0, 100, False, 9),
        (lift_types_map["Rod Pump"], "Structural Load", "structural_load", "%", 0, 200, False, 10),
        (lift_types_map["Rod Pump"], "Inferred Production", "inferred_production", "bbls", 1, 1000, False, 11),
        (lift_types_map["Rod Pump"], "Pump Intake Pressure", "pump_intake_pressure", "psi", 0, 5000, False, 12),
        (lift_types_map["Rod Pump"], "Dry Rod Weight", "dry_rod_weight", "lbs", 1000, 150000, False, 13),
        (lift_types_map["Rod Pump"], "Buoyant Rod Weight", "buoyant_rod_weight", "lbs", 1000, 150000, False, 14),
        (lift_types_map["Rod Pump"], "Pump Friction", "pump_friction", "lbs", 0, 1000, False, 15),
        (lift_types_map["Rod Pump"], "Pump Diameter", "pump_diameter", "inches", 1.0, 3.25, False, 16),
        (lift_types_map["Rod Pump"], "Barrel Length", "barrel_length", "feet", 5, 50, False, 17),
        (lift_types_map["Rod Pump"], "Pump Depth", "pump_depth", "feet", 500, 12000, False, 18),
        (lift_types_map["Rod Pump"], "Idle Time Set-point", "idle_time_setpoint", "minutes", 0, 120, False, 19),
    ]
    
    # ESP with VSD sensors - ALL data from Excel
    esp_sensors = [
        (lift_types_map["ESP"], "Motor Temperature", "motor_temp", "°F", 50, 1500, True, 1),
        (lift_types_map["ESP"], "Motor Current", "motor_current", "A", 0, 200, True, 2),
        (lift_types_map["ESP"], "Discharge Pressure", "discharge_pressure", "psi", 0, 5000, True, 3),
        (lift_types_map["ESP"], "Pump Intake Pressure", "pump_intake_pressure", "psi", 0, 5000, False, 4),
        (lift_types_map["ESP"], "Motor Voltage", "motor_voltage", "V", 0, 5000, True, 5),
        (lift_types_map["ESP"], "Intake/Fluid Temperature", "intake_fluid_temp", "°F", 50, 900, False, 6),
        (lift_types_map["ESP"], "Vibration X", "vibration_x", "in/s", 0.0, 1.25, False, 7),
        (lift_types_map["ESP"], "Vibration Y", "vibration_y", "in/s", 0.0, 1.25, False, 8),
        (lift_types_map["ESP"], "Discharge Temperature", "discharge_temp", "°F", 0, 1500, False, 9),
        (lift_types_map["ESP"], "Downhole Flow Rate", "downhole_flow_rate", "bpd", 0, 10000, False, 10),
        (lift_types_map["ESP"], "Drive Frequency", "drive_frequency", "Hz", 0, 100, False, 11),
        (lift_types_map["ESP"], "Tubing Pressure", "tubing_pressure", "psi", 0, 2000, False, 12),
        (lift_types_map["ESP"], "Casing Pressure", "casing_pressure", "psi", 0, 2000, False, 13),
        (lift_types_map["ESP"], "Drive Input Voltage", "drive_input_voltage", "V", 340, 550, False, 14),
        (lift_types_map["ESP"], "Output Voltage", "output_voltage", "V", 460, 4200, False, 15),
        (lift_types_map["ESP"], "Input Current", "input_current", "A", 10, 200, False, 16),
        (lift_types_map["ESP"], "Motor Speed", "motor_speed", "rpm", 0, 7000, False, 17),
        (lift_types_map["ESP"], "VSD Temperature", "vsd_temp", "°F", 0, 1000, False, 18),
        (lift_types_map["ESP"], "Total Harmonic Distortion", "total_harmonic_distortion", "%", 0, 100, False, 19),
        (lift_types_map["ESP"], "Motor Load", "motor_load", "%", 0, 150, False, 20),
        (lift_types_map["ESP"], "Set Frequency", "set_frequency", "Hz", 0, 100, False, 21),
        (lift_types_map["ESP"], "Acceleration Ramp Time", "acceleration_ramp_time", "seconds", 0, 60, False, 22),
        (lift_types_map["ESP"], "Cycle", "cycle", "cycles", 0, 100, False, 23),
    ]
    
    # Gas Lift and Plunger Lift sensors - ALL data from Excel
    gas_lift_sensors = [
        (lift_types_map["Gas Lift"], "Injection Rate", "injection_rate", "mcf", 0, 2000, True, 1),
        (lift_types_map["Gas Lift"], "Injection Temperature", "injection_temperature", "°F", 0, 1000, True, 2),
        (lift_types_map["Gas Lift"], "Bottomhole Pressure", "bottomhole_pressure", "psi", 50, 5000, True, 3),
        (lift_types_map["Gas Lift"], "Injection Pressure", "injection_pressure", "psi", 100, 5000, True, 4),
        (lift_types_map["Gas Lift"], "Cycle Time", "cycle_time", "minutes", 5, 180, False, 5),
        (lift_types_map["Gas Lift"], "Bottomhole Temperature", "bottomhole_temp", "°F", 50, 500, False, 6),
        (lift_types_map["Gas Lift"], "Plunger Arrival Time", "plunger_arrival_time", "minutes", 0, 45, False, 7),
        (lift_types_map["Gas Lift"], "Plunger Velocity", "plunger_velocity", "ft/min", 0, 2000, False, 8),
        (lift_types_map["Gas Lift"], "Arrival Count", "arrival_count", "count", 0, 200, False, 9),
        (lift_types_map["Gas Lift"], "Missed Arrivals", "missed_arrivals", "count", 0, 200, False, 10),
        (lift_types_map["Gas Lift"], "Shut-in Time", "shutin_time", "minutes", 5, 60, False, 11),
        (lift_types_map["Gas Lift"], "Afterflow Time", "afterflow_time", "minutes", 2, 45, False, 12),
        (lift_types_map["Gas Lift"], "Flow Time", "flow_time", "minutes", 5, 60, False, 13),
        (lift_types_map["Gas Lift"], "Plunger Drop Time", "plunger_drop_time", "minutes", 1, 20, False, 14),
        (lift_types_map["Gas Lift"], "Minimum Shut-in Pressure", "min_shutin_pressure", "psi", 0, 1000, False, 15),
        (lift_types_map["Gas Lift"], "Maximum Shut-in Pressure", "max_shutin_pressure", "psi", 0, 1000, False, 16),
        (lift_types_map["Gas Lift"], "Open Differential Pressure", "open_diff_pressure", "psi", 50, 300, False, 17),
        (lift_types_map["Gas Lift"], "Well Open", "well_open", "psi", 100, 500, False, 18),
        (lift_types_map["Gas Lift"], "Well Close", "well_close", "psi", 20, 100, False, 19),
        (lift_types_map["Gas Lift"], "Velocity Limit", "velocity_limit", "ft/min", 400, 2000, False, 20),
    ]
    
    all_sensors = rod_pump_sensors + esp_sensors + gas_lift_sensors
    
    for lift_type_id, sensor_name, field_name, unit, min_val, max_val, is_form, order in all_sensors:
        try:
            cursor.execute("""
                INSERT INTO sensor_definitions 
                (lift_type_id, sensor_name, field_name, unit, min_value, max_value, is_form_field, display_order)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (lift_type_id, sensor_name, field_name, unit, min_val, max_val, is_form, order))
        except:
            pass
    
    logger.info("✅ Inserted sensor definitions")
    
    conn.commit()
    conn.close()
    logger.info("✅ ✅ ✅ Dynamic configuration created and seeded!")

if __name__ == "__main__":
    main()
