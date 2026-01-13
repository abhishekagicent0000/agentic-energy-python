#!/usr/bin/env python3
"""
Dynamic configuration loader - fetches sensor configs and rules from Snowflake.
"""

import os
import logging
from dotenv import load_dotenv
from snowflake.connector import connect as snowflake_connect

load_dotenv()

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

# Cache for config
_config_cache = {}

def get_db_connection():
    """Get Snowflake connection."""
    return snowflake_connect(**SNOWFLAKE_CONFIG)

def get_lift_types():
    """Fetch all lift types from DB."""
    if 'lift_types' in _config_cache:
        return _config_cache['lift_types']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, lift_type_name FROM lift_types ORDER BY lift_type_name")
    lift_types = {row[1]: row[0] for row in cursor.fetchall()}
    
    conn.close()
    _config_cache['lift_types'] = lift_types
    return lift_types

def get_sensors_for_lift_type(lift_type_name):
    """Fetch sensor definitions for a specific lift type."""
    cache_key = f'sensors_{lift_type_name}'
    if cache_key in _config_cache:
        return _config_cache[cache_key]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT sd.id, sd.sensor_name, sd.field_name, sd.unit, 
               sd.min_value, sd.max_value, sd.is_form_field, sd.display_order
        FROM sensor_definitions sd
        JOIN lift_types lt ON sd.lift_type_id = lt.id
        WHERE lt.lift_type_name = %s
        ORDER BY sd.display_order
    """, (lift_type_name,))
    
    sensors = []
    for row in cursor.fetchall():
        sensors.append({
            'id': row[0],
            'name': row[1],
            'field_name': row[2],
            'unit': row[3],
            'min': row[4],
            'max': row[5],
            'is_form_field': row[6],
            'display_order': row[7]
        })
    
    conn.close()
    _config_cache[cache_key] = sensors
    return sensors

def get_form_sensors(lift_type_name):
    """Get only form fields for a lift type."""
    sensors = get_sensors_for_lift_type(lift_type_name)
    return [s for s in sensors if s['is_form_field']]

def get_sensor_ranges(lift_type_name):
    """Get sensor field names and ranges for a lift type."""
    sensors = get_sensors_for_lift_type(lift_type_name)
    ranges = {}
    for sensor in sensors:
        ranges[sensor['field_name']] = {
            'min': sensor['min'],
            'max': sensor['max'],
            'unit': sensor['unit']
        }
    return ranges

def get_form_fields(lift_type_name):
    """Get form field names for a lift type."""
    sensors = get_form_sensors(lift_type_name)
    return [s['field_name'] for s in sensors]

def get_anomaly_rules(lift_type_name):
    """Fetch anomaly rules for a lift type."""
    cache_key = f'rules_{lift_type_name}'
    if cache_key in _config_cache:
        return _config_cache[cache_key]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ar.sensor_id, sd.field_name, ar.lower_bound, ar.upper_bound, ar.severity
        FROM anomaly_rules ar
        JOIN sensor_definitions sd ON ar.sensor_id = sd.id
        JOIN lift_types lt ON ar.lift_type_id = lt.id
        WHERE lt.lift_type_name = %s
    """, (lift_type_name,))
    
    rules = {}
    for row in cursor.fetchall():
        field_name = row[1]
        rules[field_name] = {
            'lower': row[2],
            'upper': row[3],
            'severity': row[4]
        }
    
    conn.close()
    _config_cache[cache_key] = rules
    return rules

def clear_cache():
    """Clear configuration cache."""
    global _config_cache
    _config_cache = {}

def get_all_sensor_info():
    """Get all sensor info for all lift types."""
    lift_types = get_lift_types()
    all_info = {}
    
    for lift_type_name in lift_types.keys():
        all_info[lift_type_name] = {
            'sensors': get_sensors_for_lift_type(lift_type_name),
            'form_fields': get_form_fields(lift_type_name),
            'ranges': get_sensor_ranges(lift_type_name),
            'rules': get_anomaly_rules(lift_type_name)
        }
    
    return all_info

# Test
if __name__ == "__main__":
    print("=== Lift Types ===")
    lift_types = get_lift_types()
    print(lift_types)
    
    print("\n=== Rod Pump Sensors ===")
    sensors = get_sensors_for_lift_type("Rod Pump")
    for s in sensors:
        print(f"  {s['name']}: {s['field_name']} ({s['unit']}) - Form: {s['is_form_field']}")
    
    print("\n=== Rod Pump Form Fields ===")
    form_fields = get_form_fields("Rod Pump")
    print(form_fields)
    
    print("\n=== Rod Pump Ranges ===")
    ranges = get_sensor_ranges("Rod Pump")
    print(ranges)
