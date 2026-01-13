# Lift-Type Database Integration - Complete Setup Guide

## Overview

This document describes the complete lift-type specific database integration for Rod Pump, ESP, and Gas Lift sensors.

## Architecture

### Database Schema

Three lift-type specific tables store comprehensive sensor data:

#### Rod Pump (rod_pump_readings) - 24 columns

**Form Fields (5):**

- strokes_per_minute (0.5-12.0)
- torque (100k-1.8M in-lbs)
- polish_rod_load (0-125k lbs)
- pump_fillage (0-100%)
- tubing_pressure (0-5000 psi)

**Additional Attributes (19):**

- surface_stroke_length, downhole_gross_stroke_length, downhole_net_stroke_length
- runtime, cycles_per_day, structural_load, inferred_production
- pump_intake_pressure, pumping_unit_type, rod_string
- dry_rod_weight, buoyant_rod_weight, pump_friction
- pump_diameter, barrel_length, pump_depth
- controller_mode, idle_time_setpoint, vfd_settings, alarm

#### ESP (esp_readings) - 26 columns

**Form Fields (5):**

- motor_temp (50-1500°F)
- motor_current (0-200A)
- discharge_pressure (0-5000 psi)
- pump_intake_pressure (0-5000 psi)
- motor_voltage (0-5000V)

**Additional Attributes (21):**

- intake_fluid_temp, vibration_x, vibration_y, discharge_temp
- downhole_flow_rate, drive_frequency, tubing_pressure, casing_pressure
- drive_input_voltage, output_voltage, input_current, motor_speed
- vsd_temp, total_harmonic_distortion, motor_load
- run_stop_status, set_frequency, acceleration_ramp_time, cycle, run_mode, alarm, flags

#### Gas Lift (gas_lift_readings) - 22 columns

**Form Fields (5):**

- injection_rate (0-2000 scf/d)
- injection_temperature (0-1000°F)
- bottomhole_pressure (50-5000 psi)
- injection_pressure (100-5000 psi)
- cycle_time (5-180 min)

**Additional Attributes (17):**

- bottomhole_temp, plunger_arrival_time, plunger_velocity
- arrival_count, missed_arrivals, shut_in_time, afterflow_time, flow_time
- plunger_drop_time, min_shut_in_pressure, max_shut_in_pressure
- open_differential_pressure, well_open, well_close, velocity_limit, alarm, flags

## Setup Steps

### Step 1: Create Tables in Snowflake

```bash
python migrate_lift_type_tables.py
```

This script will:

- Connect to Snowflake using environment variables
- Create rod_pump_readings table (24 columns)
- Create esp_readings table (26 columns)
- Create gas_lift_readings table (22 columns)
- Create indexes on (well_id, timestamp DESC) for performance

### Step 2: Verify Installation

```bash
python test_lift_type_integration.py
```

This script will:

- Test insert for each lift type
- Test retrieval for each lift type
- Display column information
- Verify connectivity and schema

## Data Flow

### Insert Pipeline

```
Web Form (5 fields)
    ↓
API /api/insert_reading endpoint
    ↓
Convert types (numpy → Python)
    ↓
Insert to well_sensor_readings (main table)
    ↓
Insert to lift_type_specific_table
    (rod_pump_readings OR esp_readings OR gas_lift_readings)
    ↓
Run anomaly detection
    ↓
Insert anomalies if detected
```

### Retrieve Pipeline

```
API /api/well-history/<well_id>
    ↓
Query well_sensor_readings for well metadata
    ↓
Detect lift_type from readings
    ↓
Query appropriate lift_type table
    (rod_pump_readings OR esp_readings OR gas_lift_readings)
    ↓
Query anomaly_suggestions from PostgreSQL
    ↓
Merge and return all data
```

## Module Details

### lift_type_db_operations.py

**Functions:**

- `insert_rod_pump_reading(well_id, timestamp, readings)` - Insert rod pump data
- `insert_esp_reading(well_id, timestamp, readings)` - Insert ESP data
- `insert_gas_lift_reading(well_id, timestamp, readings)` - Insert gas lift data
- `insert_lift_type_specific_reading(well_id, timestamp, lift_type, readings)` - Router function
- `get_rod_pump_history(well_id, limit)` - Fetch rod pump data
- `get_esp_history(well_id, limit)` - Fetch ESP data
- `get_gas_lift_history(well_id, limit)` - Fetch gas lift data
- `get_lift_type_specific_history(well_id, lift_type, limit)` - Router for fetch

**Features:**

- Dynamic column extraction from readings dict
- Type conversion (handles None, numpy types, floats)
- Automatic NULL assignment for form fields not provided
- Connection pooling with try/finally blocks
- Comprehensive error logging

### app.py Updates

**Modified Endpoints:**

- `/api/insert_reading` - Now also populates lift-type tables (line 510)
- `/api/well-history/<well_id>` - Now fetches from lift-type tables (lines 698-850)
- `/api/wells` - Now returns well_id AND lift_type (lines 854-872)

**New Functions:**

- Enhanced `get_well_history()` with dual-source retrieval
- Lift-type detection logic
- NaN/inf handling for JSON serialization

## Usage Examples

### Insert Rod Pump Data

```python
from lift_type_db_operations import insert_rod_pump_reading

well_id = "WELL_001"
timestamp = "2024-01-15 14:30:00"
readings = {
    'strokes_per_minute': 8.5,
    'torque': 750000.0,
    'polish_rod_load': 45000.0,
    'pump_fillage': 82.0,
    'tubing_pressure': 3200.0
}

insert_rod_pump_reading(well_id, timestamp, readings)
```

### Fetch ESP History

```python
from lift_type_db_operations import get_esp_history

well_id = "WELL_002"
df = get_esp_history(well_id, limit=10)
print(df.head())
```

### Query via API

```bash
# Get well list with lift types
curl http://localhost:5000/api/wells

# Response:
# {
#   "wells": [
#     {"well_id": "WELL_001", "lift_type": "Rod Pump"},
#     {"well_id": "WELL_002", "lift_type": "ESP"}
#   ]
# }

# Get well history
curl http://localhost:5000/api/well-history/WELL_001

# Response includes:
# {
#   "well_id": "WELL_001",
#   "lift_type": "Rod Pump",
#   "readings": [...],
#   "lift_type_specific_readings": [...],
#   "anomalies": [...]
# }
```

## Environment Variables Required

```
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_ROLE=your_role
```

## Troubleshooting

### Issue: Table doesn't exist error

**Solution:** Run migration script

```bash
python migrate_lift_type_tables.py
```

### Issue: Cannot import lift_type_db_operations

**Solution:** Ensure file is in same directory as app.py

```bash
ls -la lift_type_db_operations.py
```

### Issue: Connection refused

**Solution:** Verify Snowflake credentials and network connectivity

```bash
python test_lift_type_integration.py
```

### Issue: Fields showing as NULL

**Solution:** Only form fields (5 per type) are populated from web form. Additional attributes require direct data input or integration with sensor systems.

## Migration Path

### Before

- Single well_sensor_readings table for all data
- Form limited to 5 fields per well type
- No dedicated storage for additional attributes

### After

- well_sensor_readings remains (backward compatible)
- Three lift-type specific tables (rod_pump, esp, gas_lift)
- Each with 22-26 columns for comprehensive data
- Form still shows 5 fields, but 19-21 additional attributes can be stored

### Backward Compatibility

- All existing code still works
- Historical data remains in well_sensor_readings
- Lift-type tables are additive (INSERT-only pattern)
- Migration is non-destructive

## Performance

### Indexes

Each lift-type table has composite index on (well_id, timestamp DESC):

- Optimizes well-history queries
- Supports sorting by timestamp
- Typical query time: <100ms for 20 records

### Query Examples

```sql
-- Get latest 20 rod pump readings for a well
SELECT * FROM rod_pump_readings
WHERE well_id = 'WELL_001'
ORDER BY timestamp DESC
LIMIT 20;

-- Get ESP data for a time range
SELECT * FROM esp_readings
WHERE well_id = 'WELL_002'
  AND timestamp BETWEEN '2024-01-01' AND '2024-01-31'
ORDER BY timestamp DESC;

-- Count gas lift records per well
SELECT well_id, COUNT(*) as record_count
FROM gas_lift_readings
GROUP BY well_id;
```

## Next Steps

1. **Execute Migration**

   ```bash
   python migrate_lift_type_tables.py
   ```

2. **Test Integration**

   ```bash
   python test_lift_type_integration.py
   ```

3. **Submit Form Data**

   - Open web interface
   - Select well type (Rod Pump / ESP / Gas Lift)
   - Fill 5 form fields
   - Click submit

4. **Verify Data**

   ```bash
   curl http://localhost:5000/api/well-history/WELL_001
   ```

5. **Monitor Performance**
   - Check query times in logs
   - Monitor Snowflake warehouse usage
   - Track anomaly detection accuracy

## Support

For issues or questions:

1. Check logs: `tail -f app.log`
2. Test migration: `python migrate_lift_type_tables.py`
3. Verify Snowflake: `python test_lift_type_integration.py`
4. Check environment: `env | grep SNOWFLAKE`
