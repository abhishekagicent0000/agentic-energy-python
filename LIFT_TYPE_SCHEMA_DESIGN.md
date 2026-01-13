# Lift-Type Specific Database Schema Implementation

## Overview

Implemented **separate lift-type tables** for Rod Pump, ESP, and Gas Lift sensor readings with support for all attributes while only displaying 5 in the form.

## Database Design

### Tables Created:

1. **rod_pump_readings** - 24 total columns (5 form fields + 19 additional)
2. **esp_readings** - 26 total columns (5 form fields + 21 additional)
3. **gas_lift_readings** - 22 total columns (5 form fields + 17 additional)

### Key Features:

✅ **Type-Safe**: Each table has properly typed columns for specific well type  
✅ **Complete Coverage**: All sensor attributes documented and stored  
✅ **Flexible**: Additional fields initialized as NULL, can be populated later  
✅ **Indexed**: Optimal query performance with timestamp + well_id indexes  
✅ **Organized**: All data for a well type in one table  
✅ **Scalable**: Easy to add new attributes or sensors

## Form vs Database

| Well Type    | Form Fields (5)                                                                     | Total DB Fields | Additional Fields (NULL initially)                                                                    |
| ------------ | ----------------------------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------- |
| **Rod Pump** | Strokes/Min, Torque, Load, Fillage, Pressure                                        | 24              | Surface stroke, downhole stroke, runtime, cycles, structural load, production, etc.                   |
| **ESP**      | Motor Temp, Motor Current, Discharge Pressure, Intake Pressure, Voltage             | 26              | Fluid temp, vibration, discharge temp, flow rate, tubing/casing pressure, VSD temp, motor speed, etc. |
| **Gas Lift** | Injection Rate, Injection Temp, Bottomhole Pressure, Injection Pressure, Cycle Time | 22              | Bottomhole temp, plunger metrics, shut-in times, velocity, differential pressure, etc.                |

## Implementation

### Code Structure:

1. **schema_lift_type_tables.sql** - Complete Snowflake DDL with all attributes
2. **lift_type_db_operations.py** - Helper functions for lift-type specific inserts
3. **app.py** - Updated to call lift-type specific insert after main insert

### Data Flow:

```
User Form (5 fields)
        ↓
Flask API (/api/insert-reading)
        ↓
Insert to well_sensor_readings (main table)
        ↓
Insert to rod_pump_readings / esp_readings / gas_lift_readings
        (5 values populated, others as NULL)
```

### Usage in Code:

```python
from lift_type_db_operations import insert_lift_type_specific_reading

# After inserting to main table:
insert_lift_type_specific_reading(
    well_id="Well_001",
    timestamp_str="2024-01-13 10:00:00",
    lift_type="Rod Pump",  # or "ESP" or "Gas Lift"
    readings={
        'strokes_per_minute': 6.5,
        'torque': 900000,
        'polish_rod_load': 50000,
        'pump_fillage': 80,
        'tubing_pressure': 2500,
        # ... other fields optional (NULL if not provided)
    }
)
```

## Advantages of This Approach

✅ **Clear Separation**: Rod Pump, ESP, and Gas Lift data in dedicated tables  
✅ **Query Performance**: Optimized indexes and schema per well type  
✅ **Data Integrity**: Type-specific columns prevent invalid data  
✅ **Future Ready**: Can add lift-type specific validation, alerts, or calculations  
✅ **Backward Compatible**: Form still shows 5 fields, backend handles all attributes  
✅ **Easy Migration**: Can populate additional fields without changing form

## Future Enhancements

1. **Expand Form**: Add more fields to the form to capture additional attributes
2. **Historical Analysis**: Query historical data across all attributes
3. **Predictive Maintenance**: Use additional fields for ML models
4. **Custom Alerts**: Set thresholds for any attribute in the table
5. **Data Export**: Export complete well type data for analysis

## Queries

### Get Complete Rod Pump Data for a Well:

```sql
SELECT * FROM rod_pump_readings
WHERE well_id = 'Well_001'
AND timestamp > NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;
```

### Get All Anomalies with Lift-Type Data:

```sql
SELECT
    a.well_id, a.timestamp, a.alert_title, a.severity,
    r.strokes_per_minute, r.torque, r.pump_fillage
FROM anomaly_suggestions a
JOIN rod_pump_readings r ON a.well_id = r.well_id AND a.timestamp = r.timestamp
WHERE a.lift_type = 'Rod Pump'
ORDER BY a.timestamp DESC;
```

## Files Modified/Created

- ✅ `schema_lift_type_tables.sql` (NEW) - Complete database schema
- ✅ `lift_type_db_operations.py` (NEW) - Helper functions for inserts
- ✅ `app.py` (UPDATED) - Added lift-type specific inserts
- ✅ Form validation (ALREADY DONE) - Min/max values match detector ranges
