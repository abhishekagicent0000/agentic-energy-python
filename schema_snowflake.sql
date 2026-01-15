-- Snowflake Schema for At-Risk Assets Project

-- Create Sequences first
CREATE SEQUENCE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.seq_sensor_id;
CREATE SEQUENCE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.seq_production_id;
CREATE SEQUENCE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.seq_anomalies_id;

-- Table 1: High-Frequency Sensor Data
CREATE TABLE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.well_sensor_readings (
    id BIGINT DEFAULT (AT_RISK_ASSETS.SENSOR_DATA.seq_sensor_id.nextval),
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    surface_pressure FLOAT,
    tubing_pressure FLOAT,
    casing_pressure FLOAT,
    motor_temp FLOAT,
    wellhead_temp FLOAT,
    motor_current FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (id)
);

-- Table 2: Daily Production Data
CREATE TABLE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.well_daily_production (
    id BIGINT DEFAULT (AT_RISK_ASSETS.SENSOR_DATA.seq_production_id.nextval),
    well_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    oil_volume FLOAT,
    gas_volume FLOAT,
    water_volume FLOAT,
    lift_type VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (id)
);

-- Table 3: Anomalies
CREATE TABLE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.well_anomalies (
    id BIGINT DEFAULT (AT_RISK_ASSETS.SENSOR_DATA.seq_anomalies_id.nextval),
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    anomaly_type VARCHAR,
    anomaly_score FLOAT,
    raw_values VARIANT,
    processed_features VARIANT,
    model_name VARCHAR,
    status VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (id)
);

-- Table 4: Anomaly Reviews
CREATE TABLE IF NOT EXISTS AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW (
    well_id VARCHAR,
    event_date DATE,
    detected_at TIMESTAMP,
    anomaly_code VARCHAR,
    category VARCHAR,
    severity VARCHAR,
    title VARCHAR,
    ui_text VARIANT,
    impact_value FLOAT,
    impact_unit VARCHAR,
    chart_data VARIANT,
    status VARCHAR,
    PRIMARY KEY (well_id, anomaly_code, event_date)
);
