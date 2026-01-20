-- Table 1: High-Frequency Sensor Data
CREATE SEQUENCE IF NOT EXISTS seq_sensor_id;
CREATE TABLE IF NOT EXISTS well_sensor_readings (
    id BIGINT DEFAULT nextval('seq_sensor_id'), 
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    surface_pressure DOUBLE,
    tubing_pressure DOUBLE,
    casing_pressure DOUBLE,
    motor_temp DOUBLE,
    wellhead_temp DOUBLE,
    motor_current DOUBLE,
    lift_type VARCHAR,
    PRIMARY KEY (id)
);

-- Table 2: Daily Production Data
CREATE SEQUENCE IF NOT EXISTS seq_production_id;
CREATE TABLE IF NOT EXISTS well_daily_production (
    id BIGINT DEFAULT nextval('seq_production_id'),
    well_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    oil_volume DOUBLE,
    gas_volume DOUBLE,
    water_volume DOUBLE,
    lift_type VARCHAR,
    PRIMARY KEY (id)
);

-- Table 3: Anomalies
CREATE SEQUENCE IF NOT EXISTS seq_anomalies_id;
CREATE TABLE IF NOT EXISTS well_anomalies (
    id BIGINT DEFAULT nextval('seq_anomalies_id'),
    well_id VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    anomaly_type VARCHAR,
    anomaly_score DOUBLE,
    raw_values JSON,
    processed_features JSON,
    model_name VARCHAR,
    status VARCHAR,
    PRIMARY KEY (id)
);

-- Table 4: Anomaly Reviews (Matches anomaly_review_service.py)
CREATE TABLE IF NOT EXISTS anomaly_review (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    well_id         VARCHAR(100)    NOT NULL,
    event_date      DATE            NOT NULL,
    detected_at     TIMESTAMP       NOT NULL,
    anomaly_code    VARCHAR(100)    NOT NULL,
    category        VARCHAR(50),
    severity        VARCHAR(20),
    title           VARCHAR(200),
    ui_text         JSONB,
    chart_data      JSONB,
    impact_value    FLOAT,
    impact_unit     VARCHAR(50),
    status          VARCHAR(50),
    CONSTRAINT anomaly_review_unique_business_key 
        UNIQUE (well_id, anomaly_code, event_date)
);