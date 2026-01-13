-- Main sensor readings table (references to specific readings)
CREATE TABLE IF NOT EXISTS well_sensor_readings (
    well_id VARCHAR(50) NOT NULL,
    lift_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (well_id, lift_type, timestamp)
);

-- ==================== ROD PUMP TABLE ====================
CREATE TABLE IF NOT EXISTS rod_pump_readings (
    well_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Form fields (displayed - 5 fields)
    strokes_per_minute FLOAT,
    torque FLOAT,
    polish_rod_load FLOAT,
    pump_fillage FLOAT,
    tubing_pressure FLOAT,
    
    -- Additional Rod Pump attributes (stored as NULL initially)
    surface_stroke_length FLOAT,
    downhole_gross_stroke_length FLOAT,
    downhole_net_stroke_length FLOAT,
    runtime FLOAT,
    cycles_per_day FLOAT,
    structural_load FLOAT,
    inferred_production FLOAT,
    pump_intake_pressure FLOAT,
    pumping_unit_type VARCHAR(255),
    rod_string VARCHAR(1000),
    dry_rod_weight FLOAT,
    buoyant_rod_weight FLOAT,
    pump_friction FLOAT,
    pump_diameter FLOAT,
    barrel_length FLOAT,
    pump_depth FLOAT,
    controller_mode VARCHAR(255),
    idle_time_setpoint FLOAT,
    vfd_settings VARCHAR(2000),
    alarm VARCHAR(2000),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (well_id, timestamp),
    FOREIGN KEY (well_id) REFERENCES well_sensor_readings(well_id)
);

-- ==================== ESP TABLE ====================
CREATE TABLE IF NOT EXISTS esp_readings (
    well_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Form fields (displayed - 5 fields)
    motor_temp FLOAT,
    motor_current FLOAT,
    discharge_pressure FLOAT,
    pump_intake_pressure FLOAT,
    motor_voltage FLOAT,
    
    -- Additional ESP attributes (stored as NULL initially)
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
    run_stop_status VARCHAR(50),
    set_frequency FLOAT,
    acceleration_ramp_time FLOAT,
    cycle FLOAT,
    run_mode VARCHAR(255),
    alarm VARCHAR(2000),
    flags VARCHAR(2000),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (well_id, timestamp),
    FOREIGN KEY (well_id) REFERENCES well_sensor_readings(well_id)
);

-- ==================== GAS LIFT TABLE ====================
CREATE TABLE IF NOT EXISTS gas_lift_readings (
    well_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Form fields (displayed - 5 fields)
    injection_rate FLOAT,
    injection_temperature FLOAT,
    bottomhole_pressure FLOAT,
    injection_pressure FLOAT,
    cycle_time FLOAT,
    
    -- Additional Gas Lift attributes (stored as NULL initially)
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
    alarm VARCHAR(2000),
    flags VARCHAR(2000),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (well_id, timestamp),
    FOREIGN KEY (well_id) REFERENCES well_sensor_readings(well_id)
);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_rod_pump_well_time ON rod_pump_readings(well_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_esp_well_time ON esp_readings(well_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_gas_lift_well_time ON gas_lift_readings(well_id, timestamp DESC);
