"""
Anomaly Detection Configuration
===============================
Central configuration for thresholds, weights, well types, and feature definitions.
"""

# Detection Thresholds
ANOMALY_THRESHOLD = 0.75
ML_ONLY_ANOMALY_THRESHOLD = 0.85

# Model Ensemble Weights
WEIGHTS = {
    'rules': 0.4, 
    'iso': 0.3, 
    'lstm': 0.3
}

# Rolling Window Sizes (Hours)
ROLLING_WINDOWS = [1, 3, 6, 12]

# Well Types
WELL_TYPES = ['Rod Pump', 'ESP', 'Gas Lift']

# Feature Definitions per Well Type
WELL_TYPE_FEATURES = {
    'Rod Pump': [
        'strokes_per_minute', 'torque', 'polish_rod_load', 
        'pump_fillage', 'tubing_pressure'
    ],
    'ESP': [
        'motor_temp', 'motor_current', 'discharge_pressure',
        'pump_intake_pressure', 'motor_voltage'
    ],
    'Gas Lift': [
        'injection_rate', 'injection_temperature', 'bottomhole_pressure',
        'injection_pressure', 'cycle_time'
    ]
}

# Isolation Forest Config
CONTAMINATION_RATIO = 0.1

# Data Quality Configuration
CRITICAL_SENSORS = {
    'Rod Pump': ['tubing_pressure', 'strokes_per_minute'],
    'ESP': ['motor_current', 'motor_temp'],
    'Gas Lift': ['injection_rate', 'injection_pressure']
}
