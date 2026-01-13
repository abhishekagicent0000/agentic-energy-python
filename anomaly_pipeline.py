"""
Anomaly Pipeline - ML-based Anomaly Detection using Isolation Forest and LSTM
Now integrated with Snowflake and compatible with anomaly_detector.py
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Optional, Tuple
from snowflake.connector import connect as snowflake_connect
from dotenv import load_dotenv

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Snowflake Configuration
SNOWFLAKE_CONFIG = {
    'user': os.getenv('SNOWFLAKE_USER'),
    'password': os.getenv('SNOWFLAKE_PASSWORD'),
    'account': os.getenv('SNOWFLAKE_ACCOUNT'),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
    'database': os.getenv('SNOWFLAKE_DATABASE'),
    'schema': os.getenv('SNOWFLAKE_SCHEMA'),
    'role': os.getenv('SNOWFLAKE_ROLE')
}

# Configuration
ROLLING_WINDOWS = [1, 3, 6, 12]  # Hours
ANOMALY_THRESHOLD = 0.75
WEIGHTS = {'rules': 0.4, 'iso': 0.3, 'lstm': 0.3}


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for sequence-based anomaly detection."""
    def __init__(self, input_dim, hidden_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        param = hidden.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
        decoded, _ = self.decoder(param)
        return decoded


def get_snowflake_connection():
    """Create and return a Snowflake connection."""
    try:
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        logger.info("✓ Connected to Snowflake")
        return conn
    except Exception as e:
        logger.error(f"✗ Failed to connect to Snowflake: {e}")
        raise


def load_sensor_data(well_id: str, limit: int = 5000) -> pd.DataFrame:
    """Load recent sensor data for a specific well from Snowflake."""
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        query = f"""
            SELECT 
                well_id, timestamp, 
                surface_pressure, tubing_pressure, casing_pressure, 
                motor_temp, wellhead_temp, motor_current
            FROM well_sensor_readings 
            WHERE well_id = '{well_id}'
            ORDER BY timestamp ASC 
            LIMIT {limit}
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0].lower() for desc in cursor.description]
        df = pd.DataFrame(rows, columns=cols)
        
        cursor.close()
        conn.close()
        
        return df
    except Exception as e:
        logger.error(f"Error loading sensor data for {well_id}: {e}")
        return pd.DataFrame()


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering: interpolation, rolling windows, rate of change.
    """
    if df.empty:
        return df
    
    df = df.sort_values('timestamp').copy()
    
    # Numeric sensor columns
    numeric_cols = ['surface_pressure', 'tubing_pressure', 'casing_pressure', 
                    'motor_temp', 'wellhead_temp', 'motor_current']
    
    # Ensure columns exist
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    
    # Interpolation for missing values
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill().fillna(0)
    
    # Feature Engineering
    features = df.copy()
    
    for col in numeric_cols:
        # Rate of Change
        features[f'{col}_pct_change'] = features[col].pct_change(fill_method=None).fillna(0)
        
        # Rolling Windows
        for window in ROLLING_WINDOWS:
            features[f'{col}_roll_mean_{window}h'] = features[col].rolling(window=window, min_periods=1).mean()
            features[f'{col}_roll_std_{window}h'] = features[col].rolling(window=window, min_periods=1).std().fillna(0)
    
    # Cross-sensor relationships
    if 'casing_pressure' in features.columns and 'tubing_pressure' in features.columns:
        features['casing_tubing_delta'] = features['casing_pressure'] - features['tubing_pressure']
    
    # Clean infinities
    features.replace([np.inf, -np.inf], 0, inplace=True)
    
    return features.fillna(0)


def train_isolation_forest(data: np.ndarray) -> Optional[Tuple]:
    """Train Isolation Forest on data and return model + scaler."""
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available; skipping Isolation Forest")
        return None
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_scaled)
        
        logger.info("✓ Isolation Forest trained")
        return (model, scaler)
    except Exception as e:
        logger.error(f"Error training Isolation Forest: {e}")
        return None


def predict_isolation_forest(model_scaler: Tuple, data: np.ndarray) -> np.ndarray:
    """Predict anomaly scores using trained Isolation Forest."""
    model, scaler = model_scaler
    X_scaled = scaler.transform(data)
    scores_raw = model.decision_function(X_scaled)
    # Sigmoid normalization
    scores = 1 / (1 + np.exp(scores_raw))
    return scores


def train_lstm_autoencoder(data: np.ndarray, input_dim: int, epochs: int = 5) -> Optional[Tuple]:
    """Train LSTM Autoencoder on data and return model + scaler + seq_len."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available; skipping LSTM")
        return None
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        SEQ_LEN = 10
        sequences = []
        for i in range(len(X_scaled) - SEQ_LEN):
            sequences.append(X_scaled[i:i+SEQ_LEN])
        
        if not sequences:
            logger.warning("Not enough data for LSTM sequences")
            return None
        
        X_tensor = torch.FloatTensor(np.array(sequences))
        
        model = LSTMAutoencoder(input_dim=input_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, X_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        logger.info(f"✓ LSTM Autoencoder trained for {epochs} epochs")
        return (model, scaler, SEQ_LEN)
    except Exception as e:
        logger.error(f"Error training LSTM: {e}")
        return None


def predict_lstm_autoencoder(model_info: Tuple, data: np.ndarray) -> np.ndarray:
    """Predict anomaly scores using trained LSTM."""
    model, scaler, seq_len = model_info
    X_scaled = scaler.transform(data)
    
    # Create sequences
    sequences = []
    for i in range(len(X_scaled) - seq_len):
        sequences.append(X_scaled[i:i+seq_len])
    
    if not sequences:
        return np.zeros(len(data))
    
    X_tensor = torch.FloatTensor(np.array(sequences))
    
    with torch.no_grad():
        reconstructed = model(X_tensor)
        mse = np.mean(np.power(X_tensor.numpy() - reconstructed.numpy(), 2), axis=(1, 2))
    
    # Pad to original length
    padded_mse = np.pad(mse, (seq_len, 0), 'constant', constant_values=0)
    
    # Normalize
    if padded_mse.max() > 0:
        scores = padded_mse / padded_mse.max()
    else:
        scores = padded_mse
    
    return scores


def run_ensemble_pipeline(well_id: str) -> dict:
    """
    Run complete ensemble anomaly detection pipeline for a well.
    
    Steps:
    1. Load sensor data from Snowflake
    2. Process features (interpolation, rolling windows, rate of change)
    3. Train Isolation Forest
    4. Train LSTM Autoencoder
    5. Compute ensemble scores
    6. Store results
    
    Returns:
        Summary dict with anomalies detected
    """
    logger.info(f"Running ensemble pipeline for {well_id}...")
    
    # 1. Load Data
    raw_df = load_sensor_data(well_id)
    if raw_df.empty:
        logger.warning(f"No data found for {well_id}")
        return {"well_id": well_id, "status": "no_data", "anomalies_detected": 0}
    
    # 2. Process Features
    features_df = process_features(raw_df)
    
    # 3. Prepare ML data
    ml_cols = [c for c in features_df.columns if c not in ['well_id', 'timestamp', 'id']]
    ml_data = features_df[ml_cols].fillna(0).values
    
    if ml_data.shape[0] < 20:
        logger.warning(f"Not enough data slices for {well_id}")
        return {"well_id": well_id, "status": "insufficient_data", "anomalies_detected": 0}
    
    anomalies_detected = 0
    
    # 4. Train Models
    iso_model_scaler = train_isolation_forest(ml_data)
    lstm_model_info = train_lstm_autoencoder(ml_data, input_dim=ml_data.shape[1])
    
    # 5. Compute scores for each timestamp
    iso_scores = predict_isolation_forest(iso_model_scaler, ml_data) if iso_model_scaler else np.zeros(len(ml_data))
    lstm_scores = predict_lstm_autoencoder(lstm_model_info, ml_data) if lstm_model_info else np.zeros(len(ml_data))
    
    # 6. Store results (optional)
    # For now, we just compute and log
    for idx in range(len(features_df)):
        s_iso = iso_scores[idx]
        s_lstm = lstm_scores[idx]
        
        # Ensemble score
        final_score = (WEIGHTS['iso'] * s_iso) + (WEIGHTS['lstm'] * s_lstm)
        
        if final_score > ANOMALY_THRESHOLD:
            anomalies_detected += 1
            logger.info(f"  Anomaly at index {idx}: score {final_score:.3f}")
    
    logger.info(f"✓ Pipeline complete for {well_id}; detected {anomalies_detected} anomalies")
    
    return {
        "well_id": well_id,
        "status": "success",
        "anomalies_detected": anomalies_detected,
        "total_records": len(features_df),
        "iso_available": iso_model_scaler is not None,
        "lstm_available": lstm_model_info is not None
    }


def main():
    """Run pipeline for all available wells."""
    try:
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT well_id FROM well_sensor_readings ORDER BY well_id LIMIT 5")
        wells = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        if not wells:
            logger.warning("No wells found in database")
            return
        
        logger.info(f"Processing {len(wells)} wells...")
        for well_id in wells:
            result = run_ensemble_pipeline(well_id)
            logger.info(f"Result: {result}")
    
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")


if __name__ == "__main__":
    main()


def load_data(well_id, limit=5000):
    conn = get_db_connection()
    # Load recent data for a specific well
    query = f"""
        SELECT * 
        FROM well_sensor_readings 
        WHERE well_id = '{well_id}' 
        ORDER BY timestamp ASC 
        LIMIT {limit}
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def process_features(df):
    """
    Implements Data Processing & Feature Engineering layers:
    - Cleaning/Interpolation
    - Rolling Windows
    - Rate of Change
    """
    df = df.sort_values('timestamp').copy()
    
    # 1. Handling Missing Values (Interpolation)
    numeric_cols = ['surface_pressure', 'tubing_pressure', 'casing_pressure', 'motor_temp', 'motor_current']
    
    # Ensure they exist (avoid KeyErrors)
    for c in numeric_cols:
        if c not in df.columns: df[c] = np.nan
        
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').ffill().bfill().fillna(0)
    
    # 2. Feature Engineering
    features = df.copy()
    
    for col in numeric_cols:
        # Rate of Change
        features[f'{col}_pct_change'] = features[col].pct_change(fill_method=None).fillna(0)
        
        # Rolling Windows
        for window in ROLLING_WINDOWS:
            # Assuming 1 row = 1 hour ideally, but using window size as rows for simplicity
            features[f'{col}_roll_mean_{window}h'] = features[col].rolling(window=window, min_periods=1).mean()
            features[f'{col}_roll_std_{window}h'] = features[col].rolling(window=window, min_periods=1).std().fillna(0)
            
    # Cross-sensor relationships
    features['casing_tubing_delta'] = features['casing_pressure'] - features['tubing_pressure']
    
    # 3. Clean Infinite/NaN
    features.replace([np.inf, -np.inf], 0, inplace=True)
    
    return features.fillna(0) # Careful with dropna on sensor data that might be genuinely missing for some well types

def train_isolation_forest(data):
    """
    Model B: Isolation Forest
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    # Score: -1 (anomaly) to 1 (normal). We map -1 to 1.0 (high anomaly score)
    scores_raw = model.decision_function(X)
    # Normalize decision function to 0-1 probability-like score
    # Lower decision function = more anomalous.
    # Simple sigmoid or min-max can work. Let's use simple inversion for demo.
    scores = 1 / (1 + np.exp(scores_raw)) # Sigmoid-ish
    
    return scores

def train_lstm_autoencoder(data, input_dim):
    """
    Model C: LSTM Autoencoder
    """
    # Prepare Tensors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Reshape for LSTM [samples, time_steps, features] -> Using window of 1 for point detection or N for sequence
    # For point anomaly with context, we usually slice. For simplicity here: Sequence Length = 1
    # Check architecture: "Detects pattern-based anomalies". 
    # Let's use sequence length 10
    SEQ_LEN = 10
    sequences = []
    for i in range(len(X_scaled) - SEQ_LEN):
        sequences.append(X_scaled[i:i+SEQ_LEN])
    
    if not sequences:
        return np.zeros(len(data))

    X_tensor = torch.FloatTensor(np.array(sequences))
    
    model = LSTMAutoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Fast training for demo
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        mse = np.mean(np.power(X_tensor.numpy() - reconstructed.numpy(), 2), axis=(1, 2))
    
    # Pad beginning
    padded_mse = np.pad(mse, (SEQ_LEN, 0), 'constant', constant_values=0)
    
    # Normalize MSE to 0-1
    if padded_mse.max() > 0:
        scores = padded_mse / padded_mse.max()
    else:
        scores = padded_mse
        
    return scores

def get_statistical_score(well_id, timestamp):
    """
    Model A: Statistical Rules (Look up previous results)
    """
    conn = get_db_connection()
    res = conn.execute(f"""
        SELECT anomaly_score FROM well_anomalies 
        WHERE well_id = '{well_id}' 
          AND timestamp = '{timestamp}' 
          AND model_name = 'Statistical_Rules'
    """).fetchone()
    conn.close()
    return res[0] if res else 0.0

def classify_anomaly(row_data, rule_score, iso_score, lstm_score):
    """
    Step 6: Anomaly Classification
    """
    cols = row_data.index
    reasons = []
    
    # Use raw rules if available
    if rule_score > 0:
        reasons.append("Rule Violation")
        
    # Heuristics based on sensor deviations
    if row_data.get('motor_current_pct_change', 0) > 0.5:
        reasons.append("ESP Motor Current Spike")
    if row_data.get('surface_pressure_pct_change', 0) < -0.2:
        reasons.append("Critical Pressure Drop")
    if row_data.get('motor_temp', 0) > 200: # Example threshold
        reasons.append("Overheating Event")
        
    if not reasons and (iso_score > 0.8 or lstm_score > 0.8):
        reasons.append("Unknown Pattern Anomaly")
        
    return ", ".join(reasons) if reasons else "General Anomaly"

def run_pipeline(well_id):
    logging.info(f"Running pipeline for {well_id}...")
    
    # 1. Ingest
    raw_df = load_data(well_id)
    if raw_df.empty:
        logging.warning("No data found.")
        return
        
    # 2 & 3. Process & Feature Engineering
    features_df = process_features(raw_df)
    
    # Select numeric features for ML
    ml_cols = [c for c in features_df.columns if c not in ['id', 'well_id', 'timestamp']]
    ml_data = features_df[ml_cols].fillna(0).values
    
    if ml_data.shape[0] < 10: # Need at least simple history for LSTM/ISO
        logging.warning(f"Not enough data slices for {well_id} after processing.")
        return
    
    # 4. Models
    # B. Isolation Forest
    iso_scores = train_isolation_forest(ml_data)
    
    # C. LSTM
    lstm_scores = train_lstm_autoencoder(ml_data, input_dim=ml_data.shape[1])
    
    # 5. Fusion
    conn = get_db_connection()
    cursor = conn.cursor()
    
    anomalies_to_insert = []
    
    for idx, (timestamp, row) in enumerate(features_df.iterrows()):
        # Align indices (approximate since iso/lstm alignment relies on array order)
        # Note: In production, rigorous index alignment is needed.
        if idx >= len(iso_scores) or idx >= len(lstm_scores): continue
        
        ts = row['timestamp']
        
        # A. Rule Answer
        rule_score = get_statistical_score(well_id, ts)
        
        s_iso = iso_scores[idx]
        s_lstm = lstm_scores[idx]
        
        final_score = (WEIGHTS['rules'] * rule_score) + \
                      (WEIGHTS['iso'] * s_iso) + \
                      (WEIGHTS['lstm'] * s_lstm)
                      
        if final_score > ANOMALY_THRESHOLD:
            anomaly_type = classify_anomaly(row, rule_score, s_iso, s_lstm)
            
            # Prepare JSONs
            raw_vals = json.dumps(row[ml_cols].to_dict()) # Storing processed for now as raw is in separate DF
            
            anomalies_to_insert.append((
                well_id, ts, anomaly_type, final_score, 
                raw_vals, 
                'Ensemble_Fusion', 'New'
            ))
            
    # 7. Storage
    if anomalies_to_insert:
        logging.info(f"Detected {len(anomalies_to_insert)} ensemble anomalies.")
        cursor.executemany("""
            INSERT INTO well_anomalies (well_id, timestamp, anomaly_type, anomaly_score, raw_values, model_name, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, anomalies_to_insert)
    else:
        logging.info("No ensemble anomalies detected.")
        
    conn.close()

def main():
    # Run for a few sample wells
    conn = get_db_connection()
    wells = conn.execute("SELECT DISTINCT well_id FROM well_sensor_readings LIMIT 3").fetchall()
    conn.close()
    
    for (w_id,) in wells:
        run_pipeline(w_id)

if __name__ == "__main__":
    main()
