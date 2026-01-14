"""
Unified Anomaly Detection Module
Integrates statistical rules, Isolation Forest, and LSTM for comprehensive anomaly detection.
Loads sensor configurations dynamically from Snowflake instead of hardcoding.
"""

import pandas as pd
import numpy as np
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

try:
    from dynamic_config import get_sensor_ranges, get_anomaly_rules
    DYNAMIC_CONFIG_AVAILABLE = True
except ImportError:
    DYNAMIC_CONFIG_AVAILABLE = False
    logging.warning("dynamic_config module not available; using fallback rules")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available; Isolation Forest disabled")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available; LSTM disabled")
    torch = None
    nn = None

from anomaly_config import (
    ANOMALY_THRESHOLD, 
    ML_ONLY_ANOMALY_THRESHOLD,
    WEIGHTS, 
    ROLLING_WINDOWS,
    WELL_TYPE_FEATURES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback for when PyTorch is not available
class FakeLSTMModule:
    """Dummy module when torch is not available."""
    pass

if nn is None:
    nn = FakeLSTMModule()
    nn.Module = object


if TORCH_AVAILABLE:
    class LSTMAutoencoder(nn.Module):
        """LSTM-based autoencoder for sequence anomaly detection."""
        def __init__(self, input_dim, hidden_dim=64):
            super(LSTMAutoencoder, self).__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
            
        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            param = hidden.repeat(x.shape[1], 1, 1).permute(1, 0, 2)
            decoded, _ = self.decoder(param)
            return decoded
else:
    class LSTMAutoencoder:
        """Dummy LSTM when PyTorch not available."""
        def __init__(self, *args, **kwargs):
            pass
        
        def forward(self, x):
            return x


# Dynamic rules loader function
def get_well_type_rules(lift_type):
    """Get sensor rules for a well type from dynamic config (Snowflake)."""
    if not DYNAMIC_CONFIG_AVAILABLE:
        raise ValueError("Dynamic config not available. Ensure Snowflake is configured and tables are created.")
    
    try:
        ranges = get_sensor_ranges(lift_type)
        rules = {}
        for field_name, range_info in ranges.items():
            rules[field_name] = {
                "min": range_info.get('min'),
                "max": range_info.get('max')
            }
        return rules
    except Exception as e:
        logger.error(f"Error loading dynamic rules for {lift_type}: {e}")
        raise




class AnomalyDetector:
    """Unified anomaly detection combining rules, Isolation Forest, and LSTM."""
    
    def __init__(self):
        """Initialize detector with rules from dynamic Snowflake config."""
        if not DYNAMIC_CONFIG_AVAILABLE:
            raise RuntimeError("Dynamic config (Snowflake) is required. Ensure tables are created with create_dynamic_config.py")
        
        self.iso_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        
        # Load pre-trained models from model_training.py
        self.trained_models = self._load_trained_models()
        
        logger.info(f"✓ Anomaly detector initialized (using dynamic Snowflake config)")
        if self.trained_models:
            logger.info(f"✓ Loaded trained models: {', '.join(self.trained_models.keys())}")
    
    def _load_trained_models(self) -> Dict:
        """Load pre-trained models from models/ directory."""
        trained_models = {}
        models_dir = 'models'
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found at {models_dir}. Run: python model_training.py")
            return trained_models
        
        # Load Isolation Forest models
        well_types = ['rod_pump', 'esp', 'gas_lift']
        for well_type in well_types:
            iso_file = os.path.join(models_dir, f'isolation_forest_{well_type}.joblib')
            stat_file = os.path.join(models_dir, f'statistical_model_{well_type}.joblib')
            
            if os.path.exists(iso_file):
                try:
                    trained_models[f'iso_{well_type}'] = joblib.load(iso_file)
                    logger.info(f"✓ Loaded isolation forest model for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load {iso_file}: {e}")
            
            if os.path.exists(stat_file):
                try:
                    trained_models[f'stat_{well_type}'] = joblib.load(stat_file)
                    logger.info(f"✓ Loaded statistical model for {well_type}")
                except Exception as e:
                    logger.error(f"Failed to load {stat_file}: {e}")
        
        return trained_models
    

    

    

    

    
    def check_anomaly(self, readings: Dict, well_id: str = None, lift_type: str = None) -> Dict:
        """
        Check for anomalies using statistical rules and trained ML models.
        
        Args:
            readings: Dict of sensor values {field: value}
            well_id: Well identifier (optional, for logging)
            lift_type: Well type (Rod Pump/ESP/Gas Lift) for context-aware rules
        
        Returns:
            Dict with is_anomaly, anomaly_score, violations, summary, detection_method
        """
        violations = []
        rule_anomaly_score = 0.0
        ml_anomaly_score = 0.0
        
        # Use lift-type aware rules from dynamic config
        if lift_type:
            active_rules = get_well_type_rules(lift_type)
        else:
            active_rules = self.rules
        
        # --- RULE-BASED DETECTION ---
        for field, value in readings.items():
            if field not in active_rules or value is None:
                continue
            
            rule = active_rules[field]
            
            # Skip if rule is None (e.g., Gas Lift has no motor_current)
            if rule is None or not isinstance(rule, dict):
                continue
            
            rule_min = rule.get("min")
            rule_max = rule.get("max")
            
            if rule_min is None or rule_max is None:
                continue
            
            if value < rule_min or value > rule_max:
                # Calculate deviation percentage
                range_width = rule_max - rule_min
                if range_width == 0:
                    deviation_pct = 0
                else:
                    if value < rule_min:
                        deviation_pct = ((rule_min - value) / range_width) * 100
                    else:
                        deviation_pct = ((value - rule_max) / range_width) * 100
                
                # Get unit from sensor ranges
                sensor_unit = ""
                if lift_type:
                    sensor_ranges = get_sensor_ranges(lift_type)
                    if field in sensor_ranges:
                        sensor_unit = sensor_ranges[field].get('unit', "")
                
                # Format violation message including units to give the downstream AI explicit context
                try:
                    min_str = f"{rule_min:.2f}" if isinstance(rule_min, float) else str(rule_min)
                except Exception:
                    min_str = str(rule_min)
                try:
                    max_str = f"{rule_max:.2f}" if isinstance(rule_max, float) else str(rule_max)
                except Exception:
                    max_str = str(rule_max)
                try:
                    val_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                except Exception:
                    val_str = str(value)

                unit_str = f" {sensor_unit}" if sensor_unit else ""
                violation_text = f"Out of range. Expected {min_str}{unit_str} - {max_str}{unit_str}, got {val_str}{unit_str} ({deviation_pct:.1f}% deviation)"

                # Log exact rule values used for debugging mismatched range issues
                logger.info(f"[AnomalyDetector] lift_type={lift_type} field={field} min={rule_min} max={rule_max} unit={sensor_unit} value={value}")

                violations.append({
                    "field": field,
                    "value": value,
                    "min": rule_min,
                    "max": rule_max,
                    "unit": sensor_unit,
                    "deviation_pct": round(deviation_pct, 2),
                    "violation": violation_text
                })
                # Deviation-weighted contribution: base + extra proportional to deviation
                try:
                    dev_factor = float(deviation_pct) / 100.0
                except Exception:
                    dev_factor = 0.0

                # Increased-sensitivity contribution for a rule violation:
                # - larger base contribution so single violations matter more
                # - stronger scaling of deviation (dev_factor) and higher cap for extra
                base_contrib = 0.35
                extra = min(0.7, dev_factor * 0.9)
                contrib = base_contrib + extra

                rule_anomaly_score += contrib
        
        rule_anomaly_score = min(1.0, rule_anomaly_score)
        
        # --- ML-BASED DETECTION (using trained models) ---
        detection_method = "statistical_rules"
        if lift_type and self.trained_models:
            ml_anomaly_score = self._detect_anomaly_with_trained_models(readings, lift_type)
            if ml_anomaly_score > 0.5:
                detection_method = "statistical_rules + trained_ml_model"
        
        # Combine scores using configured weights (favor rule signal by default)
        try:
            rules_w = float(WEIGHTS.get('rules', 0.4))
        except Exception:
            rules_w = 0.4
        ml_w = max(0.0, 1.0 - rules_w)

        combined_score = min(1.0, rules_w * rule_anomaly_score + ml_w * ml_anomaly_score)

        # Log scores for debugging and to clarify why an anomaly was flagged
        logger.info(f"[AnomalyDetector] rule_score={rule_anomaly_score:.3f} ml_score={ml_anomaly_score:.3f} combined_score={combined_score:.3f}")

        # Determine if anomaly. If there are explicit rule violations, flag immediately.
        # If there are NO rule violations, require a higher ML-only threshold to avoid false positives
        if len(violations) > 0:
            is_anomaly = True
        else:
            is_anomaly = ml_anomaly_score >= ML_ONLY_ANOMALY_THRESHOLD

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(combined_score, 3),
            "rule_score": round(rule_anomaly_score, 3),
            "ml_score": round(ml_anomaly_score, 3),
            "violations": violations,
            "summary": f"{len(violations)} rule violation(s) detected" if violations else "No violations detected",
            "detection_method": detection_method
        }
    
    def _detect_anomaly_with_trained_models(self, readings: Dict, lift_type: str) -> float:
        """Use trained ML models to detect anomalies."""
        try:
            well_type_clean = lift_type.lower().replace(" ", "_")
            iso_key = f'iso_{well_type_clean}'
            stat_key = f'stat_{well_type_clean}'
            
            if iso_key not in self.trained_models and stat_key not in self.trained_models:
                return 0.0
            
            # Get features for this well type
            features = self._get_features_for_well_type(lift_type)
            if not features:
                return 0.0
            
            # Extract values in correct order
            values = np.array([[readings.get(f, 0.0) for f in features]])
            
            anomaly_scores = []
            
            # Isolation Forest
            if iso_key in self.trained_models:
                model_data = self.trained_models[iso_key]
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    scaler = model_data['scaler']
                    values_scaled = scaler.transform(values)
                    score = model.decision_function(values_scaled)[0]
                    # Normalize to 0-1
                    normalized_score = 1 / (1 + np.exp(score))
                    anomaly_scores.append(normalized_score)
            
            # Statistical model
            if stat_key in self.trained_models:
                stat_model = self.trained_models[stat_key]
                if isinstance(stat_model, dict) and 'mean' in stat_model:
                    mean = np.array(stat_model['mean'])
                    std = np.array(stat_model['std'])
                    z_scores = np.abs((values[0] - mean) / (std + 1e-8))
                    max_z = np.max(z_scores)
                    # Flag if > 3 standard deviations
                    stat_anomaly = min(1.0, max_z / 3.0)
                    anomaly_scores.append(stat_anomaly)
            
            # Return average anomaly score from all models
            if anomaly_scores:
                return np.mean(anomaly_scores)
            
            return 0.0
        
        except Exception as e:
            logger.warning(f"Error in ML-based detection: {e}")
            return 0.0
    
    def _get_features_for_well_type(self, lift_type: str) -> List[str]:
        """Get feature list for the well type."""
        return WELL_TYPE_FEATURES.get(lift_type, [])
    
    
    def train_isolation_forest(self, data: np.ndarray) -> Optional[IsolationForest]:
        """Train Isolation Forest on historical sensor data."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available; skipping Isolation Forest training")
            return None
        
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)
            
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X_scaled)
            
            self.iso_model = (model, scaler)
            logger.info("✓ Isolation Forest trained")
            return model
        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}")
            return None
    
    def predict_isolation_forest(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Predict anomaly scores using Isolation Forest."""
        if not self.iso_model or not SKLEARN_AVAILABLE:
            return None
        
        try:
            model, scaler = self.iso_model
            X_scaled = scaler.transform(data)
            scores_raw = model.decision_function(X_scaled)
            # Normalize to 0-1 using sigmoid
            scores = 1 / (1 + np.exp(scores_raw))
            return scores
        except Exception as e:
            logger.error(f"Error predicting with Isolation Forest: {e}")
            return None
    
    def train_lstm_autoencoder(self, data: np.ndarray, epochs: int = 5) -> Optional[Tuple]:
        """Train LSTM Autoencoder on historical sensor data."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available; skipping LSTM training")
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
            input_dim = X_scaled.shape[1]
            
            model = LSTMAutoencoder(input_dim=input_dim)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Train
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, X_tensor)
                loss.backward()
                optimizer.step()
            
            model.eval()
            self.lstm_model = (model, scaler, SEQ_LEN)
            logger.info(f"✓ LSTM Autoencoder trained for {epochs} epochs")
            return (model, scaler, SEQ_LEN)
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return None
    
    def predict_lstm_autoencoder(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Predict anomaly scores using LSTM Autoencoder."""
        if not self.lstm_model or not TORCH_AVAILABLE:
            return None
        
        try:
            model, scaler, seq_len = self.lstm_model
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
            
            # Pad to match original data length
            padded_mse = np.pad(mse, (seq_len, 0), 'constant', constant_values=0)
            
            # Normalize to 0-1
            if padded_mse.max() > 0:
                scores = padded_mse / padded_mse.max()
            else:
                scores = padded_mse
            
            return scores
        except Exception as e:
            logger.error(f"Error predicting with LSTM: {e}")
            return None
    
    def ensemble_score(self, readings: Dict, historical_data: Optional[np.ndarray] = None,
                      rule_score: float = None, iso_score: float = None, 
                      lstm_score: float = None) -> Dict:
        """
        Compute ensemble anomaly score combining rules, ISO, and LSTM.
        
        Args:
            readings: Current sensor readings
            historical_data: Optional array of historical readings for ML models
            rule_score, iso_score, lstm_score: Pre-computed scores (optional)
        
        Returns:
            Dict with final_score, method_scores, and ensemble details
        """
        # Rule score (always computed)
        if rule_score is None:
            rule_result = self.check_anomaly(readings)
            rule_score = rule_result['anomaly_score']
        
        # ISO score (optional)
        if iso_score is None and historical_data is not None:
            iso_scores = self.predict_isolation_forest(historical_data)
            iso_score = float(iso_scores[-1]) if iso_scores is not None else 0.0
        
        # LSTM score (optional)
        if lstm_score is None and historical_data is not None:
            lstm_scores = self.predict_lstm_autoencoder(historical_data)
            lstm_score = float(lstm_scores[-1]) if lstm_scores is not None else 0.0
        
        iso_score = iso_score or 0.0
        lstm_score = lstm_score or 0.0
        
        # Weighted ensemble
        final_score = (
            WEIGHTS['rules'] * rule_score +
            WEIGHTS['iso'] * iso_score +
            WEIGHTS['lstm'] * lstm_score
        )
        
        return {
            "final_score": final_score,
            "method_scores": {
                "rules": rule_score,
                "isolation_forest": iso_score,
                "lstm_autoencoder": lstm_score
            },
            "weights": WEIGHTS,
            "is_anomaly": final_score > ANOMALY_THRESHOLD,
            "threshold": ANOMALY_THRESHOLD
        }


# Singleton instance
_detector = None

def get_detector() -> AnomalyDetector:
    """Get or create singleton anomaly detector."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector

def check_anomaly(readings: Dict, well_id: str = None, lift_type: str = None) -> Dict:
    """Convenience function: check for anomalies using default detector."""
    detector = get_detector()
    return detector.check_anomaly(readings, well_id, lift_type)

def ensemble_score(readings: Dict, historical_data: Optional[np.ndarray] = None) -> Dict:
    """Convenience function: compute ensemble score using default detector."""
    detector = get_detector()
    return detector.ensemble_score(readings, historical_data)
