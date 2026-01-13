"""
Unified Anomaly Detection Module
Integrates statistical rules, Isolation Forest, and LSTM for comprehensive anomaly detection.
Replaces hardcoded RULES in app.py with data-driven detection from Excel.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EXCEL_FILE = 'data types and ranges.xlsx'
ROLLING_WINDOWS = [1, 3, 6, 12]  # Hours
ANOMALY_THRESHOLD = 0.75
WEIGHTS = {'rules': 0.4, 'iso': 0.3, 'lstm': 0.3}

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


# Well-type aware rules (based on actual Excel data)
WELL_TYPE_RULES = {
    "Rod Pump": {
        "strokes_per_minute": {"min": 5, "max": 25},  # Realistic range
        "torque": {"min": 100, "max": 5000},  # Realistic range
        "polish_rod_load": {"min": 500, "max": 10000},  # Realistic range
        "pump_fillage": {"min": 20, "max": 100},  # 20-100%
        "tubing_pressure": {"min": 100, "max": 3000},  # 100-3000 PSI
    },
    "ESP": {
        "motor_temp": {"min": 50, "max": 150},  # 50-150°C
        "motor_current": {"min": 50, "max": 200},  # 50-200A
        "discharge_pressure": {"min": 1000, "max": 4500},  # 1000-4500 PSI
        "pump_intake_pressure": {"min": 100, "max": 1000},  # 100-1000 PSI
        "motor_voltage": {"min": 400, "max": 480},  # 400-480V
    },
    "Gas Lift": {
        "injection_rate": {"min": 2, "max": 20},  # 2-20 MMscf/d
        "injection_temperature": {"min": 20, "max": 80},  # 20-80°C
        "bottomhole_pressure": {"min": 1000, "max": 3500},  # 1000-3500 PSI
        "injection_pressure": {"min": 500, "max": 2000},  # 500-2000 PSI
        "cycle_time": {"min": 10, "max": 120},  # 10-120 seconds
    }
}

UNITS_MAP = {
    "strokes_per_minute": "SPM",
    "torque": "in-lbs",
    "polish_rod_load": "lbs",
    "pump_fillage": "%",
    "tubing_pressure": "psi",
    "motor_temp": "°F",
    "motor_current": "A",
    "discharge_pressure": "psi",
    "pump_intake_pressure": "psi",
    "motor_voltage": "V",
    "injection_rate": "scf/d",
    "injection_temperature": "°F",
    "bottomhole_pressure": "psi",
    "injection_pressure": "psi",
    "cycle_time": "minutes",
    "oil_volume": "bbl",
    "gas_volume": "mcf",
    "water_volume": "bbl",
}


class AnomalyDetector:
    """Unified anomaly detection combining rules, Isolation Forest, and LSTM."""
    
    def __init__(self, excel_file: str = EXCEL_FILE):
        """Initialize detector with rules from Excel or fallback defaults."""
        self.rules = self._load_rules_from_excel(excel_file)
        self.iso_model = None
        self.lstm_model = None
        self.lstm_scaler = None
        
        # Load pre-trained models from model_training.py
        self.trained_models = self._load_trained_models()
        
        logger.info(f"✓ Anomaly detector initialized with {len(self.rules)} rules")
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
    
    def _load_rules_from_excel(self, excel_file: str) -> Dict:
        """Load rules from Excel file, fallback to defaults if file not found."""
        rules = {}
        
        try:
            if not os.path.exists(excel_file):
                logger.warning(f"Excel file {excel_file} not found; using default rules")
                return self._get_default_rules()
            
            df = pd.read_excel(excel_file, header=None, engine='openpyxl')
            
            # Parse header rows
            categories = df.iloc[1].ffill() if len(df) > 1 else []
            data_names = df.iloc[2].fillna("Unknown") if len(df) > 2 else []
            
            # Find range row
            range_row_idx = None
            try:
                range_row_idx = df[df[0].astype(str).str.lower().str.contains("range", na=False)].index[0]
            except (IndexError, TypeError):
                logger.warning("Could not find 'range' row in Excel; using defaults")
                return self._get_default_rules()
            
            ranges = df.iloc[range_row_idx]
            
            # Extract rules
            for i in range(1, len(data_names)):
                raw_name = str(data_names[i])
                rng_str = str(ranges[i]) if i < len(ranges) else ""
                parsed = self._parse_range(rng_str)
                
                if parsed:
                    db_col = self._normalize_column_name(raw_name)
                    if db_col:
                        rules[db_col] = {
                            "min": parsed['min'],
                            "max": parsed['max'],
                            "norm_min": parsed.get('norm_min', parsed['min']),
                            "norm_max": parsed.get('norm_max', parsed['max']),
                        }
            
            if rules:
                logger.info(f"✓ Loaded {len(rules)} rules from Excel: {', '.join(rules.keys())}")
                return rules
            else:
                logger.warning("No rules parsed from Excel; using defaults")
                return self._get_default_rules()
                
        except Exception as e:
            logger.error(f"Error loading Excel rules: {e}; using defaults")
            return self._get_default_rules()
    
    def _parse_range(self, range_str: str) -> Optional[Dict]:
        """Parse range string like '100-200' or '100-200 150-180' into min/max."""
        if not isinstance(range_str, str):
            return None
        
        range_str = range_str.strip()
        pairs = re.findall(r"([\d,.]+)\s*-\s*([\d,.]+)", range_str)
        
        if not pairs:
            return None
        
        def to_float(s):
            try:
                return float(s.replace(',', ''))
            except:
                return 0.0
        
        vals = [(to_float(a), to_float(b)) for a, b in pairs]
        
        if not vals:
            return None
        
        abs_min, abs_max = vals[0]
        norm_min, norm_max = vals[1] if len(vals) > 1 else (abs_min, abs_max)
        
        # Ensure Min < Max
        if abs_min > abs_max:
            abs_min, abs_max = abs_max, abs_min
        if norm_min > norm_max:
            norm_min, norm_max = norm_max, norm_min
        
        return {
            "min": abs_min,
            "max": abs_max,
            "norm_min": norm_min,
            "norm_max": norm_max
        }
    
    def _normalize_column_name(self, raw_name: str) -> Optional[str]:
        """Map Excel column names to database column names."""
        if not isinstance(raw_name, str):
            return None
        
        raw_name = str(raw_name).strip()
        
        if "Tubing" in raw_name and "Pressure" in raw_name:
            return "tubing_pressure"
        elif "Casing" in raw_name and "Pressure" in raw_name:
            return "casing_pressure"
        elif "Surface" in raw_name and "Pressure" in raw_name:
            return "surface_pressure"
        elif "Pump Intake" in raw_name and "Pressure" in raw_name:
            return "surface_pressure"
        elif "Motor" in raw_name and "Temp" in raw_name:
            return "motor_temp"
        elif "Discharge" in raw_name and "Temp" in raw_name:
            return "wellhead_temp"
        elif "Intake" in raw_name and "Temp" in raw_name:
            return "wellhead_temp"
        elif "Fluid" in raw_name and "Temp" in raw_name:
            return "wellhead_temp"
        elif "Motor" in raw_name and "Current" in raw_name:
            return "motor_current"
        
        return None
    
    def _get_default_rules(self) -> Dict:
        """Return default hardcoded rules if Excel loading fails."""
        return {
            "surface_pressure": {"min": 100, "max": 400, "norm_min": 100, "norm_max": 400},
            "tubing_pressure": {"min": 200, "max": 600, "norm_min": 200, "norm_max": 600},
            "casing_pressure": {"min": 300, "max": 800, "norm_min": 300, "norm_max": 800},
            "motor_temp": {"min": 100, "max": 250, "norm_min": 100, "norm_max": 250},
            "wellhead_temp": {"min": 50, "max": 200, "norm_min": 50, "norm_max": 200},
            "motor_current": {"min": 10, "max": 150, "norm_min": 10, "norm_max": 150},
        }
    
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
        
        # Use lift-type aware rules if available
        active_rules = self.rules
        if lift_type and lift_type in WELL_TYPE_RULES:
            active_rules = WELL_TYPE_RULES[lift_type]
        
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
                
                violations.append({
                    "field": field,
                    "value": value,
                    "min": rule_min,
                    "max": rule_max,
                    "unit": UNITS_MAP.get(field, ""),
                    "deviation_pct": round(deviation_pct, 2),
                    "violation": f"Out of range. Expected {rule_min}-{rule_max}, got {value} ({deviation_pct:.1f}% deviation)"
                })
                rule_anomaly_score += 0.25
        
        rule_anomaly_score = min(1.0, rule_anomaly_score)
        
        # --- ML-BASED DETECTION (using trained models) ---
        detection_method = "statistical_rules"
        if lift_type and self.trained_models:
            ml_anomaly_score = self._detect_anomaly_with_trained_models(readings, lift_type)
            if ml_anomaly_score > 0.5:
                detection_method = "statistical_rules + trained_ml_model"
        
        # Combine scores
        combined_score = max(rule_anomaly_score, ml_anomaly_score)
        is_anomaly = len(violations) > 0 or ml_anomaly_score > 0.5
        
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
        well_type_features = {
            'Rod Pump': ['strokes_per_minute', 'torque', 'polish_rod_load', 'pump_fillage', 'tubing_pressure'],
            'ESP': ['motor_temp', 'motor_current', 'discharge_pressure', 'pump_intake_pressure', 'motor_voltage'],
            'Gas Lift': ['injection_rate', 'injection_temperature', 'bottomhole_pressure', 'injection_pressure', 'cycle_time']
        }
        return well_type_features.get(lift_type, [])
    
    
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

def get_detector(excel_file: str = EXCEL_FILE) -> AnomalyDetector:
    """Get or create singleton anomaly detector."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector(excel_file)
    return _detector

def check_anomaly(readings: Dict, well_id: str = None, lift_type: str = None) -> Dict:
    """Convenience function: check for anomalies using default detector."""
    detector = get_detector()
    return detector.check_anomaly(readings, well_id, lift_type)

def ensemble_score(readings: Dict, historical_data: Optional[np.ndarray] = None) -> Dict:
    """Convenience function: compute ensemble score using default detector."""
    detector = get_detector()
    return detector.ensemble_score(readings, historical_data)
