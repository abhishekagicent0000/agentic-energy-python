import json
import uuid
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Action -> Category mapping and category descriptions
ACTION_CATEGORY_MAP = {
    'Increase ESP frequency': 'Production',
    'Adjust pump/polish choke': 'Production',
    'Optimize gas lift injection rate': 'Optimization',
    # fallback actions that may be used in future
    'Inspect ESP equipment': 'Maintenance',
}

# Human-readable category descriptions used for UI/context
CATEGORY_DESCRIPTIONS = {
    'Production': 'Recommendations that directly aim to increase oil or gas production. May involve higher operating cost or equipment wear.',
    'Optimization': 'Recommendations focused on cost/efficiency optimization that typically reduce operating cost with modest production impact.',
    'Maintenance': 'Recommendations focused on preventing failure, extending equipment life, or reducing downtime.',
}

# canonical feature list used across training/prediction
FEATURES = [
    'oil_volume_slope', 'motor_current_slope', 'motor_temp_slope', 'surface_pressure_slope', 'gas_volume_slope',
    'motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'oil_volume_roll_mean', 'gas_volume_roll_mean',
    'missing_rate'
]

# Action mapping from measured field violations to recommended actions
ACTION_MAPPING = {
    'motor_current': {
        'class': 'Increase ESP frequency',
        'category': 'Production',
        'templates': [
            'Increase ESP frequency by 3 Hz',
            'Increase ESP speed by ~5%',
            'Increase ESP frequency to improve motor loading'
        ]
    },
    'motor_temp': {
        'class': 'Inspect ESP equipment',
        'category': 'Maintenance',
        'templates': [
            'Inspect ESP motor and perform impeller alignment within 7 days',
            'Replace impeller and perform alignment check within 7 days',
            'Schedule motor inspection and thermal check'
        ]
    },
    'surface_pressure': {
        'class': 'Adjust separator pressure',
        'category': 'Optimization',
        'templates': [
            'Reduce separator pressure from 125 PSI to 110 PSI',
            'Adjust separator pressure setpoint to reduce backpressure',
            'Lower separator pressure by ~10-15 PSI to improve throughput'
        ]
    },
    'oil_volume': {
        'class': 'Adjust pump/choke',
        'category': 'Production',
        'templates': [
            'Increase pump/choke opening to raise oil rate',
            'Adjust pump/polish choke to increase oil production',
            'Increase pumping speed moderately to recover oil rate'
        ]
    },
    'gas_volume': {
        'class': 'Optimize gas lift injection rate',
        'category': 'Optimization',
        'templates': [
            'Increase gas injection rate from 2.5 MMSCF to 3.2 MMSCF',
            'Optimize gas lift injection rate to improve recovery',
            'Tune gas lift injection by +20% to restore gas balance'
        ]
    }
}

# Map action -> high-level category
ACTION_CATEGORY_MAP = {
    'Increase ESP frequency': 'Production',
    'Adjust pump/polish choke': 'Production',
    'Optimize gas lift injection rate': 'Optimization',
}

# Human-facing category descriptions
CATEGORY_DESCRIPTIONS = {
    'Production': "Recommendations that directly aim to increase oil or gas production. May increase operating cost or equipment wear; valid if net economic benefit is positive.",
    'Optimization': "Recommendations focused on cost/efficiency optimization (reduce chemical usage, operating costs) without necessarily increasing production significantly.",
    'Maintenance': "Maintenance / Failure Prevention: extend equipment life, prevent failures, reduce downtime. These are recommendations, not execution actions."
}

# Optional ML imports
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib
except Exception:
    DecisionTreeClassifier = None
    train_test_split = None
    classification_report = None
    joblib = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
except Exception:
    RandomForestClassifier = None
    CalibratedClassifierCV = None

class TrendAnalyzer:
    """Compute simple trends and rolling stats from a dataframe of readings.

    Expects a DataFrame indexed by timestamp with columns including:
    'oil_volume', 'gas_volume', 'motor_current', 'motor_temp', 'surface_pressure', ...
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_index()

    def rolling_means(self, window='24H'):
        return self.df.rolling(window).mean()

    def slope(self, series: pd.Series, window_points: int = 24):
        """Compute slope over the last `window_points` samples using linear fit.
        Returns None if insufficient data.
        """
        y = series.dropna()
        if len(y) < max(3, window_points):
            return None
        y_tail = y.iloc[-window_points:]
        x = np.arange(len(y_tail))
        try:
            m = np.polyfit(x, y_tail.values, 1)[0]
            return float(m)
        except Exception:
            return None

    def baseline_stats(self, field: str, lookback_points: int = 168, exclude_last: int = 24):
        """Compute baseline mean/std and slope for a field using a lookback window.

        - lookback_points: number of historical samples to use (e.g., 168 for 7 days hourly)
        - exclude_last: number of most recent samples to exclude from baseline (e.g., 24 hours)
        Returns dict with mean, std, slope, slope_std
        """
        if field not in self.df.columns:
            return {'mean': None, 'std': None, 'slope': None, 'slope_std': None}
        series = self.df[field].dropna()
        if len(series) < (exclude_last + 3):
            return {'mean': None, 'std': None, 'slope': None, 'slope_std': None}
        end_idx = len(series) - exclude_last
        start_idx = max(0, end_idx - lookback_points)
        window = series.iloc[start_idx:end_idx]
        mean = float(window.mean()) if not window.empty else None
        std = float(window.std()) if not window.empty else None
        # compute slope over the baseline window
        slope = None
        slope_std = None
        try:
            if len(window) >= 3:
                x = np.arange(len(window))
                m = np.polyfit(x, window.values, 1)[0]
                slope = float(m)
                # approximate slope std by sampling smaller windows
                sls = []
                w = 24
                for i in range(0, max(1, len(window) - w), max(1, w)):
                    part = window.iloc[i:i + w]
                    if len(part) >= 3:
                        xm = np.arange(len(part))
                        sls.append(np.polyfit(xm, part.values, 1)[0])
                slope_std = float(np.std(sls)) if sls else 0.0
        except Exception:
            slope = None
            slope_std = None

        return {'mean': mean, 'std': std, 'slope': slope, 'slope_std': slope_std}

    def compute_metrics(self):
        metrics = {}
        # slopes
        for col in ['oil_volume', 'motor_current', 'motor_temp', 'surface_pressure', 'gas_volume']:
            if col in self.df.columns:
                metrics[f'{col}_slope'] = self.slope(self.df[col], window_points=24)
            else:
                metrics[f'{col}_slope'] = None

        # rolling means last 24 samples
        rm = self.df.rolling(window=24).mean().iloc[-1] if len(self.df) >= 24 else self.df.mean()
        for col in ['motor_current', 'motor_temp', 'surface_pressure', 'oil_volume', 'gas_volume']:
            metrics[f'{col}_roll_mean'] = float(rm[col]) if col in rm and not pd.isna(rm[col]) else None

        # anomaly frequency placeholder (requires anomaly labels) -> count NaNs as proxy
        metrics['missing_rate'] = float(self.df.isna().mean().mean())

        # correlations (motor_current vs motor_temp)
        if 'motor_current' in self.df.columns and 'motor_temp' in self.df.columns:
            metrics['motor_current_motor_temp_corr'] = float(self.df['motor_current'].corr(self.df['motor_temp']))
        else:
            metrics['motor_current_motor_temp_corr'] = None

        return metrics


class RuleEngine:
    """Apply deterministic rules to metrics and generate recommendations."""

    def __init__(self, rules: dict = None):
        """Initialize with optional rules dict of the form {field: {"min":..., "max":...}}"""
        self.rules = rules or {}

    def load_rules_from_excel(self, path: str):
        """Load field ranges from an Excel file. Expects columns: 'field','min','max'."""
        try:
            df = pd.read_excel(path)
        except Exception:
            return {}
        rules = {}
        for _, row in df.iterrows():
            field = str(row.get('field') or row.get('Field') or '')
            if not field:
                continue
            # support alternate column names that users might have
            min_candidates = ['min', 'min_value', 'minimum', 'lower']
            max_candidates = ['max', 'max_value', 'maximum', 'upper']
            minv = None
            maxv = None
            for c in min_candidates:
                if c in row and not pd.isna(row.get(c)):
                    try:
                        minv = float(row.get(c))
                        break
                    except Exception:
                        minv = None
            for c in max_candidates:
                if c in row and not pd.isna(row.get(c)):
                    try:
                        maxv = float(row.get(c))
                        break
                    except Exception:
                        maxv = None
            rules[field] = {'min': minv, 'max': maxv}
        self.rules = rules
        return rules

    def check_ranges(self, readings: dict):
        """Return violations for readings according to loaded rules."""
        violations = []
        for field, bounds in (self.rules or {}).items():
            if field not in readings:
                continue
            val = readings.get(field)
            if val is None:
                continue
            mn = bounds.get('min')
            mx = bounds.get('max')
            if mn is not None and val < mn:
                violations.append({'field': field, 'value': val, 'min': mn, 'max': mx, 'violation': 'below_min'})
            if mx is not None and val > mx:
                violations.append({'field': field, 'value': val, 'min': mn, 'max': mx, 'violation': 'above_max'})
        return violations

    def apply(self, well_id: str, metrics: dict, last_reading: dict = None, trend_analyzer: TrendAnalyzer = None, adaptive_k: float = 2.0):
        recs = []

        # Rule: motor_current high and motor_temp rising -> Increase ESP frequency
        mc = metrics.get('motor_current_roll_mean')
        mt_slope = metrics.get('motor_temp_slope')
        mc_mt_corr = metrics.get('motor_current_motor_temp_corr')
        # determine threshold: excel rule override, else adaptive
        mc_thresh = None
        if self.rules and 'motor_current' in self.rules and self.rules['motor_current'].get('max') is not None:
            mc_thresh = self.rules['motor_current']['max']
        elif trend_analyzer is not None:
            stats = trend_analyzer.baseline_stats('motor_current')
            if stats['mean'] is not None:
                mc_thresh = stats['mean'] + adaptive_k * (stats['std'] or 0.0)

        if mc is not None and mt_slope is not None and mc_thresh is not None:
            if mc > mc_thresh and mt_slope > 0.01:
                action = 'Increase ESP frequency'
                recs.append({
                    'well_id': well_id,
                    'action': action,
                    'category': ACTION_CATEGORY_MAP.get(action, 'Production'),
                    'priority': 'HIGH',
                    'status': 'New',
                    'reason': f'motor_current {mc:.1f} > threshold {mc_thresh:.1f} and motor_temp rising',
                    'expected_impact': 'Estimate based on similar cases',
                    'confidence': 0.9,
                })

        # Rule: surface_pressure declining AND oil_volume dropping -> Pump adjustment
        sp_slope = metrics.get('surface_pressure_slope')
        oil_slope = metrics.get('oil_volume_slope')
        # slope thresholds: use rule if provided, else compare to baseline slope
        slope_thresh = None
        if self.rules and 'surface_pressure' in self.rules:
            # excel rules likely provide absolute min/max not slope; keep default small thresholds
            slope_thresh = -0.01
        elif trend_analyzer is not None:
            sp_stats = trend_analyzer.baseline_stats('surface_pressure')
            oil_stats = trend_analyzer.baseline_stats('oil_volume')
            # consider significant negative slope if below baseline slope minus k * slope_std
            if sp_stats['slope'] is not None and sp_stats['slope_std'] is not None:
                slope_thresh = sp_stats['slope'] - adaptive_k * (sp_stats['slope_std'] or 0.0)
            else:
                slope_thresh = -0.01

        if sp_slope is not None and oil_slope is not None and slope_thresh is not None:
            if sp_slope < slope_thresh and oil_slope < (oil_stats['slope'] - adaptive_k * (oil_stats['slope_std'] or 0.0)):
                action = 'Adjust pump/polish choke'
                recs.append({
                    'well_id': well_id,
                    'action': action,
                    'category': ACTION_CATEGORY_MAP.get(action, 'Production'),
                    'priority': 'HIGH',
                    'status': 'New',
                    'reason': 'surface_pressure and oil production both declining relative to baseline',
                    'expected_impact': 'Estimate based on similar cases',
                    'confidence': 0.85,
                })

        # Rule: gas volume low -> gas lift optimization
        gv = metrics.get('gas_volume_roll_mean')
        # gas threshold from excel or adaptive
        gv_thresh = None
        if self.rules and 'gas_volume' in self.rules and self.rules['gas_volume'].get('min') is not None:
            gv_thresh = self.rules['gas_volume']['min']
        elif trend_analyzer is not None:
            gv_stats = trend_analyzer.baseline_stats('gas_volume')
            if gv_stats['mean'] is not None:
                gv_thresh = gv_stats['mean'] - adaptive_k * (gv_stats['std'] or 0.0)

        if gv is not None and gv_thresh is not None and gv < gv_thresh:
            action = 'Optimize gas lift injection rate'
            recs.append({
                'well_id': well_id,
                'action': action,
                'category': ACTION_CATEGORY_MAP.get(action, 'Optimization'),
                'priority': 'MEDIUM',
                'status': 'New',
                'reason': 'gas volume below threshold',
                'expected_impact': 'Estimate based on similar cases',
                'confidence': 0.8,
            })

        return recs


class ImpactEstimator:
    """Minimal placeholder. Real implementation requires historical adjustment logs.
    For now, this class returns a textual placeholder and does not compute monetary impact.
    """

    def estimate(self, recommendation: dict, metrics: dict):
        # Attach a dummy uplift estimate (percent) based on action type
        action = recommendation.get('action', '').lower()
        if 'esp' in action or 'frequency' in action:
            uplift = 0.15
        elif 'pump' in action or 'choke' in action:
            uplift = 0.08
        elif 'gas lift' in action or 'gas' in action:
            uplift = 0.10
        else:
            uplift = 0.05

        return {'expected_uplift_pct': uplift, 'confidence': recommendation.get('confidence', 0.5)}


def _create_table_sql(table_name: str = 'OPERATION_RECOMMENDATION'):
    # Unified table for both rule-based and ML-based recommendations
    # Use an uppercase, quoted identifier to avoid identifier case issues in Snowflake
    tn = table_name.upper()
    return f"""
    CREATE TABLE IF NOT EXISTS "{tn}" (
        id VARCHAR PRIMARY KEY,
        well_id VARCHAR NOT NULL,
        category VARCHAR,
        action VARCHAR,
        priority VARCHAR,
        status VARCHAR,
        reason VARCHAR,
        expected_impact VARCHAR,
        confidence FLOAT,
        prediction_source VARCHAR,
        probability FLOAT,
        oil_volume_slope FLOAT,
        oil_volume_roll_mean FLOAT,
        gas_volume_slope FLOAT,
        gas_volume_roll_mean FLOAT,
        motor_current_roll_mean FLOAT,
        motor_temp_roll_mean FLOAT,
        surface_pressure_roll_mean FLOAT,
        details VARCHAR,
        created_at TIMESTAMP_NTZ NOT NULL
    );
    """


def save_recommendation_to_snowflake(conn, rec: dict, table_name: str = 'OPERATION_RECOMMENDATION', metrics: dict = None):
    """Save a single recommendation dict into Snowflake (rule-based or ML-based).

    `conn` must be a snowflake.connector connection with `.cursor().execute()` API.
    `metrics` optional dict with metric values to store alongside recommendation.
    """
    cur = conn.cursor()
    try:
        cur.execute(_create_table_sql(table_name))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        rec_id = rec.get('id') or str(uuid.uuid4())
        details = json.dumps(rec).replace("'", "''")  # escape single quotes
        metrics = metrics or rec.get('metrics', {})
        # ensure category present (may be added by rule engine or ML model)
        if not rec.get('category'):
            try:
                from sklearn.externals import joblib as _jb
            except Exception:
                _jb = joblib
            try:
                model_path = os.path.join('models', 'category_classifier.joblib')
                if os.path.exists(model_path) and joblib is not None:
                    mdl = joblib.load(model_path)
                    features = mdl.get('features') if isinstance(mdl, dict) else None
                    clf = mdl.get('model') if isinstance(mdl, dict) else mdl
                    if features and clf is not None:
                        fv = []
                        for f in features:
                            fv.append(metrics.get(f) if metrics and f in metrics else 0.0)
                        try:
                            probas = clf.predict_proba([fv])[0]
                            pred = clf.classes_[probas.argmax()]
                            rec['category'] = pred
                        except Exception:
                            rec['category'] = 'Production'
            except Exception:
                rec['category'] = rec.get('category', 'Production')
        
        def _to_native(v):
            """Convert common numpy/pandas types and complex objects to plain python types supported by DB drivers."""
            try:
                # numpy scalars
                import numpy as _np
                if isinstance(v, (_np.generic,)):
                    return v.item()
            except Exception:
                pass
            # datetime -> native
            try:
                from datetime import datetime
                if isinstance(v, datetime):
                    return v
            except Exception:
                pass
            # basic python types
            if v is None:
                return None
            if isinstance(v, (int, float, str, bool)):
                return v
            # containers -> json string
            try:
                return json.dumps(v)
            except Exception:
                return str(v)
        
        # ensure table name quoting/case to match Snowflake's identifiers
        tn = table_name.upper()

        # Original desired columns and params order
        orig_cols = [
            "ID", "WELL_ID", "CATEGORY", "ACTION", "PRIORITY", "STATUS", "REASON", "EXPECTED_IMPACT", "CONFIDENCE", "PREDICTION_SOURCE", "PROBABILITY",
            "OIL_VOLUME_SLOPE", "OIL_VOLUME_ROLL_MEAN", "GAS_VOLUME_SLOPE", "GAS_VOLUME_ROLL_MEAN",
            "MOTOR_CURRENT_ROLL_MEAN", "MOTOR_TEMP_ROLL_MEAN", "SURFACE_PRESSURE_ROLL_MEAN", "DETAILS", "CREATED_AT"
        ]

        # Build params tuple in the same order
        params = (
            rec_id,
            rec.get('well_id') or '',
            rec.get('category') or 'Production',
            rec.get('action') or '',
            rec.get('priority') or '',
            rec.get('status') or '',
            rec.get('reason') or '',
            rec.get('expected_impact') or '',
            float(rec.get('confidence') or 0.0),
            rec.get('prediction_source') or 'rule_based',
            float(rec.get('probability')) if rec.get('probability') is not None else None,
            metrics.get('oil_volume_slope'),
            metrics.get('oil_volume_roll_mean'),
            metrics.get('gas_volume_slope'),
            metrics.get('gas_volume_roll_mean'),
            metrics.get('motor_current_roll_mean'),
            metrics.get('motor_temp_roll_mean'),
            metrics.get('surface_pressure_roll_mean'),
            details,
            now
        )

        # Try to adapt to the actual table schema: prefer to describe table and use intersection
        try:
            cur.execute(f'DESCRIBE TABLE "{tn}"')
            existing_cols = [r[0].upper() for r in cur.fetchall()]
        except Exception:
            existing_cols = orig_cols

        # If columns are missing in the target table, try to add them to match expected schema
        missing_cols = [c for c in orig_cols if c not in existing_cols]
        if missing_cols:
            logger.info(f"Missing columns detected in {tn}: {missing_cols}; attempting to ALTER TABLE ADD COLUMN...")
            # mapping of column -> SQL type
            col_types = {
                "ID": "VARCHAR",
                "WELL_ID": "VARCHAR",
                "CATEGORY": "VARCHAR",
                "ACTION": "VARCHAR",
                "PRIORITY": "VARCHAR",
                "STATUS": "VARCHAR",
                "REASON": "VARCHAR",
                "EXPECTED_IMPACT": "VARCHAR",
                "CONFIDENCE": "FLOAT",
                "PREDICTION_SOURCE": "VARCHAR",
                "PROBABILITY": "FLOAT",
                "OIL_VOLUME_SLOPE": "FLOAT",
                "OIL_VOLUME_ROLL_MEAN": "FLOAT",
                "GAS_VOLUME_SLOPE": "FLOAT",
                "GAS_VOLUME_ROLL_MEAN": "FLOAT",
                "MOTOR_CURRENT_ROLL_MEAN": "FLOAT",
                "MOTOR_TEMP_ROLL_MEAN": "FLOAT",
                "SURFACE_PRESSURE_ROLL_MEAN": "FLOAT",
                "DETAILS": "VARCHAR",
                "CREATED_AT": "TIMESTAMP_NTZ"
            }
            for c in missing_cols:
                try:
                    sql_type = col_types.get(c, 'VARCHAR')
                    alter_sql = f'ALTER TABLE "{tn}" ADD COLUMN "{c}" {sql_type} NULL'
                    logger.info(f"Altering table {tn}: {alter_sql}")
                    cur.execute(alter_sql)
                    # commit per column to make schema change durable
                    conn.commit()
                    logger.info(f"Added column {c} to {tn}")
                except Exception as alter_ex:
                    logger.warning(f"Failed to add column {c} to {tn}: {alter_ex}")
            # refresh existing columns list
            try:
                cur.execute(f'DESCRIBE TABLE "{tn}"')
                existing_cols = [r[0].upper() for r in cur.fetchall()]
            except Exception:
                existing_cols = orig_cols

        insert_cols = [c for c in orig_cols if c in existing_cols]
        if not insert_cols:
            # fallback to original full insert (may fail and be handled later)
            insert_cols = orig_cols

        insert_cols_sql = ', '.join([f'"{c}"' for c in insert_cols])
        placeholders = ', '.join(['%s'] * len(insert_cols))
        insert_sql = f'INSERT INTO "{tn}" ({insert_cols_sql}) VALUES ({placeholders})'

        params = (
            rec_id,
            rec.get('well_id') or '',
            rec.get('category') or 'Production',
            rec.get('action') or '',
            rec.get('priority') or '',
            rec.get('status') or '',
            rec.get('reason') or '',
            rec.get('expected_impact') or '',
            float(rec.get('confidence') or 0.0),
            rec.get('prediction_source') or 'rule_based',
            float(rec.get('probability')) if rec.get('probability') is not None else None,
            metrics.get('oil_volume_slope'),
            metrics.get('oil_volume_roll_mean'),
            metrics.get('gas_volume_slope'),
            metrics.get('gas_volume_roll_mean'),
            metrics.get('motor_current_roll_mean'),
            metrics.get('motor_temp_roll_mean'),
            metrics.get('surface_pressure_roll_mean'),
            details,
            now
        )

        logger.info(f"Inserting recommendation for {rec.get('well_id')}: {rec.get('action')}")
        safe_params = tuple(_to_native(p) for p in params)
        try:
            cur.execute(insert_sql, safe_params)
            conn.commit()
            logger.info(f"Successfully saved recommendation {rec_id}")
            return rec_id
        except Exception as ex:
            # Log the failure and attempt smarter fallbacks
            logger.warning(f"Primary insert failed: {ex}")
            logger.debug(f"Insert SQL: {insert_sql}")
            logger.debug(f"Params: {safe_params}")

            # First fallback: positional VALUES (original approach)
            try:
                logger.warning(f"Attempting positional fallback insert")
                placeholders = ','.join(['%s'] * len(params))
                alt_sql = f'INSERT INTO "{tn}" VALUES ({placeholders})'
                cur.execute(alt_sql, safe_params)
                conn.commit()
                logger.info(f"Successfully saved recommendation {rec_id} via positional fallback")
                return rec_id
            except Exception as ex2:
                logger.error(f"Positional fallback also failed: {ex2}")
                # Attempt adaptive fallback: discover table columns and insert only matching columns
                try:
                    logger.info("Attempting adaptive fallback by inspecting table columns...")
                    # Describe table to get column list
                    desc_sql = f'DESCRIBE TABLE "{tn}"'
                    cur.execute(desc_sql)
                    cols = [r[0].upper() for r in cur.fetchall()]

                    # Original insert column order
                    orig_cols = [
                        "ID", "WELL_ID", "CATEGORY", "ACTION", "PRIORITY", "STATUS", "REASON", "EXPECTED_IMPACT", "CONFIDENCE", "PREDICTION_SOURCE", "PROBABILITY",
                        "OIL_VOLUME_SLOPE", "OIL_VOLUME_ROLL_MEAN", "GAS_VOLUME_SLOPE", "GAS_VOLUME_ROLL_MEAN",
                        "MOTOR_CURRENT_ROLL_MEAN", "MOTOR_TEMP_ROLL_MEAN", "SURFACE_PRESSURE_ROLL_MEAN", "DETAILS", "CREATED_AT"
                    ]

                    param_map = {c: v for c, v in zip(orig_cols, safe_params)}

                    # choose intersection columns in original order
                    insert_cols = [c for c in orig_cols if c in cols]
                    if not insert_cols:
                        raise RuntimeError(f"No matching columns found in table {tn} for adaptive fallback")

                    insert_placeholders = ','.join(['%s'] * len(insert_cols))
                    insert_cols_sql = ', '.join([f'"{c}"' for c in insert_cols])
                    adaptive_sql = f'INSERT INTO "{tn}" ({insert_cols_sql}) VALUES ({insert_placeholders})'
                    adaptive_params = tuple(param_map[c] for c in insert_cols)

                    logger.debug(f"Adaptive insert SQL: {adaptive_sql}")
                    logger.debug(f"Adaptive params: {adaptive_params}")
                    cur.execute(adaptive_sql, adaptive_params)
                    conn.commit()
                    logger.info(f"Successfully saved recommendation {rec_id} via adaptive fallback")
                    return rec_id
                except Exception as ex3:
                    logger.error(f"Adaptive fallback failed: {ex3}")
                    # Re-raise the original error for upstream handling with full context
                    logger.error(f"Original insert error: {ex}")
                    raise
    except Exception as e:
        logger.error(f"Error saving recommendation: {e}", exc_info=True)
        raise
    finally:
        cur.close()


def predict_category(metrics: dict, model_path: str = 'models/category_classifier.joblib') -> dict:
    """Predict recommendation category from a metrics dict using a trained model.

    Returns {'category': str, 'probability': float} or {} if unavailable.
    """
    if joblib is None:
        return {}
    try:
        if not os.path.exists(model_path):
            return {}
        mdl = joblib.load(model_path)
        if isinstance(mdl, dict):
            clf = mdl.get('model')
            features = mdl.get('features', [])
            le = mdl.get('le')
        else:
            clf = mdl
            features = ['oil_volume_roll_mean', 'gas_volume_roll_mean', 'motor_current_roll_mean',
                        'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'oil_volume_slope',
                        'gas_volume_slope', 'motor_temp_slope', 'missing_rate', 'motor_current_motor_temp_corr']
            le = None

        fv = []
        for f in features:
            fv.append(metrics.get(f) if metrics and f in metrics else 0.0)

        probs = clf.predict_proba([fv])[0]
        idx = probs.argmax()
        pred = clf.classes_[idx]
        # if label encoder present
        if le is not None:
            try:
                pred = le.inverse_transform([pred])[0]
            except Exception:
                pass
        return {'category': pred, 'probability': float(probs[idx])}
    except Exception:
        return {}


def train_category_model(snowflake_conn, model_path: str = 'models/category_classifier.joblib') -> dict:
    """Train a category classifier using synthetic labels generated by the rule engine.

    - Fetch recent sensor readings per well
    - Compute metrics via TrendAnalyzer
    - Use RuleEngine.apply to create synthetic recommendation labels and their categories
    - Train a RandomForestClassifier and save via joblib
    Returns training summary dict.
    """
    if RandomForestClassifier is None or joblib is None:
        raise RuntimeError('Required sklearn or joblib not available')

    cur = snowflake_conn.cursor()
    X = []
    y = []
    wells = []
    try:
        cur.execute("SELECT DISTINCT well_id FROM well_sensor_readings")
        wells = [r[0] for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching wells for training: {e}")
        raise

    re = RuleEngine()

    for well in wells:
        try:
            cur.execute("SELECT timestamp, oil_volume, gas_volume, motor_current, motor_temp, surface_pressure FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 168", (well,))
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [d[0] for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            # coerce numeric columns
            for c in ['oil_volume', 'gas_volume', 'motor_current', 'motor_temp', 'surface_pressure']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')

            ta = TrendAnalyzer(df.set_index('timestamp')) if 'timestamp' in df.columns else TrendAnalyzer(df)
            metrics = ta.compute_metrics()
            recs = re.apply(well, metrics, trend_analyzer=ta)
            for rec in recs:
                cat = rec.get('category')
                if not cat:
                    continue
                feat = [
                    metrics.get('oil_volume_roll_mean'),
                    metrics.get('gas_volume_roll_mean'),
                    metrics.get('motor_current_roll_mean'),
                    metrics.get('motor_temp_roll_mean'),
                    metrics.get('surface_pressure_roll_mean'),
                    metrics.get('oil_volume_slope'),
                    metrics.get('gas_volume_slope'),
                    metrics.get('motor_temp_slope'),
                    metrics.get('missing_rate'),
                    metrics.get('motor_current_motor_temp_corr')
                ]
                X.append([0.0 if v is None else v for v in feat])
                y.append(cat)
        except Exception as e:
            logger.warning(f"Error processing well {well} for category training: {e}")
            continue

    if not X:
        raise RuntimeError('No training samples generated')

    # encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_enc)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': clf, 'features': ['oil_volume_roll_mean', 'gas_volume_roll_mean', 'motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'oil_volume_slope', 'gas_volume_slope', 'motor_temp_slope', 'missing_rate', 'motor_current_motor_temp_corr'], 'le': le, 'classes_': le.classes_.tolist()}, model_path)

    # basic report
    preds = clf.predict(X)
    from sklearn.metrics import classification_report
    report = classification_report(y_enc, preds, target_names=le.classes_.tolist(), zero_division=0)
    summary = {'n_samples': len(X), 'classes': le.classes_.tolist(), 'report': report, 'model_path': model_path}
    logger.info(f"Trained category classifier: {summary}")
    return summary


def train_with_synthetic_labels(snowflake_conn, model_path: str = 'models/decision_tree_synthetic.joblib'):
    """Train a model using synthetic action labels based on metric variance.
    
    - Fetches metrics for all wells
    - Assigns action labels based on how far metrics deviate from mean
    - Trains DecisionTree to recognize these patterns
    - Returns model report with accuracy and class distribution
    """
    if DecisionTreeClassifier is None:
        raise RuntimeError('scikit-learn not available')
    
    cur = snowflake_conn.cursor()
    try:
        # get all wells
        cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
        wells = [r[0] for r in cur.fetchall()]
        logger.info(f"Found {len(wells)} wells for training")
        X = []
        y = []
        all_metrics = []
        
        # collect metrics for all wells
        for well in wells:
            try:
                cur.execute("SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 168", (well,))
                rows = cur.fetchall()
                if not rows:
                    continue
                cols = [d[0].lower() for d in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                ts_col = None
                for c in df.columns:
                    if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                        ts_col = c
                        break
                if ts_col is None:
                    ts_col = df.columns[0]
                df['timestamp'] = pd.to_datetime(df[ts_col])
                df = df.set_index('timestamp')
                ta = TrendAnalyzer(df)
                metrics = ta.compute_metrics()

                # production data - use last 60 days for better slope estimation
                cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
                prod_rows = cur.fetchall()
                if prod_rows:
                    prod_cols = [d[0].lower() for d in cur.description]
                    prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                    prod_df['date'] = pd.to_datetime(prod_df['date'])
                    prod_df = prod_df.set_index('date')
                    try:
                        if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['oil_volume'].dropna().values[::-1]
                            x = np.arange(len(yv))
                            metrics['oil_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                            metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['oil_volume_slope'] = None
                        metrics['oil_volume_roll_mean'] = None
                    try:
                        if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['gas_volume'].dropna().values[::-1]
                            x = np.arange(len(yv))
                            metrics['gas_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                            metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['gas_volume_slope'] = None
                        metrics['gas_volume_roll_mean'] = None
                
                all_metrics.append((well, metrics))
            except Exception as e:
                logger.warning(f"Error processing well {well}: {e}")
                pass
        
        if not all_metrics:
            raise ValueError('No metrics collected from Snowflake')
        
        logger.info(f"Collected metrics for {len(all_metrics)} wells")
        
        # compute mean and std for each metric field
        metric_keys = list(all_metrics[0][1].keys())
        field_means = {}
        field_stds = {}
        for key in metric_keys:
            vals = [metrics.get(key) for well, metrics in all_metrics if metrics.get(key) is not None]
            if vals:
                field_means[key] = np.mean(vals)
                field_stds[key] = np.std(vals)
            else:
                field_means[key] = 0.0
                field_stds[key] = 1.0
        
        # assign labels based on deviation from mean using z-scores
        label_distribution = {}

        import random

        for well, metrics in all_metrics:
            feats, _ = _features_from_metrics(metrics)
            X.append(feats)

            # compute z-scores for metric fields to pick candidate actions
            candidates = []
            for key, fmap in [('motor_current_roll_mean', 'motor_current'), ('motor_temp_slope', 'motor_temp'), ('surface_pressure_slope', 'surface_pressure'), ('oil_volume_slope', 'oil_volume'), ('gas_volume_slope', 'gas_volume')]:
                val = metrics.get(key)
                std = field_stds.get(key, 1.0)
                mean = field_means.get(key, 0.0)
                if val is not None and std and abs(std) > 1e-6:
                    z_score = (val - mean) / (std + 1e-6)
                    # positive or negative deviation matters for action type
                    if abs(z_score) > 1.2:
                        # pick action templates for the measured field
                        amap = ACTION_MAPPING.get(fmap)
                        if amap and isinstance(amap, dict):
                            # use canonical class label for training, templates remain for downstream generation
                            act_class = amap.get('class') or amap.get('category')
                            cat = amap.get('category') or ACTION_CATEGORY_MAP.get(act_class, 'Production')
                            candidates.append((act_class, cat, abs(z_score)))

            # choose the strongest candidate by z-score magnitude
            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                chosen_action, chosen_cat, _ = candidates[0]
                label = chosen_action
            else:
                label = 'NoAction'

            y.append(label)
            label_distribution[label] = label_distribution.get(label, 0) + 1
        
        logger.info(f"Label distribution: {label_distribution}")
        
        X = np.vstack(X)
        y = np.array(y)
        
        # Handle imbalanced classes: use simple train/test split without stratification
        # if some classes have only 1 sample
        min_samples_per_class = min(label_distribution.values()) if label_distribution else 0
        
        if min_samples_per_class >= 2:
            # Use stratified split if balanced enough
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            # Use regular split for imbalanced data
            logger.warning(f"Imbalanced classes detected: using non-stratified split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a stronger classifier (RandomForest) and calibrate probabilities
        if RandomForestClassifier is None or CalibratedClassifierCV is None:
            # fall back to DecisionTree if RandomForest / calibration unavailable
            clf = DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)
            calibrated_clf = clf
        else:
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            # Use CalibratedClassifierCV to produce meaningful probabilities
            try:
                calibrated = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
                calibrated.fit(X_train, y_train)
                calibrated_clf = calibrated
            except Exception:
                # If calibration fails (small data), fit RF directly
                rf.fit(X_train, y_train)
                calibrated_clf = rf

        # evaluate
        try:
            y_pred = calibrated_clf.predict(X_test)
            y_pred_train = calibrated_clf.predict(X_train)
        except Exception:
            # in unlikely case predict fails, fall back to naive predictions
            y_pred = np.array(['NoAction'] * len(X_test))
            y_pred_train = np.array(['NoAction'] * len(X_train))

        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred)
        logger.info(f"Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")

        try:
            report = classification_report(y_test, y_pred, output_dict=True)
        except Exception:
            report = {}

        # save model (calibrated classifier or fallback)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_clf, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return {
            'model_path': model_path,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'trained_on': int(len(y)),
            'label_distribution': label_distribution,
            'report': report
        }
    finally:
        cur.close()


def train_from_snowflake(snowflake_conn, lookback_days: int = 30, model_path: str = 'models/decision_tree.joblib', stricter: bool = False):
    """Build a training set from Snowflake using Excel rules to label examples, then train a DecisionTree.

    - Loads rules from 'data types and ranges.xlsx' if present.
    - For each well, computes recent sensor + production metrics and uses rule violations as labels.
    - If stricter=True, applies stricter thresholds (±20% adjustment) to detect more anomalies.
    - Trains and saves a DecisionTreeClassifier to `model_path`.
    Returns training report dict.
    """
    if DecisionTreeClassifier is None:
        raise RuntimeError('scikit-learn not available')

    re = RuleEngine()
    try:
        re.load_rules_from_excel('data types and ranges.xlsx')
    except Exception:
        pass
    
    # Apply stricter rules if requested
    if stricter and re.rules:
        for field in re.rules:
            rule = re.rules[field]
            if rule.get('min') is not None:
                rule['min'] = rule['min'] * 0.8  # 20% lower threshold
            if rule.get('max') is not None:
                rule['max'] = rule['max'] * 1.2  # 20% higher threshold

    cur = snowflake_conn.cursor()
    try:
        # get wells
        cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
        wells = [r[0] for r in cur.fetchall()]
        X = []
        y = []
        for well in wells:
            # fetch sensor recent (7 days hourly) and daily production (lookback_days)
            cur.execute("SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 168", (well,))
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [d[0].lower() for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            ts_col = None
            for c in df.columns:
                if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                    ts_col = c
                    break
            if ts_col is None:
                ts_col = df.columns[0]
            df['timestamp'] = pd.to_datetime(df[ts_col])
            df = df.set_index('timestamp')
            ta = TrendAnalyzer(df)
            metrics = ta.compute_metrics()

            # production
            cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT %s", (well, lookback_days))
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [d[0].lower() for d in cur.description]
                prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                prod_df['date'] = pd.to_datetime(prod_df['date'])
                prod_df = prod_df.set_index('date')
                try:
                    if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['oil_volume'].dropna().values[::-1]
                        x = np.arange(len(yv))
                        metrics['oil_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                        metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    metrics['oil_volume_slope'] = None
                    metrics['oil_volume_roll_mean'] = None
                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['gas_volume'].dropna().values[::-1]
                        x = np.arange(len(yv))
                        metrics['gas_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    metrics['gas_volume_slope'] = None
                    metrics['gas_volume_roll_mean'] = None

            # label by rule violations on last reading
            last_reading = df.iloc[-1].to_dict() if len(df) else {}
            violations = re.check_ranges(last_reading)
            if not violations:
                label = 'NoAction'
            else:
                # map field to action
                mapping = {'motor_current': 'Increase ESP frequency', 'motor_temp': 'Increase ESP frequency', 'surface_pressure': 'Adjust pump/polish choke', 'oil_volume': 'Adjust pump/polish choke', 'gas_volume': 'Optimize gas lift injection rate'}
                mapped = None
                for v in violations:
                    mapped = mapping.get(v['field']) or mapped
                label = mapped or 'NoAction'

            feats, _ = _features_from_metrics(metrics)
            X.append(feats)
            y.append(label)

        if not X:
            raise ValueError('No training data assembled from Snowflake')

        X = np.vstack(X)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)
        return {'model_path': model_path, 'report': report, 'trained_on': int(len(y))}
    finally:
        cur.close()


def train_on_two_wells(snowflake_conn, well_ids: list, model_path: str = 'models/decision_tree_two_wells.joblib'):
    """Train a DecisionTree on two wells using both sensor and production-derived metrics.

    This helper is useful to quickly test if training on concrete wells produces detectable
    candidates. Labels are derived from `data types and ranges.xlsx` via `RuleEngine`.
    """
    if DecisionTreeClassifier is None or joblib is None:
        raise RuntimeError('scikit-learn and joblib required')

    if not isinstance(well_ids, (list, tuple)) or len(well_ids) < 2:
        raise ValueError('Provide at least two well ids')

    re = RuleEngine()
    try:
        re.load_rules_from_excel('data types and ranges.xlsx')
    except Exception:
        pass

    cur = snowflake_conn.cursor()
    X = []
    y = []
    try:
        for well in well_ids:
            # fetch sensor readings
            cur.execute("SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 168", (well,))
            rows = cur.fetchall()
            if not rows:
                continue
            cols = [d[0].lower() for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            ts_col = None
            for c in df.columns:
                if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                    ts_col = c
                    break
            if ts_col is None:
                ts_col = df.columns[0]
            df['timestamp'] = pd.to_datetime(df[ts_col])
            df = df.set_index('timestamp')
            ta = TrendAnalyzer(df)
            metrics = ta.compute_metrics()

            # production-derived metrics
            cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [d[0].lower() for d in cur.description]
                prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                prod_df['date'] = pd.to_datetime(prod_df['date'])
                prod_df = prod_df.set_index('date')
                try:
                    if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['oil_volume'].dropna().values[::-1]
                        x = np.arange(len(yv))
                        metrics['oil_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                        metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    metrics['oil_volume_slope'] = metrics.get('oil_volume_slope')
                    metrics['oil_volume_roll_mean'] = metrics.get('oil_volume_roll_mean')
                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        yv = prod_df['gas_volume'].dropna().values[::-1]
                        x = np.arange(len(yv))
                        metrics['gas_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                except Exception:
                    metrics['gas_volume_slope'] = metrics.get('gas_volume_slope')
                    metrics['gas_volume_roll_mean'] = metrics.get('gas_volume_roll_mean')

            # label by rule violations on last available reading (use production ranges as fallback)
            last_reading = df.iloc[-1].to_dict() if len(df) else {}
            # merge production roll means into last_reading for rule checks
            if metrics.get('oil_volume_roll_mean') is not None:
                last_reading['oil_volume'] = last_reading.get('oil_volume') or metrics.get('oil_volume_roll_mean')
            if metrics.get('gas_volume_roll_mean') is not None:
                last_reading['gas_volume'] = last_reading.get('gas_volume') or metrics.get('gas_volume_roll_mean')

            violations = re.check_ranges(last_reading)
            if not violations:
                label = 'NoAction'
            else:
                mapping = {'motor_current': 'Increase ESP frequency', 'motor_temp': 'Increase ESP frequency', 'surface_pressure': 'Adjust pump/polish choke', 'oil_volume': 'Adjust pump/polish choke', 'gas_volume': 'Optimize gas lift injection rate'}
                mapped = None
                for v in violations:
                    mapped = mapping.get(v['field']) or mapped
                label = mapped or 'NoAction'

            feat, _ = _features_from_metrics(metrics)
            X.append(feat)
            y.append(label)

        if not X:
            raise RuntimeError('No training samples from provided wells')

        X = np.vstack(X)
        y = np.array(y)
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X, y)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)
        return {'model_path': model_path, 'trained_on': len(y), 'label_distribution': {k: int((y==k).sum()) for k in np.unique(y)}}
    finally:
        cur.close()


def predict_all_wells(snowflake_conn, model_path: str = 'models/decision_tree.joblib', table_name: str = 'OPERATION_RECOMMENDATION', 
                     confidence_threshold: float = 0.6, dedup_days: int = 7):
    """Run the saved model for all wells and return a list of wells predicted to need action.
    
    Implements client requirements:
    - Only recommend on wells that are NOT in anomaly/alert state
    - Apply ML model with confidence_threshold (default 0.75)
    - Deduplicate: avoid duplicate well+action within dedup_days
    - Persist recommendations to unified `OPERATION_RECOMMENDATION` table in Snowflake
    
    Args:
        snowflake_conn: Snowflake connection
        model_path: Path to trained DecisionTree model
        table_name: Target table for recommendations
        confidence_threshold: Minimum confidence to persist (default 0.75)
        dedup_days: Deduplication window in days (default 7)
    
    Returns:
        List of dicts with detected recommendations
    """
    clf = load_model(model_path)
    cur = snowflake_conn.cursor()
    results = []
    try:
        # fetch all wells
        cur.execute("SELECT DISTINCT well_id FROM well_daily_production ORDER BY well_id")
        wells = [r[0] for r in cur.fetchall()]
        logger.info(f"predict_all_wells: Processing {len(wells)} wells")
        
        for well in wells:
            try:
                # No anomaly pre-filter: use sensor and production data directly
                # STEP 2: Compute metrics
                cur.execute("SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s ORDER BY timestamp DESC LIMIT 168", (well,))
                rows = cur.fetchall()
                if not rows:
                    continue
                cols = [d[0].lower() for d in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                ts_col = None
                for c in df.columns:
                    if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                        ts_col = c
                        break
                if ts_col is None:
                    ts_col = df.columns[0]
                df['timestamp'] = pd.to_datetime(df[ts_col])
                df = df.set_index('timestamp')
                ta = TrendAnalyzer(df)
                metrics = ta.compute_metrics()

                # production metrics (last 60 days for better slope estimation)
                cur.execute("SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s ORDER BY date DESC LIMIT 60", (well,))
                prod_rows = cur.fetchall()
                if prod_rows:
                    prod_cols = [d[0].lower() for d in cur.description]
                    prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                    prod_df['date'] = pd.to_datetime(prod_df['date'])
                    prod_df = prod_df.set_index('date')
                    try:
                        if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['oil_volume'].dropna().values[::-1]
                            x = np.arange(len(yv))
                            metrics['oil_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                            metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['oil_volume_slope'] = None
                        metrics['oil_volume_roll_mean'] = None
                    try:
                        if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                            yv = prod_df['gas_volume'].dropna().values[::-1]
                            x = np.arange(len(yv))
                            metrics['gas_volume_slope'] = float(np.polyfit(x, yv, 1)[0])
                            metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=min(7, len(prod_df))).mean().iloc[-1])
                    except Exception:
                        metrics['gas_volume_slope'] = None
                        metrics['gas_volume_roll_mean'] = None

                # STEP 3: Make prediction
                feats, _ = _features_from_metrics(metrics)
                # produce probabilities for all classes and allow multiple recommendations per well
                probas = None
                classes = None
                try:
                    probas = clf.predict_proba([feats])[0]
                    classes = clf.classes_
                except Exception:
                    # predict_proba unavailable; fall back to single prediction
                    pred = clf.predict([feats])[0]
                    probas = None
                    classes = [pred]

                # collect candidate actions above threshold (exclude NoAction)
                candidates = []
                if probas is not None and classes is not None:
                    for cls_name, p in zip(classes, probas):
                        if cls_name == 'NoAction':
                            continue
                        if p >= confidence_threshold:
                            candidates.append((cls_name, float(p)))
                else:
                    # single prediction fallback
                    if pred != 'NoAction':
                        candidates.append((pred, None))

                # Require detection signals from BOTH sensor readings and production tables
                # Treat production detection as presence of a computed oil/gas slope
                prod_detect = False
                if metrics.get('oil_volume_slope') is not None or metrics.get('gas_volume_slope') is not None:
                    # simple threshold check: non-zero slope implies production signal available
                    prod_detect = abs(float(metrics.get('oil_volume_slope') or 0.0)) > 0.0 or abs(float(metrics.get('gas_volume_slope') or 0.0)) > 0.0

                # sensor detection: presence of non-null slopes in sensor-derived metrics
                sensor_detect = any(
                    (metrics.get(f'{c}_slope') is not None and abs(float(metrics.get(f'{c}_slope') or 0.0)) > 0.0)
                    for c in ('motor_current', 'motor_temp', 'surface_pressure')
                )

                if not prod_detect or not sensor_detect:
                    logger.debug(f"Well {well}: skipping candidate actions because detection not present in both sensor and production data (prod_detect={prod_detect}, sensor_detect={sensor_detect})")
                    continue

                if not candidates:
                    logger.debug(f"Well {well}: no candidate actions above threshold {confidence_threshold}")
                    continue

                # persist each candidate
                for pred_action, proba in candidates:
                    logger.info(f"Well {well}: candidate action '{pred_action}' (prob={proba})")
                    # dedup check per action
                    try:
                        cutoff_date = f"DATEADD(day, -{dedup_days}, CURRENT_TIMESTAMP())"
                        # coerce params to native python types to avoid Snowflake binding errors
                        cur.execute(f"""
                            SELECT COUNT(*) FROM {table_name}
                            WHERE well_id = %s AND action = %s AND created_at > {cutoff_date}
                        """, (str(well), str(pred_action)))
                        recent_count = cur.fetchone()[0]
                        if recent_count > 0:
                            logger.info(f"Well {well}: duplicate recommendation for '{pred_action}' within {dedup_days} days, skipping")
                            continue
                    except Exception as e:
                        logger.warning(f"Deduplication check failed for well {well}: {e}")

                    results.append({'well_id': well, 'action': pred_action, 'probability': proba, 'metrics': metrics, 'category': ACTION_CATEGORY_MAP.get(pred_action)})
                    try:
                        rec = {
                            'well_id': well,
                            'action': pred_action,
                            'priority': 'HIGH' if (proba or 0) > 0.85 else 'MEDIUM',
                            'status': 'New',
                            'reason': f'ML model prediction (confidence: {int((proba or 0)*100)}%)',
                            'expected_impact': '',
                            'confidence': proba or 0.0,
                            'prediction_source': 'ml_model',
                            'probability': proba,
                            'metrics': metrics,
                            'created_at': datetime.utcnow().isoformat(),
                            'id': str(uuid.uuid4())
                        }
                        # Use model to predict category (category is source-of-truth)
                        try:
                            cat_pred = predict_category(metrics)
                            if cat_pred:
                                rec['category'] = cat_pred.get('category')
                                # store model probability if available
                                if cat_pred.get('probability') is not None:
                                    rec['probability'] = float(cat_pred.get('probability'))
                        except Exception:
                            rec['category'] = ACTION_CATEGORY_MAP.get(pred_action)

                        # prepare Snowflake detection record (store category from model)
                        rec_sf = rec.copy()
                        # keep the detected action as well for traceability
                        rec_sf['action'] = pred_action
                        save_recommendation_to_snowflake(snowflake_conn, rec_sf, table_name=table_name, metrics=metrics)
                        logger.info(f"Saved recommendation for well {well}: {pred_action}")
                    except Exception as e:
                        logger.error(f"Failed to save recommendation for well {well}: {e}", exc_info=True)
            
            except Exception as e:
                logger.error(f"Error processing well {well}: {e}", exc_info=True)
                continue

        logger.info(f"predict_all_wells completed: {len(results)} wells recommended")
        return results
    finally:
        cur.close()


def _features_from_metrics(metrics: dict):
    """Convert metrics dict to a flat numeric feature vector and feature names."""
    # Use the canonical FEATURES list and prefer production-derived roll means when available
    keys = FEATURES
    features = []
    for k in keys:
        v = metrics.get(k)
        try:
            features.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0)
        except Exception:
            features.append(0.0)
    return np.array(features), keys


def load_model(model_path: str = 'models/decision_tree.joblib'):
    if joblib is None:
        raise RuntimeError('joblib not available')
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return joblib.load(model_path)


def predict_with_model(model_path: str, metrics: dict):
    clf = load_model(model_path)
    feats, feature_names = _features_from_metrics(metrics)
    pred = clf.predict([feats])[0]
    proba = None
    try:
        proba = clf.predict_proba([feats])[0].max()
    except Exception:
        proba = None
    return {'prediction': pred, 'probability': float(proba) if proba is not None else None}


def generate_recommendations_for_well(snowflake_conn, well_id: str, start_ts: str = None, end_ts: str = None):
    """Fetch data from Snowflake for a well, compute metrics, apply rules, estimate impact, and save recommendations.

    Returns list of saved recommendation dicts (with generated `id`).
    """
    cur = snowflake_conn.cursor()
    try:
        # basic query: fetch sensor readings (production is stored in well_daily_production)
        q = f"SELECT timestamp, surface_pressure, tubing_pressure, casing_pressure, motor_temp, wellhead_temp, motor_current FROM well_sensor_readings WHERE well_id = %s"
        params = [well_id]
        if start_ts:
            q += " AND timestamp >= %s"
            params.append(start_ts)
        if end_ts:
            q += " AND timestamp <= %s"
            params.append(end_ts)
        q += " ORDER BY timestamp"
        cur.execute(q, params)
        rows = cur.fetchall()
        if not rows:
            return []
        # normalize column names (Snowflake may return uppercase names)
        cols = [d[0].lower() for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        # locate timestamp-like column robustly
        ts_col = None
        for c in df.columns:
            if 'timestamp' in c or c == 'time' or c.endswith('_ts') or 'date' in c:
                ts_col = c
                break
        if ts_col is None:
            ts_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[ts_col])
        df = df.set_index('timestamp')
        ta = TrendAnalyzer(df)
        metrics = ta.compute_metrics()

        # Fetch daily production for the well and compute production-based metrics
        try:
            # use same cursor to query production
            prod_q = "SELECT date, oil_volume, gas_volume FROM well_daily_production WHERE well_id = %s"
            prod_params = [well_id]
            if start_ts:
                prod_q += " AND date >= %s"
                prod_params.append(start_ts.split('T')[0] if 'T' in start_ts else start_ts)
            if end_ts:
                prod_q += " AND date <= %s"
                prod_params.append(end_ts.split('T')[0] if 'T' in end_ts else end_ts)
            prod_q += " ORDER BY date"
            cur.execute(prod_q, prod_params)
            prod_rows = cur.fetchall()
            if prod_rows:
                prod_cols = [d[0] for d in cur.description]
                prod_df = pd.DataFrame(prod_rows, columns=prod_cols)
                prod_df['date'] = pd.to_datetime(prod_df['date'])
                prod_df = prod_df.set_index('date')
                # compute slopes for oil and gas if available
                try:
                    if 'oil_volume' in prod_df.columns and len(prod_df) >= 3:
                        y = prod_df['oil_volume'].dropna().values
                        x = np.arange(len(y))
                        metrics['oil_volume_slope'] = float(np.polyfit(x, y, 1)[0])
                        metrics['oil_volume_roll_mean'] = float(prod_df['oil_volume'].rolling(window=7).mean().iloc[-1])
                    else:
                        metrics['oil_volume_slope'] = None
                        metrics['oil_volume_roll_mean'] = None
                except Exception:
                    metrics['oil_volume_slope'] = None
                    metrics['oil_volume_roll_mean'] = None

                try:
                    if 'gas_volume' in prod_df.columns and len(prod_df) >= 3:
                        y = prod_df['gas_volume'].dropna().values
                        x = np.arange(len(y))
                        metrics['gas_volume_slope'] = float(np.polyfit(x, y, 1)[0])
                        metrics['gas_volume_roll_mean'] = float(prod_df['gas_volume'].rolling(window=7).mean().iloc[-1])
                    else:
                        metrics['gas_volume_slope'] = None
                        metrics['gas_volume_roll_mean'] = None
                except Exception:
                    metrics['gas_volume_slope'] = None
                    metrics['gas_volume_roll_mean'] = None
        except Exception:
            # if production fetch fails, continue with sensor-only metrics
            pass
        re = RuleEngine()
        # load excel rules if present
        try:
            re.load_rules_from_excel('data types and ranges.xlsx')
        except Exception:
            pass
        raw_recs = re.apply(well_id, metrics, last_reading=df.iloc[-1].to_dict(), trend_analyzer=ta)

        ie = ImpactEstimator()
        saved = []
        for r in raw_recs:
            impact = ie.estimate(r, metrics)
            r['impact'] = impact
            r['metrics'] = metrics
            r['created_at'] = datetime.utcnow().isoformat()
            r['id'] = str(uuid.uuid4())
            # Save to Snowflake
            try:
                save_recommendation_to_snowflake(snowflake_conn, r)
            except Exception:
                # swallow to allow returning recommendations even if save fails
                pass
            saved.append(r)

        return saved
    finally:
        cur.close()


def detect_and_persist(snowflake_conn, pg_conn=None, model_path: str = 'models/decision_tree.joblib',
                       retrain: bool = False, confidence_threshold: float = 0.6,
                       snowflake_table: str = 'operational_recommendation'):
    """Run training (optional), detect recommendations for all wells, and persist to Snowflake and PostgreSQL.

    Args:
        snowflake_conn: active Snowflake connection (cursor/execute API expected)
        pg_conn: optional psycopg2 connection to PostgreSQL; if None the function will try to open one using
                 DATABASE_URL environment variable via psycopg2.connect
        model_path: path to model file
        retrain: if True, call `train_from_snowflake` to retrain model before prediction
        confidence_threshold: minimum probability to persist to Postgres as well
        snowflake_table: target Snowflake table name

    Returns:
        list of persisted recommendation ids (snowflake ids)
    """
    persisted_ids = []

    # optional retrain
    if retrain:
        try:
            logger.info("Retraining model from Snowflake before detection")
            train_from_snowflake(snowflake_conn, model_path=model_path)
        except Exception as e:
            logger.warning(f"Retraining failed: {e}")

    # run predictions for all wells
    try:
        recs = predict_all_wells(snowflake_conn, model_path=model_path, table_name=snowflake_table, confidence_threshold=confidence_threshold)
    except Exception as e:
        logger.error(f"Error running predict_all_wells: {e}")
        return persisted_ids

    # If no pg_conn provided, try to open one from env
    local_conn_opened = False
    if pg_conn is None:
        try:
            import psycopg2
            from urllib.parse import urlparse, urlunparse
            pg_url = os.getenv('DATABASE_URL')
            if pg_url:
                parsed = urlparse(pg_url)
                clean = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, '', parsed.fragment))
                pg_conn = psycopg2.connect(clean)
                local_conn_opened = True
            else:
                logger.info('No DATABASE_URL set; skipping Postgres persistence')
        except Exception as e:
            logger.warning(f'Could not open Postgres connection from env: {e}')

    # persist each recommendation
    for rec in recs:
        try:
            # ensure category exists (use category predictor if missing)
            if not rec.get('category'):
                try:
                    cat = predict_category(rec.get('metrics', {}))
                    if cat:
                        rec['category'] = cat.get('category')
                        # allow saving probability under probability key
                        if cat.get('probability') is not None:
                            rec['probability'] = float(cat.get('probability'))
                except Exception:
                    pass

            # save to Snowflake (will create table if needed)
            try:
                sf_id = save_recommendation_to_snowflake(snowflake_conn, rec, table_name=snowflake_table, metrics=rec.get('metrics'))
                persisted_ids.append(sf_id)
            except Exception as e:
                logger.error(f"Failed to save to Snowflake for well {rec.get('well_id')}: {e}")

            # persist to Postgres if connection available and confidence passes threshold
            conf = rec.get('probability') or rec.get('confidence') or 0.0
            # convert to percent if probability in [0,1]
            conf_pct = float(conf * 100.0) if conf <= 1.0 else float(conf)
            if pg_conn and (conf_pct >= (confidence_threshold * 100.0)):
                try:
                    # Import here to avoid circular imports at module load time
                    try:
                        from operation import save_operation_suggestion
                    except Exception:
                        save_operation_suggestion = None

                    prod_metrics = {k: rec.get('metrics', {}).get(k) for k in ('oil_volume_slope', 'oil_volume_roll_mean', 'gas_volume_slope', 'gas_volume_roll_mean')}
                    sensor_metrics = {k: rec.get('metrics', {}).get(k) for k in ('motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'missing_rate')}

                    if save_operation_suggestion is not None:
                        ok = save_operation_suggestion(
                            rec.get('well_id'),
                            rec.get('action') or rec.get('recommendation') or 'Recommendation',
                            category=rec.get('category') or 'Production',
                            status=rec.get('status') or 'New',
                            priority=rec.get('priority') or 'MEDIUM',
                            confidence=conf_pct,
                            production_data=prod_metrics,
                            sensor_metrics=sensor_metrics,
                            reason=rec.get('reason') or '',
                            expected_impact=rec.get('expected_impact') or ''
                        )
                        if not ok:
                            logger.error(f"Failed to save operation suggestion via operation.save_operation_suggestion for well {rec.get('well_id')}")
                    else:
                        logger.warning("operation.save_operation_suggestion not available; skipping Postgres insert")
                except Exception as e:
                    logger.error(f"Failed to save to Postgres for well {rec.get('well_id')}: {e}")

        except Exception as e:
            logger.warning(f"Error processing recommendation for persistence: {e}")

    # close local pg_conn if opened here
    if local_conn_opened and pg_conn is not None:
        try:
            pg_conn.close()
        except Exception:
            pass

    logger.info(f"detect_and_persist: persisted {len(persisted_ids)} recommendations to Snowflake")
    return persisted_ids
