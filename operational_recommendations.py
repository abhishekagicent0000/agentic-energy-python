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
            try:
                minv = float(row.get('min')) if not pd.isna(row.get('min')) else None
            except Exception:
                minv = None
            try:
                maxv = float(row.get('max')) if not pd.isna(row.get('max')) else None
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
                recs.append({
                    'well_id': well_id,
                    'action': 'Increase ESP frequency',
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
                recs.append({
                    'well_id': well_id,
                    'action': 'Adjust pump/polish choke',
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
            recs.append({
                'well_id': well_id,
                'action': 'Optimize gas lift injection rate',
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


def _create_table_sql(table_name: str = 'operation_recommendation'):
    # Unified table for both rule-based and ML-based recommendations
    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id VARCHAR PRIMARY KEY,
        well_id VARCHAR NOT NULL,
        action VARCHAR NOT NULL,
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


def save_recommendation_to_snowflake(conn, rec: dict, table_name: str = 'operation_recommendation', metrics: dict = None):
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
        
        def sql_value(v):
            """Convert Python value to SQL literal"""
            if v is None:
                return 'NULL'
            elif isinstance(v, bool):
                return 'TRUE' if v else 'FALSE'
            elif isinstance(v, (int, float)):
                return str(v)
            elif isinstance(v, str):
                return f"'{v.replace(chr(39), chr(39)+chr(39))}'"
            else:
                return f"'{str(v)}'"
        
        insert_sql = f"""INSERT INTO {table_name}
        (id, well_id, action, priority, status, reason, expected_impact, confidence, prediction_source, probability,
         oil_volume_slope, oil_volume_roll_mean, gas_volume_slope, gas_volume_roll_mean,
         motor_current_roll_mean, motor_temp_roll_mean, surface_pressure_roll_mean, details, created_at) 
        VALUES (
            {sql_value(rec_id)},
            {sql_value(rec.get('well_id'))},
            {sql_value(rec.get('action'))},
            {sql_value(rec.get('priority'))},
            {sql_value(rec.get('status'))},
            {sql_value(rec.get('reason'))},
            {sql_value(rec.get('expected_impact'))},
            {sql_value(rec.get('confidence'))},
            {sql_value(rec.get('prediction_source') or 'rule_based')},
            {sql_value(rec.get('probability'))},
            {sql_value(metrics.get('oil_volume_slope'))},
            {sql_value(metrics.get('oil_volume_roll_mean'))},
            {sql_value(metrics.get('gas_volume_slope'))},
            {sql_value(metrics.get('gas_volume_roll_mean'))},
            {sql_value(metrics.get('motor_current_roll_mean'))},
            {sql_value(metrics.get('motor_temp_roll_mean'))},
            {sql_value(metrics.get('surface_pressure_roll_mean'))},
            {sql_value(details)},
            {sql_value(now)}
        )"""
        
        logger.info(f"Inserting recommendation for {rec.get('well_id')}: {rec.get('action')}")
        cur.execute(insert_sql)
        conn.commit()
        logger.info(f"Successfully saved recommendation {rec_id}")
        return rec_id
    except Exception as e:
        logger.error(f"Error saving recommendation: {e}", exc_info=True)
        raise
    finally:
        cur.close()


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
        action_labels = ['NoAction', 'Increase ESP frequency', 'Adjust pump/polish choke', 'Optimize gas lift injection rate']
        label_distribution = {lbl: 0 for lbl in action_labels}
        
        for well, metrics in all_metrics:
            feats, _ = _features_from_metrics(metrics)
            X.append(feats)
            
            # compute z-scores for key metrics to determine action
            deviation_count = 0
            for key in ['motor_current_roll_mean', 'motor_temp_slope', 'surface_pressure_slope', 'oil_volume_slope', 'gas_volume_slope']:
                val = metrics.get(key)
                if val is not None and field_stds.get(key, 1.0) > 0:
                    z_score = abs((val - field_means.get(key, 0.0)) / (field_stds.get(key, 1.0) + 1e-6))
                    if z_score > 1.5:  # more than 1.5 std deviations
                        deviation_count += 1
            
            # assign action based on deviation count (more conservative)
            if deviation_count >= 3:
                label = 'Optimize gas lift injection rate'
            elif deviation_count >= 2:
                label = 'Adjust pump/polish choke'
            elif deviation_count >= 1:
                label = 'Increase ESP frequency'
            else:
                label = 'NoAction'
            
            y.append(label)
            label_distribution[label] += 1
        
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


def predict_all_wells(snowflake_conn, model_path: str = 'models/decision_tree.joblib', table_name: str = 'operation_recommendation', 
                     confidence_threshold: float = 0.6, dedup_days: int = 7):
    """Run the saved model for all wells and return a list of wells predicted to need action.
    
    Implements client requirements:
    - Only recommend on wells that are NOT in anomaly/alert state
    - Apply ML model with confidence_threshold (default 0.75)
    - Deduplicate: avoid duplicate well+action within dedup_days
    - Persist recommendations to unified `operation_recommendation` table in Snowflake
    
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
                # STEP 1: Pre-filter - skip wells with active anomalies/alerts
                # (Assuming well_anomalies table has alert_level, severity, or similar)
                try:
                    cur.execute("""
                        SELECT COUNT(*) FROM well_anomalies 
                        WHERE well_id = %s AND severity IN ('HIGH', 'CRITICAL')
                        AND detected_at > DATEADD(day, -7, CURRENT_TIMESTAMP())
                    """, (well,))
                    anomaly_count = cur.fetchone()[0]
                    if anomaly_count > 0:
                        logger.info(f"Skipping well {well}: {anomaly_count} active anomalies detected")
                        continue
                except Exception as e:
                    logger.warning(f"Could not check anomalies for well {well}: {e}")
                
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

                if not candidates:
                    logger.debug(f"Well {well}: no candidate actions above threshold {confidence_threshold}")
                    continue

                # persist each candidate
                for pred_action, proba in candidates:
                    logger.info(f"Well {well}: candidate action '{pred_action}' (prob={proba})")
                    # dedup check per action
                    try:
                        cutoff_date = f"DATEADD(day, -{dedup_days}, CURRENT_TIMESTAMP())"
                        cur.execute(f"""
                            SELECT COUNT(*) FROM {table_name}
                            WHERE well_id = %s AND action = %s AND created_at > {cutoff_date}
                        """, (well, pred_action))
                        recent_count = cur.fetchone()[0]
                        if recent_count > 0:
                            logger.info(f"Well {well}: duplicate recommendation for '{pred_action}' within {dedup_days} days, skipping")
                            continue
                    except Exception as e:
                        logger.warning(f"Deduplication check failed for well {well}: {e}")

                    results.append({'well_id': well, 'action': pred_action, 'probability': proba, 'metrics': metrics})
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
                        save_recommendation_to_snowflake(snowflake_conn, rec, table_name=table_name, metrics=metrics)
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
    keys = [
        'oil_volume_slope', 'motor_current_slope', 'motor_temp_slope', 'surface_pressure_slope', 'gas_volume_slope',
        'motor_current_roll_mean', 'motor_temp_roll_mean', 'surface_pressure_roll_mean', 'oil_volume_roll_mean', 'gas_volume_roll_mean',
        'missing_rate'
    ]
    features = []
    for k in keys:
        v = metrics.get(k)
        try:
            features.append(float(v) if v is not None else 0.0)
        except Exception:
            features.append(0.0)
    return np.array(features), keys


def train_decision_tree_from_history(snowflake_conn, events_table: str = 'operation_recommendation_events', model_path: str = 'models/decision_tree.joblib'):
    """Train a DecisionTreeClassifier from an events table.

    The `events_table` is expected to contain at least two columns:
      - pre_metrics (JSON or VARIANT) : JSON with same keys used by `_features_from_metrics`
      - action_label (VARCHAR) : categorical label for action taken/recommended

    The function saves the trained model to `model_path` and returns training metrics.
    """
    if DecisionTreeClassifier is None:
        raise RuntimeError('scikit-learn or joblib not available in the environment')

    cur = snowflake_conn.cursor()
    try:
        q = f"SELECT pre_metrics, action_label FROM {events_table} WHERE pre_metrics IS NOT NULL AND action_label IS NOT NULL"
        cur.execute(q)
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f'No training rows found in {events_table}')

        X = []
        y = []
        for pre_metrics, action_label in rows:
            # pre_metrics may be a string or VARIANT; ensure dict
            if isinstance(pre_metrics, str):
                try:
                    pm = json.loads(pre_metrics)
                except Exception:
                    pm = {}
            elif pre_metrics is None:
                pm = {}
            else:
                pm = dict(pre_metrics)

            feats, _ = _features_from_metrics(pm)
            X.append(feats)
            y.append(action_label)

        X = np.vstack(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # ensure model directory
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, model_path)

        return {'model_path': model_path, 'report': report}
    finally:
        cur.close()


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
