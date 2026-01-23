import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import psycopg2
from collections import defaultdict
from anomaly_review_utils import *

def prioritize_anomalies(anomalies: List[Dict]) -> List[Dict]:
    if not anomalies: return []
    
    ghost_dates = set()
    for a in anomalies:
        if a['anomaly_code'] == 'GHOST_PROD':
            ghost_dates.add(a['event_date'])
            
    cleaned_list = []
    
    for a in anomalies:
        code = a['anomaly_code']
        date = a['event_date']
        
        if date in ghost_dates and code in ['EFFICIENCY_DEGRADATION', 'NEW_PRODUCTION_BASELINE', 'MONTHLY_DECLINE']:
            continue
            
        has_baseline_crash = any(
            x['anomaly_code'] == 'NEW_PRODUCTION_BASELINE' and x['event_date'] == date 
            for x in anomalies
        )
        if has_baseline_crash and code == 'MONTHLY_DECLINE':
            continue

        cleaned_list.append(a)
        
    return cleaned_list

def detect_anomalies(well_id: str, df: Optional[pd.DataFrame] = None, lookback_days: int = DETECTION_WINDOW_DAYS) -> List[Dict]:
    logger.info(f"Fetching all historical data for {well_id} to train model...")
    full_df = fetch_well_data(well_id, days=None)
    
    if full_df.empty or len(full_df) < 30:
        logger.warning(f"Insufficient data for {well_id}")
        return []

    full_df['lift_type'] = full_df['lift_type'].ffill().fillna('Rod Pump')
    
    logger.info("Calculating rolling baselines (Explicit Time-Based)...")
    
    temp_df = full_df.set_index('timestamp').sort_index()
    
    full_df['oil_baseline_90d'] = temp_df['oil_volume'].rolling('90D').mean().values
    full_df['gas_baseline_90d'] = temp_df['gas_volume'].rolling('90D').mean().values
    full_df['oil_std_90d'] = temp_df['oil_volume'].rolling('90D').std().values
    
    full_df['oil_avg_30d'] = temp_df['oil_volume'].rolling('30D').mean().values
    full_df['gas_avg_30d'] = temp_df['gas_volume'].rolling('30D').mean().values
    full_df['water_avg_30d'] = temp_df['water_volume'].rolling('30D').mean().values
    
    full_df['oil_avg_7d'] = temp_df['oil_volume'].rolling('7D').mean().values
    full_df['gas_avg_7d'] = temp_df['gas_volume'].rolling('7D').mean().values
    
    full_df['tp_avg'] = temp_df['tubing_pressure'].rolling('30D').mean().values
    full_df['tp_std'] = temp_df['tubing_pressure'].rolling('3D').std().values 
    full_df['amps_avg'] = temp_df['motor_current'].rolling('30D').mean().values
    full_df['spm_avg'] = temp_df['strokes_per_minute'].rolling('30D').mean().values
    
    full_df['is_active'] = (full_df['strokes_per_minute'] > 0.1) | (full_df['motor_current'] > 5)

    logger.info(f"Training predictive model...")
    model_art = get_or_train_model(well_id, full_df)
    full_df['predicted_oil'] = np.nan
    
    if model_art:
        try:
            X = model_art['scaler'].transform(
                model_art['imputer'].transform(full_df[model_art['features']])
            )
            full_df['predicted_oil'] = model_art['model'].predict(X)
            full_df['efficiency'] = (full_df['oil_volume'] / full_df['predicted_oil'].replace(0, np.nan)) * 100
            
            temp_df['efficiency'] = full_df['efficiency'].values
            full_df['efficiency_avg_30d'] = temp_df['efficiency'].rolling('30D').mean().values
            
            full_df['efficiency'] = full_df['efficiency'].replace([np.inf, -np.inf], np.nan).fillna(100)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")

    all_checkers = [
        check_efficiency_degradation,
        check_sustained_production_change, 
        check_pressure_instability,
        lambda ctx: check_monthly_production_decline(ctx, full_df) if ctx.get('timestamp') == full_df['timestamp'].max() else None,
        check_ghost_production,
        lambda ctx: check_operational_pattern_shift(ctx, full_df)
    ]
    
    logger.info(f"Detecting anomalies in last {lookback_days} days...")
    window = full_df[full_df['timestamp'] > (full_df['timestamp'].max() - pd.Timedelta(days=lookback_days))]
    raw_findings = []
    
    for _, row in window.iterrows():
        ctx = row.to_dict()
        ctx['is_active'] = (ctx.get('strokes_per_minute',0) > 0.1) or (ctx.get('motor_current',0) > 5)
        
        ctx['actual_oil'] = ctx.get('oil_volume', 0)
        ctx['actual_gas'] = ctx.get('gas_volume', 0) 
        ctx['actual_water'] = ctx.get('water_volume', 0)
        ctx['avg_oil'] = ctx.get('oil_avg_30d', 0)
        ctx['avg_water'] = ctx.get('water_avg_30d', 0)
        ctx['pred_oil'] = ctx.get('predicted_oil', ctx['actual_oil'])
        if pd.isna(ctx['pred_oil']): ctx['pred_oil'] = ctx['actual_oil']
        
        for field in ['tubing_pressure', 'tp_avg', 'tp_std', 'spm', 'spm_avg', 'motor_current', 'amps_avg', 'oil_avg_7d', 'oil_baseline_90d', 'efficiency_avg_30d', 'gas_baseline_90d', 'gas_avg_30d', 'gas_avg_7d']:
            ctx[field] = ctx.get(field, 0)
            if pd.isna(ctx[field]): ctx[field] = 0

        for check in all_checkers:
            res = check(ctx)
            if res:
                severity = determine_severity(res['code'], res.get('severity_metrics', {}))
                raw_findings.append({
                    "well_id": well_id,
                    "event_date": str(row['timestamp'].date()),
                    "timestamp": row['timestamp'],
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "anomaly_code": res['code'],
                    "category": res['category'],
                    "severity": severity,
                    "raw_data": res['raw_data'],
                    "chart_metric": res['chart_metric'],
                    "impact_value": res['impact_value'],
                    "impact_payload": res['impact_payload'],
                    "lift_type": str(row['lift_type']),
                    "diagnosis": res.get('diagnosis'),
                    "severity_metrics": res.get('severity_metrics', {})
                })

    raw_findings = prioritize_anomalies(raw_findings)
    final_anomalies = []
    
    findings_by_code = defaultdict(list)
    for f in raw_findings:
        findings_by_code[f['anomaly_code']].append(f)
        
    for code, findings in findings_by_code.items():
        sorted_findings = sorted(findings, key=lambda x: x['timestamp'])
        latest_finding = sorted_findings[-1]

        history_str = get_recent_anomaly_history(well_id, code, exclude_date=latest_finding['event_date'])
        
        llm_content = generate_llm_content(
            latest_finding['raw_data'], 
            latest_finding['lift_type'], 
            latest_finding['chart_metric'], 
            history_str, 
            latest_finding['detected_at'],
            diagnosis=latest_finding.get('diagnosis')
        )
        
        chart_config = generate_chart_config(
            latest_finding['raw_data'], 
            latest_finding['chart_metric'], 
            full_df,
            anomaly_code=code,
            severity_metrics=latest_finding.get('severity_metrics', {})
        )
        
        final_anomalies.append({
            "well_id": well_id,
            "event_date": latest_finding['event_date'],
            "detected_at": latest_finding['detected_at'],
            "anomaly_code": code,
            "category": latest_finding['category'],
            "severity": latest_finding['severity'],
            "title": llm_content.get('title', 'Anomaly Detected'),
            "status": "ACTIVE",
            "ui_text": llm_content,
            "impact_metrics": latest_finding['impact_payload'],
            "impact_value": latest_finding['impact_value'],
            "chart_data": chart_config
        })

    logger.info(f"Detected {len(final_anomalies)} actionable anomalies for {well_id}")
    save_reviews(final_anomalies)
    return final_anomalies
    
def save_reviews(reviews):
    if not reviews: return

    def clean_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: clean_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_numpy(v) for v in obj]
        return obj

    cleaned_reviews = [clean_numpy(r) for r in reviews]
    
    try:
        url = get_db_url()
        if url:
            with psycopg2.connect(url) as conn:
                with conn.cursor() as cur:
                    q = """
                        INSERT INTO anomaly_review
                        (well_id, event_date, detected_at, anomaly_code, category, severity, 
                         title, ui_text, impact_value, impact_metrics, chart_data, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (well_id, anomaly_code, event_date) DO NOTHING
                    """
                    for r in cleaned_reviews:
                        cur.execute(q, (
                            r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                            r['category'], r['severity'], r['title'],
                            json.dumps(r['ui_text']),
                            r['impact_value'],               
                            json.dumps(r['impact_metrics']), 
                            json.dumps(r['chart_data']),     
                            r['status']
                        ))
                conn.commit()
    except Exception as e:
        logger.error(f"Postgres Save failed: {e}")

    try:
        with get_snowflake_conn() as conn:
            cur = conn.cursor()
            check_q = "SELECT count(*) FROM AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW WHERE well_id=%s AND anomaly_code=%s AND event_date=%s"
            insert_q = """
                INSERT INTO AT_RISK_ASSETS.SENSOR_DATA.ANOMALY_REVIEW
                (id, well_id, event_date, detected_at, anomaly_code, category, severity, 
                 title, ui_text, impact_value, impact_metrics, chart_data, status)
                SELECT UUID_STRING(), %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s, PARSE_JSON(%s), PARSE_JSON(%s), %s
            """
            for r in cleaned_reviews:
                cur.execute(check_q, (r['well_id'], r['anomaly_code'], r['event_date']))
                if cur.fetchone()[0] == 0:
                    cur.execute(insert_q, (
                        r['well_id'], r['event_date'], r['detected_at'], r['anomaly_code'],
                        r['category'], r['severity'], r['title'],
                        json.dumps(r['ui_text']),
                        r['impact_value'],               
                        json.dumps(r['impact_metrics']), 
                        json.dumps(r['chart_data']),     
                        r['status']
                    ))
            conn.commit()
    except Exception as e:
        logger.error(f"Snowflake Save failed: {e}")

if __name__ == "__main__":
    wid = sys.argv[1] if len(sys.argv) > 1 else "Well_001"
    print(json.dumps(detect_anomalies(wid), indent=2, default=str))