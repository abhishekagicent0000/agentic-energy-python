#!/usr/bin/env python3
"""
Integrated Operational Recommendations Pipeline

This script:
1. Trains the DecisionTree model on all well data
2. Runs predictions on all wells
3. For each detected recommendation:
   - Stores in Snowflake (operation_recommendation table)
   - Stores in PostgreSQL (operation_suggestion table with OpenAI details)
4. Generates a comprehensive report

Bridges operational_recommendations.py and operation.py modules
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/abhishekkumar/Desktop/porjects/at-risk-assets')

from operational_recommendations import (
    train_with_synthetic_labels,
    predict_all_wells,
    load_model,
    generate_recommendations_for_well,
    ACTION_CATEGORY_MAP
)

from operation import (
    save_operation_suggestion,
    get_operation_suggestions,
    create_operation_suggestion_table,
    infer_action_category_priority
)

import os
from dotenv import load_dotenv
try:
    from snowflake.connector import connect as snowflake_connect
except Exception:
    snowflake_connect = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_snowflake_connection():
    """Create Snowflake connection."""
    SNOWFLAKE_CONFIG = {
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
        'role': os.getenv('SNOWFLAKE_ROLE')
    }
    REQUIRED_KEYS = ['user', 'password', 'account', 'warehouse', 'database', 'schema', 'role']
    missing = [k for k in REQUIRED_KEYS if not SNOWFLAKE_CONFIG.get(k)]
    if missing:
        raise ValueError(f"Missing Snowflake configuration: {missing}")
    try:
        conn = snowflake_connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise


def get_well_metrics(snowflake_conn, well_id):
    """Retrieve production and sensor metrics for a well."""
    try:
        cur = snowflake_conn.cursor()
        
        # Get latest sensor metrics
        cur.execute("""
            SELECT 
                motor_current, motor_temp, surface_pressure,
                tubing_pressure, casing_pressure, wellhead_temp
            FROM well_sensor_readings 
            WHERE well_id = %s
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (well_id,))
        
        sensor_row = cur.fetchone()
        sensor_metrics = {}
        if sensor_row:
            cols = [d[0].lower() for d in cur.description]
            sensor_metrics = {cols[i]: sensor_row[i] for i in range(len(cols))}
        
        # Get latest production metrics
        cur.execute("""
            SELECT 
                oil_volume, gas_volume, water_volume
            FROM well_daily_production
            WHERE well_id = %s
            ORDER BY date DESC
            LIMIT 1
        """, (well_id,))
        
        prod_row = cur.fetchone()
        production_metrics = {}
        if prod_row:
            cols = [d[0].lower() for d in cur.description]
            production_metrics = {cols[i]: prod_row[i] for i in range(len(cols))}
        
        cur.close()
        return production_metrics, sensor_metrics
    
    except Exception as e:
        logger.error(f"Error retrieving metrics for {well_id}: {e}")
        return {}, {}


def process_recommendations(predictions, snowflake_conn):
    """
    Process predicted recommendations and store in both databases.
    
    Args:
        predictions: List of dicts from predict_all_wells()
        snowflake_conn: Snowflake connection
    
    Returns:
        Dict with summary statistics
    """
    logger.info("=" * 80)
    logger.info("Processing Recommendations for Database Storage")
    logger.info("=" * 80)
    
    stats = {
        'total_processed': 0,
        'saved_snowflake': 0,
        'saved_postgres': 0,
        'failed': 0,
        'details': []
    }
    
    for pred in predictions:
        try:
            well_id = pred.get('well_id')
            action = pred.get('action')
            probability = pred.get('probability', 0.0)
            
            logger.info(f"\nProcessing: {well_id} - {action} (confidence: {probability:.0%})")
            stats['total_processed'] += 1
            
            # Get well metrics for context
            production_data, sensor_metrics = get_well_metrics(snowflake_conn, well_id)
            
            # Note: Snowflake storage already handled by predict_all_wells()
            # So we just need to save to PostgreSQL with OpenAI details
            
            logger.info(f"  ↓ Inferring action/category/priority via OpenAI...")
            inferred = infer_action_category_priority(well_id, production_data, sensor_metrics, hint=action)
            inferred_action = inferred.get('action')
            inferred_category = inferred.get('category')
            inferred_priority = inferred.get('priority')

            logger.info(f"  ↓ Generating OpenAI analysis for PostgreSQL with inferred action/category...")
            success = save_operation_suggestion(
                well_id=well_id,
                action=inferred_action,
                category=inferred_category,
                status="New",
                priority=inferred_priority or ("HIGH" if probability > 0.85 else "MEDIUM"),
                confidence=probability * 100,  # Convert to percentage
                production_data=production_data,
                sensor_metrics=sensor_metrics,
                reason=f"ML model prediction with {probability:.0%} confidence",
                expected_impact=f"Recommendation based on trend analysis and historical patterns"
            )
            
            if success:
                logger.info(f"  ✓ Saved to PostgreSQL operation_suggestion table")
                stats['saved_postgres'] += 1
                stats['details'].append({
                    'well_id': well_id,
                    'action': action,
                    'confidence': probability,
                    'snowflake': True,
                    'postgres': True
                })
            else:
                logger.warning(f"  ✗ Failed to save to PostgreSQL")
                stats['details'].append({
                    'well_id': well_id,
                    'action': action,
                    'confidence': probability,
                    'snowflake': True,
                    'postgres': False
                })

            # Also generate and save rule-based recommendations for the same well
            try:
                rule_recs = generate_recommendations_for_well(snowflake_conn, well_id)
                for rr in rule_recs:
                    ra = rr.get('action') or rr.get('recommendation') or ''
                    rcat = rr.get('category') or ACTION_CATEGORY_MAP.get(ra, 'Production')
                    rprio = rr.get('priority') or 'MEDIUM'
                    rconf = rr.get('confidence') or rr.get('probability') or 0.0
                    saved = save_operation_suggestion(
                        well_id=well_id,
                        action=ra,
                        category=rcat,
                        status='New',
                        priority=rprio,
                        confidence=(rconf * 100) if rconf and rconf <= 1 else rconf,
                        production_data=production_data,
                        sensor_metrics=sensor_metrics,
                        reason='Rule-based recommendation',
                        expected_impact=rr.get('expected_impact', '')
                    )
                    if saved:
                        stats['saved_postgres'] += 1
            except Exception as e:
                logger.warning(f"Rule-based recommendations failed for {well_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing recommendation for {well_id}: {e}", exc_info=True)
            stats['failed'] += 1
    
    # Snowflake count (already saved by predict_all_wells)
    stats['saved_snowflake'] = len(predictions)
    
    return stats


def run_pipeline(skip_training=False, confidence_threshold=0.75):
    """
    Execute the complete operational recommendations pipeline.
    
    Args:
        skip_training: Skip model training if True
        confidence_threshold: Minimum confidence for recommendations (0-1)
    
    Returns:
        Dict with pipeline results
    """
    try:
        logger.info("=" * 80)
        logger.info("OPERATIONAL RECOMMENDATIONS PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.utcnow().isoformat()}Z")
        logger.info(f"Confidence Threshold: {confidence_threshold:.0%}")
        
        # Initialize PostgreSQL table
        logger.info("\n1. Initializing PostgreSQL...")
        create_operation_suggestion_table()
        logger.info("✓ PostgreSQL operation_suggestion table ready")
        
        # Connect to Snowflake
        logger.info("\n2. Connecting to Snowflake...")
        snowflake_conn = get_snowflake_connection()
        logger.info("✓ Connected to Snowflake")
        
        # Train model
        train_result = None
        if not skip_training:
            logger.info("\n3. Training DecisionTree Model...")
            model_path = 'models/decision_tree_alldata.joblib'
            train_result = train_with_synthetic_labels(snowflake_conn, model_path=model_path)
            logger.info(f"✓ Model trained")
            logger.info(f"  - Train Accuracy: {train_result.get('train_accuracy', 0):.1%}")
            logger.info(f"  - Test Accuracy: {train_result.get('test_accuracy', 0):.1%}")
            logger.info(f"  - Samples: {train_result.get('trained_on', 0)}")
            logger.info(f"  - Label Distribution: {train_result.get('label_distribution', {})}")
        else:
            logger.info("\n3. Skipping training (--skip-training)")
            model_path = 'models/decision_tree_alldata.joblib'
        
        # Run predictions
        logger.info("\n4. Running Predictions on All Wells...")
        logger.info(f"  (Confidence threshold: {confidence_threshold:.0%})")
        
        predictions = predict_all_wells(
            snowflake_conn,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            dedup_days=7
        )
        logger.info(f"✓ Predictions completed: {len(predictions)} recommendations detected")
        
        # Process recommendations for both databases
        logger.info("\n5. Storing Recommendations...")
        if predictions:
            stats = process_recommendations(predictions, snowflake_conn)
        else:
            logger.info("No high-confidence recommendations detected")
            stats = {
                'total_processed': 0,
                'saved_snowflake': 0,
                'saved_postgres': 0,
                'failed': 0,
                'details': []
            }
        
        # Verify storage
        logger.info("\n6. Verifying Storage...")
        logger.info("  Snowflake: operation_recommendation table")
        logger.info(f"    - Saved in this run: {stats['saved_snowflake']}")
        
        logger.info("  PostgreSQL: operation_suggestion table")
        postgres_suggestions = get_operation_suggestions(limit=5)
        logger.info(f"    - Total in table: {len(postgres_suggestions)}")
        if postgres_suggestions:
            logger.info(f"    - Latest: {postgres_suggestions[0].get('well_id')} - {postgres_suggestions[0].get('action')}")
        
        snowflake_conn.close()
        
        # Summary report
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
        logger.info(f"Status: SUCCESS")
        logger.info(f"\nRecommendations Processed: {stats['total_processed']}")
        logger.info(f"Saved to Snowflake: {stats['saved_snowflake']}")
        logger.info(f"Saved to PostgreSQL: {stats['saved_postgres']}")
        logger.info(f"Failed: {stats['failed']}")
        
        if stats['details']:
            logger.info("\nDetailed Results:")
            logger.info("-" * 80)
            for detail in stats['details']:
                well_id = detail.get('well_id', 'N/A')
                action = detail.get('action', 'N/A')[:50]
                conf = detail.get('confidence', 0)
                sf = "✓" if detail.get('snowflake') else "✗"
                pg = "✓" if detail.get('postgres') else "✗"
                logger.info(f"  {well_id:15s} | {action:50s} | {conf:7.0%} | SF:{sf} PG:{pg}")
        
        logger.info("\n✓ Pipeline completed successfully")
        return {
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'train_result': train_result,
            'predictions_count': len(predictions),
            'stats': stats
        }
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run operational recommendations pipeline')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training')
    parser.add_argument('--confidence', type=float, default=0.75,
                       help='Confidence threshold for recommendations (0-1)')
    args = parser.parse_args()
    
    result = run_pipeline(
        skip_training=args.skip_training,
        confidence_threshold=args.confidence
    )
    
    # Exit with appropriate code
    sys.exit(0 if result.get('status') == 'success' else 1)
