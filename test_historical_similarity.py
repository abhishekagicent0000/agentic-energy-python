#!/usr/bin/env python3
"""
Test Script for Historical Similarity Search Functionality

This script tests the find_similar_anomalies function to verify that
historical context finding is working correctly.
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import the function to test
from app import find_similar_anomalies, get_historical_anomalies, get_db_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HistoricalSimilarityTest')

load_dotenv()


def create_mock_historical_data():
    """Create mock historical anomaly data for testing."""
    mock_data = [
        {
            'well_id': 'WELL_001',
            'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
            'violation_summary': 'Motor Current High',
            'severity': 'HIGH',
            'raw_values': json.dumps({
                'motor_current': 85.0,
                'motor_temp': 240.0,
                'discharge_pressure': 1200.0,
                'pump_intake_pressure': 500.0,
                'motor_voltage': 480.0
            })
        },
        {
            'well_id': 'WELL_002',
            'timestamp': (datetime.now() - timedelta(days=10)).isoformat(),
            'violation_summary': 'Motor Temperature Spike',
            'severity': 'CRITICAL',
            'raw_values': json.dumps({
                'motor_current': 82.0,
                'motor_temp': 265.0,
                'discharge_pressure': 1150.0,
                'pump_intake_pressure': 480.0,
                'motor_voltage': 475.0
            })
        },
        {
            'well_id': 'WELL_003',
            'timestamp': (datetime.now() - timedelta(days=15)).isoformat(),
            'violation_summary': 'Pressure Anomaly',
            'severity': 'MEDIUM',
            'raw_values': json.dumps({
                'motor_current': 60.0,
                'motor_temp': 180.0,
                'discharge_pressure': 1500.0,
                'pump_intake_pressure': 600.0,
                'motor_voltage': 480.0
            })
        },
        {
            'well_id': 'WELL_001',
            'timestamp': (datetime.now() - timedelta(days=20)).isoformat(),
            'violation_summary': 'Rod Pump Torque High',
            'severity': 'HIGH',
            'raw_values': json.dumps({
                'strokes_per_minute': 12.0,
                'torque': 950.0,
                'polish_rod_load': 8500.0,
                'pump_fillage': 85.0,
                'tubing_pressure': 450.0
            })
        },
        {
            'well_id': 'WELL_004',
            'timestamp': (datetime.now() - timedelta(days=3)).isoformat(),
            'violation_summary': 'Motor Current High',
            'severity': 'HIGH',
            'raw_values': json.dumps({
                'motor_current': 87.0,
                'motor_temp': 245.0,
                'discharge_pressure': 1180.0,
                'pump_intake_pressure': 510.0,
                'motor_voltage': 478.0
            })
        }
    ]
    
    df = pd.DataFrame(mock_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def test_similarity_with_mock_data():
    """Test similarity search with mock data."""
    logger.info("=" * 70)
    logger.info("TEST 1: Similarity Search with Mock Data")
    logger.info("=" * 70)
    
    # Create current anomaly (similar to historical ESP motor current issue)
    current_readings = {
        'motor_current': 84.0,
        'motor_temp': 242.0,
        'discharge_pressure': 1190.0,
        'pump_intake_pressure': 505.0,
        'motor_voltage': 479.0
    }
    
    current_violations = [
        {'field': 'motor_current', 'value': 84.0, 'violation': 'Motor current exceeds 80A'},
        {'field': 'motor_temp', 'value': 242.0, 'violation': 'Motor temperature exceeds 240°F'}
    ]
    
    # Get mock historical data
    historical_df = create_mock_historical_data()
    
    logger.info(f"\nCurrent Readings: {json.dumps(current_readings, indent=2)}")
    logger.info(f"\nCurrent Violations: {json.dumps(current_violations, indent=2)}")
    logger.info(f"\nHistorical Records Available: {len(historical_df)}")
    
    # Test with different similarity thresholds
    for threshold in [0.8, 0.7, 0.6]:
        logger.info(f"\n--- Testing with similarity threshold: {threshold} ---")
        
        similar = find_similar_anomalies(
            current_readings=current_readings,
            current_violations=current_violations,
            historical_df=historical_df,
            lift_type='ESP',
            similarity_threshold=threshold
        )
        
        logger.info(f"Found {len(similar)} similar incidents:")
        for i, incident in enumerate(similar, 1):
            logger.info(f"\n  Incident {i}:")
            logger.info(f"    Well: {incident['well_id']}")
            logger.info(f"    Date: {incident['timestamp']}")
            logger.info(f"    Issue: {incident['alert_title']}")
            logger.info(f"    Severity: {incident['severity']}")
            logger.info(f"    Similarity Score: {incident['similarity_score']:.2%}")
            logger.info(f"    Match Count: {incident['match_count']}")
            
            if 'raw_anomaly_data' in incident and 'violations' in incident['raw_anomaly_data']:
                logger.info(f"    Violations:")
                for v in incident['raw_anomaly_data']['violations']:
                    logger.info(f"      - {v['field']}: {v['value']}")
    
    return len(similar) > 0


def test_similarity_with_real_database():
    """Test similarity search with real database data."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Similarity Search with Real Database")
    logger.info("=" * 70)
    
    try:
        # Fetch real historical anomalies
        logger.info("\nFetching historical anomalies from database...")
        historical_df = get_historical_anomalies(limit=50)
        
        if historical_df.empty:
            logger.warning("⚠ No historical anomalies found in database!")
            logger.info("Please ensure there are anomalies in the well_anomalies table.")
            return False
        
        logger.info(f"✓ Found {len(historical_df)} historical anomalies")
        
        # Display sample of historical data
        logger.info("\nSample of historical data:")
        for idx, row in historical_df.head(3).iterrows():
            logger.info(f"\n  Record {idx + 1}:")
            logger.info(f"    Well: {row.get('well_id', 'N/A')}")
            logger.info(f"    Timestamp: {row.get('timestamp', 'N/A')}")
            logger.info(f"    Severity: {row.get('severity', 'N/A')}")
            logger.info(f"    Violation: {row.get('violation_summary', 'N/A')}")
            
            # Parse raw_values
            try:
                raw_vals = row.get('raw_values', '{}')
                if isinstance(raw_vals, str):
                    raw_vals = json.loads(raw_vals)
                logger.info(f"    Readings: {list(raw_vals.keys())[:5]}")
            except:
                pass
        
        # Create a test current anomaly based on the first record
        if len(historical_df) > 0:
            first_record = historical_df.iloc[0]
            try:
                raw_vals = first_record['raw_values']
                if isinstance(raw_vals, str):
                    raw_vals = json.loads(raw_vals)
                
                # Create similar readings (with slight variations)
                current_readings = {}
                current_violations = []
                
                for key, value in list(raw_vals.items())[:5]:
                    if isinstance(value, (int, float)):
                        # Add 5% variation
                        varied_value = value * 1.05
                        current_readings[key] = varied_value
                        current_violations.append({
                            'field': key,
                            'value': varied_value,
                            'violation': f'{key} out of range'
                        })
                
                logger.info(f"\n--- Testing with synthetic current anomaly ---")
                logger.info(f"Current Readings: {json.dumps(current_readings, indent=2)}")
                
                similar = find_similar_anomalies(
                    current_readings=current_readings,
                    current_violations=current_violations,
                    historical_df=historical_df,
                    similarity_threshold=0.8
                )
                
                logger.info(f"\n✓ Found {len(similar)} similar incidents")
                for i, incident in enumerate(similar, 1):
                    logger.info(f"\n  Similar Incident {i}:")
                    logger.info(f"    Well: {incident['well_id']}")
                    logger.info(f"    Date: {incident['timestamp']}")
                    logger.info(f"    Similarity: {incident['similarity_score']:.2%}")
                    logger.info(f"    Matches: {incident['match_count']}")
                
                return len(similar) > 0
                
            except Exception as e:
                logger.error(f"Error processing historical record: {e}")
                return False
        
    except Exception as e:
        logger.error(f"Error in real database test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_edge_cases():
    """Test edge cases for similarity search."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Edge Cases")
    logger.info("=" * 70)
    
    # Test 1: Empty historical data
    logger.info("\n--- Test 3.1: Empty historical data ---")
    empty_df = pd.DataFrame()
    current_readings = {'motor_current': 80.0}
    current_violations = [{'field': 'motor_current', 'value': 80.0}]
    
    result = find_similar_anomalies(current_readings, current_violations, empty_df)
    logger.info(f"Result with empty history: {len(result)} matches (expected: 0)")
    assert len(result) == 0, "Should return empty list for empty historical data"
    
    # Test 2: No violations
    logger.info("\n--- Test 3.2: No current violations ---")
    historical_df = create_mock_historical_data()
    result = find_similar_anomalies(current_readings, [], historical_df)
    logger.info(f"Result with no violations: {len(result)} matches (expected: 0)")
    assert len(result) == 0, "Should return empty list when no violations"
    
    # Test 3: Different lift types (no common keys)
    logger.info("\n--- Test 3.3: Different lift types (no common sensors) ---")
    esp_readings = {'motor_current': 80.0, 'motor_temp': 240.0}
    esp_violations = [{'field': 'motor_current', 'value': 80.0}]
    
    # Historical data with only Rod Pump sensors
    rod_pump_data = pd.DataFrame([{
        'well_id': 'WELL_RP',
        'timestamp': datetime.now().isoformat(),
        'violation_summary': 'Torque High',
        'severity': 'HIGH',
        'raw_values': json.dumps({
            'strokes_per_minute': 12.0,
            'torque': 950.0,
            'polish_rod_load': 8500.0
        })
    }])
    
    result = find_similar_anomalies(esp_readings, esp_violations, rod_pump_data)
    logger.info(f"Result with different lift types: {len(result)} matches (expected: 0)")
    logger.info("✓ Should return 0 matches when lift types don't share sensors")
    
    logger.info("\n✓ All edge case tests passed!")
    return True


def test_similarity_threshold_behavior():
    """Test how different thresholds affect results."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Similarity Threshold Behavior")
    logger.info("=" * 70)
    
    historical_df = create_mock_historical_data()
    
    # Current anomaly with exact match to one historical record
    current_readings = {
        'motor_current': 85.0,
        'motor_temp': 240.0,
        'discharge_pressure': 1200.0,
        'pump_intake_pressure': 500.0,
        'motor_voltage': 480.0
    }
    
    current_violations = [
        {'field': 'motor_current', 'value': 85.0},
        {'field': 'motor_temp', 'value': 240.0}
    ]
    
    thresholds = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]
    
    logger.info("\nTesting different similarity thresholds:")
    logger.info(f"Current readings: motor_current=85.0, motor_temp=240.0")
    logger.info("\nThreshold | Matches Found")
    logger.info("-" * 30)
    
    for threshold in thresholds:
        result = find_similar_anomalies(
            current_readings, 
            current_violations, 
            historical_df, 
            similarity_threshold=threshold
        )
        logger.info(f"  {threshold:.2f}    |     {len(result)}")
    
    logger.info("\n✓ Threshold behavior test complete!")
    return True


def run_all_tests():
    """Run all test suites."""
    logger.info("\n" + "=" * 70)
    logger.info("HISTORICAL SIMILARITY SEARCH - COMPREHENSIVE TEST SUITE")
    logger.info("=" * 70)
    
    results = {
        'mock_data_test': False,
        'real_database_test': False,
        'edge_cases_test': False,
        'threshold_behavior_test': False
    }
    
    try:
        results['mock_data_test'] = test_similarity_with_mock_data()
    except Exception as e:
        logger.error(f"Mock data test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        results['real_database_test'] = test_similarity_with_real_database()
    except Exception as e:
        logger.error(f"Real database test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        results['edge_cases_test'] = test_edge_cases()
    except Exception as e:
        logger.error(f"Edge cases test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    try:
        results['threshold_behavior_test'] = test_similarity_threshold_behavior()
    except Exception as e:
        logger.error(f"Threshold behavior test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    logger.info("=" * 70)
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
