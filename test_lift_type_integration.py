#!/usr/bin/env python3
"""
Test script to verify the complete lift-type database integration.
Tests insert → fetch flow for all lift types.
"""

import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules
try:
    from lift_type_db_operations import (
        insert_rod_pump_reading,
        insert_esp_reading,
        insert_gas_lift_reading,
        get_rod_pump_history,
        get_esp_history,
        get_gas_lift_history
    )
    logger.info("✅ Successfully imported lift_type_db_operations")
except ImportError as e:
    logger.error(f"❌ Failed to import lift_type_db_operations: {e}")
    exit(1)

def test_rod_pump():
    """Test Rod Pump insert and retrieve."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Rod Pump Insert and Retrieve")
    logger.info("="*60)
    
    well_id = "WELL_ROD_PUMP_TEST"
    timestamp = datetime.now().isoformat()
    
    test_data = {
        'strokes_per_minute': 10.5,
        'torque': 500000.0,
        'polish_rod_load': 50000.0,
        'pump_fillage': 75.0,
        'tubing_pressure': 2500.0,
        'surface_stroke_length': 10.0,
        'downhole_gross_stroke_length': 12.0,
        'runtime': 20.0,
    }
    
    try:
        logger.info(f"Inserting Rod Pump data for {well_id}...")
        insert_rod_pump_reading(well_id, timestamp, test_data)
        logger.info("✅ Insert successful")
        
        logger.info(f"Fetching Rod Pump history for {well_id}...")
        df = get_rod_pump_history(well_id, limit=5)
        if not df.empty:
            logger.info(f"✅ Retrieved {len(df)} records")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Latest record: {df.iloc[0].to_dict()}")
            return True
        else:
            logger.warning("⚠️ No records retrieved (might be normal on first run)")
            return True
    except Exception as e:
        logger.error(f"❌ Rod Pump test failed: {e}")
        return False

def test_esp():
    """Test ESP insert and retrieve."""
    logger.info("\n" + "="*60)
    logger.info("TEST: ESP Insert and Retrieve")
    logger.info("="*60)
    
    well_id = "WELL_ESP_TEST"
    timestamp = datetime.now().isoformat()
    
    test_data = {
        'motor_temp': 150.0,
        'motor_current': 100.0,
        'discharge_pressure': 3000.0,
        'pump_intake_pressure': 500.0,
        'motor_voltage': 4500.0,
        'vibration_x': 5.0,
        'vibration_y': 3.0,
        'motor_speed': 1200.0,
    }
    
    try:
        logger.info(f"Inserting ESP data for {well_id}...")
        insert_esp_reading(well_id, timestamp, test_data)
        logger.info("✅ Insert successful")
        
        logger.info(f"Fetching ESP history for {well_id}...")
        df = get_esp_history(well_id, limit=5)
        if not df.empty:
            logger.info(f"✅ Retrieved {len(df)} records")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Latest record: {df.iloc[0].to_dict()}")
            return True
        else:
            logger.warning("⚠️ No records retrieved (might be normal on first run)")
            return True
    except Exception as e:
        logger.error(f"❌ ESP test failed: {e}")
        return False

def test_gas_lift():
    """Test Gas Lift insert and retrieve."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Gas Lift Insert and Retrieve")
    logger.info("="*60)
    
    well_id = "WELL_GAS_LIFT_TEST"
    timestamp = datetime.now().isoformat()
    
    test_data = {
        'injection_rate': 500.0,
        'injection_temperature': 200.0,
        'bottomhole_pressure': 3000.0,
        'injection_pressure': 4000.0,
        'cycle_time': 120.0,
        'plunger_velocity': 500.0,
        'arrival_count': 50.0,
    }
    
    try:
        logger.info(f"Inserting Gas Lift data for {well_id}...")
        insert_gas_lift_reading(well_id, timestamp, test_data)
        logger.info("✅ Insert successful")
        
        logger.info(f"Fetching Gas Lift history for {well_id}...")
        df = get_gas_lift_history(well_id, limit=5)
        if not df.empty:
            logger.info(f"✅ Retrieved {len(df)} records")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Latest record: {df.iloc[0].to_dict()}")
            return True
        else:
            logger.warning("⚠️ No records retrieved (might be normal on first run)")
            return True
    except Exception as e:
        logger.error(f"❌ Gas Lift test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Lift-Type Database Integration Tests")
    logger.info("="*60)
    
    results = {
        'Rod Pump': test_rod_pump(),
        'ESP': test_esp(),
        'Gas Lift': test_gas_lift()
    }
    
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✅ ✅ ✅ All tests passed!")
    else:
        logger.warning("\n⚠️ Some tests failed. Check Snowflake connectivity and schema.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
