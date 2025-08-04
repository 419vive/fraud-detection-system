#!/usr/bin/env python3
"""
Test script for the IEEE-CIS Fraud Detection Demo Application
============================================================

This script tests the core functionality of the demo application
to ensure everything works correctly before deployment.
"""

import sys
import logging
import traceback
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        import streamlit
        logger.info(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        logger.error(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas
        logger.info(f"✅ Pandas {pandas.__version__}")
    except ImportError as e:
        logger.error(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        logger.info(f"✅ NumPy {numpy.__version__}")
    except ImportError as e:
        logger.error(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import plotly
        logger.info(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        logger.error(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import sklearn
        logger.info(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        logger.error(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_demo_utils():
    """Test the demo utilities module"""
    logger.info("Testing demo utilities...")
    
    try:
        from demo_utils import SampleDataGenerator, ModelSimulator
        
        # Test data generation
        generator = SampleDataGenerator()
        sample_df = generator.generate_transaction_batch(50)
        
        if len(sample_df) != 50:
            logger.error(f"❌ Expected 50 transactions, got {len(sample_df)}")
            return False
        
        required_columns = ['TransactionID', 'TransactionAmt', 'isFraud']
        missing_columns = [col for col in required_columns if col not in sample_df.columns]
        
        if missing_columns:
            logger.error(f"❌ Missing required columns: {missing_columns}")
            return False
        
        logger.info("✅ Sample data generation works")
        
        # Test model simulation
        simulator = ModelSimulator()
        test_transaction = {
            'TransactionAmt': 100.0,
            'hour': 14,
            'P_emaildomain': 'gmail.com'
        }
        
        prob, contributions = simulator.predict_fraud_probability(test_transaction)
        
        if not (0 <= prob <= 1):
            logger.error(f"❌ Invalid probability: {prob}")
            return False
        
        if not isinstance(contributions, dict):
            logger.error(f"❌ Invalid contributions type: {type(contributions)}")
            return False
        
        logger.info("✅ Model simulation works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo utils test failed: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist"""
    logger.info("Testing file structure...")
    
    required_files = [
        "demo_app.py",
        "demo_utils.py",
        "run_demo.py",
        "demo_requirements.txt",
        "README_DEMO.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    logger.info("✅ All required files present")
    return True

def test_demo_app_components():
    """Test that the demo app components can be loaded"""
    logger.info("Testing demo app components...")
    
    try:
        # Add current directory to path so we can import demo_app
        sys.path.insert(0, str(Path.cwd()))
        
        from demo_app import FraudDetectionDemo
        
        # Test demo app initialization
        demo = FraudDetectionDemo()
        
        if not hasattr(demo, 'sample_data'):
            logger.error("❌ Demo app missing sample_data attribute")
            return False
        
        if not hasattr(demo, 'models'):
            logger.error("❌ Demo app missing models attribute")
            return False
        
        # Test that sample data is a DataFrame
        if not isinstance(demo.sample_data, pd.DataFrame):
            logger.error(f"❌ Sample data should be DataFrame, got {type(demo.sample_data)}")
            return False
        
        if len(demo.sample_data) == 0:
            logger.error("❌ Sample data is empty")
            return False
        
        logger.info(f"✅ Demo app loaded with {len(demo.sample_data)} sample transactions")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo app component test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading and validation"""
    logger.info("Testing configuration...")
    
    try:
        from demo_utils import generate_sample_config, save_demo_config, load_demo_config
        
        # Test config generation
        config = generate_sample_config()
        
        required_sections = ['demo_settings', 'model_settings', 'ui_settings', 'business_settings']
        missing_sections = [section for section in required_sections if section not in config]
        
        if missing_sections:
            logger.error(f"❌ Missing config sections: {missing_sections}")
            return False
        
        # Test config save/load
        test_config_file = "test_config.json"
        save_demo_config(config, test_config_file)
        
        if not Path(test_config_file).exists():
            logger.error("❌ Config file was not saved")
            return False
        
        loaded_config = load_demo_config(test_config_file)
        
        if loaded_config != config:
            logger.error("❌ Loaded config doesn't match saved config")
            return False
        
        # Cleanup
        Path(test_config_file).unlink()
        
        logger.info("✅ Configuration system works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_data_generation():
    """Test sample data generation with different parameters"""
    logger.info("Testing data generation...")
    
    try:
        from demo_utils import SampleDataGenerator
        
        generator = SampleDataGenerator()
        
        # Test different fraud rates
        test_cases = [
            (100, 0.01),   # Low fraud rate
            (100, 0.05),   # Medium fraud rate  
            (100, 0.10),   # High fraud rate
        ]
        
        for n_samples, fraud_rate in test_cases:
            df = generator.generate_transaction_batch(n_samples, fraud_rate)
            
            actual_fraud_rate = df['isFraud'].mean()
            expected_fraud = n_samples * fraud_rate
            actual_fraud = df['isFraud'].sum()
            
            # Allow some tolerance for randomness
            if abs(actual_fraud - expected_fraud) > expected_fraud * 0.3:
                logger.warning(f"⚠️ Fraud rate deviation: expected ~{expected_fraud}, got {actual_fraud}")
            
            # Check data quality
            if df['TransactionAmt'].min() <= 0:
                logger.error("❌ Invalid transaction amounts (negative or zero)")
                return False
            
            if df['TransactionID'].nunique() != len(df):
                logger.error("❌ Duplicate transaction IDs")
                return False
        
        logger.info("✅ Data generation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data generation test failed: {e}")
        traceback.print_exc()
        return False

def test_model_simulation():
    """Test model simulation accuracy and consistency"""
    logger.info("Testing model simulation...")
    
    try:
        from demo_utils import ModelSimulator
        
        simulator = ModelSimulator()
        
        # Test different transaction types
        test_transactions = [
            # Legitimate transaction
            {
                'TransactionAmt': 50.0,
                'hour': 14,
                'P_emaildomain': 'gmail.com'
            },
            # Suspicious transaction
            {
                'TransactionAmt': 2000.0,
                'hour': 3,
                'P_emaildomain': 'temp-mail.org'
            },
            # Card testing
            {
                'TransactionAmt': 1.0,
                'hour': 4,
                'P_emaildomain': 'guerrillamail.com'
            }
        ]
        
        for i, transaction in enumerate(test_transactions):
            prob, contributions = simulator.predict_fraud_probability(transaction)
            
            # Check probability bounds
            if not (0 <= prob <= 1):
                logger.error(f"❌ Invalid probability for transaction {i}: {prob}")
                return False
            
            # Check contributions
            if not isinstance(contributions, dict):
                logger.error(f"❌ Invalid contributions type for transaction {i}")
                return False
            
            # Test consistency - same input should give similar output
            prob2, _ = simulator.predict_fraud_probability(transaction)
            if abs(prob - prob2) > 0.1:  # Allow some randomness
                logger.warning(f"⚠️ Inconsistent predictions: {prob} vs {prob2}")
        
        # Test different models
        for model_name in ['lightgbm', 'xgboost', 'random_forest']:
            metrics = simulator.get_model_performance_metrics(model_name)
            
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            missing_metrics = [m for m in required_metrics if m not in metrics]
            
            if missing_metrics:
                logger.error(f"❌ Missing metrics for {model_name}: {missing_metrics}")
                return False
        
        logger.info("✅ Model simulation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model simulation test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and return overall result"""
    logger.info("=" * 50)
    logger.info("Running IEEE-CIS Fraud Detection Demo Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Demo Utils Test", test_demo_utils),
        ("Demo App Components Test", test_demo_app_components),
        ("Configuration Test", test_configuration),
        ("Data Generation Test", test_data_generation),
        ("Model Simulation Test", test_model_simulation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("=" * 50)
    
    if failed == 0:
        logger.info("🎉 All tests passed! Demo app is ready to run.")
        return True
    else:
        logger.error(f"💥 {failed} test(s) failed. Please fix issues before running demo.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)