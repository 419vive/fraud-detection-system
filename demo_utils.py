"""
Utility functions for the fraud detection demo app
==================================================

This module provides helper functions for data generation, model simulation,
and various utility functions needed by the demo application.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Generate realistic sample transaction data for demo purposes"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
        # Define realistic value ranges and distributions
        self.product_codes = ['W', 'C', 'R', 'H', 'S']
        self.product_weights = [0.4, 0.25, 0.15, 0.15, 0.05]  # W is most common
        
        self.card_types = ['visa', 'mastercard', 'american express', 'discover']
        self.card_type_weights = [0.45, 0.35, 0.15, 0.05]
        
        self.card_categories = ['debit', 'credit']
        self.card_category_weights = [0.6, 0.4]
        
        self.email_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'aol.com', 'icloud.com', 'company.com', 'university.edu'
        ]
        self.email_weights = [0.35, 0.20, 0.15, 0.12, 0.05, 0.05, 0.05, 0.03]
        
        # Fraud indicators
        self.suspicious_domains = [
            'temp-mail.org', '10minutemail.com', 'guerrillamail.com',
            'mailinator.com', 'tempmail.com'
        ]
        
    def generate_transaction_batch(self, n_samples: int = 1000, 
                                  fraud_rate: float = 0.035) -> pd.DataFrame:
        """Generate a batch of realistic transaction data"""
        
        transactions = []
        n_fraud = int(n_samples * fraud_rate)
        n_legitimate = n_samples - n_fraud
        
        # Generate legitimate transactions
        for i in range(n_legitimate):
            transaction = self._generate_legitimate_transaction(i + 1)
            transactions.append(transaction)
        
        # Generate fraudulent transactions
        for i in range(n_fraud):
            transaction = self._generate_fraudulent_transaction(n_legitimate + i + 1)
            transactions.append(transaction)
        
        # Shuffle the transactions
        np.random.shuffle(transactions)
        
        # Update transaction IDs to be sequential
        for i, transaction in enumerate(transactions):
            transaction['TransactionID'] = i + 1
        
        df = pd.DataFrame(transactions)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Generated {n_samples} transactions with {fraud_rate:.1%} fraud rate")
        
        return df
    
    def _generate_legitimate_transaction(self, transaction_id: int) -> Dict:
        """Generate a legitimate transaction"""
        
        # Normal business hours are more common for legitimate transactions
        hour = np.random.choice(
            range(24), 
            p=self._get_hourly_distribution(legitimate=True)
        )
        
        # Transaction amount follows log-normal distribution for legitimate transactions
        # Most transactions are small, but some larger ones exist
        amount = np.random.lognormal(mean=4.0, sigma=1.2)  # Mean around $55
        amount = max(1.0, min(amount, 10000.0))  # Clamp between $1 and $10,000
        
        # Product code affects typical amounts
        product_cd = np.random.choice(self.product_codes, p=self.product_weights)
        if product_cd == 'W':  # Web/digital products tend to be cheaper
            amount *= 0.7
        elif product_cd == 'H':  # Hardware tends to be more expensive
            amount *= 1.5
        
        transaction = {
            'TransactionID': transaction_id,
            'TransactionDT': self._generate_timestamp(hour),
            'TransactionAmt': round(amount, 2),
            'ProductCD': product_cd,
            'card1': np.random.randint(1000, 20000),
            'card2': np.random.randint(100, 600),
            'card3': np.random.randint(100, 300),
            'card4': np.random.choice(self.card_types, p=self.card_type_weights),
            'card5': np.random.randint(100, 250),
            'card6': np.random.choice(self.card_categories, p=self.card_category_weights),
            'addr1': np.random.randint(100, 500),
            'addr2': np.random.randint(10, 100),
            'dist1': np.random.exponential(50),  # Distance features
            'dist2': np.random.exponential(75),
            'P_emaildomain': np.random.choice(self.email_domains, p=self.email_weights),
            'R_emaildomain': np.random.choice(self.email_domains, p=self.email_weights),
            'isFraud': 0
        }
        
        # Add C features (count features) - normal ranges for legitimate
        for i in range(1, 15):
            transaction[f'C{i}'] = np.random.poisson(2)  # Low counts are normal
        
        # Add D features (timedelta features)
        for i in [1, 2, 3, 4, 5, 10, 15]:
            transaction[f'D{i}'] = np.random.exponential(100) if np.random.random() > 0.3 else np.nan
        
        # Add M features (match features) - mostly True for legitimate
        for i in range(1, 10):
            transaction[f'M{i}'] = np.random.choice(['T', 'F'], p=[0.8, 0.2])
        
        # Add V features (Vesta features) - normal distribution
        for i in range(1, 340):  # Vesta has many features
            if np.random.random() > 0.7:  # Only include some features
                transaction[f'V{i}'] = np.random.normal(0, 1)
        
        return transaction
    
    def _generate_fraudulent_transaction(self, transaction_id: int) -> Dict:
        """Generate a fraudulent transaction with suspicious patterns"""
        
        # Fraudulent transactions are more common at odd hours
        hour = np.random.choice(
            range(24),
            p=self._get_hourly_distribution(legitimate=False)
        )
        
        # Fraudulent transactions often have suspicious amounts
        fraud_type = np.random.choice(['card_testing', 'high_value', 'round_amount'], 
                                    p=[0.3, 0.5, 0.2])
        
        if fraud_type == 'card_testing':
            amount = np.random.uniform(1.0, 10.0)  # Small amounts for testing
        elif fraud_type == 'high_value':
            amount = np.random.uniform(500.0, 5000.0)  # High value fraud
        else:  # round_amount
            amount = np.random.choice([100, 200, 500, 1000, 1500, 2000])  # Round amounts
        
        transaction = {
            'TransactionID': transaction_id,
            'TransactionDT': self._generate_timestamp(hour),
            'TransactionAmt': round(amount, 2),
            'ProductCD': np.random.choice(self.product_codes),  # More random for fraud
            'card1': np.random.randint(1000, 20000),
            'card2': np.random.randint(100, 600),
            'card3': np.random.randint(100, 300),
            'card4': np.random.choice(self.card_types),
            'card5': np.random.randint(100, 250),
            'card6': np.random.choice(self.card_categories),
            'addr1': np.random.randint(100, 500),
            'addr2': np.random.randint(10, 100),
            'dist1': np.random.exponential(200),  # Higher distances can be suspicious
            'dist2': np.random.exponential(250),
            'isFraud': 1
        }
        
        # Email domains - mix of legitimate and suspicious
        if np.random.random() < 0.3:  # 30% use suspicious domains
            transaction['P_emaildomain'] = np.random.choice(self.suspicious_domains)
            transaction['R_emaildomain'] = np.random.choice(self.suspicious_domains)
        else:
            transaction['P_emaildomain'] = np.random.choice(self.email_domains)
            transaction['R_emaildomain'] = np.random.choice(self.email_domains)
        
        # C features - higher counts can indicate fraud
        for i in range(1, 15):
            if np.random.random() < 0.3:  # Some features have suspiciously high counts
                transaction[f'C{i}'] = np.random.poisson(10)  # Higher counts
            else:
                transaction[f'C{i}'] = np.random.poisson(2)
        
        # D features - some patterns in timing
        for i in [1, 2, 3, 4, 5, 10, 15]:
            if np.random.random() < 0.2:  # Some suspicious timing patterns
                transaction[f'D{i}'] = 0  # Immediate actions can be suspicious
            else:
                transaction[f'D{i}'] = np.random.exponential(50) if np.random.random() > 0.4 else np.nan
        
        # M features - more mismatches in fraud
        for i in range(1, 10):
            transaction[f'M{i}'] = np.random.choice(['T', 'F'], p=[0.5, 0.5])  # More random
        
        # V features - different distributions for fraud
        for i in range(1, 340):
            if np.random.random() > 0.8:  # Fewer features present
                transaction[f'V{i}'] = np.random.normal(0.5, 1.5)  # Different distribution
        
        return transaction
    
    def _get_hourly_distribution(self, legitimate: bool = True) -> List[float]:
        """Get probability distribution for transaction hours"""
        if legitimate:
            # Legitimate transactions follow business hours
            hours = np.array([
                0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM (low)
                0.03, 0.04, 0.05, 0.06, 0.07, 0.08,  # 6-11 AM (morning)
                0.09, 0.08, 0.07, 0.06, 0.05, 0.04,  # 12-5 PM (afternoon)
                0.03, 0.04, 0.05, 0.04, 0.03, 0.02   # 6-11 PM (evening)
            ])
        else:
            # Fraudulent transactions more common at odd hours
            hours = np.array([
                0.06, 0.07, 0.08, 0.06, 0.04, 0.03,  # 0-5 AM (high)
                0.02, 0.02, 0.03, 0.04, 0.05, 0.05,  # 6-11 AM 
                0.04, 0.04, 0.04, 0.04, 0.04, 0.04,  # 12-5 PM
                0.03, 0.03, 0.04, 0.05, 0.06, 0.07   # 6-11 PM (high)
            ])
        
        return hours / hours.sum()  # Normalize to sum to 1
    
    def _generate_timestamp(self, hour: int) -> float:
        """Generate timestamp for given hour"""
        # Random day in the last 30 days
        days_ago = np.random.randint(0, 30)
        base_time = datetime.now() - timedelta(days=days_ago)
        
        # Set to specific hour with random minutes/seconds
        target_time = base_time.replace(
            hour=hour,
            minute=np.random.randint(0, 60),
            second=np.random.randint(0, 60)
        )
        
        return target_time.timestamp()
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features commonly used in fraud detection"""
        
        # Time-based features
        df['hour'] = ((df['TransactionDT'] / 3600) % 24).astype(int)
        df['day_of_week'] = ((df['TransactionDT'] / 86400) % 7).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Amount-based features
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        df['is_round_amount'] = (df['TransactionAmt'] % 1 == 0).astype(int)
        df['is_small_amount'] = (df['TransactionAmt'] < 10).astype(int)
        df['is_large_amount'] = (df['TransactionAmt'] > 1000).astype(int)
        
        # Email domain features
        df['P_email_is_suspicious'] = df['P_emaildomain'].isin(self.suspicious_domains).astype(int)
        df['R_email_is_suspicious'] = df['R_emaildomain'].isin(self.suspicious_domains).astype(int)
        df['emails_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)
        
        # Card features
        df['is_american_express'] = (df['card4'] == 'american express').astype(int)
        df['is_credit_card'] = (df['card6'] == 'credit').astype(int)
        
        return df

class ModelSimulator:
    """Simulate model predictions for demo purposes"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Define feature importance weights for different "models"
        self.model_weights = {
            'lightgbm': {
                'TransactionAmt': 0.15,
                'TransactionAmt_log': 0.12,
                'hour': 0.08,
                'is_weekend': 0.06,
                'P_email_is_suspicious': 0.10,
                'is_round_amount': 0.07,
                'card4_encoded': 0.05,
                'C1': 0.04,
                'C2': 0.04,
                'D1': 0.03,
                'dist1': 0.03,
                'is_large_amount': 0.08,
                'V1': 0.02,
                'base_risk': 0.17  # Base fraud probability
            },
            'xgboost': {
                'TransactionAmt': 0.13,
                'TransactionAmt_log': 0.14,
                'hour': 0.09,
                'is_weekend': 0.05,
                'P_email_is_suspicious': 0.11,
                'is_round_amount': 0.06,
                'card4_encoded': 0.06,
                'C1': 0.05,
                'C2': 0.03,
                'D1': 0.04,
                'dist1': 0.02,
                'is_large_amount': 0.09,
                'V1': 0.03,
                'base_risk': 0.15
            },
            'random_forest': {
                'TransactionAmt': 0.12,
                'TransactionAmt_log': 0.10,
                'hour': 0.07,
                'is_weekend': 0.04,
                'P_email_is_suspicious': 0.09,
                'is_round_amount': 0.05,
                'card4_encoded': 0.04,
                'C1': 0.03,
                'C2': 0.03,
                'D1': 0.02,
                'dist1': 0.02,
                'is_large_amount': 0.07,
                'V1': 0.02,
                'base_risk': 0.30  # More conservative
            }
        }
    
    def predict_fraud_probability(self, transaction_data: Dict, 
                                model_name: str = 'lightgbm') -> Tuple[float, Dict]:
        """Simulate fraud probability prediction"""
        
        if model_name not in self.model_weights:
            model_name = 'lightgbm'  # Default fallback
        
        weights = self.model_weights[model_name]
        
        # Calculate weighted score based on features
        score = weights['base_risk']
        feature_contributions = {}
        
        # Transaction amount contribution
        if 'TransactionAmt' in transaction_data:
            amt = transaction_data['TransactionAmt']
            if amt < 5:  # Very small amounts are suspicious
                amt_score = 0.8
            elif amt > 2000:  # Very large amounts are suspicious
                amt_score = 0.7
            elif amt > 1000:
                amt_score = 0.4
            else:
                amt_score = 0.1
            
            contribution = weights['TransactionAmt'] * amt_score
            score += contribution
            feature_contributions['TransactionAmt'] = contribution
        
        # Time-based features
        if 'hour' in transaction_data:
            hour = transaction_data.get('hour', (transaction_data.get('TransactionDT', 0) / 3600) % 24)
            if hour < 6 or hour > 22:  # Late night/early morning
                time_score = 0.7
            elif 9 <= hour <= 17:  # Business hours
                time_score = 0.2
            else:
                time_score = 0.4
            
            contribution = weights['hour'] * time_score
            score += contribution
            feature_contributions['hour'] = contribution
        
        # Email domain
        if 'P_emaildomain' in transaction_data:
            domain = transaction_data['P_emaildomain']
            suspicious_domains = ['temp-mail.org', '10minutemail.com', 'guerrillamail.com']
            
            if domain in suspicious_domains:
                email_score = 0.9
            elif domain in ['gmail.com', 'yahoo.com']:
                email_score = 0.2
            else:
                email_score = 0.4
            
            contribution = weights['P_email_is_suspicious'] * email_score
            score += contribution
            feature_contributions['P_emaildomain'] = contribution
        
        # Round amount detection
        if 'TransactionAmt' in transaction_data:
            amt = transaction_data['TransactionAmt']
            is_round = amt % 1 == 0 and amt in [100, 200, 500, 1000, 1500, 2000]
            round_score = 0.6 if is_round else 0.1
            
            contribution = weights['is_round_amount'] * round_score
            score += contribution
            feature_contributions['is_round_amount'] = contribution
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05)
        score += noise
        
        # Ensure score is between 0 and 1
        score = max(0, min(1, score))
        
        return score, feature_contributions
    
    def get_model_performance_metrics(self, model_name: str) -> Dict:
        """Get simulated performance metrics for a model"""
        
        base_metrics = {
            'lightgbm': {
                'accuracy': 0.9891,
                'precision': 0.8234,
                'recall': 0.7654,
                'f1_score': 0.7932,
                'roc_auc': 0.9456,
                'training_time_min': 12.3,
                'inference_time_ms': 2.1
            },
            'xgboost': {
                'accuracy': 0.9883,
                'precision': 0.8145,
                'recall': 0.7543,
                'f1_score': 0.7831,
                'roc_auc': 0.9423,
                'training_time_min': 15.7,
                'inference_time_ms': 3.2
            },
            'random_forest': {
                'accuracy': 0.9867,
                'precision': 0.7923,
                'recall': 0.7321,
                'f1_score': 0.7618,
                'roc_auc': 0.9289,
                'training_time_min': 8.9,
                'inference_time_ms': 1.8
            },
            'catboost': {
                'accuracy': 0.9879,
                'precision': 0.8089,
                'recall': 0.7445,
                'f1_score': 0.7758,
                'roc_auc': 0.9378,
                'training_time_min': 18.2,
                'inference_time_ms': 4.1
            },
            'logistic': {
                'accuracy': 0.9723,
                'precision': 0.7234,
                'recall': 0.6789,
                'f1_score': 0.7001,
                'roc_auc': 0.8934,
                'training_time_min': 2.1,
                'inference_time_ms': 0.5
            }
        }
        
        return base_metrics.get(model_name, base_metrics['lightgbm'])

def generate_sample_config():
    """Generate sample configuration file for the demo"""
    config = {
        "demo_settings": {
            "default_model": "lightgbm",
            "fraud_threshold": 0.5,
            "max_batch_size": 10000,
            "enable_explanations": True
        },
        "model_settings": {
            "available_models": ["lightgbm", "xgboost", "random_forest", "catboost", "logistic"],
            "model_refresh_interval": 3600,
            "fallback_model": "random_forest"
        },
        "ui_settings": {
            "default_page": "Dashboard Overview",
            "enable_advanced_features": True,
            "max_display_rows": 1000
        },
        "business_settings": {
            "average_transaction_value": 150.0,
            "investigation_cost": 25.0,
            "chargeback_cost": 75.0,
            "system_cost_monthly": 50000.0
        }
    }
    
    return config

def save_demo_config(config: Dict, filepath: str = "demo_config.json"):
    """Save demo configuration to file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Demo configuration saved to {filepath}")

def load_demo_config(filepath: str = "demo_config.json") -> Dict:
    """Load demo configuration from file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        logger.info(f"Demo configuration loaded from {filepath}")
        return config
    else:
        logger.info("No configuration file found, using defaults")
        return generate_sample_config()

def create_sample_datasets():
    """Create sample datasets for demo purposes"""
    generator = SampleDataGenerator()
    
    # Create different sized datasets
    datasets = {
        'small': generator.generate_transaction_batch(100, fraud_rate=0.05),
        'medium': generator.generate_transaction_batch(1000, fraud_rate=0.035),
        'large': generator.generate_transaction_batch(10000, fraud_rate=0.03)
    }
    
    # Save datasets
    os.makedirs('demo_data', exist_ok=True)
    
    for name, df in datasets.items():
        filepath = f'demo_data/sample_transactions_{name}.csv'
        df.to_csv(filepath, index=False)
        logger.info(f"Sample dataset '{name}' saved to {filepath}")
    
    return datasets

if __name__ == "__main__":
    # Test the utility functions
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    generator = SampleDataGenerator()
    sample_df = generator.generate_transaction_batch(100)
    print(f"Generated {len(sample_df)} transactions")
    print(f"Fraud rate: {sample_df['isFraud'].mean():.2%}")
    
    # Test model simulator
    simulator = ModelSimulator()
    test_transaction = {
        'TransactionAmt': 1500.0,
        'hour': 2,
        'P_emaildomain': 'temp-mail.org'
    }
    
    prob, contributions = simulator.predict_fraud_probability(test_transaction)
    print(f"Fraud probability: {prob:.3f}")
    print(f"Feature contributions: {contributions}")
    
    # Create sample config
    config = generate_sample_config()
    save_demo_config(config)
    
    print("Demo utilities tested successfully!")