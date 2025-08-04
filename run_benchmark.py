#!/usr/bin/env python3
"""
Quick Benchmark Runner - IEEE-CIS Fraud Detection Project
Convenient script to run performance benchmarks with different configurations
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.performance_benchmarking import run_comprehensive_fraud_detection_benchmark
from src.benchmark_visualization import create_benchmark_visualizations
from src.automated_benchmark_runner import AutomatedBenchmarkRunner

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_sample_data(output_path: str, size: int = 10000):
    """Create sample fraud detection dataset for testing"""
    import pandas as pd
    import numpy as np
    
    print(f"Creating sample dataset with {size} records...")
    
    np.random.seed(42)
    
    # Generate synthetic fraud detection data
    data = {
        'TransactionID': range(size),
        'TransactionDT': np.random.randint(86400, 86400*30, size),  # 30 days worth
        'TransactionAmt': np.random.lognormal(3, 1.5, size),  # Log-normal distribution for amounts
        'ProductCD': np.random.choice(['W', 'C', 'R', 'H', 'S'], size),
        'card1': np.random.randint(1000, 20000, size),
        'card2': np.random.randint(100, 999, size),
        'card3': np.random.randint(100, 999, size),
        'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], size),
        'card5': np.random.randint(100, 300, size),
        'card6': np.random.choice(['debit', 'credit'], size),
        'addr1': np.random.randint(1, 500, size),
        'addr2': np.random.randint(1, 100, size),
        'C1': np.random.randint(0, 5000, size),
        'C2': np.random.randint(0, 5000, size),
        'C3': np.random.randint(0, 100, size),
        'C4': np.random.randint(0, 100, size),
        'C5': np.random.randint(0, 300, size),
        'D1': np.random.randint(-100, 500, size),
        'D2': np.random.randint(-100, 500, size),
        'D3': np.random.randint(-100, 500, size),
        'V1': np.random.randn(size),
        'V2': np.random.randn(size),
        'V3': np.random.randn(size),
        'V4': np.random.randn(size),
        'V5': np.random.randn(size),
        'id_01': np.random.randint(-10, 100, size),
        'id_02': np.random.randint(-100, 500, size),
        'id_03': np.random.randint(-100, 200, size),
        'id_30': np.random.choice(['Android', 'iOS', 'Windows'], size),
        'id_31': np.random.choice(['chrome', 'safari', 'firefox', 'edge'], size),
        'DeviceType': np.random.choice(['desktop', 'mobile'], size),
        'DeviceInfo': np.random.choice(['Windows', 'MacOS', 'iOS', 'Android'], size),
    }
    
    # Create fraud labels (imbalanced - ~3.5% fraud rate)
    fraud_rate = 0.035
    fraud_indices = np.random.choice(size, int(size * fraud_rate), replace=False)
    data['isFraud'] = np.zeros(size, dtype=int)
    data['isFraud'][fraud_indices] = 1
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Make fraudulent transactions slightly different patterns
    fraud_mask = df['isFraud'] == 1
    df.loc[fraud_mask, 'TransactionAmt'] *= np.random.uniform(1.5, 3.0, fraud_mask.sum())  # Higher amounts
    df.loc[fraud_mask, 'C1'] *= np.random.uniform(0.5, 1.5, fraud_mask.sum())  # Different C1 pattern
    
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created: {output_path}")
    print(f"Total records: {len(df):,}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    
    return output_path

def quick_benchmark(data_path: str, output_dir: str = "quick_benchmark_results"):
    """Run a quick benchmark with minimal models"""
    print("Running quick benchmark...")
    
    results = run_comprehensive_fraud_detection_benchmark(
        df=None,  # Will load from data_path
        models_to_test=['lightgbm', 'xgboost'],
        output_dir=output_dir
    )
    
    print("Quick benchmark completed!")
    return results

def full_benchmark(data_path: str, output_dir: str = "full_benchmark_results"):
    """Run full comprehensive benchmark"""
    print("Running full comprehensive benchmark...")
    
    results = run_comprehensive_fraud_detection_benchmark(
        df=None,  # Will load from data_path
        models_to_test=['lightgbm', 'xgboost', 'catboost', 'random_forest'],
        output_dir=output_dir
    )
    
    print("Full benchmark completed!")
    return results

def automated_benchmark(config_path: str, data_path: str = None):
    """Run automated benchmark with configuration"""
    print("Running automated benchmark...")
    
    runner = AutomatedBenchmarkRunner(config_path)
    results = runner.run_single_benchmark(data_path)
    
    print("Automated benchmark completed!")
    return results

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection Model Benchmark Runner")
    
    parser.add_argument('mode', choices=['quick', 'full', 'auto', 'create-data', 'visualize'],
                       help="Benchmark mode to run")
    
    parser.add_argument('--data', type=str, help="Path to dataset CSV file")
    parser.add_argument('--config', type=str, default="benchmark_config.yaml",
                       help="Path to configuration file (for auto mode)")
    parser.add_argument('--output', type=str, help="Output directory")
    parser.add_argument('--size', type=int, default=10000, 
                       help="Size of sample dataset (for create-data mode)")
    parser.add_argument('--results', type=str, help="Path to results JSON file (for visualize mode)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        if args.mode == 'create-data':
            # Create sample dataset
            output_path = args.data or f"sample_fraud_data_{args.size}.csv"
            create_sample_data(output_path, args.size)
            print(f"Sample dataset created: {output_path}")
            print("You can now run benchmarks with: python run_benchmark.py quick --data", output_path)
        
        elif args.mode == 'quick':
            # Quick benchmark
            if not args.data:
                print("Error: --data argument required for benchmark modes")
                return
            
            if not os.path.exists(args.data):
                print(f"Error: Dataset file not found: {args.data}")
                return
            
            output_dir = args.output or "quick_benchmark_results"
            results = quick_benchmark(args.data, output_dir)
            
            # Show summary
            if results.get('comparative_analysis', {}).get('overall_ranking'):
                best_model = results['comparative_analysis']['overall_ranking'][0]
                print(f"\n🏆 Best Model: {best_model['model_name']} (Score: {best_model['overall_score']:.3f})")
        
        elif args.mode == 'full':
            # Full benchmark  
            if not args.data:
                print("Error: --data argument required for benchmark modes")
                return
            
            if not os.path.exists(args.data):
                print(f"Error: Dataset file not found: {args.data}")
                return
            
            output_dir = args.output or "full_benchmark_results"
            results = full_benchmark(args.data, output_dir)
            
            # Show summary
            if results.get('comparative_analysis', {}).get('overall_ranking'):
                print("\n🏆 MODEL RANKINGS:")
                for i, model_info in enumerate(results['comparative_analysis']['overall_ranking'][:3]):
                    print(f"{i+1}. {model_info['model_name']}: {model_info['overall_score']:.3f}")
        
        elif args.mode == 'auto':
            # Automated benchmark
            if not os.path.exists(args.config):
                print(f"Error: Config file not found: {args.config}")
                return
            
            results = automated_benchmark(args.config, args.data)
            
            # Show summary with alerts
            violations = results.get('automated_run_metadata', {}).get('threshold_violations', [])
            regressions = results.get('automated_run_metadata', {}).get('regression_report', {})
            
            if violations:
                print(f"\n⚠️  Performance Violations: {len(violations)}")
                for v in violations[:3]:
                    print(f"   - {v['type']}: {v['model']} ({v['violation_percent']:.1f}% over threshold)")
            
            if regressions.get('regressions_detected'):
                print(f"\n🚨 Regressions Detected: {regressions['regression_count']}")
            
            if not violations and not regressions.get('regressions_detected'):
                print("\n✅ All performance checks passed!")
        
        elif args.mode == 'visualize':
            # Generate visualizations
            if not args.results:
                print("Error: --results argument required for visualize mode")
                return
            
            if not os.path.exists(args.results):
                print(f"Error: Results file not found: {args.results}")
                return
            
            viz_dir = args.output or "benchmark_visualizations"
            viz_paths = create_benchmark_visualizations(args.results, viz_dir)
            
            print("Visualizations created:")
            for name, path in viz_paths.items():
                if path:
                    print(f"  - {name}: {path}")
            
            index_path = os.path.join(viz_dir, "index.html")
            if os.path.exists(index_path):
                print(f"\n📊 Open visualization dashboard: {index_path}")
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()