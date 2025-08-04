"""
Automated Benchmark Runner - IEEE-CIS Fraud Detection Project
Automated benchmarking system with CI/CD integration and regression detection
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import yaml
from dataclasses import dataclass, asdict
import schedule
import threading

# Import benchmarking components
from .performance_benchmarking import (
    PerformanceBenchmarkSuite, 
    run_comprehensive_fraud_detection_benchmark
)
from .benchmark_visualization import create_benchmark_visualizations
from .optimized_modeling import OptimizedFraudDetectionModel
from .config import get_config

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for automated benchmarking"""
    dataset_path: str = "data/sample_data.csv"
    target_column: str = "isFraud"
    test_size: float = 0.2
    models_to_test: List[str] = None
    output_directory: str = "benchmark_results"
    visualization_directory: str = "benchmark_visualizations"
    
    # Performance thresholds for alerts
    max_training_time_minutes: float = 30.0
    min_roc_auc: float = 0.85
    max_inference_latency_ms: float = 100.0
    min_throughput_per_second: int = 100
    
    # Regression detection
    enable_regression_detection: bool = True
    regression_threshold_percent: float = 5.0
    historical_results_keep_days: int = 30
    
    # Automation settings
    enable_scheduled_runs: bool = False
    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM
    enable_alerts: bool = True
    alert_webhook_url: Optional[str] = None
    alert_email_recipients: List[str] = None
    
    def __post_init__(self):
        if self.models_to_test is None:
            self.models_to_test = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
        if self.alert_email_recipients is None:
            self.alert_email_recipients = []

class PerformanceRegression:
    """Performance regression detection system"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.historical_data_path = os.path.join(
            config.output_directory, "historical_results.json"
        )
        self.baseline_metrics_path = os.path.join(
            config.output_directory, "baseline_metrics.json"
        )
    
    def load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical benchmark results"""
        if os.path.exists(self.historical_data_path):
            try:
                with open(self.historical_data_path, 'r') as f:
                    data = json.load(f)
                
                # Filter recent data within keep_days
                cutoff_date = datetime.now() - timedelta(days=self.config.historical_results_keep_days)
                recent_data = []
                
                for result in data:
                    result_date = datetime.fromisoformat(result['benchmark_metadata']['timestamp'])
                    if result_date >= cutoff_date:
                        recent_data.append(result)
                
                return recent_data
            except Exception as e:
                logger.warning(f"Failed to load historical data: {e}")
        
        return []
    
    def save_benchmark_result(self, result: Dict[str, Any]):
        """Save benchmark result to historical data"""
        historical_data = self.load_historical_data()
        historical_data.append(result)
        
        try:
            os.makedirs(os.path.dirname(self.historical_data_path), exist_ok=True)
            with open(self.historical_data_path, 'w') as f:
                json.dump(historical_data, f, indent=2, default=str)
            logger.info(f"Benchmark result saved to historical data")
        except Exception as e:
            logger.error(f"Failed to save benchmark result: {e}")
    
    def detect_regression(self, current_result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance regressions compared to historical data"""
        logger.info("Detecting performance regressions...")
        
        historical_data = self.load_historical_data()
        if len(historical_data) < 2:
            logger.info("Insufficient historical data for regression detection")
            return {"regressions_detected": False, "message": "Insufficient historical data"}
        
        # Calculate baseline metrics from recent history
        baseline_metrics = self._calculate_baseline_metrics(historical_data[:-1])  # Exclude current
        
        # Compare current results against baseline
        regressions = []
        improvements = []
        
        # Training performance regression
        if current_result.get('training_benchmarks'):
            training_regressions = self._check_training_regression(
                current_result['training_benchmarks'], baseline_metrics
            )
            regressions.extend(training_regressions)
        
        # Accuracy regression
        if current_result.get('accuracy_benchmarks'):
            accuracy_regressions = self._check_accuracy_regression(
                current_result['accuracy_benchmarks'], baseline_metrics
            )
            regressions.extend(accuracy_regressions)
        
        # Inference performance regression  
        if current_result.get('inference_benchmarks'):
            inference_regressions = self._check_inference_regression(
                current_result['inference_benchmarks'], baseline_metrics
            )
            regressions.extend(inference_regressions)
        
        regression_report = {
            "regressions_detected": len(regressions) > 0,
            "regression_count": len(regressions),
            "regressions": regressions,
            "improvements": improvements,
            "baseline_metrics": baseline_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        if regressions:
            logger.warning(f"Performance regressions detected: {len(regressions)} issues found")
        else:
            logger.info("No performance regressions detected")
        
        return regression_report
    
    def _calculate_baseline_metrics(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate baseline metrics from historical data"""
        baseline = {}
        
        # Training metrics
        training_times = []
        memory_usage = []
        
        for result in historical_data:
            if result.get('training_benchmarks'):
                for benchmark in result['training_benchmarks']:
                    training_times.append(benchmark['training_time'])
                    memory_usage.append(benchmark['memory_usage_peak'])
        
        if training_times:
            baseline['training_time_mean'] = np.mean(training_times)
            baseline['training_time_std'] = np.std(training_times)
            baseline['memory_usage_mean'] = np.mean(memory_usage)
            baseline['memory_usage_std'] = np.std(memory_usage)
        
        # Accuracy metrics
        roc_aucs = []
        f1_scores = []
        
        for result in historical_data:
            if result.get('accuracy_benchmarks'):
                for benchmark in result['accuracy_benchmarks']:
                    roc_aucs.append(benchmark['roc_auc'])
                    f1_scores.append(benchmark['f1_score'])
        
        if roc_aucs:
            baseline['roc_auc_mean'] = np.mean(roc_aucs)
            baseline['roc_auc_std'] = np.std(roc_aucs)
            baseline['f1_score_mean'] = np.mean(f1_scores)
            baseline['f1_score_std'] = np.std(f1_scores)
        
        # Inference metrics
        throughputs = []
        latencies = []
        
        for result in historical_data:
            if result.get('inference_benchmarks'):
                for benchmark in result['inference_benchmarks']:
                    throughputs.append(benchmark['throughput'])
                    latencies.append(benchmark['latency_p95'])
        
        if throughputs:
            baseline['throughput_mean'] = np.mean(throughputs)
            baseline['throughput_std'] = np.std(throughputs)
            baseline['latency_p95_mean'] = np.mean(latencies)
            baseline['latency_p95_std'] = np.std(latencies)
        
        return baseline
    
    def _check_training_regression(self, current_training: List[Dict], baseline: Dict) -> List[Dict]:
        """Check for training performance regressions"""
        regressions = []
        threshold = self.config.regression_threshold_percent / 100
        
        for benchmark in current_training:
            model_name = benchmark['model_name']
            
            # Training time regression
            if 'training_time_mean' in baseline:
                current_time = benchmark['training_time']
                baseline_time = baseline['training_time_mean']
                
                if current_time > baseline_time * (1 + threshold):
                    regression_percent = ((current_time - baseline_time) / baseline_time) * 100
                    regressions.append({
                        "type": "training_time_regression",
                        "model": model_name,
                        "metric": "training_time",
                        "current_value": current_time,
                        "baseline_value": baseline_time,
                        "regression_percent": regression_percent,
                        "severity": "high" if regression_percent > 20 else "medium"
                    })
            
            # Memory usage regression
            if 'memory_usage_mean' in baseline:
                current_memory = benchmark['memory_usage_peak']
                baseline_memory = baseline['memory_usage_mean']
                
                if current_memory > baseline_memory * (1 + threshold):
                    regression_percent = ((current_memory - baseline_memory) / baseline_memory) * 100
                    regressions.append({
                        "type": "memory_usage_regression",
                        "model": model_name,
                        "metric": "memory_usage_peak",
                        "current_value": current_memory,
                        "baseline_value": baseline_memory,
                        "regression_percent": regression_percent,
                        "severity": "medium" if regression_percent > 10 else "low"
                    })
        
        return regressions
    
    def _check_accuracy_regression(self, current_accuracy: List[Dict], baseline: Dict) -> List[Dict]:
        """Check for accuracy regressions"""
        regressions = []
        threshold = self.config.regression_threshold_percent / 100
        
        for benchmark in current_accuracy:
            model_name = benchmark['model_name']
            
            # ROC-AUC regression
            if 'roc_auc_mean' in baseline:
                current_auc = benchmark['roc_auc']
                baseline_auc = baseline['roc_auc_mean']
                
                if current_auc < baseline_auc * (1 - threshold):
                    regression_percent = ((baseline_auc - current_auc) / baseline_auc) * 100
                    regressions.append({
                        "type": "accuracy_regression",
                        "model": model_name,
                        "metric": "roc_auc",
                        "current_value": current_auc,
                        "baseline_value": baseline_auc,
                        "regression_percent": regression_percent,
                        "severity": "critical" if regression_percent > 5 else "high"
                    })
            
            # F1 Score regression
            if 'f1_score_mean' in baseline:
                current_f1 = benchmark['f1_score']
                baseline_f1 = baseline['f1_score_mean']
                
                if current_f1 < baseline_f1 * (1 - threshold):
                    regression_percent = ((baseline_f1 - current_f1) / baseline_f1) * 100
                    regressions.append({
                        "type": "f1_regression",
                        "model": model_name,
                        "metric": "f1_score",
                        "current_value": current_f1,
                        "baseline_value": baseline_f1,
                        "regression_percent": regression_percent,
                        "severity": "high" if regression_percent > 3 else "medium"
                    })
        
        return regressions
    
    def _check_inference_regression(self, current_inference: List[Dict], baseline: Dict) -> List[Dict]:
        """Check for inference performance regressions"""
        regressions = []
        threshold = self.config.regression_threshold_percent / 100
        
        # Group by model and get best performance
        model_best_performance = {}
        for benchmark in current_inference:
            model_name = benchmark['model_name']
            if model_name not in model_best_performance or benchmark['throughput'] > model_best_performance[model_name]['throughput']:
                model_best_performance[model_name] = benchmark
        
        for model_name, benchmark in model_best_performance.items():
            # Throughput regression
            if 'throughput_mean' in baseline:
                current_throughput = benchmark['throughput']
                baseline_throughput = baseline['throughput_mean']
                
                if current_throughput < baseline_throughput * (1 - threshold):
                    regression_percent = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
                    regressions.append({
                        "type": "throughput_regression",
                        "model": model_name,
                        "metric": "throughput",
                        "current_value": current_throughput,
                        "baseline_value": baseline_throughput,
                        "regression_percent": regression_percent,
                        "severity": "high" if regression_percent > 10 else "medium"
                    })
            
            # Latency regression
            if 'latency_p95_mean' in baseline:
                current_latency = benchmark['latency_p95']
                baseline_latency = baseline['latency_p95_mean']
                
                if current_latency > baseline_latency * (1 + threshold):
                    regression_percent = ((current_latency - baseline_latency) / baseline_latency) * 100
                    regressions.append({
                        "type": "latency_regression",
                        "model": model_name,
                        "metric": "latency_p95",
                        "current_value": current_latency,
                        "baseline_value": baseline_latency,
                        "regression_percent": regression_percent,
                        "severity": "high" if regression_percent > 15 else "medium"
                    })
        
        return regressions

class AlertManager:
    """Alert and notification management"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def send_regression_alert(self, regression_report: Dict[str, Any]):
        """Send alert for performance regressions"""
        if not self.config.enable_alerts or not regression_report.get("regressions_detected"):
            return
        
        logger.info("Sending regression alert...")
        
        # Prepare alert message
        message = self._format_regression_message(regression_report)
        
        # Send via webhook
        if self.config.alert_webhook_url:
            self._send_webhook_alert(message)
        
        # Send via email
        if self.config.alert_email_recipients:
            self._send_email_alert(message)
        
        logger.info("Regression alerts sent")
    
    def send_threshold_violation_alert(self, violations: List[Dict[str, Any]]):
        """Send alert for threshold violations"""
        if not self.config.enable_alerts or not violations:
            return
        
        logger.info("Sending threshold violation alert...")
        
        message = self._format_violation_message(violations)
        
        if self.config.alert_webhook_url:
            self._send_webhook_alert(message)
        
        if self.config.alert_email_recipients:
            self._send_email_alert(message)
    
    def _format_regression_message(self, regression_report: Dict[str, Any]) -> str:
        """Format regression report as alert message"""
        lines = [
            "🚨 PERFORMANCE REGRESSION DETECTED 🚨",
            f"Timestamp: {regression_report['timestamp']}",
            f"Regressions Found: {regression_report['regression_count']}",
            "",
            "REGRESSION DETAILS:"
        ]
        
        for regression in regression_report['regressions']:
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠', 
                'medium': '🟡',
                'low': '🟢'
            }.get(regression['severity'], '⚪')
            
            lines.extend([
                f"{severity_emoji} {regression['type'].upper()}",
                f"  Model: {regression['model']}",
                f"  Metric: {regression['metric']}",
                f"  Current: {regression['current_value']:.4f}",
                f"  Baseline: {regression['baseline_value']:.4f}",
                f"  Regression: {regression['regression_percent']:.2f}%",
                ""
            ])
        
        lines.append("Please investigate and address these performance issues.")
        
        return "\n".join(lines)
    
    def _format_violation_message(self, violations: List[Dict[str, Any]]) -> str:
        """Format threshold violations as alert message"""
        lines = [
            "⚠️ PERFORMANCE THRESHOLD VIOLATIONS ⚠️",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Violations Found: {len(violations)}",
            "",
            "VIOLATION DETAILS:"
        ]
        
        for violation in violations:
            lines.extend([
                f"❌ {violation['type'].upper()}",
                f"  Model: {violation['model']}",
                f"  Current: {violation['current_value']:.4f}",
                f"  Threshold: {violation['threshold']:.4f}",
                f"  Violation: {violation['violation_percent']:.2f}%",
                ""
            ])
        
        return "\n".join(lines)
    
    def _send_webhook_alert(self, message: str):
        """Send alert via webhook"""
        try:
            import requests
            
            payload = {
                "text": message,
                "username": "Fraud Detection Benchmark Bot",
                "icon_emoji": ":warning:"
            }
            
            response = requests.post(self.config.alert_webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Webhook alert sent successfully")
            else:
                logger.error(f"Webhook alert failed: {response.status_code}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_email_alert(self, message: str):
        """Send alert via email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # This would need SMTP configuration
            logger.info("Email alert functionality available (requires SMTP config)")
            # Implementation would depend on specific email service
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class AutomatedBenchmarkRunner:
    """Main automated benchmark runner"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = self._load_config_from_file(config_path)
        else:
            self.config = BenchmarkConfig()
        
        # Initialize components
        self.regression_detector = PerformanceRegression(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Setup directories
        os.makedirs(self.config.output_directory, exist_ok=True)
        os.makedirs(self.config.visualization_directory, exist_ok=True)
        
        logger.info("Automated Benchmark Runner initialized")
    
    def _load_config_from_file(self, config_path: str) -> BenchmarkConfig:
        """Load configuration from YAML or JSON file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    import yaml
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            return BenchmarkConfig(**config_dict)
        
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return BenchmarkConfig()
    
    def run_single_benchmark(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Run a single comprehensive benchmark"""
        logger.info("Starting automated benchmark run...")
        
        # Use provided dataset or default
        data_path = dataset_path or self.config.dataset_path
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        # Check for required columns
        if self.config.target_column not in df.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in dataset")
        
        # Run comprehensive benchmark
        benchmark_results = run_comprehensive_fraud_detection_benchmark(
            df=df,
            target_col=self.config.target_column,
            test_size=self.config.test_size,
            models_to_test=self.config.models_to_test,
            output_dir=self.config.output_directory
        )
        
        # Check performance thresholds
        threshold_violations = self._check_performance_thresholds(benchmark_results)
        
        # Detect regressions
        regression_report = None
        if self.config.enable_regression_detection:
            regression_report = self.regression_detector.detect_regression(benchmark_results)
            self.regression_detector.save_benchmark_result(benchmark_results)
        
        # Generate visualizations
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.config.output_directory, 
                f"benchmark_results_{timestamp}.json"
            )
            
            if os.path.exists(results_file):
                viz_paths = create_benchmark_visualizations(
                    results_file, 
                    self.config.visualization_directory
                )
                benchmark_results['visualization_paths'] = viz_paths
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
        
        # Send alerts
        if threshold_violations:
            self.alert_manager.send_threshold_violation_alert(threshold_violations)
        
        if regression_report and regression_report.get('regressions_detected'):
            self.alert_manager.send_regression_alert(regression_report)
        
        # Add metadata
        benchmark_results['automated_run_metadata'] = {
            'config': asdict(self.config),
            'threshold_violations': threshold_violations,
            'regression_report': regression_report,
            'run_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Automated benchmark run completed")
        return benchmark_results
    
    def _check_performance_thresholds(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check performance against configured thresholds"""
        violations = []
        
        # Training time threshold
        if results.get('training_benchmarks'):
            for benchmark in results['training_benchmarks']:
                training_time_minutes = benchmark['training_time'] / 60
                if training_time_minutes > self.config.max_training_time_minutes:
                    violations.append({
                        "type": "training_time_violation",
                        "model": benchmark['model_name'],
                        "current_value": training_time_minutes,
                        "threshold": self.config.max_training_time_minutes,
                        "violation_percent": ((training_time_minutes - self.config.max_training_time_minutes) / self.config.max_training_time_minutes) * 100
                    })
        
        # Accuracy threshold
        if results.get('accuracy_benchmarks'):
            for benchmark in results['accuracy_benchmarks']:
                if benchmark['roc_auc'] < self.config.min_roc_auc:
                    violations.append({
                        "type": "accuracy_violation",
                        "model": benchmark['model_name'],
                        "current_value": benchmark['roc_auc'],
                        "threshold": self.config.min_roc_auc,
                        "violation_percent": ((self.config.min_roc_auc - benchmark['roc_auc']) / self.config.min_roc_auc) * 100
                    })
        
        # Inference performance thresholds
        if results.get('inference_benchmarks'):
            # Group by model and check best performance
            model_best_perf = {}
            for benchmark in results['inference_benchmarks']:
                model_name = benchmark['model_name']
                if model_name not in model_best_perf or benchmark['throughput'] > model_best_perf[model_name]['throughput']:
                    model_best_perf[model_name] = benchmark
            
            for model_name, benchmark in model_best_perf.items():
                # Latency violation
                if benchmark['latency_p95'] > self.config.max_inference_latency_ms:
                    violations.append({
                        "type": "latency_violation",
                        "model": model_name,
                        "current_value": benchmark['latency_p95'],
                        "threshold": self.config.max_inference_latency_ms,
                        "violation_percent": ((benchmark['latency_p95'] - self.config.max_inference_latency_ms) / self.config.max_inference_latency_ms) * 100
                    })
                
                # Throughput violation
                if benchmark['throughput'] < self.config.min_throughput_per_second:
                    violations.append({
                        "type": "throughput_violation",
                        "model": model_name,
                        "current_value": benchmark['throughput'],
                        "threshold": self.config.min_throughput_per_second,
                        "violation_percent": ((self.config.min_throughput_per_second - benchmark['throughput']) / self.config.min_throughput_per_second) * 100
                    })
        
        if violations:
            logger.warning(f"Performance threshold violations detected: {len(violations)}")
        
        return violations
    
    def setup_scheduled_runs(self):
        """Setup scheduled benchmark runs"""
        if not self.config.enable_scheduled_runs:
            logger.info("Scheduled runs are disabled")
            return
        
        logger.info(f"Setting up scheduled benchmark runs: {self.config.schedule_cron}")
        
        # Simple daily schedule (for more complex cron, would need additional library)
        schedule.every().day.at("02:00").do(self._scheduled_benchmark_job)
        
        # Run scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduled benchmark runs configured")
    
    def _scheduled_benchmark_job(self):
        """Scheduled benchmark job"""
        try:
            logger.info("Running scheduled benchmark...")
            self.run_single_benchmark()
            logger.info("Scheduled benchmark completed successfully")
        
        except Exception as e:
            logger.error(f"Scheduled benchmark failed: {e}")
            
            # Send failure alert
            if self.config.enable_alerts:
                message = f"Scheduled benchmark failed: {str(e)}"
                try:
                    if self.config.alert_webhook_url:
                        self.alert_manager._send_webhook_alert(message)
                except:
                    pass
    
    def run_ci_cd_benchmark(self, dataset_path: str, commit_hash: str = None, 
                           branch_name: str = None) -> Dict[str, Any]:
        """Run benchmark for CI/CD integration"""
        logger.info("Running CI/CD benchmark...")
        
        # Add CI/CD specific metadata
        ci_metadata = {
            "commit_hash": commit_hash or os.environ.get('GITHUB_SHA', 'unknown'),
            "branch_name": branch_name or os.environ.get('GITHUB_REF_NAME', 'unknown'),
            "build_number": os.environ.get('GITHUB_RUN_NUMBER', 'unknown'),
            "ci_system": "github_actions",  # or detect automatically
            "trigger": "automated_ci"
        }
        
        # Run benchmark
        results = self.run_single_benchmark(dataset_path)
        results['ci_cd_metadata'] = ci_metadata
        
        # Generate CI/CD specific outputs
        self._generate_ci_cd_outputs(results)
        
        return results
    
    def _generate_ci_cd_outputs(self, results: Dict[str, Any]):
        """Generate CI/CD specific output files"""
        # Generate performance summary for CI/CD
        summary_path = os.path.join(self.config.output_directory, "ci_cd_summary.txt")
        
        summary_lines = []
        summary_lines.append("=== FRAUD DETECTION MODEL BENCHMARK SUMMARY ===")
        
        # Overall ranking
        if results.get('comparative_analysis', {}).get('overall_ranking'):
            best_model = results['comparative_analysis']['overall_ranking'][0]
            summary_lines.append(f"Best Overall Model: {best_model['model_name']} (Score: {best_model['overall_score']:.3f})")
        
        # Performance violations
        violations = results.get('automated_run_metadata', {}).get('threshold_violations', [])
        if violations:
            summary_lines.append(f"Performance Violations: {len(violations)}")
            for violation in violations[:3]:  # Show top 3
                summary_lines.append(f"  - {violation['type']}: {violation['model']} ({violation['violation_percent']:.1f}% over threshold)")
        
        # Regressions
        regression_report = results.get('automated_run_metadata', {}).get('regression_report')
        if regression_report and regression_report.get('regressions_detected'):
            summary_lines.append(f"Regressions Detected: {regression_report['regression_count']}")
        
        # Key metrics
        if results.get('accuracy_benchmarks'):
            best_auc = max(b['roc_auc'] for b in results['accuracy_benchmarks'])
            summary_lines.append(f"Best ROC-AUC: {best_auc:.4f}")
        
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"CI/CD summary generated: {summary_path}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Automated Fraud Detection Model Benchmark Runner")
    
    parser.add_argument('--config', type=str, help="Path to configuration file")
    parser.add_argument('--dataset', type=str, help="Path to dataset file")
    parser.add_argument('--mode', choices=['single', 'scheduled', 'ci-cd'], 
                       default='single', help="Benchmark mode")
    parser.add_argument('--commit-hash', type=str, help="Git commit hash (for CI/CD mode)")
    parser.add_argument('--branch', type=str, help="Git branch name (for CI/CD mode)")
    parser.add_argument('--output-dir', type=str, help="Output directory")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize runner
        runner = AutomatedBenchmarkRunner(args.config)
        
        # Override config with CLI args
        if args.output_dir:
            runner.config.output_directory = args.output_dir
        
        if args.mode == 'single':
            results = runner.run_single_benchmark(args.dataset)
            print("Benchmark completed successfully!")
            
            if results.get('comparative_analysis', {}).get('overall_ranking'):
                best_model = results['comparative_analysis']['overall_ranking'][0]
                print(f"Best Model: {best_model['model_name']} (Score: {best_model['overall_score']:.3f})")
        
        elif args.mode == 'scheduled':
            runner.setup_scheduled_runs()
            print("Scheduled benchmark runner started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Stopping scheduled runner...")
        
        elif args.mode == 'ci-cd':
            if not args.dataset:
                raise ValueError("Dataset path required for CI/CD mode")
            
            results = runner.run_ci_cd_benchmark(
                args.dataset, 
                args.commit_hash, 
                args.branch
            )
            
            print("CI/CD benchmark completed!")
            
            # Exit with error code if regressions detected
            regression_report = results.get('automated_run_metadata', {}).get('regression_report')
            if regression_report and regression_report.get('regressions_detected'):
                print("Performance regressions detected!")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Benchmark runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()