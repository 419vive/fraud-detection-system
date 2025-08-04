"""
Performance Benchmarking System - IEEE-CIS Fraud Detection Project
Comprehensive benchmarking framework for ML model performance analysis
"""

import pandas as pd
import numpy as np
import time
import logging
import json
import pickle
import psutil
import gc
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing as mp
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_curve, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Optional performance monitoring
try:
    import gpustat
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profile = lambda x: x

from .config import get_config
from .memory_optimizer import MemoryProfiler, optimize_memory_usage
from .performance_optimizer import InferenceOptimizer
from .optimized_modeling import OptimizedFraudDetectionModel

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    model_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    dataset_size: int
    hardware_info: Dict[str, Any]
    optimization_level: str = "none"
    additional_info: Dict[str, Any] = None

@dataclass
class TrainingBenchmark:
    """Training performance benchmark"""
    model_name: str
    training_time: float
    memory_usage_peak: float
    memory_usage_average: float
    cpu_usage_peak: float
    cpu_usage_average: float
    gpu_usage_peak: float = 0.0
    gpu_usage_average: float = 0.0
    dataset_size: int = 0
    feature_count: int = 0
    convergence_iteration: int = 0
    final_score: float = 0.0

@dataclass
class InferenceBenchmark:
    """Inference performance benchmark"""
    model_name: str
    batch_size: int
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    predictions_per_second: float
    optimization_level: str = "none"

@dataclass
class AccuracyBenchmark:
    """Model accuracy benchmark"""
    model_name: str
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score_mean: float
    cross_val_score_std: float
    training_score: float
    validation_score: float

class SystemMonitor:
    """System resource monitoring utility"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics = defaultdict(list)
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.metrics.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.debug("System monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        # Calculate statistics
        stats = {}
        for metric, values in self.metrics.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'std': np.std(values),
                    'samples': len(values)
                }
        
        logger.debug("System monitoring stopped")
        return stats
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            try:
                # CPU and Memory
                memory_info = MemoryProfiler.get_memory_usage()
                cpu_percent = psutil.cpu_percent(interval=None)
                
                self.metrics['memory_rss_gb'].append(memory_info['rss_gb'])
                self.metrics['memory_percent'].append(memory_info['percent'])
                self.metrics['cpu_percent'].append(cpu_percent)
                
                # GPU monitoring if available
                if GPU_AVAILABLE:
                    try:
                        gpu_stats = gpustat.new_query()
                        for gpu in gpu_stats:
                            self.metrics[f'gpu_{gpu.index}_utilization'].append(gpu.utilization)
                            self.metrics[f'gpu_{gpu.index}_memory_used'].append(gpu.memory_used)
                    except:
                        pass
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                time.sleep(self.interval)

class ModelTrainingBenchmark:
    """Model training performance benchmark"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.results = []
        self.monitor = SystemMonitor()
    
    def benchmark_training_speed(self, models: Dict[str, Any], X_train: pd.DataFrame, 
                                y_train: pd.Series, X_val: Optional[pd.DataFrame] = None,
                                y_val: Optional[pd.Series] = None) -> List[TrainingBenchmark]:
        """Benchmark training speed across different models"""
        logger.info("Starting training speed benchmark...")
        
        results = []
        
        for model_name, model_class_or_instance in models.items():
            logger.info(f"Benchmarking training speed for {model_name}")
            
            # Start monitoring
            self.monitor.start_monitoring()
            initial_memory = MemoryProfiler.get_memory_usage()['rss_gb']
            
            start_time = time.time()
            
            try:
                # Train model
                if callable(model_class_or_instance):
                    # If it's a class or function, call it
                    model = model_class_or_instance()
                    if hasattr(model, 'fit'):
                        if X_val is not None and y_val is not None:
                            # Use validation set if available
                            model.fit(X_train, y_train, 
                                    eval_set=[(X_val, y_val)] if hasattr(model, 'eval_set') else None)
                        else:
                            model.fit(X_train, y_train)
                else:
                    # If it's already an instance
                    model = model_class_or_instance
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                
                # Stop monitoring and collect metrics
                system_stats = self.monitor.stop_monitoring()
                final_memory = MemoryProfiler.get_memory_usage()['rss_gb']
                
                # Calculate convergence iteration
                convergence_iter = 0
                if hasattr(model, 'best_iteration_'):
                    convergence_iter = model.best_iteration_
                elif hasattr(model, 'best_iteration'):
                    convergence_iter = model.best_iteration
                elif hasattr(model, 'n_estimators'):
                    convergence_iter = model.n_estimators
                
                # Calculate final score
                final_score = 0.0
                try:
                    if X_val is not None and y_val is not None:
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_val)[:, 1]
                            final_score = roc_auc_score(y_val, y_pred_proba)
                        else:
                            y_pred = model.predict(X_val)
                            final_score = accuracy_score(y_val, y_pred)
                except:
                    pass
                
                # Create benchmark result
                benchmark = TrainingBenchmark(
                    model_name=model_name,
                    training_time=training_time,
                    memory_usage_peak=system_stats.get('memory_rss_gb', {}).get('max', final_memory),
                    memory_usage_average=system_stats.get('memory_rss_gb', {}).get('mean', final_memory),
                    cpu_usage_peak=system_stats.get('cpu_percent', {}).get('max', 0),
                    cpu_usage_average=system_stats.get('cpu_percent', {}).get('mean', 0),
                    dataset_size=len(X_train),
                    feature_count=X_train.shape[1],
                    convergence_iteration=convergence_iter,
                    final_score=final_score
                )
                
                results.append(benchmark)
                logger.info(f"{model_name} training completed in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Training benchmark failed for {model_name}: {e}")
                self.monitor.stop_monitoring()
                continue
        
        self.results.extend(results)
        return results
    
    def benchmark_scalability(self, model_class, data_sizes: List[int], 
                            X_full: pd.DataFrame, y_full: pd.Series) -> Dict[str, Any]:
        """Benchmark model scalability with different data sizes"""
        logger.info("Starting scalability benchmark...")
        
        scalability_results = {
            'data_sizes': data_sizes,
            'training_times': [],
            'memory_usage': [],
            'scores': []
        }
        
        for size in data_sizes:
            logger.info(f"Testing with {size:,} samples")
            
            # Sample data
            if size >= len(X_full):
                X_sample, y_sample = X_full, y_full
            else:
                sample_indices = np.random.choice(len(X_full), size, replace=False)
                X_sample = X_full.iloc[sample_indices]
                y_sample = y_full.iloc[sample_indices]
            
            # Benchmark training
            self.monitor.start_monitoring()
            start_time = time.time()
            
            try:
                model = model_class()
                model.fit(X_sample, y_sample)
                
                training_time = time.time() - start_time
                system_stats = self.monitor.stop_monitoring()
                
                # Test score
                score = 0.0
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_sample)[:, 1]
                        score = roc_auc_score(y_sample, y_pred_proba)
                except:
                    pass
                
                scalability_results['training_times'].append(training_time)
                scalability_results['memory_usage'].append(
                    system_stats.get('memory_rss_gb', {}).get('max', 0)
                )
                scalability_results['scores'].append(score)
                
            except Exception as e:
                logger.error(f"Scalability test failed for size {size}: {e}")
                self.monitor.stop_monitoring()
                scalability_results['training_times'].append(float('inf'))
                scalability_results['memory_usage'].append(0)
                scalability_results['scores'].append(0)
        
        return scalability_results

class ModelInferenceBenchmark:
    """Model inference performance benchmark"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.results = []
        self.monitor = SystemMonitor()
    
    def benchmark_inference_latency(self, models: Dict[str, Any], X_test: pd.DataFrame,
                                  batch_sizes: List[int] = [1, 10, 100, 1000],
                                  num_runs: int = 100) -> List[InferenceBenchmark]:
        """Benchmark inference latency for different batch sizes"""
        logger.info("Starting inference latency benchmark...")
        
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Benchmarking inference for {model_name}")
            
            for batch_size in batch_sizes:
                if batch_size > len(X_test):
                    continue
                
                # Prepare test batch
                test_batch = X_test.head(batch_size)
                latencies = []
                
                # Warmup runs
                for _ in range(5):
                    try:
                        if hasattr(model, 'predict_proba'):
                            _ = model.predict_proba(test_batch)
                        else:
                            _ = model.predict(test_batch)
                    except:
                        break
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                # Actual benchmark runs
                for run in range(num_runs):
                    start_time = time.perf_counter()
                    
                    try:
                        if hasattr(model, 'predict_proba'):
                            _ = model.predict_proba(test_batch)
                        else:
                            _ = model.predict(test_batch)
                        
                        latency = time.perf_counter() - start_time
                        latencies.append(latency)
                        
                    except Exception as e:
                        logger.warning(f"Inference failed for {model_name}: {e}")
                        break
                
                # Stop monitoring
                system_stats = self.monitor.stop_monitoring()
                
                if latencies:
                    # Calculate statistics
                    latencies_ms = [l * 1000 for l in latencies]  # Convert to milliseconds
                    
                    benchmark = InferenceBenchmark(
                        model_name=model_name,
                        batch_size=batch_size,
                        latency_mean=np.mean(latencies_ms),
                        latency_p50=np.percentile(latencies_ms, 50),
                        latency_p95=np.percentile(latencies_ms, 95),
                        latency_p99=np.percentile(latencies_ms, 99),
                        throughput=batch_size / np.mean(latencies),
                        memory_usage=system_stats.get('memory_rss_gb', {}).get('mean', 0),
                        cpu_usage=system_stats.get('cpu_percent', {}).get('mean', 0),
                        predictions_per_second=batch_size / np.mean(latencies)
                    )
                    
                    results.append(benchmark)
                    
                    logger.info(f"{model_name} batch_size={batch_size}: "
                              f"mean_latency={benchmark.latency_mean:.2f}ms, "
                              f"throughput={benchmark.throughput:.0f} pred/s")
        
        self.results.extend(results)
        return results
    
    def benchmark_real_time_performance(self, models: Dict[str, Any], X_test: pd.DataFrame,
                                      duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark real-time prediction performance"""
        logger.info(f"Starting {duration_seconds}s real-time performance benchmark...")
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Real-time benchmark for {model_name}")
            
            # Prepare single sample for real-time prediction
            single_sample = X_test.head(1)
            
            prediction_count = 0
            latencies = []
            start_time = time.time()
            
            # Monitor system during benchmark
            self.monitor.start_monitoring()
            
            while time.time() - start_time < duration_seconds:
                pred_start = time.perf_counter()
                
                try:
                    if hasattr(model, 'predict_proba'):
                        _ = model.predict_proba(single_sample)
                    else:
                        _ = model.predict(single_sample)
                    
                    latency = time.perf_counter() - pred_start
                    latencies.append(latency * 1000)  # Convert to ms
                    prediction_count += 1
                    
                except Exception as e:
                    logger.warning(f"Real-time prediction failed: {e}")
                    break
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
            
            # Stop monitoring
            system_stats = self.monitor.stop_monitoring()
            
            if latencies:
                results[model_name] = {
                    'total_predictions': prediction_count,
                    'predictions_per_second': prediction_count / duration_seconds,
                    'avg_latency_ms': np.mean(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95),
                    'p99_latency_ms': np.percentile(latencies, 99),
                    'max_latency_ms': np.max(latencies),
                    'system_stats': system_stats
                }
                
                logger.info(f"{model_name}: {prediction_count} predictions in {duration_seconds}s "
                          f"({prediction_count/duration_seconds:.0f} pred/s)")
        
        return results

class ModelAccuracyBenchmark:
    """Model accuracy and performance benchmark"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.results = []
    
    def comprehensive_accuracy_benchmark(self, models: Dict[str, Any], 
                                       X_train: pd.DataFrame, y_train: pd.Series,
                                       X_test: pd.DataFrame, y_test: pd.Series,
                                       cv_folds: int = 5) -> List[AccuracyBenchmark]:
        """Comprehensive accuracy benchmark with cross-validation"""
        logger.info("Starting comprehensive accuracy benchmark...")
        
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Accuracy benchmark for {model_name}")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, 
                                          cv=cv_folds, scoring='roc_auc', n_jobs=-1)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Training score
                if hasattr(model, 'predict_proba'):
                    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                    training_score = roc_auc_score(y_train, y_train_pred_proba)
                else:
                    y_train_pred = model.predict(X_train)
                    training_score = accuracy_score(y_train, y_train_pred)
                
                # Test predictions
                if hasattr(model, 'predict_proba'):
                    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_test_pred = (y_test_pred_proba > 0.5).astype(int)
                    validation_score = roc_auc_score(y_test, y_test_pred_proba)
                    roc_auc = validation_score
                else:
                    y_test_pred = model.predict(X_test)
                    y_test_pred_proba = None
                    validation_score = accuracy_score(y_test, y_test_pred)
                    roc_auc = 0.0
                
                # Calculate all metrics
                accuracy = accuracy_score(y_test, y_test_pred)
                precision = precision_score(y_test, y_test_pred, zero_division=0)
                recall = recall_score(y_test, y_test_pred, zero_division=0)
                f1 = f1_score(y_test, y_test_pred, zero_division=0)
                
                benchmark = AccuracyBenchmark(
                    model_name=model_name,
                    roc_auc=roc_auc,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    cross_val_score_mean=cv_scores.mean(),
                    cross_val_score_std=cv_scores.std(),
                    training_score=training_score,
                    validation_score=validation_score
                )
                
                results.append(benchmark)
                
                logger.info(f"{model_name}: ROC-AUC={roc_auc:.4f}, "
                          f"F1={f1:.4f}, CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Accuracy benchmark failed for {model_name}: {e}")
                continue
        
        self.results.extend(results)
        return results
    
    def learning_curve_benchmark(self, model, X: pd.DataFrame, y: pd.Series,
                                train_sizes: Optional[List[float]] = None) -> Dict[str, Any]:
        """Generate learning curves for model performance analysis"""
        logger.info("Generating learning curves...")
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=5, 
                scoring='roc_auc', n_jobs=-1
            )
            
            results = {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'val_scores_mean': val_scores.mean(axis=1).tolist(),
                'val_scores_std': val_scores.std(axis=1).tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Learning curve generation failed: {e}")
            return {}

class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite"""
    
    def __init__(self, config=None, output_dir: str = "benchmark_results"):
        self.config = config or get_config()
        self.output_dir = output_dir
        self.training_benchmark = ModelTrainingBenchmark(config)
        self.inference_benchmark = ModelInferenceBenchmark(config)
        self.accuracy_benchmark = ModelAccuracyBenchmark(config)
        self.inference_optimizer = InferenceOptimizer(config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.all_results = {
            'training_benchmarks': [],
            'inference_benchmarks': [],
            'accuracy_benchmarks': [],
            'system_benchmarks': [],
            'optimization_benchmarks': [],
            'comparative_analysis': {},
            'hardware_info': self._get_hardware_info(),
            'benchmark_metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            }
        }
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get system hardware information"""
        try:
            memory_info = psutil.virtual_memory()
            cpu_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            memory_info_dict = {
                'total_gb': memory_info.total / (1024**3),
                'available_gb': memory_info.available / (1024**3)
            }
            
            gpu_info = {}
            if GPU_AVAILABLE:
                try:
                    gpu_stats = gpustat.new_query()
                    gpu_info = {
                        'gpu_count': len(gpu_stats),
                        'gpus': [{'name': gpu.name, 'memory_total': gpu.memory_total} 
                                for gpu in gpu_stats]
                    }
                except:
                    gpu_info = {'gpu_count': 0}
            
            return {
                'cpu': cpu_info,
                'memory': memory_info_dict,
                'gpu': gpu_info
            }
        except Exception as e:
            logger.warning(f"Failed to get hardware info: {e}")
            return {}
    
    def run_comprehensive_benchmark(self, models: Dict[str, Any], 
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  X_val: Optional[pd.DataFrame] = None,
                                  y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark suite...")
        
        # 1. Training Performance Benchmarks
        logger.info("=== Training Performance Benchmarks ===")
        training_results = self.training_benchmark.benchmark_training_speed(
            models, X_train, y_train, X_val, y_val
        )
        self.all_results['training_benchmarks'] = [asdict(r) for r in training_results]
        
        # 2. Accuracy Benchmarks
        logger.info("=== Accuracy Benchmarks ===")
        accuracy_results = self.accuracy_benchmark.comprehensive_accuracy_benchmark(
            models, X_train, y_train, X_test, y_test
        )
        self.all_results['accuracy_benchmarks'] = [asdict(r) for r in accuracy_results]
        
        # 3. Inference Performance Benchmarks
        logger.info("=== Inference Performance Benchmarks ===")
        inference_results = self.inference_benchmark.benchmark_inference_latency(
            models, X_test
        )
        self.all_results['inference_benchmarks'] = [asdict(r) for r in inference_results]
        
        # 4. Real-time Performance
        logger.info("=== Real-time Performance Benchmarks ===")
        realtime_results = self.inference_benchmark.benchmark_real_time_performance(
            models, X_test, duration_seconds=30
        )
        self.all_results['realtime_benchmarks'] = realtime_results
        
        # 5. System Resource Benchmarks
        logger.info("=== System Resource Benchmarks ===")
        system_results = self._benchmark_system_resources(models, X_test)
        self.all_results['system_benchmarks'] = system_results
        
        # 6. Optimization Benchmarks
        logger.info("=== Optimization Benchmarks ===")
        optimization_results = self._benchmark_optimizations(models, X_test)
        self.all_results['optimization_benchmarks'] = optimization_results
        
        # 7. Comparative Analysis
        logger.info("=== Comparative Analysis ===")
        comparative_results = self._generate_comparative_analysis()
        self.all_results['comparative_analysis'] = comparative_results
        
        # 8. Generate recommendations
        recommendations = self._generate_recommendations()
        self.all_results['recommendations'] = recommendations
        
        # Save results
        self._save_results()
        
        logger.info("Comprehensive benchmark suite completed!")
        return self.all_results
    
    def _benchmark_system_resources(self, models: Dict[str, Any], 
                                   X_test: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark system resource utilization"""
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"System resource benchmark for {model_name}")
            
            # Memory efficiency test
            initial_memory = MemoryProfiler.get_memory_usage()['rss_gb']
            
            # Load test - multiple concurrent predictions
            batch_sizes = [100, 500, 1000, 2000]
            memory_usage = []
            
            for batch_size in batch_sizes:
                if batch_size > len(X_test):
                    continue
                
                test_batch = X_test.head(batch_size)
                
                try:
                    gc.collect()  # Clean up before test
                    memory_before = MemoryProfiler.get_memory_usage()['rss_gb']
                    
                    # Prediction
                    if hasattr(model, 'predict_proba'):
                        _ = model.predict_proba(test_batch)
                    else:
                        _ = model.predict(test_batch)
                    
                    memory_after = MemoryProfiler.get_memory_usage()['rss_gb']
                    memory_increase = memory_after - memory_before
                    
                    memory_usage.append({
                        'batch_size': batch_size,
                        'memory_increase_gb': memory_increase,
                        'memory_per_sample_mb': (memory_increase * 1024) / batch_size
                    })
                    
                except Exception as e:
                    logger.warning(f"Memory test failed for {model_name}: {e}")
            
            results[model_name] = {
                'memory_usage_by_batch': memory_usage,
                'base_memory_gb': initial_memory
            }
        
        return results
    
    def _benchmark_optimizations(self, models: Dict[str, Any], 
                               X_test: pd.DataFrame) -> Dict[str, Any]:
        """Benchmark different optimization techniques"""
        results = {}
        optimization_levels = ['light', 'medium', 'aggressive']
        
        for model_name, model in models.items():
            logger.info(f"Optimization benchmark for {model_name}")
            
            model_results = {}
            
            # Baseline performance
            baseline_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                if hasattr(model, 'predict_proba'):
                    _ = model.predict_proba(X_test.head(100))
                else:
                    _ = model.predict(X_test.head(100))
                baseline_times.append(time.perf_counter() - start_time)
            
            model_results['baseline'] = {
                'mean_time_ms': np.mean(baseline_times) * 1000,
                'std_time_ms': np.std(baseline_times) * 1000
            }
            
            # Test different optimization levels
            for opt_level in optimization_levels:
                try:
                    # Optimize model
                    optimized_model = self.inference_optimizer.optimize_model_for_inference(
                        model, model_name, opt_level
                    )
                    
                    # Benchmark optimized model
                    opt_times = []
                    for _ in range(10):
                        start_time = time.perf_counter()
                        if hasattr(optimized_model, 'predict_proba'):
                            _ = optimized_model.predict_proba(X_test.head(100))
                        else:
                            _ = optimized_model.predict(X_test.head(100))
                        opt_times.append(time.perf_counter() - start_time)
                    
                    model_results[opt_level] = {
                        'mean_time_ms': np.mean(opt_times) * 1000,
                        'std_time_ms': np.std(opt_times) * 1000,
                        'speedup': np.mean(baseline_times) / np.mean(opt_times)
                    }
                    
                except Exception as e:
                    logger.warning(f"Optimization {opt_level} failed for {model_name}: {e}")
                    model_results[opt_level] = {'error': str(e)}
            
            results[model_name] = model_results
        
        return results
    
    def _generate_comparative_analysis(self) -> Dict[str, Any]:
        """Generate comparative analysis across all benchmarks"""
        analysis = {}
        
        # Training Performance Analysis
        if self.all_results['training_benchmarks']:
            training_data = self.all_results['training_benchmarks']
            
            # Find fastest training model
            fastest_model = min(training_data, key=lambda x: x['training_time'])
            slowest_model = max(training_data, key=lambda x: x['training_time'])
            
            analysis['training_performance'] = {
                'fastest_model': {
                    'name': fastest_model['model_name'],
                    'time': fastest_model['training_time']
                },
                'slowest_model': {
                    'name': slowest_model['model_name'],
                    'time': slowest_model['training_time']
                },
                'speed_difference': slowest_model['training_time'] / fastest_model['training_time']
            }
        
        # Accuracy Analysis
        if self.all_results['accuracy_benchmarks']:
            accuracy_data = self.all_results['accuracy_benchmarks']
            
            # Find best performing model
            best_model = max(accuracy_data, key=lambda x: x['roc_auc'])
            worst_model = min(accuracy_data, key=lambda x: x['roc_auc'])
            
            analysis['accuracy_performance'] = {
                'best_model': {
                    'name': best_model['model_name'],
                    'roc_auc': best_model['roc_auc'],
                    'f1_score': best_model['f1_score']
                },
                'worst_model': {
                    'name': worst_model['model_name'],
                    'roc_auc': worst_model['roc_auc'],
                    'f1_score': worst_model['f1_score']
                }
            }
        
        # Inference Performance Analysis
        if self.all_results['inference_benchmarks']:
            inference_data = self.all_results['inference_benchmarks']
            
            # Group by model and find best batch size for each
            model_inference = defaultdict(list)
            for result in inference_data:
                model_inference[result['model_name']].append(result)
            
            best_inference = {}
            for model_name, results in model_inference.items():
                best_result = max(results, key=lambda x: x['throughput'])
                best_inference[model_name] = {
                    'best_batch_size': best_result['batch_size'],
                    'best_throughput': best_result['throughput'],
                    'latency_p95': best_result['latency_p95']
                }
            
            analysis['inference_performance'] = best_inference
        
        # Overall Performance Score
        analysis['overall_ranking'] = self._calculate_overall_ranking()
        
        return analysis
    
    def _calculate_overall_ranking(self) -> List[Dict[str, Any]]:
        """Calculate overall performance ranking considering multiple factors"""
        model_scores = defaultdict(lambda: {'scores': [], 'metrics': {}})
        
        # Training speed score (normalized, faster = better)
        if self.all_results['training_benchmarks']:
            times = [r['training_time'] for r in self.all_results['training_benchmarks']]
            max_time = max(times)
            
            for result in self.all_results['training_benchmarks']:
                # Inverse score - faster training gets higher score
                score = (max_time - result['training_time']) / max_time
                model_scores[result['model_name']]['scores'].append(score)
                model_scores[result['model_name']]['metrics']['training_speed_score'] = score
        
        # Accuracy score
        if self.all_results['accuracy_benchmarks']:
            for result in self.all_results['accuracy_benchmarks']:
                # ROC-AUC is already normalized 0-1
                score = result['roc_auc']
                model_scores[result['model_name']]['scores'].append(score)
                model_scores[result['model_name']]['metrics']['accuracy_score'] = score
        
        # Inference speed score
        if self.all_results['inference_benchmarks']:
            # Get best throughput for each model
            model_throughput = {}
            for result in self.all_results['inference_benchmarks']:
                model_name = result['model_name']
                if model_name not in model_throughput or result['throughput'] > model_throughput[model_name]:
                    model_throughput[model_name] = result['throughput']
            
            if model_throughput:
                max_throughput = max(model_throughput.values())
                for model_name, throughput in model_throughput.items():
                    score = throughput / max_throughput
                    model_scores[model_name]['scores'].append(score)
                    model_scores[model_name]['metrics']['inference_speed_score'] = score
        
        # Calculate overall score and create ranking
        ranking = []
        for model_name, data in model_scores.items():
            if data['scores']:
                overall_score = np.mean(data['scores'])
                ranking.append({
                    'model_name': model_name,
                    'overall_score': overall_score,
                    'individual_scores': data['metrics']
                })
        
        # Sort by overall score (descending)
        ranking.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return ranking
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate performance optimization recommendations"""
        recommendations = {
            'model_selection': {},
            'optimization_strategies': {},
            'infrastructure_recommendations': {},
            'performance_targets': {}
        }
        
        # Model selection recommendations
        if self.all_results['comparative_analysis'].get('overall_ranking'):
            best_model = self.all_results['comparative_analysis']['overall_ranking'][0]
            recommendations['model_selection'] = {
                'recommended_model': best_model['model_name'],
                'reasoning': f"Best overall score: {best_model['overall_score']:.3f}",
                'alternative_models': [
                    r['model_name'] for r in 
                    self.all_results['comparative_analysis']['overall_ranking'][1:3]
                ]
            }
        
        # Optimization strategies
        if self.all_results.get('optimization_benchmarks'):
            opt_recommendations = []
            
            for model_name, opt_results in self.all_results['optimization_benchmarks'].items():
                best_opt = None
                best_speedup = 1.0
                
                for opt_level, result in opt_results.items():
                    if opt_level != 'baseline' and 'speedup' in result:
                        if result['speedup'] > best_speedup:
                            best_speedup = result['speedup']
                            best_opt = opt_level
                
                if best_opt:
                    opt_recommendations.append({
                        'model': model_name,
                        'recommended_optimization': best_opt,
                        'expected_speedup': f"{best_speedup:.2f}x"
                    })
            
            recommendations['optimization_strategies'] = opt_recommendations
        
        # Infrastructure recommendations
        hardware_info = self.all_results['hardware_info']
        infra_recommendations = []
        
        # Memory recommendations
        if self.all_results.get('system_benchmarks'):
            max_memory_usage = 0
            for model_results in self.all_results['system_benchmarks'].values():
                for usage_info in model_results.get('memory_usage_by_batch', []):
                    max_memory_usage = max(max_memory_usage, usage_info.get('memory_increase_gb', 0))
            
            if max_memory_usage > hardware_info.get('memory', {}).get('available_gb', 0) * 0.8:
                infra_recommendations.append(
                    "Consider increasing system memory or using batch processing for large workloads"
                )
        
        # CPU recommendations
        cpu_count = hardware_info.get('cpu', {}).get('cpu_count', 1)
        if cpu_count < 8:
            infra_recommendations.append(
                "Consider upgrading to a multi-core CPU for better parallel processing performance"
            )
        
        # GPU recommendations
        gpu_count = hardware_info.get('gpu', {}).get('gpu_count', 0)
        if gpu_count == 0:
            infra_recommendations.append(
                "Consider adding GPU acceleration for large-scale training and inference"
            )
        
        recommendations['infrastructure_recommendations'] = infra_recommendations
        
        # Performance targets
        if self.all_results.get('realtime_benchmarks'):
            avg_latencies = []
            for model_results in self.all_results['realtime_benchmarks'].values():
                avg_latencies.append(model_results.get('avg_latency_ms', 0))
            
            if avg_latencies:
                median_latency = np.median(avg_latencies)
                recommendations['performance_targets'] = {
                    'target_latency_p95_ms': median_latency * 1.5,
                    'target_throughput_min': 100,  # predictions per second
                    'target_accuracy_min': 0.85,   # minimum ROC-AUC
                    'target_memory_usage_max_gb': 4.0
                }
        
        return recommendations
    
    def _save_results(self):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive JSON results
        json_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        
        # Save summary report
        report_path = os.path.join(self.output_dir, f"benchmark_summary_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(self._generate_text_report())
        
        # Save CSV summaries for easy analysis
        self._save_csv_summaries(timestamp)
        
        logger.info(f"Benchmark results saved to {self.output_dir}")
    
    def _generate_text_report(self) -> str:
        """Generate human-readable text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("FRAUD DETECTION MODEL PERFORMANCE BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Hardware Info
        lines.append("HARDWARE CONFIGURATION")
        lines.append("-" * 40)
        hw_info = self.all_results['hardware_info']
        if hw_info.get('cpu'):
            lines.append(f"CPU Cores: {hw_info['cpu'].get('cpu_count', 'N/A')}")
        if hw_info.get('memory'):
            lines.append(f"Memory: {hw_info['memory'].get('total_gb', 0):.1f} GB")
        if hw_info.get('gpu'):
            lines.append(f"GPUs: {hw_info['gpu'].get('gpu_count', 0)}")
        lines.append("")
        
        # Overall Rankings
        if self.all_results['comparative_analysis'].get('overall_ranking'):
            lines.append("OVERALL MODEL RANKING")
            lines.append("-" * 40)
            for i, model_info in enumerate(self.all_results['comparative_analysis']['overall_ranking'][:5]):
                lines.append(f"{i+1}. {model_info['model_name']}: {model_info['overall_score']:.3f}")
            lines.append("")
        
        # Training Performance
        if self.all_results['training_benchmarks']:
            lines.append("TRAINING PERFORMANCE")
            lines.append("-" * 40)
            for result in self.all_results['training_benchmarks']:
                lines.append(f"{result['model_name']}: {result['training_time']:.2f}s "
                           f"(Memory: {result['memory_usage_peak']:.2f} GB)")
            lines.append("")
        
        # Accuracy Results
        if self.all_results['accuracy_benchmarks']:
            lines.append("ACCURACY RESULTS")
            lines.append("-" * 40)
            for result in self.all_results['accuracy_benchmarks']:
                lines.append(f"{result['model_name']}: ROC-AUC={result['roc_auc']:.4f}, "
                           f"F1={result['f1_score']:.4f}")
            lines.append("")
        
        # Recommendations
        if self.all_results.get('recommendations'):
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 40)
            rec = self.all_results['recommendations']
            
            if rec.get('model_selection', {}).get('recommended_model'):
                lines.append(f"Recommended Model: {rec['model_selection']['recommended_model']}")
            
            if rec.get('infrastructure_recommendations'):
                lines.append("Infrastructure Recommendations:")
                for rec_item in rec['infrastructure_recommendations']:
                    lines.append(f"  - {rec_item}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _save_csv_summaries(self, timestamp: str):
        """Save CSV summaries for easy analysis"""
        # Training results CSV
        if self.all_results['training_benchmarks']:
            training_df = pd.DataFrame(self.all_results['training_benchmarks'])
            training_path = os.path.join(self.output_dir, f"training_benchmarks_{timestamp}.csv")
            training_df.to_csv(training_path, index=False)
        
        # Accuracy results CSV
        if self.all_results['accuracy_benchmarks']:
            accuracy_df = pd.DataFrame(self.all_results['accuracy_benchmarks'])
            accuracy_path = os.path.join(self.output_dir, f"accuracy_benchmarks_{timestamp}.csv")
            accuracy_df.to_csv(accuracy_path, index=False)
        
        # Inference results CSV
        if self.all_results['inference_benchmarks']:
            inference_df = pd.DataFrame(self.all_results['inference_benchmarks'])
            inference_path = os.path.join(self.output_dir, f"inference_benchmarks_{timestamp}.csv")
            inference_df.to_csv(inference_path, index=False)

def run_comprehensive_fraud_detection_benchmark(
    df: pd.DataFrame, 
    target_col: str = 'isFraud',
    test_size: float = 0.2,
    models_to_test: Optional[List[str]] = None,
    output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark for fraud detection models
    
    Args:
        df: Input dataset
        target_col: Target column name
        test_size: Test set size ratio
        models_to_test: List of model names to test (None for all)
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing all benchmark results
    """
    logger.info("Starting comprehensive fraud detection benchmark...")
    
    # Initialize benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite(output_dir=output_dir)
    
    # Prepare data using optimized modeling
    model_trainer = OptimizedFraudDetectionModel()
    data_splits = model_trainer.prepare_data(df, target_col, test_size=test_size)
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits[:6]
    
    # Define models to benchmark
    if models_to_test is None:
        models_to_test = ['lightgbm', 'xgboost', 'catboost', 'random_forest']
    
    models_to_benchmark = {}
    
    for model_name in models_to_test:
        try:
            if model_name == 'lightgbm':
                models_to_benchmark[model_name] = lgb.LGBMClassifier(
                    n_estimators=100, random_state=42, verbosity=-1
                )
            elif model_name == 'xgboost':
                models_to_benchmark[model_name] = xgb.XGBClassifier(
                    n_estimators=100, random_state=42, eval_metric='logloss'
                )
            elif model_name == 'catboost':
                models_to_benchmark[model_name] = cb.CatBoostClassifier(
                    iterations=100, random_seed=42, verbose=False
                )
            elif model_name == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                models_to_benchmark[model_name] = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1
                )
        except Exception as e:
            logger.warning(f"Failed to initialize {model_name}: {e}")
    
    # Run comprehensive benchmark
    results = benchmark_suite.run_comprehensive_benchmark(
        models_to_benchmark, X_train, y_train, X_test, y_test, X_val, y_val
    )
    
    return results

if __name__ == "__main__":
    print("Performance Benchmarking System loaded successfully!")
    logger.info("Performance Benchmarking System ready for use")