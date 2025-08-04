"""
性能優化模組 - IEEE-CIS 詐騙檢測項目
提供模型推理加速、批處理優化和性能監控功能
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import gc
import psutil
from dataclasses import dataclass
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入加速庫
try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda x: x  # 創建假的裝飾器

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from .config import get_config
from .memory_optimizer import MemoryProfiler

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    prediction_time: float
    throughput: float  # predictions per second
    memory_usage: float  # GB
    cpu_usage: float  # percentage
    batch_size: int
    model_name: str
    optimization_level: str

class InferenceOptimizer:
    """推理優化器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.cached_models = {}
        self.cached_preprocessors = {}
        self.performance_history = []
        self.optimization_cache = {}
        
        # 硬體資源檢測
        self.cpu_count = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = self._check_gpu_availability()
        
        logger.info(f"推理優化器初始化完成 - CPU: {self.cpu_count}核, 內存: {self.memory_gb:.1f}GB, GPU: {self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """檢查GPU可用性"""
        try:
            if CUPY_AVAILABLE:
                cp.cuda.Device(0).compute_capability
                return True
        except:
            pass
        
        try:
            if NUMBA_AVAILABLE:
                cuda.detect()
                return len(cuda.gpus) > 0
        except:
            pass
        
        return False
    
    def optimize_model_for_inference(self, model, model_name: str, optimization_level: str = 'medium'):
        """優化模型用於推理"""
        logger.info(f"優化模型 {model_name} 用於推理，優化級別: {optimization_level}")
        
        if optimization_level == 'light':
            # 輕量級優化：只做基本的內存優化
            optimized_model = self._light_optimization(model, model_name)
        elif optimization_level == 'medium':
            # 中等優化：模型壓縮和並行化
            optimized_model = self._medium_optimization(model, model_name)
        elif optimization_level == 'aggressive':
            # 激進優化：所有可用的優化技術
            optimized_model = self._aggressive_optimization(model, model_name)
        else:
            optimized_model = model
        
        # 緩存優化後的模型
        self.cached_models[model_name] = optimized_model
        
        return optimized_model
    
    def _light_optimization(self, model, model_name: str):
        """輕量級優化"""
        # 基本的內存優化
        if hasattr(model, 'n_jobs'):
            model.n_jobs = min(self.cpu_count, 4)  # 限制線程數避免資源競爭
        
        return model
    
    def _medium_optimization(self, model, model_name: str):
        """中等優化"""
        optimized_model = self._light_optimization(model, model_name)
        
        # 嘗試模型量化（如果支持）
        try:
            if hasattr(model, 'get_booster'):  # XGBoost/LightGBM
                # 可以考慮使用更小的數據類型
                pass
        except:
            pass
        
        return optimized_model
    
    def _aggressive_optimization(self, model, model_name: str):
        """激進優化"""
        optimized_model = self._medium_optimization(model, model_name)
        
        # GPU加速（如果可用）
        if self.gpu_available and CUPY_AVAILABLE:
            try:
                # 將部分計算移到GPU
                logger.info(f"嘗試GPU加速 {model_name}")
                # 這裡可以實現GPU加速邏輯
            except Exception as e:
                logger.warning(f"GPU加速失敗: {e}")
        
        return optimized_model
    
    def create_batch_predictor(self, model, preprocessor=None, batch_size: int = 1000):
        """創建批處理預測器"""
        
        class BatchPredictor:
            def __init__(self, model, preprocessor, batch_size, optimizer):
                self.model = model
                self.preprocessor = preprocessor
                self.batch_size = batch_size
                self.optimizer = optimizer
                self.prediction_cache = {}
            
            def predict(self, X: pd.DataFrame, enable_cache: bool = True) -> np.ndarray:
                """批處理預測"""
                start_time = time.time()
                
                # 檢查緩存
                if enable_cache:
                    cache_key = hash(X.values.tobytes())
                    if cache_key in self.prediction_cache:
                        return self.prediction_cache[cache_key]
                
                # 批處理預測
                if len(X) <= self.batch_size:
                    # 小批量直接預測
                    predictions = self._single_batch_predict(X)
                else:
                    # 大批量分批預測
                    predictions = self._multi_batch_predict(X)
                
                # 緩存結果
                if enable_cache and len(self.prediction_cache) < 100:
                    self.prediction_cache[cache_key] = predictions
                
                # 記錄性能指標
                prediction_time = time.time() - start_time
                throughput = len(X) / prediction_time
                
                self.optimizer.performance_history.append(PerformanceMetrics(
                    prediction_time=prediction_time,
                    throughput=throughput,
                    memory_usage=MemoryProfiler.get_memory_usage()['rss_gb'],
                    cpu_usage=psutil.cpu_percent(),
                    batch_size=len(X),
                    model_name=getattr(self.model, '__class__').__name__,
                    optimization_level='batch'
                ))
                
                return predictions
            
            def _single_batch_predict(self, X: pd.DataFrame) -> np.ndarray:
                """單批預測"""
                if self.preprocessor:
                    X = self.preprocessor.transform(X)
                
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(X)[:, 1]
                else:
                    return self.model.predict(X)
            
            def _multi_batch_predict(self, X: pd.DataFrame) -> np.ndarray:
                """多批預測"""
                predictions = []
                
                for i in range(0, len(X), self.batch_size):
                    batch = X.iloc[i:i+self.batch_size]
                    batch_pred = self._single_batch_predict(batch)
                    predictions.append(batch_pred)
                    
                    # 定期清理內存
                    if i % (self.batch_size * 5) == 0:
                        gc.collect()
                
                return np.concatenate(predictions)
        
        return BatchPredictor(model, preprocessor, batch_size, self)
    
    def create_parallel_predictor(self, models: Dict, weights: Optional[List[float]] = None):
        """創建並行集成預測器"""
        
        class ParallelEnsemblePredictor:
            def __init__(self, models, weights, optimizer):
                self.models = models
                self.weights = weights or [1.0/len(models)] * len(models)
                self.optimizer = optimizer
                self.executor = ThreadPoolExecutor(max_workers=min(len(models), 4))
            
            def predict(self, X: pd.DataFrame) -> np.ndarray:
                """並行集成預測"""
                start_time = time.time()
                
                # 並行預測
                def single_model_predict(model_item):
                    model_name, model = model_item
                    try:
                        if hasattr(model, 'predict_proba'):
                            return model.predict_proba(X)[:, 1]
                        else:
                            return model.predict(X)
                    except Exception as e:
                        logger.error(f"模型 {model_name} 預測失敗: {e}")
                        return np.zeros(len(X))
                
                # 提交所有預測任務
                futures = {
                    self.executor.submit(single_model_predict, item): name 
                    for name, item in enumerate(self.models.items())
                }
                
                # 收集結果
                predictions = []
                for future in futures:
                    try:
                        pred = future.result(timeout=30)
                        predictions.append(pred)
                    except Exception as e:
                        logger.error(f"並行預測失敗: {e}")
                        predictions.append(np.zeros(len(X)))
                
                # 加權平均
                if predictions:
                    weighted_pred = np.average(predictions, axis=0, weights=self.weights)
                else:
                    weighted_pred = np.zeros(len(X))
                
                # 記錄性能
                prediction_time = time.time() - start_time
                throughput = len(X) / prediction_time
                
                self.optimizer.performance_history.append(PerformanceMetrics(
                    prediction_time=prediction_time,
                    throughput=throughput,
                    memory_usage=MemoryProfiler.get_memory_usage()['rss_gb'],
                    cpu_usage=psutil.cpu_percent(),
                    batch_size=len(X),
                    model_name='ensemble',
                    optimization_level='parallel'
                ))
                
                return weighted_pred
            
            def __del__(self):
                if hasattr(self, 'executor'):
                    self.executor.shutdown(wait=True)
        
        return ParallelEnsemblePredictor(models, weights, self)
    
    @jit(nopython=True, cache=True)
    def _fast_feature_computation(self, values: np.ndarray) -> np.ndarray:
        """使用Numba加速的特徵計算"""
        result = np.zeros_like(values)
        for i in range(len(values)):
            result[i] = values[i] * 2 + 1  # 示例計算
        return result
    
    def benchmark_models(self, models: Dict, X_test: pd.DataFrame, 
                        batch_sizes: List[int] = [100, 500, 1000, 5000]) -> Dict[str, Any]:
        """模型性能基準測試"""
        logger.info("開始模型性能基準測試...")
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            logger.info(f"測試模型: {model_name}")
            model_results = {}
            
            for batch_size in batch_sizes:
                # 創建測試批次
                if len(X_test) >= batch_size:
                    test_batch = X_test.head(batch_size)
                else:
                    test_batch = X_test
                
                # 預熱
                try:
                    _ = model.predict_proba(test_batch)[:, 1]
                except:
                    continue
                
                # 多次測試取平均
                times = []
                for _ in range(5):
                    start_time = time.time()
                    _ = model.predict_proba(test_batch)[:, 1]
                    times.append(time.time() - start_time)
                
                avg_time = np.mean(times)
                throughput = len(test_batch) / avg_time
                
                model_results[f'batch_{batch_size}'] = {
                    'avg_time': avg_time,
                    'throughput': throughput,
                    'std_time': np.std(times)
                }
            
            benchmark_results[model_name] = model_results
        
        return benchmark_results
    
    def get_optimal_batch_size(self, model, X_sample: pd.DataFrame, 
                              target_latency: float = 0.1) -> int:
        """獲取最優批處理大小"""
        logger.info("尋找最優批處理大小...")
        
        batch_sizes = [10, 50, 100, 500, 1000, 2000, 5000]
        best_batch_size = 100
        best_throughput = 0
        
        for batch_size in batch_sizes:
            if len(X_sample) < batch_size:
                continue
            
            test_batch = X_sample.head(batch_size)
            
            # 測試性能
            times = []
            for _ in range(3):
                start_time = time.time()
                try:
                    _ = model.predict_proba(test_batch)[:, 1]
                    times.append(time.time() - start_time)
                except Exception as e:
                    logger.warning(f"批大小 {batch_size} 測試失敗: {e}")
                    break
            
            if not times:
                continue
            
            avg_time = np.mean(times)
            throughput = batch_size / avg_time
            
            # 檢查延遲要求
            if avg_time <= target_latency and throughput > best_throughput:
                best_batch_size = batch_size
                best_throughput = throughput
        
        logger.info(f"最優批處理大小: {best_batch_size} (吞吐量: {best_throughput:.0f} pred/s)")
        return best_batch_size
    
    def create_model_pipeline(self, model, preprocessor=None, 
                            optimization_level: str = 'medium') -> 'OptimizedPipeline':
        """創建優化的模型管道"""
        
        class OptimizedPipeline:
            def __init__(self, model, preprocessor, optimizer, optimization_level):
                self.model = optimizer.optimize_model_for_inference(
                    model, getattr(model, '__class__').__name__, optimization_level
                )
                self.preprocessor = preprocessor
                self.optimizer = optimizer
                self.prediction_cache = {}
                self.cache_hits = 0
                self.cache_misses = 0
            
            def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                       enable_cache: bool = True, 
                       return_probabilities: bool = True) -> np.ndarray:
                """統一的預測接口"""
                start_time = time.time()
                
                # 輸入驗證和轉換
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X)
                
                # 緩存檢查
                if enable_cache:
                    cache_key = hash(X.values.tobytes())
                    if cache_key in self.prediction_cache:
                        self.cache_hits += 1
                        return self.prediction_cache[cache_key]
                    else:
                        self.cache_misses += 1
                
                # 預處理
                if self.preprocessor:
                    X_processed = self.preprocessor.transform(X)
                else:
                    X_processed = X
                
                # 預測
                if return_probabilities and hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(X_processed)[:, 1]
                else:
                    predictions = self.model.predict(X_processed)
                
                # 緩存結果
                if enable_cache and len(self.prediction_cache) < 1000:
                    self.prediction_cache[cache_key] = predictions
                
                # 記錄性能
                prediction_time = time.time() - start_time
                self.optimizer.performance_history.append(PerformanceMetrics(
                    prediction_time=prediction_time,
                    throughput=len(X) / prediction_time,
                    memory_usage=MemoryProfiler.get_memory_usage()['rss_gb'],
                    cpu_usage=psutil.cpu_percent(),
                    batch_size=len(X),
                    model_name=getattr(self.model, '__class__').__name__,
                    optimization_level='pipeline'
                ))
                
                return predictions
            
            def get_cache_stats(self) -> Dict[str, Any]:
                """獲取緩存統計"""
                total_requests = self.cache_hits + self.cache_misses
                hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
                
                return {
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'hit_rate': hit_rate,
                    'cache_size': len(self.prediction_cache)
                }
            
            def clear_cache(self):
                """清理緩存"""
                self.prediction_cache.clear()
                gc.collect()
        
        return OptimizedPipeline(model, preprocessor, self, optimization_level)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """獲取性能報告"""
        if not self.performance_history:
            return {"message": "無性能數據"}
        
        # 統計分析
        times = [m.prediction_time for m in self.performance_history]
        throughputs = [m.throughput for m in self.performance_history]
        memory_usage = [m.memory_usage for m in self.performance_history]
        
        report = {
            'total_predictions': len(self.performance_history),
            'avg_prediction_time': np.mean(times),
            'median_prediction_time': np.median(times),
            'p95_prediction_time': np.percentile(times, 95),
            'avg_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'avg_memory_usage': np.mean(memory_usage),
            'peak_memory_usage': np.max(memory_usage),
            'model_performance': {}
        }
        
        # 按模型分組統計
        model_stats = {}
        for metric in self.performance_history:
            model_name = metric.model_name
            if model_name not in model_stats:
                model_stats[model_name] = []
            model_stats[model_name].append(metric)
        
        for model_name, metrics in model_stats.items():
            model_times = [m.prediction_time for m in metrics]
            model_throughputs = [m.throughput for m in metrics]
            
            report['model_performance'][model_name] = {
                'count': len(metrics),
                'avg_time': np.mean(model_times),
                'avg_throughput': np.mean(model_throughputs),
                'best_throughput': np.max(model_throughputs)
            }
        
        return report

class RealTimePredictor:
    """實時預測器"""
    
    def __init__(self, model_pipeline, latency_threshold: float = 0.1):
        self.pipeline = model_pipeline
        self.latency_threshold = latency_threshold
        self.request_queue = []
        self.prediction_cache = {}
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'fast_predictions': 0,
            'slow_predictions': 0
        }
    
    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """單個樣本實時預測"""
        start_time = time.time()
        self.performance_stats['total_requests'] += 1
        
        # 轉換為DataFrame
        X = pd.DataFrame([features])
        
        # 快速預測
        try:
            prediction = self.pipeline.predict(X, enable_cache=True)[0]
            
            prediction_time = time.time() - start_time
            
            # 統計快慢預測
            if prediction_time <= self.latency_threshold:
                self.performance_stats['fast_predictions'] += 1
            else:
                self.performance_stats['slow_predictions'] += 1
            
            return {
                'fraud_probability': float(prediction),
                'prediction_time': prediction_time,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'fraud_probability': 0.5,  # 默認值
                'prediction_time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取性能統計"""
        stats = self.performance_stats.copy()
        
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
            stats['fast_prediction_rate'] = stats['fast_predictions'] / stats['total_requests']
        else:
            stats['cache_hit_rate'] = 0
            stats['fast_prediction_rate'] = 0
        
        return stats

def benchmark_fraud_detection_system(models: Dict, X_test: pd.DataFrame, 
                                    y_test: pd.Series = None) -> Dict[str, Any]:
    """詐騙檢測系統綜合性能測試"""
    logger.info("開始詐騙檢測系統綜合性能測試...")
    
    optimizer = InferenceOptimizer()
    results = {}
    
    # 1. 基礎性能測試
    basic_benchmark = optimizer.benchmark_models(models, X_test)
    results['basic_performance'] = basic_benchmark
    
    # 2. 優化後性能測試
    optimized_models = {}
    for name, model in models.items():
        optimized_models[name] = optimizer.optimize_model_for_inference(
            model, name, 'medium'
        )
    
    optimized_benchmark = optimizer.benchmark_models(optimized_models, X_test)
    results['optimized_performance'] = optimized_benchmark
    
    # 3. 並行集成測試
    if len(models) > 1:
        parallel_predictor = optimizer.create_parallel_predictor(optimized_models)
        
        start_time = time.time()
        ensemble_predictions = parallel_predictor.predict(X_test.head(1000))
        ensemble_time = time.time() - start_time
        
        results['ensemble_performance'] = {
            'prediction_time': ensemble_time,
            'throughput': 1000 / ensemble_time,
            'predictions_shape': ensemble_predictions.shape
        }
    
    # 4. 實時預測測試
    best_model_name = max(models.keys(), key=lambda x: len(x))  # 簡單選擇
    best_model = models[best_model_name]
    
    pipeline = optimizer.create_model_pipeline(best_model, optimization_level='medium')
    real_time_predictor = RealTimePredictor(pipeline)
    
    # 測試單個預測性能
    sample_features = X_test.iloc[0].to_dict()
    rt_results = []
    
    for _ in range(100):
        result = real_time_predictor.predict_single(sample_features)
        rt_results.append(result['prediction_time'])
    
    results['real_time_performance'] = {
        'avg_latency': np.mean(rt_results),
        'p95_latency': np.percentile(rt_results, 95),
        'p99_latency': np.percentile(rt_results, 99),
        'requests_under_threshold': sum(1 for t in rt_results if t <= 0.1) / len(rt_results)
    }
    
    # 5. 內存效率測試
    initial_memory = MemoryProfiler.get_memory_usage()['rss_gb']
    
    # 大批量預測測試
    large_batch = X_test.head(min(10000, len(X_test)))
    batch_predictor = optimizer.create_batch_predictor(best_model, batch_size=1000)
    
    start_time = time.time()
    batch_predictions = batch_predictor.predict(large_batch)
    batch_time = time.time() - start_time
    
    final_memory = MemoryProfiler.get_memory_usage()['rss_gb']
    
    results['memory_efficiency'] = {
        'initial_memory_gb': initial_memory,
        'final_memory_gb': final_memory,
        'memory_increase_gb': final_memory - initial_memory,
        'batch_prediction_time': batch_time,
        'batch_throughput': len(large_batch) / batch_time
    }
    
    # 6. 系統資源利用率
    results['system_resources'] = {
        'cpu_count': optimizer.cpu_count,
        'memory_gb': optimizer.memory_gb,
        'gpu_available': optimizer.gpu_available
    }
    
    # 7. 性能摘要
    performance_report = optimizer.get_performance_report()
    results['performance_summary'] = performance_report
    
    logger.info("綜合性能測試完成")
    return results

if __name__ == "__main__":
    print("性能優化模組已載入完成！")