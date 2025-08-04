"""
性能和內存回歸測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import gc
import sys
from typing import Dict, List, Callable, Any
import threading
import warnings
from unittest.mock import patch

sys.path.append('../src')

from src.data_processing import DataProcessor
from src.data_validation import DataValidator
from src.feature_engineering import FeatureEngineer
from src.modeling import FraudDetectionModel
from src.memory_optimizer import MemoryProfiler, ChunkProcessor
from src.model_monitoring import ModelMonitor
from src.config import ConfigManager


class PerformanceBenchmark:
    """性能基準測試工具"""
    
    def __init__(self):
        self.benchmarks = {}
        self.memory_snapshots = {}
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """測量函數執行時間和資源使用"""
        # 強制垃圾回收
        gc.collect()
        
        # 記錄初始狀態
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        start_cpu = time.process_time()
        
        # 執行函數
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # 記錄結束狀態
        end_time = time.time()
        end_cpu = time.process_time()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'success': success,
            'error': error,
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': final_memory - initial_memory,
            'peak_memory_mb': max(initial_memory, final_memory)
        }
    
    def store_benchmark(self, name: str, metrics: Dict[str, Any]):
        """存儲基準測試結果"""
        self.benchmarks[name] = metrics
    
    def compare_with_baseline(self, name: str, current_metrics: Dict[str, Any], 
                            tolerance: float = 0.1) -> Dict[str, Any]:
        """與基線性能比較"""
        if name not in self.benchmarks:
            self.store_benchmark(name, current_metrics)
            return {'status': 'baseline_set', 'metrics': current_metrics}
        
        baseline = self.benchmarks[name]
        comparison = {}
        
        # 比較執行時間
        time_change = (current_metrics['wall_time'] - baseline['wall_time']) / baseline['wall_time']
        comparison['time_regression'] = time_change > tolerance
        comparison['time_change_percent'] = time_change * 100
        
        # 比較內存使用
        memory_change = (current_metrics['peak_memory_mb'] - baseline['peak_memory_mb']) / baseline['peak_memory_mb']
        comparison['memory_regression'] = memory_change > tolerance
        comparison['memory_change_percent'] = memory_change * 100
        
        # 整體評估
        comparison['overall_regression'] = comparison['time_regression'] or comparison['memory_regression']
        comparison['baseline'] = baseline
        comparison['current'] = current_metrics
        
        return comparison


class TestDataProcessingPerformance:
    """測試數據處理性能"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
        
        # 創建不同大小的測試數據集
        self.datasets = {}
        for size in [1000, 5000, 10000]:
            self.datasets[f'data_{size}'] = self._generate_test_data(size)
    
    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """生成測試數據"""
        np.random.seed(42)
        return pd.DataFrame({
            'TransactionID': range(1, size + 1),
            'isFraud': np.random.choice([0, 1], size, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, size),
            'TransactionAmt': np.random.lognormal(3, 1, size),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], size),
            **{f'C{i}': np.random.randn(size) for i in range(1, 11)},
            **{f'categorical_{i}': np.random.choice(['A', 'B', 'C'], size) for i in range(3)}
        })
    
    def test_data_loading_performance(self):
        """測試數據載入性能"""
        for data_name, data in self.datasets.items():
            # 模擬數據載入（實際上是數據複製）
            metrics = self.benchmark.measure_execution_time(
                lambda df: df.copy(), data
            )
            
            assert metrics['success']
            assert metrics['wall_time'] < 10.0  # 10秒內完成
            assert metrics['memory_delta_mb'] >= 0  # 內存不應該減少
            
            self.benchmark.store_benchmark(f'data_loading_{data_name}', metrics)
            print(f"Data loading {data_name}: {metrics['wall_time']:.3f}s, "
                  f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_basic_preprocessing_performance(self):
        """測試基本預處理性能"""
        processor = DataProcessor()
        
        for data_name, data in self.datasets.items():
            metrics = self.benchmark.measure_execution_time(
                processor.basic_preprocessing, data
            )
            
            assert metrics['success']
            
            # 性能基準
            data_size = len(data)
            expected_max_time = data_size / 1000 * 2  # 每1000行最多2秒
            assert metrics['wall_time'] < expected_max_time
            
            # 內存使用合理性
            assert metrics['memory_delta_mb'] < data_size / 100  # 每100行最多1MB額外內存
            
            self.benchmark.store_benchmark(f'preprocessing_{data_name}', metrics)
            print(f"Preprocessing {data_name}: {metrics['wall_time']:.3f}s, "
                  f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_missing_value_analysis_performance(self):
        """測試缺失值分析性能"""
        processor = DataProcessor()
        
        # 添加缺失值
        for data_name, data in self.datasets.items():
            data_with_missing = data.copy()
            missing_mask = np.random.random(data.shape) < 0.1
            data_with_missing = data_with_missing.mask(missing_mask)
            
            metrics = self.benchmark.measure_execution_time(
                processor.analyze_missing_values, data_with_missing
            )
            
            assert metrics['success']
            assert metrics['wall_time'] < 5.0  # 5秒內完成
            
            result = metrics['result']
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0  # 應該有缺失值報告
            
            print(f"Missing analysis {data_name}: {metrics['wall_time']:.3f}s")
    
    def test_outlier_handling_performance(self):
        """測試異常值處理性能"""
        processor = DataProcessor()
        
        for data_name, data in self.datasets.items():
            # 選擇數值特徵
            numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_features = [col for col in numeric_features if col not in ['TransactionID', 'isFraud']]
            
            metrics = self.benchmark.measure_execution_time(
                processor.handle_outliers, data, numeric_features
            )
            
            assert metrics['success']
            
            # 性能要求
            feature_count = len(numeric_features)
            expected_max_time = feature_count * len(data) / 10000  # 每10000個數據點每特徵最多1秒
            assert metrics['wall_time'] < max(expected_max_time, 1.0)
            
            print(f"Outlier handling {data_name}: {metrics['wall_time']:.3f}s, "
                  f"Features: {feature_count}")


class TestFeatureEngineeringPerformance:
    """測試特徵工程性能"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
        self.config = ConfigManager()
        
        # 創建測試數據
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'TransactionID': range(1, 10001),
            'isFraud': np.random.choice([0, 1], 10000, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, 10000),
            'TransactionAmt': np.random.lognormal(3, 1, 10000),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], 10000),
            'card1': np.random.randint(1000, 20000, 10000),
            'DeviceType': np.random.choice(['desktop', 'mobile', 'tablet'], 10000),
            **{f'C{i}': np.random.randn(10000) for i in range(1, 6)}
        })
    
    def test_time_feature_creation_performance(self):
        """測試時間特徵創建性能"""
        engineer = FeatureEngineer(self.config)
        
        metrics = self.benchmark.measure_execution_time(
            engineer.create_time_features, self.test_data
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 5.0  # 5秒內完成
        
        result = metrics['result']
        assert len(result) == len(self.test_data)
        
        # 檢查新特徵
        expected_features = ['hour', 'day', 'week', 'is_weekend']
        for feature in expected_features:
            assert feature in result.columns
        
        print(f"Time features: {metrics['wall_time']:.3f}s, "
              f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_amount_feature_creation_performance(self):
        """測試金額特徵創建性能"""
        engineer = FeatureEngineer(self.config)
        
        metrics = self.benchmark.measure_execution_time(
            engineer.create_transaction_amount_features, self.test_data
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 3.0  # 3秒內完成
        
        result = metrics['result']
        assert len(result) == len(self.test_data)
        
        # 檢查新特徵
        expected_features = ['TransactionAmt_log', 'TransactionAmt_sqrt', 'amt_range']
        for feature in expected_features:
            assert feature in result.columns
        
        print(f"Amount features: {metrics['wall_time']:.3f}s")
    
    def test_aggregation_features_performance(self):
        """測試聚合特徵性能"""
        engineer = FeatureEngineer(self.config)
        
        group_cols = ['card1', 'ProductCD']
        agg_cols = ['TransactionAmt', 'C1']
        agg_funcs = ['mean', 'count', 'std']
        
        # 測試順序處理
        metrics_sequential = self.benchmark.measure_execution_time(
            engineer.create_aggregation_features, 
            self.test_data, group_cols, agg_cols, agg_funcs
        )
        
        assert metrics_sequential['success']
        
        # 測試並行處理
        engineer_parallel = FeatureEngineer(self.config, enable_parallel=True)
        metrics_parallel = self.benchmark.measure_execution_time(
            engineer_parallel.create_aggregation_features,
            self.test_data, group_cols, agg_cols, agg_funcs
        )
        
        assert metrics_parallel['success']
        
        # 並行處理應該更快或相當（小數據集可能差異不大）
        speedup = metrics_sequential['wall_time'] / metrics_parallel['wall_time']
        print(f"Aggregation - Sequential: {metrics_sequential['wall_time']:.3f}s, "
              f"Parallel: {metrics_parallel['wall_time']:.3f}s, "
              f"Speedup: {speedup:.2f}x")
        
        # 對於較大的數據集，並行處理不應該更慢
        assert metrics_parallel['wall_time'] <= metrics_sequential['wall_time'] * 1.2
    
    def test_full_pipeline_performance(self):
        """測試完整特徵工程流水線性能"""
        engineer = FeatureEngineer(self.config)
        
        # 測試基本特徵工程
        metrics_basic = self.benchmark.measure_execution_time(
            engineer.full_feature_engineering_pipeline,
            self.test_data, target_col='isFraud', enable_advanced_features=False
        )
        
        assert metrics_basic['success']
        assert metrics_basic['wall_time'] < 30.0  # 30秒內完成
        
        # 測試高級特徵工程
        metrics_advanced = self.benchmark.measure_execution_time(
            engineer.full_feature_engineering_pipeline,
            self.test_data, target_col='isFraud', enable_advanced_features=True
        )
        
        assert metrics_advanced['success']
        assert metrics_advanced['wall_time'] < 60.0  # 60秒內完成
        
        # 高級特徵工程應該產生更多特徵
        basic_result = metrics_basic['result']
        advanced_result = metrics_advanced['result']
        assert len(advanced_result.columns) >= len(basic_result.columns)
        
        print(f"Full pipeline - Basic: {metrics_basic['wall_time']:.3f}s, "
              f"Advanced: {metrics_advanced['wall_time']:.3f}s")
    
    def test_categorical_encoding_performance(self):
        """測試類別編碼性能"""
        engineer = FeatureEngineer(self.config)
        
        # 創建高基數類別特徵
        high_cardinality_data = self.test_data.copy()
        high_cardinality_data['high_card_feature'] = [f'category_{i}' for i in range(len(self.test_data))]
        
        categorical_features = ['ProductCD', 'DeviceType', 'high_card_feature']
        
        # 測試標籤編碼
        metrics_label = self.benchmark.measure_execution_time(
            engineer.encode_categorical_features,
            high_cardinality_data, categorical_features, method='label'
        )
        
        assert metrics_label['success']
        assert metrics_label['wall_time'] < 10.0
        
        # 測試頻率編碼
        metrics_freq = self.benchmark.measure_execution_time(
            engineer.encode_categorical_features,
            high_cardinality_data, categorical_features, method='frequency'
        )
        
        assert metrics_freq['success']
        assert metrics_freq['wall_time'] < 15.0
        
        print(f"Categorical encoding - Label: {metrics_label['wall_time']:.3f}s, "
              f"Frequency: {metrics_freq['wall_time']:.3f}s")


class TestModelingPerformance:
    """測試建模性能"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
        
        # 創建訓練數據
        np.random.seed(42)
        size = 5000
        self.train_data = pd.DataFrame({
            'TransactionID': range(1, size + 1),
            'isFraud': np.random.choice([0, 1], size, p=[0.965, 0.035]),
            **{f'feature_{i}': np.random.randn(size) for i in range(20)}
        })
        
        # 準備訓練測試集
        self.model_trainer = FraudDetectionModel()
        self.X_train, self.X_test, self.y_train, self.y_test = self.model_trainer.prepare_data(
            self.train_data, target_col='isFraud', test_size=0.2
        )
    
    def test_random_forest_training_performance(self):
        """測試隨機森林訓練性能"""
        metrics = self.benchmark.measure_execution_time(
            self.model_trainer.train_random_forest, self.X_train, self.y_train
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 30.0  # 30秒內完成
        assert 'random_forest' in self.model_trainer.models
        
        print(f"Random Forest training: {metrics['wall_time']:.3f}s, "
              f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_xgboost_training_performance(self):
        """測試XGBoost訓練性能"""
        metrics = self.benchmark.measure_execution_time(
            self.model_trainer.train_xgboost, self.X_train, self.y_train
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 45.0  # 45秒內完成
        assert 'xgboost' in self.model_trainer.models
        
        print(f"XGBoost training: {metrics['wall_time']:.3f}s, "
              f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_lightgbm_training_performance(self):
        """測試LightGBM訓練性能"""
        metrics = self.benchmark.measure_execution_time(
            self.model_trainer.train_lightgbm, self.X_train, self.y_train
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 20.0  # 20秒內完成（LightGBM通常更快）
        assert 'lightgbm' in self.model_trainer.models
        
        print(f"LightGBM training: {metrics['wall_time']:.3f}s, "
              f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_model_evaluation_performance(self):
        """測試模型評估性能"""
        # 先訓練一個模型
        self.model_trainer.train_random_forest(self.X_train, self.y_train)
        
        metrics = self.benchmark.measure_execution_time(
            self.model_trainer.evaluate_model, 'random_forest', self.X_test, self.y_test
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 5.0  # 5秒內完成
        
        evaluation = metrics['result']
        assert 0 <= evaluation['roc_auc'] <= 1
        assert 0 <= evaluation['accuracy'] <= 1
        
        print(f"Model evaluation: {metrics['wall_time']:.3f}s")
    
    def test_prediction_performance(self):
        """測試預測性能"""
        # 訓練模型
        self.model_trainer.train_random_forest(self.X_train, self.y_train)
        model = self.model_trainer.models['random_forest']
        
        # 測試批量預測
        metrics_batch = self.benchmark.measure_execution_time(
            model.predict, self.X_test
        )
        
        assert metrics_batch['success']
        assert metrics_batch['wall_time'] < 1.0  # 1秒內完成
        
        # 測試單樣本預測性能
        single_sample = self.X_test.iloc[:1]
        metrics_single = self.benchmark.measure_execution_time(
            model.predict, single_sample
        )
        
        assert metrics_single['success']
        assert metrics_single['wall_time'] < 0.1  # 0.1秒內完成
        
        print(f"Batch prediction: {metrics_batch['wall_time']:.4f}s, "
              f"Single prediction: {metrics_single['wall_time']:.4f}s")
    
    def test_model_serialization_performance(self):
        """測試模型序列化性能"""
        import tempfile
        import os
        
        # 訓練模型
        self.model_trainer.train_random_forest(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            # 測試保存性能
            metrics_save = self.benchmark.measure_execution_time(
                self.model_trainer.save_model, 'random_forest', temp_path
            )
            
            assert metrics_save['success']
            assert metrics_save['wall_time'] < 5.0
            assert os.path.exists(temp_path)
            
            # 測試載入性能
            new_trainer = FraudDetectionModel()
            metrics_load = self.benchmark.measure_execution_time(
                new_trainer.load_model, temp_path, 'loaded_model'
            )
            
            assert metrics_load['success']
            assert metrics_load['wall_time'] < 5.0
            assert 'loaded_model' in new_trainer.models
            
            print(f"Model save: {metrics_save['wall_time']:.3f}s, "
                  f"Model load: {metrics_load['wall_time']:.3f}s")
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestMemoryOptimizationPerformance:
    """測試內存優化性能"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
        
        # 創建大型測試數據
        np.random.seed(42)
        size = 50000
        self.large_data = pd.DataFrame({
            'int_small': np.random.randint(0, 100, size),
            'int_large': np.random.randint(0, 100000, size),
            'float_data': np.random.randn(size),
            'category_low': np.random.choice(['A', 'B', 'C'], size),
            'category_high': [f'cat_{i % 1000}' for i in range(size)],
            'string_data': [f'string_{i}' for i in range(size)]
        })
    
    def test_memory_profiling_performance(self):
        """測試內存分析性能"""
        # 測試內存使用估算
        metrics = self.benchmark.measure_execution_time(
            MemoryProfiler.estimate_dataframe_memory, self.large_data
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 1.0  # 1秒內完成
        
        memory_usage = metrics['result']
        assert memory_usage > 0
        
        print(f"Memory estimation: {metrics['wall_time']:.4f}s, "
              f"Estimated: {memory_usage:.2f}GB")
    
    def test_dtype_optimization_performance(self):
        """測試數據類型優化性能"""
        # 測試獲取最優數據類型
        metrics_analyze = self.benchmark.measure_execution_time(
            MemoryProfiler.get_optimal_dtypes, self.large_data
        )
        
        assert metrics_analyze['success']
        assert metrics_analyze['wall_time'] < 5.0
        
        # 測試內存優化
        metrics_optimize = self.benchmark.measure_execution_time(
            MemoryProfiler.optimize_dataframe_memory, self.large_data
        )
        
        assert metrics_optimize['success']
        assert metrics_optimize['wall_time'] < 10.0
        
        optimized_data = metrics_optimize['result']
        assert len(optimized_data) == len(self.large_data)
        
        # 計算內存節省
        original_memory = MemoryProfiler.estimate_dataframe_memory(self.large_data)
        optimized_memory = MemoryProfiler.estimate_dataframe_memory(optimized_data)
        memory_saving = (original_memory - optimized_memory) / original_memory * 100
        
        print(f"Memory optimization: {metrics_optimize['wall_time']:.3f}s, "
              f"Memory saving: {memory_saving:.1f}%")
    
    def test_chunk_processing_performance(self):
        """測試分塊處理性能"""
        processor = ChunkProcessor(chunk_size=10000)
        
        def simple_transform(df):
            df['new_feature'] = df['float_data'] * 2
            return df
        
        # 測試分塊處理
        metrics = self.benchmark.measure_execution_time(
            processor.process_dataframe_in_chunks, self.large_data, simple_transform
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 30.0
        
        result = metrics['result']
        assert len(result) == len(self.large_data)
        assert 'new_feature' in result.columns
        
        print(f"Chunk processing: {metrics['wall_time']:.3f}s, "
              f"Memory: {metrics['memory_delta_mb']:.1f}MB")
    
    def test_memory_monitoring_overhead(self):
        """測試內存監控開銷"""
        from src.memory_optimizer import memory_monitor
        
        @memory_monitor(threshold_gb=10.0)
        def simple_function():
            return sum(range(100000))
        
        # 測試帶監控的函數
        metrics_with_monitor = self.benchmark.measure_execution_time(simple_function)
        
        assert metrics_with_monitor['success']
        
        # 測試不帶監控的函數
        def simple_function_no_monitor():
            return sum(range(100000))
        
        metrics_no_monitor = self.benchmark.measure_execution_time(simple_function_no_monitor)
        
        assert metrics_no_monitor['success']
        
        # 監控開銷應該很小
        overhead = metrics_with_monitor['wall_time'] - metrics_no_monitor['wall_time']
        overhead_percent = overhead / metrics_no_monitor['wall_time'] * 100
        
        assert overhead_percent < 50  # 開銷不應該超過50%
        
        print(f"Memory monitor overhead: {overhead:.4f}s ({overhead_percent:.1f}%)")


class TestDataValidationPerformance:
    """測試數據驗證性能"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
        self.validator = DataValidator()
        
        # 創建測試數據
        np.random.seed(42)
        size = 20000
        self.test_data = pd.DataFrame({
            'TransactionID': range(1, size + 1),
            'isFraud': np.random.choice([0, 1], size, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, size),
            'TransactionAmt': np.random.lognormal(3, 1, size),
            **{f'feature_{i}': np.random.randn(size) for i in range(10)},
            **{f'category_{i}': np.random.choice(['A', 'B', 'C'], size) for i in range(5)}
        })
    
    def test_structure_validation_performance(self):
        """測試結構驗證性能"""
        expected_columns = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt']
        
        metrics = self.benchmark.measure_execution_time(
            self.validator.validate_data_structure, self.test_data, expected_columns
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 2.0  # 2秒內完成
        
        result = metrics['result']
        assert 'structure_valid' in result
        
        print(f"Structure validation: {metrics['wall_time']:.3f}s")
    
    def test_quality_validation_performance(self):
        """測試品質驗證性能"""
        metrics = self.benchmark.measure_execution_time(
            self.validator.validate_data_quality, self.test_data
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 10.0  # 10秒內完成
        
        result = metrics['result']
        assert 'quality_score' in result
        assert 'column_quality' in result
        
        print(f"Quality validation: {metrics['wall_time']:.3f}s")
    
    def test_business_rules_validation_performance(self):
        """測試業務規則驗證性能"""
        metrics = self.benchmark.measure_execution_time(
            self.validator.validate_business_rules, self.test_data
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 5.0  # 5秒內完成
        
        result = metrics['result']
        assert 'overall_valid' in result
        assert 'rule_results' in result
        
        print(f"Business rules validation: {metrics['wall_time']:.3f}s")


class TestModelMonitoringPerformance:
    """測試模型監控性能"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
        
        # 創建參考和當前數據
        np.random.seed(42)
        size = 10000
        self.reference_data = pd.DataFrame({
            **{f'feature_{i}': np.random.randn(size) for i in range(20)}
        })
        
        self.current_data = pd.DataFrame({
            **{f'feature_{i}': np.random.randn(size) + 0.1 for i in range(20)}  # 輕微漂移
        })
        
        self.monitor = ModelMonitor('test_model', self.reference_data)
    
    def test_performance_logging_performance(self):
        """測試性能記錄性能"""
        y_true = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        y_pred = np.random.choice([0, 1], 1000, p=[0.93, 0.07])
        y_pred_proba = np.random.rand(1000)
        
        metrics = self.benchmark.measure_execution_time(
            self.monitor.log_performance, y_true, y_pred, y_pred_proba
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 1.0  # 1秒內完成
        assert len(self.monitor.performance_history) == 1
        
        print(f"Performance logging: {metrics['wall_time']:.4f}s")
    
    def test_drift_detection_performance(self):
        """測試漂移檢測性能"""
        features = [f'feature_{i}' for i in range(10)]  # 測試前10個特徵
        
        metrics = self.benchmark.measure_execution_time(
            self.monitor.detect_data_drift, self.current_data, features
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 10.0  # 10秒內完成
        
        drift_results = metrics['result']
        assert len(drift_results) == len(features)
        
        print(f"Drift detection ({len(features)} features): {metrics['wall_time']:.3f}s")
    
    def test_monitoring_report_generation_performance(self):
        """測試監控報告生成性能"""
        # 添加一些歷史數據
        for i in range(10):
            y_true = np.random.choice([0, 1], 100, p=[0.95, 0.05])
            y_pred = np.random.choice([0, 1], 100, p=[0.93, 0.07])
            y_pred_proba = np.random.rand(100)
            self.monitor.log_performance(y_true, y_pred, y_pred_proba)
        
        # 添加漂移記錄
        self.monitor.detect_data_drift(self.current_data, features=['feature_0', 'feature_1'])
        
        metrics = self.benchmark.measure_execution_time(
            self.monitor.generate_monitoring_report, days=7
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 2.0  # 2秒內完成
        
        report = metrics['result']
        assert 'performance_statistics' in report
        assert 'drift_statistics' in report
        
        print(f"Report generation: {metrics['wall_time']:.3f}s")


class TestRegressionDetection:
    """測試性能回歸檢測"""
    
    def setup_method(self):
        """測試設置"""
        self.benchmark = PerformanceBenchmark()
    
    def test_detect_performance_regression(self):
        """測試性能回歸檢測"""
        # 模擬基線性能
        baseline_metrics = {
            'wall_time': 1.0,
            'cpu_time': 0.8,
            'peak_memory_mb': 100.0,
            'success': True
        }
        
        # 模擬當前性能（性能退化）
        current_metrics = {
            'wall_time': 1.5,  # 50%變慢
            'cpu_time': 1.2,
            'peak_memory_mb': 150.0,  # 50%更多內存
            'success': True
        }
        
        # 存儲基線
        self.benchmark.store_benchmark('test_function', baseline_metrics)
        
        # 比較性能
        comparison = self.benchmark.compare_with_baseline('test_function', current_metrics, tolerance=0.1)
        
        # 應該檢測到回歸
        assert comparison['overall_regression'] == True
        assert comparison['time_regression'] == True
        assert comparison['memory_regression'] == True
        assert comparison['time_change_percent'] == 50.0
        assert comparison['memory_change_percent'] == 50.0
    
    def test_acceptable_performance_variation(self):
        """測試可接受的性能變化"""
        baseline_metrics = {
            'wall_time': 1.0,
            'cpu_time': 0.8,
            'peak_memory_mb': 100.0,
            'success': True
        }
        
        # 模擬輕微的性能變化（在容忍度內）
        current_metrics = {
            'wall_time': 1.05,  # 5%變慢
            'cpu_time': 0.85,
            'peak_memory_mb': 105.0,  # 5%更多內存
            'success': True
        }
        
        self.benchmark.store_benchmark('acceptable_function', baseline_metrics)
        comparison = self.benchmark.compare_with_baseline('acceptable_function', current_metrics, tolerance=0.1)
        
        # 不應該檢測到回歸
        assert comparison['overall_regression'] == False
        assert comparison['time_regression'] == False
        assert comparison['memory_regression'] == False


class TestStressTests:
    """壓力測試"""
    
    def test_large_dataset_stress_test(self):
        """大數據集壓力測試"""
        benchmark = PerformanceBenchmark()
        
        # 創建大數據集
        size = 100000
        large_data = pd.DataFrame({
            'id': range(size),
            **{f'feature_{i}': np.random.randn(size) for i in range(50)}
        })
        
        print(f"Testing with {size:,} rows and {len(large_data.columns)} columns")
        print(f"Estimated memory: {MemoryProfiler.estimate_dataframe_memory(large_data):.2f}GB")
        
        # 測試內存優化
        metrics = benchmark.measure_execution_time(
            MemoryProfiler.optimize_dataframe_memory, large_data
        )
        
        assert metrics['success']
        assert metrics['wall_time'] < 120  # 2分鐘內完成
        
        optimized_data = metrics['result']
        memory_saving = (
            MemoryProfiler.estimate_dataframe_memory(large_data) - 
            MemoryProfiler.estimate_dataframe_memory(optimized_data)
        ) / MemoryProfiler.estimate_dataframe_memory(large_data) * 100
        
        print(f"Large dataset optimization: {metrics['wall_time']:.1f}s, "
              f"Memory saving: {memory_saving:.1f}%")
    
    def test_concurrent_processing_stress(self):
        """並發處理壓力測試"""
        import threading
        import queue
        
        def worker_function(data_queue, result_queue):
            """工作線程函數"""
            while True:
                try:
                    data = data_queue.get(timeout=1)
                    if data is None:
                        break
                    
                    # 執行一些處理
                    result = data.sum().sum()
                    result_queue.put(result)
                    data_queue.task_done()
                    
                except queue.Empty:
                    break
        
        # 創建測試數據
        datasets = []
        for i in range(10):
            datasets.append(pd.DataFrame(np.random.randn(1000, 10)))
        
        # 創建隊列
        data_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # 添加數據到隊列
        for data in datasets:
            data_queue.put(data)
        
        # 創建工作線程
        threads = []
        for i in range(4):  # 4個工作線程
            t = threading.Thread(target=worker_function, args=(data_queue, result_queue))
            t.start()
            threads.append(t)
        
        # 等待完成
        start_time = time.time()
        data_queue.join()
        
        # 停止線程
        for i in range(4):
            data_queue.put(None)
        
        for t in threads:
            t.join()
        
        execution_time = time.time() - start_time
        
        # 檢查結果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == len(datasets)
        assert execution_time < 10.0  # 10秒內完成
        
        print(f"Concurrent processing: {execution_time:.3f}s, "
              f"Processed {len(datasets)} datasets with 4 threads")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])