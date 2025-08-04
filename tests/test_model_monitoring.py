"""
模型監控與漂移檢測模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.append('../src')

from src.model_monitoring import (
    PerformanceMetrics, DriftMetrics, KSTestDriftDetector, ChiSquareDriftDetector,
    MeanShiftDetector, ModelMonitor, create_model_monitor, quick_drift_check
)
from src.exceptions import ModelError, DataValidationError


class TestPerformanceMetrics:
    """測試性能指標數據類"""
    
    def test_performance_metrics_creation(self):
        """測試性能指標創建"""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.96,
            prediction_count=1000,
            fraud_rate=0.035,
            average_prediction_time=0.001
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.92
        assert metrics.recall == 0.88
        assert metrics.f1_score == 0.90
        assert metrics.roc_auc == 0.96
        assert metrics.prediction_count == 1000
        assert metrics.fraud_rate == 0.035
        assert metrics.average_prediction_time == 0.001
    
    def test_performance_metrics_to_dict(self):
        """測試轉換為字典"""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.96,
            prediction_count=1000,
            fraud_rate=0.035,
            average_prediction_time=0.001
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'timestamp' in metrics_dict
        assert isinstance(metrics_dict['timestamp'], str)  # 應該被轉換為ISO格式
        assert metrics_dict['accuracy'] == 0.95
        assert metrics_dict['precision'] == 0.92
    
    def test_performance_metrics_from_dict(self):
        """測試從字典創建實例"""
        timestamp = datetime.now()
        original_metrics = PerformanceMetrics(
            timestamp=timestamp,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            roc_auc=0.96,
            prediction_count=1000,
            fraud_rate=0.035,
            average_prediction_time=0.001
        )
        
        metrics_dict = original_metrics.to_dict()
        restored_metrics = PerformanceMetrics.from_dict(metrics_dict)
        
        assert restored_metrics.timestamp == original_metrics.timestamp
        assert restored_metrics.accuracy == original_metrics.accuracy
        assert restored_metrics.precision == original_metrics.precision


class TestDriftMetrics:
    """測試漂移指標數據類"""
    
    def test_drift_metrics_creation(self):
        """測試漂移指標創建"""
        timestamp = datetime.now()
        metrics = DriftMetrics(
            timestamp=timestamp,
            feature_name='feature1',
            drift_score=0.1,
            p_value=0.03,
            drift_detected=True,
            drift_type='distribution',
            reference_stats={'mean': 10.0, 'std': 2.0},
            current_stats={'mean': 12.0, 'std': 2.5}
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.feature_name == 'feature1'
        assert metrics.drift_score == 0.1
        assert metrics.p_value == 0.03
        assert metrics.drift_detected == True
        assert metrics.drift_type == 'distribution'
        assert metrics.reference_stats == {'mean': 10.0, 'std': 2.0}
        assert metrics.current_stats == {'mean': 12.0, 'std': 2.5}
    
    def test_drift_metrics_serialization(self):
        """測試漂移指標序列化"""
        timestamp = datetime.now()
        original_metrics = DriftMetrics(
            timestamp=timestamp,
            feature_name='feature1',
            drift_score=0.1,
            p_value=0.03,
            drift_detected=True,
            drift_type='distribution',
            reference_stats={'mean': 10.0},
            current_stats={'mean': 12.0}
        )
        
        # 轉換為字典再恢復
        metrics_dict = original_metrics.to_dict()
        restored_metrics = DriftMetrics.from_dict(metrics_dict)
        
        assert restored_metrics.timestamp == original_metrics.timestamp
        assert restored_metrics.feature_name == original_metrics.feature_name
        assert restored_metrics.drift_score == original_metrics.drift_score
        assert restored_metrics.drift_detected == original_metrics.drift_detected


class TestKSTestDriftDetector:
    """測試Kolmogorov-Smirnov漂移檢測器"""
    
    def setup_method(self):
        """測試設置"""
        self.detector = KSTestDriftDetector()
    
    def test_no_drift_detection(self):
        """測試無漂移情況"""
        # 生成相似分佈的數據
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(0.05, 1.02, 1000)  # 輕微差異
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data, threshold=0.05
        )
        
        # 應該不會檢測到顯著漂移
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert isinstance(p_value, float)
        assert 0 <= drift_score <= 1
        assert 0 <= p_value <= 1
    
    def test_drift_detection(self):
        """測試檢測到漂移的情況"""
        # 生成明顯不同分佈的數據
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(2, 1, 1000)  # 明顯不同的均值
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data, threshold=0.05
        )
        
        # 應該檢測到顯著漂移
        assert drift_detected == True
        assert drift_score > 0
        assert p_value < 0.05
    
    def test_nan_handling(self):
        """測試NaN值處理"""
        reference_data = np.array([1, 2, 3, np.nan, 5])
        current_data = np.array([1.1, 2.1, np.nan, 4.1, 5.1])
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data
        )
        
        # 應該能處理NaN值而不崩潰
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert isinstance(p_value, float)
    
    def test_empty_data_handling(self):
        """測試空數據處理"""
        reference_data = np.array([])
        current_data = np.array([1, 2, 3])
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data
        )
        
        # 應該返回無漂移
        assert drift_detected == False
        assert drift_score == 0.0
        assert p_value == 1.0
    
    def test_all_nan_data(self):
        """測試全為NaN的數據"""
        reference_data = np.array([np.nan, np.nan, np.nan])
        current_data = np.array([np.nan, np.nan, np.nan])
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data
        )
        
        # 應該返回無漂移
        assert drift_detected == False
        assert drift_score == 0.0
        assert p_value == 1.0


class TestChiSquareDriftDetector:
    """測試卡方漂移檢測器"""
    
    def setup_method(self):
        """測試設置"""
        self.detector = ChiSquareDriftDetector()
    
    def test_no_categorical_drift(self):
        """測試無類別漂移情況"""
        # 相似的類別分佈
        reference_data = np.array(['A', 'B', 'C'] * 100 + ['A'] * 50)
        current_data = np.array(['A', 'B', 'C'] * 95 + ['A'] * 65)
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data, threshold=0.05
        )
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert isinstance(p_value, float)
        assert drift_score >= 0
    
    def test_categorical_drift_detection(self):
        """測試檢測類別漂移"""
        # 明顯不同的類別分佈
        reference_data = np.array(['A'] * 800 + ['B'] * 200)
        current_data = np.array(['A'] * 200 + ['B'] * 800)  # 比例顛倒
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data, threshold=0.05
        )
        
        # 應該檢測到顯著漂移
        assert drift_detected == True
        assert drift_score > 0
        assert p_value < 0.05
    
    def test_new_categories(self):
        """測試新類別的處理"""
        reference_data = np.array(['A', 'B', 'C'] * 100)
        current_data = np.array(['A', 'B', 'C', 'D'] * 75)  # 新增類別D
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data
        )
        
        # 應該能處理新類別
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert isinstance(p_value, float)


class TestMeanShiftDetector:
    """測試均值偏移檢測器"""
    
    def setup_method(self):
        """測試設置"""
        self.detector = MeanShiftDetector()
    
    def test_no_mean_shift(self):
        """測試無均值偏移"""
        np.random.seed(42)
        reference_data = np.random.normal(10, 2, 1000)
        current_data = np.random.normal(10.1, 2, 1000)  # 輕微差異
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data, threshold=0.5
        )
        
        assert isinstance(drift_detected, bool)
        assert isinstance(drift_score, float)
        assert isinstance(p_value, float)
        assert drift_score >= 0
    
    def test_mean_shift_detection(self):
        """測試均值偏移檢測"""
        np.random.seed(42)
        reference_data = np.random.normal(10, 2, 1000)
        current_data = np.random.normal(15, 2, 1000)  # 明顯的均值差異
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data, threshold=0.5
        )
        
        # 應該檢測到均值偏移
        assert drift_detected == True
        assert drift_score > 0.5
    
    def test_zero_std_handling(self):
        """測試零標準差處理"""
        reference_data = np.array([5, 5, 5, 5, 5])  # 零方差
        current_data = np.array([6, 6, 6, 6, 6])
        
        drift_detected, drift_score, p_value = self.detector.detect_drift(
            reference_data, current_data
        )
        
        # 應該返回無漂移（因為無法計算標準化差異）
        assert drift_detected == False
        assert drift_score == 0.0
        assert p_value == 1.0


class TestModelMonitor:
    """測試模型監控器"""
    
    def setup_method(self):
        """測試設置"""
        # 創建參考數據
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        self.monitor = ModelMonitor('test_model', self.reference_data)
    
    def test_monitor_initialization(self):
        """測試監控器初始化"""
        assert self.monitor.model_name == 'test_model'
        assert self.monitor.reference_data is not None
        assert len(self.monitor.performance_history) == 0
        assert len(self.monitor.drift_history) == 0
        assert 'numerical' in self.monitor.drift_detectors
        assert 'categorical' in self.monitor.drift_detectors
        assert 'mean_shift' in self.monitor.drift_detectors
    
    def test_log_performance(self):
        """測試性能記錄"""
        # 模擬預測結果
        y_true = np.array([0, 1, 0, 1, 0] * 100)
        y_pred = np.array([0, 1, 0, 0, 0] * 100)  # 一些錯誤預測
        y_pred_proba = np.random.rand(500)
        prediction_times = [0.001] * 500
        
        initial_count = len(self.monitor.performance_history)
        
        self.monitor.log_performance(y_true, y_pred, y_pred_proba, prediction_times)
        
        # 檢查性能記錄是否被添加
        assert len(self.monitor.performance_history) == initial_count + 1
        
        latest_metrics = self.monitor.performance_history[-1]
        assert isinstance(latest_metrics, PerformanceMetrics)
        assert 0 <= latest_metrics.accuracy <= 1
        assert 0 <= latest_metrics.precision <= 1
        assert 0 <= latest_metrics.recall <= 1
        assert 0 <= latest_metrics.f1_score <= 1
        assert 0 <= latest_metrics.roc_auc <= 1
        assert latest_metrics.prediction_count == 500
    
    def test_log_performance_edge_cases(self):
        """測試性能記錄邊界情況"""
        # 全部預測為同一類
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        self.monitor.log_performance(y_true, y_pred, y_pred_proba)
        
        # 應該能處理邊界情況而不崩潰
        assert len(self.monitor.performance_history) > 0
        
        latest_metrics = self.monitor.performance_history[-1]
        assert latest_metrics.accuracy == 1.0  # 全部預測正確
    
    def test_detect_data_drift_no_reference(self):
        """測試無參考數據的漂移檢測"""
        monitor_no_ref = ModelMonitor('test_model_no_ref')
        
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100)
        })
        
        drift_results = monitor_no_ref.detect_data_drift(current_data)
        
        # 應該返回空列表
        assert drift_results == []
    
    def test_detect_data_drift_success(self):
        """測試成功的漂移檢測"""
        # 創建有漂移的當前數據
        current_data = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 1000),  # 均值漂移
            'feature2': np.random.normal(10, 2, 1000),  # 無漂移
            'categorical_feature': np.random.choice(['A', 'D', 'E'], 1000)  # 類別漂移
        })
        
        drift_results = self.monitor.detect_data_drift(current_data)
        
        # 檢查漂移檢測結果
        assert len(drift_results) > 0
        
        for drift in drift_results:
            assert isinstance(drift, DriftMetrics)
            assert drift.feature_name in current_data.columns
            assert isinstance(drift.drift_detected, bool)
            assert isinstance(drift.drift_score, float)
            assert isinstance(drift.p_value, float)
    
    def test_detect_data_drift_specific_features(self):
        """測試指定特徵的漂移檢測"""
        current_data = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.normal(0, 1, 1000)  # 新特徵
        })
        
        # 只檢測特定特徵
        drift_results = self.monitor.detect_data_drift(current_data, features=['feature1', 'feature2'])
        
        # 檢查只檢測了指定特徵
        detected_features = [drift.feature_name for drift in drift_results]
        assert 'feature1' in detected_features
        assert 'feature2' in detected_features
        assert 'feature3' not in detected_features
    
    def test_calculate_feature_stats_numerical(self):
        """測試數值特徵統計計算"""
        data = np.array([1, 2, 3, 4, 5, np.nan])
        stats = self.monitor._calculate_feature_stats(data, 'numerical')
        
        expected_keys = ['mean', 'std', 'min', 'max', 'q25', 'q50', 'q75', 'missing_rate']
        for key in expected_keys:
            assert key in stats
        
        assert stats['mean'] == 3.0  # (1+2+3+4+5)/5
        assert stats['missing_rate'] == 1/6  # 1 NaN out of 6 values
    
    def test_calculate_feature_stats_categorical(self):
        """測試類別特徵統計計算"""
        data = np.array(['A', 'B', 'A', 'C', 'A', None])
        stats = self.monitor._calculate_feature_stats(data, 'categorical')
        
        expected_keys = ['unique_values', 'top_values', 'missing_rate']
        for key in expected_keys:
            assert key in stats
        
        assert stats['unique_values'] > 0
        assert 'A' in stats['top_values']  # A是最頻繁的值
        assert stats['missing_rate'] == 1/6
    
    def test_check_performance_degradation(self):
        """測試性能退化檢查"""
        # 添加一些歷史性能記錄
        for i in range(6):
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(days=i),
                accuracy=0.9,
                precision=0.9,
                recall=0.9,
                f1_score=0.9,
                roc_auc=0.95 - i * 0.01,  # 逐漸下降
                prediction_count=1000,
                fraud_rate=0.035,
                average_prediction_time=0.001
            )
            self.monitor.performance_history.append(metrics)
        
        # 模擬一個性能大幅下降的新記錄
        poor_metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=0.8,
            precision=0.8,
            recall=0.8,
            f1_score=0.8,
            roc_auc=0.85,  # 顯著下降
            prediction_count=1000,
            fraud_rate=0.035,
            average_prediction_time=0.001
        )
        
        # 測試性能退化檢查（這會在log_performance中自動調用）
        with patch.object(self.monitor, '_trigger_alert') as mock_alert:
            self.monitor._check_performance_degradation(poor_metrics)
            # 檢查是否觸發了警告
            mock_alert.assert_called_once()
    
    def test_generate_monitoring_report(self):
        """測試生成監控報告"""
        # 添加一些歷史數據
        for i in range(3):
            # 性能記錄
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(days=i),
                accuracy=0.9,
                precision=0.9,
                recall=0.9,
                f1_score=0.9,
                roc_auc=0.95,
                prediction_count=1000,
                fraud_rate=0.035,
                average_prediction_time=0.001
            )
            self.monitor.performance_history.append(metrics)
            
            # 漂移記錄
            drift = DriftMetrics(
                timestamp=datetime.now() - timedelta(days=i),
                feature_name=f'feature_{i}',
                drift_score=0.1,
                p_value=0.05,
                drift_detected=i == 0,  # 只有第一個檢測到漂移
                drift_type='distribution',
                reference_stats={'mean': 0},
                current_stats={'mean': 1}
            )
            self.monitor.drift_history.append(drift)
        
        report = self.monitor.generate_monitoring_report(days=7)
        
        # 檢查報告結構
        assert 'report_period' in report
        assert 'generated_at' in report
        assert 'model_name' in report
        assert 'performance_statistics' in report
        assert 'drift_statistics' in report
        
        # 檢查性能統計
        perf_stats = report['performance_statistics']
        assert 'average_auc' in perf_stats
        assert 'total_predictions' in perf_stats
        assert 'performance_trend' in perf_stats
        
        # 檢查漂移統計
        drift_stats = report['drift_statistics']
        assert 'total_features_monitored' in drift_stats
        assert 'features_with_drift' in drift_stats
        assert 'drift_rate' in drift_stats
    
    def test_calculate_performance_trend(self):
        """測試性能趨勢計算"""
        # 測試上升趨勢
        improving_metrics = [
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.90, 1000, 0.035, 0.001),
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.92, 1000, 0.035, 0.001),
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.94, 1000, 0.035, 0.001)
        ]
        trend = self.monitor._calculate_performance_trend(improving_metrics)
        assert trend == 'improving'
        
        # 測試下降趨勢
        declining_metrics = [
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.94, 1000, 0.035, 0.001),
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.92, 1000, 0.035, 0.001),
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.90, 1000, 0.035, 0.001)
        ]
        trend = self.monitor._calculate_performance_trend(declining_metrics)
        assert trend == 'declining'
        
        # 測試穩定趨勢
        stable_metrics = [
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.92, 1000, 0.035, 0.001),
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.92, 1000, 0.035, 0.001),
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.92, 1000, 0.035, 0.001)
        ]
        trend = self.monitor._calculate_performance_trend(stable_metrics)
        assert trend == 'stable'
        
        # 測試數據不足
        insufficient_metrics = [
            PerformanceMetrics(datetime.now(), 0.9, 0.9, 0.9, 0.9, 0.92, 1000, 0.035, 0.001)
        ]
        trend = self.monitor._calculate_performance_trend(insufficient_metrics)
        assert trend == 'insufficient_data'
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_history(self, mock_show):
        """測試繪製性能歷史圖"""
        # 添加一些歷史數據
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(days=i),
                accuracy=0.9,
                precision=0.9,
                recall=0.9,
                f1_score=0.9,
                roc_auc=0.95 - i * 0.01,
                prediction_count=1000,
                fraud_rate=0.035,
                average_prediction_time=0.001
            )
            self.monitor.performance_history.append(metrics)
        
        # 測試繪圖（不實際顯示）
        self.monitor.plot_performance_history(days=10)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    @patch('matplotlib.pyplot.show')
    def test_plot_performance_history_no_data(self, mock_show):
        """測試無歷史數據時的繪圖"""
        # 清空歷史數據
        self.monitor.performance_history = []
        
        # 應該不會崩潰，只會發出警告
        self.monitor.plot_performance_history()
        
        # matplotlib.show不應該被調用
        assert not mock_show.called
    
    @patch('matplotlib.pyplot.show')
    def test_plot_drift_summary(self, mock_show):
        """測試繪製漂移摘要圖"""
        # 添加一些漂移數據
        for i in range(3):
            drift = DriftMetrics(
                timestamp=datetime.now() - timedelta(days=i),
                feature_name=f'feature_{i}',
                drift_score=0.1,
                p_value=0.01,
                drift_detected=True,
                drift_type='distribution',
                reference_stats={'mean': 0},
                current_stats={'mean': 1}
            )
            self.monitor.drift_history.append(drift)
        
        # 測試繪圖
        self.monitor.plot_drift_summary(days=7)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    def test_save_and_load_monitoring_data(self):
        """測試保存和載入監控數據"""
        # 添加一些測試數據
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            accuracy=0.9, precision=0.9, recall=0.9, f1_score=0.9, roc_auc=0.95,
            prediction_count=1000, fraud_rate=0.035, average_prediction_time=0.001
        )
        self.monitor.performance_history.append(metrics)
        
        drift = DriftMetrics(
            timestamp=datetime.now(),
            feature_name='test_feature',
            drift_score=0.1, p_value=0.05, drift_detected=True, drift_type='distribution',
            reference_stats={'mean': 0}, current_stats={'mean': 1}
        )
        self.monitor.drift_history.append(drift)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存數據
            self.monitor.save_monitoring_data(temp_path)
            
            # 檢查文件是否被創建
            assert os.path.exists(temp_path)
            
            # 創建新的監控器並載入數據
            new_monitor = ModelMonitor('new_model')
            new_monitor.load_monitoring_data(temp_path)
            
            # 檢查數據是否正確載入
            assert new_monitor.model_name == self.monitor.model_name
            assert len(new_monitor.performance_history) == len(self.monitor.performance_history)
            assert len(new_monitor.drift_history) == len(self.monitor.drift_history)
            
            # 檢查具體數據
            loaded_metrics = new_monitor.performance_history[0]
            assert loaded_metrics.accuracy == metrics.accuracy
            assert loaded_metrics.roc_auc == metrics.roc_auc
            
            loaded_drift = new_monitor.drift_history[0]
            assert loaded_drift.feature_name == drift.feature_name
            assert loaded_drift.drift_detected == drift.drift_detected
        
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_monitoring_data_file_not_found(self):
        """測試載入不存在的監控數據文件"""
        # 應該不會崩潰，只會記錄錯誤
        self.monitor.load_monitoring_data('nonexistent_file.json')
        
        # 數據應該保持不變
        assert self.monitor.model_name == 'test_model'


class TestUtilityFunctions:
    """測試工具函數"""
    
    def test_create_model_monitor(self):
        """測試創建模型監控器"""
        reference_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'A', 'B']
        })
        
        monitor = create_model_monitor('test_model', reference_data)
        
        assert isinstance(monitor, ModelMonitor)
        assert monitor.model_name == 'test_model'
        assert monitor.reference_data is not None
        pd.testing.assert_frame_equal(monitor.reference_data, reference_data)
    
    def test_create_model_monitor_no_reference(self):
        """測試創建無參考數據的監控器"""
        monitor = create_model_monitor('test_model')
        
        assert isinstance(monitor, ModelMonitor)
        assert monitor.model_name == 'test_model'
        assert monitor.reference_data is None
    
    def test_quick_drift_check(self):
        """測試快速漂移檢查"""
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        # 創建有漂移的當前數據
        current_data = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 1000),  # 有漂移
            'feature2': np.random.normal(10, 2, 1000),  # 無漂移
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)  # 可能有漂移
        })
        
        drift_results = quick_drift_check(reference_data, current_data)
        
        # 檢查返回結果
        assert isinstance(drift_results, dict)
        assert 'feature1' in drift_results
        assert 'feature2' in drift_results
        
        # 檢查值類型
        for feature, has_drift in drift_results.items():
            assert isinstance(has_drift, bool)
    
    def test_quick_drift_check_specific_features(self):
        """測試指定特徵的快速漂移檢查"""
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.normal(5, 1, 100)
        })
        
        current_data = pd.DataFrame({
            'feature1': np.random.normal(2, 1, 100),
            'feature2': np.random.normal(10, 2, 100),
            'feature3': np.random.normal(5, 1, 100)
        })
        
        # 只檢查指定特徵
        drift_results = quick_drift_check(reference_data, current_data, features=['feature1', 'feature2'])
        
        assert 'feature1' in drift_results
        assert 'feature2' in drift_results
        assert 'feature3' not in drift_results


class TestModelMonitoringIntegration:
    """測試模型監控集成功能"""
    
    def test_complete_monitoring_workflow(self):
        """測試完整的監控工作流程"""
        # 1. 創建參考數據和監控器
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(10, 2, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        monitor = ModelMonitor('fraud_detection_model', reference_data)
        
        # 2. 模擬一段時間的性能記錄
        for i in range(10):
            y_true = np.random.choice([0, 1], 100, p=[0.95, 0.05])
            y_pred = np.random.choice([0, 1], 100, p=[0.93, 0.07])
            y_pred_proba = np.random.rand(100)
            
            monitor.log_performance(y_true, y_pred, y_pred_proba)
        
        # 3. 進行漂移檢測
        current_data = pd.DataFrame({
            'feature1': np.random.normal(1, 1, 500),  # 有漂移
            'feature2': np.random.normal(10, 2, 500),  # 無漂移
            'category': np.random.choice(['A', 'B', 'D'], 500)  # 新類別
        })
        
        drift_results = monitor.detect_data_drift(current_data)
        
        # 4. 生成監控報告
        report = monitor.generate_monitoring_report(days=1)
        
        # 5. 驗證整個流程
        assert len(monitor.performance_history) == 10
        assert len(drift_results) > 0
        assert report['model_name'] == 'fraud_detection_model'
        assert 'performance_statistics' in report
        assert 'drift_statistics' in report
        
        # 檢查是否檢測到漂移
        drift_detected_features = [d.feature_name for d in drift_results if d.drift_detected]
        assert len(drift_detected_features) > 0
    
    def test_continuous_monitoring_simulation(self):
        """測試連續監控模擬"""
        # 創建基準數據
        baseline_data = pd.DataFrame({
            'amount': np.random.lognormal(3, 1, 1000),
            'hour': np.random.randint(0, 24, 1000),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail'], 1000)
        })
        
        monitor = ModelMonitor('continuous_monitor', baseline_data)
        
        # 模擬7天的連續監控
        for day in range(7):
            # 模擬該天的新數據（逐漸產生漂移）
            drift_factor = day * 0.1
            daily_data = pd.DataFrame({
                'amount': np.random.lognormal(3 + drift_factor, 1, 200),  # 逐漸增加
                'hour': np.random.randint(0, 24, 200),
                'merchant_category': np.random.choice(
                    ['grocery', 'gas', 'restaurant', 'retail', 'online'] if day > 3 else 
                    ['grocery', 'gas', 'restaurant', 'retail'], 200
                )  # 第4天後出現新類別
            })
            
            # 漂移檢測
            drift_results = monitor.detect_data_drift(daily_data)
            
            # 模擬性能數據
            y_true = np.random.choice([0, 1], 200, p=[0.95, 0.05])
            y_pred = np.random.choice([0, 1], 200, p=[0.93 - day*0.01, 0.07 + day*0.01])  # 性能逐漸下降
            y_pred_proba = np.random.rand(200)
            
            monitor.log_performance(y_true, y_pred, y_pred_proba)
        
        # 檢查監控結果
        assert len(monitor.performance_history) == 7
        assert len(monitor.drift_history) > 0
        
        # 生成最終報告
        final_report = monitor.generate_monitoring_report(days=7)
        
        # 檢查報告內容
        assert final_report['performance_statistics']['performance_trend'] in ['stable', 'declining', 'improving']
        assert final_report['drift_statistics']['total_features_monitored'] == 3
        
        # 檢查是否檢測到性能退化
        performance_trend = final_report['performance_statistics']['performance_trend']
        assert performance_trend in ['declining', 'stable']  # 由於模擬的性能下降


class TestModelMonitoringEdgeCases:
    """測試模型監控邊界情況"""
    
    def test_monitoring_with_small_data(self):
        """測試小數據集監控"""
        small_reference = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['A', 'B', 'C']
        })
        
        monitor = ModelMonitor('small_data_model', small_reference)
        
        small_current = pd.DataFrame({
            'feature1': [1.1, 2.1, 3.1],
            'feature2': ['A', 'B', 'C']
        })
        
        # 應該能處理小數據集
        drift_results = monitor.detect_data_drift(small_current)
        assert isinstance(drift_results, list)
    
    def test_monitoring_with_missing_features(self):
        """測試缺少特徵的監控"""
        reference_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        monitor = ModelMonitor('missing_features_model', reference_data)
        
        # 當前數據缺少某些特徵
        current_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
            # feature3 缺失
        })
        
        drift_results = monitor.detect_data_drift(current_data)
        
        # 應該只檢測存在的特徵
        detected_features = [d.feature_name for d in drift_results]
        assert 'feature1' in detected_features
        assert 'feature2' in detected_features
        assert 'feature3' not in detected_features
    
    def test_monitoring_with_all_constant_features(self):
        """測試全為常數的特徵監控"""
        constant_reference = pd.DataFrame({
            'constant_feature': [1] * 100
        })
        
        monitor = ModelMonitor('constant_model', constant_reference)
        
        constant_current = pd.DataFrame({
            'constant_feature': [1] * 50
        })
        
        # 應該能處理常數特徵
        drift_results = monitor.detect_data_drift(constant_current)
        assert len(drift_results) >= 0  # 可能沒有檢測到漂移，但不應該崩潰


if __name__ == "__main__":
    pytest.main([__file__, "-v"])