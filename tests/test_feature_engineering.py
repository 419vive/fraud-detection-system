"""
特徵工程模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime
import gc

sys.path.append('../src')

from src.feature_engineering import (
    FeatureEngineer, engineer_features, fast_feature_engineering
)
from src.exceptions import (
    FeatureEngineeringError, FeatureCreationError, FeatureSelectionError
)
from src.config import ConfigManager


class TestFeatureEngineer:
    """測試特徵工程器類"""
    
    def setup_method(self):
        """測試設置"""
        self.config_manager = ConfigManager()
        self.engineer = FeatureEngineer(self.config_manager)
        
        # 創建測試數據
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'TransactionID': range(1, 101),
            'isFraud': np.random.choice([0, 1], 100, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, 100),
            'TransactionAmt': np.random.lognormal(3, 1, 100),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], 100),
            'card1': np.random.randint(1000, 20000, 100),
            'card2': np.random.randint(100, 600, 100),
            'addr1': np.random.choice(range(100, 500), 100),
            'C1': np.random.randn(100),
            'C2': np.random.randn(100),
            'DeviceType': np.random.choice(['desktop', 'mobile', 'tablet'], 100),
            'id_30': np.random.choice(['Windows', 'iOS', 'Android'], 100),
            'id_31': np.random.choice(['chrome', 'safari', 'firefox'], 100)
        })
        
        # 添加一些缺失值
        missing_indices = np.random.choice(100, 20, replace=False)
        self.sample_data.loc[missing_indices[:10], 'C1'] = np.nan
        self.sample_data.loc[missing_indices[10:], 'card2'] = np.nan
    
    def test_initialization(self):
        """測試初始化"""
        assert isinstance(self.engineer.label_encoders, dict)
        assert isinstance(self.engineer.scalers, dict)
        assert isinstance(self.engineer.feature_importance, dict)
        assert self.engineer.config is not None
        assert self.engineer.enable_parallel in [True, False]
        assert self.engineer.enable_caching in [True, False]
    
    def test_create_time_features_success(self):
        """測試成功創建時間特徵"""
        result = self.engineer.create_time_features(self.sample_data)
        
        # 檢查新特徵是否被創建
        expected_features = ['hour', 'day', 'week', 'time_of_day', 'is_weekend', 
                           'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        for feature in expected_features:
            assert feature in result.columns
        
        # 檢查特徵值的合理性
        assert result['hour'].min() >= 0
        assert result['hour'].max() < 24
        assert result['day'].min() >= 0
        assert result['day'].max() < 7
        assert result['is_weekend'].isin([0, 1]).all()
        assert result['time_of_day'].min() >= 0
        
        # 檢查sin/cos特徵的範圍
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1
        assert result['hour_cos'].min() >= -1
        assert result['hour_cos'].max() <= 1
    
    def test_create_time_features_missing_column(self):
        """測試時間列不存在的情況"""
        data_without_time = self.sample_data.drop(columns=['TransactionDT'])
        
        with pytest.raises(FeatureCreationError, match="時間列 TransactionDT 不存在於數據中"):
            self.engineer.create_time_features(data_without_time)
    
    def test_create_time_features_null_values(self):
        """測試時間列全為空的情況"""
        data_null_time = self.sample_data.copy()
        data_null_time['TransactionDT'] = np.nan
        
        with pytest.raises(FeatureCreationError, match="時間列 TransactionDT 完全為空"):
            self.engineer.create_time_features(data_null_time)
    
    def test_create_time_features_negative_values(self):
        """測試時間戳包含負值的情況"""
        data_negative_time = self.sample_data.copy()
        data_negative_time.loc[0, 'TransactionDT'] = -100
        
        with pytest.raises(FeatureCreationError, match="時間戳包含負值"):
            self.engineer.create_time_features(data_negative_time)
    
    def test_create_time_features_caching(self):
        """測試時間特徵緩存功能"""
        self.engineer.enable_caching = True
        
        # 第一次調用
        result1 = self.engineer.create_time_features(self.sample_data)
        cache_size_after_first = len(self.engineer.feature_cache)
        
        # 第二次調用相同數據
        result2 = self.engineer.create_time_features(self.sample_data)
        cache_size_after_second = len(self.engineer.feature_cache)
        
        # 檢查結果相同且緩存被使用
        pd.testing.assert_frame_equal(result1, result2)
        assert cache_size_after_first == cache_size_after_second == 1
    
    def test_create_transaction_amount_features_success(self):
        """測試成功創建交易金額特徵"""
        result = self.engineer.create_transaction_amount_features(self.sample_data)
        
        # 檢查新特徵是否被創建
        expected_features = ['TransactionAmt_log', 'TransactionAmt_sqrt', 'amt_range', 
                           'is_round_amount', 'is_common_amount']
        for feature in expected_features:
            assert feature in result.columns
        
        # 檢查特徵值的合理性
        assert result['TransactionAmt_log'].min() >= 0  # log1p ensures non-negative
        assert result['TransactionAmt_sqrt'].min() >= 0
        assert result['is_round_amount'].isin([0, 1]).all()
        assert result['is_common_amount'].isin([0, 1]).all()
    
    def test_create_transaction_amount_features_missing_column(self):
        """測試金額列不存在的情況"""
        data_without_amount = self.sample_data.drop(columns=['TransactionAmt'])
        
        with pytest.raises(FeatureCreationError, match="金額列 TransactionAmt 不存在於數據中"):
            self.engineer.create_transaction_amount_features(data_without_amount)
    
    def test_create_transaction_amount_features_negative_amounts(self):
        """測試交易金額包含負值的情況"""
        data_negative_amount = self.sample_data.copy()
        data_negative_amount.loc[0, 'TransactionAmt'] = -100
        
        with pytest.raises(FeatureCreationError, match="交易金額包含負值"):
            self.engineer.create_transaction_amount_features(data_negative_amount)
    
    def test_create_aggregation_features_success(self):
        """測試成功創建聚合特徵"""
        group_cols = ['card1', 'ProductCD']
        agg_cols = ['TransactionAmt', 'C1']
        agg_funcs = ['mean', 'count']
        
        result = self.engineer.create_aggregation_features(
            self.sample_data, group_cols, agg_cols, agg_funcs
        )
        
        # 檢查結果DataFrame形狀
        assert len(result) == len(self.sample_data)
        assert len(result.columns) > len(self.sample_data.columns)
        
        # 檢查是否創建了聚合特徵
        expected_feature_patterns = ['card1_TransactionAmt_mean', 'ProductCD_C1_count']
        created_features = [col for col in result.columns if any(pattern in col for pattern in expected_feature_patterns)]
        assert len(created_features) > 0
    
    def test_create_aggregation_features_parallel_processing(self):
        """測試並行處理聚合特徵"""
        self.engineer.enable_parallel = True
        
        group_cols = ['card1', 'ProductCD', 'addr1']
        agg_cols = ['TransactionAmt', 'C1', 'C2']  
        agg_funcs = ['mean', 'std', 'count']
        
        result = self.engineer.create_aggregation_features(
            self.sample_data, group_cols, agg_cols, agg_funcs
        )
        
        # 檢查結果
        assert len(result) == len(self.sample_data)
        assert len(result.columns) > len(self.sample_data.columns)
        
        # 檢查處理時間是否被記錄
        assert 'aggregation_features' in self.engineer.processing_times
        assert 'aggregation_features' in self.engineer.feature_counts
    
    def test_create_card_based_features(self):
        """測試創建基於卡片的特徵"""
        result = self.engineer.create_card_based_features(self.sample_data)
        
        # 檢查是否創建了卡片相關特徵
        card_feature_patterns = ['card1_TransactionAmt', 'card2_TransactionAmt']
        created_features = [col for col in result.columns 
                          if any(pattern in col for pattern in card_feature_patterns)]
        assert len(created_features) > 0
    
    def test_create_address_based_features(self):
        """測試創建基於地址的特徵"""
        result = self.engineer.create_address_based_features(self.sample_data)
        
        # 檢查是否創建了地址相關特徵
        addr_feature_patterns = ['addr1_TransactionAmt']
        created_features = [col for col in result.columns 
                          if any(pattern in col for pattern in addr_feature_patterns)]
        assert len(created_features) > 0
    
    def test_create_device_based_features(self):
        """測試創建基於設備的特徵"""
        result = self.engineer.create_device_based_features(self.sample_data)
        
        # 檢查是否創建了設備相關特徵
        device_feature_patterns = ['DeviceType_TransactionAmt', 'id_30_TransactionAmt']
        created_features = [col for col in result.columns 
                          if any(pattern in col for pattern in device_feature_patterns)]
        assert len(created_features) > 0
    
    def test_create_interaction_features(self):
        """測試創建交互特徵"""
        feature_pairs = [('TransactionAmt', 'C1'), ('card1', 'C2')]
        result = self.engineer.create_interaction_features(self.sample_data, feature_pairs)
        
        # 檢查交互特徵是否被創建
        expected_features = [
            'TransactionAmt_x_C1', 'TransactionAmt_div_C1', 'TransactionAmt_minus_C1',
            'card1_x_C2', 'card1_div_C2', 'card1_minus_C2'
        ]
        for feature in expected_features:
            assert feature in result.columns
        
        # 檢查除零處理
        assert not result['TransactionAmt_div_C1'].isin([np.inf, -np.inf]).any()
    
    def test_encode_categorical_features_label(self):
        """測試標籤編碼"""
        categorical_features = ['ProductCD', 'DeviceType']
        result = self.engineer.encode_categorical_features(
            self.sample_data, categorical_features, method='label'
        )
        
        # 檢查編碼結果
        for feature in categorical_features:
            assert result[feature].dtype in ['int32', 'int64']
            assert feature in self.engineer.label_encoders
        
    def test_encode_categorical_features_frequency(self):
        """測試頻率編碼"""
        categorical_features = ['ProductCD']
        result = self.engineer.encode_categorical_features(
            self.sample_data, categorical_features, method='frequency'
        )
        
        # 檢查頻率編碼特徵
        freq_feature = 'ProductCD_freq'
        assert freq_feature in result.columns
        assert result[freq_feature].min() >= 1  # 至少出現1次
    
    def test_handle_missing_values_default(self):
        """測試默認缺失值處理"""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[:20, 'C1'] = np.nan
        data_with_missing.loc[:15, 'ProductCD'] = np.nan
        
        result = self.engineer.handle_missing_values(data_with_missing)
        
        # 檢查缺失值是否被填補
        assert result['C1'].isnull().sum() == 0
        assert result['ProductCD'].isnull().sum() == 0
    
    def test_handle_missing_values_custom_strategy(self):
        """測試自定義缺失值處理策略"""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[:20, 'C1'] = np.nan
        
        strategy = {'numerical': 'mean', 'categorical': 'mode'}
        result = self.engineer.handle_missing_values(data_with_missing, strategy)
        
        # 檢查缺失值是否被填補
        assert result['C1'].isnull().sum() == 0
    
    def test_select_features(self):
        """測試特徵選擇"""
        X = self.sample_data.drop(columns=['isFraud', 'TransactionID'])
        y = self.sample_data['isFraud']
        
        selected_features = self.engineer.select_features(X, y, k=5)
        
        # 檢查選擇的特徵數量
        assert len(selected_features) <= 5
        assert len(selected_features) > 0
        
        # 檢查特徵重要性是否被保存
        assert len(self.engineer.feature_importance) > 0
    
    def test_handle_class_imbalance_smote(self):
        """測試SMOTE處理類別不平衡"""
        X = self.sample_data.select_dtypes(include=[np.number]).drop(columns=['isFraud', 'TransactionID'])
        y = self.sample_data['isFraud']
        
        X_resampled, y_resampled = self.engineer.handle_class_imbalance(X, y, method='smote')
        
        # 檢查重採樣結果
        assert len(X_resampled) >= len(X)  # SMOTE增加樣本
        assert len(y_resampled) == len(X_resampled)
        assert y_resampled.value_counts().min() > y.value_counts().min()  # 少數類樣本增加
    
    def test_handle_class_imbalance_undersampling(self):
        """測試欠採樣處理類別不平衡"""
        X = self.sample_data.select_dtypes(include=[np.number]).drop(columns=['isFraud', 'TransactionID'])
        y = self.sample_data['isFraud']
        
        X_resampled, y_resampled = self.engineer.handle_class_imbalance(X, y, method='undersampling')
        
        # 檢查重採樣結果
        assert len(X_resampled) <= len(X)  # 欠採樣減少樣本
        assert len(y_resampled) == len(X_resampled)
    
    def test_full_feature_engineering_pipeline(self):
        """測試完整特徵工程流水線"""
        result = self.engineer.full_feature_engineering_pipeline(
            self.sample_data, target_col='isFraud', enable_advanced_features=True
        )
        
        # 檢查結果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
        assert len(result.columns) > len(self.sample_data.columns)
        
        # 檢查處理統計是否被記錄
        assert len(self.engineer.processing_times) > 0
        assert len(self.engineer.feature_counts) > 0
    
    def test_full_feature_engineering_pipeline_basic(self):
        """測試基本特徵工程流水線"""
        result = self.engineer.full_feature_engineering_pipeline(
            self.sample_data, target_col='isFraud', enable_advanced_features=False
        )
        
        # 檢查結果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_data)
        # 基本版本創建的特徵應該較少
        assert len(result.columns) >= len(self.sample_data.columns)
    
    def test_get_pipeline_summary(self):
        """測試獲取流水線摘要"""
        self.engineer.full_feature_engineering_pipeline(self.sample_data)
        summary = self.engineer.get_pipeline_summary()
        
        # 檢查摘要內容
        assert 'processing_times' in summary
        assert 'feature_counts' in summary
        assert 'total_time' in summary
        assert 'total_features_created' in summary
        assert isinstance(summary['total_time'], (int, float))
        assert isinstance(summary['total_features_created'], (int, float))


class TestFeatureEngineeringEdgeCases:
    """測試特徵工程邊界情況"""
    
    def setup_method(self):
        """測試設置"""
        self.engineer = FeatureEngineer()
    
    def test_empty_dataframe(self):
        """測試空數據框"""
        empty_df = pd.DataFrame()
        
        # 大多數方法應該能處理空數據框而不崩潰
        result = self.engineer.handle_missing_values(empty_df)
        assert len(result) == 0
        
        num_features, cat_features = self.engineer.identify_feature_types(empty_df)
        assert len(num_features) == 0
        assert len(cat_features) == 0
    
    def test_single_row_dataframe(self):
        """測試單行數據框"""
        single_row_df = pd.DataFrame({
            'TransactionDT': [86400],
            'TransactionAmt': [100.0],
            'isFraud': [0]
        })
        
        result = self.engineer.create_time_features(single_row_df)
        assert len(result) == 1
        assert 'hour' in result.columns
    
    def test_all_missing_categorical_features(self):
        """測試所有類別特徵都缺失的情況"""
        data = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'cat_feature': [np.nan, np.nan, np.nan],
            'num_feature': [1, 2, 3]
        })
        
        result = self.engineer.handle_missing_values(data)
        assert result['cat_feature'].notna().all()
    
    def test_extreme_outliers(self):
        """測試極端異常值處理"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 1e10],  # 極大值
            'feature2': [1, 2, 3, -1e10]  # 極小值
        })
        
        result = self.engineer.handle_outliers(data, ['feature1', 'feature2'])
        
        # 極端值應該被截斷
        assert result['feature1'].max() < 1e10
        assert result['feature2'].min() > -1e10
    
    def test_high_cardinality_categorical(self):
        """測試高基數類別特徵"""
        # 創建高基數類別特徵（每個值都不同）
        data = pd.DataFrame({
            'high_cardinality': [f'category_{i}' for i in range(100)],
            'target': np.random.choice([0, 1], 100)
        })
        
        result = self.engineer._intelligent_categorical_encoding(data, ['high_cardinality'])
        
        # 高基數特徵應該被處理（可能被移除或使用頻率編碼）
        assert isinstance(result, pd.DataFrame)


class TestFeatureEngineeringUtilityFunctions:
    """測試特徵工程工具函數"""
    
    def test_engineer_features_function(self):
        """測試便捷函數"""
        # 創建測試數據
        test_data = pd.DataFrame({
            'TransactionID': range(1, 51),
            'isFraud': np.random.choice([0, 1], 50, p=[0.9, 0.1]),
            'TransactionDT': np.random.randint(86400, 86400*7, 50),
            'TransactionAmt': np.random.lognormal(3, 1, 50),
            'ProductCD': np.random.choice(['W', 'C'], 50)
        })
        
        result_df, summary = engineer_features(test_data, enable_parallel=False, enable_advanced=False)
        
        # 檢查結果
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert len(result_df) == len(test_data)
        assert 'processing_times' in summary
        assert 'feature_counts' in summary
    
    def test_fast_feature_engineering_function(self):
        """測試快速特徵工程函數"""
        # 創建測試數據
        test_data = pd.DataFrame({
            'TransactionID': range(1, 51),
            'isFraud': np.random.choice([0, 1], 50, p=[0.9, 0.1]),
            'TransactionDT': np.random.randint(86400, 86400*7, 50),
            'TransactionAmt': np.random.lognormal(3, 1, 50),
            'ProductCD': np.random.choice(['W', 'C'], 50)
        })
        
        result = fast_feature_engineering(test_data)
        
        # 檢查結果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)
        assert len(result.columns) >= len(test_data.columns)


class TestFeatureEngineeringPerformance:
    """測試特徵工程性能"""
    
    def test_memory_usage_tracking(self):
        """測試內存使用量追踪"""
        # 創建較大的測試數據
        large_data = pd.DataFrame({
            'TransactionID': range(1, 1001),
            'isFraud': np.random.choice([0, 1], 1000, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, 1000),
            'TransactionAmt': np.random.lognormal(3, 1, 1000),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], 1000)
        })
        
        engineer = FeatureEngineer()
        
        # 執行特徵工程並檢查是否有內存使用量統計
        result = engineer.full_feature_engineering_pipeline(large_data, enable_advanced_features=False)
        
        # 檢查處理時間是否被記錄
        summary = engineer.get_pipeline_summary()
        assert summary['total_time'] > 0
    
    def test_parallel_processing_performance(self):
        """測試並行處理性能"""
        # 創建測試數據
        test_data = pd.DataFrame({
            'TransactionID': range(1, 501),
            'isFraud': np.random.choice([0, 1], 500, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, 500),
            'TransactionAmt': np.random.lognormal(3, 1, 500),
            'card1': np.random.randint(1000, 20000, 500),
            'card2': np.random.randint(100, 600, 500),
            'addr1': np.random.choice(range(100, 500), 500)
        })
        
        # 測試並行處理
        engineer_parallel = FeatureEngineer(enable_parallel=True)
        result_parallel = engineer_parallel.create_aggregation_features(
            test_data, ['card1', 'addr1'], ['TransactionAmt'], ['mean', 'count']
        )
        
        # 測試順序處理
        engineer_sequential = FeatureEngineer(enable_parallel=False)
        result_sequential = engineer_sequential.create_aggregation_features(
            test_data, ['card1', 'addr1'], ['TransactionAmt'], ['mean', 'count'] 
        )
        
        # 結果應該相同
        assert len(result_parallel) == len(result_sequential)
        # 都應該創建了新特徵
        assert len(result_parallel.columns) > len(test_data.columns)
        assert len(result_sequential.columns) > len(test_data.columns)


class TestFeatureEngineeringErrorHandling:
    """測試特徵工程錯誤處理"""
    
    def test_invalid_aggregation_parameters(self):
        """測試無效聚合參數"""
        engineer = FeatureEngineer()
        test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['A', 'B', 'C']
        })
        
        # 測試在文本列上使用數值聚合函數
        result = engineer.create_aggregation_features(
            test_data, ['col2'], ['col2'], ['mean']  # 在文本列上計算mean
        )
        
        # 應該優雅地處理錯誤，不崩潰
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(test_data)
    
    def test_feature_creation_with_invalid_config(self):
        """測試使用無效配置創建特徵"""
        # 創建有問題的配置
        config = ConfigManager()
        config.feature_config.time_bins = []  # 空的時間分組
        
        engineer = FeatureEngineer(config)
        test_data = pd.DataFrame({
            'TransactionDT': [86400, 86500, 86600],
            'TransactionAmt': [100, 200, 300]
        })
        
        # 應該能處理配置錯誤
        with pytest.raises((FeatureCreationError, ValueError, IndexError)):
            engineer.create_time_features(test_data)
    
    def test_memory_optimization_with_large_features(self):
        """測試大量特徵的內存優化"""
        # 創建包含大量特徵的數據
        large_feature_data = pd.DataFrame(
            np.random.randn(100, 200)  # 100行, 200列
        )
        large_feature_data.columns = [f'feature_{i}' for i in range(200)]
        large_feature_data['TransactionID'] = range(100)
        large_feature_data['isFraud'] = np.random.choice([0, 1], 100)
        
        engineer = FeatureEngineer()
        
        # 測試最終優化步驟
        result = engineer._final_optimization(large_feature_data, 'isFraud')
        
        # 檢查結果
        assert isinstance(result, pd.DataFrame)
        # 常數特徵應該被移除
        constant_features = [col for col in result.columns 
                           if result[col].nunique() <= 1 and col != 'isFraud']
        assert len(constant_features) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])