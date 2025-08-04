"""
數據處理模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
sys.path.append('../src')

from src.data_processing import DataProcessor, load_and_preprocess_data
from src.exceptions import DataLoadError, DataValidationError

class TestDataProcessor:
    """測試數據處理器類"""
    
    def setup_method(self):
        """測試設置"""
        self.processor = DataProcessor()
        
        # 創建測試數據
        self.sample_transaction_data = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'isFraud': [0, 1, 0, 1, 0],
            'TransactionDT': [86400, 86500, 86600, 86700, 86800],
            'TransactionAmt': [100.0, 50.0, 200.0, 75.0, 150.0],
            'ProductCD': ['W', 'C', 'W', 'H', 'W'],
            'card1': [13553, 4590, 2755, 8845, 13553],
            'C1': [1.0, np.nan, 2.0, np.nan, 1.0],
            'C2': [np.nan, 1.0, np.nan, 2.0, np.nan]
        })
        
        self.sample_identity_data = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'id_01': [0.0, 1.0, 0.0],
            'id_02': [142617.0, np.nan, 142617.0],
            'DeviceType': ['desktop', 'mobile', 'desktop']
        })
    
    def create_temp_csv(self, data, suffix='.csv'):
        """創建臨時CSV文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    def test_initialization(self):
        """測試初始化"""
        assert self.processor.missing_threshold == 0.9
        assert isinstance(self.processor.categorical_features, list)
        assert isinstance(self.processor.numerical_features, list)
    
    def test_load_data_success(self):
        """測試成功載入數據"""
        # 創建臨時CSV文件
        transaction_path = self.create_temp_csv(self.sample_transaction_data)
        identity_path = self.create_temp_csv(self.sample_identity_data)
        
        try:
            result = self.processor.load_data(transaction_path, identity_path)
            
            # 驗證結果
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 5  # 應該有5行（基於transaction數據）
            assert 'TransactionID' in result.columns
            assert 'isFraud' in result.columns
            assert 'id_01' in result.columns  # 來自identity數據
            
            # 檢查合併結果
            assert result.loc[0, 'id_01'] == 0.0  # TransactionID=1的記錄
            assert pd.isna(result.loc[3, 'id_01'])  # TransactionID=4沒有identity數據
            
        finally:
            # 清理臨時文件
            os.unlink(transaction_path)
            os.unlink(identity_path)
    
    def test_load_data_file_not_found(self):
        """測試文件不存在的情況"""
        with pytest.raises(FileNotFoundError):
            self.processor.load_data('nonexistent_transaction.csv', 'nonexistent_identity.csv')
    
    def test_analyze_missing_values(self):
        """測試缺失值分析"""
        result = self.processor.analyze_missing_values(self.sample_transaction_data)
        
        assert isinstance(result, pd.DataFrame)
        assert '缺失數量' in result.columns
        assert '缺失百分比' in result.columns
        
        # C1和C2都有2個缺失值
        assert result.loc['C1', '缺失數量'] == 2
        assert result.loc['C2', '缺失數量'] == 2
        assert result.loc['C1', '缺失百分比'] == 40.0  # 2/5 * 100
    
    def test_remove_high_missing_features(self):
        """測試移除高缺失率特徵"""
        # 創建有高缺失率的測試數據
        high_missing_data = self.sample_transaction_data.copy()
        high_missing_data['high_missing_col'] = [np.nan] * 5  # 100%缺失
        
        # 設置較低的閾值來測試
        self.processor.missing_threshold = 0.5
        result = self.processor.remove_high_missing_features(high_missing_data)
        
        # 高缺失率的列應該被移除
        assert 'high_missing_col' not in result.columns
        # C1和C2有40%缺失，應該保留
        assert 'C1' in result.columns
        assert 'C2' in result.columns
    
    def test_identify_feature_types(self):
        """測試特徵類型識別"""
        numerical_features, categorical_features = self.processor.identify_feature_types(
            self.sample_transaction_data
        )
        
        # 數值型特徵（排除isFraud和TransactionID）
        expected_numerical = ['TransactionDT', 'TransactionAmt', 'card1', 'C1', 'C2']
        assert set(numerical_features) == set(expected_numerical)
        
        # 類別型特徵
        expected_categorical = ['ProductCD']
        assert set(categorical_features) == set(expected_categorical)
        
        # 檢查實例變量是否正確設置
        assert set(self.processor.numerical_features) == set(expected_numerical)
        assert set(self.processor.categorical_features) == set(expected_categorical)
    
    def test_handle_outliers_iqr_method(self):
        """測試IQR方法處理異常值"""
        # 創建包含異常值的數據
        data_with_outliers = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'outlier_col': [1, 2, 3, 4, 1000]  # 1000是異常值
        })
        
        result = self.processor.handle_outliers(
            data_with_outliers, 
            ['outlier_col'], 
            method='iqr'
        )
        
        # 異常值應該被截斷
        assert result['outlier_col'].max() < 1000
        assert result['normal_col'].equals(data_with_outliers['normal_col'])  # 正常列不變
    
    def test_basic_preprocessing_pipeline(self):
        """測試基本預處理流水線"""
        # 創建包含各種問題的測試數據
        problematic_data = self.sample_transaction_data.copy()
        problematic_data['high_missing'] = [np.nan] * 5  # 100%缺失
        problematic_data['outlier_col'] = [1, 2, 3, 4, 1000]  # 包含異常值
        
        # 設置低閾值以測試高缺失特徵移除
        self.processor.missing_threshold = 0.8
        
        result = self.processor.basic_preprocessing(problematic_data)
        
        # 高缺失特徵應該被移除
        assert 'high_missing' not in result.columns
        # 原始特徵應該保留
        assert 'TransactionID' in result.columns
        assert 'isFraud' in result.columns
        # 異常值應該被處理
        assert result['outlier_col'].max() < 1000
    
    def test_basic_preprocessing_empty_dataframe(self):
        """測試空數據框的基本預處理"""
        empty_df = pd.DataFrame()
        result = self.processor.basic_preprocessing(empty_df)
        assert len(result) == 0
        assert len(result.columns) == 0

class TestDataProcessorIntegration:
    """測試數據處理器集成功能"""
    
    def test_load_and_preprocess_data_function(self):
        """測試便捷函數"""
        # 創建測試數據文件
        transaction_data = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'isFraud': [0, 1, 0],
            'TransactionAmt': [100.0, 50.0, 200.0],
            'high_missing_col': [np.nan, np.nan, np.nan]  # 100%缺失
        })
        
        identity_data = pd.DataFrame({
            'TransactionID': [1, 2],
            'id_01': [0.0, 1.0]
        })
        
        transaction_path = None
        identity_path = None
        
        try:
            # 創建臨時文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                transaction_data.to_csv(f.name, index=False)
                transaction_path = f.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                identity_data.to_csv(f.name, index=False)
                identity_path = f.name
            
            # 測試便捷函數
            result = load_and_preprocess_data(transaction_path, identity_path)
            
            # 驗證結果
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'TransactionID' in result.columns
            assert 'id_01' in result.columns
            # 高缺失列應該被移除
            assert 'high_missing_col' not in result.columns
            
        finally:
            # 清理臨時文件
            if transaction_path and os.path.exists(transaction_path):
                os.unlink(transaction_path)
            if identity_path and os.path.exists(identity_path):
                os.unlink(identity_path)
    
    def test_processor_with_real_world_scenarios(self):
        """測試真實世界場景"""
        processor = DataProcessor()
        
        # 模擬真實數據的特徵
        realistic_data = pd.DataFrame({
            'TransactionID': range(1, 101),
            'isFraud': np.random.choice([0, 1], 100, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, 100),
            'TransactionAmt': np.random.lognormal(3, 1, 100),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], 100),
            'card1': np.random.randint(1000, 20000, 100),
            # 模擬不同缺失率的特徵
            'low_missing': np.where(np.random.random(100) < 0.1, np.nan, np.random.randn(100)),
            'medium_missing': np.where(np.random.random(100) < 0.5, np.nan, np.random.randn(100)),
            'high_missing': np.where(np.random.random(100) < 0.95, np.nan, np.random.randn(100)),
            # 包含異常值的特徵
            'outlier_feature': np.concatenate([np.random.randn(95), [100, -100, 200, -200, 300]])
        })
        
        # 執行完整的預處理流水線
        result = processor.basic_preprocessing(realistic_data)
        
        # 驗證結果
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100  # 行數不變
        assert 'high_missing' not in result.columns  # 高缺失特徵被移除
        assert 'low_missing' in result.columns  # 低缺失特徵保留
        
        # 檢查異常值處理
        outlier_col = 'outlier_feature'
        if outlier_col in result.columns:
            # 極端異常值應該被截斷
            assert result[outlier_col].max() < 300
            assert result[outlier_col].min() > -300

class TestDataProcessorEdgeCases:
    """測試數據處理器邊界情況"""
    
    def test_empty_dataframe_handling(self):
        """測試空數據框處理"""
        processor = DataProcessor()
        empty_df = pd.DataFrame()
        
        # 各個方法都應該能處理空數據框
        missing_analysis = processor.analyze_missing_values(empty_df)
        assert len(missing_analysis) == 0
        
        cleaned_df = processor.remove_high_missing_features(empty_df)
        assert len(cleaned_df) == 0
        
        num_features, cat_features = processor.identify_feature_types(empty_df)
        assert len(num_features) == 0
        assert len(cat_features) == 0
    
    def test_single_column_dataframe(self):
        """測試單列數據框"""
        processor = DataProcessor()
        single_col_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        
        result = processor.basic_preprocessing(single_col_df)
        assert len(result.columns) == 1
        assert 'single_col' in result.columns
    
    def test_all_missing_values(self):
        """測試所有值都缺失的情況"""
        processor = DataProcessor()
        all_missing_df = pd.DataFrame({
            'col1': [np.nan] * 5,
            'col2': [np.nan] * 5,
            'col3': [np.nan] * 5
        })
        
        result = processor.basic_preprocessing(all_missing_df)
        # 所有列都應該被移除（假設missing_threshold < 1.0）
        assert len(result.columns) == 0
    
    def test_no_missing_values(self):
        """測試沒有缺失值的情況"""
        processor = DataProcessor()
        no_missing_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        missing_analysis = processor.analyze_missing_values(no_missing_df)
        assert len(missing_analysis) == 0  # 沒有缺失值的列不會出現在結果中

if __name__ == "__main__":
    pytest.main([__file__])