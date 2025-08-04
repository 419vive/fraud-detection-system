"""
數據驗證模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import sys
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

sys.path.append('../src')

from src.data_validation import (
    DataValidator, validate_fraud_detection_data
)
from src.exceptions import DataValidationError
from src.config import ConfigManager


class TestDataValidator:
    """測試數據驗證器類"""
    
    def setup_method(self):
        """測試設置"""
        self.validator = DataValidator()
        
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
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # 添加一些缺失值和異常值
        missing_indices = np.random.choice(100, 20, replace=False)
        self.sample_data.loc[missing_indices[:10], 'C1'] = np.nan
        self.sample_data.loc[missing_indices[10:], 'card2'] = np.nan
        
        # 添加異常值
        self.sample_data.loc[0, 'TransactionAmt'] = 1000000  # 極大值
        self.sample_data.loc[1, 'C1'] = -1000  # 極小值
    
    def test_initialization(self):
        """測試初始化"""
        assert isinstance(self.validator.validation_rules, dict)
        assert isinstance(self.validator.validation_results, dict)
        assert isinstance(self.validator.data_profile, dict)
        assert self.validator.config is not None
    
    def test_validate_data_structure_success(self):
        """測試成功的數據結構驗證"""
        expected_columns = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt']
        result = self.validator.validate_data_structure(self.sample_data, expected_columns)
        
        # 檢查返回結果結構
        assert 'timestamp' in result
        assert 'total_rows' in result
        assert 'total_columns' in result
        assert 'column_names' in result
        assert 'data_types' in result
        assert 'memory_usage' in result
        assert 'structure_valid' in result
        
        # 檢查值
        assert result['total_rows'] == 100
        assert result['total_columns'] == len(self.sample_data.columns)
        assert result['structure_valid'] == True  # 所有預期列都存在
        assert len(result['missing_columns']) == 0
    
    def test_validate_data_structure_missing_columns(self):
        """測試缺少預期列的情況"""
        expected_columns = ['TransactionID', 'isFraud', 'NonExistentColumn']
        result = self.validator.validate_data_structure(self.sample_data, expected_columns)
        
        # 檢查結果
        assert result['structure_valid'] == False
        assert 'NonExistentColumn' in result['missing_columns']
        assert len(result['missing_columns']) == 1
    
    def test_validate_data_structure_extra_columns(self):
        """測試有額外列的情況"""
        expected_columns = ['TransactionID', 'isFraud']  # 只期望這兩列
        result = self.validator.validate_data_structure(self.sample_data, expected_columns)
        
        # 檢查結果
        assert result['structure_valid'] == True  # 不缺少預期列
        assert len(result['extra_columns']) > 0  # 有額外列
        assert 'TransactionDT' in result['extra_columns']
    
    def test_validate_data_quality_success(self):
        """測試數據品質驗證"""
        result = self.validator.validate_data_quality(self.sample_data)
        
        # 檢查返回結果結構
        assert 'timestamp' in result
        assert 'missing_values' in result
        assert 'duplicate_rows' in result
        assert 'data_completeness' in result
        assert 'column_quality' in result
        assert 'quality_score' in result
        
        # 檢查missing_values結構
        assert 'counts' in result['missing_values']
        assert 'percentages' in result['missing_values']
        assert 'high_missing_features' in result['missing_values']
        
        # 檢查數值合理性
        assert 0 <= result['quality_score'] <= 100
        assert 0 <= result['data_completeness'] <= 100
        assert result['duplicate_rows'] >= 0
    
    def test_assess_column_quality_numeric(self):
        """測試數值型列的品質評估"""
        numeric_series = self.sample_data['TransactionAmt']
        quality = self.validator._assess_column_quality(numeric_series)
        
        # 檢查返回結果
        assert 'missing_count' in quality
        assert 'missing_percentage' in quality
        assert 'unique_count' in quality
        assert 'unique_percentage' in quality
        assert 'data_type' in quality
        assert 'quality_issues' in quality
        
        # 檢查數值型特定指標
        assert 'min_value' in quality
        assert 'max_value' in quality
        assert 'mean_value' in quality
        assert 'outlier_count' in quality
    
    def test_assess_column_quality_categorical(self):
        """測試類別型列的品質評估"""
        categorical_series = self.sample_data['ProductCD']
        quality = self.validator._assess_column_quality(categorical_series)
        
        # 檢查基本指標
        assert 'missing_count' in quality
        assert 'unique_count' in quality
        assert 'data_type' in quality
        
        # 類別型不應該有數值型特定指標
        assert 'min_value' not in quality
        assert 'max_value' not in quality
    
    def test_assess_numeric_column_outliers(self):
        """測試數值型列異常值檢測"""
        # 創建包含明顯異常值的數據
        data_with_outliers = pd.Series([1, 2, 3, 4, 5, 1000])  # 1000是明顯異常值
        quality = self.validator._assess_numeric_column(data_with_outliers)
        
        assert 'outlier_count' in quality
        assert 'outlier_percentage' in quality
        assert quality['outlier_count'] > 0
        assert quality['outlier_percentage'] > 0
    
    def test_validate_business_rules_success(self):
        """測試業務規則驗證成功"""
        result = self.validator.validate_business_rules(self.sample_data)
        
        # 檢查返回結果結構
        assert 'timestamp' in result
        assert 'rules_checked' in result
        assert 'rules_passed' in result
        assert 'rules_failed' in result
        assert 'rule_results' in result
        assert 'overall_valid' in result
        
        # 檢查規則結果
        assert result['rules_checked'] > 0
        assert isinstance(result['rule_results'], dict)
    
    def test_validate_business_rules_custom_rules(self):
        """測試自定義業務規則"""
        custom_rules = {
            'amount_positive': {
                'type': 'numeric_range',
                'column': 'TransactionAmt',
                'min_value': 0,
                'description': '交易金額必須為正數'
            },
            'fraud_binary': {
                'type': 'categorical_values',
                'column': 'isFraud',
                'allowed_values': [0, 1],
                'description': '詐騙標籤必須為0或1'
            }
        }
        
        result = self.validator.validate_business_rules(self.sample_data, custom_rules)
        
        # 檢查自定義規則是否被執行
        assert result['rules_checked'] == 2
        assert 'amount_positive' in result['rule_results']
        assert 'fraud_binary' in result['rule_results']
    
    def test_apply_business_rule_numeric_range(self):
        """測試數值範圍業務規則"""
        rule_config = {
            'type': 'numeric_range',
            'column': 'TransactionAmt',
            'min_value': 0,
            'max_value': 10000,
            'description': '交易金額範圍檢查'
        }
        
        result = self.validator._apply_business_rule(self.sample_data, 'test_rule', rule_config)
        
        # 檢查結果
        assert 'passed' in result
        assert 'violation_count' in result
        assert 'violation_percentage' in result
        assert 'description' in result
        
        # 應該檢測到極大值異常
        assert result['violation_count'] > 0
    
    def test_apply_business_rule_uniqueness(self):
        """測試唯一性業務規則"""
        # 創建有重複值的數據
        data_with_duplicates = self.sample_data.copy()
        data_with_duplicates.loc[50, 'TransactionID'] = 1  # 創建重複ID
        
        rule_config = {
            'type': 'uniqueness',
            'column': 'TransactionID',
            'description': 'ID必須唯一'
        }
        
        result = self.validator._apply_business_rule(data_with_duplicates, 'unique_test', rule_config)
        
        # 應該檢測到重複值
        assert result['violation_count'] > 0
        assert not result['passed']
    
    def test_apply_business_rule_categorical_values(self):
        """測試類別值業務規則"""
        rule_config = {
            'type': 'categorical_values',
            'column': 'isFraud',
            'allowed_values': [0, 1],
            'description': '詐騙標籤檢查'
        }
        
        result = self.validator._apply_business_rule(self.sample_data, 'categorical_test', rule_config)
        
        # isFraud只包含0和1，應該通過驗證
        assert result['passed'] == True
        assert result['violation_count'] == 0
    
    def test_apply_business_rule_not_null(self):
        """測試非空業務規則"""
        rule_config = {
            'type': 'not_null',
            'column': 'C1',  # 這一列有缺失值
            'description': '不能為空'
        }
        
        result = self.validator._apply_business_rule(self.sample_data, 'not_null_test', rule_config)
        
        # C1有缺失值，應該失敗
        assert not result['passed']
        assert result['violation_count'] > 0
    
    def test_apply_business_rule_missing_column(self):
        """測試業務規則應用於不存在的列"""
        rule_config = {
            'type': 'not_null',
            'column': 'NonExistentColumn',
            'description': '測試不存在的列'
        }
        
        result = self.validator._apply_business_rule(self.sample_data, 'missing_col_test', rule_config)
        
        # 應該返回失敗並說明原因
        assert not result['passed']
        assert 'error' in result
        assert 'NonExistentColumn' in result['error']
    
    def test_validate_data_distribution(self):
        """測試數據分佈驗證"""
        result = self.validator.validate_data_distribution(self.sample_data)
        
        # 檢查返回結果結構
        assert 'timestamp' in result
        assert 'current_stats' in result
        assert 'distribution_shifts' in result
        assert 'overall_shift_detected' in result
        
        # 檢查當前統計信息
        assert isinstance(result['current_stats'], dict)
        assert len(result['current_stats']) > 0
    
    def test_validate_data_distribution_with_reference(self):
        """測試與參考數據的分佈對比"""
        # 計算當前數據統計
        current_stats = self.validator._calculate_distribution_stats(self.sample_data)
        
        # 創建略有不同的參考統計
        reference_stats = current_stats.copy()
        if 'TransactionAmt' in reference_stats:
            reference_stats['TransactionAmt']['mean'] *= 1.5  # 增加50%
        
        result = self.validator.validate_data_distribution(self.sample_data, reference_stats)
        
        # 檢查是否檢測到分佈偏移
        assert 'distribution_shifts' in result
        if 'TransactionAmt' in result['distribution_shifts']:
            shift_info = result['distribution_shifts']['TransactionAmt']
            assert 'significant_shift' in shift_info
    
    def test_calculate_distribution_stats_numeric(self):
        """測試數值型特徵分佈統計計算"""
        stats = self.validator._calculate_distribution_stats(self.sample_data)
        
        # 檢查數值型特徵統計
        numeric_feature = 'TransactionAmt'
        if numeric_feature in stats:
            feature_stats = stats[numeric_feature]
            expected_keys = ['mean', 'std', 'min', 'max', 'q25', 'q50', 'q75', 'skewness', 'kurtosis']
            for key in expected_keys:
                assert key in feature_stats
    
    def test_calculate_distribution_stats_categorical(self):
        """測試類別型特徵分佈統計計算"""
        stats = self.validator._calculate_distribution_stats(self.sample_data)
        
        # 檢查類別型特徵統計
        categorical_feature = 'ProductCD'
        if categorical_feature in stats:
            feature_stats = stats[categorical_feature]
            expected_keys = ['unique_count', 'top_values', 'entropy']
            for key in expected_keys:
                assert key in feature_stats
    
    def test_calculate_entropy(self):
        """測試熵值計算"""
        # 均勻分佈（最大熵）
        uniform_series = pd.Series(['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25)
        uniform_counts = uniform_series.value_counts()
        uniform_entropy = self.validator._calculate_entropy(uniform_counts)
        
        # 極不均勻分佈（最小熵）
        skewed_series = pd.Series(['A'] * 99 + ['B'] * 1)
        skewed_counts = skewed_series.value_counts()
        skewed_entropy = self.validator._calculate_entropy(skewed_counts)
        
        # 均勻分佈的熵應該大於不均勻分佈
        assert uniform_entropy > skewed_entropy
        assert uniform_entropy > 0
        assert skewed_entropy >= 0
    
    def test_detect_distribution_shift(self):
        """測試分佈偏移檢測"""
        current_stats = {'mean': 100, 'std': 10, 'q50': 95}
        reference_stats = {'mean': 90, 'std': 12, 'q50': 88}  # 有明顯差異
        
        result = self.validator._detect_distribution_shift(current_stats, reference_stats)
        
        # 檢查結果結構
        assert 'significant_shift' in result
        assert 'shift_metrics' in result
        assert 'shift_threshold' in result
        
        # 應該檢測到顯著偏移
        assert result['significant_shift'] == True
    
    def test_generate_validation_report(self):
        """測試生成驗證報告"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name
        
        try:
            report = self.validator.generate_validation_report(self.sample_data, temp_path)
            
            # 檢查報告內容
            assert isinstance(report, str)
            assert len(report) > 0
            assert '數據驗證報告' in report
            assert '數據結構驗證' in report
            assert '數據品質驗證' in report
            assert '業務規則驗證' in report
            
            # 檢查文件是否被創建
            assert os.path.exists(temp_path)
            
            # 讀取文件內容
            with open(temp_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            assert len(file_content) > 0
            assert file_content == report
            
        finally:
            # 清理臨時文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_missing_value_patterns(self, mock_show):
        """測試缺失值模式繪圖"""
        # 創建有更多缺失值的數據用於繪圖
        plot_data = self.sample_data.copy()
        for i in range(10):
            plot_data[f'missing_col_{i}'] = np.nan
        
        # 測試繪圖函數（不實際顯示圖表）
        self.validator.plot_missing_value_patterns(plot_data, top_n=10)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    @patch('matplotlib.pyplot.show')
    def test_plot_correlation_analysis(self, mock_show):
        """測試相關性分析繪圖"""
        self.validator.plot_correlation_analysis(self.sample_data, target_col='isFraud')
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    @patch('matplotlib.pyplot.show')
    def test_plot_distribution_analysis(self, mock_show):
        """測試分佈分析繪圖"""
        self.validator.plot_distribution_analysis(self.sample_data, target_col='isFraud', max_features=5)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    @patch('matplotlib.pyplot.show')
    def test_plot_outlier_analysis(self, mock_show):
        """測試異常值分析繪圖"""
        self.validator.plot_outlier_analysis(self.sample_data, max_features=5)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    def test_create_interactive_data_profile(self):
        """測試創建交互式數據概覽"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            profile = self.validator.create_interactive_data_profile(self.sample_data, temp_path)
            
            # 檢查概覽結構
            assert isinstance(profile, dict)
            assert 'generated_at' in profile
            assert 'basic_statistics' in profile
            assert 'data_types' in profile
            assert 'missing_values' in profile
            assert 'numeric_features' in profile
            assert 'categorical_features' in profile
            
            # 檢查基本統計
            basic_stats = profile['basic_statistics']
            assert basic_stats['total_rows'] == 100
            assert basic_stats['total_columns'] == len(self.sample_data.columns)
            
            # 檢查文件是否被創建
            assert os.path.exists(temp_path)
            
            # 讀取並驗證JSON文件
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_profile = json.load(f)
            assert saved_profile == profile
            
        finally:
            # 清理臨時文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_comprehensive_data_validation_report(self, mock_show, mock_savefig):
        """測試綜合數據驗證報告"""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_files = self.validator.comprehensive_data_validation_report(
                self.sample_data, 
                target_col='isFraud',
                output_dir=temp_dir
            )
            
            # 檢查返回的文件路徑
            expected_keys = [
                'text_report', 'dashboard', 'data_profile',
                'missing_analysis', 'correlation_analysis', 
                'distribution_analysis', 'outlier_analysis'
            ]
            for key in expected_keys:
                assert key in report_files
                # 檢查某些文件是否實際存在
                if key in ['text_report', 'data_profile']:
                    assert os.path.exists(report_files[key])


class TestDataValidatorEdgeCases:
    """測試數據驗證器邊界情況"""
    
    def setup_method(self):
        """測試設置"""
        self.validator = DataValidator()
    
    def test_empty_dataframe_validation(self):
        """測試空數據框驗證"""
        empty_df = pd.DataFrame()
        
        # 結構驗證
        structure_result = self.validator.validate_data_structure(empty_df)
        assert structure_result['total_rows'] == 0
        assert structure_result['total_columns'] == 0
        
        # 品質驗證
        quality_result = self.validator.validate_data_quality(empty_df)
        assert quality_result['data_completeness'] == 0 or np.isnan(quality_result['data_completeness'])
        assert quality_result['duplicate_rows'] == 0
    
    def test_single_row_dataframe(self):
        """測試單行數據框"""
        single_row_df = pd.DataFrame({
            'col1': [1],
            'col2': ['A'],
            'col3': [np.nan]
        })
        
        result = self.validator.validate_data_quality(single_row_df)
        
        # 檢查結果合理性
        assert result['total_rows'] == 1
        assert result['data_completeness'] < 100  # 由於有缺失值
    
    def test_all_missing_values(self):
        """測試所有值都缺失的數據框"""
        all_missing_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        result = self.validator.validate_data_quality(all_missing_df)
        
        # 數據完整性應該為0
        assert result['data_completeness'] == 0
        # 品質分數應該很低
        assert result['quality_score'] <= 20
    
    def test_constant_features(self):
        """測試常數特徵"""
        constant_df = pd.DataFrame({
            'constant_col': [1, 1, 1, 1, 1],
            'normal_col': [1, 2, 3, 4, 5]
        })
        
        quality_result = self.validator.validate_data_quality(constant_df)
        
        # 檢查常數列是否被標識
        constant_col_quality = quality_result['column_quality']['constant_col']
        assert '常數欄位' in constant_col_quality['quality_issues']
    
    def test_high_cardinality_features(self):
        """測試高基數特徵"""
        # 每個值都不同（可能是ID欄位）
        high_cardinality_df = pd.DataFrame({
            'id_col': [f'id_{i}' for i in range(100)],
            'normal_col': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        quality_result = self.validator.validate_data_quality(high_cardinality_df)
        
        # 檢查高基數列是否被標識
        id_col_quality = quality_result['column_quality']['id_col']
        assert '可能為ID欄位' in id_col_quality['quality_issues']
    
    def test_extreme_outliers(self):
        """測試極端異常值"""
        extreme_outlier_df = pd.DataFrame({
            'normal_data': [1, 2, 3, 4, 5],
            'with_outliers': [1, 2, 3, 4, 1e10]  # 極大值
        })
        
        quality_result = self.validator.validate_data_quality(extreme_outlier_df)
        
        # 檢查異常值是否被檢測
        outlier_col_quality = quality_result['column_quality']['with_outliers']
        assert outlier_col_quality['outlier_count'] > 0
    
    def test_mixed_data_types(self):
        """測試混合數據類型"""
        mixed_df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['A', 'B', 'C'],
            'bool_col': [True, False, True],
            'datetime_col': pd.date_range('2023-01-01', periods=3)
        })
        
        structure_result = self.validator.validate_data_structure(mixed_df)
        
        # 檢查數據類型是否被正確識別
        assert len(structure_result['data_types']) == 5
        assert 'int_col' in structure_result['data_types']
        assert 'str_col' in structure_result['data_types']


class TestDataValidatorBusinessRules:
    """測試數據驗證器業務規則"""
    
    def setup_method(self):
        """測試設置"""
        self.validator = DataValidator()
    
    def test_default_fraud_detection_rules(self):
        """測試默認詐騙檢測業務規則"""
        rules = self.validator._get_default_fraud_detection_rules()
        
        # 檢查必要的規則是否存在
        expected_rules = [
            'transaction_amount_positive',
            'transaction_id_unique', 
            'fraud_label_binary',
            'transaction_dt_reasonable'
        ]
        
        for rule in expected_rules:
            assert rule in rules
            assert 'type' in rules[rule]
            assert 'column' in rules[rule]
            assert 'description' in rules[rule]
    
    def test_business_rules_with_violations(self):
        """測試包含違規的業務規則驗證"""
        # 創建包含違規的數據
        violating_data = pd.DataFrame({
            'TransactionID': [1, 2, 2],  # 重複ID
            'isFraud': [0, 1, 2],  # 無效的詐騙標籤
            'TransactionDT': [86400, -100, 86500],  # 負數時間戳
            'TransactionAmt': [100, -50, 200]  # 負數金額
        })
        
        result = self.validator.validate_business_rules(violating_data)
        
        # 應該檢測到多個違規
        assert not result['overall_valid']
        assert result['rules_failed'] > 0
        
        # 檢查具體違規
        if 'transaction_amount_positive' in result['rule_results']:
            amount_rule = result['rule_results']['transaction_amount_positive']
            assert not amount_rule['passed']
            assert amount_rule['violation_count'] > 0
    
    def test_custom_business_rules_complex(self):
        """測試複雜的自定義業務規則"""
        # 創建測試數據
        test_data = pd.DataFrame({
            'user_age': [25, 30, 15, 35, 200],  # 包含不合理年齡
            'purchase_amount': [100, 500, 10000, 200, 50],
            'user_type': ['premium', 'basic', 'invalid', 'premium', 'basic']
        })
        
        # 定義複雜業務規則
        complex_rules = {
            'age_reasonable': {
                'type': 'numeric_range',
                'column': 'user_age',
                'min_value': 18,
                'max_value': 120,
                'description': '用戶年齡必須在合理範圍內'
            },
            'high_value_transaction': {
                'type': 'numeric_range',
                'column': 'purchase_amount',
                'max_value': 5000,
                'description': '單筆交易金額不能過高'
            },
            'valid_user_type': {
                'type': 'categorical_values',
                'column': 'user_type',
                'allowed_values': ['premium', 'basic', 'guest'],
                'description': '用戶類型必須有效'
            }
        }
        
        result = self.validator.validate_business_rules(test_data, complex_rules)
        
        # 檢查規則執行結果
        assert result['rules_checked'] == 3
        assert result['rules_failed'] > 0  # 應該有違規
        
        # 檢查具體規則結果
        age_result = result['rule_results']['age_reasonable']
        assert not age_result['passed']  # 年齡15和200不合理
        
        amount_result = result['rule_results']['high_value_transaction']
        assert not amount_result['passed']  # 10000超過限額
        
        type_result = result['rule_results']['valid_user_type']
        assert not type_result['passed']  # 'invalid'不是有效類型


class TestDataValidatorUtilityFunctions:
    """測試數據驗證器工具函數"""
    
    def test_validate_fraud_detection_data_function(self):
        """測試便捷驗證函數"""
        # 創建符合詐騙檢測要求的數據
        fraud_data = pd.DataFrame({
            'TransactionID': range(1, 51),
            'isFraud': np.random.choice([0, 1], 50, p=[0.9, 0.1]),
            'TransactionDT': np.random.randint(86400, 86400*7, 50),
            'TransactionAmt': np.random.lognormal(3, 1, 50),
        })
        
        results = validate_fraud_detection_data(fraud_data)
        
        # 檢查返回結果結構
        assert 'structure' in results
        assert 'quality' in results
        assert 'business_rules' in results
        
        # 檢查各項驗證結果
        assert 'structure_valid' in results['structure']
        assert 'quality_score' in results['quality']
        assert 'overall_valid' in results['business_rules']
    
    def test_validate_fraud_detection_data_missing_columns(self):
        """測試缺少必要列的詐騙檢測數據"""
        incomplete_data = pd.DataFrame({
            'TransactionID': [1, 2, 3],
            'TransactionAmt': [100, 200, 300]
            # 缺少 isFraud 和 TransactionDT
        })
        
        results = validate_fraud_detection_data(incomplete_data)
        
        # 結構驗證應該失敗
        assert not results['structure']['structure_valid']
        assert 'isFraud' in results['structure']['missing_columns']
        assert 'TransactionDT' in results['structure']['missing_columns']


class TestDataValidatorPerformance:
    """測試數據驗證器性能"""
    
    def test_large_dataset_validation(self):
        """測試大數據集驗證性能"""
        # 創建較大的測試數據集
        large_data = pd.DataFrame({
            'id': range(10000),
            'feature1': np.random.randn(10000),
            'feature2': np.random.choice(['A', 'B', 'C'], 10000),
            'feature3': np.random.lognormal(0, 1, 10000),
            'target': np.random.choice([0, 1], 10000, p=[0.9, 0.1])
        })
        
        # 添加一些缺失值
        missing_indices = np.random.choice(10000, 1000, replace=False)
        large_data.loc[missing_indices, 'feature1'] = np.nan
        
        validator = DataValidator()
        
        # 測試各種驗證是否能在合理時間內完成
        import time
        
        start_time = time.time()
        structure_result = validator.validate_data_structure(large_data)
        structure_time = time.time() - start_time
        
        start_time = time.time()
        quality_result = validator.validate_data_quality(large_data)
        quality_time = time.time() - start_time
        
        start_time = time.time()
        business_result = validator.validate_business_rules(large_data)
        business_time = time.time() - start_time
        
        # 檢查結果正確性
        assert structure_result['total_rows'] == 10000
        assert quality_result['quality_score'] >= 0
        assert business_result['rules_checked'] > 0
        
        # 簡單的性能檢查（不應該過慢）
        assert structure_time < 10  # 結構驗證應該很快
        assert quality_time < 30   # 品質驗證可能稍慢
        assert business_time < 20  # 業務規則驗證應該合理
    
    def test_memory_usage_with_large_features(self):
        """測試大量特徵的內存使用"""
        # 創建包含大量特徵的數據
        many_features_data = pd.DataFrame(
            np.random.randn(1000, 100)  # 1000行, 100列
        )
        many_features_data.columns = [f'feature_{i}' for i in range(100)]
        
        validator = DataValidator()
        
        # 測試驗證過程不會導致內存問題
        result = validator.validate_data_quality(many_features_data)
        
        # 檢查結果合理性
        assert result['total_columns'] == 100
        assert len(result['column_quality']) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])