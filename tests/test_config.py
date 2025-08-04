"""
配置管理模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open
import sys
sys.path.append('../src')

from src.config import ConfigManager, DataConfig, ModelConfig, FeatureEngineeringConfig
from src.exceptions import ConfigLoadError, InvalidConfigError

class TestDataConfig:
    """測試數據配置類"""
    
    def test_default_values(self):
        """測試默認值設置"""
        config = DataConfig()
        assert config.missing_threshold == 0.9
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.train_transaction_path == "train_transaction.csv"
    
    def test_custom_values(self):
        """測試自定義值設置"""
        config = DataConfig(
            missing_threshold=0.8,
            test_size=0.3,
            random_state=123
        )
        assert config.missing_threshold == 0.8
        assert config.test_size == 0.3
        assert config.random_state == 123

class TestModelConfig:
    """測試模型配置類"""
    
    def test_default_model_params(self):
        """測試默認模型參數"""
        config = ModelConfig()
        assert config.random_state == 42
        assert config.rf_n_estimators == 100
        assert config.xgb_learning_rate == 0.1
        assert config.lgb_class_weight == "balanced"
    
    def test_custom_model_params(self):
        """測試自定義模型參數"""
        config = ModelConfig(
            rf_n_estimators=200,
            xgb_learning_rate=0.05
        )
        assert config.rf_n_estimators == 200
        assert config.xgb_learning_rate == 0.05

class TestFeatureEngineeringConfig:
    """測試特徵工程配置類"""
    
    def test_default_post_init(self):
        """測試 __post_init__ 默認值設置"""
        config = FeatureEngineeringConfig()
        assert config.time_bins == [0, 6, 12, 18, 24]
        assert config.time_labels == ['夜晚', '早晨', '下午', '晚上']
        assert config.amount_bins == [0, 50, 100, 500, 1000, float('inf')]
        assert 'mean' in config.agg_functions
    
    def test_custom_values_override_post_init(self):
        """測試自定義值覆蓋 __post_init__"""
        custom_bins = [0, 8, 16, 24]
        config = FeatureEngineeringConfig(time_bins=custom_bins)
        assert config.time_bins == custom_bins

class TestConfigManager:
    """測試配置管理器"""
    
    def setup_method(self):
        """測試設置"""
        self.config_manager = ConfigManager()
    
    def test_default_initialization(self):
        """測試默認初始化"""
        assert isinstance(self.config_manager.data_config, DataConfig)
        assert isinstance(self.config_manager.model_config, ModelConfig)
        assert isinstance(self.config_manager.feature_config, FeatureEngineeringConfig)
    
    def test_get_model_params_logistic(self):
        """測試獲取邏輯回歸參數"""
        params = self.config_manager.get_model_params('logistic')
        expected_keys = ['random_state', 'max_iter', 'class_weight']
        assert all(key in params for key in expected_keys)
        assert params['random_state'] == 42
        assert params['max_iter'] == 1000
        assert params['class_weight'] == "balanced"
    
    def test_get_model_params_random_forest(self):
        """測試獲取隨機森林參數"""
        params = self.config_manager.get_model_params('random_forest')
        expected_keys = ['random_state', 'n_estimators', 'max_depth', 'class_weight', 'n_jobs']
        assert all(key in params for key in expected_keys)
        assert params['n_estimators'] == 100
        assert params['max_depth'] == 10
    
    def test_get_model_params_xgboost(self):
        """測試獲取XGBoost參數"""
        params = self.config_manager.get_model_params('xgboost')
        expected_keys = ['random_state', 'n_estimators', 'max_depth', 'learning_rate', 'eval_metric']
        assert all(key in params for key in expected_keys)
        assert params['learning_rate'] == 0.1
        assert params['eval_metric'] == "auc"
    
    def test_get_model_params_invalid_model(self):
        """測試獲取無效模型參數"""
        params = self.config_manager.get_model_params('invalid_model')
        # 應該只返回 random_state
        assert params == {'random_state': 42}
    
    def test_update_config_valid(self):
        """測試有效配置更新"""
        self.config_manager.update_config('data', 'missing_threshold', 0.8)
        assert self.config_manager.data_config.missing_threshold == 0.8
    
    def test_update_config_invalid_section(self):
        """測試無效配置節更新"""
        with pytest.raises(ValueError, match="無效的配置項"):
            self.config_manager.update_config('invalid_section', 'key', 'value')
    
    def test_update_config_invalid_key(self):
        """測試無效配置鍵更新"""
        with pytest.raises(ValueError, match="無效的配置項"):
            self.config_manager.update_config('data', 'invalid_key', 'value')
    
    def test_get_business_rules(self):
        """測試獲取業務規則"""
        rules = self.config_manager.get_business_rules()
        assert isinstance(rules, dict)
        assert 'transaction_amount_positive' in rules
        assert 'fraud_label_binary' in rules
        assert rules['transaction_amount_positive']['min_value'] == 0
        assert rules['fraud_label_binary']['allowed_values'] == [0, 1]
    
    def test_save_and_load_config(self):
        """測試配置保存和載入"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # 修改一些配置
            self.config_manager.update_config('data', 'missing_threshold', 0.8)
            self.config_manager.update_config('model', 'rf_n_estimators', 200)
            
            # 保存配置
            self.config_manager.save_to_file(temp_path)
            
            # 創建新的配置管理器並載入
            new_config_manager = ConfigManager(temp_path)
            
            # 驗證配置已正確載入
            assert new_config_manager.data_config.missing_threshold == 0.8
            assert new_config_manager.model_config.rf_n_estimators == 200
            
        finally:
            # 清理臨時文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_from_invalid_file(self):
        """測試從無效文件載入配置"""
        with pytest.raises(Exception):  # 應該拋出異常
            ConfigManager("nonexistent_file.json")
    
    def test_save_to_invalid_path(self):
        """測試保存到無效路徑"""
        with pytest.raises(Exception):
            self.config_manager.save_to_file("/invalid/path/config.json")
    
    @patch("builtins.open", mock_open(read_data='{"invalid": "json"'))
    def test_load_malformed_json(self):
        """測試載入格式錯誤的JSON"""
        with pytest.raises(Exception):
            ConfigManager("test_config.json")

class TestConfigIntegration:
    """測試配置集成功能"""
    
    def test_config_consistency(self):
        """測試配置一致性"""
        config_manager = ConfigManager()
        
        # 確保所有模型都能獲取到參數
        models = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        for model in models:
            params = config_manager.get_model_params(model)
            assert 'random_state' in params
            assert params['random_state'] == 42
    
    def test_feature_config_completeness(self):
        """測試特徵配置完整性"""
        config_manager = ConfigManager()
        feature_config = config_manager.feature_config
        
        # 確保所有必要的配置都存在
        assert hasattr(feature_config, 'time_bins')
        assert hasattr(feature_config, 'time_labels')
        assert hasattr(feature_config, 'amount_bins')
        assert hasattr(feature_config, 'amount_labels')
        assert hasattr(feature_config, 'agg_functions')
        
        # 確保配置值合理
        assert len(feature_config.time_bins) == len(feature_config.time_labels) + 1
        assert len(feature_config.amount_bins) == len(feature_config.amount_labels) + 1
    
    def test_system_config_directories(self):
        """測試系統配置目錄創建"""
        config_manager = ConfigManager()
        system_config = config_manager.system_config
        
        # 檢查目錄是否被創建
        directories = [
            system_config.models_dir,
            system_config.reports_dir,
            system_config.logs_dir,
            system_config.cache_dir
        ]
        
        for directory in directories:
            assert os.path.exists(directory)

if __name__ == "__main__":
    pytest.main([__file__])