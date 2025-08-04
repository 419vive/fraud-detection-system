"""
配置管理模組 - IEEE-CIS 詐騙檢測項目
集中管理所有模型參數、路徑和系統配置
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """數據配置"""
    train_transaction_path: str = "train_transaction.csv"
    train_identity_path: str = "train_identity.csv" 
    test_transaction_path: str = "test_transaction.csv"
    test_identity_path: str = "test_identity.csv"
    missing_threshold: float = 0.9
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    """模型配置"""
    # 通用參數
    random_state: int = 42
    n_jobs: int = -1
    
    # Logistic Regression
    logistic_max_iter: int = 1000
    logistic_class_weight: str = "balanced"
    
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_class_weight: str = "balanced"
    
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_eval_metric: str = "auc"
    
    # LightGBM
    lgb_n_estimators: int = 100
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_class_weight: str = "balanced"
    
    # CatBoost
    cb_iterations: int = 100
    cb_depth: int = 6
    cb_learning_rate: float = 0.1

@dataclass
class FeatureEngineeringConfig:
    """特徵工程配置"""
    # 時間特徵
    time_bins: list = None
    time_labels: list = None
    
    # 交易金額分組
    amount_bins: list = None
    amount_labels: list = None
    
    # 聚合特徵參數
    agg_functions: list = None
    
    # 特徵選擇
    feature_selection_k: int = 100
    
    def __post_init__(self):
        if self.time_bins is None:
            self.time_bins = [0, 6, 12, 18, 24]
        if self.time_labels is None:
            self.time_labels = ['夜晚', '早晨', '下午', '晚上']
        if self.amount_bins is None:
            self.amount_bins = [0, 50, 100, 500, 1000, float('inf')]
        if self.amount_labels is None:
            self.amount_labels = ['小額', '中小額', '中額', '大額', '超大額']
        if self.agg_functions is None:
            self.agg_functions = ['mean', 'std', 'count', 'sum', 'max', 'min']

@dataclass
class ValidationConfig:
    """驗證配置"""
    cv_folds: int = 5
    scoring_metrics: list = None
    validation_split: float = 0.2
    
    def __post_init__(self):
        if self.scoring_metrics is None:
            self.scoring_metrics = ['roc_auc', 'f1', 'precision', 'recall']

@dataclass
class SystemConfig:
    """系統配置"""
    # 路徑配置
    models_dir: str = "models"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    
    # API配置
    api_host: str = "localhost"
    api_port: int = 8000
    api_debug: bool = False
    
    # 日誌配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 內存管理
    chunk_size: int = 10000
    memory_limit_gb: float = 8.0
    
    # 業務規則閾值
    fraud_probability_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    
    def __post_init__(self):
        # 創建必要目錄
        for directory in [self.models_dir, self.reports_dir, self.logs_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.data_config = DataConfig()
        self.model_config = ModelConfig()
        self.feature_config = FeatureEngineeringConfig()
        self.validation_config = ValidationConfig()
        self.system_config = SystemConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """從文件載入配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 更新各個配置
            if 'data' in config_dict:
                self._update_config(self.data_config, config_dict['data'])
            if 'model' in config_dict:
                self._update_config(self.model_config, config_dict['model'])
            if 'feature_engineering' in config_dict:
                self._update_config(self.feature_config, config_dict['feature_engineering'])
            if 'validation' in config_dict:
                self._update_config(self.validation_config, config_dict['validation'])
            if 'system' in config_dict:
                self._update_config(self.system_config, config_dict['system'])
            
            logger.info(f"配置已從 {config_file} 載入")
            
        except Exception as e:
            logger.error(f"載入配置文件失敗: {e}")
            raise
    
    def _update_config(self, config_obj, config_dict: Dict[str, Any]):
        """更新配置對象"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        config_dict = {
            'data': self.data_config.__dict__,
            'model': self.model_config.__dict__,
            'feature_engineering': self.feature_config.__dict__,
            'validation': self.validation_config.__dict__,
            'system': self.system_config.__dict__
        }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"配置已保存至 {config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失敗: {e}")
            raise
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """獲取特定模型的參數"""
        params = {'random_state': self.model_config.random_state}
        
        if model_name == 'logistic':
            params.update({
                'max_iter': self.model_config.logistic_max_iter,
                'class_weight': self.model_config.logistic_class_weight
            })
        elif model_name == 'random_forest':
            params.update({
                'n_estimators': self.model_config.rf_n_estimators,
                'max_depth': self.model_config.rf_max_depth,
                'class_weight': self.model_config.rf_class_weight,
                'n_jobs': self.model_config.n_jobs
            })
        elif model_name == 'xgboost':
            params.update({
                'n_estimators': self.model_config.xgb_n_estimators,
                'max_depth': self.model_config.xgb_max_depth,
                'learning_rate': self.model_config.xgb_learning_rate,
                'eval_metric': self.model_config.xgb_eval_metric
            })
        elif model_name == 'lightgbm':
            params.update({
                'n_estimators': self.model_config.lgb_n_estimators,
                'max_depth': self.model_config.lgb_max_depth,
                'learning_rate': self.model_config.lgb_learning_rate,
                'class_weight': self.model_config.lgb_class_weight,
                'verbosity': -1
            })
        elif model_name == 'catboost':
            params.update({
                'iterations': self.model_config.cb_iterations,
                'depth': self.model_config.cb_depth,
                'learning_rate': self.model_config.cb_learning_rate,
                'verbose': False
            })
        
        return params
    
    def update_config(self, section: str, key: str, value: Any):
        """動態更新配置"""
        config_map = {
            'data': self.data_config,
            'model': self.model_config,
            'feature_engineering': self.feature_config,
            'validation': self.validation_config,
            'system': self.system_config
        }
        
        if section in config_map and hasattr(config_map[section], key):
            setattr(config_map[section], key, value)
            logger.info(f"更新配置: {section}.{key} = {value}")
        else:
            raise ValueError(f"無效的配置項: {section}.{key}")
    
    def get_business_rules(self) -> Dict[str, Any]:
        """獲取業務規則配置"""
        return {
            'transaction_amount_positive': {
                'type': 'numeric_range',
                'column': 'TransactionAmt',
                'min_value': 0,
                'description': '交易金額必須為正數'
            },
            'transaction_id_unique': {
                'type': 'uniqueness',
                'column': 'TransactionID',
                'description': '交易ID必須唯一'
            },
            'fraud_label_binary': {
                'type': 'categorical_values',
                'column': 'isFraud',
                'allowed_values': [0, 1],
                'description': '詐騙標籤必須為0或1'
            },
            'transaction_dt_reasonable': {
                'type': 'numeric_range',
                'column': 'TransactionDT',
                'min_value': 0,
                'description': '交易時間戳必須合理'
            },
            'fraud_probability_threshold': {
                'type': 'numeric_range',
                'column': 'fraud_probability',
                'min_value': 0,
                'max_value': 1,
                'threshold': self.system_config.fraud_probability_threshold,
                'description': f'詐騙概率閾值: {self.system_config.fraud_probability_threshold}'
            }
        }

# 全局配置實例
config_manager = ConfigManager()

# 便捷函數
def get_config() -> ConfigManager:
    """獲取全局配置管理器"""
    return config_manager

def load_config(config_file: str):
    """載入配置文件"""
    global config_manager
    config_manager = ConfigManager(config_file)

def save_config(config_file: str):
    """保存當前配置"""
    config_manager.save_to_file(config_file)

if __name__ == "__main__":
    # 測試配置管理器
    cm = ConfigManager()
    print("配置管理模組已載入完成！")
    
    # 保存默認配置示例
    cm.save_to_file("config.json")
    print("默認配置已保存至 config.json")