"""
自定義異常模組 - IEEE-CIS 詐騙檢測項目
定義項目特定的異常類型，提供更精確的錯誤處理
"""

class FraudDetectionException(Exception):
    """詐騙檢測系統基礎異常類"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DataProcessingError(FraudDetectionException):
    """數據處理相關異常"""
    pass

class DataLoadError(DataProcessingError):
    """數據載入異常"""
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        message = f"數據載入失敗 - 文件: {file_path}, 原因: {reason}"
        super().__init__(message, "DATA_LOAD_ERROR")

class DataValidationError(DataProcessingError):
    """數據驗證異常"""
    def __init__(self, validation_type: str, details: str):
        self.validation_type = validation_type
        self.details = details
        message = f"數據驗證失敗 - 類型: {validation_type}, 詳情: {details}"
        super().__init__(message, "DATA_VALIDATION_ERROR")

class FeatureEngineeringError(FraudDetectionException):
    """特徵工程相關異常"""
    pass

class FeatureCreationError(FeatureEngineeringError):
    """特徵創建異常"""
    def __init__(self, feature_name: str, reason: str):
        self.feature_name = feature_name
        self.reason = reason
        message = f"特徵創建失敗 - 特徵: {feature_name}, 原因: {reason}"
        super().__init__(message, "FEATURE_CREATION_ERROR")

class FeatureSelectionError(FeatureEngineeringError):
    """特徵選擇異常"""
    def __init__(self, method: str, reason: str):
        self.method = method
        self.reason = reason
        message = f"特徵選擇失敗 - 方法: {method}, 原因: {reason}"
        super().__init__(message, "FEATURE_SELECTION_ERROR")

class ModelError(FraudDetectionException):
    """模型相關異常"""
    pass

class ModelTrainingError(ModelError):
    """模型訓練異常"""
    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        message = f"模型訓練失敗 - 模型: {model_name}, 原因: {reason}"
        super().__init__(message, "MODEL_TRAINING_ERROR")

class ModelPredictionError(ModelError):
    """模型預測異常"""
    def __init__(self, model_name: str, reason: str):
        self.model_name = model_name
        self.reason = reason
        message = f"模型預測失敗 - 模型: {model_name}, 原因: {reason}"
        super().__init__(message, "MODEL_PREDICTION_ERROR")

class ModelSaveError(ModelError):
    """模型保存異常"""
    def __init__(self, model_name: str, file_path: str, reason: str):
        self.model_name = model_name
        self.file_path = file_path
        self.reason = reason
        message = f"模型保存失敗 - 模型: {model_name}, 路徑: {file_path}, 原因: {reason}"
        super().__init__(message, "MODEL_SAVE_ERROR")

class ModelLoadError(ModelError):
    """模型載入異常"""
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        message = f"模型載入失敗 - 路徑: {file_path}, 原因: {reason}"
        super().__init__(message, "MODEL_LOAD_ERROR")

class ConfigurationError(FraudDetectionException):
    """配置相關異常"""
    pass

class InvalidConfigError(ConfigurationError):
    """無效配置異常"""
    def __init__(self, config_key: str, config_value: str, reason: str):
        self.config_key = config_key
        self.config_value = config_value
        self.reason = reason
        message = f"無效配置 - 鍵: {config_key}, 值: {config_value}, 原因: {reason}"
        super().__init__(message, "INVALID_CONFIG_ERROR")

class ConfigLoadError(ConfigurationError):
    """配置載入異常"""
    def __init__(self, config_file: str, reason: str):
        self.config_file = config_file
        self.reason = reason
        message = f"配置載入失敗 - 文件: {config_file}, 原因: {reason}"
        super().__init__(message, "CONFIG_LOAD_ERROR")

class APIError(FraudDetectionException):
    """API相關異常"""
    pass

class PredictionServiceError(APIError):
    """預測服務異常"""
    def __init__(self, endpoint: str, reason: str):
        self.endpoint = endpoint
        self.reason = reason
        message = f"預測服務錯誤 - 端點: {endpoint}, 原因: {reason}"
        super().__init__(message, "PREDICTION_SERVICE_ERROR")

class InvalidInputError(APIError):
    """無效輸入異常"""
    def __init__(self, input_field: str, input_value: str, expected_format: str):
        self.input_field = input_field
        self.input_value = input_value
        self.expected_format = expected_format
        message = f"無效輸入 - 欄位: {input_field}, 值: {input_value}, 期望格式: {expected_format}"
        super().__init__(message, "INVALID_INPUT_ERROR")

class BusinessRuleViolationError(FraudDetectionException):
    """業務規則違反異常"""
    def __init__(self, rule_name: str, violation_details: str):
        self.rule_name = rule_name
        self.violation_details = violation_details
        message = f"業務規則違反 - 規則: {rule_name}, 詳情: {violation_details}"
        super().__init__(message, "BUSINESS_RULE_VIOLATION")

class MemoryError(FraudDetectionException):
    """內存相關異常"""
    pass

class InsufficientMemoryError(MemoryError):
    """內存不足異常"""
    def __init__(self, required_memory: float, available_memory: float):
        self.required_memory = required_memory
        self.available_memory = available_memory
        message = f"內存不足 - 需要: {required_memory:.2f}GB, 可用: {available_memory:.2f}GB"
        super().__init__(message, "INSUFFICIENT_MEMORY_ERROR")

class DataTooLargeError(MemoryError):
    """數據過大異常"""
    def __init__(self, data_size: int, max_size: int):
        self.data_size = data_size
        self.max_size = max_size
        message = f"數據過大 - 當前大小: {data_size:,}行, 最大限制: {max_size:,}行"
        super().__init__(message, "DATA_TOO_LARGE_ERROR")

# 異常處理工具函數
def handle_exception(func):
    """異常處理裝飾器"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FraudDetectionException:
            # 重新拋出我們的自定義異常
            raise
        except FileNotFoundError as e:
            raise DataLoadError(str(e.filename), f"文件不存在: {e}")
        except PermissionError as e:
            raise DataLoadError(str(e.filename), f"權限不足: {e}")
        except pd.errors.EmptyDataError as e:
            raise DataLoadError("unknown", f"空數據文件: {e}")
        except pd.errors.ParserError as e:
            raise DataLoadError("unknown", f"數據解析錯誤: {e}")
        except ValueError as e:
            raise DataValidationError("value_error", str(e))
        except KeyError as e:
            raise DataValidationError("key_error", f"缺少必要欄位: {e}")
        except MemoryError as e:
            raise InsufficientMemoryError(0, 0)  # 無法確定具體數值
        except Exception as e:
            # 其他未預期的異常
            raise FraudDetectionException(f"未預期的錯誤: {e}", "UNEXPECTED_ERROR")
    
    return wrapper

# 異常恢復策略
class ExceptionRecoveryStrategy:
    """異常恢復策略"""
    
    @staticmethod
    def handle_data_load_error(error: DataLoadError, alternative_paths: list = None):
        """處理數據載入錯誤"""
        if alternative_paths:
            for alt_path in alternative_paths:
                try:
                    if os.path.exists(alt_path):
                        return alt_path
                except:
                    continue
        
        raise error
    
    @staticmethod
    def handle_memory_error(error: InsufficientMemoryError, chunk_size: int = 10000):
        """處理內存錯誤，建議分塊處理"""
        return {
            'strategy': 'chunk_processing',
            'recommended_chunk_size': chunk_size,
            'error': error.message
        }
    
    @staticmethod
    def handle_model_training_error(error: ModelTrainingError, fallback_models: list = None):
        """處理模型訓練錯誤"""
        if fallback_models:
            return {
                'strategy': 'fallback_models',
                'recommended_models': fallback_models,
                'error': error.message
            }
        
        return {
            'strategy': 'reduce_complexity',
            'recommendations': [
                '減少特徵數量',
                '使用更簡單的模型',
                '減少樣本數量',
                '調整超參數'
            ],
            'error': error.message
        }

if __name__ == "__main__":
    print("自定義異常模組已載入完成！")
    
    # 測試異常
    try:
        raise ModelTrainingError("XGBoost", "數據不平衡嚴重")
    except ModelTrainingError as e:
        print(f"捕獲異常: {e.error_code} - {e.message}")