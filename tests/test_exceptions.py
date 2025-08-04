"""
異常處理模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import sys
sys.path.append('../src')

from src.exceptions import (
    FraudDetectionException, DataProcessingError, DataLoadError, DataValidationError,
    FeatureEngineeringError, FeatureCreationError, FeatureSelectionError,
    ModelError, ModelTrainingError, ModelPredictionError, ModelSaveError, ModelLoadError,
    ConfigurationError, InvalidConfigError, ConfigLoadError,
    APIError, PredictionServiceError, InvalidInputError,
    BusinessRuleViolationError, MemoryError, InsufficientMemoryError, DataTooLargeError,
    handle_exception, ExceptionRecoveryStrategy
)

class TestCustomExceptions:
    """測試自定義異常類"""
    
    def test_fraud_detection_exception_base(self):
        """測試基礎異常類"""
        message = "測試錯誤"
        error_code = "TEST_ERROR"
        exception = FraudDetectionException(message, error_code)
        
        assert str(exception) == message
        assert exception.message == message
        assert exception.error_code == error_code
    
    def test_data_load_error(self):
        """測試數據載入異常"""
        file_path = "test.csv"
        reason = "文件不存在"
        exception = DataLoadError(file_path, reason)
        
        assert exception.file_path == file_path
        assert exception.reason == reason
        assert exception.error_code == "DATA_LOAD_ERROR"
        assert file_path in str(exception)
        assert reason in str(exception)
    
    def test_data_validation_error(self):
        """測試數據驗證異常"""
        validation_type = "missing_values"
        details = "超過50%的值缺失"
        exception = DataValidationError(validation_type, details)
        
        assert exception.validation_type == validation_type
        assert exception.details == details
        assert exception.error_code == "DATA_VALIDATION_ERROR"
    
    def test_feature_creation_error(self):
        """測試特徵創建異常"""
        feature_name = "time_features"
        reason = "時間列不存在"
        exception = FeatureCreationError(feature_name, reason)
        
        assert exception.feature_name == feature_name
        assert exception.reason == reason
        assert exception.error_code == "FEATURE_CREATION_ERROR"
    
    def test_model_training_error(self):
        """測試模型訓練異常"""
        model_name = "XGBoost"
        reason = "內存不足"
        exception = ModelTrainingError(model_name, reason)
        
        assert exception.model_name == model_name
        assert exception.reason == reason
        assert exception.error_code == "MODEL_TRAINING_ERROR"
    
    def test_invalid_config_error(self):
        """測試無效配置異常"""
        config_key = "learning_rate"
        config_value = "-0.1"
        reason = "學習率必須為正數"
        exception = InvalidConfigError(config_key, config_value, reason)
        
        assert exception.config_key == config_key
        assert exception.config_value == config_value
        assert exception.reason == reason
        assert exception.error_code == "INVALID_CONFIG_ERROR"
    
    def test_insufficient_memory_error(self):
        """測試內存不足異常"""
        required_memory = 16.0
        available_memory = 8.0
        exception = InsufficientMemoryError(required_memory, available_memory)
        
        assert exception.required_memory == required_memory
        assert exception.available_memory == available_memory
        assert exception.error_code == "INSUFFICIENT_MEMORY_ERROR"
        assert "16.00GB" in str(exception)
        assert "8.00GB" in str(exception)
    
    def test_data_too_large_error(self):
        """測試數據過大異常"""
        data_size = 1000000
        max_size = 500000
        exception = DataTooLargeError(data_size, max_size)
        
        assert exception.data_size == data_size
        assert exception.max_size == max_size
        assert exception.error_code == "DATA_TOO_LARGE_ERROR"
        assert "1,000,000" in str(exception)
        assert "500,000" in str(exception)

class TestExceptionDecorator:
    """測試異常處理裝飾器"""
    
    def test_handle_exception_reraise_custom(self):
        """測試重新拋出自定義異常"""
        @handle_exception
        def test_function():
            raise FeatureCreationError("test_feature", "測試錯誤")
        
        with pytest.raises(FeatureCreationError):
            test_function()
    
    def test_handle_exception_file_not_found(self):
        """測試處理文件不存在異常"""
        @handle_exception
        def test_function():
            raise FileNotFoundError("test.csv")
        
        with pytest.raises(DataLoadError):
            test_function()
    
    def test_handle_exception_permission_error(self):
        """測試處理權限錯誤"""
        @handle_exception
        def test_function():
            error = PermissionError("Permission denied")
            error.filename = "test.csv"
            raise error
        
        with pytest.raises(DataLoadError):
            test_function()
    
    def test_handle_exception_value_error(self):
        """測試處理值錯誤"""
        @handle_exception
        def test_function():
            raise ValueError("無效的數值")
        
        with pytest.raises(DataValidationError):
            test_function()
    
    def test_handle_exception_key_error(self):
        """測試處理鍵錯誤"""
        @handle_exception
        def test_function():
            raise KeyError("missing_column")
        
        with pytest.raises(DataValidationError):
            test_function()
    
    def test_handle_exception_memory_error(self):
        """測試處理內存錯誤"""
        @handle_exception
        def test_function():
            raise MemoryError("內存不足")
        
        with pytest.raises(InsufficientMemoryError):
            test_function()
    
    def test_handle_exception_pandas_empty_data(self):
        """測試處理pandas空數據異常"""
        @handle_exception
        def test_function():
            raise pd.errors.EmptyDataError("空文件")
        
        with pytest.raises(DataLoadError):
            test_function()
    
    def test_handle_exception_pandas_parser_error(self):
        """測試處理pandas解析錯誤"""
        @handle_exception
        def test_function():
            raise pd.errors.ParserError("解析錯誤")
        
        with pytest.raises(DataLoadError):
            test_function()
    
    def test_handle_exception_generic_exception(self):
        """測試處理通用異常"""
        @handle_exception
        def test_function():
            raise RuntimeError("運行時錯誤")
        
        with pytest.raises(FraudDetectionException) as exc_info:
            test_function()
        
        assert exc_info.value.error_code == "UNEXPECTED_ERROR"
        assert "未預期的錯誤" in str(exc_info.value)

class TestExceptionRecoveryStrategy:
    """測試異常恢復策略"""
    
    def test_handle_data_load_error_with_alternatives(self):
        """測試處理數據載入錯誤（有替代路徑）"""
        import tempfile
        import os
        
        # 創建臨時文件作為替代路徑
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            alternative_path = tmp_file.name
        
        try:
            error = DataLoadError("nonexistent.csv", "文件不存在")
            result = ExceptionRecoveryStrategy.handle_data_load_error(
                error, [alternative_path]
            )
            assert result == alternative_path
        finally:
            os.unlink(alternative_path)
    
    def test_handle_data_load_error_no_alternatives(self):
        """測試處理數據載入錯誤（無替代路徑）"""
        error = DataLoadError("nonexistent.csv", "文件不存在")
        
        with pytest.raises(DataLoadError):
            ExceptionRecoveryStrategy.handle_data_load_error(error, [])
    
    def test_handle_memory_error(self):
        """測試處理內存錯誤"""
        error = InsufficientMemoryError(16.0, 8.0)
        result = ExceptionRecoveryStrategy.handle_memory_error(error, 5000)
        
        assert result['strategy'] == 'chunk_processing'
        assert result['recommended_chunk_size'] == 5000
        assert 'error' in result
    
    def test_handle_model_training_error_with_fallback(self):
        """測試處理模型訓練錯誤（有替代模型）"""
        error = ModelTrainingError("XGBoost", "內存不足")
        fallback_models = ["LogisticRegression", "RandomForest"]
        
        result = ExceptionRecoveryStrategy.handle_model_training_error(
            error, fallback_models
        )
        
        assert result['strategy'] == 'fallback_models'
        assert result['recommended_models'] == fallback_models
        assert 'error' in result
    
    def test_handle_model_training_error_no_fallback(self):
        """測試處理模型訓練錯誤（無替代模型）"""
        error = ModelTrainingError("XGBoost", "內存不足")
        
        result = ExceptionRecoveryStrategy.handle_model_training_error(error)
        
        assert result['strategy'] == 'reduce_complexity'
        assert 'recommendations' in result
        assert len(result['recommendations']) > 0
        assert 'error' in result

class TestExceptionIntegration:
    """測試異常集成功能"""
    
    def test_exception_hierarchy(self):
        """測試異常層次結構"""
        # 所有自定義異常都應該繼承自基類
        exceptions_to_test = [
            DataProcessingError, DataLoadError, DataValidationError,
            FeatureEngineeringError, FeatureCreationError, FeatureSelectionError,
            ModelError, ModelTrainingError, ModelPredictionError,
            ConfigurationError, InvalidConfigError, APIError
        ]
        
        for exception_class in exceptions_to_test:
            assert issubclass(exception_class, FraudDetectionException)
    
    def test_error_codes_uniqueness(self):
        """測試錯誤代碼唯一性"""
        error_codes = [
            "DATA_LOAD_ERROR", "DATA_VALIDATION_ERROR", "FEATURE_CREATION_ERROR",
            "FEATURE_SELECTION_ERROR", "MODEL_TRAINING_ERROR", "MODEL_PREDICTION_ERROR",
            "MODEL_SAVE_ERROR", "MODEL_LOAD_ERROR", "INVALID_CONFIG_ERROR",
            "CONFIG_LOAD_ERROR", "PREDICTION_SERVICE_ERROR", "INVALID_INPUT_ERROR",
            "BUSINESS_RULE_VIOLATION", "INSUFFICIENT_MEMORY_ERROR", "DATA_TOO_LARGE_ERROR"
        ]
        
        # 檢查錯誤代碼沒有重複
        assert len(error_codes) == len(set(error_codes))
    
    def test_exception_messages_localization(self):
        """測試異常消息本地化"""
        # 測試中文錯誤消息
        exceptions = [
            DataLoadError("test.csv", "文件不存在"),
            FeatureCreationError("時間特徵", "時間列缺失"),
            ModelTrainingError("XGBoost", "數據不平衡"),
            InvalidConfigError("learning_rate", "-0.1", "必須為正數")
        ]
        
        for exception in exceptions:
            message = str(exception)
            # 確保消息包含中文字符
            assert any('\u4e00' <= char <= '\u9fff' for char in message)

if __name__ == "__main__":
    pytest.main([__file__])