"""
集成測試 - IEEE-CIS 詐騙檢測項目
測試完整的機器學習流水線集成
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import time
import json

sys.path.append('../src')

from src.data_processing import DataProcessor
from src.data_validation import DataValidator
from src.feature_engineering import FeatureEngineer
from src.modeling import FraudDetectionModel
from src.memory_optimizer import MemoryProfiler, ChunkProcessor
from src.model_monitoring import ModelMonitor
from src.config import ConfigManager
from src.exceptions import (
    DataProcessingError, ModelTrainingError, FeatureEngineeringError
)


class TestCompleteMLPipeline:
    """測試完整的機器學習流水線"""
    
    def setup_method(self):
        """測試設置"""
        # 創建完整的詐騙檢測測試數據
        np.random.seed(42)
        self.sample_size = 5000
        
        # 生成真實場景的詐騙檢測數據
        self.fraud_data = self._generate_fraud_detection_data()
        
        # 分離交易和身份數據（模擬真實場景）
        self.transaction_data = self.fraud_data[
            ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD'] +
            [col for col in self.fraud_data.columns if col.startswith('C') or col.startswith('D')]
        ].copy()
        
        self.identity_data = self.fraud_data[
            ['TransactionID'] + 
            [col for col in self.fraud_data.columns if col.startswith('id_') or col == 'DeviceType']
        ].copy()
        
        # 配置管理器
        self.config = ConfigManager()
    
    def _generate_fraud_detection_data(self) -> pd.DataFrame:
        """生成真實場景的詐騙檢測數據"""
        data = pd.DataFrame()
        
        # 基本交易信息
        data['TransactionID'] = range(1, self.sample_size + 1)
        data['TransactionDT'] = np.random.randint(86400, 86400*30, self.sample_size)
        data['TransactionAmt'] = np.random.lognormal(3, 1.5, self.sample_size)
        
        # 產品類別
        data['ProductCD'] = np.random.choice(['W', 'C', 'H', 'R', 'S'], self.sample_size, 
                                           p=[0.4, 0.25, 0.15, 0.1, 0.1])
        
        # 卡片信息
        data['card1'] = np.random.randint(1000, 20000, self.sample_size)
        data['card2'] = np.random.randint(100, 600, self.sample_size)
        data['card3'] = np.random.choice(['visa', 'mastercard', 'american_express'], self.sample_size)
        
        # 地址信息
        data['addr1'] = np.random.choice(range(100, 500), self.sample_size)
        data['addr2'] = np.random.choice(range(10, 100), self.sample_size)
        
        # C類特徵（金額相關）
        for i in range(1, 15):
            if np.random.random() > 0.3:  # 30%的特徵有缺失
                missing_mask = np.random.random(self.sample_size) < 0.1
                values = np.random.randn(self.sample_size)
                values[missing_mask] = np.nan
                data[f'C{i}'] = values
        
        # D類特徵（時間相關）
        for i in range(1, 16):
            if np.random.random() > 0.4:  # 40%的特徵有缺失
                missing_mask = np.random.random(self.sample_size) < 0.15
                values = np.random.randint(0, 365, self.sample_size).astype(float)
                values[missing_mask] = np.nan
                data[f'D{i}'] = values
        
        # 身份信息
        data['DeviceType'] = np.random.choice(['desktop', 'mobile', 'tablet'], self.sample_size)
        
        for i in range(1, 39):
            if np.random.random() > 0.5:  # 50%的身份特徵有缺失
                if i in [12, 13, 14, 15]:  # 某些id特徵是類別型
                    values = np.random.choice(['chrome', 'safari', 'firefox', 'edge'], self.sample_size)
                else:
                    missing_mask = np.random.random(self.sample_size) < 0.2
                    values = np.random.randn(self.sample_size)
                    values[missing_mask] = np.nan
                data[f'id_{i:02d}'] = values
        
        # 生成詐騙標籤（基於特徵的複雜規則）
        fraud_probability = np.zeros(self.sample_size)
        
        # 大額交易更可能是詐騙
        fraud_probability += (data['TransactionAmt'] > 1000) * 0.2
        
        # 特定產品類別風險更高
        fraud_probability += (data['ProductCD'].isin(['H', 'R'])) * 0.1
        
        # 夜間交易風險更高
        hour = (data['TransactionDT'] / 3600) % 24
        fraud_probability += ((hour < 6) | (hour > 22)) * 0.15
        
        # 某些C特徵異常值
        if 'C1' in data.columns:
            fraud_probability += (np.abs(data['C1']) > 2) * 0.1
        
        # 添加隨機性
        fraud_probability += np.random.random(self.sample_size) * 0.1
        
        # 生成最終標籤
        data['isFraud'] = (fraud_probability > 0.3).astype(int)
        
        # 確保詐騙率合理
        current_fraud_rate = data['isFraud'].mean()
        if current_fraud_rate > 0.1:  # 如果詐騙率過高，隨機降低
            fraud_indices = data[data['isFraud'] == 1].index
            reduce_count = int((current_fraud_rate - 0.05) * len(data))
            reduce_indices = np.random.choice(fraud_indices, reduce_count, replace=False)
            data.loc[reduce_indices, 'isFraud'] = 0
        
        return data
    
    def test_complete_pipeline_success(self):
        """測試完整流水線成功執行"""
        # 1. 數據載入和預處理
        processor = DataProcessor()
        
        # 模擬從文件載入（實際使用內存數據）
        merged_data = self.transaction_data.merge(
            self.identity_data, on='TransactionID', how='left'
        )
        
        # 基本預處理
        processed_data = processor.basic_preprocessing(merged_data)
        
        # 2. 數據驗證
        validator = DataValidator()
        
        # 結構驗證
        structure_result = validator.validate_data_structure(processed_data)
        assert structure_result['structure_valid'] == True
        
        # 品質驗證
        quality_result = validator.validate_data_quality(processed_data)
        assert quality_result['quality_score'] > 0
        
        # 業務規則驗證
        business_result = validator.validate_business_rules(processed_data)
        assert business_result['rules_checked'] > 0
        
        # 3. 特徵工程
        engineer = FeatureEngineer(self.config)
        
        # 執行特徵工程
        engineered_data = engineer.full_feature_engineering_pipeline(
            processed_data, target_col='isFraud', enable_advanced_features=False
        )
        
        # 4. 模型訓練和評估
        model_trainer = FraudDetectionModel()
        
        # 準備數據
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            engineered_data, target_col='isFraud', test_size=0.2
        )
        
        # 訓練幾個模型（不是全部，節省時間）
        model_trainer.train_random_forest(X_train, y_train)
        model_trainer.train_xgboost(X_train, y_train)
        
        # 評估模型
        rf_result = model_trainer.evaluate_model('random_forest', X_test, y_test)
        xgb_result = model_trainer.evaluate_model('xgboost', X_test, y_test)
        
        # 5. 模型監控
        monitor = ModelMonitor('integration_test_model', X_train)
        
        # 模擬監控數據
        y_pred_rf = model_trainer.models['random_forest'].predict(X_test)
        y_pred_proba_rf = model_trainer.models['random_forest'].predict_proba(X_test)[:, 1]
        
        monitor.log_performance(y_test, y_pred_rf, y_pred_proba_rf)
        
        # 檢測數據漂移
        drift_results = monitor.detect_data_drift(X_test, features=X_test.columns[:5].tolist())
        
        # 驗證整個流水線結果
        assert len(processed_data) > 0
        assert len(engineered_data.columns) > len(processed_data.columns)
        assert 0 <= rf_result['roc_auc'] <= 1
        assert 0 <= xgb_result['roc_auc'] <= 1
        assert len(monitor.performance_history) == 1
        assert isinstance(drift_results, list)
        
        # 檢查模型性能合理性
        assert rf_result['accuracy'] > 0.5  # 至少比隨機猜測好
        assert xgb_result['accuracy'] > 0.5
    
    def test_pipeline_with_memory_optimization(self):
        """測試帶內存優化的流水線"""
        # 1. 內存優化的數據處理
        original_memory = MemoryProfiler.estimate_dataframe_memory(self.fraud_data)
        optimized_data = MemoryProfiler.optimize_dataframe_memory(self.fraud_data)
        optimized_memory = MemoryProfiler.estimate_dataframe_memory(optimized_data)
        
        # 內存應該減少或持平
        assert optimized_memory <= original_memory
        
        # 2. 分塊處理大數據
        chunk_processor = ChunkProcessor(chunk_size=1000)
        
        def simple_transform(df):
            # 簡單的變換操作
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            return df
        
        processed_chunks = chunk_processor.process_dataframe_in_chunks(
            optimized_data, simple_transform
        )
        
        # 3. 繼續流水線
        engineer = FeatureEngineer(enable_parallel=False)  # 關閉並行處理節省資源
        
        # 只使用基本特徵工程
        basic_features = engineer.create_time_features(processed_chunks)
        basic_features = engineer.create_transaction_amount_features(basic_features)
        
        # 4. 簡單模型訓練
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(basic_features)
        
        # 只訓練一個快速模型
        model_trainer.train_random_forest(X_train, y_train)
        evaluation = model_trainer.evaluate_model('random_forest', X_test, y_test)
        
        # 驗證結果
        assert len(processed_chunks) == len(optimized_data)
        assert 'TransactionAmt_log' in processed_chunks.columns
        assert len(basic_features.columns) > len(processed_chunks.columns)
        assert 0 <= evaluation['roc_auc'] <= 1
    
    def test_pipeline_error_handling(self):
        """測試流水線錯誤處理"""
        # 1. 測試缺少必要列的數據
        incomplete_data = self.fraud_data.drop(columns=['isFraud'])
        
        processor = DataProcessor()
        processed_incomplete = processor.basic_preprocessing(incomplete_data)
        
        # 2. 嘗試特徵工程（應該處理缺失目標列）
        engineer = FeatureEngineer()
        
        # 創建時間特徵應該成功
        time_features = engineer.create_time_features(processed_incomplete)
        assert 'hour' in time_features.columns
        
        # 3. 測試模型訓練錯誤處理
        model_trainer = FraudDetectionModel()
        
        # 沒有目標列應該引發錯誤
        with pytest.raises(KeyError):
            model_trainer.prepare_data(processed_incomplete, target_col='isFraud')
        
        # 4. 測試數據驗證對問題數據的處理
        validator = DataValidator()
        
        # 缺失關鍵列的驗證
        structure_result = validator.validate_data_structure(
            processed_incomplete, 
            expected_columns=['TransactionID', 'isFraud', 'TransactionAmt']
        )
        assert not structure_result['structure_valid']
        assert 'isFraud' in structure_result['missing_columns']
    
    def test_pipeline_with_small_dataset(self):
        """測試小數據集的流水線"""
        # 創建很小的數據集
        small_data = self.fraud_data.head(100).copy()
        
        # 1. 數據處理
        processor = DataProcessor()
        processed_small = processor.basic_preprocessing(small_data)
        
        # 2. 特徵工程
        engineer = FeatureEngineer()
        
        # 只做基本特徵工程
        engineered_small = engineer.create_time_features(processed_small)
        engineered_small = engineer.create_transaction_amount_features(engineered_small)
        
        # 3. 模型訓練
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(
            engineered_small, test_size=0.3  # 小數據集用更大的測試集
        )
        
        # 訓練簡單模型
        model_trainer.train_random_forest(X_train, y_train)
        evaluation = model_trainer.evaluate_model('random_forest', X_test, y_test)
        
        # 小數據集的結果可能不穩定，但應該能完成
        assert isinstance(evaluation['accuracy'], float)
        assert 0 <= evaluation['roc_auc'] <= 1
        assert len(X_train) > 0
        assert len(X_test) > 0
    
    def test_pipeline_performance_tracking(self):
        """測試流水線性能追踪"""
        performance_log = {}
        
        # 1. 數據處理性能
        start_time = time.time()
        processor = DataProcessor()
        processed_data = processor.basic_preprocessing(self.fraud_data)
        performance_log['data_processing'] = time.time() - start_time
        
        # 2. 特徵工程性能
        start_time = time.time()
        engineer = FeatureEngineer()
        engineered_data = engineer.full_feature_engineering_pipeline(
            processed_data, enable_advanced_features=False
        )
        performance_log['feature_engineering'] = time.time() - start_time
        
        # 3. 模型訓練性能
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_data)
        
        start_time = time.time()
        model_trainer.train_random_forest(X_train, y_train)
        performance_log['model_training'] = time.time() - start_time
        
        # 4. 模型評估性能
        start_time = time.time()
        evaluation = model_trainer.evaluate_model('random_forest', X_test, y_test)
        performance_log['model_evaluation'] = time.time() - start_time
        
        # 檢查所有步驟都在合理時間內完成
        assert performance_log['data_processing'] < 60  # 1分鐘
        assert performance_log['feature_engineering'] < 120  # 2分鐘
        assert performance_log['model_training'] < 180  # 3分鐘
        assert performance_log['model_evaluation'] < 30  # 30秒
        
        # 檢查總體性能
        total_time = sum(performance_log.values())
        assert total_time < 300  # 總共5分鐘內
        
        # 記錄性能日誌供調試
        print(f"Pipeline Performance Log: {performance_log}")
        print(f"Total Time: {total_time:.2f} seconds")
    
    def test_end_to_end_prediction_workflow(self):
        """測試端到端預測工作流程"""
        # 1. 準備訓練數據
        train_data = self.fraud_data.iloc[:3000].copy()  # 前3000條作訓練
        test_data = self.fraud_data.iloc[3000:].copy()   # 後面作測試
        
        # 2. 完整訓練流程
        processor = DataProcessor()
        train_processed = processor.basic_preprocessing(train_data)
        
        engineer = FeatureEngineer()
        train_engineered = engineer.full_feature_engineering_pipeline(
            train_processed, enable_advanced_features=False
        )
        
        model_trainer = FraudDetectionModel()
        X_train, X_val, y_train, y_val = model_trainer.prepare_data(train_engineered)
        
        # 訓練模型
        model_trainer.train_random_forest(X_train, y_train)
        train_evaluation = model_trainer.evaluate_model('random_forest', X_val, y_val)
        
        # 3. 新數據預測流程
        # 移除目標列模擬真實預測場景
        test_features = test_data.drop(columns=['isFraud'])
        
        # 相同的預處理步驟
        test_processed = processor.basic_preprocessing(test_features)
        
        # 使用訓練時的特徵工程器
        test_engineered = engineer.create_time_features(test_processed)
        test_engineered = engineer.create_transaction_amount_features(test_engineered)
        test_engineered = engineer.handle_missing_values(test_engineered)
        
        # 確保特徵列匹配
        train_features = set(X_train.columns)
        test_feature_cols = [col for col in test_engineered.columns if col in train_features]
        missing_features = train_features - set(test_feature_cols)
        
        # 添加缺失特徵（用0填充）
        for feature in missing_features:
            test_engineered[feature] = 0
        
        # 選擇相同的特徵列
        test_X = test_engineered[X_train.columns]
        
        # 4. 模型預測
        model = model_trainer.models['random_forest']
        predictions = model.predict(test_X)
        prediction_probabilities = model.predict_proba(test_X)[:, 1]
        
        # 5. 預測後處理和監控
        y_test_true = test_data['isFraud'].values
        
        # 計算實際性能
        from sklearn.metrics import accuracy_score, roc_auc_score
        test_accuracy = accuracy_score(y_test_true, predictions)
        test_auc = roc_auc_score(y_test_true, prediction_probabilities)
        
        # 設置監控
        monitor = ModelMonitor('production_model', X_train)
        monitor.log_performance(y_test_true, predictions, prediction_probabilities)
        
        # 檢測數據漂移
        drift_results = monitor.detect_data_drift(test_X, features=X_train.columns[:10].tolist())
        
        # 驗證端到端結果
        assert len(predictions) == len(test_data)
        assert len(prediction_probabilities) == len(test_data)
        assert 0 <= test_accuracy <= 1
        assert 0 <= test_auc <= 1
        assert len(monitor.performance_history) == 1
        
        # 檢查預測結果合理性
        fraud_predictions = np.sum(predictions)
        total_predictions = len(predictions)
        predicted_fraud_rate = fraud_predictions / total_predictions
        
        # 預測的詐騙率應該在合理範圍內
        assert 0 <= predicted_fraud_rate <= 0.2  # 最多20%
    
    def test_pipeline_configuration_management(self):
        """測試流水線配置管理"""
        # 1. 測試配置載入和使用
        config = ConfigManager()
        
        # 修改一些配置
        config.update_config('data', 'missing_threshold', 0.8)
        config.update_config('model', 'rf_n_estimators', 50)
        
        # 2. 使用配置進行數據處理
        processor = DataProcessor()
        processor.missing_threshold = config.data_config.missing_threshold
        
        processed_data = processor.basic_preprocessing(self.fraud_data)
        
        # 3. 使用配置進行特徵工程
        engineer = FeatureEngineer(config)
        engineered_data = engineer.full_feature_engineering_pipeline(
            processed_data, enable_advanced_features=False
        )
        
        # 4. 使用配置進行模型訓練
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_data)
        
        # 使用配置中的參數
        rf_params = config.get_model_params('random_forest')
        rf_params['n_estimators'] = 50  # 使用修改的配置
        
        from sklearn.ensemble import RandomForestClassifier
        custom_rf = RandomForestClassifier(**rf_params)
        custom_rf.fit(X_train, y_train)
        
        # 5. 驗證配置被正確使用
        assert processor.missing_threshold == 0.8
        assert custom_rf.n_estimators == 50
        assert custom_rf.random_state == 42  # 默認配置
        
        # 評估定制模型
        predictions = custom_rf.predict(X_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, predictions)
        assert 0 <= accuracy <= 1
    
    def test_pipeline_failure_recovery(self):
        """測試流水線失敗恢復"""
        # 1. 模擬數據處理階段失敗
        corrupted_data = self.fraud_data.copy()
        corrupted_data.loc[0, 'TransactionAmt'] = 'invalid_value'  # 引入無效數據
        
        processor = DataProcessor()
        
        # 應該能處理無效數據而不崩潰
        try:
            processed_data = processor.basic_preprocessing(corrupted_data)
            # 如果成功處理，檢查結果
            assert len(processed_data) > 0
        except Exception as e:
            # 如果失敗，確保是預期的異常類型
            assert isinstance(e, (ValueError, TypeError, DataProcessingError))
        
        # 2. 使用乾淨數據繼續流水線
        clean_data = self.fraud_data.copy()
        processed_clean = processor.basic_preprocessing(clean_data)
        
        # 3. 模擬特徵工程階段失敗
        engineer = FeatureEngineer()
        
        try:
            # 嘗試在沒有必要列的數據上進行特徵工程
            no_time_data = processed_clean.drop(columns=['TransactionDT'])
            time_features = engineer.create_time_features(no_time_data)
        except FeatureEngineeringError:
            # 預期的異常
            pass
        
        # 使用完整數據繼續
        engineered_data = engineer.create_time_features(processed_clean)
        engineered_data = engineer.create_transaction_amount_features(engineered_data)
        
        # 4. 模擬模型訓練階段失敗
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered_data)
        
        # 模擬單類問題（移除所有詐騙案例）
        y_single_class = np.zeros_like(y_train)
        
        try:
            # 嘗試訓練模型（可能會有警告但不應該崩潰）
            model_trainer.train_random_forest(X_train, y_single_class)
            
            # 如果成功，進行評估
            if 'random_forest' in model_trainer.models:
                evaluation = model_trainer.evaluate_model('random_forest', X_test, np.zeros_like(y_test))
                assert isinstance(evaluation, dict)
        except (ModelTrainingError, ValueError) as e:
            # 預期的異常
            pass
        
        # 5. 恢復到正常流程
        model_trainer.train_random_forest(X_train, y_train)
        final_evaluation = model_trainer.evaluate_model('random_forest', X_test, y_test)
        
        # 驗證最終恢復成功
        assert 0 <= final_evaluation['roc_auc'] <= 1
        assert final_evaluation['accuracy'] > 0


class TestPipelineDataFlow:
    """測試流水線數據流轉"""
    
    def setup_method(self):
        """測試設置"""
        np.random.seed(42)
        self.base_data = pd.DataFrame({
            'TransactionID': range(1, 1001),
            'isFraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
            'TransactionDT': np.random.randint(86400, 86400*7, 1000),
            'TransactionAmt': np.random.lognormal(3, 1, 1000),
            'ProductCD': np.random.choice(['W', 'C', 'H'], 1000),
            'C1': np.random.randn(1000),
            'C2': np.random.randn(1000)
        })
    
    def test_data_shape_consistency(self):
        """測試數據形狀一致性"""
        original_rows = len(self.base_data)
        
        # 1. 數據處理
        processor = DataProcessor()
        processed = processor.basic_preprocessing(self.base_data)
        
        # 行數應該保持不變
        assert len(processed) == original_rows
        
        # 2. 特徵工程
        engineer = FeatureEngineer()
        
        # 時間特徵
        with_time = engineer.create_time_features(processed)
        assert len(with_time) == original_rows
        
        # 交易金額特徵
        with_amount = engineer.create_transaction_amount_features(with_time)
        assert len(with_amount) == original_rows
        
        # 缺失值處理
        with_imputed = engineer.handle_missing_values(with_amount)
        assert len(with_imputed) == original_rows
        
        # 3. 模型準備
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(with_imputed)
        
        # 訓練集和測試集行數之和應該等於原始數據
        assert len(X_train) + len(X_test) == original_rows
        assert len(y_train) + len(y_test) == original_rows
    
    def test_feature_column_tracking(self):
        """測試特徵列追踪"""
        original_columns = set(self.base_data.columns)
        
        # 1. 數據處理階段
        processor = DataProcessor()
        processed = processor.basic_preprocessing(self.base_data)
        processed_columns = set(processed.columns)
        
        # 高缺失特徵可能被移除，但核心列應該保留
        core_columns = {'TransactionID', 'isFraud', 'TransactionAmt', 'TransactionDT'}
        assert core_columns.issubset(processed_columns)
        
        # 2. 特徵工程階段
        engineer = FeatureEngineer()
        
        # 時間特徵
        with_time = engineer.create_time_features(processed)
        time_columns = set(with_time.columns)
        
        # 應該增加時間相關特徵
        expected_time_features = {'hour', 'day', 'week', 'is_weekend'}
        assert expected_time_features.issubset(time_columns)
        
        # 金額特徵
        with_amount = engineer.create_transaction_amount_features(with_time)
        amount_columns = set(with_amount.columns)
        
        # 應該增加金額相關特徵
        expected_amount_features = {'TransactionAmt_log', 'TransactionAmt_sqrt', 'amt_range'}
        assert expected_amount_features.issubset(amount_columns)
        
        # 3. 模型準備階段
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(with_amount)
        
        # 目標列和ID列應該被移除
        model_columns = set(X_train.columns)
        assert 'isFraud' not in model_columns
        assert 'TransactionID' not in model_columns
        
        # 特徵列應該都是數值型
        assert all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
    
    def test_data_type_consistency(self):
        """測試數據類型一致性"""
        # 1. 檢查原始數據類型
        original_dtypes = self.base_data.dtypes.to_dict()
        
        # 2. 數據處理後的類型
        processor = DataProcessor()
        processed = processor.basic_preprocessing(self.base_data)
        processed_dtypes = processed.dtypes.to_dict()
        
        # 核心數值列應該保持數值型
        numeric_columns = ['TransactionDT', 'TransactionAmt', 'C1', 'C2']
        for col in numeric_columns:
            if col in processed_dtypes:
                assert np.issubdtype(processed_dtypes[col], np.number)
        
        # 3. 特徵工程後的類型
        engineer = FeatureEngineer()
        engineered = engineer.full_feature_engineering_pipeline(
            processed, enable_advanced_features=False
        )
        
        # 模型準備
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered)
        
        # 所有特徵都應該是數值型
        assert all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
        assert all(X_test.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
        
        # 目標變數應該是整數型
        assert np.issubdtype(y_train.dtype, np.integer)
        assert np.issubdtype(y_test.dtype, np.integer)
    
    def test_missing_value_handling(self):
        """測試缺失值處理"""
        # 1. 引入缺失值
        data_with_missing = self.base_data.copy()
        missing_indices = np.random.choice(1000, 200, replace=False)
        data_with_missing.loc[missing_indices[:100], 'C1'] = np.nan
        data_with_missing.loc[missing_indices[100:], 'C2'] = np.nan
        
        # 檢查初始缺失值
        initial_missing = data_with_missing.isnull().sum().sum()
        assert initial_missing > 0
        
        # 2. 數據處理
        processor = DataProcessor()
        processed = processor.basic_preprocessing(data_with_missing)
        
        # 3. 特徵工程（包含缺失值處理）
        engineer = FeatureEngineer()
        engineered = engineer.full_feature_engineering_pipeline(
            processed, enable_advanced_features=False
        )
        
        # 4. 模型準備
        model_trainer = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered)
        
        # 檢查最終缺失值
        final_missing_train = X_train.isnull().sum().sum()
        final_missing_test = X_test.isnull().sum().sum()
        
        # 模型輸入不應該有缺失值
        assert final_missing_train == 0
        assert final_missing_test == 0


class TestPipelineScalability:
    """測試流水線可擴展性"""
    
    def test_pipeline_with_different_data_sizes(self):
        """測試不同數據大小的流水線性能"""
        data_sizes = [100, 500, 1000, 2000]
        performance_results = {}
        
        for size in data_sizes:
            print(f"Testing pipeline with {size} samples...")
            
            # 生成指定大小的數據
            test_data = pd.DataFrame({
                'TransactionID': range(1, size + 1),
                'isFraud': np.random.choice([0, 1], size, p=[0.95, 0.05]),
                'TransactionDT': np.random.randint(86400, 86400*7, size),
                'TransactionAmt': np.random.lognormal(3, 1, size),
                'ProductCD': np.random.choice(['W', 'C', 'H'], size),
                'C1': np.random.randn(size),
                'C2': np.random.randn(size)
            })
            
            # 測試流水線執行時間
            start_time = time.time()
            
            # 簡化的流水線
            processor = DataProcessor()
            processed = processor.basic_preprocessing(test_data)
            
            engineer = FeatureEngineer()
            engineered = engineer.create_time_features(processed)
            engineered = engineer.create_transaction_amount_features(engineered)
            
            model_trainer = FraudDetectionModel()
            X_train, X_test, y_train, y_test = model_trainer.prepare_data(engineered, test_size=0.3)
            
            # 只訓練快速模型
            model_trainer.train_random_forest(X_train, y_train)
            evaluation = model_trainer.evaluate_model('random_forest', X_test, y_test)
            
            execution_time = time.time() - start_time
            performance_results[size] = {
                'execution_time': execution_time,
                'roc_auc': evaluation['roc_auc']
            }
            
            print(f"Size {size}: Time={execution_time:.2f}s, AUC={evaluation['roc_auc']:.4f}")
        
        # 檢查性能合理性
        for size, result in performance_results.items():
            assert result['execution_time'] < 300  # 每個大小都應該在5分鐘內完成
            assert 0 <= result['roc_auc'] <= 1
        
        # 檢查時間複雜度合理性（不應該指數增長）
        times = [result['execution_time'] for result in performance_results.values()]
        for i in range(1, len(times)):
            # 時間增長不應該超過數據增長的平方
            time_ratio = times[i] / times[i-1]
            data_ratio = data_sizes[i] / data_sizes[i-1]
            assert time_ratio <= data_ratio ** 1.5  # 允許一定的非線性增長
    
    def test_pipeline_memory_efficiency(self):
        """測試流水線內存效率"""
        # 創建較大的測試數據
        large_data = pd.DataFrame({
            'TransactionID': range(1, 10001),
            'isFraud': np.random.choice([0, 1], 10000, p=[0.95, 0.05]),
            'TransactionDT': np.random.randint(86400, 86400*30, 10000),
            'TransactionAmt': np.random.lognormal(3, 1, 10000),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], 10000)
        })
        
        # 添加更多特徵
        for i in range(1, 21):
            large_data[f'C{i}'] = np.random.randn(10000)
        
        # 監控內存使用
        initial_memory = MemoryProfiler.get_memory_usage()
        
        # 1. 內存優化的數據處理
        optimized_data = MemoryProfiler.optimize_dataframe_memory(large_data)
        
        after_optimization_memory = MemoryProfiler.get_memory_usage()
        
        # 2. 分塊處理
        chunk_processor = ChunkProcessor(chunk_size=2000)
        
        def feature_engineering_chunk(chunk):
            engineer = FeatureEngineer()
            chunk = engineer.create_time_features(chunk)
            chunk = engineer.create_transaction_amount_features(chunk)
            return chunk
        
        processed_data = chunk_processor.process_dataframe_in_chunks(
            optimized_data, feature_engineering_chunk
        )
        
        final_memory = MemoryProfiler.get_memory_usage()
        
        # 3. 驗證內存效率
        # 內存增長應該是合理的
        memory_increase = final_memory['rss_gb'] - initial_memory['rss_gb']
        assert memory_increase < 2.0  # 不應該增加超過2GB
        
        # 最終數據應該完整
        assert len(processed_data) == len(large_data)
        assert len(processed_data.columns) > len(large_data.columns)  # 增加了特徵
        
        print(f"Memory usage - Initial: {initial_memory['rss_gb']:.2f}GB, "
              f"Final: {final_memory['rss_gb']:.2f}GB, "
              f"Increase: {memory_increase:.2f}GB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])