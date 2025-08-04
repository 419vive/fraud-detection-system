"""
機器學習模型模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import sys
from unittest.mock import patch, MagicMock
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

sys.path.append('../src')

from src.modeling import (
    FraudDetectionModel, train_and_evaluate_models
)
from src.exceptions import ModelTrainingError, ModelSaveError, ModelLoadError


class TestFraudDetectionModel:
    """測試詐騙檢測模型類"""
    
    def setup_method(self):
        """測試設置"""
        self.model = FraudDetectionModel()
        
        # 創建測試數據
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'TransactionID': range(1, 1001),
            'isFraud': np.random.choice([0, 1], 1000, p=[0.965, 0.035]),
            'TransactionDT': np.random.randint(86400, 86400*30, 1000),
            'TransactionAmt': np.random.lognormal(3, 1, 1000),
            'ProductCD': np.random.choice(['W', 'C', 'H', 'R', 'S'], 1000),
            'card1': np.random.randint(1000, 20000, 1000),
            'card2': np.random.randint(100, 600, 1000),
            'addr1': np.random.choice(range(100, 500), 1000),
            'C1': np.random.randn(1000),
            'C2': np.random.randn(1000),
            'C3': np.random.randn(1000),
            'C4': np.random.randn(1000),
            'C5': np.random.randn(1000),
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000)
        })
        
        # 準備訓練和測試數據
        self.X_train, self.X_test, self.y_train, self.y_test = self.model.prepare_data(
            self.sample_data, target_col='isFraud', test_size=0.2, random_state=42
        )
    
    def test_initialization(self):
        """測試初始化"""
        assert isinstance(self.model.models, dict)
        assert isinstance(self.model.scalers, dict)
        assert isinstance(self.model.evaluation_results, dict)
        assert isinstance(self.model.feature_importance, dict)
        assert len(self.model.models) == 0
    
    def test_prepare_data_success(self):
        """測試成功準備數據"""
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            self.sample_data, target_col='isFraud', test_size=0.3, random_state=123
        )
        
        # 檢查數據分割結果
        total_samples = len(self.sample_data)
        assert len(X_train) + len(X_test) == total_samples
        assert len(y_train) + len(y_test) == total_samples
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # 檢查測試集大小
        expected_test_size = int(total_samples * 0.3)
        assert abs(len(X_test) - expected_test_size) <= 1  # 允許四捨五入誤差
        
        # 檢查特徵是否為數值型
        assert X_train.select_dtypes(include=[np.number]).shape[1] == X_train.shape[1]
        
        # 檢查不包含目標列和ID列
        assert 'isFraud' not in X_train.columns
        assert 'TransactionID' not in X_train.columns
    
    def test_train_logistic_regression_with_scaling(self):
        """測試訓練邏輯回歸模型（帶縮放）"""
        model = self.model.train_logistic_regression(
            self.X_train, self.y_train, scale_features=True
        )
        
        # 檢查模型是否被訓練和保存
        assert isinstance(model, LogisticRegression)
        assert 'logistic' in self.model.models
        assert 'logistic' in self.model.scalers  # 應該有縮放器
        
        # 檢查模型能否進行預測
        predictions = model.predict(self.model.scalers['logistic'].transform(self.X_test))
        assert len(predictions) == len(self.X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_train_logistic_regression_without_scaling(self):
        """測試訓練邏輯回歸模型（不縮放）"""
        model = self.model.train_logistic_regression(
            self.X_train, self.y_train, scale_features=False
        )
        
        # 檢查模型是否被訓練
        assert isinstance(model, LogisticRegression)
        assert 'logistic' in self.model.models
        assert 'logistic' not in self.model.scalers  # 應該沒有縮放器
    
    def test_train_random_forest(self):
        """測試訓練隨機森林模型"""
        model = self.model.train_random_forest(self.X_train, self.y_train)
        
        # 檢查模型是否被訓練和保存
        assert isinstance(model, RandomForestClassifier)
        assert 'random_forest' in self.model.models
        
        # 檢查特徵重要性是否被保存
        assert 'random_forest' in self.model.feature_importance
        assert len(self.model.feature_importance['random_forest']) == len(self.X_train.columns)
        
        # 檢查模型能否進行預測
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_train_xgboost(self):
        """測試訓練XGBoost模型"""
        model = self.model.train_xgboost(self.X_train, self.y_train)
        
        # 檢查模型是否被訓練和保存
        assert isinstance(model, xgb.XGBClassifier)
        assert 'xgboost' in self.model.models
        
        # 檢查特徵重要性是否被保存
        assert 'xgboost' in self.model.feature_importance
        assert len(self.model.feature_importance['xgboost']) == len(self.X_train.columns)
        
        # 檢查模型能否進行預測
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_train_lightgbm(self):
        """測試訓練LightGBM模型"""
        model = self.model.train_lightgbm(self.X_train, self.y_train)
        
        # 檢查模型是否被訓練和保存
        assert isinstance(model, lgb.LGBMClassifier)
        assert 'lightgbm' in self.model.models
        
        # 檢查特徵重要性是否被保存
        assert 'lightgbm' in self.model.feature_importance
        assert len(self.model.feature_importance['lightgbm']) == len(self.X_train.columns)
        
        # 檢查模型能否進行預測
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_train_catboost(self):
        """測試訓練CatBoost模型"""
        model = self.model.train_catboost(self.X_train, self.y_train)
        
        # 檢查模型是否被訓練和保存
        assert isinstance(model, cb.CatBoostClassifier)
        assert 'catboost' in self.model.models
        
        # 檢查特徵重要性是否被保存
        assert 'catboost' in self.model.feature_importance
        assert len(self.model.feature_importance['catboost']) == len(self.X_train.columns)
        
        # 檢查模型能否進行預測
        predictions = model.predict(self.X_test)
        assert len(predictions) == len(self.X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_evaluate_model_success(self):
        """測試成功評估模型"""
        # 先訓練一個模型
        self.model.train_random_forest(self.X_train, self.y_train)
        
        # 評估模型
        evaluation = self.model.evaluate_model('random_forest', self.X_test, self.y_test)
        
        # 檢查評估結果結構
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'confusion_matrix']
        for metric in expected_metrics:
            assert metric in evaluation
        
        # 檢查指標值的合理性
        assert 0 <= evaluation['accuracy'] <= 1
        assert 0 <= evaluation['precision'] <= 1
        assert 0 <= evaluation['recall'] <= 1
        assert 0 <= evaluation['f1_score'] <= 1
        assert 0 <= evaluation['roc_auc'] <= 1
        
        # 檢查混淆矩陣
        confusion_matrix = evaluation['confusion_matrix']
        assert len(confusion_matrix) == 2
        assert len(confusion_matrix[0]) == 2
        
        # 檢查評估結果是否被保存
        assert 'random_forest' in self.model.evaluation_results
    
    def test_evaluate_model_with_logistic_scaling(self):
        """測試評估帶縮放的邏輯回歸模型"""
        # 訓練帶縮放的邏輯回歸
        self.model.train_logistic_regression(self.X_train, self.y_train, scale_features=True)
        
        # 評估模型
        evaluation = self.model.evaluate_model('logistic', self.X_test, self.y_test)
        
        # 檢查評估結果
        assert 'accuracy' in evaluation
        assert 0 <= evaluation['roc_auc'] <= 1
    
    def test_evaluate_model_not_trained(self):
        """測試評估未訓練的模型"""
        with pytest.raises(ValueError, match="模型 untrained_model 尚未訓練"):
            self.model.evaluate_model('untrained_model', self.X_test, self.y_test)
    
    @patch('matplotlib.pyplot.show')
    def test_plot_feature_importance(self, mock_show):
        """測試繪製特徵重要性"""
        # 訓練模型以獲得特徵重要性
        self.model.train_random_forest(self.X_train, self.y_train)
        
        # 測試繪圖
        self.model.plot_feature_importance('random_forest', top_n=10)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    def test_plot_feature_importance_no_importance(self):
        """測試繪製沒有特徵重要性的模型"""
        # 不訓練模型，直接嘗試繪製
        self.model.plot_feature_importance('nonexistent_model')
        # 應該不會拋出異常，只是發出警告
    
    @patch('matplotlib.pyplot.show')
    def test_plot_roc_curves(self, mock_show):
        """測試繪製ROC曲線"""
        # 訓練幾個模型
        self.model.train_random_forest(self.X_train, self.y_train)
        self.model.train_xgboost(self.X_train, self.y_train)
        
        # 繪製ROC曲線
        self.model.plot_roc_curves(self.X_test, self.y_test)
        
        # 檢查matplotlib.show是否被調用
        assert mock_show.called
    
    def test_get_evaluation_summary_empty(self):
        """測試獲取空評估結果摘要"""
        summary = self.model.get_evaluation_summary()
        
        # 應該返回空DataFrame
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
    
    def test_get_evaluation_summary_with_results(self):
        """測試獲取有結果的評估摘要"""
        # 訓練和評估幾個模型
        self.model.train_random_forest(self.X_train, self.y_train)
        self.model.train_xgboost(self.X_train, self.y_train)
        
        self.model.evaluate_model('random_forest', self.X_test, self.y_test)
        self.model.evaluate_model('xgboost', self.X_test, self.y_test)
        
        # 獲取摘要
        summary = self.model.get_evaluation_summary()
        
        # 檢查摘要結構
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # 兩個模型
        
        expected_columns = ['模型', '準確率', '精確率', '召回率', 'F1分數', 'ROC-AUC']
        for col in expected_columns:
            assert col in summary.columns
        
        # 檢查模型名稱
        model_names = summary['模型'].tolist()
        assert 'random_forest' in model_names
        assert 'xgboost' in model_names
    
    def test_train_all_models(self):
        """測試訓練所有模型"""
        self.model.train_all_models(self.X_train, self.y_train)
        
        # 檢查所有模型是否被訓練
        expected_models = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        for model_name in expected_models:
            assert model_name in self.model.models
        
        # 檢查特徵重要性（除了邏輯回歸）
        for model_name in ['random_forest', 'xgboost', 'lightgbm', 'catboost']:
            assert model_name in self.model.feature_importance
    
    def test_save_model_success(self):
        """測試成功保存模型"""
        # 訓練模型
        self.model.train_random_forest(self.X_train, self.y_train)
        self.model.evaluate_model('random_forest', self.X_test, self.y_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            
            # 保存模型
            self.model.save_model('random_forest', filepath, include_metadata=True)
            
            # 檢查文件是否被創建
            assert os.path.exists(filepath)
            
            # 檢查元數據文件是否被創建
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            assert os.path.exists(metadata_path)
            
            # 檢查元數據內容
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            assert metadata['model_name'] == 'random_forest'
            assert 'save_timestamp' in metadata
            assert 'feature_importance' in metadata
            assert 'evaluation_results' in metadata
    
    def test_save_model_with_scaler(self):
        """測試保存帶縮放器的模型"""
        # 訓練帶縮放的邏輯回歸
        self.model.train_logistic_regression(self.X_train, self.y_train, scale_features=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'logistic_model.pkl')
            
            # 保存模型
            self.model.save_model('logistic', filepath)
            
            # 檢查縮放器文件是否被創建
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            assert os.path.exists(scaler_path)
    
    def test_save_model_not_exists(self):
        """測試保存不存在的模型"""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'nonexistent_model.pkl')
            
            with pytest.raises(ValueError, match="模型 nonexistent_model 不存在"):
                self.model.save_model('nonexistent_model', filepath)
    
    def test_load_model_success(self):
        """測試成功載入模型"""
        # 先訓練和保存模型
        self.model.train_random_forest(self.X_train, self.y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_model.pkl')
            self.model.save_model('random_forest', filepath, include_metadata=True)
            
            # 創建新的模型實例並載入
            new_model = FraudDetectionModel()
            loaded_model = new_model.load_model(filepath, 'loaded_random_forest')
            
            # 檢查模型是否被載入
            assert isinstance(loaded_model, RandomForestClassifier)
            assert 'loaded_random_forest' in new_model.models
            
            # 檢查特徵重要性和評估結果是否被載入
            if 'loaded_random_forest' in new_model.feature_importance:
                assert len(new_model.feature_importance['loaded_random_forest']) > 0
    
    def test_load_model_with_scaler(self):
        """測試載入帶縮放器的模型"""
        # 訓練和保存帶縮放的邏輯回歸
        self.model.train_logistic_regression(self.X_train, self.y_train, scale_features=True)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'logistic_model.pkl')
            self.model.save_model('logistic', filepath)
            
            # 載入模型
            new_model = FraudDetectionModel()
            new_model.load_model(filepath, 'loaded_logistic')
            
            # 檢查縮放器是否被載入
            assert 'loaded_logistic' in new_model.scalers
    
    def test_load_model_file_not_exists(self):
        """測試載入不存在的模型文件"""
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            self.model.load_model('nonexistent_file.pkl')
    
    def test_save_all_models(self):
        """測試保存所有模型"""
        # 訓練幾個模型
        self.model.train_random_forest(self.X_train, self.y_train)
        self.model.train_xgboost(self.X_train, self.y_train)
        
        # 評估模型以生成對比結果
        self.model.evaluate_model('random_forest', self.X_test, self.y_test)
        self.model.evaluate_model('xgboost', self.X_test, self.y_test)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_directory = self.model.save_all_models(temp_dir)
            
            # 檢查保存目錄是否被創建
            assert os.path.exists(save_directory)
            
            # 檢查模型文件是否被創建
            assert os.path.exists(os.path.join(save_directory, 'random_forest.pkl'))
            assert os.path.exists(os.path.join(save_directory, 'xgboost.pkl'))
            
            # 檢查配置文件是否被創建
            assert os.path.exists(os.path.join(save_directory, 'config.json'))
            
            # 檢查對比結果文件是否被創建
            assert os.path.exists(os.path.join(save_directory, 'model_comparison.csv'))
    
    def test_save_all_models_empty(self):
        """測試保存空的模型集合"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 沒有訓練任何模型
            result = self.model.save_all_models(temp_dir)
            
            # 應該返回None或給出警告
            assert result is None
    
    def test_load_model_from_directory(self):
        """測試從目錄載入模型"""
        # 先保存幾個模型
        self.model.train_random_forest(self.X_train, self.y_train)
        self.model.train_xgboost(self.X_train, self.y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_directory = self.model.save_all_models(temp_dir)
            
            # 從目錄載入所有模型
            new_model = FraudDetectionModel()
            available_models = new_model.load_model_from_directory(save_directory)
            
            # 檢查模型是否被載入
            assert 'random_forest' in available_models
            assert 'xgboost' in available_models
            assert 'random_forest' in new_model.models
            assert 'xgboost' in new_model.models
    
    def test_load_specific_model_from_directory(self):
        """測試從目錄載入特定模型"""
        # 先保存幾個模型
        self.model.train_random_forest(self.X_train, self.y_train)
        self.model.train_xgboost(self.X_train, self.y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_directory = self.model.save_all_models(temp_dir)
            
            # 只載入特定模型
            new_model = FraudDetectionModel()
            available_models = new_model.load_model_from_directory(save_directory, 'random_forest')
            
            # 檢查只有指定的模型被載入
            assert 'random_forest' in new_model.models
            assert 'xgboost' not in new_model.models
    
    def test_load_model_from_nonexistent_directory(self):
        """測試從不存在的目錄載入模型"""
        with pytest.raises(FileNotFoundError, match="模型目錄不存在"):
            self.model.load_model_from_directory('nonexistent_directory')


class TestFraudDetectionModelEdgeCases:
    """測試詐騙檢測模型邊界情況"""
    
    def setup_method(self):
        """測試設置"""
        self.model = FraudDetectionModel()
    
    def test_prepare_data_no_target_column(self):
        """測試數據中沒有目標列"""
        data_no_target = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        with pytest.raises(KeyError):
            self.model.prepare_data(data_no_target, target_col='nonexistent_target')
    
    def test_prepare_data_no_numeric_features(self):
        """測試數據中沒有數值型特徵"""
        data_no_numeric = pd.DataFrame({
            'TransactionID': [1, 2, 3, 4, 5],
            'isFraud': [0, 1, 0, 1, 0],
            'categorical1': ['A', 'B', 'C', 'A', 'B'],
            'categorical2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            data_no_numeric, target_col='isFraud'
        )
        
        # 應該沒有特徵（所有非數值特徵被移除）
        assert X_train.shape[1] == 0
        assert X_test.shape[1] == 0
    
    def test_train_with_single_class(self):
        """測試只有單一類別的數據訓練"""
        # 創建只有一個類別的數據
        single_class_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'isFraud': [0] * 100  # 只有類別0
        })
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            single_class_data, target_col='isFraud'
        )
        
        # 嘗試訓練模型（可能會有警告但不應該崩潰）
        try:
            model = self.model.train_random_forest(X_train, y_train)
            assert model is not None
        except Exception as e:
            # 某些算法可能無法處理單類問題
            assert "single class" in str(e).lower() or "stratify" in str(e).lower()
    
    def test_train_with_small_dataset(self):
        """測試小數據集訓練"""
        # 創建非常小的數據集
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [1.1, 2.2, 3.3],
            'isFraud': [0, 1, 0]
        })
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            small_data, target_col='isFraud', test_size=0.33
        )
        
        # 訓練模型
        model = self.model.train_random_forest(X_train, y_train)
        
        # 檢查模型是否能處理小數據集
        assert model is not None
        assert len(X_train) >= 1
        assert len(X_test) >= 1
    
    def test_evaluate_with_perfect_predictions(self):
        """測試完美預測的評估"""
        # 創建簡單的線性可分數據
        perfect_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 10, 11, 12, 13],
            'isFraud': [0, 0, 0, 0, 1, 1, 1, 1]
        })
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            perfect_data, target_col='isFraud'
        )
        
        # 訓練模型
        self.model.train_random_forest(X_train, y_train)
        
        # 評估模型
        evaluation = self.model.evaluate_model('random_forest', X_test, y_test)
        
        # 檢查指標是否合理（可能接近完美）
        assert 0 <= evaluation['accuracy'] <= 1
        assert 0 <= evaluation['roc_auc'] <= 1
    
    def test_train_with_missing_values(self):
        """測試包含缺失值的數據訓練"""
        # 創建包含缺失值的數據
        data_with_nan = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8],
            'feature2': [1.1, np.nan, 3.3, np.nan, 5.5, 6.6, 7.7, 8.8],
            'isFraud': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            data_with_nan, target_col='isFraud'
        )
        
        # 某些算法應該能處理缺失值，某些不能
        try:
            # XGBoost 和 LightGBM 通常能處理缺失值
            model = self.model.train_xgboost(X_train, y_train)
            assert model is not None
        except Exception:
            # 如果不能處理缺失值，應該有相應的錯誤處理
            pass
    
    def test_model_with_zero_variance_features(self):
        """測試包含零方差特徵的模型"""
        # 創建包含常數特徵的數據
        constant_feature_data = pd.DataFrame({
            'constant_feature': [1] * 100,  # 零方差
            'normal_feature': np.random.randn(100),
            'isFraud': np.random.choice([0, 1], 100)
        })
        
        X_train, X_test, y_train, y_test = self.model.prepare_data(
            constant_feature_data, target_col='isFraud'
        )
        
        # 訓練模型（應該能處理常數特徵）
        model = self.model.train_random_forest(X_train, y_train)
        assert model is not None
        
        # 特徵重要性中常數特徵的重要性應該很低或為0
        importance = self.model.feature_importance['random_forest']
        assert 'constant_feature' in importance
        # 常數特徵的重要性應該很低
        assert importance['constant_feature'] <= importance['normal_feature']


class TestFraudDetectionModelPerformance:
    """測試詐騙檢測模型性能"""
    
    def test_training_time_tracking(self):
        """測試訓練時間追踪"""
        # 創建中等大小的數據集
        medium_data = pd.DataFrame({
            'feature1': np.random.randn(5000),
            'feature2': np.random.randn(5000),
            'feature3': np.random.randn(5000),
            'isFraud': np.random.choice([0, 1], 5000, p=[0.95, 0.05])
        })
        
        model = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model.prepare_data(medium_data)
        
        import time
        
        # 測試不同算法的訓練時間
        algorithms = [
            ('random_forest', model.train_random_forest),
            ('xgboost', model.train_xgboost),
            ('lightgbm', model.train_lightgbm)
        ]
        
        training_times = {}
        for name, train_func in algorithms:
            start_time = time.time()
            train_func(X_train, y_train)
            training_times[name] = time.time() - start_time
        
        # 檢查所有算法都能在合理時間內完成
        for name, train_time in training_times.items():
            assert train_time < 60  # 應該在1分鐘內完成
            assert name in model.models
    
    def test_memory_usage_with_large_features(self):
        """測試大量特徵的內存使用"""
        # 創建具有大量特徵的數據
        many_features_data = pd.DataFrame(
            np.random.randn(1000, 50)  # 1000行, 50個特徵
        )
        many_features_data.columns = [f'feature_{i}' for i in range(50)]
        many_features_data['isFraud'] = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        
        model = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model.prepare_data(many_features_data)
        
        # 訓練模型並檢查內存使用
        model.train_random_forest(X_train, y_train)
        
        # 檢查特徵重要性是否包含所有特徵
        importance = model.feature_importance['random_forest']
        assert len(importance) == 50
    
    def test_prediction_speed(self):
        """測試預測速度"""
        # 創建測試數據
        test_data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
            'isFraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
        })
        
        model = FraudDetectionModel()
        X_train, X_test, y_train, y_test = model.prepare_data(test_data)
        
        # 訓練模型
        model.train_random_forest(X_train, y_train)
        
        import time
        
        # 測試預測速度
        start_time = time.time()
        predictions = model.models['random_forest'].predict(X_test)
        prediction_time = time.time() - start_time
        
        # 預測應該很快
        assert prediction_time < 1  # 應該在1秒內完成
        assert len(predictions) == len(X_test)


class TestFraudDetectionModelUtilityFunctions:
    """測試詐騙檢測模型工具函數"""
    
    def test_train_and_evaluate_models_function(self):
        """測試便捷訓練和評估函數"""
        # 創建測試數據
        test_data = pd.DataFrame({
            'TransactionID': range(1, 501),
            'isFraud': np.random.choice([0, 1], 500, p=[0.965, 0.035]),
            'feature1': np.random.randn(500),
            'feature2': np.random.randn(500),
            'feature3': np.random.randn(500)
        })
        
        # 使用便捷函數
        model_trainer = train_and_evaluate_models(test_data, target_col='isFraud')
        
        # 檢查所有模型是否被訓練
        expected_models = ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost']
        for model_name in expected_models:
            assert model_name in model_trainer.models
            assert model_name in model_trainer.evaluation_results
        
        # 檢查評估結果
        summary = model_trainer.get_evaluation_summary()
        assert len(summary) == 5  # 5個模型
        assert '模型' in summary.columns
        assert 'ROC-AUC' in summary.columns


class TestFraudDetectionModelIntegration:
    """測試詐騙檢測模型集成功能"""
    
    def test_complete_workflow(self):
        """測試完整的建模工作流程"""
        # 創建真實場景的測試數據
        realistic_data = pd.DataFrame({
            'TransactionID': range(1, 1001),
            'isFraud': np.random.choice([0, 1], 1000, p=[0.965, 0.035]),
            'TransactionAmt': np.random.lognormal(3, 1, 1000),
            'card1': np.random.randint(1000, 20000, 1000),
            'addr1': np.random.choice(range(100, 500), 1000),
            'C1': np.random.randn(1000),
            'C2': np.random.randn(1000),
            'hour': np.random.randint(0, 24, 1000),
            'day': np.random.randint(0, 7, 1000)
        })
        
        model = FraudDetectionModel()
        
        # 1. 準備數據
        X_train, X_test, y_train, y_test = model.prepare_data(realistic_data)
        
        # 2. 訓練多個模型
        model.train_all_models(X_train, y_train)
        
        # 3. 評估所有模型
        for model_name in model.models.keys():
            model.evaluate_model(model_name, X_test, y_test)
        
        # 4. 獲取評估摘要
        summary = model.get_evaluation_summary()
        
        # 5. 保存最佳模型
        best_model = summary.loc[summary['ROC-AUC'].idxmax(), '模型']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存所有模型
            save_dir = model.save_all_models(temp_dir)
            
            # 載入最佳模型進行驗證
            new_model = FraudDetectionModel()
            new_model.load_model_from_directory(save_dir, best_model)
            
            # 驗證載入的模型能正常預測
            loaded_model_obj = new_model.models[best_model]
            predictions = loaded_model_obj.predict(X_test)
            
            assert len(predictions) == len(X_test)
            assert set(predictions).issubset({0, 1})
        
        # 檢查整個流程的結果
        assert len(summary) == 5  # 5個模型
        assert all(summary['ROC-AUC'] >= 0)
        assert all(summary['ROC-AUC'] <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])