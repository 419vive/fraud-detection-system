"""
機器學習模型模組 - IEEE-CIS 詐騙檢測項目
包含多種機器學習算法、模型訓練、評估和調優功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import joblib
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """詐騙檢測模型類別"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.evaluation_results = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'isFraud', 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """準備訓練和測試數據"""
        # 分離特徵和目標變數
        X = df.drop(columns=[target_col, 'TransactionID'], errors='ignore')
        y = df[target_col]
        
        # 確保所有特徵都是數值型
        X = X.select_dtypes(include=[np.number])
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"訓練集: {X_train.shape}, 測試集: {X_test.shape}")
        logger.info(f"訓練集詐騙率: {y_train.mean():.4f}, 測試集詐騙率: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 scale_features: bool = True) -> LogisticRegression:
        """訓練邏輯迴歸模型"""
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers['logistic'] = scaler
        else:
            X_train_scaled = X_train
        
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        self.models['logistic'] = model
        logger.info("邏輯迴歸模型訓練完成")
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """訓練隨機森林模型"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        # 保存特徵重要性
        self.feature_importance['random_forest'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        logger.info("隨機森林模型訓練完成")
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """訓練 XGBoost 模型"""
        # 計算類別權重
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc'
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        # 保存特徵重要性
        self.feature_importance['xgboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        logger.info("XGBoost 模型訓練完成")
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
        """訓練 LightGBM 模型"""
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        
        # 保存特徵重要性
        self.feature_importance['lightgbm'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        logger.info("LightGBM 模型訓練完成")
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> cb.CatBoostClassifier:
        """訓練 CatBoost 模型"""
        # 計算類別權重
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            class_weights=[1, scale_pos_weight],
            random_state=42,
            verbose=False
        )
        
        model.fit(X_train, y_train)
        self.models['catboost'] = model
        
        # 保存特徵重要性
        self.feature_importance['catboost'] = dict(zip(
            X_train.columns, model.feature_importances_
        ))
        
        logger.info("CatBoost 模型訓練完成")
        return model
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """評估模型性能"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 尚未訓練")
        
        model = self.models[model_name]
        
        # 預測
        if model_name == 'logistic' and model_name in self.scalers:
            X_test_scaled = self.scalers[model_name].transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 計算評估指標
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.evaluation_results[model_name] = evaluation
        
        logger.info(f"{model_name} 模型評估完成")
        logger.info(f"ROC-AUC: {evaluation['roc_auc']:.4f}")
        logger.info(f"F1-Score: {evaluation['f1_score']:.4f}")
        
        return evaluation
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20):
        """繪製特徵重要性圖"""
        if model_name not in self.feature_importance:
            logger.warning(f"模型 {model_name} 沒有特徵重要性信息")
            return
        
        importance = self.feature_importance[model_name]
        
        # 排序並選取前N個特徵
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, scores = zip(*sorted_importance)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('特徵重要性')
        plt.title(f'{model_name} 模型 - Top {top_n} 重要特徵')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series):
        """繪製所有模型的ROC曲線"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            # 預測概率
            if model_name == 'logistic' and model_name in self.scalers:
                X_test_scaled = self.scalers[model_name].transform(X_test)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 計算ROC曲線
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('假陽性率 (False Positive Rate)')
        plt.ylabel('真陽性率 (True Positive Rate)')
        plt.title('ROC 曲線比較')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """獲取所有模型的評估結果摘要"""
        if not self.evaluation_results:
            logger.warning("沒有可用的評估結果")
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                '模型': model_name,
                '準確率': results['accuracy'],
                '精確率': results['precision'],
                '召回率': results['recall'],
                'F1分數': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            })
        
        return pd.DataFrame(summary_data).round(4)
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """訓練所有模型"""
        logger.info("開始訓練所有模型...")
        
        # 訓練各種模型
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        self.train_catboost(X_train, y_train)
        
        logger.info("所有模型訓練完成")
    
    def save_model(self, model_name: str, filepath: str, include_metadata: bool = True):
        """保存模型到文件"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        
        model = self.models[model_name]
        
        # 創建目錄（如果不存在）
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存模型
        if model_name in ['xgboost', 'lightgbm', 'catboost']:
            # 使用模型自帶的保存方法
            if model_name == 'xgboost':
                model.save_model(filepath.replace('.pkl', '.json'))
            elif model_name == 'lightgbm':
                model.booster_.save_model(filepath.replace('.pkl', '.txt'))
            elif model_name == 'catboost':
                model.save_model(filepath.replace('.pkl', '.cbm'))
        
        # 使用joblib保存（通用方法）
        joblib.dump(model, filepath)
        
        # 保存元數據
        if include_metadata:
            metadata = {
                'model_name': model_name,
                'model_type': str(type(model)),
                'save_timestamp': datetime.now().isoformat(),
                'feature_importance': self.feature_importance.get(model_name, {}),
                'evaluation_results': self.evaluation_results.get(model_name, {}),
                'scaler_info': model_name in self.scalers
            }
            
            metadata_path = filepath.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存縮放器（如果存在）
        if model_name in self.scalers:
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            joblib.dump(self.scalers[model_name], scaler_path)
        
        logger.info(f"模型 {model_name} 已保存至 {filepath}")
    
    def load_model(self, filepath: str, model_name: str = None):
        """從文件載入模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 載入模型
        model = joblib.load(filepath)
        
        # 推斷模型名稱
        if model_name is None:
            if 'xgb' in str(type(model)).lower():
                model_name = 'xgboost'
            elif 'lgb' in str(type(model)).lower():
                model_name = 'lightgbm'
            elif 'catboost' in str(type(model)).lower():
                model_name = 'catboost'
            elif 'randomforest' in str(type(model)).lower():
                model_name = 'random_forest'
            elif 'logistic' in str(type(model)).lower():
                model_name = 'logistic'
            else:
                model_name = 'loaded_model'
        
        self.models[model_name] = model
        
        # 載入元數據
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            if 'feature_importance' in metadata:
                self.feature_importance[model_name] = metadata['feature_importance']
            
            if 'evaluation_results' in metadata:
                self.evaluation_results[model_name] = metadata['evaluation_results']
        
        # 載入縮放器
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scalers[model_name] = joblib.load(scaler_path)
        
        logger.info(f"模型 {model_name} 已從 {filepath} 載入")
        return model
    
    def save_all_models(self, base_directory: str = "models"):
        """保存所有訓練好的模型"""
        if not self.models:
            logger.warning("沒有可保存的模型")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = os.path.join(base_directory, f"fraud_detection_{timestamp}")
        os.makedirs(save_directory, exist_ok=True)
        
        for model_name in self.models.keys():
            filepath = os.path.join(save_directory, f"{model_name}.pkl")
            self.save_model(model_name, filepath)
        
        # 保存模型對比結果
        if self.evaluation_results:
            summary_df = self.get_evaluation_summary()
            summary_path = os.path.join(save_directory, "model_comparison.csv")
            summary_df.to_csv(summary_path, index=False)
        
        # 保存配置信息
        config = {
            'save_timestamp': datetime.now().isoformat(),
            'models_saved': list(self.models.keys()),
            'save_directory': save_directory
        }
        
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"所有模型已保存至目錄: {save_directory}")
        return save_directory
    
    def load_model_from_directory(self, model_directory: str, model_name: str = None):
        """從目錄載入模型"""
        if not os.path.exists(model_directory):
            raise FileNotFoundError(f"模型目錄不存在: {model_directory}")
        
        # 載入配置
        config_path = os.path.join(model_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            available_models = config.get('models_saved', [])
        else:
            # 掃描目錄中的模型文件
            available_models = []
            for file in os.listdir(model_directory):
                if file.endswith('.pkl') and not file.endswith('_scaler.pkl'):
                    available_models.append(file.replace('.pkl', ''))
        
        if model_name is None:
            # 載入所有模型
            for model in available_models:
                model_path = os.path.join(model_directory, f"{model}.pkl")
                if os.path.exists(model_path):
                    self.load_model(model_path, model)
            logger.info(f"已載入 {len(available_models)} 個模型")
        else:
            # 載入特定模型
            if model_name not in available_models:
                raise ValueError(f"模型 {model_name} 在目錄中不存在。可用模型: {available_models}")
            
            model_path = os.path.join(model_directory, f"{model_name}.pkl")
            self.load_model(model_path, model_name)
        
        return available_models

def train_and_evaluate_models(df: pd.DataFrame, target_col: str = 'isFraud') -> FraudDetectionModel:
    """便捷函數：訓練和評估所有模型"""
    model_trainer = FraudDetectionModel()
    
    # 準備數據
    X_train, X_test, y_train, y_test = model_trainer.prepare_data(df, target_col)
    
    # 訓練所有模型
    model_trainer.train_all_models(X_train, y_train)
    
    # 評估所有模型
    for model_name in model_trainer.models.keys():
        model_trainer.evaluate_model(model_name, X_test, y_test)
    
    return model_trainer

if __name__ == "__main__":
    # 測試代碼
    model_trainer = FraudDetectionModel()
    print("機器學習模型模組已載入完成！") 