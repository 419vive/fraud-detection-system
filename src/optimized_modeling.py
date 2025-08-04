"""
優化的機器學習模型模組 - IEEE-CIS 詐騙檢測項目
提供高性能的模型訓練和推理功能，包含多種優化策略
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from .config import get_config
from .memory_optimizer import MemoryProfiler, optimize_memory_usage

logger = logging.getLogger(__name__)

class OptimizedFraudDetectionModel:
    """優化的詐騙檢測模型類"""
    
    def __init__(self, config=None):
        self.models = {}
        self.evaluation_results = {}
        self.feature_importance = {}
        self.preprocessing_pipeline = {}
        self.training_history = {}
        self.inference_cache = {}
        self.config = config or get_config()
        
        # 性能優化設置
        self.enable_early_stopping = True
        self.enable_feature_selection = True
        self.enable_model_ensemble = True
        self.enable_cache = True
        
        # 並行處理設置
        self.n_jobs = min(-1, psutil.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'isFraud', 
                    test_size: float = 0.2, validation_size: float = 0.1,
                    optimize_memory: bool = True, feature_selection: bool = True) -> Tuple:
        """優化的數據準備流程"""
        logger.info("開始優化數據準備流程...")
        start_time = time.time()
        
        # 內存優化
        if optimize_memory:
            df = optimize_memory_usage(df)
        
        # 特徵和目標分離
        drop_cols = [target_col, 'TransactionID'] + [col for col in df.columns if 'ID' in col.upper()]
        X = df.drop(columns=drop_cols, errors='ignore')
        y = df[target_col]
        
        # 只保留數值型特徵（更高效）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # 處理缺失值（高效版本）
        X = self._fast_missing_value_imputation(X)
        
        # 智能特徵選擇
        if feature_selection and self.enable_feature_selection:
            X = self._intelligent_feature_selection(X, y)
        
        # 三分割：訓練/驗證/測試
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # 標準化（只對需要的模型）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.preprocessing_pipeline['scaler'] = scaler
        self.preprocessing_pipeline['feature_names'] = X.columns.tolist()
        
        prep_time = time.time() - start_time
        logger.info(f"數據準備完成 - 訓練集: {X_train.shape}, 驗證集: {X_val.shape}, 測試集: {X_test.shape}")
        logger.info(f"準備時間: {prep_time:.2f}秒")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test, 
                X_train_scaled, X_val_scaled, X_test_scaled)
    
    def _fast_missing_value_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """快速缺失值填補"""
        # 使用median填補（比mean更robust且更快）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        return X
    
    def _intelligent_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                     max_features: int = 200) -> pd.DataFrame:
        """智能特徵選擇"""
        logger.info("執行智能特徵選擇...")
        
        # 移除低方差特徵
        variance_threshold = 0.01
        high_variance_features = X.var() > variance_threshold
        X_filtered = X.loc[:, high_variance_features]
        
        # 使用輕量級模型進行特徵選擇
        if len(X_filtered.columns) > max_features:
            lgb_selector = lgb.LGBMClassifier(
                n_estimators=50, random_state=42, verbosity=-1,
                n_jobs=self.n_jobs
            )
            
            # 快速訓練用於特徵選擇
            sample_size = min(50000, len(X_filtered))
            sample_idx = np.random.choice(len(X_filtered), sample_size, replace=False)
            
            lgb_selector.fit(X_filtered.iloc[sample_idx], y.iloc[sample_idx])
            
            # 選擇重要特徵
            selector = SelectFromModel(lgb_selector, max_features=max_features, prefit=True)
            X_selected = pd.DataFrame(
                selector.transform(X_filtered),
                columns=X_filtered.columns[selector.get_support()],
                index=X_filtered.index
            )
            
            logger.info(f"特徵選擇完成: {len(X_filtered.columns)} -> {len(X_selected.columns)}")
            return X_selected
        
        return X_filtered
    
    def train_optimized_lightgbm(self, X_train, y_train, X_val=None, y_val=None):
        """訓練優化的LightGBM模型"""
        logger.info("開始訓練優化的LightGBM模型...")
        start_time = time.time()
        
        # LightGBM通常是最快的gradient boosting實現
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=self.n_jobs,
            verbosity=-1,
            early_stopping_rounds=50 if self.enable_early_stopping else None
        )
        
        # 使用早停機制
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='auc'
        )
        
        # 記錄特徵重要性
        self.feature_importance['lightgbm'] = dict(
            zip(self.preprocessing_pipeline.get('feature_names', []), 
                model.feature_importances_)
        )
        
        training_time = time.time() - start_time
        self.training_history['lightgbm'] = {
            'training_time': training_time,
            'best_iteration': getattr(model, 'best_iteration_', model.n_estimators),
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        self.models['lightgbm'] = model
        logger.info(f"LightGBM模型訓練完成，耗時: {training_time:.2f}秒")
        return model
    
    def train_optimized_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """訓練優化的XGBoost模型"""
        logger.info("開始訓練優化的XGBoost模型...")
        start_time = time.time()
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # 優化的XGBoost參數
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=self.n_jobs,
            tree_method='hist',  # 更快的訓練
            eval_metric='auc',
            early_stopping_rounds=50 if self.enable_early_stopping else None
        )
        
        # 使用早停機制
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # 記錄特徵重要性
        self.feature_importance['xgboost'] = model.get_booster().get_score(importance_type='weight')
        
        training_time = time.time() - start_time
        self.training_history['xgboost'] = {
            'training_time': training_time,
            'best_iteration': getattr(model, 'best_iteration', model.n_estimators),
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        self.models['xgboost'] = model
        logger.info(f"XGBoost模型訓練完成，耗時: {training_time:.2f}秒")
        return model
    
    def train_optimized_catboost(self, X_train, y_train, X_val=None, y_val=None):
        """訓練優化的CatBoost模型"""
        logger.info("開始訓練優化的CatBoost模型...")
        start_time = time.time()
        
        model = cb.CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            class_weights=[1, (y_train == 0).sum() / (y_train == 1).sum()],
            random_seed=42,
            thread_count=self.n_jobs,
            verbose=False,
            early_stopping_rounds=50 if self.enable_early_stopping else None
        )
        
        # 使用早停機制
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            use_best_model=True if eval_set else False
        )
        
        # 記錄特徵重要性
        self.feature_importance['catboost'] = dict(
            zip(self.preprocessing_pipeline.get('feature_names', []), 
                model.feature_importances_)
        )
        
        training_time = time.time() - start_time
        self.training_history['catboost'] = {
            'training_time': training_time,
            'best_iteration': getattr(model, 'best_iteration_', model.tree_count_),
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        self.models['catboost'] = model
        logger.info(f"CatBoost模型訓練完成，耗時: {training_time:.2f}秒")
        return model
    
    def train_optimized_random_forest(self, X_train, y_train, X_val=None, y_val=None):
        """訓練優化的隨機森林模型"""
        logger.info("開始訓練優化的隨機森林模型...")
        start_time = time.time()
        
        # 優化的隨機森林參數
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=self.n_jobs,
            warm_start=True  # 支持增量訓練
        )
        
        model.fit(X_train, y_train)
        
        # 記錄特徵重要性
        self.feature_importance['random_forest'] = dict(
            zip(self.preprocessing_pipeline.get('feature_names', []), 
                model.feature_importances_)
        )
        
        training_time = time.time() - start_time
        self.training_history['random_forest'] = {
            'training_time': training_time,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        self.models['random_forest'] = model
        logger.info(f"隨機森林模型訓練完成，耗時: {training_time:.2f}秒")
        return model
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val):
        """訓練集成模型"""
        if not self.enable_model_ensemble:
            return None
        
        logger.info("訓練集成模型...")
        start_time = time.time()
        
        # 並行訓練基礎模型
        with self.thread_pool as executor:
            futures = {
                executor.submit(self.train_optimized_lightgbm, X_train, y_train, X_val, y_val): 'lightgbm',
                executor.submit(self.train_optimized_xgboost, X_train, y_train, X_val, y_val): 'xgboost',
                executor.submit(self.train_optimized_catboost, X_train, y_train, X_val, y_val): 'catboost'
            }
            
            for future in futures:
                try:
                    future.result(timeout=300)  # 5分鐘超時
                except Exception as e:
                    logger.error(f"模型 {futures[future]} 訓練失敗: {e}")
        
        # 簡單的集成策略：加權平均
        ensemble_weights = self._calculate_ensemble_weights(X_val, y_val)
        self.models['ensemble'] = {
            'models': ['lightgbm', 'xgboost', 'catboost'],
            'weights': ensemble_weights
        }
        
        training_time = time.time() - start_time
        logger.info(f"集成模型訓練完成，耗時: {training_time:.2f}秒")
        
        return self.models['ensemble']
    
    def _calculate_ensemble_weights(self, X_val, y_val) -> List[float]:
        """計算集成權重"""
        model_names = ['lightgbm', 'xgboost', 'catboost']
        scores = []
        
        for model_name in model_names:
            if model_name in self.models:
                try:
                    y_pred = self.models[model_name].predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
                    scores.append(score)
                except:
                    scores.append(0.5)  # 默認分數
            else:
                scores.append(0.5)
        
        # 歸一化權重
        total_score = sum(scores)
        weights = [score / total_score for score in scores] if total_score > 0 else [1/len(scores)] * len(scores)
        
        return weights
    
    def evaluate_model(self, model_name: str, X_test, y_test, detailed: bool = True) -> Dict[str, Any]:
        """綜合模型評估"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未找到")
        
        logger.info(f"評估模型: {model_name}")
        start_time = time.time()
        
        model = self.models[model_name]
        
        # 預測
        if model_name == 'ensemble':
            y_pred_proba = self.predict_ensemble(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 基本指標
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'inference_time': time.time() - start_time,
            'prediction_speed': len(X_test) / (time.time() - start_time)
        }
        
        if detailed and model_name != 'ensemble':
            # 詳細評估
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            
            # 找到最佳閾值
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            results.update({
                'optimal_threshold': optimal_threshold,
                'fpr': fpr.tolist()[:100],  # 限制長度以節省內存
                'tpr': tpr.tolist()[:100],
                'precision_curve': precision.tolist()[:100],
                'recall_curve': recall.tolist()[:100],
                'feature_importance_top10': self._get_top_features(model_name, 10)
            })
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"{model_name} 評估完成 - AUC: {results['roc_auc']:.4f}, F1: {results['f1_score']:.4f}")
        logger.info(f"推理速度: {results['prediction_speed']:.0f} predictions/sec")
        
        return results
    
    def _get_top_features(self, model_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """獲取最重要的特徵"""
        if model_name not in self.feature_importance:
            return []
        
        importance_dict = self.feature_importance[model_name]
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:top_k]
    
    def fast_predict(self, X: pd.DataFrame, model_name: str = 'best') -> np.ndarray:
        """快速預測（針對生產環境優化）"""
        if model_name == 'best':
            model_name = self._get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 未找到")
        
        # 使用緩存
        cache_key = f"{model_name}_{hash(X.values.tobytes())}"
        if self.enable_cache and cache_key in self.inference_cache:
            return self.inference_cache[cache_key]
        
        # 預處理
        X_processed = self._preprocess_for_inference(X)
        
        # 預測
        if model_name == 'ensemble':
            predictions = self.predict_ensemble(X_processed)
        else:
            model = self.models[model_name]
            predictions = model.predict_proba(X_processed)[:, 1]
        
        # 緩存結果
        if self.enable_cache and len(self.inference_cache) < 100:  # 限制緩存大小
            self.inference_cache[cache_key] = predictions
        
        return predictions
    
    def _preprocess_for_inference(self, X: pd.DataFrame) -> np.ndarray:
        """推理時的預處理"""
        # 選擇特徵
        if 'feature_names' in self.preprocessing_pipeline:
            feature_names = self.preprocessing_pipeline['feature_names']
            X = X[feature_names]
        
        # 缺失值處理
        X = X.fillna(X.median())
        
        # 標準化（如果需要）
        if 'scaler' in self.preprocessing_pipeline:
            X = self.preprocessing_pipeline['scaler'].transform(X)
        
        return X
    
    def _get_best_model(self) -> str:
        """獲取最佳模型名稱"""
        if not self.evaluation_results:
            return list(self.models.keys())[0] if self.models else None
        
        best_model = max(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['roc_auc'])
        return best_model
    
    def predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """集成模型預測"""
        if 'ensemble' not in self.models:
            raise ValueError("集成模型未訓練")
        
        ensemble_info = self.models['ensemble']
        model_names = ensemble_info['models']
        weights = ensemble_info['weights']
        
        predictions = []
        for model_name, weight in zip(model_names, weights):
            if model_name in self.models:
                pred = self.models[model_name].predict_proba(X)[:, 1]
                predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0) if predictions else np.zeros(len(X))
    
    def save_models(self, directory: str):
        """保存所有模型"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name != 'ensemble':  # 集成模型單獨處理
                model_path = os.path.join(directory, f'{model_name}.joblib')
                joblib.dump(model, model_path)
                logger.info(f"模型 {model_name} 已保存至 {model_path}")
        
        # 保存預處理管道
        pipeline_path = os.path.join(directory, 'preprocessing_pipeline.joblib')
        joblib.dump(self.preprocessing_pipeline, pipeline_path)
        
        # 保存集成信息
        if 'ensemble' in self.models:
            ensemble_path = os.path.join(directory, 'ensemble_info.joblib')
            joblib.dump(self.models['ensemble'], ensemble_path)
    
    def load_models(self, directory: str):
        """載入所有模型"""
        import os
        
        # 載入預處理管道
        pipeline_path = os.path.join(directory, 'preprocessing_pipeline.joblib')
        if os.path.exists(pipeline_path):
            self.preprocessing_pipeline = joblib.load(pipeline_path)
        
        # 載入模型
        for model_file in os.listdir(directory):
            if model_file.endswith('.joblib') and model_file != 'preprocessing_pipeline.joblib':
                model_name = model_file.replace('.joblib', '')
                if model_name == 'ensemble_info':
                    self.models['ensemble'] = joblib.load(os.path.join(directory, model_file))
                else:
                    model_path = os.path.join(directory, model_file)
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"模型 {model_name} 已載入")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """獲取性能摘要"""
        summary = {
            'models_trained': list(self.models.keys()),
            'training_history': self.training_history,
            'evaluation_results': self.evaluation_results,
            'best_model': self._get_best_model(),
            'total_training_time': sum(
                history.get('training_time', 0) 
                for history in self.training_history.values()
            )
        }
        
        if self.evaluation_results:
            best_model_name = self._get_best_model()
            if best_model_name:
                summary['best_performance'] = self.evaluation_results[best_model_name]
        
        return summary
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        gc.collect()

def train_optimized_models(df: pd.DataFrame, enable_ensemble: bool = True, 
                          config: Dict = None) -> OptimizedFraudDetectionModel:
    """訓練優化的模型集合"""
    logger.info("開始訓練優化的詐騙檢測模型...")
    
    model_trainer = OptimizedFraudDetectionModel(config)
    model_trainer.enable_model_ensemble = enable_ensemble
    
    # 準備數據
    data_splits = model_trainer.prepare_data(df, feature_selection=True, optimize_memory=True)
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_scaled, X_val_scaled, X_test_scaled = data_splits
    
    # 訓練模型
    if enable_ensemble:
        model_trainer.train_ensemble_model(X_train, y_train, X_val, y_val)
    else:
        # 單獨訓練各個模型
        model_trainer.train_optimized_lightgbm(X_train, y_train, X_val, y_val)
        model_trainer.train_optimized_xgboost(X_train, y_train, X_val, y_val)
        model_trainer.train_optimized_random_forest(X_train, y_train, X_val, y_val)
    
    # 評估所有模型
    for model_name in model_trainer.models.keys():
        if model_name != 'ensemble':
            model_trainer.evaluate_model(model_name, X_test, y_test)
    
    # 評估集成模型
    if enable_ensemble and 'ensemble' in model_trainer.models:
        try:
            model_trainer.evaluate_model('ensemble', X_test, y_test)
        except Exception as e:
            logger.error(f"集成模型評估失敗: {e}")
    
    # 性能摘要
    summary = model_trainer.get_performance_summary()
    logger.info(f"訓練完成 - 最佳模型: {summary['best_model']}")
    logger.info(f"總訓練時間: {summary['total_training_time']:.2f}秒")
    
    return model_trainer

# 保持向後兼容
FraudDetectionModel = OptimizedFraudDetectionModel
train_models = train_optimized_models

if __name__ == "__main__":
    print("優化的機器學習模型模組已載入完成！")