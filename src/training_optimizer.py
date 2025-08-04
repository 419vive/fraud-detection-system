"""
訓練優化模組 - IEEE-CIS 詐騙檢測項目
提供高效的模型訓練策略、自動調參和訓練加速功能
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import gc
import psutil
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 機器學習庫
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 嘗試導入加速庫
try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .config import get_config
from .memory_optimizer import MemoryProfiler, memory_monitor

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """訓練指標數據類"""
    model_name: str
    training_time: float
    validation_score: float
    best_iteration: int
    memory_peak: float
    cpu_usage: float
    hyperparameters: Dict[str, Any]
    optimization_method: str

class EarlyStoppingCallback:
    """自定義早停回調"""
    
    def __init__(self, patience: int = 50, min_delta: float = 1e-4, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = -np.inf
        self.best_iteration = 0
        self.wait = 0
        self.stopped_iteration = 0
        self.best_weights = None
    
    def __call__(self, iteration: int, score: float, model=None) -> bool:
        """檢查是否應該早停"""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_iteration = iteration
            self.wait = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = self._get_model_weights(model)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_iteration = iteration
                if self.restore_best_weights and self.best_weights is not None:
                    self._set_model_weights(model, self.best_weights)
                return True
        
        return False
    
    def _get_model_weights(self, model):
        """獲取模型權重（簡化版本）"""
        # 這裡可以根據具體模型類型實現權重獲取
        return None
    
    def _set_model_weights(self, model, weights):
        """設置模型權重（簡化版本）"""
        # 這裡可以根據具體模型類型實現權重設置
        pass

class HyperparameterOptimizer:
    """超參數優化器"""
    
    def __init__(self, config=None, n_trials: int = 100, timeout: int = 3600):
        self.config = config or get_config()
        self.n_trials = n_trials
        self.timeout = timeout
        self.optimization_history = []
        
        # 設置優化器
        self.study = None
        self.best_params = {}
        
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, 
                        scoring_metric: str = 'roc_auc') -> Dict[str, Any]:
        """XGBoost超參數優化"""
        logger.info("開始XGBoost超參數優化...")
        
        def objective(trial):
            # 定義搜索空間
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist'
            }
            
            # 訓練模型
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # 評估
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            return score
        
        # 創建研究
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        logger.info(f"XGBoost最佳參數: {best_params}")
        logger.info(f"XGBoost最佳分數: {best_score:.4f}")
        
        self.best_params['xgboost'] = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val, 
                         scoring_metric: str = 'roc_auc') -> Dict[str, Any]:
        """LightGBM超參數優化"""
        logger.info("開始LightGBM超參數優化...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=50
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            return score
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        logger.info(f"LightGBM最佳參數: {best_params}")
        logger.info(f"LightGBM最佳分數: {best_score:.4f}")
        
        self.best_params['lightgbm'] = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val, 
                         scoring_metric: str = 'roc_auc') -> Dict[str, Any]:
        """CatBoost超參數優化"""
        logger.info("開始CatBoost超參數優化...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
                'class_weights': [1, (y_train == 0).sum() / (y_train == 1).sum()],
                'random_seed': 42,
                'thread_count': -1,
                'verbose': False
            }
            
            model = cb.CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                use_best_model=True
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            
            return score
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        logger.info(f"CatBoost最佳參數: {best_params}")
        logger.info(f"CatBoost最佳分數: {best_score:.4f}")
        
        self.best_params['catboost'] = best_params
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(self.study.trials)
        }
    
    def optimize_all_models(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """優化所有模型的超參數"""
        logger.info("開始優化所有模型的超參數...")
        start_time = time.time()
        
        results = {}
        
        # 並行優化不同模型
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_xgb = executor.submit(
                self.optimize_xgboost, X_train, y_train, X_val, y_val
            )
            future_lgb = executor.submit(
                self.optimize_lightgbm, X_train, y_train, X_val, y_val
            )
            future_cb = executor.submit(
                self.optimize_catboost, X_train, y_train, X_val, y_val
            )
            
            try:
                results['xgboost'] = future_xgb.result(timeout=self.timeout)
            except Exception as e:
                logger.error(f"XGBoost優化失敗: {e}")
                results['xgboost'] = {'error': str(e)}
            
            try:
                results['lightgbm'] = future_lgb.result(timeout=self.timeout)
            except Exception as e:
                logger.error(f"LightGBM優化失敗: {e}")
                results['lightgbm'] = {'error': str(e)}
            
            try:
                results['catboost'] = future_cb.result(timeout=self.timeout)
            except Exception as e:
                logger.error(f"CatBoost優化失敗: {e}")
                results['catboost'] = {'error': str(e)}
        
        total_time = time.time() - start_time
        results['optimization_summary'] = {
            'total_time': total_time,
            'best_model': self._get_best_model(results),
            'all_best_params': self.best_params
        }
        
        logger.info(f"超參數優化完成，總耗時: {total_time:.2f}秒")
        return results
    
    def _get_best_model(self, results: Dict[str, Any]) -> str:
        """獲取最佳模型"""
        best_score = -1
        best_model = None
        
        for model_name, result in results.items():
            if isinstance(result, dict) and 'best_score' in result:
                if result['best_score'] > best_score:
                    best_score = result['best_score']
                    best_model = model_name
        
        return best_model

class TrainingAccelerator:
    """訓練加速器"""
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.cpu_count = mp.cpu_count()
        self.training_metrics = []
    
    @memory_monitor()
    def accelerated_training(self, model_configs: Dict[str, Dict], 
                           X_train, y_train, X_val, y_val,
                           early_stopping_patience: int = 50) -> Dict[str, Any]:
        """加速模型訓練"""
        logger.info("開始加速模型訓練...")
        start_time = time.time()
        
        trained_models = {}
        training_results = {}
        
        # 並行訓練多個模型
        if self.enable_parallel and len(model_configs) > 1:
            logger.info("使用並行訓練")
            trained_models = self._parallel_training(
                model_configs, X_train, y_train, X_val, y_val, early_stopping_patience
            )
        else:
            logger.info("使用順序訓練")
            trained_models = self._sequential_training(
                model_configs, X_train, y_train, X_val, y_val, early_stopping_patience
            )
        
        # 評估所有模型
        for model_name, model in trained_models.items():
            if model is not None:
                try:
                    y_pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
                    
                    training_results[model_name] = {
                        'model': model,
                        'validation_score': score,
                        'training_completed': True
                    }
                except Exception as e:
                    logger.error(f"模型 {model_name} 評估失敗: {e}")
                    training_results[model_name] = {
                        'model': None,
                        'validation_score': 0,
                        'training_completed': False,
                        'error': str(e)
                    }
        
        total_time = time.time() - start_time
        
        # 找出最佳模型
        best_model_name = max(
            training_results.keys(), 
            key=lambda x: training_results[x]['validation_score']
        )
        
        summary = {
            'trained_models': training_results,
            'best_model': best_model_name,
            'best_score': training_results[best_model_name]['validation_score'],
            'total_training_time': total_time,
            'training_metrics': self.training_metrics
        }
        
        logger.info(f"加速訓練完成 - 最佳模型: {best_model_name}")
        logger.info(f"最佳分數: {summary['best_score']:.4f}")
        logger.info(f"總訓練時間: {total_time:.2f}秒")
        
        return summary
    
    def _parallel_training(self, model_configs, X_train, y_train, X_val, y_val, patience):
        """並行訓練"""
        def train_single_model(args):
            model_name, config = args
            return model_name, self._train_single_model(
                model_name, config, X_train, y_train, X_val, y_val, patience
            )
        
        trained_models = {}
        
        # 使用線程池而不是進程池，避免模型序列化問題
        with ThreadPoolExecutor(max_workers=min(len(model_configs), 4)) as executor:
            futures = {
                executor.submit(train_single_model, item): item[0] 
                for item in model_configs.items()
            }
            
            for future in futures:
                try:
                    model_name, model = future.result(timeout=1800)  # 30分鐘超時
                    trained_models[model_name] = model
                except Exception as e:
                    model_name = futures[future]
                    logger.error(f"並行訓練模型 {model_name} 失敗: {e}")
                    trained_models[model_name] = None
        
        return trained_models
    
    def _sequential_training(self, model_configs, X_train, y_train, X_val, y_val, patience):
        """順序訓練"""
        trained_models = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"訓練模型: {model_name}")
            try:
                model = self._train_single_model(
                    model_name, config, X_train, y_train, X_val, y_val, patience
                )
                trained_models[model_name] = model
            except Exception as e:
                logger.error(f"模型 {model_name} 訓練失敗: {e}")
                trained_models[model_name] = None
        
        return trained_models
    
    def _train_single_model(self, model_name, config, X_train, y_train, X_val, y_val, patience):
        """訓練單個模型"""
        start_time = time.time()
        initial_memory = MemoryProfiler.get_memory_usage()['rss_gb']
        
        try:
            if model_name.lower() == 'lightgbm':
                model = lgb.LGBMClassifier(**config)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    early_stopping_rounds=patience
                )
                best_iteration = getattr(model, 'best_iteration_', model.n_estimators)
                
            elif model_name.lower() == 'xgboost':
                model = xgb.XGBClassifier(**config)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=patience,
                    verbose=False
                )
                best_iteration = getattr(model, 'best_iteration', model.n_estimators)
                
            elif model_name.lower() == 'catboost':
                model = cb.CatBoostClassifier(**config)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=patience,
                    use_best_model=True
                )
                best_iteration = getattr(model, 'best_iteration_', model.tree_count_)
                
            else:
                raise ValueError(f"不支持的模型類型: {model_name}")
            
            # 評估模型
            y_pred = model.predict_proba(X_val)[:, 1]
            validation_score = roc_auc_score(y_val, y_pred)
            
            # 記錄訓練指標
            training_time = time.time() - start_time
            peak_memory = MemoryProfiler.get_memory_usage()['rss_gb']
            
            metric = TrainingMetrics(
                model_name=model_name,
                training_time=training_time,
                validation_score=validation_score,
                best_iteration=best_iteration,
                memory_peak=peak_memory - initial_memory,
                cpu_usage=psutil.cpu_percent(),
                hyperparameters=config,
                optimization_method='accelerated'
            )
            
            self.training_metrics.append(metric)
            
            logger.info(f"{model_name} 訓練完成 - 分數: {validation_score:.4f}, 耗時: {training_time:.2f}秒")
            
            return model
            
        except Exception as e:
            logger.error(f"模型 {model_name} 訓練失敗: {e}")
            return None

class AutoMLPipeline:
    """自動化機器學習管道"""
    
    def __init__(self, config=None, time_budget: int = 3600):
        self.config = config or get_config()
        self.time_budget = time_budget
        self.hyperopt = HyperparameterOptimizer(config, timeout=time_budget//2)
        self.accelerator = TrainingAccelerator()
        self.pipeline_results = {}
    
    def auto_train_fraud_detection_models(self, X_train, y_train, X_val, y_val, 
                                        X_test=None, y_test=None) -> Dict[str, Any]:
        """自動化詐騙檢測模型訓練"""
        logger.info("開始自動化詐騙檢測模型訓練...")
        total_start_time = time.time()
        
        results = {
            'hyperparameter_optimization': {},
            'accelerated_training': {},
            'model_evaluation': {},
            'best_model_info': {},
            'pipeline_summary': {}
        }
        
        # 階段1：超參數優化
        logger.info("階段1: 超參數優化")
        hyper_start_time = time.time()
        
        try:
            hyperopt_results = self.hyperopt.optimize_all_models(X_train, y_train, X_val, y_val)
            results['hyperparameter_optimization'] = hyperopt_results
            
            hyper_time = time.time() - hyper_start_time
            logger.info(f"超參數優化完成，耗時: {hyper_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"超參數優化失敗: {e}")
            # 使用默認參數
            hyperopt_results = self._get_default_configs()
            results['hyperparameter_optimization'] = {'error': str(e), 'using_defaults': True}
        
        # 階段2：加速訓練
        logger.info("階段2: 加速模型訓練")
        train_start_time = time.time()
        
        # 準備訓練配置
        if 'optimization_summary' in hyperopt_results and 'all_best_params' in hyperopt_results['optimization_summary']:\n            training_configs = hyperopt_results['optimization_summary']['all_best_params']\n        else:\n            training_configs = self._get_default_configs()\n        \n        try:\n            training_results = self.accelerator.accelerated_training(\n                training_configs, X_train, y_train, X_val, y_val\n            )\n            results['accelerated_training'] = training_results\n            \n            train_time = time.time() - train_start_time\n            logger.info(f\"加速訓練完成，耗時: {train_time:.2f}秒\")\n            \n        except Exception as e:\n            logger.error(f\"加速訓練失敗: {e}\")\n            results['accelerated_training'] = {'error': str(e)}\n            return results\n        \n        # 階段3：模型評估（如果有測試集）\n        if X_test is not None and y_test is not None:\n            logger.info(\"階段3: 模型評估\")\n            eval_start_time = time.time()\n            \n            evaluation_results = self._evaluate_models(\n                training_results['trained_models'], X_test, y_test\n            )\n            results['model_evaluation'] = evaluation_results\n            \n            eval_time = time.time() - eval_start_time\n            logger.info(f\"模型評估完成，耗時: {eval_time:.2f}秒\")\n        \n        # 最佳模型信息\n        best_model_name = training_results.get('best_model', 'unknown')\n        if best_model_name != 'unknown' and best_model_name in training_results['trained_models']:\n            results['best_model_info'] = {\n                'model_name': best_model_name,\n                'validation_score': training_results['best_score'],\n                'model_object': training_results['trained_models'][best_model_name]['model']\n            }\n            \n            # 如果有測試集結果，也包含進來\n            if 'model_evaluation' in results and best_model_name in results['model_evaluation']:\n                results['best_model_info']['test_score'] = results['model_evaluation'][best_model_name]['test_score']\n        \n        # 管道摘要\n        total_time = time.time() - total_start_time\n        results['pipeline_summary'] = {\n            'total_time': total_time,\n            'hyperopt_time': hyper_time if 'hyper_time' in locals() else 0,\n            'training_time': train_time if 'train_time' in locals() else 0,\n            'evaluation_time': eval_time if 'eval_time' in locals() else 0,\n            'models_trained': len(training_results.get('trained_models', {})),\n            'success': True\n        }\n        \n        logger.info(f\"自動化訓練管道完成，總耗時: {total_time:.2f}秒\")\n        logger.info(f\"最佳模型: {best_model_name}\")\n        \n        return results\n    \n    def _get_default_configs(self) -> Dict[str, Dict]:\n        \"\"\"獲取默認配置\"\"\"\n        return {\n            'lightgbm': {\n                'n_estimators': 1000,\n                'max_depth': 8,\n                'learning_rate': 0.05,\n                'subsample': 0.8,\n                'colsample_bytree': 0.8,\n                'class_weight': 'balanced',\n                'random_state': 42,\n                'n_jobs': -1,\n                'verbosity': -1\n            },\n            'xgboost': {\n                'n_estimators': 1000,\n                'max_depth': 8,\n                'learning_rate': 0.05,\n                'subsample': 0.8,\n                'colsample_bytree': 0.8,\n                'scale_pos_weight': 10,  # 假設的類別不平衡比例\n                'random_state': 42,\n                'n_jobs': -1,\n                'tree_method': 'hist'\n            },\n            'catboost': {\n                'iterations': 1000,\n                'depth': 8,\n                'learning_rate': 0.05,\n                'l2_leaf_reg': 3,\n                'class_weights': [1, 10],\n                'random_seed': 42,\n                'thread_count': -1,\n                'verbose': False\n            }\n        }\n    \n    def _evaluate_models(self, trained_models: Dict, X_test, y_test) -> Dict[str, Any]:\n        \"\"\"評估訓練好的模型\"\"\"\n        evaluation_results = {}\n        \n        for model_name, model_info in trained_models.items():\n            if model_info['model'] is not None and model_info['training_completed']:\n                try:\n                    model = model_info['model']\n                    y_pred = model.predict_proba(X_test)[:, 1]\n                    test_score = roc_auc_score(y_test, y_pred)\n                    \n                    evaluation_results[model_name] = {\n                        'test_score': test_score,\n                        'validation_score': model_info['validation_score'],\n                        'score_difference': test_score - model_info['validation_score']\n                    }\n                    \n                except Exception as e:\n                    logger.error(f\"評估模型 {model_name} 失敗: {e}\")\n                    evaluation_results[model_name] = {\n                        'test_score': 0,\n                        'validation_score': model_info['validation_score'],\n                        'error': str(e)\n                    }\n        \n        return evaluation_results\n\ndef quick_model_training(X_train, y_train, X_val, y_val, \n                        models_to_train: List[str] = None,\n                        time_budget: int = 1800) -> Dict[str, Any]:\n    \"\"\"快速模型訓練（生產環境友好版本）\"\"\"\n    logger.info(\"開始快速模型訓練...\")\n    \n    if models_to_train is None:\n        models_to_train = ['lightgbm', 'xgboost']\n    \n    accelerator = TrainingAccelerator()\n    \n    # 使用優化的默認配置\n    default_configs = {\n        'lightgbm': {\n            'n_estimators': 500,\n            'max_depth': 6,\n            'learning_rate': 0.1,\n            'class_weight': 'balanced',\n            'random_state': 42,\n            'n_jobs': -1,\n            'verbosity': -1\n        },\n        'xgboost': {\n            'n_estimators': 500,\n            'max_depth': 6,\n            'learning_rate': 0.1,\n            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),\n            'random_state': 42,\n            'n_jobs': -1,\n            'tree_method': 'hist'\n        }\n    }\n    \n    # 只訓練指定的模型\n    configs_to_use = {k: v for k, v in default_configs.items() if k in models_to_train}\n    \n    results = accelerator.accelerated_training(\n        configs_to_use, X_train, y_train, X_val, y_val, early_stopping_patience=30\n    )\n    \n    logger.info(f\"快速訓練完成 - 最佳模型: {results['best_model']}\")\n    return results\n\nif __name__ == \"__main__\":\n    print(\"訓練優化模組已載入完成！\")