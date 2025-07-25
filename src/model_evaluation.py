"""
模型評估與驗證模組 - IEEE-CIS 詐騙檢測項目
包含交叉驗證、性能監控、模型比較等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelValidator:
    """模型驗證器類別"""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.validation_results = {}
        self.cross_validation_scores = {}
        
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                           scoring: str = 'roc_auc') -> Dict[str, float]:
        """執行交叉驗證"""
        logger.info(f"開始 {self.cv_folds} 折交叉驗證...")
        
        # 設置分層K折交叉驗證
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # 執行交叉驗證
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        
        results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max()
        }
        
        logger.info(f"交叉驗證結果 - {scoring}: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        
        return results
    
    def validate_multiple_models(self, models: Dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """驗證多個模型"""
        validation_results = []
        
        for model_name, model in models.items():
            logger.info(f"驗證模型: {model_name}")
            
            # ROC-AUC 交叉驗證
            auc_results = self.cross_validate_model(model, X, y, scoring='roc_auc')
            
            # F1-Score 交叉驗證
            f1_results = self.cross_validate_model(model, X, y, scoring='f1')
            
            # Precision 交叉驗證
            precision_results = self.cross_validate_model(model, X, y, scoring='precision')
            
            # Recall 交叉驗證
            recall_results = self.cross_validate_model(model, X, y, scoring='recall')
            
            validation_results.append({
                'model': model_name,
                'auc_mean': auc_results['cv_mean'],
                'auc_std': auc_results['cv_std'],
                'f1_mean': f1_results['cv_mean'],
                'f1_std': f1_results['cv_std'],
                'precision_mean': precision_results['cv_mean'],
                'precision_std': precision_results['cv_std'],
                'recall_mean': recall_results['cv_mean'],
                'recall_std': recall_results['cv_std']
            })
            
            # 保存詳細結果
            self.cross_validation_scores[model_name] = {
                'auc': auc_results,
                'f1': f1_results,
                'precision': precision_results,
                'recall': recall_results
            }
        
        return pd.DataFrame(validation_results).round(4)

class ModelEvaluator:
    """詳細模型評估器"""
    
    def __init__(self):
        self.evaluation_results = {}
        self.performance_history = []
        
    def comprehensive_evaluation(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                               model_name: str = 'model') -> Dict[str, Any]:
        """全面評估模型性能"""
        logger.info(f"開始全面評估模型: {model_name}")
        
        # 預測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 基本指標
        basic_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # PR-AUC (Precision-Recall AUC)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 進階指標
        advanced_metrics = {
            'pr_auc': pr_auc,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        # 業務指標
        business_metrics = {
            'fraud_detection_rate': recall_score(y_test, y_pred),
            'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'precision_at_high_confidence': self._precision_at_threshold(y_test, y_pred_proba, 0.8),
            'recall_at_high_confidence': self._recall_at_threshold(y_test, y_pred_proba, 0.8)
        }
        
        # 合併所有指標
        evaluation = {
            'model_name': model_name,
            'evaluation_date': datetime.now().isoformat(),
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'business_metrics': business_metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.evaluation_results[model_name] = evaluation
        self.performance_history.append(evaluation)
        
        # 輸出主要指標
        logger.info(f"模型 {model_name} 評估完成:")
        logger.info(f"  ROC-AUC: {basic_metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {advanced_metrics['pr_auc']:.4f}")
        logger.info(f"  F1-Score: {basic_metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {basic_metrics['precision']:.4f}")
        logger.info(f"  Recall: {basic_metrics['recall']:.4f}")
        
        return evaluation
    
    def _precision_at_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
        """計算特定閾值下的精確率"""
        y_pred_threshold = (y_scores >= threshold).astype(int)
        if np.sum(y_pred_threshold) == 0:
            return 0.0
        return precision_score(y_true, y_pred_threshold)
    
    def _recall_at_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
        """計算特定閾值下的召回率"""
        y_pred_threshold = (y_scores >= threshold).astype(int)
        return recall_score(y_true, y_pred_threshold)
    
    def plot_performance_comparison(self, metrics: List[str] = ['roc_auc', 'f1_score', 'precision', 'recall']):
        """繪製模型性能比較圖"""
        if not self.evaluation_results:
            logger.warning("沒有可用的評估結果")
            return
        
        # 準備數據
        models = list(self.evaluation_results.keys())
        metric_values = {metric: [] for metric in metrics}
        
        for model_name in models:
            eval_result = self.evaluation_results[model_name]
            for metric in metrics:
                if metric in eval_result['basic_metrics']:
                    metric_values[metric].append(eval_result['basic_metrics'][metric])
                elif metric in eval_result['advanced_metrics']:
                    metric_values[metric].append(eval_result['advanced_metrics'][metric])
                else:
                    metric_values[metric].append(0)
        
        # 繪製比較圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].bar(models, metric_values[metric])
                axes[i].set_title(f'{metric.upper()} 比較')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                
                # 添加數值標籤
                for j, v in enumerate(metric_values[metric]):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self):
        """繪製所有模型的混淆矩陣"""
        if not self.evaluation_results:
            logger.warning("沒有可用的評估結果")
            return
        
        n_models = len(self.evaluation_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.ravel()
        
        for i, (model_name, eval_result) in enumerate(self.evaluation_results.items()):
            cm = np.array(eval_result['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name} - Confusion Matrix')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
        # 隱藏多餘的子圖
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, output_path: str = None) -> str:
        """生成評估報告"""
        if not self.evaluation_results:
            logger.warning("沒有可用的評估結果")
            return ""
        
        report = []
        report.append("# 模型評估報告")
        report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 模型概覽
        report.append("## 模型概覽")
        for model_name in self.evaluation_results.keys():
            report.append(f"- {model_name}")
        report.append("")
        
        # 詳細評估結果
        for model_name, eval_result in self.evaluation_results.items():
            report.append(f"## {model_name} 評估結果")
            
            basic_metrics = eval_result['basic_metrics']
            advanced_metrics = eval_result['advanced_metrics']
            business_metrics = eval_result['business_metrics']
            
            report.append("### 基本指標")
            for metric, value in basic_metrics.items():
                report.append(f"- {metric.upper()}: {value:.4f}")
            
            report.append("\n### 進階指標")
            for metric, value in advanced_metrics.items():
                report.append(f"- {metric.upper()}: {value:.4f}")
            
            report.append("\n### 業務指標")
            for metric, value in business_metrics.items():
                report.append(f"- {metric.upper()}: {value:.4f}")
            
            report.append("")
        
        # 模型比較
        report.append("## 模型比較")
        comparison_data = []
        for model_name, eval_result in self.evaluation_results.items():
            comparison_data.append({
                '模型': model_name,
                'ROC-AUC': eval_result['basic_metrics']['roc_auc'],
                'PR-AUC': eval_result['advanced_metrics']['pr_auc'],
                'F1-Score': eval_result['basic_metrics']['f1_score'],
                'Precision': eval_result['basic_metrics']['precision'],
                'Recall': eval_result['basic_metrics']['recall']
            })
        
        comparison_df = pd.DataFrame(comparison_data).round(4)
        report.append(comparison_df.to_string(index=False))
        
        # 生成最終報告
        final_report = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            logger.info(f"評估報告已保存至: {output_path}")
        
        return final_report
    
    def save_evaluation_results(self, filepath: str):
        """保存評估結果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        logger.info(f"評估結果已保存至: {filepath}")
    
    def load_evaluation_results(self, filepath: str):
        """載入評估結果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.evaluation_results = json.load(f)
        logger.info(f"評估結果已從 {filepath} 載入")

def evaluate_model_performance(model, X_test: pd.DataFrame, y_test: pd.Series, 
                             model_name: str = 'model') -> Dict[str, Any]:
    """便捷函數：評估單個模型性能"""
    evaluator = ModelEvaluator()
    return evaluator.comprehensive_evaluation(model, X_test, y_test, model_name)

def compare_models(models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> ModelEvaluator:
    """便捷函數：比較多個模型"""
    evaluator = ModelEvaluator()
    
    for model_name, model in models.items():
        evaluator.comprehensive_evaluation(model, X_test, y_test, model_name)
    
    return evaluator

if __name__ == "__main__":
    print("模型評估與驗證模組已載入完成！")