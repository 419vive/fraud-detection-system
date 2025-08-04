"""
模型監控與漂移檢測模組 - IEEE-CIS 詐騙檢測項目
提供模型性能監控、數據漂移檢測和預警功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import pickle
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .config import get_config
from .exceptions import ModelError, DataValidationError

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    prediction_count: int
    fraud_rate: float
    average_prediction_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """從字典創建實例"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class DriftMetrics:
    """漂移指標數據類"""
    timestamp: datetime
    feature_name: str
    drift_score: float
    p_value: float
    drift_detected: bool
    drift_type: str  # 'mean', 'distribution', 'categorical'
    reference_stats: Dict[str, Any]
    current_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DriftMetrics':
        """從字典創建實例"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class DriftDetector(ABC):
    """漂移檢測器基類"""
    
    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, 
                    threshold: float = 0.05) -> Tuple[bool, float, float]:
        """檢測漂移"""
        pass

class KSTestDriftDetector(DriftDetector):
    """基於Kolmogorov-Smirnov檢驗的漂移檢測器"""
    
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, 
                    threshold: float = 0.05) -> Tuple[bool, float, float]:
        """使用KS檢驗檢測分佈漂移"""
        try:
            # 移除NaN值
            ref_clean = reference_data[~np.isnan(reference_data)]
            cur_clean = current_data[~np.isnan(current_data)]
            
            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return False, 0.0, 1.0
            
            # 執行KS檢驗
            ks_statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)
            
            drift_detected = p_value < threshold
            return drift_detected, ks_statistic, p_value
            
        except Exception as e:
            logger.error(f"KS檢驗失敗: {e}")
            return False, 0.0, 1.0

class ChiSquareDriftDetector(DriftDetector):
    """基於卡方檢驗的類別特徵漂移檢測器"""
    
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, 
                    threshold: float = 0.05) -> Tuple[bool, float, float]:
        """使用卡方檢驗檢測類別分佈漂移"""
        try:
            # 獲取所有可能的類別
            all_categories = np.unique(np.concatenate([reference_data, current_data]))
            
            # 計算頻率
            ref_counts = pd.Series(reference_data).value_counts().reindex(all_categories, fill_value=0)
            cur_counts = pd.Series(current_data).value_counts().reindex(all_categories, fill_value=0)
            
            # 避免頻率為0的情況
            ref_counts = ref_counts + 1
            cur_counts = cur_counts + 1
            
            # 執行卡方檢驗
            chi2_statistic, p_value = stats.chisquare(cur_counts, ref_counts)
            
            drift_detected = p_value < threshold
            return drift_detected, chi2_statistic, p_value
            
        except Exception as e:
            logger.error(f"卡方檢驗失敗: {e}")
            return False, 0.0, 1.0

class MeanShiftDetector(DriftDetector):
    """基於均值偏移的漂移檢測器"""
    
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray, 
                    threshold: float = 0.1) -> Tuple[bool, float, float]:
        """檢測均值漂移"""
        try:
            ref_mean = np.nanmean(reference_data)
            cur_mean = np.nanmean(current_data)
            ref_std = np.nanstd(reference_data)
            
            if ref_std == 0:
                return False, 0.0, 1.0
            
            # 計算標準化差異
            drift_score = abs(cur_mean - ref_mean) / ref_std
            
            # 使用t檢驗
            t_statistic, p_value = stats.ttest_ind(
                reference_data[~np.isnan(reference_data)],
                current_data[~np.isnan(current_data)]
            )
            
            drift_detected = drift_score > threshold
            return drift_detected, drift_score, p_value
            
        except Exception as e:
            logger.error(f"均值偏移檢測失敗: {e}")
            return False, 0.0, 1.0

class ModelMonitor:
    """模型監控器"""
    
    def __init__(self, model_name: str, reference_data: pd.DataFrame = None):
        self.model_name = model_name
        self.reference_data = reference_data
        self.performance_history: List[PerformanceMetrics] = []
        self.drift_history: List[DriftMetrics] = []
        
        # 配置漂移檢測器
        self.drift_detectors = {
            'numerical': KSTestDriftDetector(),
            'categorical': ChiSquareDriftDetector(),
            'mean_shift': MeanShiftDetector()
        }
        
        self.config = get_config()
    
    def log_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray, prediction_times: List[float] = None):
        """記錄模型性能"""
        try:
            # 計算性能指標
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1_score=f1_score(y_true, y_pred, zero_division=0),
                roc_auc=roc_auc_score(y_true, y_pred_proba),
                prediction_count=len(y_pred),
                fraud_rate=np.mean(y_true),
                average_prediction_time=np.mean(prediction_times) if prediction_times else 0.0
            )
            
            self.performance_history.append(metrics)
            
            logger.info(f"性能記錄已更新 - AUC: {metrics.roc_auc:.4f}, F1: {metrics.f1_score:.4f}")
            
            # 檢查性能退化
            self._check_performance_degradation(metrics)
            
        except Exception as e:
            logger.error(f"性能記錄失敗: {e}")
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         features: List[str] = None, 
                         drift_threshold: float = 0.05) -> List[DriftMetrics]:
        """檢測數據漂移"""
        if self.reference_data is None:
            logger.warning("未設置參考數據，無法進行漂移檢測")
            return []
        
        if features is None:
            # 只檢測數值型特徵
            features = current_data.select_dtypes(include=[np.number]).columns.tolist()
        
        drift_results = []
        
        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue
            
            try:
                # 獲取數據
                ref_data = self.reference_data[feature].values
                cur_data = current_data[feature].values
                
                # 選擇適當的檢測器
                if pd.api.types.is_numeric_dtype(current_data[feature]):
                    detector = self.drift_detectors['numerical']
                    drift_type = 'distribution'
                else:
                    detector = self.drift_detectors['categorical']
                    drift_type = 'categorical'
                
                # 檢測漂移
                drift_detected, drift_score, p_value = detector.detect_drift(
                    ref_data, cur_data, drift_threshold
                )
                
                # 計算統計信息
                ref_stats = self._calculate_feature_stats(ref_data, drift_type)
                cur_stats = self._calculate_feature_stats(cur_data, drift_type)
                
                # 創建漂移記錄
                drift_metric = DriftMetrics(
                    timestamp=datetime.now(),
                    feature_name=feature,
                    drift_score=drift_score,
                    p_value=p_value,
                    drift_detected=drift_detected,
                    drift_type=drift_type,
                    reference_stats=ref_stats,
                    current_stats=cur_stats
                )
                
                drift_results.append(drift_metric)
                
                if drift_detected:
                    logger.warning(f"檢測到特徵 {feature} 發生漂移 - 分數: {drift_score:.4f}, p值: {p_value:.4f}")
                
            except Exception as e:
                logger.error(f"特徵 {feature} 漂移檢測失敗: {e}")
        
        # 保存漂移歷史
        self.drift_history.extend(drift_results)
        
        return drift_results
    
    def _calculate_feature_stats(self, data: np.ndarray, data_type: str) -> Dict[str, Any]:
        """計算特徵統計信息"""
        stats_dict = {}
        
        if data_type in ['distribution', 'numerical']:
            clean_data = data[~np.isnan(data)]
            if len(clean_data) > 0:
                stats_dict.update({
                    'mean': float(np.mean(clean_data)),
                    'std': float(np.std(clean_data)),
                    'min': float(np.min(clean_data)),
                    'max': float(np.max(clean_data)),
                    'q25': float(np.percentile(clean_data, 25)),
                    'q50': float(np.percentile(clean_data, 50)),
                    'q75': float(np.percentile(clean_data, 75)),
                    'missing_rate': float(np.sum(np.isnan(data)) / len(data))
                })
        
        elif data_type == 'categorical':
            value_counts = pd.Series(data).value_counts()
            stats_dict.update({
                'unique_values': len(value_counts),
                'top_values': value_counts.head(10).to_dict(),
                'missing_rate': float(pd.Series(data).isna().sum() / len(data))
            })
        
        return stats_dict
    
    def _check_performance_degradation(self, current_metrics: PerformanceMetrics):
        """檢查性能退化"""
        if len(self.performance_history) < 5:  # 需要至少5個歷史記錄
            return
        
        # 獲取最近的歷史性能
        recent_history = self.performance_history[-5:-1]  # 排除當前記錄
        historical_auc = np.mean([m.roc_auc for m in recent_history])
        
        # 檢查AUC退化
        auc_degradation = historical_auc - current_metrics.roc_auc
        if auc_degradation > 0.05:  # 5%退化閾值
            logger.warning(f"檢測到模型性能退化 - AUC下降 {auc_degradation:.4f}")
            self._trigger_alert('performance_degradation', {
                'metric': 'roc_auc',
                'degradation': auc_degradation,
                'current_value': current_metrics.roc_auc,
                'historical_average': historical_auc
            })
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """觸發警告"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'model_name': self.model_name,
            'details': details
        }
        
        logger.critical(f"模型監控警告: {alert}")
        
        # 可以在這裡添加其他警告機制，如發送郵件、Slack通知等
    
    def generate_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        """生成監控報告"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 篩選最近的記錄
        recent_performance = [
            m for m in self.performance_history 
            if m.timestamp >= cutoff_date
        ]
        
        recent_drift = [
            d for d in self.drift_history 
            if d.timestamp >= cutoff_date
        ]
        
        # 性能統計
        performance_stats = {}
        if recent_performance:
            performance_stats = {
                'average_auc': np.mean([m.roc_auc for m in recent_performance]),
                'average_f1': np.mean([m.f1_score for m in recent_performance]),
                'total_predictions': sum([m.prediction_count for m in recent_performance]),
                'average_fraud_rate': np.mean([m.fraud_rate for m in recent_performance]),
                'performance_trend': self._calculate_performance_trend(recent_performance)
            }
        
        # 漂移統計
        drift_stats = {}
        if recent_drift:
            drift_features = list(set([d.feature_name for d in recent_drift]))
            drift_detected_features = [d.feature_name for d in recent_drift if d.drift_detected]
            
            drift_stats = {
                'total_features_monitored': len(drift_features),
                'features_with_drift': len(set(drift_detected_features)),
                'drift_rate': len(set(drift_detected_features)) / len(drift_features) if drift_features else 0,
                'most_drifted_features': list(set(drift_detected_features))[:10]
            }
        
        return {
            'report_period': f'{days} days',
            'generated_at': datetime.now().isoformat(),
            'model_name': self.model_name,
            'performance_statistics': performance_stats,
            'drift_statistics': drift_stats,
            'alerts_generated': self._count_recent_alerts(cutoff_date)
        }
    
    def _calculate_performance_trend(self, metrics: List[PerformanceMetrics]) -> str:
        """計算性能趨勢"""
        if len(metrics) < 3:
            return 'insufficient_data'
        
        auc_values = [m.roc_auc for m in metrics]
        
        # 簡單線性趨勢分析
        x = np.arange(len(auc_values))
        slope = np.polyfit(x, auc_values, 1)[0]
        
        if slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'declining'
        else:
            return 'stable'
    
    def _count_recent_alerts(self, cutoff_date: datetime) -> int:
        """計算最近的警告數量"""
        # 這裡可以實現警告計數邏輯
        return 0
    
    def plot_performance_history(self, days: int = 30):
        """繪製性能歷史圖"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.performance_history 
            if m.timestamp >= cutoff_date
        ]
        
        if not recent_metrics:
            logger.warning("沒有足夠的歷史數據進行繪圖")
            return
        
        timestamps = [m.timestamp for m in recent_metrics]
        auc_scores = [m.roc_auc for m in recent_metrics]
        f1_scores = [m.f1_score for m in recent_metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # AUC趨勢
        ax1.plot(timestamps, auc_scores, marker='o', linewidth=2)
        ax1.set_title(f'{self.model_name} - ROC-AUC 趨勢')
        ax1.set_ylabel('ROC-AUC')
        ax1.grid(True, alpha=0.3)
        
        # F1趨勢
        ax2.plot(timestamps, f1_scores, marker='s', color='orange', linewidth=2)
        ax2.set_title(f'{self.model_name} - F1 Score 趨勢')
        ax2.set_ylabel('F1 Score')
        ax2.set_xlabel('時間')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    
    def plot_drift_summary(self, days: int = 7):
        """繪製漂移摘要圖"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_drift = [
            d for d in self.drift_history 
            if d.timestamp >= cutoff_date
        ]
        
        if not recent_drift:
            logger.warning("沒有漂移數據進行繪圖")
            return
        
        # 統計每個特徵的漂移次數
        drift_counts = {}
        for drift in recent_drift:
            if drift.drift_detected:
                drift_counts[drift.feature_name] = drift_counts.get(drift.feature_name, 0) + 1
        
        if not drift_counts:
            logger.info("在指定期間內沒有檢測到漂移")
            return
        
        # 繪製漂移頻率圖
        features = list(drift_counts.keys())
        counts = list(drift_counts.values())
        
        plt.figure(figsize=(12, 6))
        plt.bar(features, counts)
        plt.title(f'最近 {days} 天的特徵漂移檢測')
        plt.xlabel('特徵名稱')
        plt.ylabel('漂移檢測次數')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def save_monitoring_data(self, filepath: str):
        """保存監控數據"""
        monitoring_data = {
            'model_name': self.model_name,
            'performance_history': [m.to_dict() for m in self.performance_history],
            'drift_history': [d.to_dict() for d in self.drift_history],
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"監控數據已保存至: {filepath}")
    
    def load_monitoring_data(self, filepath: str):
        """載入監控數據"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                monitoring_data = json.load(f)
            
            self.model_name = monitoring_data['model_name']
            self.performance_history = [
                PerformanceMetrics.from_dict(m) 
                for m in monitoring_data['performance_history']
            ]
            self.drift_history = [
                DriftMetrics.from_dict(d) 
                for d in monitoring_data['drift_history']
            ]
            
            logger.info(f"監控數據已從 {filepath} 載入")
            
        except Exception as e:
            logger.error(f"載入監控數據失敗: {e}")

# 便捷函數
def create_model_monitor(model_name: str, reference_data: pd.DataFrame = None) -> ModelMonitor:
    """創建模型監控器"""
    return ModelMonitor(model_name, reference_data)

def quick_drift_check(reference_data: pd.DataFrame, 
                     current_data: pd.DataFrame, 
                     features: List[str] = None) -> Dict[str, bool]:
    """快速漂移檢查"""
    monitor = ModelMonitor("quick_check", reference_data)
    drift_results = monitor.detect_data_drift(current_data, features)
    
    return {
        drift.feature_name: drift.drift_detected 
        for drift in drift_results
    }

if __name__ == "__main__":
    print("模型監控與漂移檢測模組已載入完成！")