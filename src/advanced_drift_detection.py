"""
高級數據漂移檢測模組 - IEEE-CIS 詐騙檢測項目
提供多種先進的漂移檢測算法和自動化監控功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .model_monitoring import DriftMetrics
from .config import get_config

logger = logging.getLogger(__name__)

@dataclass
class PopulationStabilityIndex:
    """人口穩定性指數（PSI）結果"""
    psi_value: float
    bins_info: List[Dict[str, Any]]
    drift_detected: bool
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'psi_value': self.psi_value,
            'bins_info': self.bins_info,
            'drift_detected': self.drift_detected,
            'interpretation': self.interpretation
        }

@dataclass
class MultivariateDriftResult:
    """多變量漂移檢測結果"""
    drift_detected: bool
    drift_score: float
    p_value: float
    affected_features: List[str]
    drift_type: str
    details: Dict[str, Any]

class AdvancedDriftDetector(ABC):
    """高級漂移檢測器基類"""
    
    @abstractmethod
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame, 
                    **kwargs) -> MultivariateDriftResult:
        """檢測漂移"""
        pass

class PSIDriftDetector(AdvancedDriftDetector):
    """基於人口穩定性指數（PSI）的漂移檢測器"""
    
    def __init__(self, bins: int = 10, psi_threshold: float = 0.1):
        self.bins = bins
        self.psi_threshold = psi_threshold
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> PopulationStabilityIndex:
        """計算PSI指數"""
        try:
            # 移除NaN值
            ref_clean = reference[~np.isnan(reference)]
            cur_clean = current[~np.isnan(current)]
            
            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return PopulationStabilityIndex(0.0, [], False, "數據不足")
            
            # 創建分箱
            _, bin_edges = np.histogram(ref_clean, bins=self.bins)
            
            # 確保邊界包含所有數據
            bin_edges[0] = min(bin_edges[0], cur_clean.min())
            bin_edges[-1] = max(bin_edges[-1], cur_clean.max())
            
            # 計算各分箱的頻率
            ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
            cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)
            
            # 轉換為比例
            ref_props = ref_counts / len(ref_clean)
            cur_props = cur_counts / len(cur_clean)
            
            # 避免零除錯誤
            ref_props = np.where(ref_props == 0, 0.0001, ref_props)
            cur_props = np.where(cur_props == 0, 0.0001, cur_props)
            
            # 計算PSI
            psi_values = (cur_props - ref_props) * np.log(cur_props / ref_props)
            psi_total = np.sum(psi_values)
            
            # 創建分箱信息
            bins_info = []
            for i in range(len(bin_edges) - 1):
                bins_info.append({
                    'bin_range': f"[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})",
                    'reference_prop': ref_props[i],
                    'current_prop': cur_props[i],
                    'psi_contribution': psi_values[i]
                })
            
            # 解釋PSI值
            if psi_total < 0.1:
                interpretation = "無顯著變化"
            elif psi_total < 0.2:
                interpretation = "輕微變化"
            else:
                interpretation = "顯著變化"
            
            drift_detected = psi_total > self.psi_threshold
            
            return PopulationStabilityIndex(
                psi_value=psi_total,
                bins_info=bins_info,
                drift_detected=drift_detected,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"PSI計算失敗: {e}")
            return PopulationStabilityIndex(0.0, [], False, f"計算錯誤: {e}")
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame, 
                    features: List[str] = None) -> MultivariateDriftResult:
        """檢測多特徵PSI漂移"""
        if features is None:
            features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_psi_results = {}
        drifted_features = []
        total_drift_score = 0.0
        
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            psi_result = self.calculate_psi(
                reference_data[feature].values,
                current_data[feature].values
            )
            
            feature_psi_results[feature] = psi_result
            total_drift_score += psi_result.psi_value
            
            if psi_result.drift_detected:
                drifted_features.append(feature)
        
        # 計算整體漂移分數
        avg_psi = total_drift_score / len(features) if features else 0.0
        drift_detected = len(drifted_features) > 0
        
        return MultivariateDriftResult(
            drift_detected=drift_detected,
            drift_score=avg_psi,
            p_value=0.0,  # PSI不提供p值
            affected_features=drifted_features,
            drift_type='psi',
            details={'feature_psi_results': {k: v.to_dict() for k, v in feature_psi_results.items()}}
        )

class PCADriftDetector(AdvancedDriftDetector):
    """基於主成分分析（PCA）的漂移檢測器"""
    
    def __init__(self, n_components: float = 0.95, threshold_percentile: float = 95):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.pca = None
        self.reference_scores = None
        self.threshold = None
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame, 
                    features: List[str] = None) -> MultivariateDriftResult:
        """使用PCA檢測多變量漂移"""
        try:
            if features is None:
                features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 準備數據
            ref_data = reference_data[features].fillna(0)
            cur_data = current_data[features].fillna(0)
            
            # 訓練PCA模型
            self.pca = PCA(n_components=self.n_components)
            ref_transformed = self.pca.fit_transform(ref_data)
            
            # 計算參考數據的重構誤差
            ref_reconstructed = self.pca.inverse_transform(ref_transformed)
            ref_errors = np.sum((ref_data.values - ref_reconstructed) ** 2, axis=1)
            
            # 設置閾值
            self.threshold = np.percentile(ref_errors, self.threshold_percentile)
            
            # 計算當前數據的重構誤差
            cur_transformed = self.pca.transform(cur_data)
            cur_reconstructed = self.pca.inverse_transform(cur_transformed)
            cur_errors = np.sum((cur_data.values - cur_reconstructed) ** 2, axis=1)
            
            # 檢測漂移
            drift_ratio = np.mean(cur_errors > self.threshold)
            drift_detected = drift_ratio > 0.1  # 如果超過10%的樣本超出閾值
            
            # 識別受影響的特徵
            feature_importance = np.abs(self.pca.components_).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-5:]  # 前5個重要特徵
            affected_features = [features[i] for i in top_features_idx]
            
            # 統計測試
            statistic, p_value = stats.ks_2samp(ref_errors, cur_errors)
            
            return MultivariateDriftResult(
                drift_detected=drift_detected,
                drift_score=drift_ratio,
                p_value=p_value,
                affected_features=affected_features,
                drift_type='pca',
                details={
                    'reconstruction_error_threshold': self.threshold,
                    'drift_ratio': drift_ratio,
                    'pca_explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
                    'n_components_used': self.pca.n_components_
                }
            )
            
        except Exception as e:
            logger.error(f"PCA漂移檢測失敗: {e}")
            return MultivariateDriftResult(
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                affected_features=[],
                drift_type='pca',
                details={'error': str(e)}
            )

class IsolationForestDriftDetector(AdvancedDriftDetector):
    """基於孤立森林的漂移檢測器"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.isolation_forest = None
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame, 
                    features: List[str] = None) -> MultivariateDriftResult:
        """使用孤立森林檢測異常漂移"""
        try:
            if features is None:
                features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 準備數據
            ref_data = reference_data[features].fillna(0)
            cur_data = current_data[features].fillna(0)
            
            # 訓練孤立森林
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42
            )
            self.isolation_forest.fit(ref_data)
            
            # 預測異常
            cur_predictions = self.isolation_forest.predict(cur_data)
            cur_scores = self.isolation_forest.decision_function(cur_data)
            
            # 計算異常比例
            anomaly_ratio = np.mean(cur_predictions == -1)
            drift_detected = anomaly_ratio > self.contamination * 2  # 異常比例是預期的2倍以上
            
            # 識別最異常的特徵
            # 通過特徵重要性估計（基於路徑長度）
            feature_importance = np.abs(cur_scores).mean()
            affected_features = features  # 孤立森林考慮所有特徵
            
            return MultivariateDriftResult(
                drift_detected=drift_detected,
                drift_score=anomaly_ratio,
                p_value=0.0,  # 孤立森林不提供p值
                affected_features=affected_features,
                drift_type='isolation_forest',
                details={
                    'anomaly_ratio': anomaly_ratio,
                    'expected_contamination': self.contamination,
                    'mean_anomaly_score': np.mean(cur_scores),
                    'std_anomaly_score': np.std(cur_scores)
                }
            )
            
        except Exception as e:
            logger.error(f"孤立森林漂移檢測失敗: {e}")
            return MultivariateDriftResult(
                drift_detected=False,
                drift_score=0.0,
                p_value=1.0,
                affected_features=[],
                drift_type='isolation_forest',
                details={'error': str(e)}
            )

class EnsembleDriftDetector:
    """集成漂移檢測器"""
    
    def __init__(self):
        self.detectors = {
            'psi': PSIDriftDetector(),
            'pca': PCADriftDetector(),
            'isolation_forest': IsolationForestDriftDetector()
        }
        self.weights = {
            'psi': 0.4,
            'pca': 0.4,
            'isolation_forest': 0.2
        }
    
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame, 
                    features: List[str] = None) -> Dict[str, MultivariateDriftResult]:
        """使用多個檢測器進行集成漂移檢測"""
        results = {}
        
        for name, detector in self.detectors.items():
            try:
                result = detector.detect_drift(reference_data, current_data, features)
                results[name] = result
                logger.info(f"{name} 漂移檢測完成 - 漂移: {result.drift_detected}")
            except Exception as e:
                logger.error(f"{name} 漂移檢測失敗: {e}")
        
        return results
    
    def get_consensus_result(self, results: Dict[str, MultivariateDriftResult]) -> MultivariateDriftResult:
        """獲取集成檢測結果"""
        # 加權投票
        weighted_score = 0.0
        weighted_detection = 0.0
        all_affected_features = set()
        
        for name, result in results.items():
            weight = self.weights.get(name, 1.0)
            weighted_score += result.drift_score * weight
            weighted_detection += (1.0 if result.drift_detected else 0.0) * weight
            all_affected_features.update(result.affected_features)
        
        # 決策
        consensus_drift = weighted_detection > 0.5
        
        return MultivariateDriftResult(
            drift_detected=consensus_drift,
            drift_score=weighted_score,
            p_value=0.0,
            affected_features=list(all_affected_features),
            drift_type='ensemble',
            details={
                'individual_results': {k: v.__dict__ for k, v in results.items()},
                'weighted_detection_score': weighted_detection
            }
        )

class ComprehensiveDriftMonitor:
    """綜合漂移監控系統"""
    
    def __init__(self, reference_data: pd.DataFrame, 
                 monitoring_features: List[str] = None):
        self.reference_data = reference_data
        self.monitoring_features = monitoring_features
        self.ensemble_detector = EnsembleDriftDetector()
        self.drift_history: List[Dict[str, Any]] = []
        
        if self.monitoring_features is None:
            self.monitoring_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
    
    def monitor_drift(self, current_data: pd.DataFrame, 
                     timestamp: datetime = None) -> Dict[str, Any]:
        """執行綜合漂移監控"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # 執行集成漂移檢測
        individual_results = self.ensemble_detector.detect_drift(
            self.reference_data, current_data, self.monitoring_features
        )
        
        # 獲取集成結果
        consensus_result = self.ensemble_detector.get_consensus_result(individual_results)
        
        # 記錄結果
        monitoring_record = {
            'timestamp': timestamp,
            'consensus_result': consensus_result,
            'individual_results': individual_results,
            'data_size': len(current_data)
        }
        
        self.drift_history.append(monitoring_record)
        
        # 記錄日誌
        if consensus_result.drift_detected:
            logger.warning(f"檢測到數據漂移 - 分數: {consensus_result.drift_score:.4f}")
            logger.warning(f"受影響的特徵: {consensus_result.affected_features}")
        else:
            logger.info("未檢測到顯著數據漂移")
        
        return monitoring_record
    
    def create_drift_report(self, days: int = 7) -> Dict[str, Any]:
        """創建漂移報告"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            r for r in self.drift_history 
            if r['timestamp'] >= cutoff_date
        ]
        
        if not recent_records:
            return {'message': '沒有足夠的監控數據'}
        
        # 統計分析
        drift_detected_count = sum(
            1 for r in recent_records 
            if r['consensus_result'].drift_detected
        )
        
        # 受影響特徵頻率統計
        feature_drift_count = {}
        for record in recent_records:
            for feature in record['consensus_result'].affected_features:
                feature_drift_count[feature] = feature_drift_count.get(feature, 0) + 1
        
        # 各檢測器統計
        detector_stats = {}
        for detector_name in ['psi', 'pca', 'isolation_forest']:
            detector_drift_count = sum(
                1 for r in recent_records 
                if r['individual_results'].get(detector_name, {}).drift_detected
            )
            detector_stats[detector_name] = {
                'drift_count': detector_drift_count,
                'drift_rate': detector_drift_count / len(recent_records)
            }
        
        return {
            'report_period': f'{days} days',
            'total_monitoring_sessions': len(recent_records),
            'drift_detected_sessions': drift_detected_count,
            'overall_drift_rate': drift_detected_count / len(recent_records),
            'most_affected_features': dict(sorted(
                feature_drift_count.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),
            'detector_performance': detector_stats,
            'generated_at': datetime.now().isoformat()
        }
    
    def visualize_drift_trends(self, days: int = 30):
        """可視化漂移趨勢"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            r for r in self.drift_history 
            if r['timestamp'] >= cutoff_date
        ]
        
        if not recent_records:
            logger.warning("沒有足夠的數據進行可視化")
            return
        
        # 準備數據
        timestamps = [r['timestamp'] for r in recent_records]
        drift_scores = [r['consensus_result'].drift_score for r in recent_records]
        drift_detected = [r['consensus_result'].drift_detected for r in recent_records]
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('漂移分數趨勢', '漂移檢測狀態'),
            vertical_spacing=0.1
        )
        
        # 漂移分數趨勢
        fig.add_trace(
            go.Scatter(
                x=timestamps, 
                y=drift_scores, 
                mode='lines+markers',
                name='漂移分數',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 漂移檢測狀態
        fig.add_trace(
            go.Scatter(
                x=timestamps, 
                y=[1 if d else 0 for d in drift_detected],
                mode='markers',
                name='漂移檢測',
                marker=dict(
                    color=['red' if d else 'green' for d in drift_detected],
                    size=8
                )
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'最近 {days} 天的數據漂移趨勢',
            height=600
        )
        
        fig.update_yaxes(title_text="漂移分數", row=1, col=1)
        fig.update_yaxes(title_text="檢測狀態", row=2, col=1)
        
        fig.show()
    
    def export_drift_data(self, filepath: str):
        """導出漂移監控數據"""
        export_data = []
        
        for record in self.drift_history:
            row = {
                'timestamp': record['timestamp'].isoformat(),
                'drift_detected': record['consensus_result'].drift_detected,
                'drift_score': record['consensus_result'].drift_score,
                'affected_features_count': len(record['consensus_result'].affected_features),
                'affected_features': ','.join(record['consensus_result'].affected_features),
                'data_size': record['data_size']
            }
            
            # 添加各檢測器結果
            for detector_name, result in record['individual_results'].items():
                row[f'{detector_name}_drift'] = result.drift_detected
                row[f'{detector_name}_score'] = result.drift_score
            
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"漂移數據已導出至: {filepath}")

# 便捷函數
def create_comprehensive_drift_monitor(reference_data: pd.DataFrame, 
                                     features: List[str] = None) -> ComprehensiveDriftMonitor:
    """創建綜合漂移監控器"""
    return ComprehensiveDriftMonitor(reference_data, features)

def quick_drift_analysis(reference_data: pd.DataFrame, 
                        current_data: pd.DataFrame, 
                        features: List[str] = None) -> Dict[str, Any]:
    """快速漂移分析"""
    detector = EnsembleDriftDetector()
    results = detector.detect_drift(reference_data, current_data, features)
    consensus = detector.get_consensus_result(results)
    
    return {
        'drift_detected': consensus.drift_detected,
        'drift_score': consensus.drift_score,
        'affected_features': consensus.affected_features,
        'individual_results': {k: v.drift_detected for k, v in results.items()}
    }

if __name__ == "__main__":
    print("高級數據漂移檢測模組已載入完成！")