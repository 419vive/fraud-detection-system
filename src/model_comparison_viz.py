"""
模型比較可視化 - IEEE-CIS 詐騙檢測項目
提供多模型性能比較、特徵重要性對比和決策邊界分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import logging
from datetime import datetime
import json

from .visualization_engine import VisualizationEngine
from .config import get_config

logger = logging.getLogger(__name__)

class ModelComparisonVisualizer:
    """模型比較可視化器"""
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.viz_engine = VisualizationEngine(config_manager)
        self.model_results = {}
        self.comparison_cache = {}
        
        # 模型顏色映射
        self.model_colors = {
            'Logistic Regression': '#1f77b4',
            'Random Forest': '#ff7f0e',
            'XGBoost': '#2ca02c',
            'LightGBM': '#d62728',
            'CatBoost': '#9467bd',
            'Neural Network': '#8c564b',
            'SVM': '#e377c2',
            'Ensemble': '#7f7f7f'
        }
    
    def add_model_results(self, model_name: str, results: Dict[str, Any]):
        """添加模型結果"""
        required_keys = ['y_true', 'y_pred', 'y_pred_proba']
        if not all(key in results for key in required_keys):
            raise ValueError(f"模型結果必須包含: {required_keys}")
        
        # 計算基本指標
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        y_pred_proba = np.array(results['y_pred_proba'])
        
        metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
        
        self.model_results[model_name] = {
            'data': results,
            'metrics': metrics,
            'feature_importance': results.get('feature_importance'),
            'feature_names': results.get('feature_names'),
            'training_time': results.get('training_time', 0),
            'prediction_time': results.get('prediction_time', 0)
        }
        
        logger.info(f"已添加模型結果: {model_name} (AUC: {metrics['roc_auc']:.4f})")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """計算模型指標"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba)
        }
    
    def create_performance_comparison_dashboard(self) -> go.Figure:
        """創建性能比較儀表板"""
        if len(self.model_results) < 2:
            raise ValueError("至少需要2個模型進行比較")
        
        logger.info(f"創建 {len(self.model_results)} 個模型的性能比較儀表板")
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'ROC曲線比較', '精確率-召回率曲線', '性能指標雷達圖',
                '混淆矩陣比較', '特徵重要性對比', '訓練/預測時間',
                '模型穩定性分析', '錯誤分析', '綜合評分'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatterpolar"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. ROC曲線比較
        for model_name, model_data in self.model_results.items():
            y_true = model_data['data']['y_true']
            y_pred_proba = model_data['data']['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            color = self.model_colors.get(model_name, '#1f77b4')
            
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC={roc_auc:.3f})',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
        
        # 添加對角線
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='隨機分類器',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Precision-Recall曲線
        for model_name, model_data in self.model_results.items():
            y_true = model_data['data']['y_true']
            y_pred_proba = model_data['data']['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = auc(recall, precision)
            
            color = self.model_colors.get(model_name, '#1f77b4')
            
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{model_name} (PR-AUC={pr_auc:.3f})',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. 性能指標雷達圖
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        for model_name, model_data in self.model_results.items():
            metrics = model_data['metrics']
            values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['roc_auc']
            ]
            
            color = self.model_colors.get(model_name, '#1f77b4')
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # 閉合多邊形
                    theta=metrics_names + [metrics_names[0]],
                    fill='toself',
                    name=model_name,
                    line_color=color,
                    opacity=0.6
                ),
                row=1, col=3
            )
        
        # 4. 混淆矩陣比較（選擇最佳模型）
        best_model = max(self.model_results.items(), key=lambda x: x[1]['metrics']['roc_auc'])
        best_model_name, best_model_data = best_model
        
        y_true = best_model_data['data']['y_true']
        y_pred = best_model_data['data']['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig.add_trace(
            go.Heatmap(
                z=cm_normalized,
                x=['預測: 正常', '預測: 詐騙'],
                y=['實際: 正常', '實際: 詐騙'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=False
            ),
            row=2, col=1
        )
        
        # 5. 特徵重要性對比（選擇有特徵重要性的模型）
        models_with_importance = {
            name: data for name, data in self.model_results.items() 
            if data.get('feature_importance') is not None
        }
        
        if models_with_importance:
            self._add_feature_importance_comparison(fig, models_with_importance, row=2, col=2)
        
        # 6. 訓練/預測時間比較
        model_names = list(self.model_results.keys())
        training_times = [self.model_results[name].get('training_time', 0) for name in model_names]
        prediction_times = [self.model_results[name].get('prediction_time', 0) for name in model_names]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=training_times,
                name='訓練時間',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=2, col=3
        )
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=prediction_times,
                name='預測時間',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=2, col=3
        )
        
        # 7. 綜合評分
        composite_scores = self._calculate_composite_scores()
        
        fig.add_trace(
            go.Bar(
                x=list(composite_scores.keys()),
                y=list(composite_scores.values()),
                marker_color=[self.model_colors.get(name, '#1f77b4') for name in composite_scores.keys()],
                text=[f'{score:.3f}' for score in composite_scores.values()],
                textposition='auto'
            ),
            row=3, col=3
        )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title={
                'text': '模型性能綜合比較儀表板',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=True,
            template='plotly_white'
        )
        
        # 更新軸標籤
        fig.update_xaxes(title_text="假正例率", row=1, col=1)
        fig.update_yaxes(title_text="真正例率", row=1, col=1)
        fig.update_xaxes(title_text="召回率", row=1, col=2)
        fig.update_yaxes(title_text="精確率", row=1, col=2)
        fig.update_xaxes(title_text="模型", row=2, col=3)
        fig.update_yaxes(title_text="時間 (秒)", row=2, col=3)
        fig.update_xaxes(title_text="模型", row=3, col=3)
        fig.update_yaxes(title_text="綜合評分", row=3, col=3)
        
        return fig
    
    def _add_feature_importance_comparison(self, fig: go.Figure, models_with_importance: Dict, row: int, col: int):
        """添加特徵重要性比較"""
        # 獲取所有特徵名稱
        all_features = set()
        for model_data in models_with_importance.values():
            if model_data.get('feature_names'):
                all_features.update(model_data['feature_names'])
        
        if not all_features:
            return
        
        # 選擇前15個最重要的特徵
        top_features = list(all_features)[:15]
        
        for i, (model_name, model_data) in enumerate(models_with_importance.items()):
            feature_importance = model_data.get('feature_importance', [])
            feature_names = model_data.get('feature_names', [])
            
            if len(feature_importance) == len(feature_names):
                # 創建特徵重要性字典
                importance_dict = dict(zip(feature_names, feature_importance))
                
                # 獲取top特徵的重要性
                importance_values = [importance_dict.get(feature, 0) for feature in top_features]
                
                fig.add_trace(
                    go.Bar(
                        x=top_features,
                        y=importance_values,
                        name=model_name,
                        marker_color=self.model_colors.get(model_name, '#1f77b4'),
                        opacity=0.7,
                        offsetgroup=i
                    ),
                    row=row, col=col
                )
    
    def _calculate_composite_scores(self) -> Dict[str, float]:
        """計算綜合評分"""
        composite_scores = {}
        
        for model_name, model_data in self.model_results.items():
            metrics = model_data['metrics']
            
            # 加權綜合評分 (可以根據業務需求調整權重)
            score = (
                metrics['roc_auc'] * 0.3 +
                metrics['f1_score'] * 0.25 +
                metrics['precision'] * 0.2 +
                metrics['recall'] * 0.15 +
                metrics['accuracy'] * 0.1
            )
            
            # 考慮時間效率 (時間越短分數越高)
            training_time = model_data.get('training_time', 1)
            time_penalty = min(1.0, 1.0 / (1 + training_time / 60))  # 60秒基準
            
            composite_scores[model_name] = score * (0.9 + 0.1 * time_penalty)
        
        return composite_scores
    
    def create_feature_importance_analysis(self, top_n: int = 20) -> go.Figure:
        """創建特徵重要性分析"""
        models_with_importance = {
            name: data for name, data in self.model_results.items() 
            if data.get('feature_importance') is not None and data.get('feature_names') is not None
        }
        
        if not models_with_importance:
            logger.warning("沒有模型提供特徵重要性數據")
            return go.Figure()
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '特徵重要性對比', '特徵重要性差異分析',
                '特徵共識分析', '特徵穩定性評估'
            ]
        )
        
        # 1. 特徵重要性對比
        all_features = set()
        for model_data in models_with_importance.values():
            all_features.update(model_data['feature_names'])
        
        # 計算每個特徵在不同模型中的平均重要性
        feature_avg_importance = {}
        for feature in all_features:
            importances = []
            for model_data in models_with_importance.values():
                feature_names = model_data['feature_names']
                feature_importance = model_data['feature_importance']
                
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    importances.append(feature_importance[idx])
            
            if importances:
                feature_avg_importance[feature] = np.mean(importances)
        
        # 選擇top特徵
        top_features = sorted(feature_avg_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [f[0] for f in top_features]
        
        # 為每個模型添加條形圖
        for i, (model_name, model_data) in enumerate(models_with_importance.items()):
            feature_names = model_data['feature_names']
            feature_importance = model_data['feature_importance']
            importance_dict = dict(zip(feature_names, feature_importance))
            
            y_values = [importance_dict.get(feature, 0) for feature in top_feature_names]
            
            fig.add_trace(
                go.Bar(
                    x=top_feature_names,
                    y=y_values,
                    name=model_name,
                    marker_color=self.model_colors.get(model_name, '#1f77b4'),
                    opacity=0.7,
                    offsetgroup=i
                ),
                row=1, col=1
            )
        
        # 2. 特徵重要性差異分析
        if len(models_with_importance) >= 2:
            model_names = list(models_with_importance.keys())
            model1_name, model2_name = model_names[0], model_names[1]
            
            model1_data = models_with_importance[model1_name]
            model2_data = models_with_importance[model2_name]
            
            # 計算特徵重要性差異
            common_features = set(model1_data['feature_names']) & set(model2_data['feature_names'])
            
            differences = []
            feature_list = []
            
            for feature in common_features:
                idx1 = model1_data['feature_names'].index(feature)
                idx2 = model2_data['feature_names'].index(feature)
                
                imp1 = model1_data['feature_importance'][idx1]
                imp2 = model2_data['feature_importance'][idx2]
                
                differences.append(imp1 - imp2)
                feature_list.append(feature)
            
            # 排序並選擇top差異
            diff_data = list(zip(feature_list, differences))
            diff_data.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_diff_features = [d[0] for d in diff_data[:15]]
            top_diff_values = [d[1] for d in diff_data[:15]]
            
            colors = ['red' if d > 0 else 'blue' for d in top_diff_values]
            
            fig.add_trace(
                go.Bar(
                    x=top_diff_features,
                    y=top_diff_values,
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. 特徵共識分析 - 顯示在多個模型中都重要的特徵
        feature_consensus = {}
        
        for feature in all_features:
            rankings = []
            for model_data in models_with_importance.values():
                feature_names = model_data['feature_names']
                feature_importance = model_data['feature_importance']
                
                if feature in feature_names:
                    # 計算特徵排名 (重要性越高排名越靠前)
                    sorted_features = sorted(zip(feature_names, feature_importance), 
                                           key=lambda x: x[1], reverse=True)
                    ranking = next(i for i, (name, _) in enumerate(sorted_features) if name == feature)
                    rankings.append(ranking)
            
            if len(rankings) >= 2:  # 至少在兩個模型中出現
                feature_consensus[feature] = np.mean(rankings)
        
        # 選擇共識度最高的特徵 (平均排名最靠前)
        consensus_features = sorted(feature_consensus.items(), key=lambda x: x[1])[:15]
        consensus_names = [f[0] for f in consensus_features]
        consensus_scores = [1.0 / (1 + f[1]) for f in consensus_features]  # 轉換為分數
        
        fig.add_trace(
            go.Bar(
                x=consensus_names,
                y=consensus_scores,
                marker_color='green',
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title={
                'text': '特徵重要性綜合分析',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            template='plotly_white'
        )
        
        # 更新軸標籤
        fig.update_xaxes(title_text="特徵名稱", row=1, col=1)
        fig.update_yaxes(title_text="重要性分數", row=1, col=1)
        fig.update_xaxes(title_text="特徵名稱", row=1, col=2)
        fig.update_yaxes(title_text="重要性差異", row=1, col=2)
        fig.update_xaxes(title_text="特徵名稱", row=2, col=1)
        fig.update_yaxes(title_text="共識分數", row=2, col=1)
        
        return fig
    
    def create_decision_boundary_analysis(self, X: np.ndarray, y: np.ndarray, 
                                        feature_names: List[str] = None) -> go.Figure:
        """創建決策邊界分析"""
        if X.shape[1] < 2:
            logger.warning("決策邊界分析需要至少2個特徵")
            return go.Figure()
        
        logger.info("創建決策邊界分析...")
        
        # 使用PCA降維到2D
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
        
        # 創建子圖
        n_models = len(self.model_results)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'{name} 決策邊界' for name in self.model_results.keys()]
        )
        
        # 創建網格點用於繪製決策邊界
        h = 0.02
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        for i, (model_name, model_data) in enumerate(self.model_results.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            # 繪製數據點
            fraud_mask = y == 1
            
            # 正常交易
            fig.add_trace(
                go.Scatter(
                    x=X_2d[~fraud_mask, 0],
                    y=X_2d[~fraud_mask, 1],
                    mode='markers',
                    name='正常交易',
                    marker=dict(color='blue', size=4, opacity=0.6),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # 詐騙交易
            fig.add_trace(
                go.Scatter(
                    x=X_2d[fraud_mask, 0],
                    y=X_2d[fraud_mask, 1],
                    mode='markers',
                    name='詐騙交易',
                    marker=dict(color='red', size=4, opacity=0.6),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=300 * rows,
            title={
                'text': '模型決策邊界比較',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def create_error_analysis(self) -> go.Figure:
        """創建錯誤分析"""
        if len(self.model_results) < 2:
            return go.Figure()
        
        logger.info("創建模型錯誤分析...")
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '錯誤類型分佈', '模型一致性分析',
                '錯誤相關性分析', '困難樣本識別'
            ]
        )
        
        # 1. 錯誤類型分佈
        error_types = {}
        for model_name, model_data in self.model_results.items():
            y_true = np.array(model_data['data']['y_true'])
            y_pred = np.array(model_data['data']['y_pred'])
            
            fp = np.sum((y_true == 0) & (y_pred == 1))  # 假正例
            fn = np.sum((y_true == 1) & (y_pred == 0))  # 假負例
            
            error_types[model_name] = {'假正例': fp, '假負例': fn}
        
        model_names = list(error_types.keys())
        fp_counts = [error_types[name]['假正例'] for name in model_names]
        fn_counts = [error_types[name]['假負例'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=fp_counts, name='假正例', marker_color='orange'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=fn_counts, name='假負例', marker_color='red'),
            row=1, col=1
        )
        
        # 2. 模型一致性分析
        if len(self.model_results) >= 2:
            model_list = list(self.model_results.items())
            consistency_matrix = np.zeros((len(model_list), len(model_list)))
            
            for i, (name1, data1) in enumerate(model_list):
                for j, (name2, data2) in enumerate(model_list):
                    y_pred1 = np.array(data1['data']['y_pred'])
                    y_pred2 = np.array(data2['data']['y_pred'])
                    
                    # 計算預測一致性
                    consistency = np.mean(y_pred1 == y_pred2)
                    consistency_matrix[i, j] = consistency
            
            fig.add_trace(
                go.Heatmap(
                    z=consistency_matrix,
                    x=[name for name, _ in model_list],
                    y=[name for name, _ in model_list],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="一致性"),
                    text=np.round(consistency_matrix, 3),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=800,
            title={
                'text': '模型錯誤分析',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def generate_comparison_report(self, output_path: str = None) -> Dict[str, Any]:
        """生成模型比較報告"""
        if not self.model_results:
            raise ValueError("沒有模型結果可供比較")
        
        logger.info("生成模型比較報告...")
        
        # 計算排名
        metrics_ranking = {}
        for metric in ['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy']:
            ranking = sorted(
                self.model_results.items(),
                key=lambda x: x[1]['metrics'][metric],
                reverse=True
            )
            metrics_ranking[metric] = [(name, data['metrics'][metric]) for name, data in ranking]
        
        # 綜合排名
        composite_scores = self._calculate_composite_scores()
        overall_ranking = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 生成報告
        report = {
            'generation_time': datetime.now().isoformat(),
            'models_compared': len(self.model_results),
            'overall_ranking': overall_ranking,
            'metrics_ranking': metrics_ranking,
            'best_model': {
                'name': overall_ranking[0][0],
                'score': overall_ranking[0][1],
                'metrics': self.model_results[overall_ranking[0][0]]['metrics']
            },
            'performance_summary': {
                name: data['metrics'] for name, data in self.model_results.items()
            }
        }
        
        # 保存報告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"模型比較報告已保存至: {output_path}")
        
        return report
    
    def create_comprehensive_comparison_dashboard(self) -> go.Figure:
        """創建綜合比較儀表板"""
        logger.info("創建綜合模型比較儀表板...")
        
        # 主儀表板
        performance_fig = self.create_performance_comparison_dashboard()
        
        return performance_fig

# 便捷函數
def compare_fraud_detection_models(model_results: Dict[str, Dict], 
                                 output_dir: str = 'model_comparison') -> Dict[str, str]:
    """比較詐騙檢測模型的便捷函數"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    comparator = ModelComparisonVisualizer()
    
    # 添加模型結果
    for model_name, results in model_results.items():
        comparator.add_model_results(model_name, results)
    
    # 生成可視化
    dashboard_fig = comparator.create_comprehensive_comparison_dashboard()
    dashboard_path = os.path.join(output_dir, 'model_comparison_dashboard.html')
    dashboard_fig.write_html(dashboard_path)
    
    # 生成特徵重要性分析
    feature_fig = comparator.create_feature_importance_analysis()
    feature_path = os.path.join(output_dir, 'feature_importance_analysis.html')
    feature_fig.write_html(feature_path)
    
    # 生成錯誤分析
    error_fig = comparator.create_error_analysis()
    error_path = os.path.join(output_dir, 'error_analysis.html')
    error_fig.write_html(error_path)
    
    # 生成比較報告
    report_path = os.path.join(output_dir, 'comparison_report.json')
    comparator.generate_comparison_report(report_path)
    
    return {
        'dashboard': dashboard_path,
        'feature_analysis': feature_path,
        'error_analysis': error_path,
        'report': report_path
    }

if __name__ == "__main__":
    print("模型比較可視化模組已載入完成！")