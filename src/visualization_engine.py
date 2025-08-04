"""
可視化引擎 - IEEE-CIS 詐騙檢測項目
提供全面的視覺化分析和儀表板功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from plotly.graph_objs import Figure
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import json
import logging
from datetime import datetime, timedelta
import os

from .config import get_config
from .exceptions import DataValidationError, handle_exception

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """核心可視化引擎"""
    
    def __init__(self, config_manager=None, theme='plotly_white'):
        self.config = config_manager or get_config()
        self.theme = theme
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'fraud': '#d62728',
            'legitimate': '#2ca02c',
            'neutral': '#7f7f7f'
        }
        
        # 設置默認樣式
        plt.style.use('default')
        sns.set_palette('husl')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def create_model_performance_dashboard(self, 
                                         y_true: np.ndarray, 
                                         y_pred: np.ndarray, 
                                         y_pred_proba: np.ndarray,
                                         model_names: List[str] = None,
                                         save_path: str = None) -> Figure:
        """創建模型性能綜合儀表板"""
        logger.info("創建模型性能儀表板...")
        
        # 確保輸入數據格式正確
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if isinstance(y_pred_proba, list):
            y_pred_proba = np.array(y_pred_proba)
        
        # 創建子圖布局
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'ROC曲線', '混淆矩陣', 'Precision-Recall曲線',
                '特徵重要性', '預測概率分佈', '模型比較',
                '性能指標雷達圖', '分類報告', '預測置信度'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatterpolar"}, {"type": "table"}, {"type": "box"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. ROC曲線
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color=self.color_palette['primary'], width=3)
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
        
        # 2. 混淆矩陣
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
            row=1, col=2
        )
        
        # 3. Precision-Recall曲線
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'PR (AUC = {pr_auc:.3f})',
                line=dict(color=self.color_palette['secondary'], width=3)
            ),
            row=1, col=3
        )
        
        # 4. 預測概率分佈
        fig.add_trace(
            go.Histogram(
                x=y_pred_proba[y_true == 0],
                name='正常交易',
                opacity=0.7,
                nbinsx=50,
                marker_color=self.color_palette['legitimate']
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=y_pred_proba[y_true == 1],
                name='詐騙交易',
                opacity=0.7,
                nbinsx=50,
                marker_color=self.color_palette['fraud']
            ),
            row=2, col=2
        )
        
        # 5. 預測置信度箱線圖
        confidence_data = [
            y_pred_proba[y_true == 0],
            y_pred_proba[y_true == 1]
        ]
        
        for i, (data, name, color) in enumerate(zip(
            confidence_data, 
            ['正常交易', '詐騙交易'],
            [self.color_palette['legitimate'], self.color_palette['fraud']]
        )):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=name,
                    marker_color=color
                ),
                row=3, col=3
            )
        
        # 更新布局
        fig.update_layout(
            height=1200,
            title={
                'text': '模型性能綜合分析儀表板',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template=self.theme,
            showlegend=True
        )
        
        # 更新子圖標題
        fig.update_xaxes(title_text="假正例率", row=1, col=1)
        fig.update_yaxes(title_text="真正例率", row=1, col=1)
        fig.update_xaxes(title_text="召回率", row=1, col=3)
        fig.update_yaxes(title_text="精確率", row=1, col=3)
        fig.update_xaxes(title_text="預測概率", row=2, col=2)
        fig.update_yaxes(title_text="頻次", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"模型性能儀表板已保存至: {save_path}")
        
        return fig
    
    def create_feature_importance_chart(self, 
                                      feature_names: List[str], 
                                      importance_scores: np.ndarray,
                                      top_n: int = 20,
                                      chart_type: str = 'horizontal_bar') -> Figure:
        """創建特徵重要性圖表"""
        logger.info(f"創建特徵重要性圖表 (Top {top_n})...")
        
        # 數據預處理
        if len(feature_names) != len(importance_scores):
            raise ValueError("特徵名稱和重要性分數長度不匹配")
        
        # 排序並選擇前N個特徵
        sorted_indices = np.argsort(importance_scores)[::-1][:top_n]
        top_features = [feature_names[i] for i in sorted_indices]
        top_scores = importance_scores[sorted_indices]
        
        if chart_type == 'horizontal_bar':
            fig = go.Figure(data=[
                go.Bar(
                    x=top_scores[::-1],
                    y=top_features[::-1],
                    orientation='h',
                    marker=dict(
                        color=top_scores[::-1],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="重要性分數")
                    ),
                    text=[f'{score:.4f}' for score in top_scores[::-1]],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f'Top {top_n} 特徵重要性',
                xaxis_title='重要性分數',
                yaxis_title='特徵名稱',
                height=max(400, top_n * 20),
                template=self.theme
            )
        
        elif chart_type == 'treemap':
            fig = go.Figure(data=[
                go.Treemap(
                    labels=top_features,
                    values=top_scores,
                    parents=[""] * len(top_features),
                    textinfo="label+value",
                    texttemplate="<b>%{label}</b><br>%{value:.4f}",
                    colorscale='Viridis'
                )
            ])
            
            fig.update_layout(
                title=f'特徵重要性樹狀圖 (Top {top_n})',
                height=600,
                template=self.theme
            )
        
        return fig
    
    def create_transaction_pattern_analysis(self, 
                                          df: pd.DataFrame,
                                          time_col: str = 'TransactionDT',
                                          amount_col: str = 'TransactionAmt',
                                          fraud_col: str = 'isFraud') -> Figure:
        """創建交易模式分析"""
        logger.info("創建交易模式分析...")
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '時間分佈分析', '金額分佈分析',
                '詐騙率時間趨勢', '交易量vs詐騙率'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}]
            ]
        )
        
        # 1. 時間分佈分析
        if time_col in df.columns:
            # 轉換時間戳為小時
            df['hour'] = (df[time_col] / 3600) % 24
            
            # 正常交易和詐騙交易的時間分佈
            normal_hours = df[df[fraud_col] == 0]['hour']
            fraud_hours = df[df[fraud_col] == 1]['hour']
            
            fig.add_trace(
                go.Histogram(
                    x=normal_hours,
                    name='正常交易',
                    opacity=0.7,
                    nbinsx=24,
                    marker_color=self.color_palette['legitimate']
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Histogram(
                    x=fraud_hours,
                    name='詐騙交易',
                    opacity=0.7,
                    nbinsx=24,
                    marker_color=self.color_palette['fraud']
                ),
                row=1, col=1
            )
        
        # 2. 金額分佈分析
        if amount_col in df.columns:
            # 使用對數尺度來更好地顯示分佈
            normal_amounts = np.log1p(df[df[fraud_col] == 0][amount_col])
            fraud_amounts = np.log1p(df[df[fraud_col] == 1][amount_col])
            
            fig.add_trace(
                go.Histogram(
                    x=normal_amounts,
                    name='正常交易',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color=self.color_palette['legitimate']
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=fraud_amounts,
                    name='詐騙交易',
                    opacity=0.7,
                    nbinsx=50,
                    marker_color=self.color_palette['fraud']
                ),
                row=1, col=2
            )
        
        # 3. 詐騙率時間趨勢（如果有時間列）
        if time_col in df.columns and 'hour' in df.columns:
            hourly_stats = df.groupby('hour').agg({
                fraud_col: ['count', 'sum']
            }).round(4)
            
            hourly_stats.columns = ['total_transactions', 'fraud_count']
            hourly_stats['fraud_rate'] = hourly_stats['fraud_count'] / hourly_stats['total_transactions']
            
            # 交易量
            fig.add_trace(
                go.Bar(
                    x=hourly_stats.index,
                    y=hourly_stats['total_transactions'],
                    name='交易量',
                    marker_color=self.color_palette['info'],
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # 詐騙率
            fig.add_trace(
                go.Scatter(
                    x=hourly_stats.index,
                    y=hourly_stats['fraud_rate'],
                    mode='lines+markers',
                    name='詐騙率',
                    line=dict(color=self.color_palette['danger'], width=3),
                    yaxis='y2'
                ),
                row=2, col=1, secondary_y=True
            )
        
        # 4. 交易量vs詐騙率散點圖
        if amount_col in df.columns:
            # 按金額區間分組
            df['amount_bin'] = pd.cut(df[amount_col], bins=20, labels=False)
            amount_analysis = df.groupby('amount_bin').agg({
                fraud_col: ['count', 'sum', 'mean']
            }).round(4)
            
            amount_analysis.columns = ['transaction_count', 'fraud_count', 'fraud_rate']
            amount_analysis = amount_analysis.dropna()
            
            fig.add_trace(
                go.Scatter(
                    x=amount_analysis['transaction_count'],
                    y=amount_analysis['fraud_rate'],
                    mode='markers',
                    marker=dict(
                        size=amount_analysis['fraud_count'],
                        sizemode='diameter',
                        sizeref=2.*max(amount_analysis['fraud_count'])/(40.**2),
                        sizemin=4,
                        color=amount_analysis['fraud_rate'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="詐騙率")
                    ),
                    text=[f'詐騙數: {int(count)}' for count in amount_analysis['fraud_count']],
                    hovertemplate='交易量: %{x}<br>詐騙率: %{y:.4f}<br>%{text}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title={
                'text': '交易模式綜合分析',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            template=self.theme,
            showlegend=True
        )
        
        # 更新軸標籤
        fig.update_xaxes(title_text="小時", row=1, col=1)
        fig.update_yaxes(title_text="交易數量", row=1, col=1)
        fig.update_xaxes(title_text="log(交易金額 + 1)", row=1, col=2)
        fig.update_yaxes(title_text="頻次", row=1, col=2)
        
        if time_col in df.columns and 'hour' in df.columns:
            fig.update_xaxes(title_text="小時", row=2, col=1)
            fig.update_yaxes(title_text="交易量", row=2, col=1)
            fig.update_yaxes(title_text="詐騙率", row=2, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="交易量", row=2, col=2)
        fig.update_yaxes(title_text="詐騙率", row=2, col=2)
        
        return fig
    
    def create_geographic_analysis(self, 
                                 df: pd.DataFrame,
                                 location_cols: List[str] = None) -> Figure:
        """創建地理位置分析"""
        logger.info("創建地理分析...")
        
        if location_cols is None:
            # 自動檢測地理相關列
            location_cols = [col for col in df.columns if any(
                keyword in col.lower() for keyword in ['addr', 'country', 'state', 'city']
            )]
        
        if not location_cols:
            logger.warning("未找到地理位置相關欄位")
            return go.Figure()
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '地區詐騙率分佈', '地區交易量分佈',
                '高風險地區排名', '地區風險熱力圖'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # 分析第一個地理欄位
        main_location_col = location_cols[0]
        if main_location_col in df.columns:
            # 計算各地區統計
            location_stats = df.groupby(main_location_col).agg({
                'isFraud': ['count', 'sum', 'mean']
            }).round(4)
            
            location_stats.columns = ['transaction_count', 'fraud_count', 'fraud_rate']
            location_stats = location_stats.sort_values('fraud_rate', ascending=False).head(20)
            
            # 1. 地區詐騙率分佈
            fig.add_trace(
                go.Bar(
                    x=location_stats.index,
                    y=location_stats['fraud_rate'],
                    marker_color=location_stats['fraud_rate'],
                    marker_colorscale='Reds',
                    text=[f'{rate:.3f}' for rate in location_stats['fraud_rate']],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # 2. 地區交易量分佈
            fig.add_trace(
                go.Bar(
                    x=location_stats.index,
                    y=location_stats['transaction_count'],
                    marker_color=self.color_palette['info'],
                    text=[f'{count:,}' for count in location_stats['transaction_count']],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. 高風險地區排名
            high_risk_regions = location_stats.head(10)
            fig.add_trace(
                go.Bar(
                    x=high_risk_regions['fraud_rate'],
                    y=high_risk_regions.index,
                    orientation='h',
                    marker_color=self.color_palette['danger'],
                    text=[f'{rate:.3f}' for rate in high_risk_regions['fraud_rate']],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title={
                'text': '地理位置風險分析',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            template=self.theme,
            showlegend=False
        )
        
        # 更新軸標籤
        fig.update_xaxes(title_text="地區", row=1, col=1)
        fig.update_yaxes(title_text="詐騙率", row=1, col=1)
        fig.update_xaxes(title_text="地區", row=1, col=2)
        fig.update_yaxes(title_text="交易量", row=1, col=2)
        fig.update_xaxes(title_text="詐騙率", row=2, col=1)
        fig.update_yaxes(title_text="地區", row=2, col=1)
        
        return fig
    
    def create_real_time_monitoring_dashboard(self, 
                                            recent_data: pd.DataFrame,
                                            alerts: List[Dict] = None,
                                            performance_metrics: Dict = None) -> Figure:
        """創建實時監控儀表板"""
        logger.info("創建實時監控儀表板...")
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '實時交易量', '詐騙檢測率', '系統性能',
                '風險分數分佈', '模型置信度', '警報狀態',
                '處理時間趨勢', '錯誤率趨勢', '系統健康度'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "histogram"}, {"type": "gauge"}, {"type": "table"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "indicator"}]
            ]
        )
        
        # 計算實時指標
        if not recent_data.empty:
            total_transactions = len(recent_data)
            fraud_count = recent_data['isFraud'].sum() if 'isFraud' in recent_data.columns else 0
            fraud_rate = fraud_count / total_transactions if total_transactions > 0 else 0
            
            # 1. 實時交易量指示器
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=total_transactions,
                    title={"text": "實時交易量"},
                    delta={'reference': total_transactions * 0.9, 'valueformat': ','},
                    number={'valueformat': ','},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=1
            )
            
            # 2. 詐騙檢測率指示器
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge+delta",
                    value=fraud_rate * 100,
                    title={"text": "詐騙檢測率 (%)"},
                    delta={'reference': 3.5, 'valueformat': '.2f'},
                    gauge={
                        'axis': {'range': [None, 10]},
                        'bar': {'color': "red"},
                        'steps': [
                            {'range': [0, 2], 'color': "lightgray"},
                            {'range': [2, 5], 'color': "yellow"},
                            {'range': [5, 10], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 5
                        }
                    },
                    number={'valueformat': '.2f'},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=2
            )
            
            # 3. 系統性能指示器
            system_performance = performance_metrics.get('average_response_time', 0.1) if performance_metrics else 0.1
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=system_performance * 1000,  # 轉換為毫秒
                    title={"text": "平均響應時間 (ms)"},
                    gauge={
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 100], 'color': "lightgreen"},
                            {'range': [100, 300], 'color': "yellow"},
                            {'range': [300, 1000], 'color': "red"}
                        ]
                    },
                    number={'valueformat': '.0f'},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=3
            )
            
            # 4. 風險分數分佈
            if 'risk_score' in recent_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=recent_data['risk_score'],
                        nbinsx=30,
                        marker_color=self.color_palette['warning'],
                        opacity=0.7
                    ),
                    row=2, col=1
                )
        
        # 5. 警報狀態表格
        if alerts:
            alert_df = pd.DataFrame(alerts)
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['時間', '類型', '嚴重程度', '狀態'],
                        fill_color='paleturquoise',
                        align='left'
                    ),
                    cells=dict(
                        values=[
                            alert_df.get('timestamp', []),
                            alert_df.get('alert_type', []),
                            alert_df.get('severity', []),
                            alert_df.get('status', [])
                        ],
                        fill_color='lavender',
                        align='left'
                    )
                ),
                row=2, col=3
            )
        
        # 更新布局
        fig.update_layout(
            height=1000,
            title={
                'text': '實時監控儀表板',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template=self.theme,
            showlegend=False
        )
        
        return fig
    
    def create_business_intelligence_report(self, 
                                          df: pd.DataFrame,
                                          financial_impact: Dict = None,
                                          time_period: str = "本月") -> Figure:
        """創建商業智能報告"""
        logger.info("創建商業智能報告...")
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '財務影響概覽', '詐騙趨勢分析',
                '風險等級分佈', '業務部門影響',
                '預防效果評估', '投資回報分析'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ]
        )
        
        # 計算關鍵指標
        if 'TransactionAmt' in df.columns and 'isFraud' in df.columns:
            total_amount = df['TransactionAmt'].sum()
            fraud_amount = df[df['isFraud'] == 1]['TransactionAmt'].sum()
            prevented_loss = financial_impact.get('prevented_loss', 0) if financial_impact else 0
            
            # 1. 財務影響指示器
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=fraud_amount,
                    title={"text": f"{time_period}詐騙損失"},
                    delta={'reference': fraud_amount * 1.1, 'valueformat': '$,.0f'},
                    number={'valueformat': '$,.0f'},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=1
            )
            
            # 2. 詐騙趨勢分析（按時間）
            if 'TransactionDT' in df.columns:
                # 按天聚合
                df['date'] = pd.to_datetime(df['TransactionDT'], unit='s')
                daily_fraud = df.groupby(df['date'].dt.date).agg({
                    'isFraud': ['sum', 'count'],
                    'TransactionAmt': 'sum'
                }).round(2)
                
                daily_fraud.columns = ['fraud_count', 'total_count', 'total_amount']
                daily_fraud['fraud_rate'] = daily_fraud['fraud_count'] / daily_fraud['total_count']
                
                fig.add_trace(
                    go.Scatter(
                        x=daily_fraud.index,
                        y=daily_fraud['fraud_rate'],
                        mode='lines+markers',
                        name='詐騙率趨勢',
                        line=dict(color=self.color_palette['danger'], width=2)
                    ),
                    row=1, col=2
                )
            
            # 3. 風險等級分佈（假設有風險分級）
            risk_levels = ['低風險', '中風險', '高風險', '極高風險']
            risk_counts = [
                len(df[(df['isFraud'] == 0) & (df.get('risk_score', 0) < 0.3)]),
                len(df[(df.get('risk_score', 0) >= 0.3) & (df.get('risk_score', 0) < 0.6)]),
                len(df[(df.get('risk_score', 0) >= 0.6) & (df.get('risk_score', 0) < 0.8)]),
                len(df[df.get('risk_score', 1) >= 0.8])
            ]
            
            fig.add_trace(
                go.Pie(
                    labels=risk_levels,
                    values=risk_counts,
                    marker_colors=['green', 'yellow', 'orange', 'red']
                ),
                row=2, col=1
            )
            
            # 4. 投資回報指示器
            if financial_impact:
                roi = financial_impact.get('roi', 0)
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=roi * 100,
                        title={"text": "投資回報率 (%)"},
                        gauge={
                            'axis': {'range': [0, 500]},
                            'bar': {'color': "green"},
                            'steps': [
                                {'range': [0, 100], 'color': "lightgray"},
                                {'range': [100, 200], 'color': "yellow"},
                                {'range': [200, 500], 'color': "green"}
                            ]
                        },
                        number={'valueformat': '.1f'},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ),
                    row=3, col=2
                )
        
        # 更新布局
        fig.update_layout(
            height=1000,
            title={
                'text': f'{time_period}商業智能報告',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def save_dashboard(self, fig: Figure, save_path: str, format_type: str = 'html'):
        """保存儀表板"""
        try:
            if format_type.lower() == 'html':
                fig.write_html(save_path)
            elif format_type.lower() == 'png':
                fig.write_image(save_path, width=1200, height=800)
            elif format_type.lower() == 'pdf':
                fig.write_image(save_path, format='pdf', width=1200, height=800)
            else:
                raise ValueError(f"不支持的格式: {format_type}")
            
            logger.info(f"儀表板已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存儀表板失敗: {e}")
            raise
    
    def create_comprehensive_report(self, 
                                  df: pd.DataFrame,
                                  model_results: Dict = None,
                                  output_dir: str = 'visualization_reports') -> Dict[str, str]:
        """創建綜合可視化報告"""
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"創建綜合可視化報告，保存至: {output_dir}")
        
        report_files = {}
        
        try:
            # 1. 交易模式分析
            if 'TransactionDT' in df.columns and 'TransactionAmt' in df.columns:
                pattern_fig = self.create_transaction_pattern_analysis(df)
                pattern_path = os.path.join(output_dir, 'transaction_patterns.html')
                self.save_dashboard(pattern_fig, pattern_path)
                report_files['transaction_patterns'] = pattern_path
            
            # 2. 地理分析
            geo_fig = self.create_geographic_analysis(df)
            if geo_fig.data:  # 只有在有數據時才保存
                geo_path = os.path.join(output_dir, 'geographic_analysis.html')
                self.save_dashboard(geo_fig, geo_path)
                report_files['geographic_analysis'] = geo_path
            
            # 3. 模型性能分析（如果有模型結果）
            if model_results:
                perf_fig = self.create_model_performance_dashboard(
                    model_results.get('y_true', []),
                    model_results.get('y_pred', []),
                    model_results.get('y_pred_proba', [])
                )
                perf_path = os.path.join(output_dir, 'model_performance.html')
                self.save_dashboard(perf_fig, perf_path)
                report_files['model_performance'] = perf_path
            
            # 4. 商業智能報告
            bi_fig = self.create_business_intelligence_report(df)
            bi_path = os.path.join(output_dir, 'business_intelligence.html')
            self.save_dashboard(bi_fig, bi_path)
            report_files['business_intelligence'] = bi_path
            
            # 5. 生成索引頁面
            index_path = self._create_report_index(output_dir, report_files)
            report_files['index'] = index_path
            
            logger.info(f"綜合報告生成完成，包含 {len(report_files)} 個文件")
            
        except Exception as e:
            logger.error(f"生成綜合報告時出錯: {e}")
            raise
        
        return report_files
    
    def _create_report_index(self, output_dir: str, report_files: Dict[str, str]) -> str:
        """創建報告索引頁面"""
        index_path = os.path.join(output_dir, 'index.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>詐騙檢測系統 - 可視化報告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
                .report-item {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa; }}
                .report-item h3 {{ margin-top: 0; color: #666; }}
                .report-link {{ display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 5px; }}
                .report-link:hover {{ background-color: #0056b3; }}
                .timestamp {{ text-align: center; color: #888; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>詐騙檢測系統 - 可視化報告中心</h1>
                
                <div class="report-item">
                    <h3>📊 交易模式分析</h3>
                    <p>分析交易的時間分佈、金額分佈和詐騙趨勢</p>
                    <a href="transaction_patterns.html" class="report-link">查看報告</a>
                </div>
                
                <div class="report-item">
                    <h3>🗺️ 地理位置分析</h3>
                    <p>分析不同地區的詐騙率和風險分佈</p>
                    <a href="geographic_analysis.html" class="report-link">查看報告</a>
                </div>
                
                <div class="report-item">
                    <h3>🎯 模型性能分析</h3>
                    <p>模型準確率、ROC曲線、混淆矩陣等性能指標</p>
                    <a href="model_performance.html" class="report-link">查看報告</a>
                </div>
                
                <div class="report-item">
                    <h3>💼 商業智能報告</h3>
                    <p>財務影響、投資回報、業務指標分析</p>
                    <a href="business_intelligence.html" class="report-link">查看報告</a>
                </div>
                
                <div class="timestamp">
                    報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return index_path

# 便捷函數
def create_fraud_detection_visualizations(df: pd.DataFrame, 
                                        model_results: Dict = None,
                                        output_dir: str = 'fraud_visualizations') -> Dict[str, str]:
    """創建詐騙檢測可視化的便捷函數"""
    viz_engine = VisualizationEngine()
    return viz_engine.create_comprehensive_report(df, model_results, output_dir)

def quick_performance_dashboard(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray,
                              save_path: str = None) -> Figure:
    """快速創建性能儀表板"""
    viz_engine = VisualizationEngine()
    return viz_engine.create_model_performance_dashboard(
        y_true, y_pred, y_pred_proba, save_path=save_path
    )

if __name__ == "__main__":
    print("可視化引擎已載入完成！")