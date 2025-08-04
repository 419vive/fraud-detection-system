"""
商業分析模組 - IEEE-CIS 詐騙檢測項目
提供深度商業洞察、財務影響分析和策略建議
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from .config import get_config
from .visualization_engine import VisualizationEngine

logger = logging.getLogger(__name__)

@dataclass
class FinancialImpact:
    """財務影響數據類"""
    prevented_fraud_amount: float
    actual_fraud_loss: float
    false_positive_cost: float
    operational_cost: float
    investigation_cost: float
    total_savings: float
    roi_percentage: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass
class BusinessMetrics:
    """業務指標數據類"""
    transaction_volume: int
    fraud_rate: float
    detection_rate: float
    false_positive_rate: float
    average_transaction_amount: float
    customer_satisfaction_impact: float
    processing_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BusinessAnalyzer:
    """商業分析器"""
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.viz_engine = VisualizationEngine(config_manager)
        
        # 業務參數配置
        self.business_params = {
            'average_investigation_cost': 50.0,  # 每筆調查成本
            'false_positive_impact': 25.0,  # 假正例對客戶體驗的成本
            'operational_cost_per_transaction': 0.05,  # 每筆交易運營成本
            'fraud_prevention_value_multiplier': 5.0,  # 防詐騙價值倍數
            'customer_retention_value': 200.0,  # 客戶留存價值
            'brand_reputation_impact': 10000.0  # 品牌聲譽影響成本
        }
    
    def calculate_financial_impact(self, 
                                 df: pd.DataFrame,
                                 predictions: np.ndarray,
                                 prediction_probabilities: np.ndarray,
                                 amount_col: str = 'TransactionAmt',
                                 fraud_col: str = 'isFraud') -> FinancialImpact:
        """計算財務影響"""
        logger.info("計算財務影響分析...")
        
        if amount_col not in df.columns or fraud_col not in df.columns:
            raise ValueError(f"缺少必要欄位: {amount_col} 或 {fraud_col}")
        
        y_true = df[fraud_col].values
        y_pred = predictions
        transaction_amounts = df[amount_col].values
        
        # 真實詐騙損失（未被檢測到的詐騙）
        undetected_fraud_mask = (y_true == 1) & (y_pred == 0)
        actual_fraud_loss = transaction_amounts[undetected_fraud_mask].sum()
        
        # 防止的詐騙損失（被正確檢測的詐騙）
        detected_fraud_mask = (y_true == 1) & (y_pred == 1)
        prevented_fraud_amount = transaction_amounts[detected_fraud_mask].sum()
        
        # 假正例成本（誤判正常交易為詐騙）
        false_positive_mask = (y_true == 0) & (y_pred == 1)
        false_positive_count = false_positive_mask.sum()
        false_positive_cost = (
            false_positive_count * self.business_params['false_positive_impact'] +
            transaction_amounts[false_positive_mask].sum() * 0.01  # 1%的交易延遲成本
        )
        
        # 調查成本
        investigation_cost = (
            (detected_fraud_mask.sum() + false_positive_mask.sum()) * 
            self.business_params['average_investigation_cost']
        )
        
        # 運營成本
        operational_cost = len(df) * self.business_params['operational_cost_per_transaction']
        
        # 總節省
        total_savings = (
            prevented_fraud_amount * self.business_params['fraud_prevention_value_multiplier'] - 
            false_positive_cost - 
            investigation_cost - 
            operational_cost
        )
        
        # ROI計算
        total_investment = investigation_cost + operational_cost
        roi_percentage = (total_savings / total_investment * 100) if total_investment > 0 else 0
        
        return FinancialImpact(
            prevented_fraud_amount=prevented_fraud_amount,
            actual_fraud_loss=actual_fraud_loss,
            false_positive_cost=false_positive_cost,
            operational_cost=operational_cost,
            investigation_cost=investigation_cost,
            total_savings=total_savings,
            roi_percentage=roi_percentage
        )
    
    def calculate_business_metrics(self, 
                                 df: pd.DataFrame,
                                 predictions: np.ndarray,
                                 amount_col: str = 'TransactionAmt',
                                 fraud_col: str = 'isFraud') -> BusinessMetrics:
        """計算業務指標"""
        logger.info("計算業務指標...")
        
        y_true = df[fraud_col].values
        y_pred = predictions
        
        # 基本指標
        transaction_volume = len(df)
        fraud_rate = y_true.mean()
        
        # 檢測率（真正例率/召回率）
        true_positives = ((y_true == 1) & (y_pred == 1)).sum()
        actual_frauds = (y_true == 1).sum()
        detection_rate = true_positives / actual_frauds if actual_frauds > 0 else 0
        
        # 假正例率
        false_positives = ((y_true == 0) & (y_pred == 1)).sum()
        actual_normals = (y_true == 0).sum()
        false_positive_rate = false_positives / actual_normals if actual_normals > 0 else 0
        
        # 平均交易金額
        average_transaction_amount = df[amount_col].mean()
        
        # 客戶滿意度影響（基於假正例率）
        customer_satisfaction_impact = max(0, 100 - false_positive_rate * 10000)
        
        # 處理效率（基於檢測率和假正例率）
        processing_efficiency = max(0, detection_rate * 100 - false_positive_rate * 50)
        
        return BusinessMetrics(
            transaction_volume=transaction_volume,
            fraud_rate=fraud_rate,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            average_transaction_amount=average_transaction_amount,
            customer_satisfaction_impact=customer_satisfaction_impact,
            processing_efficiency=processing_efficiency
        )
    
    def create_financial_impact_dashboard(self, financial_impact: FinancialImpact) -> go.Figure:
        """創建財務影響儀表板"""
        logger.info("創建財務影響儀表板...")
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'ROI指示器', '成本效益分析', '詐騙預防效果',
                '成本結構分析', '收益分解', '投資回報趨勢'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "waterfall"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. ROI指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=financial_impact.roi_percentage,
                title={"text": "投資回報率 (%)"},
                delta={'reference': 100, 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [None, 500]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                        {'range': [100, 200], 'color': "yellow"},
                        {'range': [200, 500], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 150
                    }
                },
                number={'valueformat': '.1f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # 2. 成本效益分析
        cost_benefit_data = {
            '防詐騙收益': financial_impact.prevented_fraud_amount,
            '調查成本': -financial_impact.investigation_cost,
            '假正例成本': -financial_impact.false_positive_cost,
            '運營成本': -financial_impact.operational_cost,
            '淨收益': financial_impact.total_savings
        }
        
        colors = ['green' if v > 0 else 'red' for v in cost_benefit_data.values()]
        
        fig.add_trace(
            go.Bar(
                x=list(cost_benefit_data.keys()),
                y=list(cost_benefit_data.values()),
                marker_color=colors,
                text=[f'${v:,.0f}' for v in cost_benefit_data.values()],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. 詐騙預防效果
        prevention_data = {
            '已防止詐騙': financial_impact.prevented_fraud_amount,
            '未檢測詐騙': financial_impact.actual_fraud_loss
        }
        
        fig.add_trace(
            go.Bar(
                x=list(prevention_data.keys()),
                y=list(prevention_data.values()),
                marker_color=['green', 'red'],
                text=[f'${v:,.0f}' for v in prevention_data.values()],
                textposition='auto',
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. 成本結構分析
        cost_structure = {
            '調查成本': financial_impact.investigation_cost,
            '假正例成本': financial_impact.false_positive_cost,
            '運營成本': financial_impact.operational_cost
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(cost_structure.keys()),
                values=list(cost_structure.values()),
                marker_colors=['#ff9999', '#66b3ff', '#99ff99']
            ),
            row=2, col=1
        )
        
        # 5. 收益分解（瀑布圖）
        fig.add_trace(
            go.Waterfall(
                name="收益分解",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=["防詐騙收益", "調查成本", "假正例成本", "運營成本", "總收益"],
                textposition="auto",
                text=[f"${financial_impact.prevented_fraud_amount:,.0f}",
                      f"-${financial_impact.investigation_cost:,.0f}",
                      f"-${financial_impact.false_positive_cost:,.0f}",
                      f"-${financial_impact.operational_cost:,.0f}",
                      f"${financial_impact.total_savings:,.0f}"],
                y=[financial_impact.prevented_fraud_amount,
                   -financial_impact.investigation_cost,
                   -financial_impact.false_positive_cost,
                   -financial_impact.operational_cost,
                   financial_impact.total_savings],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 6. 總節省指示器
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=financial_impact.total_savings,
                title={"text": "總節省金額 ($)"},
                delta={'reference': 0, 'valueformat': '$,.0f'},
                number={'valueformat': '$,.0f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=3
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title={
                'text': '財務影響分析儀表板',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_business_performance_dashboard(self, 
                                            business_metrics: BusinessMetrics,
                                            time_series_data: pd.DataFrame = None) -> go.Figure:
        """創建業務績效儀表板"""
        logger.info("創建業務績效儀表板...")
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                '檢測率', '假正例率', '客戶滿意度',
                '處理效率', '交易量趨勢', '詐騙率趨勢',
                '業務健康度', '績效雷達圖', '關鍵指標摘要'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "indicator"}, {"type": "scatterpolar"}, {"type": "table"}]
            ]
        )
        
        # 1. 檢測率指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=business_metrics.detection_rate * 100,
                title={"text": "詐騙檢測率 (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                },
                number={'valueformat': '.1f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # 2. 假正例率指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=business_metrics.false_positive_rate * 100,
                title={"text": "假正例率 (%)"},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 2], 'color': "green"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}
                    ]
                },
                number={'valueformat': '.2f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=2
        )
        
        # 3. 客戶滿意度指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=business_metrics.customer_satisfaction_impact,
                title={"text": "客戶滿意度影響"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                },
                number={'valueformat': '.1f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=3
        )
        
        # 4. 處理效率指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=business_metrics.processing_efficiency,
                title={"text": "處理效率"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                },
                number={'valueformat': '.1f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=1
        )
        
        # 5 & 6. 時間序列圖（如果有數據）
        if time_series_data is not None and not time_series_data.empty:
            # 交易量趨勢
            fig.add_trace(
                go.Scatter(
                    x=time_series_data.index,
                    y=time_series_data.get('transaction_volume', []),
                    mode='lines+markers',
                    name='交易量',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=2
            )
            
            # 詐騙率趨勢
            fig.add_trace(
                go.Scatter(
                    x=time_series_data.index,
                    y=time_series_data.get('fraud_rate', []),
                    mode='lines+markers',
                    name='詐騙率',
                    line=dict(color='red', width=2)
                ),
                row=2, col=3
            )
        
        # 7. 業務健康度總分
        health_score = (
            business_metrics.detection_rate * 40 +
            (1 - business_metrics.false_positive_rate) * 30 +
            business_metrics.customer_satisfaction_impact * 0.2 +
            business_metrics.processing_efficiency * 0.1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_score,
                title={"text": "業務健康度"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                },
                number={'valueformat': '.1f'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=3, col=1
        )
        
        # 8. 績效雷達圖
        radar_metrics = [
            '檢測率',
            '準確性',
            '效率',
            '客戶滿意度',
            '成本效益'
        ]
        
        radar_values = [
            business_metrics.detection_rate * 100,
            (1 - business_metrics.false_positive_rate) * 100,
            business_metrics.processing_efficiency,
            business_metrics.customer_satisfaction_impact,
            min(100, health_score)
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name='當前績效',
                line_color='blue'
            ),
            row=3, col=2
        )
        
        # 9. 關鍵指標摘要表
        summary_data = [
            ['交易總量', f'{business_metrics.transaction_volume:,}'],
            ['詐騙率', f'{business_metrics.fraud_rate:.2%}'],
            ['檢測率', f'{business_metrics.detection_rate:.2%}'],
            ['假正例率', f'{business_metrics.false_positive_rate:.2%}'],
            ['平均交易金額', f'${business_metrics.average_transaction_amount:,.2f}'],
            ['客戶滿意度', f'{business_metrics.customer_satisfaction_impact:.1f}'],
            ['處理效率', f'{business_metrics.processing_efficiency:.1f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['指標', '值'], fill_color='paleturquoise', align='left'),
                cells=dict(
                    values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=3
        )
        
        # 更新布局
        fig.update_layout(
            height=1000,
            title={
                'text': '業務績效分析儀表板',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def create_risk_assessment_dashboard(self, df: pd.DataFrame, 
                                       predictions: np.ndarray,
                                       prediction_probabilities: np.ndarray) -> go.Figure:
        """創建風險評估儀表板"""
        logger.info("創建風險評估儀表板...")
        
        # 創建風險分級
        risk_scores = prediction_probabilities
        risk_categories = pd.cut(
            risk_scores,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['低風險', '中風險', '高風險', '極高風險']
        )
        
        # 創建子圖
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                '風險分級分布', '風險金額分布', '時間風險模式',
                '地理風險分布', '風險趨勢預測', '風險控制建議'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "table"}]
            ]
        )
        
        # 1. 風險分級分布
        risk_counts = risk_categories.value_counts()
        colors = ['green', 'yellow', 'orange', 'red']
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=colors[:len(risk_counts)]
            ),
            row=1, col=1
        )
        
        # 2. 風險金額分布
        if 'TransactionAmt' in df.columns:
            risk_amounts = []
            risk_labels = []
            
            for category in ['低風險', '中風險', '高風險', '極高風險']:
                mask = risk_categories == category
                if mask.any():
                    total_amount = df.loc[mask, 'TransactionAmt'].sum()
                    risk_amounts.append(total_amount)
                    risk_labels.append(category)
            
            fig.add_trace(
                go.Bar(
                    x=risk_labels,
                    y=risk_amounts,
                    marker_color=colors[:len(risk_labels)],
                    text=[f'${amount:,.0f}' for amount in risk_amounts],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        # 3. 時間風險模式（如果有時間數據）
        if 'TransactionDT' in df.columns:
            df_temp = df.copy()
            df_temp['hour'] = (df_temp['TransactionDT'] / 3600) % 24
            df_temp['day_of_week'] = ((df_temp['TransactionDT'] / 86400) % 7).astype(int)
            df_temp['risk_score'] = risk_scores
            
            # 創建熱力圖數據
            risk_heatmap = df_temp.groupby(['day_of_week', df_temp['hour'].astype(int)])['risk_score'].mean().unstack()
            
            fig.add_trace(
                go.Heatmap(
                    z=risk_heatmap.values,
                    x=[f'{h:02d}:00' for h in risk_heatmap.columns],
                    y=['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                    colorscale='Reds',
                    showscale=True
                ),
                row=1, col=3
            )
        
        # 4. 地理風險分布（如果有地理數據）
        geo_cols = [col for col in df.columns if 'addr' in col.lower()]
        if geo_cols:
            geo_col = geo_cols[0]
            geo_risk = df.groupby(geo_col)['isFraud'].agg(['count', 'mean']).reset_index()
            geo_risk = geo_risk[geo_risk['count'] >= 10].nlargest(15, 'mean')
            
            fig.add_trace(
                go.Bar(
                    x=geo_risk[geo_col],
                    y=geo_risk['mean'],
                    marker_color='red',
                    opacity=0.7,
                    text=[f'{rate:.2%}' for rate in geo_risk['mean']],
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # 5. 風險趨勢預測（簡單移動平均）
        if len(df) > 100:
            window_size = min(50, len(df) // 10)
            risk_trend = pd.Series(risk_scores).rolling(window=window_size).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(risk_trend))),
                    y=risk_trend,
                    mode='lines',
                    name='風險趨勢',
                    line=dict(color='red', width=2)
                ),
                row=2, col=2
            )
        
        # 6. 風險控制建議
        recommendations = [
            ['極高風險交易', '立即阻斷並人工審核'],
            ['高風險交易', '延遲處理並加強驗證'],
            ['中風險交易', '增加額外驗證步驟'],
            ['低風險交易', '正常處理'],
            ['假正例優化', '調整模型閾值'],
            ['客戶體驗', '簡化低風險客戶流程']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['風險級別', '建議措施'], fill_color='paleturquoise', align='left'),
                cells=dict(
                    values=[[rec[0] for rec in recommendations], [rec[1] for rec in recommendations]],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=2, col=3
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title={
                'text': '風險評估與控制儀表板',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def generate_executive_summary(self, 
                                 financial_impact: FinancialImpact,
                                 business_metrics: BusinessMetrics,
                                 time_period: str = "本月") -> Dict[str, Any]:
        """生成執行摘要"""
        logger.info("生成執行摘要報告...")
        
        # 計算關鍵洞察
        key_insights = []
        
        if financial_impact.roi_percentage > 200:
            key_insights.append("💰 投資回報率超過200%，詐騙防護系統表現優異")
        elif financial_impact.roi_percentage > 100:
            key_insights.append("📈 投資回報率超過100%，系統投資價值顯著")
        else:
            key_insights.append("⚠️ 投資回報率偏低，建議優化系統參數")
        
        if business_metrics.detection_rate > 0.8:
            key_insights.append("🎯 詐騙檢測率超過80%，防護能力強")
        else:
            key_insights.append("📊 詐騙檢測率需要提升，建議調整模型")
        
        if business_metrics.false_positive_rate < 0.02:
            key_insights.append("✅ 假正例率控制良好，客戶體驗佳")
        else:
            key_insights.append("🔍 假正例率偏高，影響客戶體驗")
        
        # 計算總體評級
        overall_score = (
            min(100, financial_impact.roi_percentage / 2) * 0.3 +
            business_metrics.detection_rate * 100 * 0.3 +
            (1 - business_metrics.false_positive_rate) * 100 * 0.2 +
            business_metrics.customer_satisfaction_impact * 0.2
        )
        
        if overall_score >= 80:
            overall_rating = "優秀"
        elif overall_score >= 60:
            overall_rating = "良好"
        elif overall_score >= 40:
            overall_rating = "一般"
        else:
            overall_rating = "需改進"
        
        # 生成建議
        recommendations = []
        
        if financial_impact.roi_percentage < 150:
            recommendations.append("考慮調整模型閾值以提高ROI")
        
        if business_metrics.false_positive_rate > 0.03:
            recommendations.append("重點優化假正例率，提升客戶體驗")
        
        if business_metrics.detection_rate < 0.75:
            recommendations.append("增強模型檢測能力，提高詐騙識別率")
        
        recommendations.append("定期更新模型以應對新型詐騙手法")
        recommendations.append("建立持續監控機制確保系統穩定性")
        
        return {
            'generation_time': datetime.now().isoformat(),
            'time_period': time_period,
            'overall_rating': overall_rating,
            'overall_score': round(overall_score, 1),
            'key_metrics': {
                'roi_percentage': round(financial_impact.roi_percentage, 1),
                'total_savings': round(financial_impact.total_savings, 0),
                'detection_rate': round(business_metrics.detection_rate * 100, 1),
                'false_positive_rate': round(business_metrics.false_positive_rate * 100, 2),
                'transaction_volume': business_metrics.transaction_volume
            },
            'key_insights': key_insights,
            'recommendations': recommendations,
            'financial_summary': {
                'prevented_fraud': round(financial_impact.prevented_fraud_amount, 0),
                'actual_loss': round(financial_impact.actual_fraud_loss, 0),
                'total_cost': round(
                    financial_impact.investigation_cost + 
                    financial_impact.operational_cost + 
                    financial_impact.false_positive_cost, 0
                ),
                'net_benefit': round(financial_impact.total_savings, 0)
            }
        }
    
    def create_comprehensive_business_dashboard(self, 
                                              df: pd.DataFrame,
                                              predictions: np.ndarray,
                                              prediction_probabilities: np.ndarray,
                                              output_dir: str = 'business_analytics') -> Dict[str, str]:
        """創建綜合商業分析儀表板"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("創建綜合商業分析儀表板...")
        
        # 計算財務影響和業務指標
        financial_impact = self.calculate_financial_impact(df, predictions, prediction_probabilities)
        business_metrics = self.calculate_business_metrics(df, predictions)
        
        # 創建各種儀表板
        dashboards = {}
        
        # 1. 財務影響儀表板
        financial_fig = self.create_financial_impact_dashboard(financial_impact)
        financial_path = os.path.join(output_dir, 'financial_impact_dashboard.html')
        financial_fig.write_html(financial_path)
        dashboards['financial_impact'] = financial_path
        
        # 2. 業務績效儀表板
        performance_fig = self.create_business_performance_dashboard(business_metrics)
        performance_path = os.path.join(output_dir, 'business_performance_dashboard.html')
        performance_fig.write_html(performance_path)
        dashboards['business_performance'] = performance_path
        
        # 3. 風險評估儀表板
        risk_fig = self.create_risk_assessment_dashboard(df, predictions, prediction_probabilities)
        risk_path = os.path.join(output_dir, 'risk_assessment_dashboard.html')
        risk_fig.write_html(risk_path)
        dashboards['risk_assessment'] = risk_path
        
        # 4. 執行摘要
        executive_summary = self.generate_executive_summary(financial_impact, business_metrics)
        summary_path = os.path.join(output_dir, 'executive_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(executive_summary, f, ensure_ascii=False, indent=2)
        dashboards['executive_summary'] = summary_path
        
        # 5. 創建主索引頁面
        index_path = self._create_business_index(output_dir, dashboards, executive_summary)
        dashboards['index'] = index_path
        
        logger.info(f"商業分析儀表板已生成，保存在: {output_dir}")
        return dashboards
    
    def _create_business_index(self, output_dir: str, dashboards: Dict[str, str], 
                             executive_summary: Dict[str, Any]) -> str:
        """創建商業分析索引頁面"""
        index_path = os.path.join(output_dir, 'business_index.html')
        
        # 提取關鍵指標
        metrics = executive_summary['key_metrics']
        insights = executive_summary['key_insights']
        recommendations = executive_summary['recommendations']
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>詐騙檢測系統 - 商業分析中心</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Arial', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); text-align: center; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 30px 0; }}
                .dashboard-card {{ background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
                .dashboard-card h3 {{ color: #333; margin-top: 0; }}
                .dashboard-link {{ display: inline-block; padding: 12px 24px; background: linear-gradient(45deg, #667eea, #764ba2); color: white; text-decoration: none; border-radius: 25px; margin-top: 15px; transition: transform 0.3s; }}
                .dashboard-link:hover {{ transform: translateY(-2px); text-decoration: none; color: white; }}
                .insights {{ background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); margin: 20px 0; }}
                .insight-item {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #667eea; }}
                .recommendations {{ background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
                .recommendation-item {{ margin: 10px 0; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; }}
                .rating {{ display: inline-block; padding: 10px 20px; background: #28a745; color: white; border-radius: 20px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏢 詐騙檢測系統 - 商業分析中心</h1>
                    <p>總體評級: <span class="rating">{executive_summary['overall_rating']}</span></p>
                    <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{metrics['roi_percentage']}%</div>
                        <div class="metric-label">投資回報率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${metrics['total_savings']:,.0f}</div>
                        <div class="metric-label">總節省金額</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['detection_rate']}%</div>
                        <div class="metric-label">詐騙檢測率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{metrics['false_positive_rate']}%</div>
                        <div class="metric-label">假正例率</div>
                    </div>
                </div>
                
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <h3>💰 財務影響分析</h3>
                        <p>詳細分析系統的財務收益、成本結構和投資回報</p>
                        <a href="financial_impact_dashboard.html" class="dashboard-link">查看財務分析</a>
                    </div>
                    
                    <div class="dashboard-card">
                        <h3>📊 業務績效監控</h3>
                        <p>監控關鍵業務指標、客戶滿意度和系統效率</p>
                        <a href="business_performance_dashboard.html" class="dashboard-link">查看績效分析</a>
                    </div>
                    
                    <div class="dashboard-card">
                        <h3>🎯 風險評估控制</h3>
                        <p>風險分級管理、趨勢預測和控制策略建議</p>
                        <a href="risk_assessment_dashboard.html" class="dashboard-link">查看風險分析</a>
                    </div>
                </div>
                
                <div class="insights">
                    <h3>🔍 關鍵洞察</h3>
                    {''.join([f'<div class="insight-item">{insight}</div>' for insight in insights])}
                </div>
                
                <div class="recommendations">
                    <h3>💡 策略建議</h3>
                    {''.join([f'<div class="recommendation-item">• {rec}</div>' for rec in recommendations])}
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return index_path

# 便捷函數
def create_business_analytics_suite(df: pd.DataFrame, 
                                   predictions: np.ndarray,
                                   prediction_probabilities: np.ndarray,
                                   output_dir: str = 'business_analytics') -> Dict[str, str]:
    """創建完整的商業分析套件"""
    analyzer = BusinessAnalyzer()
    return analyzer.create_comprehensive_business_dashboard(
        df, predictions, prediction_probabilities, output_dir
    )

if __name__ == "__main__":
    print("商業分析模組已載入完成！")