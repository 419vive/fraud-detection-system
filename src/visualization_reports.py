"""
可視化報告模組 - IEEE-CIS 詐騙檢測項目
包含EDA圖表、模型監控視覺化、架構圖表等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體和樣式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class FraudDetectionVisualizer:
    """詐騙檢測可視化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def create_architecture_diagram(self, save_path: str = None):
        """創建系統架構圖"""
        fig = go.Figure()
        
        # 定義架構層次
        layers = [
            {"name": "呈現層 (Presentation)", "y": 3, "components": [
                "Jupyter Notebooks (EDA/Analysis)", 
                "Web Dashboard (Model Monitoring)", 
                "API Endpoints (Prediction Service)"
            ]},
            {"name": "業務層 (Business Logic)", "y": 2, "components": [
                "Model Training Pipeline", 
                "Feature Engineering Engine", 
                "Model Evaluation & Validation", 
                "Prediction Service"
            ]},
            {"name": "資料層 (Data Layer)", "y": 1, "components": [
                "Raw Data Storage", 
                "Processed Data Cache", 
                "Model Artifacts Store", 
                "Experiment Tracking"
            ]}
        ]
        
        # 繪製層次框
        colors = ['lightblue', 'lightgreen', 'lightyellow']
        for i, layer in enumerate(layers):
            fig.add_shape(
                type="rect",
                x0=-0.5, y0=layer["y"]-0.4,
                x1=3.5, y1=layer["y"]+0.4,
                line=dict(color="black", width=2),
                fillcolor=colors[i],
                opacity=0.3
            )
            
            # 添加層次標題
            fig.add_annotation(
                x=-0.3, y=layer["y"],
                text=f"<b>{layer['name']}</b>",
                showarrow=False,
                font=dict(size=14, color="black"),
                align="left"
            )
            
            # 添加組件
            for j, component in enumerate(layer["components"]):
                fig.add_annotation(
                    x=j*0.8 + 0.2, y=layer["y"]-0.15,
                    text=component,
                    showarrow=False,
                    font=dict(size=10, color="darkblue"),
                    align="center"
                )
        
        # 添加數據流箭頭
        fig.add_annotation(
            x=3.7, y=2,
            ax=3.7, ay=1.5,
            arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor="red",
            text="Data Flow",
            font=dict(size=10, color="red")
        )
        
        fig.update_layout(
            title="<b>詐騙檢測系統架構圖</b>",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.6, 4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.5, 3.5]),
            plot_bgcolor='white',
            width=1000, height=600
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_architecture.html'))
            fig.write_image(save_path.replace('.png', '_architecture.png'))
        
        fig.show()
        return fig
    
    def create_data_flow_diagram(self, save_path: str = None):
        """創建數據流程圖"""
        fig = go.Figure()
        
        # ETL Pipeline 步驟
        steps = [
            "Raw Data", "Data Validation", "Preprocessing", 
            "Feature Engineering", "Model Training", "Evaluation", "Deployment"
        ]
        
        # 創建流程圖
        for i, step in enumerate(steps):
            # 添加步驟框
            fig.add_shape(
                type="rect",
                x0=i*1.5, y0=0.8,
                x1=i*1.5+1.2, y1=1.2,
                line=dict(color="blue", width=2),
                fillcolor="lightblue",
                opacity=0.7
            )
            
            # 添加步驟文字
            fig.add_annotation(
                x=i*1.5+0.6, y=1,
                text=f"<b>{step}</b>",
                showarrow=False,
                font=dict(size=10, color="black"),
                align="center"
            )
            
            # 添加箭頭（除了最後一個步驟）
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=i*1.5+1.3, y=1,
                    ax=i*1.5+1.4, ay=1,
                    arrowhead=2, arrowsize=1, arrowwidth=2,
                    arrowcolor="red"
                )
        
        fig.update_layout(
            title="<b>數據處理流程圖 (ETL Pipeline)</b>",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                      range=[-0.5, len(steps)*1.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, 
                      range=[0.5, 1.5]),
            plot_bgcolor='white',
            width=1200, height=300
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_dataflow.html'))
            fig.write_image(save_path.replace('.png', '_dataflow.png'))
        
        fig.show()
        return fig
    
    def create_model_performance_dashboard(self, evaluation_results: Dict[str, Any], 
                                         save_path: str = None):
        """創建模型性能監控儀表板"""
        
        if not evaluation_results:
            logger.warning("沒有評估結果可以可視化")
            return None
        
        # 創建子圖佈局
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                '模型ROC-AUC比較', '模型F1-Score比較', '模型Precision vs Recall',
                '混淆矩陣熱圖', '特徵重要性TOP10', '模型性能雷達圖'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatterpolar"}]]
        )
        
        models = list(evaluation_results.keys())
        colors = self.color_palette[:len(models)]
        
        # 1. ROC-AUC 比較
        auc_scores = [evaluation_results[model]['basic_metrics']['roc_auc'] 
                     for model in models]
        fig.add_trace(
            go.Bar(x=models, y=auc_scores, name='ROC-AUC', 
                  marker_color=colors[0], showlegend=False),
            row=1, col=1
        )
        
        # 2. F1-Score 比較
        f1_scores = [evaluation_results[model]['basic_metrics']['f1_score'] 
                    for model in models]
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name='F1-Score', 
                  marker_color=colors[1], showlegend=False),
            row=1, col=2
        )
        
        # 3. Precision vs Recall 散點圖
        precisions = [evaluation_results[model]['basic_metrics']['precision'] 
                     for model in models]
        recalls = [evaluation_results[model]['basic_metrics']['recall'] 
                  for model in models]
        
        fig.add_trace(
            go.Scatter(x=recalls, y=precisions, mode='markers+text',
                      text=models, textposition="top center",
                      marker=dict(size=12, color=colors), showlegend=False),
            row=1, col=3
        )
        
        # 4. 混淆矩陣熱圖（取第一個模型作為示例）
        if models:
            first_model = models[0]
            cm = evaluation_results[first_model]['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False,
                          text=[['TN', 'FP'], ['FN', 'TP']], texttemplate="%{text}",
                          textfont={"size": 12}),
                row=2, col=1
            )
        
        # 5. 特徵重要性（如果可用）
        # 這裡我們創建一個示例，實際使用時需要傳入特徵重要性數據
        feature_names = [f'Feature_{i}' for i in range(10)]
        importance_values = np.random.random(10)
        
        fig.add_trace(
            go.Bar(x=importance_values, y=feature_names, orientation='h',
                  marker_color=colors[2], showlegend=False),
            row=2, col=2
        )
        
        # 6. 模型性能雷達圖
        metrics = ['ROC-AUC', 'F1-Score', 'Precision', 'Recall', 'Accuracy']
        
        for i, model in enumerate(models):
            values = [
                evaluation_results[model]['basic_metrics']['roc_auc'],
                evaluation_results[model]['basic_metrics']['f1_score'],
                evaluation_results[model]['basic_metrics']['precision'],
                evaluation_results[model]['basic_metrics']['recall'],
                evaluation_results[model]['basic_metrics']['accuracy']
            ]
            
            fig.add_trace(
                go.Scatterpolar(r=values, theta=metrics, fill='toself',
                              name=model, line_color=colors[i % len(colors)]),
                row=2, col=3
            )
        
        # 更新佈局
        fig.update_layout(
            title_text="<b>詐騙檢測模型性能監控儀表板</b>",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # 更新各子圖的軸標題
        fig.update_xaxes(title_text="模型", row=1, col=1)
        fig.update_yaxes(title_text="ROC-AUC", row=1, col=1)
        
        fig.update_xaxes(title_text="模型", row=1, col=2)
        fig.update_yaxes(title_text="F1-Score", row=1, col=2)
        
        fig.update_xaxes(title_text="Recall", row=1, col=3)
        fig.update_yaxes(title_text="Precision", row=1, col=3)
        
        fig.update_xaxes(title_text="特徵重要性", row=2, col=2)
        fig.update_yaxes(title_text="特徵", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_dashboard.html'))
            fig.write_image(save_path.replace('.png', '_dashboard.png'))
        
        fig.show()
        return fig
    
    def create_eda_report(self, df: pd.DataFrame, target_col: str = 'isFraud', 
                         save_path: str = None):
        """創建探索性數據分析報告"""
        
        # 創建多個子圖
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '詐騙交易分佈', '交易金額分佈', 
                '時間模式分析', '缺失值熱圖',
                '特徵相關性矩陣', '異常值檢測'
            ],
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "box"}]]
        )
        
        # 1. 詐騙交易分佈餅圖
        if target_col in df.columns:
            fraud_counts = df[target_col].value_counts()
            labels = ['正常交易', '詐騙交易']
            values = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, hole=0.3,
                      marker_colors=['lightblue', 'red']),
                row=1, col=1
            )
        
        # 2. 交易金額分佈直方圖
        if 'TransactionAmt' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['TransactionAmt'], nbinsx=50, 
                           marker_color='lightgreen', opacity=0.7),
                row=1, col=2
            )
        
        # 3. 時間模式分析（如果有時間特徵）
        if 'TransactionDT' in df.columns:
            # 創建小時特徵進行分析
            df_temp = df.copy()
            df_temp['hour'] = (df_temp['TransactionDT'] / 3600) % 24
            hourly_counts = df_temp['hour'].value_counts().sort_index()
            
            fig.add_trace(
                go.Bar(x=hourly_counts.index, y=hourly_counts.values,
                      marker_color='orange'),
                row=2, col=1
            )
        
        # 4. 缺失值熱圖
        missing_data = df.isnull().sum().head(20)  # 前20個特徵
        if len(missing_data) > 0:
            missing_matrix = missing_data.values.reshape(-1, 1)
            fig.add_trace(
                go.Heatmap(z=missing_matrix, 
                          y=missing_data.index,
                          colorscale='Reds', showscale=False),
                row=2, col=2
            )
        
        # 5. 特徵相關性矩陣
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]  # 前10個數值特徵
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=3, col=1
            )
        
        # 6. 異常值檢測箱線圖
        if 'TransactionAmt' in df.columns:
            fig.add_trace(
                go.Box(y=df['TransactionAmt'], name='TransactionAmt',
                      marker_color='purple'),
                row=3, col=2
            )
        
        fig.update_layout(
            title_text="<b>詐騙檢測數據探索性分析報告</b>",
            title_x=0.5,
            height=1000,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_eda.html'))
            fig.write_image(save_path.replace('.png', '_eda.png'))
        
        fig.show()
        return fig
    
    def create_feature_engineering_report(self, original_df: pd.DataFrame, 
                                        engineered_df: pd.DataFrame, 
                                        save_path: str = None):
        """創建特徵工程報告"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '特徵數量變化', '數據維度變化',
                '新增特徵類型分佈', '特徵工程前後對比'
            ]
        )
        
        # 1. 特徵數量變化
        original_features = original_df.shape[1]
        engineered_features = engineered_df.shape[1]
        
        fig.add_trace(
            go.Bar(x=['原始特徵', '工程後特徵'], 
                  y=[original_features, engineered_features],
                  marker_color=['lightblue', 'lightgreen']),
            row=1, col=1
        )
        
        # 2. 數據維度變化
        original_rows = original_df.shape[0]
        engineered_rows = engineered_df.shape[0]
        
        categories = ['特徵數', '樣本數']
        original_values = [original_features, original_rows]
        engineered_values = [engineered_features, engineered_rows]
        
        fig.add_trace(
            go.Bar(x=categories, y=original_values, name='原始數據',
                  marker_color='lightblue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=categories, y=engineered_values, name='工程後數據',
                  marker_color='lightgreen'),
            row=1, col=2
        )
        
        # 3. 新增特徵類型分佈
        new_features = set(engineered_df.columns) - set(original_df.columns)
        feature_types = {
            '時間特徵': len([f for f in new_features if any(t in f.lower() 
                            for t in ['hour', 'day', 'weekend', 'time'])]),
            '聚合特徵': len([f for f in new_features if any(t in f.lower() 
                            for t in ['mean', 'std', 'count', 'sum'])]),
            '交互特徵': len([f for f in new_features if any(t in f.lower() 
                            for t in ['_x_', '_div_', '_minus_'])]),
            '其他特徵': len([f for f in new_features if not any(
                t in f.lower() for t in ['hour', 'day', 'weekend', 'time', 
                'mean', 'std', 'count', 'sum', '_x_', '_div_', '_minus_'])])
        }
        
        fig.add_trace(
            go.Pie(labels=list(feature_types.keys()), 
                  values=list(feature_types.values()),
                  hole=0.3),
            row=2, col=1
        )
        
        # 4. 特徵工程前後統計對比
        original_stats = original_df.describe().loc['mean'].head(5)
        engineered_stats = engineered_df[original_stats.index].describe().loc['mean']
        
        fig.add_trace(
            go.Scatter(x=original_stats.index, y=original_stats.values,
                      mode='lines+markers', name='原始均值',
                      line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=engineered_stats.index, y=engineered_stats.values,
                      mode='lines+markers', name='工程後均值',
                      line=dict(color='green')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="<b>特徵工程分析報告</b>",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path.replace('.png', '_feature_engineering.html'))
            fig.write_image(save_path.replace('.png', '_feature_engineering.png'))
        
        fig.show()
        return fig
    
    def generate_comprehensive_report(self, df: pd.DataFrame, 
                                    evaluation_results: Dict[str, Any] = None,
                                    output_dir: str = "reports",
                                    target_col: str = 'isFraud'):
        """生成綜合可視化報告"""
        
        # 創建報告目錄
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("開始生成綜合可視化報告...")
        
        # 1. 系統架構圖
        logger.info("生成系統架構圖...")
        arch_fig = self.create_architecture_diagram(
            save_path=os.path.join(output_dir, f"architecture_{timestamp}.png")
        )
        
        # 2. 數據流程圖
        logger.info("生成數據流程圖...")
        flow_fig = self.create_data_flow_diagram(
            save_path=os.path.join(output_dir, f"dataflow_{timestamp}.png")
        )
        
        # 3. EDA報告
        logger.info("生成EDA分析報告...")
        eda_fig = self.create_eda_report(
            df, target_col=target_col,
            save_path=os.path.join(output_dir, f"eda_{timestamp}.png")
        )
        
        # 4. 模型性能儀表板
        if evaluation_results:
            logger.info("生成模型性能監控儀表板...")
            dashboard_fig = self.create_model_performance_dashboard(
                evaluation_results,
                save_path=os.path.join(output_dir, f"dashboard_{timestamp}.png")
            )
        
        # 5. 生成HTML報告摘要
        html_report = self._generate_html_summary(df, evaluation_results, timestamp)
        html_path = os.path.join(output_dir, f"fraud_detection_report_{timestamp}.html")
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"綜合報告已生成至目錄: {output_dir}")
        logger.info(f"HTML報告: {html_path}")
        
        return {
            "architecture": arch_fig,
            "dataflow": flow_fig, 
            "eda": eda_fig,
            "dashboard": dashboard_fig if evaluation_results else None,
            "html_report": html_path
        }
    
    def _generate_html_summary(self, df: pd.DataFrame, 
                              evaluation_results: Dict[str, Any],
                              timestamp: str) -> str:
        """生成HTML報告摘要"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-TW">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>詐騙檢測系統分析報告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; 
                           background: #f8f9fa; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #667eea; color: white; }}
                .success {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; font-weight: bold; }}
                .danger {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🛡️ IEEE-CIS 詐騙檢測系統</h1>
                <h2>綜合分析報告</h2>
                <p>生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 數據概覽</h2>
                <div class="metric">
                    <div class="metric-value">{df.shape[0]:,}</div>
                    <div class="metric-label">總交易數</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df.shape[1]}</div>
                    <div class="metric-label">特徵數量</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df['isFraud'].sum() if 'isFraud' in df.columns else 'N/A'}</div>
                    <div class="metric-label">詐騙交易數</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{(df['isFraud'].mean()*100):.2f}% if 'isFraud' in df.columns else 'N/A'}</div>
                    <div class="metric-label">詐騙率</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🏗️ 系統架構</h2>
                <p>本系統採用三層架構設計：</p>
                <ul>
                    <li><strong>呈現層</strong>: Jupyter Notebooks、Web Dashboard、API服務</li>
                    <li><strong>業務層</strong>: 模型訓練、特徵工程、評估驗證、預測服務</li>
                    <li><strong>資料層</strong>: 原始數據存儲、處理緩存、模型倉庫、實驗追蹤</li>
                </ul>
            </div>
        """
        
        # 添加模型性能部分
        if evaluation_results:
            html_content += """
            <div class="section">
                <h2>🤖 模型性能</h2>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>ROC-AUC</th>
                        <th>F1-Score</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>狀態</th>
                    </tr>
            """
            
            for model_name, results in evaluation_results.items():
                auc = results['basic_metrics']['roc_auc']
                f1 = results['basic_metrics']['f1_score']
                precision = results['basic_metrics']['precision']
                recall = results['basic_metrics']['recall']
                
                status_class = "success" if auc > 0.9 else "warning" if auc > 0.8 else "danger"
                status_text = "優秀" if auc > 0.9 else "良好" if auc > 0.8 else "需改進"
                
                html_content += f"""
                    <tr>
                        <td><strong>{model_name}</strong></td>
                        <td>{auc:.4f}</td>
                        <td>{f1:.4f}</td>
                        <td>{precision:.4f}</td>
                        <td>{recall:.4f}</td>
                        <td><span class="{status_class}">{status_text}</span></td>
                    </tr>
                """
            
            html_content += "</table></div>"
        
        html_content += """
            <div class="section">
                <h2>📈 關鍵指標</h2>
                <ul>
                    <li><strong>目標AUC</strong>: > 0.9 (系統要求)</li>
                    <li><strong>推論延遲</strong>: < 100ms (系統要求)</li>
                    <li><strong>系統可用性</strong>: > 99.5% (系統要求)</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🔍 技術棧</h2>
                <ul>
                    <li><strong>機器學習</strong>: LightGBM, XGBoost, CatBoost</li>
                    <li><strong>特徵工程</strong>: Pandas, Scikit-learn, Imbalanced-learn</li>
                    <li><strong>API服務</strong>: FastAPI</li>
                    <li><strong>可視化</strong>: Plotly, Matplotlib, Seaborn</li>
                    <li><strong>部署</strong>: Docker, uvicorn</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📋 報告文件</h2>
                <ul>
                    <li>系統架構圖: architecture_{timestamp}.html</li>
                    <li>數據流程圖: dataflow_{timestamp}.html</li>
                    <li>EDA分析報告: eda_{timestamp}.html</li>
                    <li>模型監控儀表板: dashboard_{timestamp}.html</li>
                </ul>
            </div>
            
            <footer style="text-align: center; margin-top: 50px; color: #666;">
                <p>© 2024 IEEE-CIS 詐騙檢測系統 | 生成時間: {timestamp}</p>
            </footer>
        </body>
        </html>
        """
        
        return html_content

def generate_fraud_detection_report(df: pd.DataFrame, 
                                  evaluation_results: Dict[str, Any] = None,
                                  output_dir: str = "reports") -> Dict[str, Any]:
    """便捷函數：生成詐騙檢測可視化報告"""
    visualizer = FraudDetectionVisualizer()
    return visualizer.generate_comprehensive_report(
        df, evaluation_results, output_dir
    )

if __name__ == "__main__":
    print("詐騙檢測可視化報告模組已載入完成！")