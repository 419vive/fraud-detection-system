"""
Benchmark Visualization Module - IEEE-CIS Fraud Detection Project
Advanced visualization tools for performance benchmark analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BenchmarkVisualizer:
    """Advanced visualization for benchmark results"""
    
    def __init__(self, results: Dict[str, Any], output_dir: str = "benchmark_visualizations"):
        self.results = results
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting styles
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def create_training_performance_dashboard(self) -> str:
        """Create training performance dashboard"""
        logger.info("Creating training performance dashboard...")
        
        if not self.results.get('training_benchmarks'):
            logger.warning("No training benchmark data available")
            return ""
        
        training_data = pd.DataFrame(self.results['training_benchmarks'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Training Time Comparison', 'Memory Usage Peak',
                'CPU Usage During Training', 'Model Convergence',
                'Training Efficiency Score', 'Resource Utilization'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "radar"}, {"type": "heatmap"}]]
        )
        
        # 1. Training Time Comparison
        fig.add_trace(
            go.Bar(
                x=training_data['model_name'],
                y=training_data['training_time'],
                name="Training Time",
                marker_color='lightblue',
                text=[f"{t:.2f}s" for t in training_data['training_time']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Memory Usage Peak
        fig.add_trace(
            go.Bar(
                x=training_data['model_name'],
                y=training_data['memory_usage_peak'],
                name="Peak Memory",
                marker_color='lightcoral',
                text=[f"{m:.2f}GB" for m in training_data['memory_usage_peak']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. CPU Usage
        fig.add_trace(
            go.Bar(
                x=training_data['model_name'],
                y=training_data['cpu_usage_peak'],
                name="Peak CPU",
                marker_color='lightgreen',
                text=[f"{c:.1f}%" for c in training_data['cpu_usage_peak']],
                textposition='auto'
            ),
            row=1, col=3
        )
        
        # 4. Model Convergence (Training Time vs Final Score)
        fig.add_trace(
            go.Scatter(
                x=training_data['training_time'],
                y=training_data['final_score'],
                mode='markers+text',
                text=training_data['model_name'],
                textposition='top center',
                marker=dict(size=12, color='purple', opacity=0.7),
                name="Training Efficiency"
            ),
            row=2, col=1
        )
        
        # 5. Training Efficiency Radar Chart
        # Calculate efficiency scores
        training_data['time_score'] = 1 - (training_data['training_time'] / training_data['training_time'].max())
        training_data['memory_score'] = 1 - (training_data['memory_usage_peak'] / training_data['memory_usage_peak'].max())
        training_data['accuracy_score'] = training_data['final_score']
        
        for i, model in enumerate(training_data['model_name']):
            model_data = training_data[training_data['model_name'] == model].iloc[0]
            fig.add_trace(
                go.Scatterpolar(
                    r=[model_data['time_score'], model_data['memory_score'], 
                       model_data['accuracy_score']],
                    theta=['Speed', 'Memory Efficiency', 'Accuracy'],
                    fill='toself',
                    name=model,
                    opacity=0.6
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Training Performance Dashboard",
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="Memory (GB)", row=1, col=2)
        fig.update_xaxes(title_text="Models", row=1, col=3)
        fig.update_yaxes(title_text="CPU Usage (%)", row=1, col=3)
        fig.update_xaxes(title_text="Training Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Final Score", row=2, col=1)
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, "training_performance_dashboard.html")
        fig.write_html(output_path)
        logger.info(f"Training performance dashboard saved to {output_path}")
        
        return output_path
    
    def create_inference_performance_dashboard(self) -> str:
        """Create inference performance dashboard"""
        logger.info("Creating inference performance dashboard...")
        
        if not self.results.get('inference_benchmarks'):
            logger.warning("No inference benchmark data available")
            return ""
        
        inference_data = pd.DataFrame(self.results['inference_benchmarks'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Latency by Batch Size', 'Throughput Comparison',
                'Latency Distribution', 'Memory vs Performance',
                'Batch Size Optimization', 'Real-time Performance'
            ]
        )
        
        # 1. Latency by Batch Size
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['batch_size'],
                    y=model_data['latency_p95'],
                    mode='lines+markers',
                    name=f"{model} P95",
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Throughput Comparison
        throughput_by_model = inference_data.groupby('model_name')['throughput'].max().reset_index()
        fig.add_trace(
            go.Bar(
                x=throughput_by_model['model_name'],
                y=throughput_by_model['throughput'],
                name="Max Throughput",
                marker_color='lightgreen',
                text=[f"{t:.0f}" for t in throughput_by_model['throughput']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Latency Distribution (Box Plot)
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model]
            fig.add_trace(
                go.Box(
                    y=[model_data['latency_mean'].mean(), model_data['latency_p95'].mean()],
                    name=model,
                    boxpoints='all'
                ),
                row=1, col=3
            )
        
        # 4. Memory vs Performance Scatter
        fig.add_trace(
            go.Scatter(
                x=inference_data['memory_usage'],
                y=inference_data['throughput'],
                mode='markers+text',
                text=inference_data['model_name'],
                textposition='top center',
                marker=dict(
                    size=inference_data['batch_size']/50,
                    color=inference_data['latency_p95'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="P95 Latency (ms)")
                ),
                name="Memory vs Throughput"
            ),
            row=2, col=1
        )
        
        # 5. Batch Size Optimization
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['batch_size'],
                    y=model_data['predictions_per_second'],
                    mode='lines+markers',
                    name=f"{model} Pred/s",
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        # 6. Real-time Performance (if available)
        if self.results.get('realtime_benchmarks'):
            rt_data = self.results['realtime_benchmarks']
            models = list(rt_data.keys())
            latencies = [rt_data[model]['avg_latency_ms'] for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=latencies,
                    name="Avg Latency",
                    marker_color='orange',
                    text=[f"{l:.2f}ms" for l in latencies],
                    textposition='auto'
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            title_text="Inference Performance Dashboard",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Batch Size", row=1, col=1)
        fig.update_yaxes(title_text="P95 Latency (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="Throughput (pred/s)", row=1, col=2)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=3)
        fig.update_xaxes(title_text="Memory Usage (GB)", row=2, col=1)
        fig.update_yaxes(title_text="Throughput (pred/s)", row=2, col=1)
        fig.update_xaxes(title_text="Batch Size", row=2, col=2)
        fig.update_yaxes(title_text="Predictions/sec", row=2, col=2)
        fig.update_xaxes(title_text="Models", row=2, col=3)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=3)
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, "inference_performance_dashboard.html")
        fig.write_html(output_path)
        logger.info(f"Inference performance dashboard saved to {output_path}")
        
        return output_path
    
    def create_accuracy_comparison_dashboard(self) -> str:
        """Create accuracy comparison dashboard"""
        logger.info("Creating accuracy comparison dashboard...")
        
        if not self.results.get('accuracy_benchmarks'):
            logger.warning("No accuracy benchmark data available")
            return ""
        
        accuracy_data = pd.DataFrame(self.results['accuracy_benchmarks'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'ROC-AUC Comparison', 'Precision vs Recall',
                'Cross-Validation Scores', 'Training vs Validation'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # 1. ROC-AUC Comparison
        fig.add_trace(
            go.Bar(
                x=accuracy_data['model_name'],
                y=accuracy_data['roc_auc'],
                name="ROC-AUC",
                marker_color='steelblue',
                text=[f"{auc:.4f}" for auc in accuracy_data['roc_auc']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Precision vs Recall
        fig.add_trace(
            go.Scatter(
                x=accuracy_data['precision'],
                y=accuracy_data['recall'],
                mode='markers+text',
                text=accuracy_data['model_name'],
                textposition='top center',
                marker=dict(
                    size=accuracy_data['f1_score'] * 20,
                    color=accuracy_data['roc_auc'],
                    colorscale='viridis',
                    showscale=True,
                    colorbar=dict(title="ROC-AUC")
                ),
                name="Precision vs Recall"
            ),
            row=1, col=2
        )
        
        # 3. Cross-Validation Scores (with error bars)
        fig.add_trace(
            go.Bar(
                x=accuracy_data['model_name'],
                y=accuracy_data['cross_val_score_mean'],
                error_y=dict(
                    type='data',
                    array=accuracy_data['cross_val_score_std'],
                    visible=True
                ),
                name="CV Score",
                marker_color='lightcoral',
                text=[f"{cv:.4f}±{std:.4f}" for cv, std in 
                     zip(accuracy_data['cross_val_score_mean'], accuracy_data['cross_val_score_std'])],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Training vs Validation Scores
        fig.add_trace(
            go.Scatter(
                x=accuracy_data['training_score'],
                y=accuracy_data['validation_score'],
                mode='markers+text',
                text=accuracy_data['model_name'],
                textposition='top center',
                marker=dict(size=12, color='green', opacity=0.7),
                name="Train vs Val"
            ),
            row=2, col=2
        )
        
        # Add diagonal line for perfect alignment
        max_score = max(accuracy_data['training_score'].max(), accuracy_data['validation_score'].max())
        min_score = min(accuracy_data['training_score'].min(), accuracy_data['validation_score'].min())
        fig.add_trace(
            go.Scatter(
                x=[min_score, max_score],
                y=[min_score, max_score],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name="Perfect Alignment",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Model Accuracy Comparison Dashboard",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="ROC-AUC", row=1, col=1)
        fig.update_xaxes(title_text="Precision", row=1, col=2)
        fig.update_yaxes(title_text="Recall", row=1, col=2)
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_yaxes(title_text="CV Score", row=2, col=1)
        fig.update_xaxes(title_text="Training Score", row=2, col=2)
        fig.update_yaxes(title_text="Validation Score", row=2, col=2)
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, "accuracy_comparison_dashboard.html")
        fig.write_html(output_path)
        logger.info(f"Accuracy comparison dashboard saved to {output_path}")
        
        return output_path
    
    def create_comprehensive_comparison_matrix(self) -> str:
        """Create comprehensive model comparison matrix"""
        logger.info("Creating comprehensive comparison matrix...")
        
        # Prepare data from all benchmark types
        comparison_data = []
        
        # Get model names from any available benchmark
        model_names = set()
        if self.results.get('training_benchmarks'):
            model_names.update([r['model_name'] for r in self.results['training_benchmarks']])
        if self.results.get('accuracy_benchmarks'):
            model_names.update([r['model_name'] for r in self.results['accuracy_benchmarks']])
        if self.results.get('inference_benchmarks'):
            model_names.update([r['model_name'] for r in self.results['inference_benchmarks']])
        
        for model_name in model_names:
            model_data = {'model_name': model_name}
            
            # Training metrics
            if self.results.get('training_benchmarks'):
                training_result = next((r for r in self.results['training_benchmarks'] 
                                      if r['model_name'] == model_name), None)
                if training_result:
                    model_data.update({
                        'training_time': training_result['training_time'],
                        'memory_peak': training_result['memory_usage_peak'],
                        'cpu_usage': training_result['cpu_usage_peak']
                    })
            
            # Accuracy metrics
            if self.results.get('accuracy_benchmarks'):
                accuracy_result = next((r for r in self.results['accuracy_benchmarks'] 
                                      if r['model_name'] == model_name), None)
                if accuracy_result:
                    model_data.update({
                        'roc_auc': accuracy_result['roc_auc'],
                        'f1_score': accuracy_result['f1_score'],
                        'precision': accuracy_result['precision'],
                        'recall': accuracy_result['recall'],
                        'cv_score': accuracy_result['cross_val_score_mean']
                    })
            
            # Inference metrics (get best performance)
            if self.results.get('inference_benchmarks'):
                inference_results = [r for r in self.results['inference_benchmarks'] 
                                   if r['model_name'] == model_name]
                if inference_results:
                    best_inference = max(inference_results, key=lambda x: x['throughput'])
                    model_data.update({
                        'best_throughput': best_inference['throughput'],
                        'best_latency_p95': best_inference['latency_p95'],
                        'optimal_batch_size': best_inference['batch_size']
                    })
            
            comparison_data.append(model_data)
        
        if not comparison_data:
            logger.warning("No comparison data available")
            return ""
        
        # Create comparison matrix visualization
        df = pd.DataFrame(comparison_data)
        
        # Normalize metrics for heatmap (0-1 scale)
        normalized_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['training_time', 'memory_peak', 'best_latency_p95']:
                # Lower is better - inverse normalization
                normalized_df[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                # Higher is better - normal normalization
                normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=normalized_df[numeric_cols].values,
            x=numeric_cols,
            y=normalized_df['model_name'],
            colorscale='RdYlGn',
            text=df[numeric_cols].round(4).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Normalized Score (0-1)")
        ))
        
        fig.update_layout(
            title="Model Comparison Matrix (Normalized Scores)",
            xaxis_title="Performance Metrics",
            yaxis_title="Models",
            height=400 + len(model_names) * 50,
            width=1200
        )
        
        # Save comparison matrix
        output_path = os.path.join(self.output_dir, "comprehensive_comparison_matrix.html")
        fig.write_html(output_path)
        logger.info(f"Comprehensive comparison matrix saved to {output_path}")
        
        return output_path
    
    def create_performance_trend_analysis(self) -> str:
        """Create performance trend analysis visualization"""
        logger.info("Creating performance trend analysis...")
        
        if not self.results.get('inference_benchmarks'):
            logger.warning("No inference data for trend analysis")
            return ""
        
        inference_data = pd.DataFrame(self.results['inference_benchmarks'])
        
        # Create multi-metric trend analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Throughput vs Batch Size Trends',
                'Latency Scaling Analysis',
                'Memory Efficiency Trends',
                'Performance Score vs Resource Usage'
            ]
        )
        
        # 1. Throughput trends
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model].sort_values('batch_size')
            fig.add_trace(
                go.Scatter(
                    x=model_data['batch_size'],
                    y=model_data['throughput'],
                    mode='lines+markers',
                    name=f"{model} Throughput",
                    line=dict(width=3)
                ),
                row=1, col=1
            )
        
        # 2. Latency scaling
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model].sort_values('batch_size')
            fig.add_trace(
                go.Scatter(
                    x=model_data['batch_size'],
                    y=model_data['latency_p95'],
                    mode='lines+markers',
                    name=f"{model} P95 Latency",
                    line=dict(width=3)
                ),
                row=1, col=2
            )
        
        # 3. Memory efficiency (throughput per GB)
        inference_data['memory_efficiency'] = inference_data['throughput'] / (inference_data['memory_usage'] + 0.1)
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model].sort_values('batch_size')
            fig.add_trace(
                go.Scatter(
                    x=model_data['batch_size'],
                    y=model_data['memory_efficiency'],
                    mode='lines+markers',
                    name=f"{model} Memory Eff",
                    line=dict(width=3)
                ),
                row=2, col=1
            )
        
        # 4. Performance vs Resource scatter
        inference_data['performance_score'] = (
            inference_data['throughput'] / inference_data['throughput'].max() * 0.5 +
            (1 - inference_data['latency_p95'] / inference_data['latency_p95'].max()) * 0.5
        )
        inference_data['resource_usage'] = (
            inference_data['memory_usage'] / inference_data['memory_usage'].max() * 0.5 +
            inference_data['cpu_usage'] / 100 * 0.5
        )
        
        for model in inference_data['model_name'].unique():
            model_data = inference_data[inference_data['model_name'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['resource_usage'],
                    y=model_data['performance_score'],
                    mode='markers',
                    name=f"{model}",
                    marker=dict(size=10, opacity=0.7),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Performance Trend Analysis",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Batch Size", row=1, col=1)
        fig.update_yaxes(title_text="Throughput (pred/s)", row=1, col=1)
        fig.update_xaxes(title_text="Batch Size", row=1, col=2)
        fig.update_yaxes(title_text="P95 Latency (ms)", row=1, col=2)
        fig.update_xaxes(title_text="Batch Size", row=2, col=1)
        fig.update_yaxes(title_text="Memory Efficiency", row=2, col=1)
        fig.update_xaxes(title_text="Resource Usage Score", row=2, col=2)
        fig.update_yaxes(title_text="Performance Score", row=2, col=2)
        
        # Save trend analysis
        output_path = os.path.join(self.output_dir, "performance_trend_analysis.html")
        fig.write_html(output_path)
        logger.info(f"Performance trend analysis saved to {output_path}")
        
        return output_path
    
    def create_optimization_impact_analysis(self) -> str:
        """Create optimization impact analysis visualization"""
        logger.info("Creating optimization impact analysis...")
        
        if not self.results.get('optimization_benchmarks'):
            logger.warning("No optimization benchmark data available")
            return ""
        
        opt_data = self.results['optimization_benchmarks']
        
        # Prepare data for visualization
        models = list(opt_data.keys())
        optimization_levels = ['baseline', 'light', 'medium', 'aggressive']
        
        # Create speedup matrix
        speedup_matrix = []
        for model in models:
            model_speedups = []
            for opt_level in optimization_levels:
                if opt_level in opt_data[model] and 'speedup' in opt_data[model][opt_level]:
                    speedup = opt_data[model][opt_level]['speedup']
                elif opt_level == 'baseline':
                    speedup = 1.0
                else:
                    speedup = np.nan
                model_speedups.append(speedup)
            speedup_matrix.append(model_speedups)
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Optimization Speedup Matrix', 'Speedup Comparison'],
            specs=[[{"type": "heatmap"}, {"type": "bar"}]]
        )
        
        # 1. Speedup Heatmap
        fig.add_trace(
            go.Heatmap(
                z=speedup_matrix,
                x=optimization_levels,
                y=models,
                text=[[f"{val:.2f}x" if not np.isnan(val) else "N/A" for val in row] for row in speedup_matrix],
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale='RdYlGn',
                colorbar=dict(title="Speedup Factor")
            ),
            row=1, col=1
        )
        
        # 2. Best speedup comparison
        best_speedups = []
        best_opt_levels = []
        for model in models:
            max_speedup = 1.0
            best_level = 'baseline'
            for opt_level, result in opt_data[model].items():
                if opt_level != 'baseline' and 'speedup' in result:
                    if result['speedup'] > max_speedup:
                        max_speedup = result['speedup']
                        best_level = opt_level
            best_speedups.append(max_speedup)
            best_opt_levels.append(best_level)
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=best_speedups,
                text=[f"{s:.2f}x ({level})" for s, level in zip(best_speedups, best_opt_levels)],
                textposition='auto',
                name="Best Speedup",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Optimization Impact Analysis",
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Optimization Level", row=1, col=1)
        fig.update_yaxes(title_text="Models", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="Speedup Factor", row=1, col=2)
        
        # Save optimization analysis
        output_path = os.path.join(self.output_dir, "optimization_impact_analysis.html")
        fig.write_html(output_path)
        logger.info(f"Optimization impact analysis saved to {output_path}")
        
        return output_path
    
    def create_executive_summary_dashboard(self) -> str:
        """Create executive summary dashboard"""
        logger.info("Creating executive summary dashboard...")
        
        # Create comprehensive executive dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Overall Model Ranking', 'Training Time vs Accuracy',
                'Inference Performance Summary', 'Resource Utilization',
                'Cost-Performance Analysis', 'Scalability Assessment',
                'Recommendation Score', 'Performance Trends', 'Key Metrics'
            ],
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "radar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Overall ranking (if available)
        if self.results.get('comparative_analysis', {}).get('overall_ranking'):
            ranking_data = self.results['comparative_analysis']['overall_ranking'][:5]
            fig.add_trace(
                go.Bar(
                    x=[r['model_name'] for r in ranking_data],
                    y=[r['overall_score'] for r in ranking_data],
                    name="Overall Score",
                    marker_color='steelblue',
                    text=[f"{r['overall_score']:.3f}" for r in ranking_data],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # 2. Training Time vs Accuracy
        if self.results.get('training_benchmarks') and self.results.get('accuracy_benchmarks'):
            training_df = pd.DataFrame(self.results['training_benchmarks'])
            accuracy_df = pd.DataFrame(self.results['accuracy_benchmarks'])
            
            # Merge data
            merged_df = training_df.merge(accuracy_df, on='model_name', how='inner')
            
            fig.add_trace(
                go.Scatter(
                    x=merged_df['training_time'],
                    y=merged_df['roc_auc'],
                    mode='markers+text',
                    text=merged_df['model_name'],
                    textposition='top center',
                    marker=dict(
                        size=15,
                        color=merged_df['f1_score'],
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="F1 Score")
                    ),
                    name="Time vs Accuracy"
                ),
                row=1, col=2
            )
        
        # 3. Inference Performance Summary
        if self.results.get('inference_benchmarks'):
            inference_df = pd.DataFrame(self.results['inference_benchmarks'])
            best_throughput = inference_df.groupby('model_name')['throughput'].max().reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=best_throughput['model_name'],
                    y=best_throughput['throughput'],
                    name="Best Throughput",
                    marker_color='lightgreen',
                    text=[f"{t:.0f}" for t in best_throughput['throughput']],
                    textposition='auto'
                ),
                row=1, col=3
            )
        
        # Add more visualizations...
        # (Additional plots would be added here based on available data)
        
        fig.update_layout(
            height=1200,
            title_text="Executive Performance Summary Dashboard",
            showlegend=True
        )
        
        # Save executive dashboard
        output_path = os.path.join(self.output_dir, "executive_summary_dashboard.html")
        fig.write_html(output_path)
        logger.info(f"Executive summary dashboard saved to {output_path}")
        
        return output_path
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generate all visualization dashboards"""
        logger.info("Generating all benchmark visualizations...")
        
        output_paths = {}
        
        try:
            output_paths['training_dashboard'] = self.create_training_performance_dashboard()
        except Exception as e:
            logger.error(f"Failed to create training dashboard: {e}")
        
        try:
            output_paths['inference_dashboard'] = self.create_inference_performance_dashboard()
        except Exception as e:
            logger.error(f"Failed to create inference dashboard: {e}")
        
        try:
            output_paths['accuracy_dashboard'] = self.create_accuracy_comparison_dashboard()
        except Exception as e:
            logger.error(f"Failed to create accuracy dashboard: {e}")
        
        try:
            output_paths['comparison_matrix'] = self.create_comprehensive_comparison_matrix()
        except Exception as e:
            logger.error(f"Failed to create comparison matrix: {e}")
        
        try:
            output_paths['trend_analysis'] = self.create_performance_trend_analysis()
        except Exception as e:
            logger.error(f"Failed to create trend analysis: {e}")
        
        try:
            output_paths['optimization_analysis'] = self.create_optimization_impact_analysis()
        except Exception as e:
            logger.error(f"Failed to create optimization analysis: {e}")
        
        try:
            output_paths['executive_dashboard'] = self.create_executive_summary_dashboard()
        except Exception as e:
            logger.error(f"Failed to create executive dashboard: {e}")
        
        # Create index page
        self._create_visualization_index(output_paths)
        
        logger.info(f"All visualizations generated and saved to {self.output_dir}")
        return output_paths
    
    def _create_visualization_index(self, output_paths: Dict[str, str]):
        """Create an index page for all visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Benchmark Visualizations</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                .dashboard-link {{ 
                    display: block; 
                    margin: 10px 0; 
                    padding: 10px; 
                    background-color: #f0f0f0; 
                    text-decoration: none; 
                    color: #333; 
                    border-radius: 5px; 
                }}
                .dashboard-link:hover {{ background-color: #e0e0e0; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>Performance Benchmark Visualizations</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Available Dashboards:</h2>
        """
        
        dashboard_names = {
            'training_dashboard': 'Training Performance Dashboard',
            'inference_dashboard': 'Inference Performance Dashboard',
            'accuracy_dashboard': 'Accuracy Comparison Dashboard',
            'comparison_matrix': 'Comprehensive Comparison Matrix',
            'trend_analysis': 'Performance Trend Analysis',
            'optimization_analysis': 'Optimization Impact Analysis',
            'executive_dashboard': 'Executive Summary Dashboard'
        }
        
        for key, path in output_paths.items():
            if path and os.path.exists(path):
                name = dashboard_names.get(key, key.replace('_', ' ').title())
                relative_path = os.path.basename(path)
                html_content += f'<a href="{relative_path}" class="dashboard-link">{name}</a>\n'
        
        html_content += """
        </body>
        </html>
        """
        
        index_path = os.path.join(self.output_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Visualization index created at {index_path}")

def create_benchmark_visualizations(results_file: str, output_dir: str = "benchmark_visualizations") -> Dict[str, str]:
    """
    Create visualizations from benchmark results file
    
    Args:
        results_file: Path to benchmark results JSON file
        output_dir: Output directory for visualizations
    
    Returns:
        Dictionary of generated visualization file paths
    """
    logger.info(f"Creating visualizations from {results_file}")
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create visualizer and generate all visualizations
    visualizer = BenchmarkVisualizer(results, output_dir)
    output_paths = visualizer.generate_all_visualizations()
    
    return output_paths

if __name__ == "__main__":
    print("Benchmark Visualization Module loaded successfully!")
    logger.info("Benchmark Visualization Module ready for use")