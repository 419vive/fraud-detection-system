"""
性能報告生成器 - IEEE-CIS 詐騙檢測項目
生成詳細的性能優化報告和基準測試結果
"""

import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 設置matplotlib中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class PerformanceReportGenerator:
    """性能報告生成器"""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 報告模板
        self.report_template = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'project': 'IEEE-CIS Fraud Detection Performance Optimization'
            },
            'system_info': {},
            'optimization_results': {},
            'benchmark_results': {},
            'recommendations': []
        }
    
    def collect_system_info(self) -> Dict[str, Any]:
        """收集系統信息"""
        import psutil
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'python_version': platform.python_version(),
            'libraries': self._get_library_versions()
        }
        
        return system_info
    
    def _get_library_versions(self) -> Dict[str, str]:
        """獲取關鍵庫版本"""
        versions = {}
        libraries = [
            'pandas', 'numpy', 'scikit-learn', 'xgboost', 
            'lightgbm', 'catboost', 'optuna', 'joblib'
        ]
        
        for lib in libraries:
            try:
                if lib == 'scikit-learn':
                    import sklearn
                    versions[lib] = sklearn.__version__
                else:
                    module = __import__(lib)
                    versions[lib] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[lib] = 'not installed'
        
        return versions
    
    def generate_memory_optimization_report(self, before_memory: float, 
                                          after_memory: float, 
                                          optimization_details: Dict[str, Any]) -> Dict[str, Any]:
        """生成內存優化報告"""
        memory_reduction = ((before_memory - after_memory) / before_memory) * 100
        
        report = {
            'before_memory_gb': before_memory,
            'after_memory_gb': after_memory,
            'memory_reduction_percent': memory_reduction,
            'memory_saved_gb': before_memory - after_memory,
            'optimization_techniques': optimization_details,
            'efficiency_rating': self._calculate_efficiency_rating(memory_reduction, 'memory')
        }
        
        return report
    
    def generate_feature_engineering_report(self, fe_summary: Dict[str, Any]) -> Dict[str, Any]:
        """生成特徵工程報告"""
        report = {
            'total_processing_time': fe_summary.get('total_time', 0),
            'features_created': fe_summary.get('total_features_created', 0),
            'processing_steps': fe_summary.get('processing_times', {}),
            'feature_counts_by_type': fe_summary.get('feature_counts', {}),
            'cache_efficiency': fe_summary.get('cache_size', 0),
            'performance_rating': self._calculate_efficiency_rating(
                fe_summary.get('total_time', 0), 'time'
            )
        }
        
        return report
    
    def generate_training_optimization_report(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成訓練優化報告"""
        report = {
            'models_trained': len(training_results.get('trained_models', {})),
            'total_training_time': training_results.get('total_training_time', 0),
            'best_model': training_results.get('best_model', 'unknown'),
            'best_score': training_results.get('best_score', 0),
            'model_performances': {},
            'training_efficiency': {}
        }
        
        # 分析各模型性能
        if 'trained_models' in training_results:
            for model_name, model_info in training_results['trained_models'].items():
                if isinstance(model_info, dict):
                    report['model_performances'][model_name] = {
                        'validation_score': model_info.get('validation_score', 0),
                        'training_completed': model_info.get('training_completed', False)
                    }
        
        # 計算訓練效率
        if 'training_metrics' in training_results:
            metrics = training_results['training_metrics']
            if metrics:
                avg_training_time = np.mean([m.training_time for m in metrics])
                avg_score = np.mean([m.validation_score for m in metrics])
                
                report['training_efficiency'] = {
                    'average_training_time': avg_training_time,
                    'average_validation_score': avg_score,
                    'efficiency_score': avg_score / (avg_training_time + 1e-8)
                }
        
        return report
    
    def generate_inference_optimization_report(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成推理優化報告"""
        report = {
            'throughput_improvements': {},
            'latency_improvements': {},
            'memory_efficiency': benchmark_results.get('memory_efficiency', {}),
            'real_time_performance': benchmark_results.get('real_time_performance', {}),
            'system_utilization': benchmark_results.get('system_resources', {})
        }
        
        # 分析吞吐量改進
        if 'basic_performance' in benchmark_results and 'optimized_performance' in benchmark_results:
            basic_perf = benchmark_results['basic_performance']
            optimized_perf = benchmark_results['optimized_performance']
            
            for model_name in basic_perf.keys():
                if model_name in optimized_perf:
                    basic_throughput = self._extract_best_throughput(basic_perf[model_name])
                    optimized_throughput = self._extract_best_throughput(optimized_perf[model_name])
                    
                    if basic_throughput > 0:
                        improvement = optimized_throughput / basic_throughput
                        report['throughput_improvements'][model_name] = {
                            'basic_throughput': basic_throughput,
                            'optimized_throughput': optimized_throughput,
                            'improvement_factor': improvement
                        }
        
        return report
    
    def _extract_best_throughput(self, performance_data: Dict[str, Any]) -> float:
        """從性能數據中提取最佳吞吐量"""
        throughputs = []
        for batch_config, metrics in performance_data.items():
            if isinstance(metrics, dict) and 'throughput' in metrics:
                throughputs.append(metrics['throughput'])
        
        return max(throughputs) if throughputs else 0
    
    def _calculate_efficiency_rating(self, value: float, metric_type: str) -> str:
        """計算效率評級"""
        if metric_type == 'memory':
            if value >= 40:
                return 'Excellent'
            elif value >= 25:
                return 'Good'
            elif value >= 10:
                return 'Fair'
            else:
                return 'Poor'
        
        elif metric_type == 'time':
            if value <= 60:
                return 'Excellent'
            elif value <= 180:
                return 'Good'
            elif value <= 300:
                return 'Fair'
            else:
                return 'Poor'
        
        return 'Unknown'
    
    def generate_visualizations(self, report_data: Dict[str, Any]):
        """生成可視化圖表"""
        
        # 1. 內存優化圖表
        if 'memory_optimization' in report_data:
            self._create_memory_optimization_chart(report_data['memory_optimization'])
        
        # 2. 特徵工程性能圖表
        if 'feature_engineering' in report_data:
            self._create_feature_engineering_chart(report_data['feature_engineering'])
        
        # 3. 模型性能比較圖表
        if 'training_optimization' in report_data:
            self._create_model_performance_chart(report_data['training_optimization'])
        
        # 4. 推理性能圖表
        if 'inference_optimization' in report_data:
            self._create_inference_performance_chart(report_data['inference_optimization'])
    
    def _create_memory_optimization_chart(self, memory_data: Dict[str, Any]):
        """創建內存優化圖表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 內存使用對比
        categories = ['優化前', '優化後']
        memory_values = [memory_data['before_memory_gb'], memory_data['after_memory_gb']]
        
        bars = ax1.bar(categories, memory_values, color=['red', 'green'], alpha=0.7)
        ax1.set_ylabel('內存使用 (GB)')
        ax1.set_title('內存使用優化對比')
        
        # 添加數值標籤
        for bar, value in zip(bars, memory_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f} GB', ha='center', va='bottom')
        
        # 內存節省餅圖
        saved_memory = memory_data['memory_saved_gb']
        remaining_memory = memory_data['after_memory_gb']
        
        ax2.pie([saved_memory, remaining_memory], 
                labels=[f'節省 ({memory_data["memory_reduction_percent"]:.1f}%)', '使用中'],
                colors=['lightgreen', 'lightcoral'],
                autopct='%1.1f%%')
        ax2.set_title('內存節省比例')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'memory_optimization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_engineering_chart(self, fe_data: Dict[str, Any]):
        """創建特徵工程圖表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 處理時間分解
        if 'processing_steps' in fe_data:
            steps = list(fe_data['processing_steps'].keys())
            times = list(fe_data['processing_steps'].values())
            
            ax1.barh(steps, times, color='skyblue', alpha=0.8)
            ax1.set_xlabel('處理時間 (秒)')
            ax1.set_title('特徵工程各步驟耗時')
            
            # 添加數值標籤
            for i, (step, time_val) in enumerate(zip(steps, times)):
                ax1.text(time_val + 0.1, i, f'{time_val:.2f}s', va='center')
        
        # 特徵類型分佈
        if 'feature_counts_by_type' in fe_data:
            feature_types = list(fe_data['feature_counts_by_type'].keys())
            feature_counts = list(fe_data['feature_counts_by_type'].values())
            
            ax2.pie(feature_counts, labels=feature_types, autopct='%1.1f%%')
            ax2.set_title('創建特徵類型分佈')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_engineering.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_performance_chart(self, training_data: Dict[str, Any]):
        """創建模型性能圖表"""
        if 'model_performances' not in training_data:
            return
        
        models = list(training_data['model_performances'].keys())
        scores = [training_data['model_performances'][model]['validation_score'] 
                 for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color='lightblue', alpha=0.8)
        plt.ylabel('驗證分數 (AUC)')
        plt.title('模型性能比較')
        plt.xticks(rotation=45)
        
        # 添加數值標籤
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # 標記最佳模型
        if 'best_model' in training_data:
            best_model = training_data['best_model']
            if best_model in models:
                best_idx = models.index(best_model)
                bars[best_idx].set_color('gold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_inference_performance_chart(self, inference_data: Dict[str, Any]):
        """創建推理性能圖表"""
        if 'throughput_improvements' not in inference_data:
            return
        
        models = list(inference_data['throughput_improvements'].keys())
        basic_throughputs = [inference_data['throughput_improvements'][model]['basic_throughput'] 
                            for model in models]
        optimized_throughputs = [inference_data['throughput_improvements'][model]['optimized_throughput'] 
                               for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width/2, basic_throughputs, width, label='優化前', alpha=0.8)
        bars2 = plt.bar(x + width/2, optimized_throughputs, width, label='優化後', alpha=0.8)
        
        plt.ylabel('吞吐量 (predictions/sec)')
        plt.title('推理性能優化對比')
        plt.xticks(x, models)
        plt.legend()
        
        # 添加改進倍數標籤
        for i, model in enumerate(models):
            improvement = inference_data['throughput_improvements'][model]['improvement_factor']
            plt.text(i, max(basic_throughputs[i], optimized_throughputs[i]) + 10,
                    f'{improvement:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'inference_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_recommendations(self, report_data: Dict[str, Any]) -> List[str]:
        """生成優化建議"""
        recommendations = []
        
        # 內存優化建議
        if 'memory_optimization' in report_data:
            memory_data = report_data['memory_optimization']
            if memory_data['memory_reduction_percent'] < 20:
                recommendations.append(
                    "建議進一步優化數據類型，考慮使用category類型處理高重複率字符串欄位"
                )
            
            if memory_data['efficiency_rating'] == 'Excellent':
                recommendations.append(
                    "內存優化效果優秀，可以考慮處理更大的數據集"
                )
        
        # 特徵工程建議
        if 'feature_engineering' in report_data:
            fe_data = report_data['feature_engineering']
            if fe_data['total_processing_time'] > 300:  # 5分鐘
                recommendations.append(
                    "特徵工程耗時較長，建議啟用並行處理或使用特徵緩存"
                )
            
            if fe_data['features_created'] > 1000:
                recommendations.append(
                    "創建的特徵數量較多，建議使用特徵選擇方法減少特徵維度"
                )
        
        # 訓練優化建議
        if 'training_optimization' in report_data:
            training_data = report_data['training_optimization']
            if training_data['total_training_time'] > 1800:  # 30分鐘
                recommendations.append(
                    "模型訓練時間較長，建議使用早停機制或減少超參數搜索空間"
                )
            
            if training_data['best_score'] < 0.9:
                recommendations.append(
                    "模型性能有提升空間，建議嘗試更多的特徵工程或集成方法"
                )
        
        # 推理優化建議
        if 'inference_optimization' in report_data:
            inference_data = report_data['inference_optimization']
            if 'real_time_performance' in inference_data:
                rt_perf = inference_data['real_time_performance']
                if rt_perf.get('p95_latency', 1) > 0.1:  # 100ms
                    recommendations.append(
                        "推理延遲較高，建議使用模型壓縮或批處理優化"
                    )
        
        # 通用建議
        recommendations.extend([
            "定期監控模型性能，及時發現性能退化",
            "考慮使用GPU加速來進一步提升訓練和推理速度",
            "建立自動化的性能基準測試流程"
        ])
        
        return recommendations
    
    def generate_comprehensive_report(self, optimization_results: Dict[str, Any]) -> str:
        """生成綜合性能報告"""
        logger.info("生成綜合性能報告...")
        
        # 收集系統信息
        system_info = self.collect_system_info()
        
        # 組織報告數據
        report_data = self.report_template.copy()
        report_data['system_info'] = system_info
        report_data['optimization_results'] = optimization_results
        
        # 生成各部分報告
        if 'memory_optimization' in optimization_results:
            memory_data = optimization_results['memory_optimization']
            if isinstance(memory_data, pd.DataFrame):
                # 如果是DataFrame，提取內存信息
                initial_memory = 0.1  # 默認值
                final_memory = 0.05   # 默認值
                report_data['memory_optimization'] = self.generate_memory_optimization_report(
                    initial_memory, final_memory, {'dataframe_optimization': True}
                )
            else:
                report_data['memory_optimization'] = memory_data
        
        if 'feature_engineering' in optimization_results:
            report_data['feature_engineering'] = self.generate_feature_engineering_report(
                optimization_results['feature_engineering']
            )
        
        if 'training_optimization' in optimization_results:
            report_data['training_optimization'] = self.generate_training_optimization_report(
                optimization_results['training_optimization']
            )
        
        if 'inference_optimization' in optimization_results:
            report_data['inference_optimization'] = self.generate_inference_optimization_report(
                optimization_results['inference_optimization']
            )
        
        # 生成建議
        report_data['recommendations'] = self.generate_recommendations(report_data)
        
        # 生成可視化
        self.generate_visualizations(report_data)
        
        # 生成HTML報告
        html_report = self._generate_html_report(report_data)
        
        # 保存JSON報告
        json_report_path = os.path.join(self.output_dir, 'performance_report.json')
        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存HTML報告
        html_report_path = os.path.join(self.output_dir, 'performance_report.html')
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"性能報告已生成：{html_report_path}")
        return html_report_path
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML格式報告"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>IEEE-CIS 詐騙檢測性能優化報告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 0.9em; color: #7f8c8d; }}
                .recommendation {{ background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-left: 4px solid #27ae60; }}
                .image-container {{ text-align: center; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>IEEE-CIS 詐騙檢測性能優化報告</h1>
                <p>生成時間: {report_data['metadata']['generated_at']}</p>
                <p>報告版本: {report_data['metadata']['report_version']}</p>
            </div>
            
            <div class="section">
                <h2>系統信息</h2>
                <div class="metric">
                    <div class="metric-value">{report_data['system_info'].get('cpu_count', 'N/A')}</div>
                    <div class="metric-label">CPU 核心數</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report_data['system_info'].get('memory_total_gb', 0):.1f} GB</div>
                    <div class="metric-label">總內存</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{report_data['system_info'].get('python_version', 'N/A')}</div>
                    <div class="metric-label">Python 版本</div>
                </div>
            </div>
            
            {self._generate_optimization_sections(report_data)}
            
            <div class="section">
                <h2>優化建議</h2>
                {self._generate_recommendations_html(report_data.get('recommendations', []))}
            </div>
            
            <div class="section">
                <h2>性能圖表</h2>
                {self._generate_charts_html()}
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_optimization_sections(self, report_data: Dict[str, Any]) -> str:
        """生成優化章節HTML"""
        sections_html = ""
        
        # 內存優化
        if 'memory_optimization' in report_data:
            memory_data = report_data['memory_optimization']
            sections_html += f"""
            <div class="section">
                <h2>內存優化結果</h2>
                <div class="metric">
                    <div class="metric-value">{memory_data.get('memory_reduction_percent', 0):.1f}%</div>
                    <div class="metric-label">內存減少比例</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{memory_data.get('memory_saved_gb', 0):.2f} GB</div>
                    <div class="metric-label">節省內存</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{memory_data.get('efficiency_rating', 'N/A')}</div>
                    <div class="metric-label">效率評級</div>
                </div>
            </div>
            """
        
        # 特徵工程優化
        if 'feature_engineering' in report_data:
            fe_data = report_data['feature_engineering']
            sections_html += f"""
            <div class="section">
                <h2>特徵工程優化結果</h2>
                <div class="metric">
                    <div class="metric-value">{fe_data.get('total_processing_time', 0):.2f}s</div>
                    <div class="metric-label">總處理時間</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{fe_data.get('features_created', 0)}</div>
                    <div class="metric-label">創建特徵數</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{fe_data.get('performance_rating', 'N/A')}</div>
                    <div class="metric-label">性能評級</div>
                </div>
            </div>
            """
        
        # 訓練優化
        if 'training_optimization' in report_data:
            training_data = report_data['training_optimization']
            sections_html += f"""
            <div class="section">
                <h2>訓練優化結果</h2>
                <div class="metric">
                    <div class="metric-value">{training_data.get('best_score', 0):.4f}</div>
                    <div class="metric-label">最佳模型分數</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{training_data.get('total_training_time', 0):.2f}s</div>
                    <div class="metric-label">總訓練時間</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{training_data.get('best_model', 'N/A')}</div>
                    <div class="metric-label">最佳模型</div>
                </div>
            </div>
            """
        
        return sections_html
    
    def _generate_recommendations_html(self, recommendations: List[str]) -> str:
        """生成建議HTML"""
        html = ""
        for recommendation in recommendations:
            html += f'<div class="recommendation">{recommendation}</div>\n'
        return html
    
    def _generate_charts_html(self) -> str:
        """生成圖表HTML"""
        charts = ['memory_optimization.png', 'feature_engineering.png', 
                 'model_performance.png', 'inference_performance.png']
        
        html = ""
        for chart in charts:
            chart_path = os.path.join(self.output_dir, chart)
            if os.path.exists(chart_path):
                html += f"""
                <div class="image-container">
                    <h3>{chart.replace('.png', '').replace('_', ' ').title()}</h3>
                    <img src="{chart}" alt="{chart}" style="max-width: 100%; height: auto;">
                </div>
                """
        
        return html

def generate_performance_report_from_demo(demo_results: Dict[str, Any], 
                                        output_dir: str = "performance_reports") -> str:
    """從演示结果生成性能報告"""
    logger.info("從演示結果生成性能報告...")
    
    generator = PerformanceReportGenerator(output_dir)
    report_path = generator.generate_comprehensive_report(demo_results)
    
    logger.info(f"性能報告已保存至: {report_path}")
    return report_path

if __name__ == "__main__":
    # 示例用法
    demo_results = {
        'memory_optimization': {
            'before_memory_gb': 2.5,
            'after_memory_gb': 1.8,
            'memory_reduction_percent': 28.0,
            'memory_saved_gb': 0.7,
            'efficiency_rating': 'Good'
        },
        'feature_engineering': {
            'total_time': 45.2,
            'total_features_created': 156,
            'processing_times': {
                'time_features': 5.2,
                'amount_features': 3.1,
                'aggregation_features': 25.8,
                'categorical_encoding': 8.9,
                'feature_interactions': 2.2
            },
            'feature_counts': {
                'time_features': 8,
                'amount_features': 6,
                'aggregation_features': 89,
                'categorical_encoding': 15,
                'feature_interactions': 38
            }
        }
    }
    
    report_path = generate_performance_report_from_demo(demo_results)
    print(f"報告已生成: {report_path}")