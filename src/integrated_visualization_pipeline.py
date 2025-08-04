"""
集成可視化流水線 - IEEE-CIS 詐騙檢測項目
整合特徵工程、模型訓練、可視化分析的完整流水線
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import os
import json
import pickle
from pathlib import Path

# 導入現有模組
from .feature_engineering import FeatureEngineer, engineer_features
from .data_validation import DataValidator
from .model_monitoring import ModelMonitor
from .visualization_engine import VisualizationEngine
from .realtime_dashboard import RealTimeMonitoringSystem
from .model_comparison_viz import ModelComparisonVisualizer
from .business_analytics import BusinessAnalyzer
from .config import get_config

# 機器學習模組
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class IntegratedVisualizationPipeline:
    """集成可視化流水線"""
    
    def __init__(self, config_manager=None, output_base_dir: str = 'fraud_detection_output'):
        self.config = config_manager or get_config()
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        
        # 初始化各個組件
        self.data_validator = DataValidator(config_manager)
        self.feature_engineer = FeatureEngineer(config_manager)
        self.viz_engine = VisualizationEngine(config_manager)
        self.model_comparator = ModelComparisonVisualizer(config_manager)
        self.business_analyzer = BusinessAnalyzer(config_manager)
        
        # 存儲結果
        self.validation_results = {}
        self.feature_engineering_results = {}
        self.model_results = {}
        self.visualization_outputs = {}
        
        # 運行統計
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_time': None,
            'stages_completed': [],
            'errors': []
        }
    
    def run_complete_pipeline(self, 
                            df: pd.DataFrame,
                            target_col: str = 'isFraud',
                            test_size: float = 0.2,
                            enable_realtime_monitoring: bool = False,
                            models_to_train: List[str] = None) -> Dict[str, Any]:
        """運行完整的可視化分析流水線"""
        
        self.pipeline_stats['start_time'] = datetime.now()
        logger.info(f"🚀 開始執行集成可視化流水線 - {self.pipeline_stats['start_time']}")
        
        try:
            # 階段1: 數據驗證
            logger.info("📊 階段1: 數據驗證與品質分析")
            validation_output = self._run_data_validation(df, target_col)
            self.pipeline_stats['stages_completed'].append('data_validation')
            
            # 階段2: 特徵工程
            logger.info("🔧 階段2: 特徵工程與數據預處理")
            processed_df = self._run_feature_engineering(df, target_col)
            self.pipeline_stats['stages_completed'].append('feature_engineering')
            
            # 階段3: 模型訓練與比較
            logger.info("🤖 階段3: 模型訓練與性能比較")
            model_results = self._run_model_training_and_comparison(
                processed_df, target_col, test_size, models_to_train
            )
            self.pipeline_stats['stages_completed'].append('model_training')
            
            # 階段4: 可視化分析
            logger.info("📈 階段4: 綜合可視化分析")
            visualization_outputs = self._run_visualization_analysis(
                df, processed_df, model_results, target_col
            )
            self.pipeline_stats['stages_completed'].append('visualization')
            
            # 階段5: 商業分析
            logger.info("💼 階段5: 商業洞察與財務分析")
            business_outputs = self._run_business_analysis(
                processed_df, model_results, target_col
            )
            self.pipeline_stats['stages_completed'].append('business_analysis')
            
            # 階段6: 實時監控（可選）
            if enable_realtime_monitoring:
                logger.info("⚡ 階段6: 實時監控系統設置")
                monitoring_setup = self._setup_realtime_monitoring(model_results)
                self.pipeline_stats['stages_completed'].append('realtime_monitoring')
            
            # 生成最終報告
            logger.info("📋 生成最終綜合報告")
            final_report = self._generate_final_report()
            
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['total_time'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()
            
            logger.info(f"✅ 流水線執行完成 - 總耗時: {self.pipeline_stats['total_time']:.2f}秒")
            
            return {
                'pipeline_stats': self.pipeline_stats,
                'validation_results': validation_output,
                'model_results': model_results,
                'visualization_outputs': visualization_outputs,
                'business_outputs': business_outputs,
                'final_report': final_report
            }
            
        except Exception as e:
            self.pipeline_stats['errors'].append(str(e))
            logger.error(f"❌ 流水線執行失敗: {e}")
            raise
    
    def _run_data_validation(self, df: pd.DataFrame, target_col: str) -> Dict[str, str]:
        """運行數據驗證"""
        validation_dir = self.output_base_dir / 'data_validation'
        validation_dir.mkdir(exist_ok=True)
        
        # 執行全面數據驗證
        validation_files = self.data_validator.comprehensive_data_validation_report(
            df, target_col, str(validation_dir)
        )
        
        self.validation_results = validation_files
        logger.info(f"數據驗證完成，生成 {len(validation_files)} 個報告文件")
        
        return validation_files
    
    def _run_feature_engineering(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """運行特徵工程"""
        logger.info("執行優化的特徵工程流水線...")
        
        # 使用現有的特徵工程模組
        processed_df, summary = engineer_features(
            df, target_col, enable_parallel=True, enable_advanced=True
        )
        
        # 保存特徵工程結果
        fe_dir = self.output_base_dir / 'feature_engineering'
        fe_dir.mkdir(exist_ok=True)
        
        # 保存處理後的數據
        processed_df.to_csv(fe_dir / 'processed_data.csv', index=False)
        
        # 保存特徵工程摘要
        with open(fe_dir / 'feature_engineering_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        self.feature_engineering_results = {
            'summary': summary,
            'processed_data_path': str(fe_dir / 'processed_data.csv'),
            'original_features': len(df.columns),
            'engineered_features': len(processed_df.columns),
            'features_added': len(processed_df.columns) - len(df.columns)
        }
        
        logger.info(f"特徵工程完成 - 原始特徵: {len(df.columns)}, 工程後特徵: {len(processed_df.columns)}")
        
        return processed_df
    
    def _run_model_training_and_comparison(self, 
                                         df: pd.DataFrame, 
                                         target_col: str,
                                         test_size: float,
                                         models_to_train: List[str] = None) -> Dict[str, Any]:
        """運行模型訓練與比較"""
        
        if models_to_train is None:
            models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'logistic']
        
        # 準備數據
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number])  # 只使用數值特徵
        y = df[target_col]
        
        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"數據分割完成 - 訓練集: {len(X_train)}, 測試集: {len(X_test)}")
        
        # 定義模型
        models = {}
        if 'random_forest' in models_to_train:
            models['Random Forest'] = RandomForestClassifier(
                **self.config.get_model_params('random_forest')
            )
        
        if 'logistic' in models_to_train:
            models['Logistic Regression'] = LogisticRegression(
                **self.config.get_model_params('logistic')
            )
        
        if 'xgboost' in models_to_train:
            models['XGBoost'] = xgb.XGBClassifier(
                **self.config.get_model_params('xgboost'),
                verbosity=0
            )
        
        if 'lightgbm' in models_to_train:
            models['LightGBM'] = lgb.LGBMClassifier(
                **self.config.get_model_params('lightgbm')
            )
        
        # 訓練和評估模型
        model_results = {}
        model_dir = self.output_base_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            logger.info(f"訓練 {name}...")
            
            # 訓練模型
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 預測
            start_time = datetime.now()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            # 交叉驗證
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1
            )
            
            # 特徵重要性
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
            
            # 保存模型
            model_path = model_dir / f'{name.lower().replace(" ", "_")}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 計算指標
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # 存儲結果
            model_results[name] = {
                'model': model,
                'model_path': str(model_path),
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_importance': feature_importance,
                'feature_names': X.columns.tolist(),
                'training_time': training_time,
                'prediction_time': prediction_time,
                'auc_score': auc_score,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{name} - AUC: {auc_score:.4f}, CV AUC: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # 模型比較分析
        comparison_dir = self.output_base_dir / 'model_comparison'
        comparison_dir.mkdir(exist_ok=True)
        
        # 添加模型結果到比較器
        for name, results in model_results.items():
            self.model_comparator.add_model_results(name, results)
        
        # 生成比較報告
        comparison_dashboard = self.model_comparator.create_comprehensive_comparison_dashboard()
        comparison_dashboard.write_html(comparison_dir / 'model_comparison_dashboard.html')
        
        feature_analysis = self.model_comparator.create_feature_importance_analysis()
        feature_analysis.write_html(comparison_dir / 'feature_importance_analysis.html')
        
        comparison_report = self.model_comparator.generate_comparison_report(
            str(comparison_dir / 'comparison_report.json')
        )
        
        self.model_results = model_results
        
        return {
            'models': model_results,
            'comparison_report': comparison_report,
            'best_model': max(model_results.items(), key=lambda x: x[1]['auc_score']),
            'comparison_files': {
                'dashboard': str(comparison_dir / 'model_comparison_dashboard.html'),
                'feature_analysis': str(comparison_dir / 'feature_importance_analysis.html'),
                'report': str(comparison_dir / 'comparison_report.json')
            }
        }
    
    def _run_visualization_analysis(self, 
                                  original_df: pd.DataFrame,
                                  processed_df: pd.DataFrame,
                                  model_results: Dict[str, Any],
                                  target_col: str) -> Dict[str, str]:
        """運行可視化分析"""
        
        viz_dir = self.output_base_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 獲取最佳模型結果
        best_model_name, best_model_data = model_results['best_model']
        
        # 1. 模型性能儀表板
        performance_fig = self.viz_engine.create_model_performance_dashboard(
            best_model_data['y_true'],
            best_model_data['y_pred'],
            best_model_data['y_pred_proba']
        )
        performance_path = viz_dir / 'model_performance_dashboard.html'
        performance_fig.write_html(performance_path)
        
        # 2. 交易模式分析
        pattern_fig = self.viz_engine.create_transaction_pattern_analysis(original_df)
        pattern_path = viz_dir / 'transaction_patterns_analysis.html'
        pattern_fig.write_html(pattern_path)
        
        # 3. 地理分析
        geo_fig = self.viz_engine.create_geographic_analysis(original_df)
        geo_path = viz_dir / 'geographic_analysis.html'
        geo_fig.write_html(geo_path)
        
        # 4. 綜合報告
        comprehensive_reports = self.viz_engine.create_comprehensive_report(
            original_df, best_model_data, str(viz_dir / 'comprehensive_analysis')
        )
        
        visualization_outputs = {
            'model_performance': str(performance_path),
            'transaction_patterns': str(pattern_path),
            'geographic_analysis': str(geo_path),
            'comprehensive_reports': comprehensive_reports
        }
        
        self.visualization_outputs = visualization_outputs
        
        return visualization_outputs
    
    def _run_business_analysis(self, 
                             df: pd.DataFrame,
                             model_results: Dict[str, Any],
                             target_col: str) -> Dict[str, str]:
        """運行商業分析"""
        
        business_dir = self.output_base_dir / 'business_analysis'
        business_dir.mkdir(exist_ok=True)
        
        # 獲取最佳模型的預測結果
        best_model_name, best_model_data = model_results['best_model']
        
        # 對整個數據集進行預測（用於商業分析）
        model = best_model_data['model']
        feature_cols = [col for col in df.columns if col != target_col]
        X_full = df[feature_cols].select_dtypes(include=[np.number])
        
        full_predictions = model.predict(X_full)
        full_prediction_probabilities = model.predict_proba(X_full)[:, 1]
        
        # 創建商業分析儀表板
        business_dashboards = self.business_analyzer.create_comprehensive_business_dashboard(
            df, full_predictions, full_prediction_probabilities, str(business_dir)
        )
        
        return business_dashboards
    
    def _setup_realtime_monitoring(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """設置實時監控系統"""
        
        monitoring_dir = self.output_base_dir / 'realtime_monitoring'
        monitoring_dir.mkdir(exist_ok=True)
        
        # 獲取最佳模型
        best_model_name, best_model_data = model_results['best_model']
        
        # 創建監控系統配置
        monitoring_config = {
            'model_name': best_model_name,
            'model_path': best_model_data['model_path'],
            'feature_names': best_model_data['feature_names'],
            'monitoring_port': 8051,
            'alert_thresholds': {
                'fraud_rate': 0.05,
                'model_performance': 0.85,
                'system_latency': 1.0,
                'data_quality': 0.95
            }
        }
        
        # 保存配置
        config_path = monitoring_dir / 'monitoring_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(monitoring_config, f, ensure_ascii=False, indent=2)
        
        # 創建監控啟動腳本
        startup_script = f"""
#!/usr/bin/env python
# 實時監控系統啟動腳本

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.realtime_dashboard import RealTimeMonitoringSystem
import json

# 載入配置
with open('{config_path}', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 創建監控系統
monitor = RealTimeMonitoringSystem()

# 啟動監控
monitor.start_monitoring(dashboard_port=config['monitoring_port'])

print(f"實時監控系統已啟動 - 訪問 http://localhost:{{config['monitoring_port']}}")
print("按 Ctrl+C 停止系統")

try:
    monitor.run_dashboard(port=config['monitoring_port'])
except KeyboardInterrupt:
    print("停止監控系統...")
    monitor.stop_monitoring()
"""
        
        script_path = monitoring_dir / 'start_monitoring.py'
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(startup_script)
        
        # 使腳本可執行
        os.chmod(script_path, 0o755)
        
        return {
            'config_path': str(config_path),
            'startup_script': str(script_path),
            'monitoring_port': monitoring_config['monitoring_port']
        }
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """生成最終綜合報告"""
        
        report = {
            'generation_time': datetime.now().isoformat(),
            'pipeline_execution': self.pipeline_stats,
            'data_summary': {
                'validation_files': len(self.validation_results),
                'feature_engineering': self.feature_engineering_results,
                'models_trained': len(self.model_results.get('models', {})),
                'visualization_outputs': len(self.visualization_outputs)
            },
            'best_model_performance': None,
            'key_insights': [],
            'recommendations': [],
            'output_files': {}
        }
        
        # 最佳模型性能
        if 'models' in self.model_results:
            best_model_name, best_model_data = self.model_results['best_model']
            report['best_model_performance'] = {
                'model_name': best_model_name,
                'auc_score': best_model_data['auc_score'],
                'cv_mean': best_model_data['cv_mean'],
                'cv_std': best_model_data['cv_std']
            }
        
        # 關鍵洞察
        if report['best_model_performance']:
            auc = report['best_model_performance']['auc_score']
            if auc > 0.9:
                report['key_insights'].append("🎯 模型性能優異，AUC超過0.9")
            elif auc > 0.8:
                report['key_insights'].append("✅ 模型性能良好，AUC超過0.8")
            else:
                report['key_insights'].append("⚠️ 模型性能需要提升")
        
        if 'features_added' in self.feature_engineering_results:
            features_added = self.feature_engineering_results['features_added']
            report['key_insights'].append(f"🔧 特徵工程新增了 {features_added} 個特徵")
        
        # 建議
        report['recommendations'].extend([
            "定期重新訓練模型以保持性能",
            "監控模型在生產環境中的表現",
            "持續收集新的特徵以改善檢測能力",
            "建立完善的警報機制"
        ])
        
        # 輸出文件整理
        all_outputs = {}
        all_outputs.update(self.validation_results)
        all_outputs.update(self.visualization_outputs)
        if 'comparison_files' in self.model_results:
            all_outputs.update(self.model_results['comparison_files'])
        
        report['output_files'] = all_outputs
        
        # 保存最終報告
        report_path = self.output_base_dir / 'final_comprehensive_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # 創建HTML版本的報告
        self._create_html_final_report(report)
        
        return report
    
    def _create_html_final_report(self, report: Dict[str, Any]):
        """創建HTML版本的最終報告"""
        
        best_model = report.get('best_model_performance', {})
        pipeline_stats = report.get('pipeline_execution', {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>詐騙檢測系統 - 綜合分析報告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center; }}
                .header h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }}
                .stat-card {{ background: white; padding: 25px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); text-align: center; }}
                .stat-value {{ font-size: 2.5em; font-weight: bold; color: #3498db; margin: 10px 0; }}
                .stat-label {{ color: #666; font-size: 0.9em; }}
                .section {{ background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); margin: 20px 0; }}
                .section h2 {{ color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .insights {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
                .insight-item {{ background: #e8f6ff; padding: 15px; border-radius: 10px; border-left: 4px solid #3498db; }}
                .file-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
                .file-item {{ background: #f8f9fa; padding: 15px; border-radius: 10px; }}
                .file-item a {{ color: #3498db; text-decoration: none; font-weight: bold; }}
                .file-item a:hover {{ text-decoration: underline; }}
                .timeline {{ background: #f8f9fa; padding: 20px; border-radius: 10px; }}
                .timeline-item {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 詐騙檢測系統綜合分析報告</h1>
                    <p>生成時間: {report['generation_time']}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{best_model.get('auc_score', 0):.3f}</div>
                        <div class="stat-label">最佳模型 AUC 分數</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(report.get('output_files', {}))}</div>
                        <div class="stat-label">生成的分析文件</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{pipeline_stats.get('total_time', 0):.1f}s</div>
                        <div class="stat-label">總執行時間</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(pipeline_stats.get('stages_completed', []))}</div>
                        <div class="stat-label">完成的分析階段</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🏆 最佳模型性能</h2>
                    <p><strong>模型名稱:</strong> {best_model.get('model_name', 'N/A')}</p>
                    <p><strong>AUC 分數:</strong> {best_model.get('auc_score', 0):.4f}</p>
                    <p><strong>交叉驗證平均:</strong> {best_model.get('cv_mean', 0):.4f} ± {best_model.get('cv_std', 0):.4f}</p>
                </div>
                
                <div class="section">
                    <h2>🔍 關鍵洞察</h2>
                    <div class="insights">
                        {''.join([f'<div class="insight-item">{insight}</div>' for insight in report.get('key_insights', [])])}
                    </div>
                </div>
                
                <div class="section">
                    <h2>💡 建議事項</h2>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in report.get('recommendations', [])])}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>📊 分析文件</h2>
                    <div class="file-grid">
                        <div class="file-item">
                            <h4>📈 數據驗證</h4>
                            <a href="data_validation/data_quality_dashboard.html">數據品質儀表板</a><br>
                            <a href="data_validation/data_validation_report.md">驗證報告</a>
                        </div>
                        <div class="file-item">
                            <h4>🤖 模型比較</h4>
                            <a href="model_comparison/model_comparison_dashboard.html">模型比較儀表板</a><br>
                            <a href="model_comparison/feature_importance_analysis.html">特徵重要性分析</a>
                        </div>
                        <div class="file-item">
                            <h4>📊 可視化分析</h4>
                            <a href="visualizations/model_performance_dashboard.html">性能分析</a><br>
                            <a href="visualizations/transaction_patterns_analysis.html">交易模式</a>
                        </div>
                        <div class="file-item">
                            <h4>💼 商業分析</h4>
                            <a href="business_analysis/business_index.html">商業分析中心</a><br>
                            <a href="business_analysis/financial_impact_dashboard.html">財務影響</a>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>⏱️ 執行時間線</h2>
                    <div class="timeline">
                        {''.join([f'<div class="timeline-item">✅ {stage.replace("_", " ").title()}</div>' for stage in pipeline_stats.get('stages_completed', [])])}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_path = self.output_base_dir / 'final_comprehensive_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"最終報告已生成: {html_path}")

# 便捷函數
def run_integrated_fraud_detection_analysis(df: pd.DataFrame,
                                          target_col: str = 'isFraud',
                                          output_dir: str = 'fraud_detection_analysis',
                                          models_to_train: List[str] = None,
                                          enable_realtime: bool = False) -> Dict[str, Any]:
    """運行完整的詐騙檢測分析流水線"""
    
    pipeline = IntegratedVisualizationPipeline(output_base_dir=output_dir)
    
    return pipeline.run_complete_pipeline(
        df=df,
        target_col=target_col,
        models_to_train=models_to_train,
        enable_realtime_monitoring=enable_realtime
    )

if __name__ == "__main__":
    print("集成可視化流水線已載入完成！")