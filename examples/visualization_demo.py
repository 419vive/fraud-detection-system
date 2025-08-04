"""
可視化系統演示 - IEEE-CIS 詐騙檢測項目
展示如何使用完整的可視化和分析系統
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 導入我們的可視化模組
from src.visualization_engine import VisualizationEngine, create_fraud_detection_visualizations
from src.realtime_dashboard import RealTimeMonitoringSystem, simulate_transaction_data
from src.model_comparison_viz import ModelComparisonVisualizer, compare_fraud_detection_models
from src.business_analytics import BusinessAnalyzer, create_business_analytics_suite

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """生成示例詐騙檢測數據"""
    logger.info(f"生成 {n_samples} 筆示例數據...")
    
    np.random.seed(42)
    
    # 基本交易特徵
    data = {
        'TransactionID': [f'TXN_{i:07d}' for i in range(n_samples)],
        'TransactionDT': np.random.randint(86400, 86400*30, n_samples),  # 30天內的時間戳
        'TransactionAmt': np.random.lognormal(4, 1.5, n_samples),  # 對數正態分佈的金額
        
        # 用戶特徵
        'card1': np.random.randint(1000, 20000, n_samples),
        'card2': np.random.randint(100, 600, n_samples),
        'card3': np.random.randint(100, 200, n_samples),
        'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], n_samples),
        'card5': np.random.randint(100, 300, n_samples),
        'card6': np.random.choice(['debit', 'credit'], n_samples),
        
        # 地址特徵
        'addr1': np.random.randint(100, 600, n_samples),
        'addr2': np.random.randint(10, 100, n_samples),
        
        # 設備特徵
        'DeviceType': np.random.choice(['desktop', 'mobile'], n_samples, p=[0.6, 0.4]),
        'DeviceInfo': [f'Device_{np.random.randint(1, 1000)}' for _ in range(n_samples)],
        
        # 用戶ID特徵
        'id_01': np.random.uniform(0, 1, n_samples),
        'id_02': np.random.uniform(0, 100, n_samples),
        'id_03': np.random.uniform(0, 1, n_samples),
        'id_04': np.random.uniform(0, 1, n_samples),
        'id_05': np.random.uniform(0, 100, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 生成詐騙標籤（基於一些規則）
    fraud_probability = (
        (df['TransactionAmt'] > 1000) * 0.3 +  # 大額交易更可能是詐騙
        (df['id_01'] > 0.8) * 0.2 +  # 某些ID特徵
        (df['card4'] == 'american express') * 0.1 +  # 某些卡片類型
        ((df['TransactionDT'] % 86400) < 21600) * 0.15 +  # 凌晨時段
        np.random.uniform(0, 0.1, n_samples)  # 隨機噪聲
    )
    
    df['isFraud'] = (fraud_probability > 0.5).astype(int)
    
    # 調整詐騙率到合理水平（約3-5%）
    fraud_indices = df[df['isFraud'] == 1].index
    keep_fraud = np.random.choice(fraud_indices, size=int(0.04 * n_samples), replace=False)
    df.loc[df.index.difference(keep_fraud), 'isFraud'] = 0
    
    logger.info(f"數據生成完成 - 詐騙率: {df['isFraud'].mean():.2%}")
    return df

def train_multiple_models(X_train, y_train, X_test, y_test):
    """訓練多個模型用於比較"""
    logger.info("訓練多個模型...")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    model_results = {}
    
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
        
        # 特徵重要性
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_[0])
        
        # 保存結果
        model_results[name] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_importance': feature_importance,
            'feature_names': X_train.columns.tolist() if hasattr(X_train, 'columns') else None,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'model': model
        }
        
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"{name} - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    
    return model_results

def demo_basic_visualizations():
    """演示基本可視化功能"""
    logger.info("=== 演示基本可視化功能 ===")
    
    # 生成示例數據
    df = generate_sample_data(5000)
    
    # 數據預處理
    categorical_cols = ['card4', 'card6', 'DeviceType']
    for col in categorical_cols:
        df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    
    # 選擇特徵
    feature_cols = [col for col in df.columns if col not in ['TransactionID', 'isFraud']]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols]
    y = df['isFraud']
    
    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 訓練模型
    model_results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # 創建可視化引擎
    viz_engine = VisualizationEngine()
    
    # 1. 模型性能儀表板
    logger.info("創建模型性能儀表板...")
    best_model_name = max(model_results.keys(), key=lambda k: roc_auc_score(
        model_results[k]['y_true'], model_results[k]['y_pred_proba']
    ))
    best_results = model_results[best_model_name]
    
    performance_fig = viz_engine.create_model_performance_dashboard(
        best_results['y_true'],
        best_results['y_pred'],
        best_results['y_pred_proba'],
        save_path='output/model_performance_dashboard.html'
    )
    
    # 2. 交易模式分析
    logger.info("創建交易模式分析...")
    pattern_fig = viz_engine.create_transaction_pattern_analysis(df)
    viz_engine.save_dashboard(pattern_fig, 'output/transaction_patterns.html')
    
    # 3. 地理分析
    logger.info("創建地理分析...")
    geo_fig = viz_engine.create_geographic_analysis(df)
    viz_engine.save_dashboard(geo_fig, 'output/geographic_analysis.html')
    
    # 4. 綜合報告
    logger.info("生成綜合可視化報告...")
    report_files = viz_engine.create_comprehensive_report(
        df, best_results, 'output/comprehensive_report'
    )
    
    logger.info("基本可視化演示完成！")
    logger.info(f"生成的文件: {list(report_files.values())}")

def demo_model_comparison():
    """演示模型比較功能"""
    logger.info("=== 演示模型比較功能 ===")
    
    # 生成數據並訓練模型
    df = generate_sample_data(3000)
    
    # 數據預處理
    categorical_cols = ['card4', 'card6', 'DeviceType']
    for col in categorical_cols:
        df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    
    feature_cols = [col for col in df.columns if col not in ['TransactionID', 'isFraud']]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols]
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 訓練模型
    model_results = train_multiple_models(X_train, y_train, X_test, y_test)
    
    # 模型比較
    logger.info("創建模型比較分析...")
    comparison_files = compare_fraud_detection_models(
        model_results, 'output/model_comparison'
    )
    
    logger.info("模型比較演示完成！")
    logger.info(f"生成的文件: {list(comparison_files.values())}")

def demo_business_analytics():
    """演示商業分析功能"""
    logger.info("=== 演示商業分析功能 ===")
    
    # 生成數據並訓練模型
    df = generate_sample_data(8000)
    
    # 數據預處理
    categorical_cols = ['card4', 'card6', 'DeviceType']
    for col in categorical_cols:
        df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    
    feature_cols = [col for col in df.columns if col not in ['TransactionID', 'isFraud']]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[numeric_cols]
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 訓練最佳模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 對完整數據集進行預測
    predictions = model.predict(X)
    prediction_probabilities = model.predict_proba(X)[:, 1]
    
    # 創建商業分析
    logger.info("創建商業分析儀表板...")
    business_files = create_business_analytics_suite(
        df, predictions, prediction_probabilities, 'output/business_analytics'
    )
    
    logger.info("商業分析演示完成！")
    logger.info(f"生成的文件: {list(business_files.values())}")

def demo_realtime_monitoring():
    """演示實時監控功能"""
    logger.info("=== 演示實時監控功能 ===")
    
    # 創建實時監控系統
    monitor = RealTimeMonitoringSystem()
    
    # 啟動監控
    monitor.start_monitoring(dashboard_port=8050)
    
    logger.info("實時監控系統已啟動...")
    logger.info("正在模擬交易數據...")
    
    # 在後台線程中模擬數據
    import threading
    sim_thread = threading.Thread(
        target=simulate_transaction_data,
        args=(monitor, 10, 30),  # 10分鐘，每分鐘30筆交易
        daemon=True
    )
    sim_thread.start()
    
    logger.info("訪問 http://localhost:8050 查看實時儀表板")
    logger.info("按 Ctrl+C 停止監控系統")
    
    try:
        # 運行儀表板服務器
        monitor.run_dashboard(debug=False, host='127.0.0.1', port=8050)
    except KeyboardInterrupt:
        logger.info("停止實時監控系統...")
        monitor.stop_monitoring()

def create_demo_index():
    """創建演示索引頁面"""
    logger.info("創建演示索引頁面...")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>詐騙檢測可視化系統演示</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 40px; }
            .demo-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
            .demo-section h2 { color: #34495e; margin-top: 0; }
            .file-list { background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .file-list a { display: block; padding: 8px 0; color: #3498db; text-decoration: none; }
            .file-list a:hover { text-decoration: underline; }
            .highlight { background: #e8f5e8; padding: 15px; border-left: 4px solid #2ecc71; margin: 15px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 詐騙檢測可視化系統演示</h1>
            
            <div class="highlight">
                <strong>系統功能概覽：</strong><br>
                • 全面的模型性能分析和比較<br>
                • 實時交易監控和警報系統<br>
                • 深度商業洞察和財務分析<br>
                • 互動式儀表板和報告生成
            </div>
            
            <div class="demo-section">
                <h2>📊 基本可視化功能</h2>
                <p>包含模型性能分析、交易模式分析、地理分佈等核心功能</p>
                <div class="file-list">
                    <a href="comprehensive_report/index.html">📈 綜合分析報告</a>
                    <a href="model_performance_dashboard.html">🎯 模型性能儀表板</a>
                    <a href="transaction_patterns.html">📊 交易模式分析</a>
                    <a href="geographic_analysis.html">🗺️ 地理分佈分析</a>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>🔍 模型比較分析</h2>
                <p>多模型性能對比、特徵重要性分析、錯誤分析等</p>
                <div class="file-list">
                    <a href="model_comparison/model_comparison_dashboard.html">⚖️ 模型比較儀表板</a>
                    <a href="model_comparison/feature_importance_analysis.html">🎯 特徵重要性分析</a>
                    <a href="model_comparison/error_analysis.html">🔍 錯誤分析報告</a>
                    <a href="model_comparison/comparison_report.json">📋 詳細比較報告</a>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>💼 商業分析套件</h2>
                <p>財務影響分析、投資回報率、風險評估和策略建議</p>
                <div class="file-list">
                    <a href="business_analytics/business_index.html">🏢 商業分析中心</a>
                    <a href="business_analytics/financial_impact_dashboard.html">💰 財務影響分析</a>
                    <a href="business_analytics/business_performance_dashboard.html">📊 業務績效監控</a>
                    <a href="business_analytics/risk_assessment_dashboard.html">🎯 風險評估控制</a>
                    <a href="business_analytics/executive_summary.json">📋 執行摘要</a>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>⚡ 實時監控系統</h2>
                <p>實時交易監控、警報系統、性能追蹤（需要運行Python腳本）</p>
                <div class="highlight">
                    <strong>啟動方式：</strong><br>
                    <code>python examples/visualization_demo.py --realtime</code><br>
                    然後訪問 <a href="http://localhost:8050">http://localhost:8050</a>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>🛠️ 使用說明</h2>
                <p><strong>運行完整演示：</strong></p>
                <pre><code>python examples/visualization_demo.py</code></pre>
                
                <p><strong>單獨運行某個功能：</strong></p>
                <pre><code>python examples/visualization_demo.py --basic
python examples/visualization_demo.py --comparison  
python examples/visualization_demo.py --business
python examples/visualization_demo.py --realtime</code></pre>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
                生成時間: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
            </div>
        </div>
    </body>
    </html>
    """
    
    os.makedirs('output', exist_ok=True)
    with open('output/demo_index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info("演示索引頁面已創建: output/demo_index.html")

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='詐騙檢測可視化系統演示')
    parser.add_argument('--basic', action='store_true', help='運行基本可視化演示')
    parser.add_argument('--comparison', action='store_true', help='運行模型比較演示')
    parser.add_argument('--business', action='store_true', help='運行商業分析演示')
    parser.add_argument('--realtime', action='store_true', help='運行實時監控演示')
    parser.add_argument('--all', action='store_true', help='運行全部演示（除實時監控）')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    os.makedirs('output', exist_ok=True)
    
    logger.info("🚀 詐騙檢測可視化系統演示開始")
    
    try:
        if args.basic or args.all:
            demo_basic_visualizations()
        
        if args.comparison or args.all:
            demo_model_comparison()
        
        if args.business or args.all:
            demo_business_analytics()
        
        if args.realtime:
            demo_realtime_monitoring()
        
        if not any([args.basic, args.comparison, args.business, args.realtime, args.all]):
            # 默認運行全部演示（除實時監控）
            demo_basic_visualizations()
            demo_model_comparison() 
            demo_business_analytics()
        
        # 創建索引頁面
        create_demo_index()
        
        logger.info("✅ 所有演示完成！")
        logger.info("📂 查看生成的文件在 output/ 目錄")
        logger.info("🌐 打開 output/demo_index.html 查看演示索引")
        
    except Exception as e:
        logger.error(f"演示過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()