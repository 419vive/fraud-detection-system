# 🎨 詐騙檢測可視化系統

這是一個為IEEE-CIS詐騙檢測項目設計的綜合可視化和分析系統，提供從數據驗證到商業洞察的完整視覺化解決方案。

## 🌟 系統特色

### 🔍 全面的可視化功能
- **模型性能分析**: ROC曲線、混淆矩陣、特徵重要性
- **數據探索**: 交易模式、地理分佈、時間序列分析
- **模型比較**: 多模型性能對比、特徵重要性差異分析
- **商業洞察**: 財務影響分析、投資回報率、風險評估

### ⚡ 實時監控
- **動態儀表板**: 實時交易監控和詐騙檢測
- **智能警報系統**: 自動異常檢測和通知
- **性能追蹤**: 模型漂移檢測和系統健康監控

### 💼 商業智能
- **財務影響分析**: 詐騙損失、預防收益、ROI計算
- **風險管理**: 風險分級、控制策略建議
- **執行報告**: 高層決策支持和關鍵指標摘要

## 📁 系統架構

```
src/
├── visualization_engine.py      # 核心可視化引擎
├── realtime_dashboard.py       # 實時監控儀表板
├── model_comparison_viz.py     # 模型比較可視化
├── business_analytics.py       # 商業分析模組
├── integrated_pipeline.py      # 集成分析流水線
├── data_validation.py          # 數據驗證（已存在）
├── feature_engineering.py      # 特徵工程（已存在）
├── model_monitoring.py         # 模型監控（已存在）
└── config.py                   # 配置管理（已存在）

examples/
└── visualization_demo.py       # 完整演示腳本
```

## 🚀 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 或使用conda環境
conda create -n fraud_viz python=3.9
conda activate fraud_viz
pip install -r requirements.txt
```

### 2. 運行完整演示

```bash
# 運行所有可視化演示
python examples/visualization_demo.py

# 運行特定功能
python examples/visualization_demo.py --basic      # 基本可視化
python examples/visualization_demo.py --comparison # 模型比較
python examples/visualization_demo.py --business   # 商業分析
python examples/visualization_demo.py --realtime   # 實時監控
```

### 3. 使用集成流水線

```python
from src.integrated_visualization_pipeline import run_integrated_fraud_detection_analysis

# 載入數據
df = pd.read_csv('your_fraud_data.csv')

# 運行完整分析
results = run_integrated_fraud_detection_analysis(
    df=df,
    target_col='isFraud',
    output_dir='analysis_output',
    models_to_train=['random_forest', 'xgboost', 'lightgbm'],
    enable_realtime=True
)
```

## 🎯 主要功能模組

### 1. 核心可視化引擎 (`visualization_engine.py`)

提供基礎的可視化功能：

```python
from src.visualization_engine import VisualizationEngine

viz_engine = VisualizationEngine()

# 創建模型性能儀表板
performance_fig = viz_engine.create_model_performance_dashboard(
    y_true, y_pred, y_pred_proba
)

# 創建交易模式分析
pattern_fig = viz_engine.create_transaction_pattern_analysis(df)

# 生成綜合報告
reports = viz_engine.create_comprehensive_report(df, model_results)
```

### 2. 實時監控系統 (`realtime_dashboard.py`)

設置實時監控和警報：

```python
from src.realtime_dashboard import RealTimeMonitoringSystem

# 創建監控系統
monitor = RealTimeMonitoringSystem()

# 啟動監控
monitor.start_monitoring(dashboard_port=8050)

# 添加交易數據
monitor.add_transaction({
    'transaction_id': 'TXN_123',
    'amount': 1500.0,
    'timestamp': datetime.now()
})

# 運行儀表板
monitor.run_dashboard()
```

### 3. 模型比較分析 (`model_comparison_viz.py`)

比較多個模型的性能：

```python
from src.model_comparison_viz import ModelComparisonVisualizer

comparator = ModelComparisonVisualizer()

# 添加模型結果
for model_name, results in model_results.items():
    comparator.add_model_results(model_name, results)

# 創建比較儀表板
dashboard = comparator.create_performance_comparison_dashboard()

# 特徵重要性分析
feature_analysis = comparator.create_feature_importance_analysis()
```

### 4. 商業分析模組 (`business_analytics.py`)

深度商業洞察和財務分析：

```python
from src.business_analytics import BusinessAnalyzer

analyzer = BusinessAnalyzer()

# 計算財務影響
financial_impact = analyzer.calculate_financial_impact(
    df, predictions, prediction_probabilities
)

# 創建商業儀表板
business_dashboards = analyzer.create_comprehensive_business_dashboard(
    df, predictions, prediction_probabilities
)
```

## 📊 可視化類型

### 1. 模型性能可視化
- **ROC曲線和AUC分數**
- **精確率-召回率曲線**
- **混淆矩陣熱力圖**
- **特徵重要性排名**
- **預測概率分佈**
- **模型性能雷達圖**

### 2. 數據分析可視化
- **交易時間分佈分析**
- **金額分佈和異常檢測**
- **地理位置風險熱力圖**
- **客戶行為模式分析**
- **詐騙趨勢時間序列**

### 3. 實時監控儀表板
- **即時交易量指示器**
- **詐騙檢測率儀表**
- **系統性能監控**
- **警報狀態面板**
- **風險分數分佈**

### 4. 商業智能可視化
- **ROI和財務影響分析**
- **成本效益瀑布圖**
- **風險評估矩陣**
- **投資回報趨勢**
- **執行摘要儀表板**

## 🛠️ 配置和自定義

### 1. 可視化主題配置

```python
# 自定義顏色方案
viz_engine = VisualizationEngine(theme='plotly_dark')

# 自定義顏色映射
color_palette = {
    'fraud': '#e74c3c',
    'legitimate': '#2ecc71',
    'warning': '#f39c12'
}
```

### 2. 業務參數配置

```python
# 自定義業務參數
business_params = {
    'average_investigation_cost': 75.0,
    'false_positive_impact': 30.0,
    'fraud_prevention_value_multiplier': 6.0
}

analyzer = BusinessAnalyzer()
analyzer.business_params.update(business_params)
```

### 3. 監控警報配置

```python
# 自定義警報閾值
alert_config = {
    'fraud_rate': 0.06,          # 6%詐騙率警報
    'model_performance': 0.80,    # 80% AUC閾值
    'system_latency': 0.5,       # 500ms延遲警報
    'data_quality': 0.90         # 90%數據品質閾值
}
```

## 📈 性能優化

### 1. 大數據處理
- **分塊處理**: 自動處理大型數據集
- **並行計算**: 多進程特徵工程和分析
- **內存優化**: 智能數據類型優化
- **緩存機制**: 重複計算結果緩存

### 2. 可視化優化
- **懶加載**: 按需生成圖表
- **採樣顯示**: 大數據集自動採樣
- **響應式設計**: 適配不同屏幕尺寸
- **批量導出**: 高效生成多個報告

## 🔧 故障排除

### 常見問題解決

1. **內存不足**
   ```python
   # 減少數據量或使用採樣
   df_sample = df.sample(n=10000, random_state=42)
   ```

2. **可視化渲染慢**
   ```python
   # 使用靜態圖表而非交互式
   viz_engine.save_dashboard(fig, 'output.png', format_type='png')
   ```

3. **實時監控連接問題**
   ```bash
   # 檢查端口是否被佔用
   lsof -i :8050
   ```

4. **依賴包衝突**
   ```bash
   # 使用虛擬環境
   python -m venv fraud_viz_env
   source fraud_viz_env/bin/activate  # Linux/Mac
   # 或 fraud_viz_env\Scripts\activate  # Windows
   ```

## 🎨 可視化範例

### 1. 模型性能儀表板
![模型性能儀表板示例](docs/images/model_performance_dashboard.png)

### 2. 實時監控界面
![實時監控界面示例](docs/images/realtime_monitoring.png)

### 3. 商業分析報告
![商業分析報告示例](docs/images/business_analytics.png)

## 🤝 擴展開發

### 添加新的可視化類型

```python
class CustomVisualizationEngine(VisualizationEngine):
    def create_custom_analysis(self, data):
        # 實現自定義分析邏輯
        fig = go.Figure()
        # ... 添加圖表元素
        return fig
```

### 自定義警報規則

```python
def custom_alert_rule(data):
    # 實現自定義警報邏輯
    return alert_triggered, alert_details

alert_system.alert_rules['custom_rule'] = {
    'condition': custom_alert_rule,
    'threshold': 0.1,
    'severity': 'high',
    'message': '自定義警報觸發'
}
```

## 📚 API文檔

詳細的API文檔請參考各模組的docstring註釋，或使用以下命令生成：

```bash
# 生成API文檔
python -m pydoc -w src.visualization_engine
python -m pydoc -w src.realtime_dashboard
python -m pydoc -w src.business_analytics
```

## 🔄 更新日誌

### v1.0.0 (2024-01-15)
- ✨ 初始版本發布
- 🎨 核心可視化引擎
- ⚡ 實時監控系統
- 💼 商業分析模組
- 🔍 模型比較功能

### 未來計劃
- 🤖 AI驅動的洞察建議
- 🌐 Web API接口
- 📱 移動端適配
- 🔐 企業級安全功能

## 📞 支持與貢獻

- **問題回報**: 請在GitHub Issues中提交
- **功能建議**: 歡迎提交Pull Request
- **文檔改進**: 協助完善使用說明

---

**開發團隊**: IEEE-CIS 詐騙檢測項目組  
**更新時間**: 2024年1月15日  
**版本**: v1.0.0

> 這個可視化系統是為了讓詐騙檢測不僅僅是技術問題，更成為業務決策的重要工具。透過直觀的可視化和深度分析，幫助組織更好地理解、監控和優化其詐騙防護能力。