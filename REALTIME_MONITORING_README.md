# 實時監控系統 - IEEE-CIS 詐騙檢測項目

## 🎯 系統概述

本實時監控系統為詐騙檢測項目提供全面的模型監控、數據漂移檢測和智能警報功能。系統採用模組化設計，支援多種檢測算法和通知渠道，確保模型在生產環境中的穩定運行。

## 🚀 核心功能

### 1. 實時模型監控
- **性能指標追蹤**：AUC、精確率、召回率、F1分數等
- **預測吞吐量監控**：處理速度、延遲統計
- **系統資源監控**：CPU、內存、磁碟使用率
- **自動性能退化檢測**：智能識別模型性能下降

### 2. 高級數據漂移檢測
- **多種檢測算法**：
  - PSI（人口穩定性指數）檢測
  - PCA（主成分分析）漂移檢測
  - 孤立森林異常檢測
  - 集成檢測器（加權投票）
- **實時漂移監控**：自動檢測特徵分佈變化
- **視覺化分析**：漂移趨勢圖表和報告

### 3. 智能警報系統
- **多渠道通知**：Slack、Teams、Email、Webhook
- **規則引擎**：可配置的警報規則和閾值
- **警報級別**：低、中、高、嚴重四個級別
- **智能去重**：防止警報風暴的冷卻機制

### 4. 互動式監控儀表板
- **實時數據展示**：動態更新的性能指標
- **警報面板**：活躍警報狀態展示
- **歷史趨勢分析**：長期性能變化追蹤
- **Web介面**：基於Dash的專業儀表板

## 📁 文件結構

```
src/
├── realtime_monitoring_system.py     # 實時監控系統主模組
├── advanced_drift_detection.py       # 高級漂移檢測算法
├── alert_notification_system.py      # 警報和通知系統
└── model_monitoring.py              # 基礎模型監控（已存在）

examples/
└── realtime_monitoring_demo.py       # 完整演示程序

REALTIME_MONITORING_README.md         # 本文檔
```

## 🛠️ 安裝和配置

### 安裝依賴

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly dash
pip install aiohttp requests psutil asyncio
```

### 基本配置

在 `src/config.py` 中添加監控配置：

```python
MONITORING_CONFIG = {
    'notifications': {
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            'username': '詐騙檢測監控',
            'icon': ':warning:'
        },
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-password',
            'from_email': 'monitoring@yourcompany.com',
            'to_emails': ['admin@yourcompany.com']
        }
    },
    'thresholds': {
        'performance_degradation': 0.05,  # 5% AUC下降
        'drift_detection': 0.05,          # p值閾值
        'system_cpu': 80.0,               # CPU使用率80%
        'system_memory': 85.0             # 記憶體使用率85%
    }
}
```

## 📊 使用方法

### 1. 快速開始

```python
from src.realtime_monitoring_system import create_realtime_monitoring_system
from src.advanced_drift_detection import create_comprehensive_drift_monitor
import pandas as pd

# 創建監控系統
monitoring_system = create_realtime_monitoring_system()

# 添加模型到監控
monitoring_system.add_model("fraud_detector", your_model, reference_data)

# 啟動監控
monitoring_system.start_monitoring()

# 記錄預測結果
monitoring_system.log_prediction("fraud_detector", {
    'y_true': y_true,
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba,
    'current_data': current_data
})
```

### 2. 運行完整演示

```bash
cd examples
python realtime_monitoring_demo.py
```

演示包含：
- 基礎監控功能
- 漂移檢測演示
- 警報系統測試
- 集成監控場景
- 互動式儀表板

### 3. 啟動監控儀表板

```python
from src.realtime_monitoring_system import run_monitoring_dashboard

# 創建並配置監控系統
monitoring_system = create_realtime_monitoring_system()
# ... 添加模型和配置 ...

# 啟動儀表板
run_monitoring_dashboard(monitoring_system, host='0.0.0.0', port=8050)
```

訪問 `http://localhost:8050` 查看監控面板

## 🔧 進階配置

### 自定義漂移檢測器

```python
from src.advanced_drift_detection import EnsembleDriftDetector

# 創建集成檢測器
detector = EnsembleDriftDetector()

# 調整檢測器權重
detector.weights = {
    'psi': 0.5,
    'pca': 0.3,
    'isolation_forest': 0.2
}

# 執行檢測
results = detector.detect_drift(reference_data, current_data)
consensus = detector.get_consensus_result(results)
```

### 自定義警報規則

```python
from src.alert_notification_system import AlertRule, AlertSeverity

# 創建自定義規則
custom_rule = AlertRule(
    name="high_fraud_rate",
    condition=lambda alert: (
        alert.alert_type == "fraud_rate" and 
        alert.details.get('fraud_rate', 0) > 0.1
    ),
    channels=['slack', 'email'],
    config={'cooldown_minutes': 30}
)

# 添加到系統
notification_system.add_rule(custom_rule)
```

### 配置通知渠道

```python
# Slack配置
notification_system.configure_channel('slack', {
    'webhook_url': 'https://hooks.slack.com/services/...',
    'username': '詐騙檢測監控',
    'icon': ':rotating_light:'
})

# Teams配置
notification_system.configure_channel('teams', {
    'webhook_url': 'https://outlook.office.com/webhook/...'
})

# Webhook配置
notification_system.configure_channel('webhook', {
    'url': 'https://your-api.com/alerts',
    'headers': {'Authorization': 'Bearer your-token'}
})
```

## 📈 監控指標

### 模型性能指標
- **ROC-AUC**：接收者操作特徵曲線下面積
- **精確率**：正確預測的詐騙交易比例
- **召回率**：成功檢測的詐騙交易比例
- **F1分數**：精確率和召回率的調和平均
- **預測延遲**：平均預測響應時間
- **吞吐量**：每秒處理的交易數量

### 漂移檢測指標
- **PSI分數**：人口穩定性指數（<0.1無變化，0.1-0.2輕微，>0.2顯著）
- **重構誤差**：PCA重構誤差分佈變化
- **異常比例**：孤立森林檢測的異常樣本比例
- **統計檢驗**：KS檢驗、卡方檢驗p值

### 系統監控指標
- **CPU使用率**：處理器負載百分比
- **記憶體使用率**：RAM使用百分比
- **磁碟使用率**：存儲空間使用百分比
- **網路流量**：輸入輸出流量統計

## 🚨 警報類型

### 性能相關警報
- `performance_degradation`：模型性能下降
- `model_failure`：模型預測失敗
- `prediction_latency`：預測延遲過高
- `throughput_drop`：處理吞吐量下降

### 數據相關警報
- `data_drift`：數據分佈漂移
- `feature_anomaly`：特徵異常值
- `missing_data`：缺失數據過多
- `data_quality`：數據質量問題

### 系統相關警報
- `system_cpu_high`：CPU使用率過高
- `system_memory_high`：記憶體使用率過高
- `disk_space_low`：磁碟空間不足
- `service_unavailable`：服務不可用

## 📋 最佳實踐

### 1. 監控策略
- **分層監控**：系統→模型→數據三個層次
- **閾值設定**：根據業務需求設定合理警報閾值
- **歷史基線**：建立穩定的性能基線參考
- **定期校準**：定期更新漂移檢測的參考數據

### 2. 警報管理
- **優先級分級**：合理設定警報優先級
- **去重機制**：避免相同類型警報的重複發送
- **升級策略**：嚴重警報的自動升級機制
- **回應追蹤**：記錄警報處理狀態和結果

### 3. 性能優化
- **批處理**：批量處理監控數據提高效率
- **異步處理**：使用異步IO處理通知發送
- **數據清理**：定期清理歷史監控數據
- **資源限制**：設定監控系統的資源使用上限

## 🔍 故障排除

### 常見問題

1. **監控數據不更新**
   - 檢查監控系統是否正常啟動
   - 驗證模型預測數據是否正確記錄
   - 查看日誌文件中的錯誤信息

2. **警報發送失敗**
   - 確認通知渠道配置正確
   - 檢查網路連接和認證信息
   - 查看警報規則是否正確匹配

3. **漂移檢測誤報**
   - 調整漂移檢測閾值
   - 更新參考數據集
   - 檢查數據預處理一致性

4. **儀表板載入緩慢**
   - 減少歷史數據展示範圍
   - 優化查詢和聚合邏輯
   - 增加服務器資源配置

### 日誌分析

```bash
# 查看監控系統日誌
tail -f monitoring.log

# 過濾警報相關日誌
grep "WARNING\|ERROR" monitoring.log

# 分析漂移檢測結果
grep "drift" monitoring.log | tail -20
```

## 🔮 未來增強

### 計劃功能
- **機器學習驅動的異常檢測**：使用ML算法自動識別異常模式
- **預測性維護**：基於歷史趨勢預測潛在問題
- **A/B測試支援**：支援多模型比較和性能測試
- **自動化響應**：基於警報類型的自動化處理流程

### 集成計劃
- **MLflow集成**：與實驗追蹤平台集成
- **Kubernetes支援**：容器化部署和擴展
- **Apache Kafka集成**：實時數據流處理
- **Prometheus/Grafana**：指標收集和可視化

## 📞 支援和貢獻

如有問題或建議，請：
1. 查看本文檔的故障排除部分
2. 檢查項目日誌文件
3. 在項目中創建Issue
4. 提交Pull Request貢獻代碼

---

**注意**：本系統專為詐騙檢測場景設計，但架構足夠通用，可適用於其他機器學習監控需求。在生產環境使用前，請進行充分測試並根據實際需求調整配置參數。