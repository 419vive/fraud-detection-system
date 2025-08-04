"""
實時監控系統演示 - IEEE-CIS 詐騙檢測項目
展示完整的實時監控功能，包括模型監控、漂移檢測和警報系統
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import sys
import os

# 添加src目錄到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime_monitoring_system import (
    RealTimeMonitoringSystem, 
    create_realtime_monitoring_system,
    run_monitoring_dashboard
)
from advanced_drift_detection import (
    ComprehensiveDriftMonitor,
    create_comprehensive_drift_monitor,
    quick_drift_analysis
)
from alert_notification_system import (
    AlertNotificationSystem,
    AlertSeverity,
    create_alert,
    create_notification_system
)
from model_monitoring import create_model_monitor

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples: int = 1000, add_drift: bool = False) -> pd.DataFrame:
    """生成示例詐騙檢測數據"""
    np.random.seed(42)
    
    # 基礎特徵
    data = {
        'TransactionAmt': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'ProductCD_W': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'card1': np.random.randint(1000, 9999, size=n_samples),
        'card2': np.random.randint(100, 999, size=n_samples),
        'card3': np.random.randint(100, 300, size=n_samples),
        'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], size=n_samples),
        'card6': np.random.choice(['debit', 'credit'], size=n_samples, p=[0.6, 0.4]),
        'C1': np.random.randint(0, 5000, size=n_samples),
        'C2': np.random.randint(0, 5000, size=n_samples),
        'C3': np.random.randint(0, 100, size=n_samples),
        'C4': np.random.randint(0, 1000, size=n_samples),
        'C5': np.random.randint(0, 500, size=n_samples),
        'D1': np.random.randint(0, 640, size=n_samples),
        'D2': np.random.randint(0, 640, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些數值特徵
    df['hour'] = np.random.randint(0, 24, size=n_samples)
    df['day_of_week'] = np.random.randint(0, 7, size=n_samples)
    
    # 如果需要添加漂移
    if add_drift:
        # 模擬交易金額漂移（通脹效應）
        df['TransactionAmt'] = df['TransactionAmt'] * np.random.uniform(1.1, 1.3, size=n_samples)
        
        # 模擬卡片類型分佈漂移
        df['ProductCD_W'] = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])  # 比例變化
        
        # 模擬時間模式漂移
        df['hour'] = np.random.choice(range(24), size=n_samples, 
                                    p=np.concatenate([np.repeat(0.02, 8), np.repeat(0.06, 8), np.repeat(0.02, 8)]))
    
    # 生成詐騙標籤（約3%的詐騙率）
    fraud_probability = 0.03 + df['TransactionAmt'] / df['TransactionAmt'].max() * 0.05
    df['isFraud'] = np.random.binomial(1, fraud_probability)
    
    return df

class MockModel:
    """模擬機器學習模型"""
    
    def __init__(self, name: str):
        self.name = name
        self.feature_importance = None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """模擬預測"""
        # 基於特徵的簡單邏輯
        scores = (
            X['TransactionAmt'] / X['TransactionAmt'].max() * 0.3 +
            X['ProductCD_W'] * 0.2 +
            X['C1'] / X['C1'].max() * 0.2 +
            (X['hour'] > 22).astype(int) * 0.3
        )
        return (scores > 0.4).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """模擬預測概率"""
        scores = (
            X['TransactionAmt'] / X['TransactionAmt'].max() * 0.3 +
            X['ProductCD_W'] * 0.2 +
            X['C1'] / X['C1'].max() * 0.2 +
            (X['hour'] > 22).astype(int) * 0.3
        )
        return scores.clip(0, 1)

async def demonstrate_basic_monitoring():
    """演示基礎監控功能"""
    logger.info("=== 基礎監控功能演示 ===")
    
    # 1. 生成訓練數據和測試數據
    logger.info("生成示例數據...")
    train_data = generate_sample_data(5000, add_drift=False)
    test_data = generate_sample_data(1000, add_drift=True)  # 包含漂移的測試數據
    
    # 2. 創建模型
    model = MockModel("fraud_detector_v1")
    
    # 3. 創建實時監控系統
    monitoring_system = create_realtime_monitoring_system()
    monitoring_system.add_model("fraud_detector_v1", model, train_data)
    
    # 4. 啟動監控
    monitoring_system.start_monitoring()
    
    # 5. 模擬預測和監控
    logger.info("開始模擬預測...")
    y_true = test_data['isFraud'].values
    y_pred = model.predict(test_data.drop('isFraud', axis=1))
    y_pred_proba = model.predict_proba(test_data.drop('isFraud', axis=1))
    
    # 記錄預測結果
    monitoring_system.log_prediction("fraud_detector_v1", {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'current_data': test_data.drop('isFraud', axis=1),
        'prediction_times': [0.1] * len(y_pred)  # 模擬預測時間
    })
    
    # 等待處理完成
    await asyncio.sleep(2)
    
    # 6. 檢查監控狀態
    status = monitoring_system.get_monitoring_status()
    logger.info(f"監控狀態: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    # 7. 停止監控
    monitoring_system.stop_monitoring()
    
    return monitoring_system

async def demonstrate_drift_detection():
    """演示漂移檢測功能"""
    logger.info("=== 漂移檢測功能演示 ===")
    
    # 1. 生成參考數據和當前數據
    reference_data = generate_sample_data(3000, add_drift=False)
    current_data = generate_sample_data(1000, add_drift=True)
    
    # 2. 創建漂移監控器
    drift_monitor = create_comprehensive_drift_monitor(
        reference_data.drop('isFraud', axis=1),
        features=['TransactionAmt', 'ProductCD_W', 'card1', 'C1', 'C2', 'hour']
    )
    
    # 3. 執行漂移檢測
    logger.info("執行漂移檢測...")
    monitoring_result = drift_monitor.monitor_drift(current_data.drop('isFraud', axis=1))
    
    # 4. 顯示結果
    consensus = monitoring_result['consensus_result']
    logger.info(f"漂移檢測結果:")
    logger.info(f"  檢測到漂移: {consensus.drift_detected}")
    logger.info(f"  漂移分數: {consensus.drift_score:.4f}")
    logger.info(f"  受影響特徵: {consensus.affected_features}")
    
    # 5. 顯示各檢測器結果
    for detector_name, result in monitoring_result['individual_results'].items():
        logger.info(f"  {detector_name}: 漂移={result.drift_detected}, 分數={result.drift_score:.4f}")
    
    # 6. 生成漂移報告
    drift_report = drift_monitor.create_drift_report(days=1)
    logger.info(f"漂移報告: {json.dumps(drift_report, indent=2, ensure_ascii=False)}")
    
    return drift_monitor

async def demonstrate_alert_system():
    """演示警報系統功能"""
    logger.info("=== 警報系統功能演示 ===")
    
    # 1. 創建通知系統
    notification_system = create_notification_system()
    
    # 2. 配置通知渠道（示例配置）
    # 注意：實際使用時需要配置真實的webhook URL和認證信息
    notification_system.configure_channel('slack', {
        'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        'username': '詐騙檢測監控',
        'icon': ':warning:'
    })
    
    # 3. 啟動通知處理
    processing_task = asyncio.create_task(notification_system.start_processing())
    
    # 4. 發送不同級別的測試警報
    test_alerts = [
        create_alert(
            alert_type="model_performance",
            severity=AlertSeverity.HIGH,
            title="模型性能下降",
            message="詐騙檢測模型的AUC分數從0.85下降至0.78",
            source="fraud_detector_v1",
            details={
                "previous_auc": 0.85,
                "current_auc": 0.78,
                "degradation": 0.07
            }
        ),
        create_alert(
            alert_type="data_drift",
            severity=AlertSeverity.MEDIUM,
            title="數據漂移檢測",
            message="交易金額特徵檢測到顯著漂移",
            source="drift_monitor",
            details={
                "feature": "TransactionAmt",
                "psi_score": 0.15,
                "drift_type": "distribution"
            }
        ),
        create_alert(
            alert_type="system_resource",
            severity=AlertSeverity.CRITICAL,
            title="系統資源警報",
            message="CPU使用率持續超過90%",
            source="system_monitor",
            details={
                "cpu_usage": 92.5,
                "memory_usage": 87.3,
                "duration_minutes": 15
            }
        )
    ]
    
    # 5. 發送警報
    logger.info("發送測試警報...")
    for alert in test_alerts:
        await notification_system.send_alert(alert)
        await asyncio.sleep(1)  # 間隔發送
    
    # 6. 等待處理完成
    await asyncio.sleep(3)
    
    # 7. 獲取統計信息
    stats = notification_system.get_alert_statistics(days=1)
    logger.info(f"警報統計: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 8. 停止處理
    notification_system.stop_processing()
    await processing_task
    
    return notification_system

async def demonstrate_integrated_monitoring():
    """演示集成監控功能"""
    logger.info("=== 集成監控功能演示 ===")
    
    # 1. 設置所有組件
    train_data = generate_sample_data(5000, add_drift=False)
    test_data = generate_sample_data(1000, add_drift=True)
    
    model = MockModel("integrated_fraud_detector")
    monitoring_system = create_realtime_monitoring_system()
    notification_system = create_notification_system()
    
    # 2. 配置集成系統
    monitoring_system.add_model("integrated_fraud_detector", model, train_data)
    
    # 3. 啟動所有系統
    monitoring_system.start_monitoring()
    processing_task = asyncio.create_task(notification_system.start_processing())
    
    # 4. 模擬真實運行場景
    logger.info("模擬真實運行場景...")
    
    for batch_idx in range(5):  # 模擬5個批次的處理
        logger.info(f"處理批次 {batch_idx + 1}/5...")
        
        # 生成當前批次數據（逐漸增加漂移）
        drift_factor = batch_idx * 0.2
        batch_data = generate_sample_data(200, add_drift=(drift_factor > 0))
        
        # 模擬預測
        y_true = batch_data['isFraud'].values
        y_pred = model.predict(batch_data.drop('isFraud', axis=1))
        y_pred_proba = model.predict_proba(batch_data.drop('isFraud', axis=1))
        
        # 記錄到監控系統
        monitoring_system.log_prediction("integrated_fraud_detector", {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'current_data': batch_data.drop('isFraud', axis=1),
            'prediction_times': np.random.uniform(0.05, 0.2, len(y_pred))
        })
        
        # 檢查是否需要發送警報
        if batch_idx >= 2:  # 從第3批開始可能觸發警報
            # 模擬性能下降警報
            if np.random.random() > 0.7:
                alert = create_alert(
                    alert_type="batch_processing",
                    severity=AlertSeverity.MEDIUM,
                    title=f"批次 {batch_idx + 1} 處理完成",
                    message=f"已處理 {len(batch_data)} 筆交易，詐騙率: {np.mean(y_true):.3f}",
                    source="integrated_fraud_detector",
                    details={
                        "batch_id": batch_idx + 1,
                        "transaction_count": len(batch_data),
                        "fraud_rate": float(np.mean(y_true)),
                        "avg_prediction_time": float(np.mean(np.random.uniform(0.05, 0.2, len(y_pred))))
                    }
                )
                await notification_system.send_alert(alert)
        
        await asyncio.sleep(2)  # 模擬處理間隔
    
    # 5. 生成最終報告
    logger.info("生成最終監控報告...")
    
    final_status = monitoring_system.get_monitoring_status()
    alert_stats = notification_system.get_alert_statistics(days=1)
    
    final_report = {
        "monitoring_session": {
            "duration": "演示模式",
            "batches_processed": 5,
            "total_transactions": 1000,
            "monitoring_status": final_status,
            "alert_statistics": alert_stats
        },
        "generated_at": datetime.now().isoformat()
    }
    
    logger.info(f"最終報告: {json.dumps(final_report, indent=2, ensure_ascii=False)}")
    
    # 6. 清理
    monitoring_system.stop_monitoring()
    notification_system.stop_processing()
    await processing_task
    
    return final_report

def run_dashboard_demo():
    """運行監控儀表板演示"""
    logger.info("=== 監控儀表板演示 ===")
    
    # 創建監控系統
    train_data = generate_sample_data(3000, add_drift=False)
    model = MockModel("dashboard_demo_model")
    monitoring_system = create_realtime_monitoring_system()
    monitoring_system.add_model("dashboard_demo_model", model, train_data)
    
    # 啟動監控
    monitoring_system.start_monitoring()
    
    # 生成一些歷史數據
    logger.info("生成監控歷史數據...")
    for i in range(10):
        test_batch = generate_sample_data(100, add_drift=(i > 5))
        y_true = test_batch['isFraud'].values
        y_pred = model.predict(test_batch.drop('isFraud', axis=1))
        y_pred_proba = model.predict_proba(test_batch.drop('isFraud', axis=1))
        
        monitoring_system.log_prediction("dashboard_demo_model", {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'current_data': test_batch.drop('isFraud', axis=1)
        })
    
    try:
        logger.info("啟動監控儀表板...")
        logger.info("請在瀏覽器中訪問 http://127.0.0.1:8050")
        logger.info("按 Ctrl+C 停止儀表板")
        
        run_monitoring_dashboard(monitoring_system, debug=True)
        
    except KeyboardInterrupt:
        logger.info("儀表板已停止")
    finally:
        monitoring_system.stop_monitoring()

async def main():
    """主演示函數"""
    logger.info("詐騙檢測實時監控系統演示開始")
    
    try:
        # 1. 基礎監控演示
        await demonstrate_basic_monitoring()
        await asyncio.sleep(1)
        
        # 2. 漂移檢測演示
        await demonstrate_drift_detection()
        await asyncio.sleep(1)
        
        # 3. 警報系統演示
        await demonstrate_alert_system()
        await asyncio.sleep(1)
        
        # 4. 集成監控演示
        await demonstrate_integrated_monitoring()
        
        logger.info("所有演示完成！")
        
        # 5. 詢問是否啟動儀表板
        try:
            response = input("\n是否啟動監控儀表板？(y/n): ")
            if response.lower() in ['y', 'yes']:
                run_dashboard_demo()
        except KeyboardInterrupt:
            logger.info("演示結束")
            
    except Exception as e:
        logger.error(f"演示過程中發生錯誤: {e}")
        raise

if __name__ == "__main__":
    # 運行演示
    asyncio.run(main())