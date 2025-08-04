"""
實時監控系統 - IEEE-CIS 詐騙檢測項目
提供實時模型監控、警報系統和儀表板功能
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import queue
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import websockets
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import psutil
import requests

from .model_monitoring import ModelMonitor, PerformanceMetrics, DriftMetrics
from .config import get_config
from .exceptions import ModelError

logger = logging.getLogger(__name__)

@dataclass
class RealTimeAlert:
    """實時警報數據類"""
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    model_name: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class SystemMetrics:
    """系統指標數據類"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    prediction_throughput: float
    active_connections: int
    queue_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class AlertManager:
    """警報管理器"""
    
    def __init__(self):
        self.alerts: List[RealTimeAlert] = []
        self.alert_handlers: Dict[str, List[Callable]] = {}
        self.notification_channels: Dict[str, Dict] = {}
        
    def add_alert_handler(self, alert_type: str, handler: Callable):
        """添加警報處理器"""
        if alert_type not in self.alert_handlers:
            self.alert_handlers[alert_type] = []
        self.alert_handlers[alert_type].append(handler)
    
    def configure_notification_channel(self, channel_name: str, config: Dict[str, Any]):
        """配置通知渠道"""
        self.notification_channels[channel_name] = config
    
    def trigger_alert(self, alert: RealTimeAlert):
        """觸發警報"""
        self.alerts.append(alert)
        logger.warning(f"觸發警報: {alert.alert_type} - {alert.message}")
        
        # 執行警報處理器
        if alert.alert_type in self.alert_handlers:
            for handler in self.alert_handlers[alert.alert_type]:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"警報處理器執行失敗: {e}")
        
        # 發送通知
        self._send_notifications(alert)
    
    def _send_notifications(self, alert: RealTimeAlert):
        """發送通知"""
        for channel_name, config in self.notification_channels.items():
            try:
                if channel_name == 'slack':
                    self._send_slack_notification(alert, config)
                elif channel_name == 'email':
                    self._send_email_notification(alert, config)
                elif channel_name == 'webhook':
                    self._send_webhook_notification(alert, config)
            except Exception as e:
                logger.error(f"通知發送失敗 ({channel_name}): {e}")
    
    def _send_slack_notification(self, alert: RealTimeAlert, config: Dict):
        """發送Slack通知"""
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return
        
        color_map = {
            'low': 'good',
            'medium': 'warning', 
            'high': 'danger',
            'critical': 'danger'
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, 'warning'),
                "title": f"模型監控警報 - {alert.alert_type}",
                "text": alert.message,
                "fields": [
                    {"title": "模型", "value": alert.model_name, "short": True},
                    {"title": "嚴重程度", "value": alert.severity, "short": True},
                    {"title": "時間", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                ],
                "footer": "詐騙檢測監控系統",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        requests.post(webhook_url, json=payload, timeout=10)
    
    def _send_webhook_notification(self, alert: RealTimeAlert, config: Dict):
        """發送Webhook通知"""
        webhook_url = config.get('url')
        if not webhook_url:
            return
        
        payload = alert.to_dict()
        requests.post(webhook_url, json=payload, timeout=10)
    
    def _send_email_notification(self, alert: RealTimeAlert, config: Dict):
        """發送郵件通知（需要配置SMTP）"""
        # 這裡可以實現郵件發送邏輯
        pass
    
    def get_active_alerts(self, hours: int = 24) -> List[RealTimeAlert]:
        """獲取最近的活躍警報"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts 
            if alert.timestamp >= cutoff_time and not alert.resolved
        ]
    
    def resolve_alert(self, alert_id: str):
        """解決警報"""
        for alert in self.alerts:
            if str(id(alert)) == alert_id:
                alert.resolved = True
                logger.info(f"警報已解決: {alert.alert_type}")
                break

class SystemMonitor:
    """系統監控器"""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: int = 30):
        """開始系統監控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("系統監控已啟動")
    
    def stop_monitoring(self):
        """停止系統監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("系統監控已停止")
    
    def _monitor_loop(self, interval: int):
        """監控循環"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 保持歷史記錄在合理範圍內
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-800:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"系統監控錯誤: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系統指標"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            prediction_throughput=0.0,  # 需要從預測服務獲取
            active_connections=0,  # 需要從應用服務獲取
            queue_size=0  # 需要從隊列系統獲取
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """獲取當前系統指標"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """獲取指標歷史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]

class RealTimeMonitoringSystem:
    """實時監控系統主類"""
    
    def __init__(self, models: Dict[str, Any] = None):
        self.models = models or {}
        self.model_monitors: Dict[str, ModelMonitor] = {}
        self.alert_manager = AlertManager()
        self.system_monitor = SystemMonitor()
        self.prediction_queue = queue.Queue()
        self.monitoring_active = False
        self.config = get_config()
        
        # 設置警報閾值
        self.alert_thresholds = {
            'performance_degradation': 0.05,  # AUC下降5%
            'drift_detection': 0.05,  # p值小於0.05
            'system_cpu': 80.0,  # CPU使用率80%
            'system_memory': 85.0,  # 記憶體使用率85%
            'prediction_latency': 1.0,  # 預測延遲1秒
            'error_rate': 0.1  # 錯誤率10%
        }
        
        # 配置默認通知渠道
        self._setup_default_notifications()
    
    def add_model(self, model_name: str, model, reference_data: pd.DataFrame = None):
        """添加模型到監控系統"""
        self.models[model_name] = model
        self.model_monitors[model_name] = ModelMonitor(model_name, reference_data)
        logger.info(f"模型 {model_name} 已添加到監控系統")
    
    def start_monitoring(self):
        """啟動實時監控"""
        self.monitoring_active = True
        
        # 啟動系統監控
        self.system_monitor.start_monitoring()
        
        # 啟動預測監控
        self._start_prediction_monitoring()
        
        logger.info("實時監控系統已啟動")
    
    def stop_monitoring(self):
        """停止實時監控"""
        self.monitoring_active = False
        self.system_monitor.stop_monitoring()
        logger.info("實時監控系統已停止")
    
    def _start_prediction_monitoring(self):
        """啟動預測監控"""
        def monitor_predictions():
            while self.monitoring_active:
                try:
                    # 檢查預測隊列
                    if not self.prediction_queue.empty():
                        prediction_data = self.prediction_queue.get_nowait()
                        self._process_prediction_monitoring(prediction_data)
                    
                    # 檢查系統指標
                    self._check_system_alerts()
                    
                    time.sleep(5)  # 每5秒檢查一次
                    
                except Exception as e:
                    logger.error(f"預測監控錯誤: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor_predictions, daemon=True)
        monitor_thread.start()
    
    def log_prediction(self, model_name: str, prediction_data: Dict[str, Any]):
        """記錄預測結果"""
        prediction_data['model_name'] = model_name
        prediction_data['timestamp'] = datetime.now()
        self.prediction_queue.put(prediction_data)
    
    def _process_prediction_monitoring(self, prediction_data: Dict[str, Any]):
        """處理預測監控"""
        model_name = prediction_data['model_name']
        
        if model_name not in self.model_monitors:
            return
        
        monitor = self.model_monitors[model_name]
        
        # 檢查是否有真實標籤用於性能監控
        if 'y_true' in prediction_data and 'y_pred' in prediction_data:
            monitor.log_performance(
                prediction_data['y_true'],
                prediction_data['y_pred'],
                prediction_data.get('y_pred_proba', prediction_data['y_pred']),
                prediction_data.get('prediction_times', [])
            )
        
        # 檢查數據漂移
        if 'current_data' in prediction_data:
            drift_results = monitor.detect_data_drift(prediction_data['current_data'])
            for drift in drift_results:
                if drift.drift_detected:
                    self._trigger_drift_alert(model_name, drift)
    
    def _check_system_alerts(self):
        """檢查系統警報"""
        current_metrics = self.system_monitor.get_current_metrics()
        if not current_metrics:
            return
        
        # CPU使用率警報
        if current_metrics.cpu_usage > self.alert_thresholds['system_cpu']:
            self.alert_manager.trigger_alert(RealTimeAlert(
                timestamp=datetime.now(),
                alert_type='system_cpu_high',
                severity='high',
                model_name='system',
                message=f"CPU使用率過高: {current_metrics.cpu_usage:.1f}%",
                details={'cpu_usage': current_metrics.cpu_usage}
            ))
        
        # 記憶體使用率警報
        if current_metrics.memory_usage > self.alert_thresholds['system_memory']:
            self.alert_manager.trigger_alert(RealTimeAlert(
                timestamp=datetime.now(),
                alert_type='system_memory_high',
                severity='high',
                model_name='system',
                message=f"記憶體使用率過高: {current_metrics.memory_usage:.1f}%",
                details={'memory_usage': current_metrics.memory_usage}
            ))
    
    def _trigger_drift_alert(self, model_name: str, drift_metric: DriftMetrics):
        """觸發漂移警報"""
        severity = 'high' if drift_metric.p_value < 0.01 else 'medium'
        
        self.alert_manager.trigger_alert(RealTimeAlert(
            timestamp=datetime.now(),
            alert_type='data_drift',
            severity=severity,
            model_name=model_name,
            message=f"檢測到特徵 {drift_metric.feature_name} 發生數據漂移",
            details={
                'feature_name': drift_metric.feature_name,
                'drift_score': drift_metric.drift_score,
                'p_value': drift_metric.p_value,
                'drift_type': drift_metric.drift_type
            }
        ))
    
    def _setup_default_notifications(self):
        """設置默認通知配置"""
        # 可以從配置文件讀取
        slack_config = self.config.get('notifications', {}).get('slack', {})
        if slack_config.get('webhook_url'):
            self.alert_manager.configure_notification_channel('slack', slack_config)
        
        webhook_config = self.config.get('notifications', {}).get('webhook', {})
        if webhook_config.get('url'):
            self.alert_manager.configure_notification_channel('webhook', webhook_config)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """獲取監控狀態"""
        active_alerts = self.alert_manager.get_active_alerts()
        system_metrics = self.system_monitor.get_current_metrics()
        
        return {
            'monitoring_active': self.monitoring_active,
            'models_monitored': list(self.model_monitors.keys()),
            'active_alerts_count': len(active_alerts),
            'system_health': self._assess_system_health(system_metrics),
            'last_update': datetime.now().isoformat()
        }
    
    def _assess_system_health(self, metrics: Optional[SystemMetrics]) -> str:
        """評估系統健康狀態"""
        if not metrics:
            return 'unknown'
        
        if (metrics.cpu_usage > 90 or 
            metrics.memory_usage > 90 or 
            metrics.disk_usage > 95):
            return 'critical'
        elif (metrics.cpu_usage > 70 or 
              metrics.memory_usage > 80 or 
              metrics.disk_usage > 85):
            return 'warning'
        else:
            return 'healthy'
    
    def create_dashboard_app(self) -> dash.Dash:
        """創建監控儀表板"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("詐騙檢測實時監控儀表板", className="header"),
            
            # 系統狀態概覽
            html.Div([
                html.H3("系統狀態"),
                html.Div(id="system-status"),
            ], className="section"),
            
            # 警報面板
            html.Div([
                html.H3("活躍警報"),
                html.Div(id="alerts-panel"),
            ], className="section"),
            
            # 系統指標圖表
            html.Div([
                html.H3("系統指標"),
                dcc.Graph(id="system-metrics-chart"),
            ], className="section"),
            
            # 模型性能圖表
            html.Div([
                html.H3("模型性能"),
                dcc.Graph(id="model-performance-chart"),
            ], className="section"),
            
            # 自動刷新
            dcc.Interval(
                id='interval-component',
                interval=10*1000,  # 10秒刷新
                n_intervals=0
            )
        ])
        
        @app.callback(
            [Output('system-status', 'children'),
             Output('alerts-panel', 'children'),
             Output('system-metrics-chart', 'figure'),
             Output('model-performance-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._update_dashboard_components()
        
        return app
    
    def _update_dashboard_components(self):
        """更新儀表板組件"""
        # 系統狀態
        status = self.get_monitoring_status()
        system_status = html.Div([
            html.P(f"監控狀態: {'運行中' if status['monitoring_active'] else '已停止'}"),
            html.P(f"監控模型: {len(status['models_monitored'])}"),
            html.P(f"活躍警報: {status['active_alerts_count']}"),
            html.P(f"系統健康: {status['system_health']}")
        ])
        
        # 警報面板
        active_alerts = self.alert_manager.get_active_alerts()
        alerts_panel = html.Div([
            html.Div([
                html.H5(f"{alert.alert_type} - {alert.severity}"),
                html.P(alert.message),
                html.Small(alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            ], className="alert-item") for alert in active_alerts[-10:]  # 顯示最近10個警報
        ])
        
        # 系統指標圖表
        metrics_history = self.system_monitor.get_metrics_history(hours=1)
        if metrics_history:
            timestamps = [m.timestamp for m in metrics_history]
            cpu_usage = [m.cpu_usage for m in metrics_history]
            memory_usage = [m.memory_usage for m in metrics_history]
            
            system_fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('CPU使用率 (%)', '記憶體使用率 (%)'),
                vertical_spacing=0.1
            )
            
            system_fig.add_trace(
                go.Scatter(x=timestamps, y=cpu_usage, name='CPU', line=dict(color='blue')),
                row=1, col=1
            )
            system_fig.add_trace(
                go.Scatter(x=timestamps, y=memory_usage, name='記憶體', line=dict(color='red')),
                row=2, col=1
            )
            
            system_fig.update_layout(height=400, title_text="系統資源使用率")
        else:
            system_fig = go.Figure()
        
        # 模型性能圖表
        model_fig = go.Figure()
        for model_name, monitor in self.model_monitors.items():
            if monitor.performance_history:
                recent_metrics = monitor.performance_history[-20:]  # 最近20個記錄
                timestamps = [m.timestamp for m in recent_metrics]
                auc_scores = [m.roc_auc for m in recent_metrics]
                
                model_fig.add_trace(go.Scatter(
                    x=timestamps, 
                    y=auc_scores, 
                    name=f"{model_name} AUC",
                    mode='lines+markers'
                ))
        
        model_fig.update_layout(title="模型性能趨勢", yaxis_title="ROC-AUC")
        
        return system_status, alerts_panel, system_fig, model_fig

# 便捷函數
def create_realtime_monitoring_system(models: Dict[str, Any] = None) -> RealTimeMonitoringSystem:
    """創建實時監控系統"""
    return RealTimeMonitoringSystem(models)

def run_monitoring_dashboard(monitoring_system: RealTimeMonitoringSystem, 
                           host: str = '127.0.0.1', 
                           port: int = 8050, 
                           debug: bool = False):
    """運行監控儀表板"""
    app = monitoring_system.create_dashboard_app()
    app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # 示例使用
    monitoring_system = create_realtime_monitoring_system()
    monitoring_system.start_monitoring()
    
    # 啟動儀表板
    print("啟動實時監控儀表板...")
    print("訪問 http://127.0.0.1:8050 查看監控面板")
    run_monitoring_dashboard(monitoring_system)