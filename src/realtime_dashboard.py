"""
實時監控儀表板 - IEEE-CIS 詐騙檢測項目
提供實時數據監控、警報系統和性能追蹤
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json
import logging
import threading
import time
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .config import get_config
from .model_monitoring import ModelMonitor, PerformanceMetrics
from .visualization_engine import VisualizationEngine

logger = logging.getLogger(__name__)

class RealTimeDataBuffer:
    """實時數據緩衝區"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.transaction_buffer = deque(maxlen=max_size)
        self.prediction_buffer = deque(maxlen=max_size)
        self.performance_buffer = deque(maxlen=1000)
        self.alert_buffer = deque(maxlen=500)
        self.system_metrics = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def add_transaction(self, transaction_data: Dict):
        """添加交易數據"""
        with self._lock:
            transaction_data['timestamp'] = datetime.now()
            self.transaction_buffer.append(transaction_data)
    
    def add_prediction(self, prediction_data: Dict):
        """添加預測結果"""
        with self._lock:
            prediction_data['timestamp'] = datetime.now()
            self.prediction_buffer.append(prediction_data)
    
    def add_performance_metric(self, metric: PerformanceMetrics):
        """添加性能指標"""
        with self._lock:
            self.performance_buffer.append(metric)
    
    def add_alert(self, alert: Dict):
        """添加警報"""
        with self._lock:
            alert['timestamp'] = datetime.now()
            self.alert_buffer.append(alert)
    
    def add_system_metric(self, metric: Dict):
        """添加系統指標"""
        with self._lock:
            metric['timestamp'] = datetime.now()
            self.system_metrics.append(metric)
    
    def get_recent_data(self, minutes: int = 60) -> Dict:
        """獲取最近的數據"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            recent_transactions = [
                t for t in self.transaction_buffer 
                if t.get('timestamp', datetime.now()) >= cutoff_time
            ]
            
            recent_predictions = [
                p for p in self.prediction_buffer 
                if p.get('timestamp', datetime.now()) >= cutoff_time
            ]
            
            recent_alerts = [
                a for a in self.alert_buffer 
                if a.get('timestamp', datetime.now()) >= cutoff_time
            ]
            
            recent_metrics = [
                m for m in self.system_metrics 
                if m.get('timestamp', datetime.now()) >= cutoff_time
            ]
        
        return {
            'transactions': recent_transactions,
            'predictions': recent_predictions,
            'alerts': recent_alerts,
            'system_metrics': recent_metrics
        }

class AlertSystem:
    """警報系統"""
    
    def __init__(self, data_buffer: RealTimeDataBuffer):
        self.data_buffer = data_buffer
        self.alert_rules = self._setup_default_rules()
        self.alert_callbacks = []
    
    def _setup_default_rules(self) -> Dict[str, Dict]:
        """設置默認警報規則"""
        return {
            'high_fraud_rate': {
                'condition': lambda data: self._check_fraud_rate(data),
                'threshold': 0.05,  # 5%
                'severity': 'high',
                'message': '詐騙率異常升高'
            },
            'low_model_performance': {
                'condition': lambda data: self._check_model_performance(data),
                'threshold': 0.85,  # AUC < 0.85
                'severity': 'medium',
                'message': '模型性能下降'
            },
            'system_latency': {
                'condition': lambda data: self._check_system_latency(data),
                'threshold': 1.0,  # 1秒
                'severity': 'medium',
                'message': '系統延遲異常'
            },
            'data_quality': {
                'condition': lambda data: self._check_data_quality(data),
                'threshold': 0.95,  # 95%完整性
                'severity': 'low',
                'message': '數據品質下降'
            }
        }
    
    def _check_fraud_rate(self, data: Dict) -> bool:
        """檢查詐騙率"""
        predictions = data.get('predictions', [])
        if len(predictions) < 100:  # 需要足夠的樣本
            return False
        
        recent_predictions = predictions[-100:]
        fraud_count = sum(1 for p in recent_predictions if p.get('prediction', 0) == 1)
        fraud_rate = fraud_count / len(recent_predictions)
        
        return fraud_rate > self.alert_rules['high_fraud_rate']['threshold']
    
    def _check_model_performance(self, data: Dict) -> bool:
        """檢查模型性能"""
        # 這裡需要實際的性能數據
        return False  # 暫時返回False
    
    def _check_system_latency(self, data: Dict) -> bool:
        """檢查系統延遲"""
        metrics = data.get('system_metrics', [])
        if not metrics:
            return False
        
        recent_metrics = metrics[-10:]
        avg_latency = np.mean([m.get('response_time', 0) for m in recent_metrics])
        
        return avg_latency > self.alert_rules['system_latency']['threshold']
    
    def _check_data_quality(self, data: Dict) -> bool:
        """檢查數據品質"""
        transactions = data.get('transactions', [])
        if not transactions:
            return False
        
        recent_transactions = transactions[-100:]
        missing_values = sum(
            1 for t in recent_transactions 
            if any(v is None or v == '' for v in t.values())
        )
        
        completeness = 1 - (missing_values / len(recent_transactions))
        return completeness < self.alert_rules['data_quality']['threshold']
    
    def check_alerts(self) -> List[Dict]:
        """檢查並生成警報"""
        recent_data = self.data_buffer.get_recent_data(minutes=10)
        alerts = []
        
        for rule_name, rule_config in self.alert_rules.items():
            try:
                if rule_config['condition'](recent_data):
                    alert = {
                        'id': f"{rule_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'rule_name': rule_name,
                        'severity': rule_config['severity'],
                        'message': rule_config['message'],
                        'timestamp': datetime.now(),
                        'status': 'active'
                    }
                    alerts.append(alert)
                    self.data_buffer.add_alert(alert)
                    
                    # 執行回調函數
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"警報回調執行失敗: {e}")
            
            except Exception as e:
                logger.error(f"警報規則 {rule_name} 檢查失敗: {e}")
        
        return alerts
    
    def add_alert_callback(self, callback: Callable[[Dict], None]):
        """添加警報回調函數"""
        self.alert_callbacks.append(callback)

class DashboardServer:
    """儀表板服務器"""
    
    def __init__(self, data_buffer: RealTimeDataBuffer, port: int = 8050):
        self.data_buffer = data_buffer
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.viz_engine = VisualizationEngine()
        self.alert_system = AlertSystem(data_buffer)
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """設置儀表板布局"""
        self.app.layout = dbc.Container([
            # 標題行
            dbc.Row([
                dbc.Col([
                    html.H1("詐騙檢測實時監控儀表板", 
                           className="text-center mb-4",
                           style={'color': '#2c3e50'})
                ])
            ]),
            
            # 實時指標卡片
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("實時交易量", className="card-title"),
                            html.H2(id="transaction-count", children="0", 
                                   className="text-primary")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("詐騙檢測率", className="card-title"),
                            html.H2(id="fraud-rate", children="0.0%", 
                                   className="text-danger")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("平均響應時間", className="card-title"),
                            html.H2(id="response-time", children="0ms", 
                                   className="text-info")
                        ])
                    ])
                ], width=3),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("活躍警報", className="card-title"),
                            html.H2(id="alert-count", children="0", 
                                   className="text-warning")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # 圖表行
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("實時交易趨勢"),
                        dbc.CardBody([
                            dcc.Graph(id="transaction-trend")
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("風險分數分佈"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-distribution")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("模型性能監控"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-metrics")
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("實時警報"),
                        dbc.CardBody([
                            html.Div(id="alert-list")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # 控制面板
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("控制面板"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("更新間隔 (秒)"),
                                    dcc.Slider(
                                        id="update-interval",
                                        min=1, max=60, value=5,
                                        marks={i: str(i) for i in [1, 5, 10, 30, 60]}
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    dbc.Label("時間範圍 (分鐘)"),
                                    dcc.Dropdown(
                                        id="time-range",
                                        options=[
                                            {'label': '5分鐘', 'value': 5},
                                            {'label': '15分鐘', 'value': 15},
                                            {'label': '30分鐘', 'value': 30},
                                            {'label': '1小時', 'value': 60},
                                            {'label': '2小時', 'value': 120}
                                        ],
                                        value=30
                                    )
                                ], width=4),
                                
                                dbc.Col([
                                    dbc.Label("警報級別過濾"),
                                    dcc.Dropdown(
                                        id="alert-filter",
                                        options=[
                                            {'label': '全部', 'value': 'all'},
                                            {'label': '高', 'value': 'high'},
                                            {'label': '中', 'value': 'medium'},
                                            {'label': '低', 'value': 'low'}
                                        ],
                                        value='all',
                                        multi=True
                                    )
                                ], width=4)
                            ])
                        ])
                    ])
                ])
            ]),
            
            # 自動更新組件
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # 5秒更新一次
                n_intervals=0
            ),
            
            # 存儲組件
            dcc.Store(id='dashboard-data')
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """設置回調函數"""
        
        @self.app.callback(
            [Output('transaction-count', 'children'),
             Output('fraud-rate', 'children'),
             Output('response-time', 'children'),
             Output('alert-count', 'children'),
             Output('dashboard-data', 'data')],
            [Input('interval-component', 'n_intervals'),
             Input('time-range', 'value')]
        )
        def update_metrics(n, time_range):
            """更新實時指標"""
            try:
                recent_data = self.data_buffer.get_recent_data(minutes=time_range or 30)
                
                # 交易量
                transaction_count = len(recent_data['transactions'])
                
                # 詐騙率
                predictions = recent_data['predictions']
                if predictions:
                    fraud_count = sum(1 for p in predictions if p.get('prediction', 0) == 1)
                    fraud_rate = f"{(fraud_count / len(predictions) * 100):.2f}%"
                else:
                    fraud_rate = "0.00%"
                
                # 響應時間
                system_metrics = recent_data['system_metrics']
                if system_metrics:
                    avg_response = np.mean([m.get('response_time', 0) for m in system_metrics])
                    response_time = f"{avg_response*1000:.0f}ms"
                else:
                    response_time = "0ms"
                
                # 警報數量
                active_alerts = [a for a in recent_data['alerts'] if a.get('status') == 'active']
                alert_count = len(active_alerts)
                
                return transaction_count, fraud_rate, response_time, alert_count, recent_data
                
            except Exception as e:
                logger.error(f"更新指標失敗: {e}")
                return "0", "0.00%", "0ms", "0", {}
        
        @self.app.callback(
            Output('transaction-trend', 'figure'),
            [Input('dashboard-data', 'data')]
        )
        def update_transaction_trend(data):
            """更新交易趨勢圖"""
            try:
                if not data or not data.get('transactions'):
                    return go.Figure()
                
                transactions = data['transactions']
                df = pd.DataFrame(transactions)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # 按分鐘聚合
                    df_grouped = df.set_index('timestamp').resample('1T').size()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_grouped.index,
                        y=df_grouped.values,
                        mode='lines+markers',
                        name='交易量',
                        line=dict(color='#3498db', width=2)
                    ))
                    
                    fig.update_layout(
                        title="交易量趨勢",
                        xaxis_title="時間",
                        yaxis_title="交易數量",
                        height=300
                    )
                    
                    return fig
                
                return go.Figure()
                
            except Exception as e:
                logger.error(f"更新交易趨勢圖失敗: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('risk-distribution', 'figure'),
            [Input('dashboard-data', 'data')]
        )
        def update_risk_distribution(data):
            """更新風險分佈圖"""
            try:
                if not data or not data.get('predictions'):
                    return go.Figure()
                
                predictions = data['predictions']
                risk_scores = [p.get('risk_score', 0) for p in predictions if 'risk_score' in p]
                
                if not risk_scores:
                    return go.Figure()
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=risk_scores,
                    nbinsx=30,
                    name='風險分數',
                    marker_color='#e74c3c',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    title="風險分數分佈",
                    xaxis_title="風險分數",
                    yaxis_title="頻次",
                    height=300
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"更新風險分佈圖失敗: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('alert-list', 'children'),
            [Input('dashboard-data', 'data'),
             Input('alert-filter', 'value')]
        )
        def update_alert_list(data, alert_filter):
            """更新警報列表"""
            try:
                if not data or not data.get('alerts'):
                    return html.P("暫無警報", className="text-muted")
                
                alerts = data['alerts']
                
                # 過濾警報
                if alert_filter and alert_filter != 'all':
                    if isinstance(alert_filter, list):
                        alerts = [a for a in alerts if a.get('severity') in alert_filter]
                    else:
                        alerts = [a for a in alerts if a.get('severity') == alert_filter]
                
                # 按時間排序
                alerts = sorted(alerts, key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
                
                alert_components = []
                for alert in alerts[:10]:  # 只顯示最近10個
                    severity_color = {
                        'high': 'danger',
                        'medium': 'warning', 
                        'low': 'info'
                    }.get(alert.get('severity', 'info'), 'info')
                    
                    alert_time = alert.get('timestamp', datetime.now())
                    if isinstance(alert_time, str):
                        alert_time = datetime.fromisoformat(alert_time)
                    
                    alert_components.append(
                        dbc.Alert([
                            html.Strong(alert.get('message', '未知警報')),
                            html.Br(),
                            html.Small(f"時間: {alert_time.strftime('%H:%M:%S')}")
                        ], color=severity_color, className="mb-2")
                    )
                
                return alert_components if alert_components else html.P("暫無警報", className="text-muted")
                
            except Exception as e:
                logger.error(f"更新警報列表失敗: {e}")
                return html.P("警報載入失敗", className="text-danger")
    
    def run_server(self, debug: bool = False, host: str = '127.0.0.1'):
        """運行儀表板服務器"""
        logger.info(f"啟動實時監控儀表板 - http://{host}:{self.port}")
        self.app.run_server(debug=debug, host=host, port=self.port)

class RealTimeMonitoringSystem:
    """實時監控系統主類"""
    
    def __init__(self, config_manager=None):
        self.config = config_manager or get_config()
        self.data_buffer = RealTimeDataBuffer()
        self.alert_system = AlertSystem(self.data_buffer)
        self.dashboard_server = None
        self.monitoring_thread = None
        self.is_running = False
        
        # 設置警報回調
        self.alert_system.add_alert_callback(self._handle_alert)
    
    def _handle_alert(self, alert: Dict):
        """處理警報"""
        logger.warning(f"觸發警報: {alert['message']} (嚴重程度: {alert['severity']})")
        
        # 這裡可以添加其他警報處理邏輯，如：
        # - 發送郵件
        # - 發送Slack通知
        # - 記錄到數據庫
        # - 觸發自動回應
    
    def start_monitoring(self, dashboard_port: int = 8050):
        """啟動監控系統"""
        if self.is_running:
            logger.warning("監控系統已在運行中")
            return
        
        self.is_running = True
        
        # 啟動警報檢查線程
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # 啟動儀表板服務器
        self.dashboard_server = DashboardServer(self.data_buffer, port=dashboard_port)
        
        logger.info("實時監控系統已啟動")
    
    def stop_monitoring(self):
        """停止監控系統"""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("實時監控系統已停止")
    
    def _monitoring_loop(self):
        """監控循環"""
        while self.is_running:
            try:
                # 檢查警報
                self.alert_system.check_alerts()
                
                # 添加系統指標
                system_metric = {
                    'cpu_usage': self._get_cpu_usage(),
                    'memory_usage': self._get_memory_usage(),
                    'response_time': np.random.uniform(0.01, 0.1)  # 模擬響應時間
                }
                self.data_buffer.add_system_metric(system_metric)
                
                time.sleep(10)  # 每10秒檢查一次
                
            except Exception as e:
                logger.error(f"監控循環出錯: {e}")
    
    def _get_cpu_usage(self) -> float:
        """獲取CPU使用率"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return np.random.uniform(10, 80)  # 模擬數據
    
    def _get_memory_usage(self) -> float:
        """獲取內存使用率"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return np.random.uniform(30, 90)  # 模擬數據
    
    def add_transaction(self, transaction_data: Dict):
        """添加交易數據"""
        self.data_buffer.add_transaction(transaction_data)
    
    def add_prediction(self, prediction_data: Dict):
        """添加預測結果"""
        self.data_buffer.add_prediction(prediction_data)
    
    def run_dashboard(self, debug: bool = False, host: str = '127.0.0.1', port: int = 8050):
        """運行儀表板"""
        if not self.dashboard_server:
            self.dashboard_server = DashboardServer(self.data_buffer, port=port)
        
        self.dashboard_server.run_server(debug=debug, host=host)

# 便捷函數
def create_realtime_monitor(config_manager=None) -> RealTimeMonitoringSystem:
    """創建實時監控系統"""
    return RealTimeMonitoringSystem(config_manager)

def simulate_transaction_data(monitor: RealTimeMonitoringSystem, 
                            duration_minutes: int = 60,
                            transactions_per_minute: int = 100):
    """模擬交易數據（用於測試）"""
    import random
    import time
    
    logger.info(f"開始模擬交易數據，持續 {duration_minutes} 分鐘")
    
    end_time = time.time() + (duration_minutes * 60)
    
    while time.time() < end_time:
        for _ in range(transactions_per_minute):
            # 模擬交易數據
            transaction = {
                'transaction_id': f"TXN_{random.randint(1000000, 9999999)}",
                'amount': random.uniform(10, 5000),
                'user_id': f"USER_{random.randint(1000, 50000)}",
                'merchant_id': f"MERCH_{random.randint(100, 1000)}"
            }
            
            # 模擬預測結果
            risk_score = random.uniform(0, 1)
            prediction = {
                'transaction_id': transaction['transaction_id'],
                'risk_score': risk_score,
                'prediction': 1 if risk_score > 0.5 else 0,
                'confidence': random.uniform(0.6, 0.99)
            }
            
            monitor.add_transaction(transaction)
            monitor.add_prediction(prediction)
        
        time.sleep(60)  # 等待1分鐘

if __name__ == "__main__":
    # 測試代碼
    monitor = create_realtime_monitor()
    
    # 在後台線程中模擬數據
    import threading
    sim_thread = threading.Thread(
        target=simulate_transaction_data, 
        args=(monitor, 60, 50),  # 60分鐘，每分鐘50筆交易
        daemon=True
    )
    sim_thread.start()
    
    # 啟動監控和儀表板
    monitor.start_monitoring()
    monitor.run_dashboard(debug=True)