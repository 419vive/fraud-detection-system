"""
使用量監控和實時計費系統 - IEEE-CIS 詐騙檢測服務
提供實時使用量追蹤、計費計算和成本分析功能
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import logging
import threading
from collections import defaultdict, deque
import statistics
from decimal import Decimal
import redis
import schedule

from .billing_system import BillingEngine, UsageType, UsageRecord
from .api_key_management import APIKeyManager, APIKey

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指標類型"""
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    COST = "cost"
    THROUGHPUT = "throughput"

@dataclass
class UsageMetrics:
    """使用量指標"""
    timestamp: datetime
    customer_id: str
    api_key_id: str
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['metric_type'] = self.metric_type.value
        return result

@dataclass
class RealTimeBill:
    """實時計費記錄"""
    customer_id: str
    current_period_start: datetime
    current_period_end: datetime
    usage_summary: Dict[str, int]
    cost_breakdown: Dict[str, Decimal]
    total_cost: Decimal
    estimated_monthly_cost: Decimal
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['current_period_start'] = self.current_period_start.isoformat()
        result['current_period_end'] = self.current_period_end.isoformat()
        result['cost_breakdown'] = {k: float(v) for k, v in self.cost_breakdown.items()}
        result['total_cost'] = float(self.total_cost)
        result['estimated_monthly_cost'] = float(self.estimated_monthly_cost)
        result['last_updated'] = self.last_updated.isoformat()
        return result

@dataclass
class UsageAlert:
    """使用量警報"""
    id: str
    customer_id: str
    alert_type: str
    threshold: float
    current_value: float
    message: str
    triggered_at: datetime
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['triggered_at'] = self.triggered_at.isoformat()
        return result

class UsageBuffer:
    """使用量緩衝區"""
    
    def __init__(self, max_size: int = 10000, flush_interval: int = 60):
        self.max_size = max_size
        self.flush_interval = flush_interval
        self.buffer: deque = deque(maxlen=max_size)
        self.last_flush = time.time()
        self.lock = threading.Lock()
    
    def add_usage(self, usage_record: Dict[str, Any]):
        """添加使用記錄到緩衝區"""
        with self.lock:
            self.buffer.append(usage_record)
            
            # 檢查是否需要刷新
            if (len(self.buffer) >= self.max_size or 
                time.time() - self.last_flush >= self.flush_interval):
                return list(self.buffer), True
            
            return [], False
    
    def flush(self) -> List[Dict[str, Any]]:
        """強制刷新緩衝區"""
        with self.lock:
            records = list(self.buffer)
            self.buffer.clear()
            self.last_flush = time.time()
            return records

class RealTimeUsageTracker:
    """實時使用量追蹤器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.usage_buffer = UsageBuffer()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.metrics_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    def track_request_start(self, customer_id: str, api_key_id: str, 
                          endpoint: str, request_id: str) -> str:
        """追蹤請求開始"""
        session_data = {
            'customer_id': customer_id,
            'api_key_id': api_key_id,
            'endpoint': endpoint,
            'start_time': time.time(),
            'request_id': request_id
        }
        
        self.active_sessions[request_id] = session_data
        
        # 更新Redis中的實時計數
        self._increment_redis_counter(f"requests:{customer_id}:current", 1)
        self._increment_redis_counter(f"requests:{api_key_id}:current", 1)
        
        return request_id
    
    def track_request_end(self, request_id: str, response_status: int,
                         response_size: int = 0, error_details: str = None) -> Optional[Dict[str, Any]]:
        """追蹤請求結束"""
        if request_id not in self.active_sessions:
            logger.warning(f"找不到請求會話: {request_id}")
            return None
        
        session = self.active_sessions.pop(request_id)
        end_time = time.time()
        response_time = end_time - session['start_time']
        
        # 創建使用記錄
        usage_record = {
            'customer_id': session['customer_id'],
            'api_key_id': session['api_key_id'],
            'endpoint': session['endpoint'],
            'response_time': response_time,
            'response_status': response_status,
            'response_size': response_size,
            'timestamp': datetime.fromtimestamp(end_time),
            'error_details': error_details
        }
        
        # 添加到緩衝區
        pending_records, should_flush = self.usage_buffer.add_usage(usage_record)
        
        # 更新實時指標
        self._update_real_time_metrics(usage_record)
        
        # 如果需要刷新，返回待處理記錄
        if should_flush:
            return {'usage_record': usage_record, 'pending_records': pending_records}
        
        return {'usage_record': usage_record}
    
    def _increment_redis_counter(self, key: str, amount: int, ttl: int = 3600):
        """遞增Redis計數器"""
        try:
            pipeline = self.redis_client.pipeline()
            pipeline.incrby(key, amount)
            pipeline.expire(key, ttl)
            pipeline.execute()
        except Exception as e:
            logger.error(f"Redis操作失敗: {e}")
    
    def _update_real_time_metrics(self, usage_record: Dict[str, Any]):
        """更新實時指標"""
        customer_id = usage_record['customer_id']
        timestamp = time.time()
        
        # 響應時間指標
        self.metrics_cache[f"{customer_id}:response_time"].append({
            'timestamp': timestamp,
            'value': usage_record['response_time']
        })
        
        # 錯誤率指標
        is_error = usage_record['response_status'] >= 400
        self.metrics_cache[f"{customer_id}:error_rate"].append({
            'timestamp': timestamp,
            'value': 1 if is_error else 0
        })
        
        # 吞吐量指標（每分鐘請求數）
        self.metrics_cache[f"{customer_id}:throughput"].append({
            'timestamp': timestamp,
            'value': 1
        })
    
    def get_real_time_metrics(self, customer_id: str, 
                            metric_type: MetricType,
                            time_window: int = 300) -> Dict[str, Any]:
        """獲取實時指標"""
        cache_key = f"{customer_id}:{metric_type.value}"
        metrics = self.metrics_cache.get(cache_key, deque())
        
        # 過濾時間窗口內的數據
        current_time = time.time()
        filtered_metrics = [
            m for m in metrics 
            if current_time - m['timestamp'] <= time_window
        ]
        
        if not filtered_metrics:
            return {'average': 0, 'count': 0, 'latest': 0}
        
        values = [m['value'] for m in filtered_metrics]
        
        result = {
            'average': statistics.mean(values),
            'count': len(values),
            'latest': values[-1] if values else 0,
            'time_window_seconds': time_window
        }
        
        # 根據指標類型添加特定統計
        if metric_type == MetricType.RESPONSE_TIME:
            result.update({
                'min': min(values),
                'max': max(values),
                'p50': statistics.median(values),
                'p95': statistics.quantiles(values, n=20)[18] if len(values) > 20 else max(values)
            })
        elif metric_type == MetricType.ERROR_RATE:
            result['error_percentage'] = (sum(values) / len(values)) * 100
        elif metric_type == MetricType.THROUGHPUT:
            # 計算每分鐘請求數
            result['requests_per_minute'] = (len(values) / time_window) * 60
        
        return result
    
    def get_active_sessions_count(self) -> int:
        """獲取活躍會話數量"""
        return len(self.active_sessions)
    
    def cleanup_stale_sessions(self, max_age: int = 3600):
        """清理過期會話"""
        current_time = time.time()
        stale_sessions = [
            request_id for request_id, session in self.active_sessions.items()
            if current_time - session['start_time'] > max_age
        ]
        
        for request_id in stale_sessions:
            logger.warning(f"清理過期會話: {request_id}")
            self.active_sessions.pop(request_id, None)
        
        return len(stale_sessions)

class RealTimeBillingEngine:
    """實時計費引擎"""
    
    def __init__(self, billing_engine: BillingEngine, 
                 api_key_manager: APIKeyManager,
                 update_interval: int = 60):
        self.billing_engine = billing_engine
        self.api_key_manager = api_key_manager
        self.update_interval = update_interval
        self.real_time_bills: Dict[str, RealTimeBill] = {}
        self.usage_alerts: List[UsageAlert] = []
        self.running = False
        self.update_thread = None
    
    def start_real_time_billing(self):
        """啟動實時計費"""
        self.running = True
        self.update_thread = threading.Thread(target=self._billing_update_loop, daemon=True)
        self.update_thread.start()
        logger.info("實時計費引擎已啟動")
    
    def stop_real_time_billing(self):
        """停止實時計費"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
        logger.info("實時計費引擎已停止")
    
    def _billing_update_loop(self):
        """計費更新循環"""
        while self.running:
            try:
                self._update_all_customer_bills()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"計費更新錯誤: {e}")
                time.sleep(self.update_interval)
    
    def _update_all_customer_bills(self):
        """更新所有客戶的計費"""
        # 獲取所有活躍客戶
        active_customers = self._get_active_customers()
        
        for customer_id in active_customers:
            try:
                self._update_customer_bill(customer_id)
            except Exception as e:
                logger.error(f"更新客戶 {customer_id} 計費失敗: {e}")
    
    def _get_active_customers(self) -> List[str]:
        """獲取活躍客戶列表"""
        # 從資料庫獲取最近有活動的客戶
        with sqlite3.connect(self.billing_engine.database.db_path) as conn:
            cursor = conn.cursor()
            
            # 獲取最近24小時有使用記錄的客戶
            yesterday = datetime.now() - timedelta(days=1)
            cursor.execute('''
                SELECT DISTINCT customer_id FROM usage_records 
                WHERE timestamp > ?
            ''', (yesterday.isoformat(),))
            
            return [row[0] for row in cursor.fetchall()]
    
    def _update_customer_bill(self, customer_id: str):
        """更新客戶實時帳單"""
        # 計算當前計費期間
        now = datetime.now()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            period_end = now.replace(year=now.year + 1, month=1, day=1) - timedelta(seconds=1)
        else:
            period_end = now.replace(month=now.month + 1, day=1) - timedelta(seconds=1)
        
        # 獲取使用記錄
        usage_records = self.billing_engine.database.get_usage_for_period(
            customer_id, period_start, now
        )
        
        # 計算使用摘要
        usage_summary = defaultdict(int)
        for record in usage_records:
            usage_summary[record.usage_type.value] += record.quantity
        
        # 計算成本分解
        cost_breakdown = {}
        total_cost = Decimal('0.00')
        
        subscription = self.billing_engine.database.get_customer_subscription(customer_id)
        if subscription:
            # 訂閱客戶
            plan_type = subscription['plan_type']
            prediction_usage = usage_summary.get('prediction', 0)
            
            subscription_cost, overage_cost = self.billing_engine.pricing_engine.calculate_subscription_cost(
                plan_type, prediction_usage
            )
            
            cost_breakdown['subscription'] = subscription_cost
            cost_breakdown['overage'] = overage_cost
            total_cost = subscription_cost + overage_cost
        else:
            # 按需付費客戶
            for usage_type_str, quantity in usage_summary.items():
                usage_type = UsageType(usage_type_str)
                cost = self.billing_engine.pricing_engine.calculate_payg_cost(usage_type, quantity)
                cost_breakdown[usage_type_str] = cost
                total_cost += cost
        
        # 估算月度成本
        days_in_period = (now - period_start).days + 1
        days_in_month = (period_end - period_start).days + 1
        estimated_monthly_cost = total_cost * (days_in_month / days_in_period)
        
        # 創建實時帳單
        real_time_bill = RealTimeBill(
            customer_id=customer_id,
            current_period_start=period_start,
            current_period_end=period_end,
            usage_summary=dict(usage_summary),
            cost_breakdown=cost_breakdown,
            total_cost=total_cost,
            estimated_monthly_cost=estimated_monthly_cost,
            last_updated=now
        )
        
        self.real_time_bills[customer_id] = real_time_bill
        
        # 檢查使用警報
        self._check_usage_alerts(customer_id, real_time_bill)
    
    def _check_usage_alerts(self, customer_id: str, bill: RealTimeBill):
        """檢查使用警報"""
        # 成本警報閾值（根據計劃類型設定）
        cost_thresholds = {
            'free': 10.0,
            'basic': 150.0,
            'professional': 450.0,
            'enterprise': 1500.0
        }
        
        subscription = self.billing_engine.database.get_customer_subscription(customer_id)
        plan_type = subscription['plan_type'].value if subscription else 'free'
        threshold = cost_thresholds.get(plan_type, 100.0)
        
        current_cost = float(bill.estimated_monthly_cost)
        
        # 檢查是否超過80%的預算
        if current_cost > threshold * 0.8:
            alert = UsageAlert(
                id=f"cost_alert_{customer_id}_{int(time.time())}",
                customer_id=customer_id,
                alert_type="high_cost",
                threshold=threshold * 0.8,
                current_value=current_cost,
                message=f"預估月度成本 ${current_cost:.2f} 已超過預算的80%",
                triggered_at=datetime.now()
            )
            
            self.usage_alerts.append(alert)
            logger.warning(f"成本警報觸發 - 客戶: {customer_id}, 成本: ${current_cost:.2f}")
    
    def get_customer_real_time_bill(self, customer_id: str) -> Optional[RealTimeBill]:
        """獲取客戶實時帳單"""
        return self.real_time_bills.get(customer_id)
    
    def get_all_real_time_bills(self) -> Dict[str, RealTimeBill]:
        """獲取所有實時帳單"""
        return self.real_time_bills.copy()
    
    def get_usage_alerts(self, customer_id: str = None, 
                        resolved: bool = None) -> List[UsageAlert]:
        """獲取使用警報"""
        alerts = self.usage_alerts
        
        if customer_id:
            alerts = [a for a in alerts if a.customer_id == customer_id]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return alerts

class UsageMonitoringSystem:
    """使用量監控系統"""
    
    def __init__(self, billing_engine: BillingEngine, 
                 api_key_manager: APIKeyManager,
                 redis_client=None):
        self.billing_engine = billing_engine
        self.api_key_manager = api_key_manager
        self.usage_tracker = RealTimeUsageTracker(redis_client)
        self.billing_engine_rt = RealTimeBillingEngine(billing_engine, api_key_manager)
        self.usage_processor_thread = None
        self.running = False
    
    def start_monitoring(self):
        """啟動監控系統"""
        self.running = True
        self.billing_engine_rt.start_real_time_billing()
        
        # 啟動使用記錄處理線程
        self.usage_processor_thread = threading.Thread(
            target=self._process_usage_records, daemon=True
        )
        self.usage_processor_thread.start()
        
        # 設定定期任務
        schedule.every(5).minutes.do(self._cleanup_stale_sessions)
        schedule.every(1).hours.do(self._generate_usage_reports)
        
        logger.info("使用量監控系統已啟動")
    
    def stop_monitoring(self):
        """停止監控系統"""
        self.running = False
        self.billing_engine_rt.stop_real_time_billing()
        
        if self.usage_processor_thread:
            self.usage_processor_thread.join()
        
        logger.info("使用量監控系統已停止")
    
    def track_api_call(self, customer_id: str, api_key_id: str, 
                      endpoint: str, usage_type: UsageType,
                      quantity: int = 1, metadata: Dict[str, Any] = None) -> str:
        """追蹤API調用"""
        request_id = f"req_{int(time.time() * 1000)}_{customer_id}"
        
        # 追蹤請求開始
        self.usage_tracker.track_request_start(
            customer_id, api_key_id, endpoint, request_id
        )
        
        # 記錄計費使用量
        usage_record = self.billing_engine.record_api_usage(
            customer_id, api_key_id, usage_type, quantity, metadata
        )
        
        return request_id
    
    def complete_api_call(self, request_id: str, response_status: int,
                         response_size: int = 0, error_details: str = None):
        """完成API調用追蹤"""
        result = self.usage_tracker.track_request_end(
            request_id, response_status, response_size, error_details
        )
        
        if result and 'pending_records' in result:
            # 處理待處理的記錄
            self._batch_process_usage_records(result['pending_records'])
    
    def _process_usage_records(self):
        """處理使用記錄線程"""
        while self.running:
            try:
                # 定期刷新緩衝區
                time.sleep(60)  # 每分鐘檢查一次
                pending_records = self.usage_tracker.usage_buffer.flush()
                
                if pending_records:
                    self._batch_process_usage_records(pending_records)
                    
            except Exception as e:
                logger.error(f"處理使用記錄錯誤: {e}")
    
    def _batch_process_usage_records(self, records: List[Dict[str, Any]]):
        """批量處理使用記錄"""
        logger.info(f"批量處理 {len(records)} 條使用記錄")
        
        # 這裡可以添加批量插入到資料庫的邏輯
        # 或者發送到消息隊列進行異步處理
        
        for record in records:
            # 更新API密鑰使用統計
            self.api_key_manager.log_api_usage(
                api_key=None,  # 需要從record中獲取
                endpoint=record['endpoint'],
                method='POST',  # 假設
                ip_address='0.0.0.0',  # 從record中獲取
                user_agent='',
                response_status=record['response_status'],
                response_time_ms=record['response_time'] * 1000
            )
    
    def _cleanup_stale_sessions(self):
        """清理過期會話"""
        cleaned = self.usage_tracker.cleanup_stale_sessions()
        if cleaned > 0:
            logger.info(f"清理了 {cleaned} 個過期會話")
    
    def _generate_usage_reports(self):
        """生成使用報告"""
        logger.info("生成使用報告...")
        # 這裡可以添加生成各種報告的邏輯
    
    def get_customer_usage_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """獲取客戶使用儀表板數據"""
        # 實時帳單
        real_time_bill = self.billing_engine_rt.get_customer_real_time_bill(customer_id)
        
        # 實時指標
        metrics = {}
        for metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.THROUGHPUT]:
            metrics[metric_type.value] = self.usage_tracker.get_real_time_metrics(
                customer_id, metric_type
            )
        
        # 警報
        alerts = self.billing_engine_rt.get_usage_alerts(customer_id, resolved=False)
        
        # API密鑰統計
        api_keys = self.api_key_manager.get_customer_keys(customer_id)
        
        return {
            'real_time_bill': real_time_bill.to_dict() if real_time_bill else None,
            'metrics': metrics,
            'alerts': [alert.to_dict() for alert in alerts],
            'api_keys_count': len(api_keys),
            'active_sessions': self.usage_tracker.get_active_sessions_count(),
            'last_updated': datetime.now().isoformat()
        }

# 便捷函數
def create_usage_monitoring_system(billing_engine: BillingEngine,
                                 api_key_manager: APIKeyManager,
                                 redis_client=None) -> UsageMonitoringSystem:
    """創建使用量監控系統"""
    return UsageMonitoringSystem(billing_engine, api_key_manager, redis_client)

if __name__ == "__main__":
    # 示例使用
    from .billing_system import create_billing_engine
    from .api_key_management import create_api_key_manager
    
    billing_engine = create_billing_engine()
    api_key_manager = create_api_key_manager()
    
    monitoring_system = create_usage_monitoring_system(
        billing_engine, api_key_manager
    )
    
    # 啟動監控
    monitoring_system.start_monitoring()
    
    print("使用量監控系統已啟動...")
    
    # 模擬API調用
    request_id = monitoring_system.track_api_call(
        customer_id="test_customer",
        api_key_id="test_key",
        endpoint="/predict",
        usage_type=UsageType.PREDICTION
    )
    
    # 完成調用
    monitoring_system.complete_api_call(request_id, 200, 1024)
    
    print(f"API調用已追蹤: {request_id}")
    
    # 模擬運行一段時間
    import time
    time.sleep(5)
    
    # 停止監控
    monitoring_system.stop_monitoring()