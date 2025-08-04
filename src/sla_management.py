"""
SLA管理系統 - IEEE-CIS 詐騙檢測服務
提供99.9%正常運行時間保證和服務等級協議管理
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import logging
import threading
import statistics
from collections import deque, defaultdict
import requests
import psutil
import subprocess

logger = logging.getLogger(__name__)

class SLAMetric(Enum):
    """SLA指標類型"""
    UPTIME = "uptime"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"

class SLAStatus(Enum):
    """SLA狀態"""
    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    SUSPENDED = "suspended"

class IncidentSeverity(Enum):
    """事件嚴重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SLATarget:
    """SLA目標"""
    metric: SLAMetric
    target_value: float
    measurement_period: timedelta
    threshold_warning: float  # 警告閾值
    threshold_breach: float   # 違約閾值
    unit: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['metric'] = self.metric.value
        result['measurement_period'] = self.measurement_period.total_seconds()
        return result

@dataclass
class SLAMeasurement:
    """SLA測量結果"""
    timestamp: datetime
    metric: SLAMetric
    measured_value: float
    target_value: float
    status: SLAStatus
    measurement_period: timedelta
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['metric'] = self.metric.value
        result['status'] = self.status.value
        result['measurement_period'] = self.measurement_period.total_seconds()
        return result

@dataclass
class Incident:
    """服務事件"""
    id: str
    title: str
    description: str
    severity: IncidentSeverity
    started_at: datetime
    resolved_at: Optional[datetime]
    affected_services: List[str]
    root_cause: Optional[str]
    resolution_steps: List[str]
    impact_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['severity'] = self.severity.value
        result['started_at'] = self.started_at.isoformat()
        result['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return result
    
    @property
    def duration(self) -> Optional[timedelta]:
        """事件持續時間"""
        if self.resolved_at:
            return self.resolved_at - self.started_at
        return datetime.now() - self.started_at
    
    @property
    def is_resolved(self) -> bool:
        """事件是否已解決"""
        return self.resolved_at is not None

@dataclass
class SLAReport:
    """SLA報告"""
    period_start: datetime
    period_end: datetime
    customer_id: Optional[str]
    measurements: List[SLAMeasurement]
    incidents: List[Incident]
    overall_compliance: float
    downtime_minutes: float
    credits_owed: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['period_start'] = self.period_start.isoformat()
        result['period_end'] = self.period_end.isoformat()
        result['measurements'] = [m.to_dict() for m in self.measurements]
        result['incidents'] = [i.to_dict() for i in self.incidents]
        return result

class ServiceHealthChecker:
    """服務健康檢查器"""
    
    def __init__(self):
        self.health_endpoints = [
            "http://localhost:8000/health",
            "http://localhost:8001/predict",
            "http://localhost:8002/monitor"
        ]
        self.check_interval = 30  # 秒
        self.timeout = 10  # 秒
        self.health_history = deque(maxlen=1000)
        self.running = False
        self.check_thread = None
    
    def start_health_checks(self):
        """啟動健康檢查"""
        self.running = True
        self.check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_thread.start()
        logger.info("服務健康檢查已啟動")
    
    def stop_health_checks(self):
        """停止健康檢查"""
        self.running = False
        if self.check_thread:
            self.check_thread.join()
        logger.info("服務健康檢查已停止")
    
    def _health_check_loop(self):
        """健康檢查循環"""
        while self.running:
            try:
                health_status = self._perform_health_check()
                self.health_history.append(health_status)
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"健康檢查錯誤: {e}")
                time.sleep(self.check_interval)
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """執行健康檢查"""
        timestamp = datetime.now()
        endpoint_results = {}
        overall_healthy = True
        response_times = []
        
        for endpoint in self.health_endpoints:
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=self.timeout)
                response_time = (time.time() - start_time) * 1000  # 毫秒
                
                is_healthy = response.status_code == 200
                endpoint_results[endpoint] = {
                    'healthy': is_healthy,
                    'response_time_ms': response_time,
                    'status_code': response.status_code
                }
                
                if is_healthy:
                    response_times.append(response_time)
                else:
                    overall_healthy = False
                    
            except Exception as e:
                endpoint_results[endpoint] = {
                    'healthy': False,
                    'error': str(e)
                }
                overall_healthy = False
        
        # 系統資源檢查
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_healthy = (
            cpu_usage < 90 and 
            memory.percent < 90 and 
            disk.percent < 95
        )
        
        overall_healthy = overall_healthy and system_healthy
        
        return {
            'timestamp': timestamp,
            'overall_healthy': overall_healthy,
            'endpoints': endpoint_results,
            'system': {
                'cpu_percent': cpu_usage,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'healthy': system_healthy
            },
            'average_response_time': statistics.mean(response_times) if response_times else None
        }
    
    def get_current_health(self) -> Optional[Dict[str, Any]]:
        """獲取當前健康狀態"""
        if self.health_history:
            return self.health_history[-1]
        return None
    
    def get_uptime_percentage(self, hours: int = 24) -> float:
        """計算正常運行時間百分比"""
        if not self.health_history:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in self.health_history
            if check['timestamp'] >= cutoff_time
        ]
        
        if not recent_checks:
            return 0.0
        
        healthy_checks = sum(1 for check in recent_checks if check['overall_healthy'])
        return (healthy_checks / len(recent_checks)) * 100
    
    def get_average_response_time(self, hours: int = 1) -> Optional[float]:
        """獲取平均響應時間"""
        if not self.health_history:
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in self.health_history
            if (check['timestamp'] >= cutoff_time and 
                check.get('average_response_time') is not None)
        ]
        
        if not recent_checks:
            return None
        
        response_times = [check['average_response_time'] for check in recent_checks]
        return statistics.mean(response_times)

class SLADatabase:
    """SLA資料庫"""
    
    def __init__(self, db_path: str = "sla.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化資料庫"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # SLA測量表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sla_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    measured_value REAL NOT NULL,
                    target_value REAL NOT NULL,
                    status TEXT NOT NULL,
                    measurement_period INTEGER NOT NULL,
                    details TEXT,
                    customer_id TEXT
                )
            ''')
            
            # 事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    resolved_at TEXT,
                    affected_services TEXT NOT NULL,
                    root_cause TEXT,
                    resolution_steps TEXT,
                    impact_assessment TEXT
                )
            ''')
            
            # SLA目標表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sla_targets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    measurement_period INTEGER NOT NULL,
                    threshold_warning REAL NOT NULL,
                    threshold_breach REAL NOT NULL,
                    unit TEXT NOT NULL,
                    description TEXT NOT NULL,
                    customer_id TEXT
                )
            ''')
            
            # 創建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON sla_measurements (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_started ON incidents (started_at)')
            
            conn.commit()
    
    def save_measurement(self, measurement: SLAMeasurement, customer_id: str = None):
        """保存SLA測量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sla_measurements 
                (timestamp, metric, measured_value, target_value, status, 
                 measurement_period, details, customer_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                measurement.timestamp.isoformat(),
                measurement.metric.value,
                measurement.measured_value,
                measurement.target_value,
                measurement.status.value,
                int(measurement.measurement_period.total_seconds()),
                json.dumps(measurement.details),
                customer_id
            ))
            conn.commit()
    
    def save_incident(self, incident: Incident):
        """保存事件記錄"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO incidents 
                (id, title, description, severity, started_at, resolved_at,
                 affected_services, root_cause, resolution_steps, impact_assessment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident.id,
                incident.title,
                incident.description,
                incident.severity.value,
                incident.started_at.isoformat(),
                incident.resolved_at.isoformat() if incident.resolved_at else None,
                json.dumps(incident.affected_services),
                incident.root_cause,
                json.dumps(incident.resolution_steps),
                json.dumps(incident.impact_assessment)
            ))
            conn.commit()
    
    def get_measurements(self, start_date: datetime, end_date: datetime,
                        customer_id: str = None) -> List[SLAMeasurement]:
        """獲取測量記錄"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT * FROM sla_measurements 
                WHERE timestamp BETWEEN ? AND ?
            '''
            params = [start_date.isoformat(), end_date.isoformat()]
            
            if customer_id:
                query += ' AND customer_id = ?'
                params.append(customer_id)
            
            query += ' ORDER BY timestamp'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            measurements = []
            for row in rows:
                measurements.append(SLAMeasurement(
                    timestamp=datetime.fromisoformat(row[1]),
                    metric=SLAMetric(row[2]),
                    measured_value=row[3],
                    target_value=row[4],
                    status=SLAStatus(row[5]),
                    measurement_period=timedelta(seconds=row[6]),
                    details=json.loads(row[7]) if row[7] else {}
                ))
            
            return measurements
    
    def get_incidents(self, start_date: datetime, end_date: datetime) -> List[Incident]:
        """獲取事件記錄"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM incidents 
                WHERE started_at BETWEEN ? AND ?
                ORDER BY started_at DESC
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            incidents = []
            
            for row in rows:
                incidents.append(Incident(
                    id=row[0],
                    title=row[1],
                    description=row[2],
                    severity=IncidentSeverity(row[3]),
                    started_at=datetime.fromisoformat(row[4]),
                    resolved_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    affected_services=json.loads(row[6]),
                    root_cause=row[7],
                    resolution_steps=json.loads(row[8]) if row[8] else [],
                    impact_assessment=json.loads(row[9]) if row[9] else {}
                ))
            
            return incidents

class SLAManager:
    """SLA管理器"""
    
    def __init__(self, db_path: str = "sla.db"):
        self.database = SLADatabase(db_path)
        self.health_checker = ServiceHealthChecker()
        self.sla_targets = self._initialize_sla_targets()
        self.active_incidents: Dict[str, Incident] = {}
        self.monitoring_active = False
        self.monitor_thread = None
    
    def _initialize_sla_targets(self) -> Dict[SLAMetric, SLATarget]:
        """初始化SLA目標"""
        return {
            SLAMetric.UPTIME: SLATarget(
                metric=SLAMetric.UPTIME,
                target_value=99.9,  # 99.9%正常運行時間
                measurement_period=timedelta(days=30),
                threshold_warning=99.5,
                threshold_breach=99.0,
                unit="%",
                description="服務正常運行時間百分比"
            ),
            SLAMetric.RESPONSE_TIME: SLATarget(
                metric=SLAMetric.RESPONSE_TIME,
                target_value=500.0,  # 500ms
                measurement_period=timedelta(hours=1),
                threshold_warning=750.0,
                threshold_breach=1000.0,
                unit="ms",
                description="平均API響應時間"
            ),
            SLAMetric.ERROR_RATE: SLATarget(
                metric=SLAMetric.ERROR_RATE,
                target_value=1.0,  # 1%錯誤率
                measurement_period=timedelta(hours=1),
                threshold_warning=2.0,
                threshold_breach=5.0,
                unit="%",
                description="API錯誤率"
            ),
            SLAMetric.AVAILABILITY: SLATarget(
                metric=SLAMetric.AVAILABILITY,
                target_value=99.9,  # 99.9%可用性
                measurement_period=timedelta(hours=24),
                threshold_warning=99.5,
                threshold_breach=99.0,
                unit="%",
                description="服務可用性"
            )
        }
    
    def start_sla_monitoring(self):
        """啟動SLA監控"""
        self.monitoring_active = True
        self.health_checker.start_health_checks()
        
        self.monitor_thread = threading.Thread(target=self._sla_monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("SLA監控已啟動")
    
    def stop_sla_monitoring(self):
        """停止SLA監控"""
        self.monitoring_active = False
        self.health_checker.stop_health_checks()
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("SLA監控已停止")
    
    def _sla_monitoring_loop(self):
        """SLA監控循環"""
        while self.monitoring_active:
            try:
                self._perform_sla_measurements()
                self._check_for_incidents()
                time.sleep(60)  # 每分鐘檢查一次
            except Exception as e:
                logger.error(f"SLA監控錯誤: {e}")
                time.sleep(60)
    
    def _perform_sla_measurements(self):
        """執行SLA測量"""
        now = datetime.now()
        
        # 正常運行時間測量
        uptime_target = self.sla_targets[SLAMetric.UPTIME]
        uptime_hours = int(uptime_target.measurement_period.total_seconds() / 3600)
        uptime_percentage = self.health_checker.get_uptime_percentage(uptime_hours)
        
        if uptime_percentage > 0:  # 只有當有數據時才記錄
            uptime_measurement = SLAMeasurement(
                timestamp=now,
                metric=SLAMetric.UPTIME,
                measured_value=uptime_percentage,
                target_value=uptime_target.target_value,
                status=self._determine_sla_status(uptime_percentage, uptime_target),
                measurement_period=uptime_target.measurement_period,
                details={'measurement_hours': uptime_hours}
            )
            self.database.save_measurement(uptime_measurement)
        
        # 響應時間測量
        response_time_target = self.sla_targets[SLAMetric.RESPONSE_TIME]
        avg_response_time = self.health_checker.get_average_response_time(1)  # 1小時
        
        if avg_response_time is not None:
            response_time_measurement = SLAMeasurement(
                timestamp=now,
                metric=SLAMetric.RESPONSE_TIME,
                measured_value=avg_response_time,
                target_value=response_time_target.target_value,
                status=self._determine_sla_status(avg_response_time, response_time_target, reverse=True),
                measurement_period=response_time_target.measurement_period,
                details={'measurement_period_hours': 1}
            )
            self.database.save_measurement(response_time_measurement)
    
    def _determine_sla_status(self, measured_value: float, target: SLATarget, 
                            reverse: bool = False) -> SLAStatus:
        """確定SLA狀態"""
        if reverse:
            # 對於響應時間等指標，值越低越好
            if measured_value <= target.target_value:
                return SLAStatus.MEETING
            elif measured_value <= target.threshold_warning:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.BREACHED
        else:
            # 對於正常運行時間等指標，值越高越好
            if measured_value >= target.target_value:
                return SLAStatus.MEETING
            elif measured_value >= target.threshold_warning:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.BREACHED
    
    def _check_for_incidents(self):
        """檢查是否需要創建事件"""
        current_health = self.health_checker.get_current_health()
        if not current_health:
            return
        
        # 檢查服務是否不健康
        if not current_health['overall_healthy']:
            incident_id = "service_outage_" + str(int(time.time()))
            
            if incident_id not in self.active_incidents:
                # 創建新事件
                incident = Incident(
                    id=incident_id,
                    title="服務可用性問題",
                    description="服務健康檢查失敗",
                    severity=IncidentSeverity.HIGH,
                    started_at=current_health['timestamp'],
                    resolved_at=None,
                    affected_services=["fraud_detection_api"],
                    root_cause=None,
                    resolution_steps=[],
                    impact_assessment={
                        'affected_endpoints': [
                            endpoint for endpoint, status in current_health['endpoints'].items()
                            if not status.get('healthy', False)
                        ],
                        'system_issues': current_health['system']
                    }
                )
                
                self.active_incidents[incident_id] = incident
                self.database.save_incident(incident)
                logger.critical(f"新事件已創建: {incident_id}")
        
        # 檢查是否有事件需要自動解決
        resolved_incidents = []
        for incident_id, incident in self.active_incidents.items():
            if not incident.is_resolved and current_health['overall_healthy']:
                # 服務恢復正常，解決事件
                incident.resolved_at = datetime.now()
                incident.resolution_steps.append("服務自動恢復正常")
                self.database.save_incident(incident)
                resolved_incidents.append(incident_id)
                logger.info(f"事件已自動解決: {incident_id}")
        
        # 移除已解決的事件
        for incident_id in resolved_incidents:
            self.active_incidents.pop(incident_id, None)
    
    def create_incident(self, title: str, description: str, 
                       severity: IncidentSeverity,
                       affected_services: List[str]) -> Incident:
        """手動創建事件"""
        incident_id = f"manual_{int(time.time())}"
        
        incident = Incident(
            id=incident_id,
            title=title,
            description=description,
            severity=severity,
            started_at=datetime.now(),
            resolved_at=None,
            affected_services=affected_services,
            root_cause=None,
            resolution_steps=[],
            impact_assessment={}
        )
        
        self.active_incidents[incident_id] = incident
        self.database.save_incident(incident)
        
        logger.warning(f"手動創建事件: {incident_id} - {title}")
        return incident
    
    def resolve_incident(self, incident_id: str, root_cause: str,
                        resolution_steps: List[str]) -> bool:
        """解決事件"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.resolved_at = datetime.now()
            incident.root_cause = root_cause
            incident.resolution_steps = resolution_steps
            
            self.database.save_incident(incident)
            self.active_incidents.pop(incident_id)
            
            logger.info(f"事件已解決: {incident_id}")
            return True
        
        return False
    
    def generate_sla_report(self, start_date: datetime, end_date: datetime,
                          customer_id: str = None) -> SLAReport:
        """生成SLA報告"""
        measurements = self.database.get_measurements(start_date, end_date, customer_id)
        incidents = self.database.get_incidents(start_date, end_date)
        
        # 計算整體合規性
        if measurements:
            meeting_count = sum(1 for m in measurements if m.status == SLAStatus.MEETING)
            overall_compliance = (meeting_count / len(measurements)) * 100
        else:
            overall_compliance = 0.0
        
        # 計算停機時間
        downtime_minutes = 0.0
        for incident in incidents:
            if incident.duration:
                downtime_minutes += incident.duration.total_seconds() / 60
        
        # 計算應賠償的信用額度（簡化計算）
        period_days = (end_date - start_date).days
        expected_uptime = period_days * 24 * 60 * 0.999  # 99.9%正常運行時間
        actual_uptime = (period_days * 24 * 60) - downtime_minutes
        uptime_percentage = (actual_uptime / (period_days * 24 * 60)) * 100
        
        # 信用額度計算（基於服務費用的百分比）
        if uptime_percentage < 99.9:
            credits_owed = max(0, (99.9 - uptime_percentage) * 10)  # 簡化計算
        else:
            credits_owed = 0.0
        
        return SLAReport(
            period_start=start_date,
            period_end=end_date,
            customer_id=customer_id,
            measurements=measurements,
            incidents=incidents,
            overall_compliance=overall_compliance,
            downtime_minutes=downtime_minutes,
            credits_owed=credits_owed
        )
    
    def get_current_sla_status(self) -> Dict[str, Any]:
        """獲取當前SLA狀態"""
        current_health = self.health_checker.get_current_health()
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy' if current_health and current_health['overall_healthy'] else 'unhealthy',
            'uptime_24h': self.health_checker.get_uptime_percentage(24),
            'uptime_30d': self.health_checker.get_uptime_percentage(24 * 30),
            'avg_response_time_1h': self.health_checker.get_average_response_time(1),
            'active_incidents': len(self.active_incidents),
            'sla_targets': {k.value: v.to_dict() for k, v in self.sla_targets.items()}
        }
        
        if current_health:
            status['current_health'] = current_health
        
        return status
    
    def get_sla_targets(self) -> Dict[SLAMetric, SLATarget]:
        """獲取SLA目標"""
        return self.sla_targets
    
    def update_sla_target(self, metric: SLAMetric, target: SLATarget):
        """更新SLA目標"""
        self.sla_targets[metric] = target
        logger.info(f"SLA目標已更新: {metric.value}")

# 便捷函數
def create_sla_manager(db_path: str = "sla.db") -> SLAManager:
    """創建SLA管理器"""
    return SLAManager(db_path)

def calculate_sla_credits(actual_uptime: float, target_uptime: float = 99.9,
                         monthly_fee: float = 100.0) -> float:
    """計算SLA信用額度"""
    if actual_uptime >= target_uptime:
        return 0.0
    
    # 根據停機時間百分比計算信用額度
    downtime_percentage = target_uptime - actual_uptime
    
    if downtime_percentage <= 0.1:  # 99.8% - 99.9%
        credit_percentage = 10
    elif downtime_percentage <= 0.5:  # 99.4% - 99.8%
        credit_percentage = 25
    elif downtime_percentage <= 1.0:  # 98.9% - 99.4%
        credit_percentage = 50
    else:  # < 98.9%
        credit_percentage = 100
    
    return monthly_fee * (credit_percentage / 100)

if __name__ == "__main__":
    # 示例使用
    sla_manager = create_sla_manager()
    
    # 啟動SLA監控
    sla_manager.start_sla_monitoring()
    
    print("SLA監控已啟動...")
    
    # 模擬運行
    import time
    time.sleep(10)
    
    # 獲取當前狀態
    status = sla_manager.get_current_sla_status()
    print(f"當前SLA狀態: {json.dumps(status, indent=2, ensure_ascii=False)}")
    
    # 停止監控
    sla_manager.stop_sla_monitoring()
    
    print("SLA監控已停止")