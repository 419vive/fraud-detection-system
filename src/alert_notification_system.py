"""
警報和通知系統 - IEEE-CIS 詐騙檢測項目
提供多渠道警報通知、規則引擎和智能通知管理
"""

import smtplib
import json
import asyncio
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
import threading
import queue
import time
from enum import Enum
import requests
import yaml

from .config import get_config
from .exceptions import ModelError

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """警報嚴重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """警報狀態"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """警報數據類"""
    id: str
    timestamp: datetime
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    details: Dict[str, Any]
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        return result
    
    def acknowledge(self, user: str):
        """確認警報"""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        
    def resolve(self):
        """解決警報"""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now()

class NotificationChannel(ABC):
    """通知渠道基類"""
    
    @abstractmethod
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """發送通知"""
        pass

class SlackNotificationChannel(NotificationChannel):
    """Slack通知渠道"""
    
    def __init__(self):
        self.session = None
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """發送Slack通知"""
        try:
            webhook_url = config.get('webhook_url')
            if not webhook_url:
                logger.error("Slack webhook URL未配置")
                return False
            
            # 顏色映射
            color_map = {
                AlertSeverity.LOW: 'good',
                AlertSeverity.MEDIUM: 'warning',
                AlertSeverity.HIGH: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            # 構建消息
            payload = {
                "username": config.get('username', '詐騙檢測監控'),
                "icon_emoji": config.get('icon', ':warning:'),
                "attachments": [{
                    "color": color_map.get(alert.severity, 'warning'),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "嚴重程度", "value": alert.severity.value, "short": True},
                        {"title": "來源", "value": alert.source, "short": True},
                        {"title": "類型", "value": alert.alert_type, "short": True},
                        {"title": "時間", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "詐騙檢測監控系統",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # 添加詳細信息
            if alert.details:
                detail_text = "\n".join([f"• {k}: {v}" for k, v in alert.details.items()])
                payload["attachments"][0]["fields"].append({
                    "title": "詳細信息",
                    "value": detail_text,
                    "short": False
                })
            
            # 發送請求
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Slack通知已發送 - 警報ID: {alert.id}")
                    return True
                else:
                    logger.error(f"Slack通知發送失敗 - 狀態碼: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Slack通知發送異常: {e}")
            return False

class EmailNotificationChannel(NotificationChannel):
    """郵件通知渠道"""
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """發送郵件通知"""
        try:
            smtp_server = config.get('smtp_server')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            from_email = config.get('from_email', username)
            to_emails = config.get('to_emails', [])
            
            if not all([smtp_server, username, password, to_emails]):
                logger.error("郵件配置不完整")
                return False
            
            # 創建郵件內容
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # 郵件正文
            body = f"""
詐騙檢測系統警報

警報類型: {alert.alert_type}
嚴重程度: {alert.severity.value}
來源: {alert.source}
時間: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

消息:
{alert.message}

詳細信息:
"""
            
            if alert.details:
                for key, value in alert.details.items():
                    body += f"  {key}: {value}\n"
            
            body += f"\n警報ID: {alert.id}"
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 發送郵件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"郵件通知已發送 - 警報ID: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"郵件通知發送異常: {e}")
            return False

class WebhookNotificationChannel(NotificationChannel):
    """Webhook通知渠道"""
    
    def __init__(self):
        self.session = None
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """發送Webhook通知"""
        try:
            url = config.get('url')
            if not url:
                logger.error("Webhook URL未配置")
                return False
            
            headers = config.get('headers', {'Content-Type': 'application/json'})
            payload = alert.to_dict()
            
            # 添加自定義字段
            if 'custom_fields' in config:
                payload.update(config['custom_fields'])
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status in [200, 201, 202]:
                    logger.info(f"Webhook通知已發送 - 警報ID: {alert.id}")
                    return True
                else:
                    logger.error(f"Webhook通知發送失敗 - 狀態碼: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Webhook通知發送異常: {e}")
            return False

class TeamsNotificationChannel(NotificationChannel):
    """Microsoft Teams通知渠道"""
    
    def __init__(self):
        self.session = None
    
    async def send(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """發送Teams通知"""
        try:
            webhook_url = config.get('webhook_url')
            if not webhook_url:
                logger.error("Teams webhook URL未配置")
                return False
            
            # 顏色映射
            color_map = {
                AlertSeverity.LOW: '00FF00',
                AlertSeverity.MEDIUM: 'FFFF00',
                AlertSeverity.HIGH: 'FF6600',
                AlertSeverity.CRITICAL: 'FF0000'
            }
            
            # 構建Teams消息卡片
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": color_map.get(alert.severity, 'FFFF00'),
                "summary": alert.title,
                "sections": [{
                    "activityTitle": alert.title,
                    "activitySubtitle": f"來源: {alert.source}",
                    "activityImage": "https://teamsnodesample.azurewebsites.net/static/img/image5.png",
                    "facts": [
                        {"name": "嚴重程度", "value": alert.severity.value},
                        {"name": "類型", "value": alert.alert_type},
                        {"name": "時間", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                    ],
                    "markdown": True,
                    "text": alert.message
                }]
            }
            
            # 添加詳細信息
            if alert.details:
                for key, value in alert.details.items():
                    payload["sections"][0]["facts"].append({
                        "name": key,
                        "value": str(value)
                    })
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Teams通知已發送 - 警報ID: {alert.id}")
                    return True
                else:
                    logger.error(f"Teams通知發送失敗 - 狀態碼: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Teams通知發送異常: {e}")
            return False

class AlertRule:
    """警報規則"""
    
    def __init__(self, name: str, condition: Callable[[Alert], bool], 
                 channels: List[str], config: Dict[str, Any] = None):
        self.name = name
        self.condition = condition
        self.channels = channels
        self.config = config or {}
        self.enabled = True
        self.last_triggered = None
        self.trigger_count = 0
    
    def should_trigger(self, alert: Alert) -> bool:
        """檢查是否應該觸發規則"""
        if not self.enabled:
            return False
        
        # 檢查冷卻期
        cooldown = self.config.get('cooldown_minutes', 0)
        if cooldown > 0 and self.last_triggered:
            if datetime.now() - self.last_triggered < timedelta(minutes=cooldown):
                return False
        
        # 檢查條件
        if self.condition(alert):
            self.last_triggered = datetime.now()
            self.trigger_count += 1
            return True
        
        return False

class AlertNotificationSystem:
    """警報通知系統主類"""
    
    def __init__(self):
        self.channels = {
            'slack': SlackNotificationChannel(),
            'email': EmailNotificationChannel(),
            'webhook': WebhookNotificationChannel(),
            'teams': TeamsNotificationChannel()
        }
        
        self.rules: List[AlertRule] = []
        self.channel_configs: Dict[str, Dict[str, Any]] = {}
        self.alert_queue = asyncio.Queue()
        self.processing = False
        self.alert_history: List[Alert] = []
        self.config = get_config()
        
        # 載入配置
        self._load_configuration()
        
    def _load_configuration(self):
        """載入通知配置"""
        notifications_config = self.config.get('notifications', {})
        
        # 載入渠道配置
        for channel_name, channel_config in notifications_config.items():
            if channel_name in self.channels:
                self.channel_configs[channel_name] = channel_config
        
        # 載入默認規則
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """設置默認警報規則"""
        # 嚴重警報規則
        critical_rule = AlertRule(
            name="critical_alerts",
            condition=lambda alert: alert.severity == AlertSeverity.CRITICAL,
            channels=['slack', 'email', 'teams'],
            config={'cooldown_minutes': 5}
        )
        
        # 高級警報規則
        high_rule = AlertRule(
            name="high_alerts",
            condition=lambda alert: alert.severity == AlertSeverity.HIGH,
            channels=['slack', 'teams'],
            config={'cooldown_minutes': 15}
        )
        
        # 模型性能警報規則
        performance_rule = AlertRule(
            name="performance_alerts",
            condition=lambda alert: alert.alert_type in ['performance_degradation', 'model_failure'],
            channels=['slack', 'email'],
            config={'cooldown_minutes': 30}
        )
        
        # 數據漂移警報規則
        drift_rule = AlertRule(
            name="drift_alerts",
            condition=lambda alert: alert.alert_type == 'data_drift',
            channels=['slack'],
            config={'cooldown_minutes': 60}
        )
        
        self.rules.extend([critical_rule, high_rule, performance_rule, drift_rule])
    
    def add_rule(self, rule: AlertRule):
        """添加警報規則"""
        self.rules.append(rule)
        logger.info(f"警報規則已添加: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """移除警報規則"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        logger.info(f"警報規則已移除: {rule_name}")
    
    def configure_channel(self, channel_name: str, config: Dict[str, Any]):
        """配置通知渠道"""
        if channel_name not in self.channels:
            raise ValueError(f"不支持的通知渠道: {channel_name}")
        
        self.channel_configs[channel_name] = config
        logger.info(f"通知渠道已配置: {channel_name}")
    
    async def send_alert(self, alert: Alert):
        """發送警報"""
        await self.alert_queue.put(alert)
    
    async def start_processing(self):
        """開始處理警報隊列"""
        self.processing = True
        logger.info("警報通知系統已啟動")
        
        while self.processing:
            try:
                # 等待警報
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # 保存到歷史記錄
                self.alert_history.append(alert)
                
                # 處理警報
                await self._process_alert(alert)
                
                # 清理舊歷史記錄
                self._cleanup_history()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"警報處理異常: {e}")
    
    async def _process_alert(self, alert: Alert):
        """處理單個警報"""
        logger.info(f"處理警報: {alert.id} - {alert.title}")
        
        # 檢查匹配的規則
        matched_rules = []
        for rule in self.rules:
            if rule.should_trigger(alert):
                matched_rules.append(rule)
        
        if not matched_rules:
            logger.info(f"警報 {alert.id} 沒有匹配的規則")
            return
        
        # 收集所有需要使用的渠道
        all_channels = set()
        for rule in matched_rules:
            all_channels.update(rule.channels)
        
        # 發送通知
        success_count = 0
        for channel_name in all_channels:
            if channel_name not in self.channel_configs:
                logger.warning(f"渠道 {channel_name} 未配置，跳過")
                continue
            
            try:
                channel = self.channels[channel_name]
                config = self.channel_configs[channel_name]
                
                success = await channel.send(alert, config)
                if success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"渠道 {channel_name} 發送失敗: {e}")
        
        logger.info(f"警報 {alert.id} 已發送到 {success_count}/{len(all_channels)} 個渠道")
    
    def _cleanup_history(self, max_age_days: int = 7, max_count: int = 10000):
        """清理歷史記錄"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # 按時間過濾
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_date
        ]
        
        # 按數量限制
        if len(self.alert_history) > max_count:
            self.alert_history = self.alert_history[-max_count:]
    
    def stop_processing(self):
        """停止處理"""
        self.processing = False
        logger.info("警報通知系統已停止")
    
    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """獲取警報統計"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp >= cutoff_date
        ]
        
        if not recent_alerts:
            return {'message': '沒有警報數據'}
        
        # 按嚴重程度統計
        severity_stats = {}
        for severity in AlertSeverity:
            count = sum(1 for alert in recent_alerts if alert.severity == severity)
            severity_stats[severity.value] = count
        
        # 按類型統計
        type_stats = {}
        for alert in recent_alerts:
            type_stats[alert.alert_type] = type_stats.get(alert.alert_type, 0) + 1
        
        # 按狀態統計
        status_stats = {}
        for status in AlertStatus:
            count = sum(1 for alert in recent_alerts if alert.status == status)
            status_stats[status.value] = count
        
        # 規則統計
        rule_stats = {}
        for rule in self.rules:
            rule_stats[rule.name] = {
                'enabled': rule.enabled,
                'trigger_count': rule.trigger_count,
                'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
            }
        
        return {
            'period': f'{days} days',
            'total_alerts': len(recent_alerts),
            'severity_distribution': severity_stats,
            'type_distribution': type_stats,
            'status_distribution': status_stats,
            'rule_statistics': rule_stats,
            'generated_at': datetime.now().isoformat()
        }
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """確認警報"""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.acknowledge(user)
                logger.info(f"警報 {alert_id} 已由 {user} 確認")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解決警報"""
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.resolve()
                logger.info(f"警報 {alert_id} 已解決")
                return True
        return False
    
    def export_alerts(self, filepath: str, days: int = 30):
        """導出警報數據"""
        cutoff_date = datetime.now() - timedelta(days=days)
        alerts_to_export = [
            alert.to_dict() for alert in self.alert_history 
            if alert.timestamp >= cutoff_date
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(alerts_to_export, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"警報數據已導出至: {filepath}")

# 便捷函數
def create_alert(alert_type: str, severity: AlertSeverity, title: str, 
                message: str, source: str, details: Dict[str, Any] = None) -> Alert:
    """創建警報"""
    return Alert(
        id=f"{alert_type}_{int(datetime.now().timestamp())}",
        timestamp=datetime.now(),
        alert_type=alert_type,
        severity=severity,
        title=title,
        message=message,
        source=source,
        details=details or {}
    )

def create_notification_system() -> AlertNotificationSystem:
    """創建通知系統"""
    return AlertNotificationSystem()

async def send_test_alert(notification_system: AlertNotificationSystem):
    """發送測試警報"""
    test_alert = create_alert(
        alert_type="test",
        severity=AlertSeverity.MEDIUM,
        title="測試警報",
        message="這是一個測試警報，用於驗證通知系統功能",
        source="test_system",
        details={"test_parameter": "test_value"}
    )
    
    await notification_system.send_alert(test_alert)

if __name__ == "__main__":
    print("警報和通知系統模組已載入完成！")