"""
API Key 管理系統 - IEEE-CIS 詐騙檢測服務
提供安全的API密鑰生成、驗證、管理和權限控制
"""

import secrets
import hashlib
import hmac
import base64
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import json
import logging
import time
from functools import wraps
import jwt
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

class KeyStatus(Enum):
    """API密鑰狀態"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    EXPIRED = "expired"

class Permission(Enum):
    """權限類型"""
    READ = "read"
    WRITE = "write"
    PREDICT = "predict"
    BATCH_PREDICT = "batch_predict"
    MONITOR = "monitor"
    ADMIN = "admin"

@dataclass
class RateLimit:
    """速率限制配置"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = None  # 突發流量限制
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class APIKey:
    """API密鑰數據模型"""
    id: str
    customer_id: str
    name: str
    key_hash: str  # 存儲哈希值，不存儲原始密鑰
    permissions: Set[Permission]
    rate_limits: RateLimit
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    allowed_ips: List[str]  # IP白名單
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['permissions'] = [p.value for p in self.permissions]
        result['rate_limits'] = self.rate_limits.to_dict()
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        result['last_used_at'] = self.last_used_at.isoformat() if self.last_used_at else None
        return result
    
    def has_permission(self, permission: Permission) -> bool:
        """檢查是否具有特定權限"""
        return permission in self.permissions or Permission.ADMIN in self.permissions
    
    def is_valid(self) -> bool:
        """檢查密鑰是否有效"""
        if self.status != KeyStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        
        return True

@dataclass
class UsageLog:
    """使用日誌"""
    id: str
    api_key_id: str
    customer_id: str
    endpoint: str
    method: str
    ip_address: str
    user_agent: str
    response_status: int
    response_time_ms: float
    timestamp: datetime
    request_size: int
    response_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class TokenGenerator:
    """安全令牌生成器"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or Fernet.generate_key()
        self.cipher = Fernet(self.secret_key)
    
    def generate_api_key(self, prefix: str = "fdk") -> str:
        """生成API密鑰"""
        # 生成隨機部分 (32字節)
        random_part = secrets.token_urlsafe(32)
        
        # 創建檢查和 (4字節)
        checksum = hashlib.sha256(random_part.encode()).hexdigest()[:8]
        
        # 組合密鑰
        api_key = f"{prefix}_{random_part}_{checksum}"
        
        return api_key
    
    def hash_api_key(self, api_key: str) -> str:
        """對API密鑰進行哈希"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key_format(self, api_key: str) -> bool:
        """驗證API密鑰格式"""
        try:
            parts = api_key.split('_')
            if len(parts) != 3:
                return False
            
            prefix, random_part, checksum = parts
            expected_checksum = hashlib.sha256(random_part.encode()).hexdigest()[:8]
            
            return checksum == expected_checksum
        except Exception:
            return False
    
    def generate_jwt_token(self, payload: Dict[str, Any], 
                          expires_in: int = 3600) -> str:
        """生成JWT令牌"""
        payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
        payload['iat'] = datetime.utcnow()
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """驗證JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT令牌已過期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("無效的JWT令牌")
            return None

class RateLimiter:
    """速率限制器"""
    
    def __init__(self):
        self.usage_records: Dict[str, List[float]] = {}
    
    def is_allowed(self, api_key_id: str, rate_limits: RateLimit) -> Tuple[bool, Dict[str, Any]]:
        """檢查是否允許請求"""
        now = time.time()
        
        # 初始化記錄
        if api_key_id not in self.usage_records:
            self.usage_records[api_key_id] = []
        
        usage_times = self.usage_records[api_key_id]
        
        # 清理過期記錄 (保留最近24小時)
        cutoff_time = now - 86400  # 24小時
        usage_times = [t for t in usage_times if t > cutoff_time]
        self.usage_records[api_key_id] = usage_times
        
        # 檢查各時間段的限制
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        requests_last_minute = len([t for t in usage_times if t > minute_ago])
        requests_last_hour = len([t for t in usage_times if t > hour_ago])
        requests_last_day = len([t for t in usage_times if t > day_ago])
        
        # 檢查是否超過限制
        if requests_last_minute >= rate_limits.requests_per_minute:
            return False, {
                'error': 'Rate limit exceeded',
                'limit_type': 'per_minute',
                'current_usage': requests_last_minute,
                'limit': rate_limits.requests_per_minute,
                'reset_at': int(minute_ago + 60)
            }
        
        if requests_last_hour >= rate_limits.requests_per_hour:
            return False, {
                'error': 'Rate limit exceeded',
                'limit_type': 'per_hour',
                'current_usage': requests_last_hour,
                'limit': rate_limits.requests_per_hour,
                'reset_at': int(hour_ago + 3600)
            }
        
        if requests_last_day >= rate_limits.requests_per_day:
            return False, {
                'error': 'Rate limit exceeded',
                'limit_type': 'per_day',
                'current_usage': requests_last_day,
                'limit': rate_limits.requests_per_day,
                'reset_at': int(day_ago + 86400)
            }
        
        # 記錄本次請求
        usage_times.append(now)
        
        return True, {
            'remaining_minute': rate_limits.requests_per_minute - requests_last_minute - 1,
            'remaining_hour': rate_limits.requests_per_hour - requests_last_hour - 1,
            'remaining_day': rate_limits.requests_per_day - requests_last_day - 1
        }

class APIKeyDatabase:
    """API密鑰資料庫"""
    
    def __init__(self, db_path: str = "api_keys.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化資料庫"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 客戶表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customers (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    company_name TEXT,
                    plan_type TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # API密鑰表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    key_hash TEXT UNIQUE NOT NULL,
                    permissions TEXT NOT NULL,
                    rate_limits TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used_at TEXT,
                    usage_count INTEGER DEFAULT 0,
                    allowed_ips TEXT,
                    metadata TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers (id)
                )
            ''')
            
            # 使用日誌表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id TEXT PRIMARY KEY,
                    api_key_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    response_status INTEGER NOT NULL,
                    response_time_ms REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    request_size INTEGER DEFAULT 0,
                    response_size INTEGER DEFAULT 0,
                    FOREIGN KEY (api_key_id) REFERENCES api_keys (id)
                )
            ''')
            
            # 創建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_logs_timestamp ON usage_logs (timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_logs_api_key ON usage_logs (api_key_id)')
            
            conn.commit()
    
    def create_customer(self, email: str, company_name: str = None, 
                       plan_type: str = "free", metadata: Dict[str, Any] = None) -> str:
        """創建客戶"""
        customer_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO customers (id, email, company_name, plan_type, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                customer_id,
                email,
                company_name,
                plan_type,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
        
        logger.info(f"客戶已創建: {customer_id} ({email})")
        return customer_id
    
    def save_api_key(self, api_key: APIKey):
        """保存API密鑰"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO api_keys 
                (id, customer_id, name, key_hash, permissions, rate_limits, status,
                 created_at, expires_at, last_used_at, usage_count, allowed_ips, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                api_key.id,
                api_key.customer_id,
                api_key.name,
                api_key.key_hash,
                json.dumps([p.value for p in api_key.permissions]),
                json.dumps(api_key.rate_limits.to_dict()),
                api_key.status.value,
                api_key.created_at.isoformat(),
                api_key.expires_at.isoformat() if api_key.expires_at else None,
                api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                api_key.usage_count,
                json.dumps(api_key.allowed_ips),
                json.dumps(api_key.metadata)
            ))
            conn.commit()
    
    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """根據哈希值獲取API密鑰"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM api_keys WHERE key_hash = ?', (key_hash,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_api_key(row)
            return None
    
    def get_customer_api_keys(self, customer_id: str) -> List[APIKey]:
        """獲取客戶的所有API密鑰"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM api_keys WHERE customer_id = ?', (customer_id,))
            rows = cursor.fetchall()
            
            return [self._row_to_api_key(row) for row in rows]
    
    def _row_to_api_key(self, row) -> APIKey:
        """將資料庫行轉換為APIKey對象"""
        return APIKey(
            id=row[0],
            customer_id=row[1],
            name=row[2],
            key_hash=row[3],
            permissions=set(Permission(p) for p in json.loads(row[4])),
            rate_limits=RateLimit(**json.loads(row[5])),
            status=KeyStatus(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
            last_used_at=datetime.fromisoformat(row[9]) if row[9] else None,
            usage_count=row[10],
            allowed_ips=json.loads(row[11]),
            metadata=json.loads(row[12]) if row[12] else {}
        )
    
    def update_api_key_usage(self, api_key_id: str):
        """更新API密鑰使用記錄"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE api_keys 
                SET last_used_at = ?, usage_count = usage_count + 1
                WHERE id = ?
            ''', (datetime.now().isoformat(), api_key_id))
            conn.commit()
    
    def log_usage(self, usage_log: UsageLog):
        """記錄使用日誌"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO usage_logs 
                (id, api_key_id, customer_id, endpoint, method, ip_address, user_agent,
                 response_status, response_time_ms, timestamp, request_size, response_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                usage_log.id,
                usage_log.api_key_id,
                usage_log.customer_id,
                usage_log.endpoint,
                usage_log.method,
                usage_log.ip_address,
                usage_log.user_agent,
                usage_log.response_status,
                usage_log.response_time_ms,
                usage_log.timestamp.isoformat(),
                usage_log.request_size,
                usage_log.response_size
            ))
            conn.commit()
    
    def get_usage_stats(self, customer_id: str, days: int = 30) -> Dict[str, Any]:
        """獲取使用統計"""
        start_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 總請求數
            cursor.execute('''
                SELECT COUNT(*) FROM usage_logs 
                WHERE customer_id = ? AND timestamp > ?
            ''', (customer_id, start_date.isoformat()))
            total_requests = cursor.fetchone()[0]
            
            # 按狀態碼統計
            cursor.execute('''
                SELECT response_status, COUNT(*) FROM usage_logs 
                WHERE customer_id = ? AND timestamp > ?
                GROUP BY response_status
            ''', (customer_id, start_date.isoformat()))
            status_stats = dict(cursor.fetchall())
            
            # 平均響應時間
            cursor.execute('''
                SELECT AVG(response_time_ms) FROM usage_logs 
                WHERE customer_id = ? AND timestamp > ?
            ''', (customer_id, start_date.isoformat()))
            avg_response_time = cursor.fetchone()[0] or 0
            
            return {
                'total_requests': total_requests,
                'status_distribution': status_stats,
                'average_response_time_ms': avg_response_time,
                'success_rate': status_stats.get(200, 0) / max(total_requests, 1) * 100
            }

class APIKeyManager:
    """API密鑰管理器"""
    
    def __init__(self, db_path: str = "api_keys.db", secret_key: str = None):
        self.database = APIKeyDatabase(db_path)
        self.token_generator = TokenGenerator(secret_key)
        self.rate_limiter = RateLimiter()
    
    def create_api_key(self, customer_id: str, name: str,
                       permissions: List[Permission],
                       rate_limits: RateLimit,
                       expires_in_days: int = None,
                       allowed_ips: List[str] = None,
                       metadata: Dict[str, Any] = None) -> Tuple[str, APIKey]:
        """創建API密鑰"""
        # 生成原始密鑰
        raw_key = self.token_generator.generate_api_key()
        key_hash = self.token_generator.hash_api_key(raw_key)
        
        # 創建APIKey對象
        api_key = APIKey(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            name=name,
            key_hash=key_hash,
            permissions=set(permissions),
            rate_limits=rate_limits,
            status=KeyStatus.ACTIVE,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None,
            last_used_at=None,
            usage_count=0,
            allowed_ips=allowed_ips or [],
            metadata=metadata or {}
        )
        
        # 保存到資料庫
        self.database.save_api_key(api_key)
        
        logger.info(f"API密鑰已創建: {api_key.id} ({name})")
        
        return raw_key, api_key
    
    def authenticate_api_key(self, raw_key: str, 
                           ip_address: str = None) -> Optional[APIKey]:
        """驗證API密鑰"""
        # 驗證格式
        if not self.token_generator.verify_api_key_format(raw_key):
            logger.warning(f"無效的API密鑰格式: {raw_key[:10]}...")
            return None
        
        # 計算哈希
        key_hash = self.token_generator.hash_api_key(raw_key)
        
        # 從資料庫獲取
        api_key = self.database.get_api_key_by_hash(key_hash)
        if not api_key:
            logger.warning(f"API密鑰不存在: {raw_key[:10]}...")
            return None
        
        # 檢查有效性
        if not api_key.is_valid():
            logger.warning(f"API密鑰無效: {api_key.id}")
            return None
        
        # 檢查IP白名單
        if api_key.allowed_ips and ip_address:
            if ip_address not in api_key.allowed_ips:
                logger.warning(f"IP地址不在白名單中: {ip_address}")
                return None
        
        return api_key
    
    def check_rate_limit(self, api_key: APIKey) -> Tuple[bool, Dict[str, Any]]:
        """檢查速率限制"""
        return self.rate_limiter.is_allowed(api_key.id, api_key.rate_limits)
    
    def log_api_usage(self, api_key: APIKey, endpoint: str, method: str,
                     ip_address: str, user_agent: str, response_status: int,
                     response_time_ms: float, request_size: int = 0,
                     response_size: int = 0):
        """記錄API使用"""
        # 更新密鑰使用統計
        self.database.update_api_key_usage(api_key.id)
        
        # 創建使用日誌
        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key.id,
            customer_id=api_key.customer_id,
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent or "",
            response_status=response_status,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(),
            request_size=request_size,
            response_size=response_size
        )
        
        self.database.log_usage(usage_log)
    
    def revoke_api_key(self, api_key_id: str) -> bool:
        """撤銷API密鑰"""
        try:
            # 獲取密鑰
            api_key = self.get_api_key_by_id(api_key_id)
            if not api_key:
                return False
            
            # 更新狀態
            api_key.status = KeyStatus.REVOKED
            self.database.save_api_key(api_key)
            
            logger.info(f"API密鑰已撤銷: {api_key_id}")
            return True
        except Exception as e:
            logger.error(f"撤銷API密鑰失敗: {e}")
            return False
    
    def get_api_key_by_id(self, api_key_id: str) -> Optional[APIKey]:
        """根據ID獲取API密鑰"""
        # 這需要在資料庫中添加相應方法
        with sqlite3.connect(self.database.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM api_keys WHERE id = ?', (api_key_id,))
            row = cursor.fetchone()
            
            if row:
                return self.database._row_to_api_key(row)
            return None
    
    def get_customer_keys(self, customer_id: str) -> List[APIKey]:
        """獲取客戶的API密鑰"""
        return self.database.get_customer_api_keys(customer_id)
    
    def get_usage_statistics(self, customer_id: str, days: int = 30) -> Dict[str, Any]:
        """獲取使用統計"""
        return self.database.get_usage_stats(customer_id, days)

# 裝飾器用於API認證
def require_api_key(permissions: List[Permission] = None):
    """API密鑰認證裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 這裡需要從請求中提取API密鑰
            # 實際實現會根據web框架而有所不同
            api_key = kwargs.get('api_key')
            if not api_key or not api_key.is_valid():
                return {'error': 'Invalid API key'}, 401
            
            # 檢查權限
            if permissions:
                for permission in permissions:
                    if not api_key.has_permission(permission):
                        return {'error': 'Insufficient permissions'}, 403
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# 便捷函數
def create_api_key_manager(db_path: str = "api_keys.db") -> APIKeyManager:
    """創建API密鑰管理器"""
    return APIKeyManager(db_path)

def get_default_rate_limits(plan_type: str) -> RateLimit:
    """獲取默認速率限制"""
    limits_map = {
        'free': RateLimit(10, 100, 1000),
        'basic': RateLimit(100, 1000, 10000),
        'professional': RateLimit(500, 5000, 50000),
        'enterprise': RateLimit(2000, 20000, 200000)
    }
    
    return limits_map.get(plan_type, limits_map['free'])

if __name__ == "__main__":
    # 示例使用
    manager = create_api_key_manager()
    
    # 創建客戶
    customer_id = manager.database.create_customer(
        email="test@example.com",
        company_name="Test Company",
        plan_type="basic"
    )
    
    # 創建API密鑰
    raw_key, api_key = manager.create_api_key(
        customer_id=customer_id,
        name="Test API Key",
        permissions=[Permission.PREDICT, Permission.READ],
        rate_limits=get_default_rate_limits("basic"),
        expires_in_days=365
    )
    
    print(f"API密鑰已創建: {raw_key}")
    print(f"密鑰ID: {api_key.id}")
    
    # 驗證密鑰
    validated_key = manager.authenticate_api_key(raw_key)
    print(f"驗證結果: {'成功' if validated_key else '失敗'}")