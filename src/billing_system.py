"""
商業計費系統 - IEEE-CIS 詐騙檢測服務
提供按次收費、訂閱制和企業級計費功能
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import logging
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import hmac

logger = logging.getLogger(__name__)

class PlanType(Enum):
    """訂閱方案類型"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class UsageType(Enum):
    """使用類型"""
    PREDICTION = "prediction"
    BATCH_PREDICTION = "batch_prediction"
    MONITORING = "monitoring"
    DRIFT_DETECTION = "drift_detection"
    CUSTOM_MODEL = "custom_model"

class BillingStatus(Enum):
    """計費狀態"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

@dataclass
class PricingTier:
    """定價層級"""
    name: str
    min_usage: int
    max_usage: Optional[int]
    price_per_unit: Decimal
    currency: str = "USD"
    
    def calculate_cost(self, usage: int) -> Decimal:
        """計算費用"""
        applicable_usage = min(usage, self.max_usage or float('inf'))
        applicable_usage = max(0, applicable_usage - self.min_usage)
        return Decimal(str(applicable_usage)) * self.price_per_unit

@dataclass
class SubscriptionPlan:
    """訂閱方案"""
    plan_type: PlanType
    name: str
    monthly_price: Decimal
    included_predictions: int
    max_requests_per_minute: int
    max_requests_per_day: int
    features: List[str]
    overage_pricing: List[PricingTier]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['plan_type'] = self.plan_type.value
        result['monthly_price'] = float(self.monthly_price)
        result['overage_pricing'] = [
            {
                'name': tier.name,
                'min_usage': tier.min_usage,
                'max_usage': tier.max_usage,
                'price_per_unit': float(tier.price_per_unit),
                'currency': tier.currency
            }
            for tier in self.overage_pricing
        ]
        return result

@dataclass
class UsageRecord:
    """使用記錄"""
    id: str
    customer_id: str
    api_key_id: str
    usage_type: UsageType
    quantity: int
    unit_cost: Decimal
    total_cost: Decimal
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['usage_type'] = self.usage_type.value
        result['unit_cost'] = float(self.unit_cost)
        result['total_cost'] = float(self.total_cost)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class Invoice:
    """發票"""
    id: str
    customer_id: str
    billing_period_start: datetime
    billing_period_end: datetime
    usage_records: List[UsageRecord]
    subscription_cost: Decimal
    usage_cost: Decimal
    total_cost: Decimal
    status: BillingStatus
    created_at: datetime
    due_date: datetime
    paid_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['billing_period_start'] = self.billing_period_start.isoformat()
        result['billing_period_end'] = self.billing_period_end.isoformat()
        result['usage_records'] = [record.to_dict() for record in self.usage_records]
        result['subscription_cost'] = float(self.subscription_cost)
        result['usage_cost'] = float(self.usage_cost)
        result['total_cost'] = float(self.total_cost)
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['due_date'] = self.due_date.isoformat()
        if self.paid_at:
            result['paid_at'] = self.paid_at.isoformat()
        return result

class PricingEngine:
    """定價引擎"""
    
    def __init__(self):
        self.plans = self._initialize_pricing_plans()
        self.pay_as_you_go_pricing = self._initialize_payg_pricing()
    
    def _initialize_pricing_plans(self) -> Dict[PlanType, SubscriptionPlan]:
        """初始化訂閱方案"""
        # 免費方案
        free_plan = SubscriptionPlan(
            plan_type=PlanType.FREE,
            name="免費方案",
            monthly_price=Decimal('0.00'),
            included_predictions=1000,
            max_requests_per_minute=10,
            max_requests_per_day=1000,
            features=[
                "基礎詐騙檢測",
                "API訪問",
                "基礎支援"
            ],
            overage_pricing=[
                PricingTier("超額使用", 1000, None, Decimal('0.01'))
            ]
        )
        
        # 基礎方案
        basic_plan = SubscriptionPlan(
            plan_type=PlanType.BASIC,
            name="基礎方案",
            monthly_price=Decimal('99.00'),
            included_predictions=10000,
            max_requests_per_minute=100,
            max_requests_per_day=10000,
            features=[
                "進階詐騙檢測",
                "批量處理",
                "基礎監控",
                "電子郵件支援"
            ],
            overage_pricing=[
                PricingTier("1-10K", 10000, 20000, Decimal('0.008')),
                PricingTier("10K+", 20000, None, Decimal('0.005'))
            ]
        )
        
        # 專業方案
        professional_plan = SubscriptionPlan(
            plan_type=PlanType.PROFESSIONAL,
            name="專業方案",
            monthly_price=Decimal('299.00'),
            included_predictions=50000,
            max_requests_per_minute=500,
            max_requests_per_day=50000,
            features=[
                "高級詐騙檢測",
                "實時監控",
                "數據漂移檢測",
                "自定義模型",
                "優先支援"
            ],
            overage_pricing=[
                PricingTier("1-25K", 50000, 75000, Decimal('0.006')),
                PricingTier("25K-100K", 75000, 150000, Decimal('0.004')),
                PricingTier("100K+", 150000, None, Decimal('0.002'))
            ]
        )
        
        # 企業方案
        enterprise_plan = SubscriptionPlan(
            plan_type=PlanType.ENTERPRISE,
            name="企業方案",
            monthly_price=Decimal('999.00'),
            included_predictions=200000,
            max_requests_per_minute=2000,
            max_requests_per_day=200000,
            features=[
                "企業級詐騙檢測",
                "24/7實時監控",
                "高級分析",
                "專屬模型訓練",
                "SLA保證",
                "專屬客戶經理"
            ],
            overage_pricing=[
                PricingTier("1-100K", 200000, 300000, Decimal('0.003')),
                PricingTier("100K-500K", 300000, 700000, Decimal('0.002')),
                PricingTier("500K+", 700000, None, Decimal('0.001'))
            ]
        )
        
        return {
            PlanType.FREE: free_plan,
            PlanType.BASIC: basic_plan,
            PlanType.PROFESSIONAL: professional_plan,
            PlanType.ENTERPRISE: enterprise_plan
        }
    
    def _initialize_payg_pricing(self) -> Dict[UsageType, List[PricingTier]]:
        """初始化按需付費定價"""
        return {
            UsageType.PREDICTION: [
                PricingTier("1-10K", 0, 10000, Decimal('0.015')),
                PricingTier("10K-100K", 10000, 100000, Decimal('0.012')),
                PricingTier("100K+", 100000, None, Decimal('0.008'))
            ],
            UsageType.BATCH_PREDICTION: [
                PricingTier("批量處理", 0, None, Decimal('0.008'))
            ],
            UsageType.MONITORING: [
                PricingTier("監控服務", 0, None, Decimal('0.002'))
            ],
            UsageType.DRIFT_DETECTION: [
                PricingTier("漂移檢測", 0, None, Decimal('0.005'))
            ],
            UsageType.CUSTOM_MODEL: [
                PricingTier("自定義模型", 0, None, Decimal('0.020'))
            ]
        }
    
    def calculate_subscription_cost(self, plan_type: PlanType, 
                                  usage: int) -> Tuple[Decimal, Decimal]:
        """計算訂閱費用"""
        plan = self.plans[plan_type]
        subscription_cost = plan.monthly_price
        
        # 計算超額費用
        overage_usage = max(0, usage - plan.included_predictions)
        overage_cost = Decimal('0.00')
        
        remaining_usage = overage_usage
        for tier in plan.overage_pricing:
            if remaining_usage <= 0:
                break
            tier_cost = tier.calculate_cost(remaining_usage)
            overage_cost += tier_cost
            if tier.max_usage:
                remaining_usage -= (tier.max_usage - tier.min_usage)
        
        return subscription_cost, overage_cost
    
    def calculate_payg_cost(self, usage_type: UsageType, 
                           quantity: int) -> Decimal:
        """計算按需付費成本"""
        pricing_tiers = self.pay_as_you_go_pricing.get(usage_type, [])
        if not pricing_tiers:
            return Decimal('0.00')
        
        total_cost = Decimal('0.00')
        remaining_quantity = quantity
        
        for tier in pricing_tiers:
            if remaining_quantity <= 0:
                break
            tier_cost = tier.calculate_cost(remaining_quantity)
            total_cost += tier_cost
            if tier.max_usage:
                remaining_quantity -= (tier.max_usage - tier.min_usage)
        
        return total_cost
    
    def get_plan_by_type(self, plan_type: PlanType) -> SubscriptionPlan:
        """根據類型獲取方案"""
        return self.plans[plan_type]

class BillingDatabase:
    """計費資料庫"""
    
    def __init__(self, db_path: str = "billing.db"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """初始化資料庫"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 使用記錄表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_records (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    api_key_id TEXT NOT NULL,
                    usage_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_cost REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers (id)
                )
            ''')
            
            # 發票表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS invoices (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    billing_period_start TEXT NOT NULL,
                    billing_period_end TEXT NOT NULL,
                    subscription_cost REAL NOT NULL,
                    usage_cost REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    due_date TEXT NOT NULL,
                    paid_at TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers (id)
                )
            ''')
            
            # 客戶訂閱表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_subscriptions (
                    customer_id TEXT PRIMARY KEY,
                    plan_type TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ends_at TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (customer_id) REFERENCES customers (id)
                )
            ''')
            
            conn.commit()
    
    def record_usage(self, usage_record: UsageRecord):
        """記錄使用量"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO usage_records 
                (id, customer_id, api_key_id, usage_type, quantity, 
                 unit_cost, total_cost, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                usage_record.id,
                usage_record.customer_id,
                usage_record.api_key_id,
                usage_record.usage_type.value,
                usage_record.quantity,
                float(usage_record.unit_cost),
                float(usage_record.total_cost),
                usage_record.timestamp.isoformat(),
                json.dumps(usage_record.metadata)
            ))
            conn.commit()
    
    def get_usage_for_period(self, customer_id: str, 
                           start_date: datetime, 
                           end_date: datetime) -> List[UsageRecord]:
        """獲取期間內的使用記錄"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM usage_records 
                WHERE customer_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (customer_id, start_date.isoformat(), end_date.isoformat()))
            
            records = []
            for row in cursor.fetchall():
                records.append(UsageRecord(
                    id=row[0],
                    customer_id=row[1],
                    api_key_id=row[2],
                    usage_type=UsageType(row[3]),
                    quantity=row[4],
                    unit_cost=Decimal(str(row[5])),
                    total_cost=Decimal(str(row[6])),
                    timestamp=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8]) if row[8] else {}
                ))
            
            return records
    
    def save_invoice(self, invoice: Invoice):
        """保存發票"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO invoices 
                (id, customer_id, billing_period_start, billing_period_end,
                 subscription_cost, usage_cost, total_cost, status,
                 created_at, due_date, paid_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                invoice.id,
                invoice.customer_id,
                invoice.billing_period_start.isoformat(),
                invoice.billing_period_end.isoformat(),
                float(invoice.subscription_cost),
                float(invoice.usage_cost),
                float(invoice.total_cost),
                invoice.status.value,
                invoice.created_at.isoformat(),
                invoice.due_date.isoformat(),
                invoice.paid_at.isoformat() if invoice.paid_at else None
            ))
            conn.commit()
    
    def get_customer_subscription(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """獲取客戶訂閱信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM customer_subscriptions 
                WHERE customer_id = ? AND is_active = 1
            ''', (customer_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'customer_id': row[0],
                    'plan_type': PlanType(row[1]),
                    'started_at': datetime.fromisoformat(row[2]),
                    'ends_at': datetime.fromisoformat(row[3]) if row[3] else None,
                    'is_active': bool(row[4])
                }
            return None
    
    def update_customer_subscription(self, customer_id: str, 
                                   plan_type: PlanType,
                                   started_at: datetime,
                                   ends_at: Optional[datetime] = None):
        """更新客戶訂閱"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO customer_subscriptions 
                (customer_id, plan_type, started_at, ends_at, is_active)
                VALUES (?, ?, ?, ?, 1)
            ''', (
                customer_id,
                plan_type.value,
                started_at.isoformat(),
                ends_at.isoformat() if ends_at else None
            ))
            conn.commit()

class BillingEngine:
    """計費引擎"""
    
    def __init__(self, db_path: str = "billing.db"):
        self.pricing_engine = PricingEngine()
        self.database = BillingDatabase(db_path)
    
    def record_api_usage(self, customer_id: str, api_key_id: str,
                        usage_type: UsageType, quantity: int = 1,
                        metadata: Dict[str, Any] = None) -> UsageRecord:
        """記錄API使用量"""
        # 獲取客戶訂閱信息
        subscription = self.database.get_customer_subscription(customer_id)
        
        if subscription:
            # 訂閱客戶 - 記錄使用量，稍後在月度賬單中計費
            unit_cost = Decimal('0.00')  # 包含在訂閱中
            total_cost = Decimal('0.00')
        else:
            # 按需付費客戶
            unit_cost = self._calculate_unit_cost(usage_type, quantity)
            total_cost = unit_cost * quantity
        
        # 創建使用記錄
        usage_record = UsageRecord(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            api_key_id=api_key_id,
            usage_type=usage_type,
            quantity=quantity,
            unit_cost=unit_cost,
            total_cost=total_cost,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # 保存到資料庫
        self.database.record_usage(usage_record)
        
        logger.info(f"記錄使用量 - 客戶: {customer_id}, 類型: {usage_type.value}, 數量: {quantity}")
        
        return usage_record
    
    def _calculate_unit_cost(self, usage_type: UsageType, quantity: int) -> Decimal:
        """計算單位成本"""
        total_cost = self.pricing_engine.calculate_payg_cost(usage_type, quantity)
        return total_cost / quantity if quantity > 0 else Decimal('0.00')
    
    def generate_monthly_invoice(self, customer_id: str, 
                               billing_date: datetime) -> Invoice:
        """生成月度發票"""
        # 計算計費期間
        billing_period_start = billing_date.replace(day=1)
        if billing_date.month == 12:
            billing_period_end = billing_date.replace(year=billing_date.year + 1, month=1, day=1)
        else:
            billing_period_end = billing_date.replace(month=billing_date.month + 1, day=1)
        billing_period_end -= timedelta(seconds=1)
        
        # 獲取使用記錄
        usage_records = self.database.get_usage_for_period(
            customer_id, billing_period_start, billing_period_end
        )
        
        # 獲取訂閱信息
        subscription = self.database.get_customer_subscription(customer_id)
        
        if subscription:
            # 訂閱客戶
            plan_type = subscription['plan_type']
            total_usage = sum(record.quantity for record in usage_records 
                            if record.usage_type == UsageType.PREDICTION)
            
            subscription_cost, usage_cost = self.pricing_engine.calculate_subscription_cost(
                plan_type, total_usage
            )
        else:
            # 按需付費客戶
            subscription_cost = Decimal('0.00')
            usage_cost = sum(record.total_cost for record in usage_records)
        
        # 創建發票
        invoice = Invoice(
            id=f"INV-{customer_id}-{billing_date.strftime('%Y%m')}",
            customer_id=customer_id,
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            usage_records=usage_records,
            subscription_cost=subscription_cost,
            usage_cost=usage_cost,
            total_cost=subscription_cost + usage_cost,
            status=BillingStatus.PENDING,
            created_at=datetime.now(),
            due_date=datetime.now() + timedelta(days=30)
        )
        
        # 保存發票
        self.database.save_invoice(invoice)
        
        logger.info(f"生成月度發票 - 客戶: {customer_id}, 總額: ${invoice.total_cost}")
        
        return invoice
    
    def get_customer_usage_summary(self, customer_id: str, 
                                 days: int = 30) -> Dict[str, Any]:
        """獲取客戶使用摘要"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        usage_records = self.database.get_usage_for_period(
            customer_id, start_date, end_date
        )
        
        # 按使用類型統計
        usage_by_type = {}
        total_cost = Decimal('0.00')
        
        for record in usage_records:
            usage_type = record.usage_type.value
            if usage_type not in usage_by_type:
                usage_by_type[usage_type] = {
                    'quantity': 0,
                    'cost': Decimal('0.00')
                }
            
            usage_by_type[usage_type]['quantity'] += record.quantity
            usage_by_type[usage_type]['cost'] += record.total_cost
            total_cost += record.total_cost
        
        # 轉換為可序列化格式
        for usage_type in usage_by_type:
            usage_by_type[usage_type]['cost'] = float(usage_by_type[usage_type]['cost'])
        
        return {
            'customer_id': customer_id,
            'period_days': days,
            'total_requests': len(usage_records),
            'total_cost': float(total_cost),
            'usage_by_type': usage_by_type,
            'subscription': self.database.get_customer_subscription(customer_id)
        }
    
    def upgrade_customer_plan(self, customer_id: str, 
                            new_plan_type: PlanType) -> bool:
        """升級客戶方案"""
        try:
            self.database.update_customer_subscription(
                customer_id, new_plan_type, datetime.now()
            )
            logger.info(f"客戶 {customer_id} 已升級至 {new_plan_type.value} 方案")
            return True
        except Exception as e:
            logger.error(f"升級方案失敗: {e}")
            return False
    
    def get_pricing_info(self) -> Dict[str, Any]:
        """獲取定價信息"""
        plans_info = {}
        for plan_type, plan in self.pricing_engine.plans.items():
            plans_info[plan_type.value] = plan.to_dict()
        
        payg_info = {}
        for usage_type, tiers in self.pricing_engine.pay_as_you_go_pricing.items():
            payg_info[usage_type.value] = [
                {
                    'name': tier.name,
                    'min_usage': tier.min_usage,
                    'max_usage': tier.max_usage,
                    'price_per_unit': float(tier.price_per_unit),
                    'currency': tier.currency
                }
                for tier in tiers
            ]
        
        return {
            'subscription_plans': plans_info,
            'pay_as_you_go': payg_info
        }

# 便捷函數
def create_billing_engine(db_path: str = "billing.db") -> BillingEngine:
    """創建計費引擎"""
    return BillingEngine(db_path)

def calculate_estimated_monthly_cost(plan_type: PlanType, 
                                   expected_usage: int,
                                   pricing_engine: PricingEngine = None) -> Dict[str, Any]:
    """計算預估月度成本"""
    if not pricing_engine:
        pricing_engine = PricingEngine()
    
    subscription_cost, overage_cost = pricing_engine.calculate_subscription_cost(
        plan_type, expected_usage
    )
    
    plan = pricing_engine.get_plan_by_type(plan_type)
    
    return {
        'plan_name': plan.name,
        'monthly_subscription': float(subscription_cost),
        'included_predictions': plan.included_predictions,
        'expected_usage': expected_usage,
        'overage_predictions': max(0, expected_usage - plan.included_predictions),
        'overage_cost': float(overage_cost),
        'total_monthly_cost': float(subscription_cost + overage_cost)
    }

if __name__ == "__main__":
    # 示例使用
    billing_engine = create_billing_engine()
    
    # 記錄使用量
    usage = billing_engine.record_api_usage(
        customer_id="customer_123",
        api_key_id="key_456",
        usage_type=UsageType.PREDICTION,
        quantity=1,
        metadata={"request_id": "req_789"}
    )
    
    print(f"使用記錄已創建: {usage.id}")
    
    # 獲取定價信息
    pricing_info = billing_engine.get_pricing_info()
    print(f"定價方案: {len(pricing_info['subscription_plans'])} 個")