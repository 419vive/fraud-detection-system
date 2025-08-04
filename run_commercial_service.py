"""
商業服務啟動腳本 - IEEE-CIS 詐騙檢測API
整合所有商業功能：計費、API管理、監控、SLA、客戶門戶
"""

import asyncio
import threading
import time
import logging
from datetime import datetime
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import streamlit.web.cli as stcli
import sys
import os

# 添加src目錄到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.billing_system import BillingEngine, UsageType, create_billing_engine
from src.api_key_management import APIKeyManager, Permission, create_api_key_manager
from src.usage_monitoring import UsageMonitoringSystem, create_usage_monitoring_system
from src.sla_management import SLAManager, create_sla_manager
from src.prediction_service import FraudDetectionService
from src.localization import get_localization_manager, t, set_language, get_error_message

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic 模型
class PredictionRequest(BaseModel):
    transaction_data: Dict[str, Any]
    model_version: str = "v1"

class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_score: float
    confidence: float
    model_version: str
    processing_time_ms: float
    request_id: str

class BatchPredictionRequest(BaseModel):
    transactions: list[Dict[str, Any]]
    model_version: str = "v1"

class APIKeyRequest(BaseModel):
    name: str
    permissions: list[str]
    expires_in_days: int = 365

class CommercialFraudDetectionAPI:
    """商業化詐騙檢測API服務"""
    
    def __init__(self):
        # 初始化核心服務
        self.billing_engine = create_billing_engine()
        self.api_key_manager = create_api_key_manager()
        self.sla_manager = create_sla_manager()
        self.usage_monitoring = create_usage_monitoring_system(
            self.billing_engine, self.api_key_manager
        )
        
        # 初始化預測服務
        self.fraud_service = FraudDetectionService()
        
        # 創建FastAPI應用
        self.app = FastAPI(
            title="詐騙檢測API - 商業服務",
            description="企業級詐騙檢測API，提供高精度實時預測服務",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # 設置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 安全依賴
        self.security = HTTPBearer()
        
        # 設置路由
        self._setup_routes()
        
        # 啟動標記
        self.services_started = False
    
    def _setup_routes(self):
        """設置API路由"""
        
        @self.app.middleware("http")
        async def add_process_time_header(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        @self.app.get("/")
        async def root():
            return {
                "service": "詐騙檢測API",
                "version": "2.0.0",
                "status": "運行中",
                "documentation": "/docs"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": {
                    "billing": "active",
                    "api_keys": "active",
                    "monitoring": "active",
                    "sla": "active"
                }
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_fraud(
            request: PredictionRequest,
            api_key: str = Depends(self._get_api_key),
            accept_language: str = Header(None, alias="Accept-Language")
        ):
            """單筆交易詐騙預測"""
            start_time = time.time()
            
            # 設置語言
            language = self._get_language_from_header(accept_language)
            set_language(language)
            
            # 驗證API密鑰
            validated_key = await self._validate_api_key(api_key, [Permission.PREDICT])
            if not validated_key:
                raise HTTPException(status_code=401, detail=get_error_message('invalid_api_key'))
            
            # 檢查速率限制
            allowed, limit_info = self.api_key_manager.check_rate_limit(validated_key)
            if not allowed:
                raise HTTPException(status_code=429, detail=get_error_message('rate_limit_exceeded'))
            
            try:
                # 追蹤API調用
                request_id = self.usage_monitoring.track_api_call(
                    customer_id=validated_key.customer_id,
                    api_key_id=validated_key.id,
                    endpoint="/predict",
                    usage_type=UsageType.PREDICTION,
                    quantity=1
                )
                
                # 執行預測
                prediction_result = self.fraud_service.predict_single(
                    request.transaction_data,
                    model_version=request.model_version
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # 完成API調用追蹤
                self.usage_monitoring.complete_api_call(request_id, 200)
                
                return PredictionResponse(
                    fraud_probability=prediction_result['fraud_probability'],
                    risk_score=prediction_result['risk_score'],
                    confidence=prediction_result['confidence'],
                    model_version=request.model_version,
                    processing_time_ms=processing_time,
                    request_id=request_id
                )
                
            except Exception as e:
                # 記錄錯誤
                self.usage_monitoring.complete_api_call(request_id, 500, error_details=str(e))
                logger.error(f"預測錯誤: {e}")
                raise HTTPException(status_code=500, detail=get_error_message('prediction_error'))
        
        @self.app.post("/predict/batch")
        async def batch_predict_fraud(
            request: BatchPredictionRequest,
            api_key: str = Depends(self._get_api_key),
            accept_language: str = Header(None, alias="Accept-Language")
        ):
            """批量交易詐騙預測"""
            start_time = time.time()
            
            # 設置語言
            language = self._get_language_from_header(accept_language)
            set_language(language)
            
            # 驗證API密鑰
            validated_key = await self._validate_api_key(api_key, [Permission.BATCH_PREDICT])
            if not validated_key:
                raise HTTPException(status_code=401, detail=get_error_message('invalid_api_key'))
            
            # 檢查批量大小限制
            if len(request.transactions) > 10000:
                raise HTTPException(status_code=400, detail=get_error_message('batch_size_limit'))
            
            try:
                # 追蹤API調用
                request_id = self.usage_monitoring.track_api_call(
                    customer_id=validated_key.customer_id,
                    api_key_id=validated_key.id,
                    endpoint="/predict/batch",
                    usage_type=UsageType.BATCH_PREDICTION,
                    quantity=len(request.transactions)
                )
                
                # 執行批量預測
                results = self.fraud_service.predict_batch(
                    request.transactions,
                    model_version=request.model_version
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # 完成API調用追蹤
                self.usage_monitoring.complete_api_call(request_id, 200)
                
                return {
                    "predictions": results,
                    "batch_size": len(request.transactions),
                    "processing_time_ms": processing_time,
                    "request_id": request_id
                }
                
            except Exception as e:
                self.usage_monitoring.complete_api_call(request_id, 500, error_details=str(e))
                logger.error(f"批量預測錯誤: {e}")
                raise HTTPException(status_code=500, detail=get_error_message('batch_prediction_error'))
        
        @self.app.get("/account/usage")
        async def get_usage_stats(api_key: str = Depends(self._get_api_key)):
            """獲取使用統計"""
            validated_key = await self._validate_api_key(api_key, [Permission.READ])
            if not validated_key:
                raise HTTPException(status_code=401, detail="無效的API密鑰")
            
            stats = self.api_key_manager.get_usage_statistics(validated_key.customer_id)
            return stats
        
        @self.app.get("/account/billing")
        async def get_billing_info(api_key: str = Depends(self._get_api_key)):
            """獲取計費信息"""
            validated_key = await self._validate_api_key(api_key, [Permission.READ])
            if not validated_key:
                raise HTTPException(status_code=401, detail="無效的API密鑰")
            
            dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(
                validated_key.customer_id
            )
            return dashboard_data.get('real_time_bill', {})
        
        @self.app.post("/account/api-keys")
        async def create_api_key(
            request: APIKeyRequest,
            api_key: str = Depends(self._get_api_key)
        ):
            """創建新的API密鑰"""
            validated_key = await self._validate_api_key(api_key, [Permission.ADMIN])
            if not validated_key:
                raise HTTPException(status_code=401, detail="無效的API密鑰或權限不足")
            
            try:
                # 轉換權限
                permissions = [Permission(p) for p in request.permissions]
                
                # 獲取速率限制
                customer_subscription = self.billing_engine.database.get_customer_subscription(
                    validated_key.customer_id
                )
                plan_type = customer_subscription['plan_type'].value if customer_subscription else 'free'
                rate_limits = get_default_rate_limits(plan_type)
                
                # 創建API密鑰
                raw_key, new_api_key = self.api_key_manager.create_api_key(
                    customer_id=validated_key.customer_id,
                    name=request.name,
                    permissions=permissions,
                    rate_limits=rate_limits,
                    expires_in_days=request.expires_in_days
                )
                
                return {
                    "api_key": raw_key,
                    "key_id": new_api_key.id,
                    "name": new_api_key.name,
                    "expires_at": new_api_key.expires_at.isoformat() if new_api_key.expires_at else None,
                    "permissions": [p.value for p in new_api_key.permissions]
                }
                
            except Exception as e:
                logger.error(f"創建API密鑰錯誤: {e}")
                raise HTTPException(status_code=500, detail="創建API密鑰失敗")
        
        @self.app.get("/status/sla")
        async def get_sla_status():
            """獲取SLA狀態"""
            status = self.sla_manager.get_current_sla_status()
            return status
        
        @self.app.get("/pricing")
        async def get_pricing_info():
            """獲取定價信息"""
            pricing = self.billing_engine.get_pricing_info()
            return pricing
    
    async def _get_api_key(self, authorization: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """從請求頭獲取API密鑰"""
        return authorization.credentials
    
    def _get_language_from_header(self, accept_language: str = None) -> str:
        """從Accept-Language header獲取語言設置"""
        if accept_language:
            # 簡單的語言檢測邏輯
            if 'zh' in accept_language.lower():
                return 'zh'
            elif 'en' in accept_language.lower():
                return 'en'
        return 'zh'  # 默認中文
    
    async def _validate_api_key(self, api_key: str, required_permissions: list = None):
        """驗證API密鑰"""
        validated_key = self.api_key_manager.authenticate_api_key(api_key)
        
        if not validated_key:
            return None
        
        # 檢查權限
        if required_permissions:
            for permission in required_permissions:
                if not validated_key.has_permission(permission):
                    return None
        
        return validated_key
    
    def start_background_services(self):
        """啟動後台服務"""
        if self.services_started:
            return
        
        logger.info("啟動後台服務...")
        
        # 啟動使用量監控
        self.usage_monitoring.start_monitoring()
        
        # 啟動SLA監控
        self.sla_manager.start_sla_monitoring()
        
        self.services_started = True
        logger.info("所有後台服務已啟動")
    
    def stop_background_services(self):
        """停止後台服務"""
        if not self.services_started:
            return
        
        logger.info("停止後台服務...")
        
        # 停止使用量監控
        self.usage_monitoring.stop_monitoring()
        
        # 停止SLA監控
        self.sla_manager.stop_sla_monitoring()
        
        self.services_started = False
        logger.info("所有後台服務已停止")

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """運行API服務器"""
    commercial_api = CommercialFraudDetectionAPI()
    
    # 啟動後台服務
    commercial_api.start_background_services()
    
    try:
        logger.info(f"啟動詐騙檢測API服務器 - http://{host}:{port}")
        logger.info(f"API文檔地址: http://{host}:{port}/docs")
        
        uvicorn.run(
            commercial_api.app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("正在停止API服務器...")
    finally:
        # 停止後台服務
        commercial_api.stop_background_services()

def run_customer_portal(host: str = "0.0.0.0", port: int = 8501):
    """運行客戶門戶"""
    logger.info(f"啟動客戶門戶 - http://{host}:{port}")
    
    # 設置Streamlit配置
    sys.argv = [
        "streamlit",
        "run",
        "src/customer_portal.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    # 運行Streamlit
    stcli.main()

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="詐騙檢測商業服務")
    parser.add_argument("--service", choices=["api", "portal", "all"], default="all",
                       help="要啟動的服務")
    parser.add_argument("--api-host", default="0.0.0.0", help="API服務器主機")
    parser.add_argument("--api-port", type=int, default=8000, help="API服務器端口")
    parser.add_argument("--portal-host", default="0.0.0.0", help="客戶門戶主機")
    parser.add_argument("--portal-port", type=int, default=8501, help="客戶門戶端口")
    
    args = parser.parse_args()
    
    print("""
    🛡️  詐騙檢測API - 商業服務
    ========================================
    
    🔧 服務功能:
    ✅ 按次收費計費系統
    ✅ API密鑰管理和認證
    ✅ 實時使用量監控
    ✅ 99.9% SLA保證
    ✅ 客戶自助服務門戶
    
    📊 商業模式:
    • 免費方案: 1,000次/月
    • 基礎方案: $99/月, 10,000次
    • 專業方案: $299/月, 50,000次
    • 企業方案: $999/月, 200,000次
    
    📡 服務地址:
    """)
    
    if args.service in ["api", "all"]:
        print(f"    API服務: http://{args.api_host}:{args.api_port}")
        print(f"    API文檔: http://{args.api_host}:{args.api_port}/docs")
    
    if args.service in ["portal", "all"]:
        print(f"    客戶門戶: http://{args.portal_host}:{args.portal_port}")
    
    print("\n" + "="*40 + "\n")
    
    if args.service == "api":
        run_api_server(args.api_host, args.api_port)
    elif args.service == "portal":
        run_customer_portal(args.portal_host, args.portal_port)
    elif args.service == "all":
        # 並行啟動兩個服務
        import multiprocessing
        
        # API服務進程
        api_process = multiprocessing.Process(
            target=run_api_server,
            args=(args.api_host, args.api_port)
        )
        
        # 客戶門戶進程
        portal_process = multiprocessing.Process(
            target=run_customer_portal,
            args=(args.portal_host, args.portal_port)
        )
        
        try:
            api_process.start()
            portal_process.start()
            
            api_process.join()
            portal_process.join()
            
        except KeyboardInterrupt:
            logger.info("正在停止所有服務...")
            api_process.terminate()
            portal_process.terminate()
            api_process.join()
            portal_process.join()

if __name__ == "__main__":
    main()