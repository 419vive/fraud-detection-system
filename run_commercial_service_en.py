"""
Commercial Service Launcher - English Version - IEEE-CIS Fraud Detection API
Integrates all commercial features: billing, API management, monitoring, SLA, customer portal
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

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.billing_system import BillingEngine, UsageType, create_billing_engine
from src.api_key_management import APIKeyManager, Permission, create_api_key_manager
from src.usage_monitoring import UsageMonitoringSystem, create_usage_monitoring_system
from src.sla_management import SLAManager, create_sla_manager
from src.prediction_service import FraudDetectionService
from src.localization import get_localization_manager, t, set_language, get_error_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models
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
    """Commercial Fraud Detection API Service"""
    
    def __init__(self, default_language='en'):
        # Set default language to English
        set_language(default_language)
        
        # Initialize core services
        self.billing_engine = create_billing_engine()
        self.api_key_manager = create_api_key_manager()
        self.sla_manager = create_sla_manager()
        self.usage_monitoring = create_usage_monitoring_system(
            self.billing_engine, self.api_key_manager
        )
        
        # Initialize prediction service
        self.fraud_service = FraudDetectionService()
        
        # Create FastAPI application
        self.app = FastAPI(
            title="Fraud Detection API - Commercial Service",
            description="Enterprise-grade fraud detection API providing high-precision real-time prediction services",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security dependencies
        self.security = HTTPBearer()
        
        # Setup routes
        self._setup_routes()
        
        # Startup flag
        self.services_started = False
    
    def _setup_routes(self):
        """Setup API routes"""
        
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
                "service": "Fraud Detection API",
                "version": "2.0.0",
                "status": "Running",
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
            """Single transaction fraud prediction"""
            start_time = time.time()
            
            # Set language
            language = self._get_language_from_header(accept_language)
            set_language(language)
            
            # Validate API key
            validated_key = await self._validate_api_key(api_key, [Permission.PREDICT])
            if not validated_key:
                raise HTTPException(status_code=401, detail=get_error_message('invalid_api_key'))
            
            # Check rate limits
            allowed, limit_info = self.api_key_manager.check_rate_limit(validated_key)
            if not allowed:
                raise HTTPException(status_code=429, detail=get_error_message('rate_limit_exceeded'))
            
            try:
                # Track API call
                request_id = self.usage_monitoring.track_api_call(
                    customer_id=validated_key.customer_id,
                    api_key_id=validated_key.id,
                    endpoint="/predict",
                    usage_type=UsageType.PREDICTION,
                    quantity=1
                )
                
                # Execute prediction
                prediction_result = self.fraud_service.predict_single(
                    request.transaction_data,
                    model_version=request.model_version
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Complete API call tracking
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
                # Log error
                self.usage_monitoring.complete_api_call(request_id, 500, error_details=str(e))
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=get_error_message('prediction_error'))
        
        @self.app.post("/predict/batch")
        async def batch_predict_fraud(
            request: BatchPredictionRequest,
            api_key: str = Depends(self._get_api_key),
            accept_language: str = Header(None, alias="Accept-Language")
        ):
            """Batch transaction fraud prediction"""
            start_time = time.time()
            
            # Set language
            language = self._get_language_from_header(accept_language)
            set_language(language)
            
            # Validate API key
            validated_key = await self._validate_api_key(api_key, [Permission.BATCH_PREDICT])
            if not validated_key:
                raise HTTPException(status_code=401, detail=get_error_message('invalid_api_key'))
            
            # Check batch size limit
            if len(request.transactions) > 10000:
                raise HTTPException(status_code=400, detail=get_error_message('batch_size_limit'))
            
            try:
                # Track API call
                request_id = self.usage_monitoring.track_api_call(
                    customer_id=validated_key.customer_id,
                    api_key_id=validated_key.id,
                    endpoint="/predict/batch",
                    usage_type=UsageType.BATCH_PREDICTION,
                    quantity=len(request.transactions)
                )
                
                # Execute batch prediction
                results = self.fraud_service.predict_batch(
                    request.transactions,
                    model_version=request.model_version
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Complete API call tracking
                self.usage_monitoring.complete_api_call(request_id, 200)
                
                return {
                    "predictions": results,
                    "batch_size": len(request.transactions),
                    "processing_time_ms": processing_time,
                    "request_id": request_id
                }
                
            except Exception as e:
                self.usage_monitoring.complete_api_call(request_id, 500, error_details=str(e))
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=get_error_message('batch_prediction_error'))
        
        @self.app.get("/account/usage")
        async def get_usage_stats(api_key: str = Depends(self._get_api_key)):
            """Get usage statistics"""
            validated_key = await self._validate_api_key(api_key, [Permission.READ])
            if not validated_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            stats = self.api_key_manager.get_usage_statistics(validated_key.customer_id)
            return stats
        
        @self.app.get("/account/billing")
        async def get_billing_info(api_key: str = Depends(self._get_api_key)):
            """Get billing information"""
            validated_key = await self._validate_api_key(api_key, [Permission.READ])
            if not validated_key:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            dashboard_data = self.usage_monitoring.get_customer_usage_dashboard(
                validated_key.customer_id
            )
            return dashboard_data.get('real_time_bill', {})
        
        @self.app.post("/account/api-keys")
        async def create_api_key(
            request: APIKeyRequest,
            api_key: str = Depends(self._get_api_key)
        ):
            """Create new API key"""
            validated_key = await self._validate_api_key(api_key, [Permission.ADMIN])
            if not validated_key:
                raise HTTPException(status_code=401, detail="Invalid API key or insufficient permissions")
            
            try:
                # Convert permissions
                permissions = [Permission(p) for p in request.permissions]
                
                # Get rate limits
                customer_subscription = self.billing_engine.database.get_customer_subscription(
                    validated_key.customer_id
                )
                plan_type = customer_subscription['plan_type'].value if customer_subscription else 'free'
                rate_limits = get_default_rate_limits(plan_type)
                
                # Create API key
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
                logger.error(f"API key creation error: {e}")
                raise HTTPException(status_code=500, detail="Failed to create API key")
        
        @self.app.get("/status/sla")
        async def get_sla_status():
            """Get SLA status"""
            status = self.sla_manager.get_current_sla_status()
            return status
        
        @self.app.get("/pricing")
        async def get_pricing_info():
            """Get pricing information"""
            pricing = self.billing_engine.get_pricing_info()
            return pricing
    
    async def _get_api_key(self, authorization: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get API key from request header"""
        return authorization.credentials
    
    def _get_language_from_header(self, accept_language: str = None) -> str:
        """Get language setting from Accept-Language header"""
        if accept_language:
            # Simple language detection logic
            if 'zh' in accept_language.lower():
                return 'zh'
            elif 'en' in accept_language.lower():
                return 'en'
        return 'en'  # Default to English for this version
    
    async def _validate_api_key(self, api_key: str, required_permissions: list = None):
        """Validate API key"""
        validated_key = self.api_key_manager.authenticate_api_key(api_key)
        
        if not validated_key:
            return None
        
        # Check permissions
        if required_permissions:
            for permission in required_permissions:
                if not validated_key.has_permission(permission):
                    return None
        
        return validated_key
    
    def start_background_services(self):
        """Start background services"""
        if self.services_started:
            return
        
        logger.info("Starting background services...")
        
        # Start usage monitoring
        self.usage_monitoring.start_monitoring()
        
        # Start SLA monitoring
        self.sla_manager.start_sla_monitoring()
        
        self.services_started = True
        logger.info("All background services started")
    
    def stop_background_services(self):
        """Stop background services"""
        if not self.services_started:
            return
        
        logger.info("Stopping background services...")
        
        # Stop usage monitoring
        self.usage_monitoring.stop_monitoring()
        
        # Stop SLA monitoring
        self.sla_manager.stop_sla_monitoring()
        
        self.services_started = False
        logger.info("All background services stopped")

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run API server"""
    commercial_api = CommercialFraudDetectionAPI()
    
    # Start background services
    commercial_api.start_background_services()
    
    try:
        logger.info(f"Starting Fraud Detection API Server - http://{host}:{port}")
        logger.info(f"API Documentation: http://{host}:{port}/docs")
        
        uvicorn.run(
            commercial_api.app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Stopping API server...")
    finally:
        # Stop background services
        commercial_api.stop_background_services()

def run_customer_portal(host: str = "0.0.0.0", port: int = 8501):
    """Run customer portal"""
    logger.info(f"Starting Customer Portal - http://{host}:{port}")
    
    # Set Streamlit configuration
    sys.argv = [
        "streamlit",
        "run",
        "src/customer_portal_i18n.py",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    # Run Streamlit
    stcli.main()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fraud Detection Commercial Service - English Version")
    parser.add_argument("--service", choices=["api", "portal", "all"], default="all",
                       help="Service to start")
    parser.add_argument("--api-host", default="0.0.0.0", help="API server host")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--portal-host", default="0.0.0.0", help="Customer portal host")
    parser.add_argument("--portal-port", type=int, default=8501, help="Customer portal port")
    
    args = parser.parse_args()
    
    print("""
    🛡️  Fraud Detection API - Commercial Service (English)
    ========================================
    
    🔧 Service Features:
    ✅ Pay-per-use billing system
    ✅ API key management and authentication
    ✅ Real-time usage monitoring
    ✅ 99.9% SLA guarantee
    ✅ Customer self-service portal
    
    📊 Business Model:
    • Free Plan: 1,000 calls/month
    • Basic Plan: $99/month, 10,000 calls
    • Professional Plan: $299/month, 50,000 calls
    • Enterprise Plan: $999/month, 200,000 calls
    
    📡 Service URLs:
    """)
    
    if args.service in ["api", "all"]:
        print(f"    API Service: http://{args.api_host}:{args.api_port}")
        print(f"    API Documentation: http://{args.api_host}:{args.api_port}/docs")
    
    if args.service in ["portal", "all"]:
        print(f"    Customer Portal: http://{args.portal_host}:{args.portal_port}")
    
    print("\n" + "="*40 + "\n")
    
    if args.service == "api":
        run_api_server(args.api_host, args.api_port)
    elif args.service == "portal":
        run_customer_portal(args.portal_host, args.portal_port)
    elif args.service == "all":
        # Start both services in parallel
        import multiprocessing
        
        # API service process
        api_process = multiprocessing.Process(
            target=run_api_server,
            args=(args.api_host, args.api_port)
        )
        
        # Customer portal process
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
            logger.info("Stopping all services...")
            api_process.terminate()
            portal_process.terminate()
            api_process.join()
            portal_process.join()

if __name__ == "__main__":
    main()