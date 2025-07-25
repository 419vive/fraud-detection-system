"""
預測服務模組 - IEEE-CIS 詐騙檢測項目
提供REST API預測服務接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import joblib
import os
from datetime import datetime
import json

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建FastAPI應用
app = FastAPI(
    title="Fraud Detection API",
    description="IEEE-CIS 詐騙檢測預測服務",
    version="1.0.0"
)

# 添加CORS中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局變量存儲載入的模型
loaded_models = {}
feature_columns = []
scalers = {}

class TransactionData(BaseModel):
    """交易數據模型"""
    TransactionID: int
    TransactionDT: float
    TransactionAmt: float
    ProductCD: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    C9: Optional[float] = None
    C10: Optional[float] = None
    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C14: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    D10: Optional[float] = None
    D15: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    M7: Optional[str] = None
    M8: Optional[str] = None
    M9: Optional[str] = None
    V1: Optional[float] = None
    V2: Optional[float] = None
    V3: Optional[float] = None
    V4: Optional[float] = None
    V5: Optional[float] = None
    # 可以根據需要添加更多特徵

class PredictionRequest(BaseModel):
    """預測請求模型"""
    transaction: TransactionData
    model_name: Optional[str] = "lightgbm"
    return_probability: bool = True
    include_feature_importance: bool = False

class PredictionResponse(BaseModel):
    """預測響應模型"""
    transaction_id: int
    prediction: int
    probability: Optional[float] = None
    confidence_level: str
    model_used: str
    prediction_timestamp: str
    feature_importance: Optional[Dict[str, float]] = None

class BatchPredictionRequest(BaseModel):
    """批量預測請求模型"""
    transactions: List[TransactionData]
    model_name: Optional[str] = "lightgbm"
    return_probability: bool = True

class ModelInfo(BaseModel):
    """模型信息模型"""
    model_name: str
    model_type: str
    load_timestamp: str
    feature_count: int
    is_ready: bool

def load_models(models_directory: str = "models"):
    """載入所有可用模型"""
    global loaded_models, feature_columns, scalers
    
    logger.info(f"從目錄載入模型: {models_directory}")
    
    if not os.path.exists(models_directory):
        logger.warning(f"模型目錄不存在: {models_directory}")
        return
    
    # 尋找最新的模型目錄
    model_dirs = [d for d in os.listdir(models_directory) 
                  if d.startswith('fraud_detection_') and os.path.isdir(os.path.join(models_directory, d))]
    
    if not model_dirs:
        logger.warning("未找到模型目錄")
        return
    
    latest_model_dir = sorted(model_dirs)[-1]
    model_path = os.path.join(models_directory, latest_model_dir)
    
    logger.info(f"使用模型目錄: {model_path}")
    
    # 載入配置
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        available_models = config.get('models_saved', [])
    else:
        available_models = []
        for file in os.listdir(model_path):
            if file.endswith('.pkl') and not file.endswith('_scaler.pkl'):
                available_models.append(file.replace('.pkl', ''))
    
    # 載入每個模型
    for model_name in available_models:
        try:
            model_file = os.path.join(model_path, f"{model_name}.pkl")
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                loaded_models[model_name] = model
                
                # 載入縮放器（如果存在）
                scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
                if os.path.exists(scaler_file):
                    scalers[model_name] = joblib.load(scaler_file)
                
                logger.info(f"成功載入模型: {model_name}")
                
        except Exception as e:
            logger.error(f"載入模型 {model_name} 失敗: {e}")
    
    logger.info(f"總共載入 {len(loaded_models)} 個模型")

def preprocess_transaction(transaction: TransactionData) -> pd.DataFrame:
    """預處理交易數據"""
    # 轉換為字典
    data_dict = transaction.dict()
    
    # 轉換為DataFrame
    df = pd.DataFrame([data_dict])
    
    # 基本特徵工程
    if 'TransactionDT' in df.columns:
        df['hour'] = (df['TransactionDT'] / 3600) % 24
        df['day'] = (df['TransactionDT'] / (3600 * 24)) % 7
        df['is_weekend'] = (df['day'] >= 5).astype(int)
    
    if 'TransactionAmt' in df.columns:
        df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
        df['is_round_amount'] = (df['TransactionAmt'] % 1 == 0).astype(int)
    
    # 處理缺失值
    df = df.fillna(0)
    
    # 編碼類別特徵（簡單的標籤編碼）
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'TransactionID':
            df[col] = df[col].astype('category').cat.codes
    
    # 確保只返回數值型特徵
    numeric_df = df.select_dtypes(include=[np.number])
    
    return numeric_df

def get_prediction_confidence(probability: float) -> str:
    """根據概率確定信心水平"""
    if probability < 0.3:
        return "低風險"
    elif probability < 0.7:
        return "中等風險"
    else:
        return "高風險"

@app.on_event("startup")
async def startup_event():
    """應用啟動時載入模型"""
    load_models()

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "詐騙檢測API服務已啟動",
        "version": "1.0.0",
        "models_loaded": len(loaded_models),
        "available_models": list(loaded_models.keys())
    }

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(loaded_models)
    }

@app.get("/models", response_model=List[ModelInfo])
async def get_available_models():
    """獲取可用模型列表"""
    model_info = []
    
    for model_name, model in loaded_models.items():
        info = ModelInfo(
            model_name=model_name,
            model_type=str(type(model).__name__),
            load_timestamp=datetime.now().isoformat(),
            feature_count=getattr(model, 'n_features_in_', 0),
            is_ready=True
        )
        model_info.append(info)
    
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: PredictionRequest):
    """單筆交易詐騙預測"""
    
    # 檢查模型是否存在
    if request.model_name not in loaded_models:
        raise HTTPException(
            status_code=400, 
            detail=f"模型 {request.model_name} 不存在。可用模型: {list(loaded_models.keys())}"
        )
    
    try:
        # 預處理數據
        processed_data = preprocess_transaction(request.transaction)
        
        # 獲取模型
        model = loaded_models[request.model_name]
        
        # 處理特徵維度不匹配問題
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            current_features = processed_data.shape[1]
            
            if current_features < expected_features:
                # 添加缺失特徵（填充0）
                for i in range(current_features, expected_features):
                    processed_data[f'feature_{i}'] = 0
            elif current_features > expected_features:
                # 截取前N個特徵
                processed_data = processed_data.iloc[:, :expected_features]
        
        # 應用縮放器（如果存在）
        if request.model_name in scalers:
            processed_data = scalers[request.model_name].transform(processed_data)
        
        # 進行預測
        prediction = model.predict(processed_data)[0]
        
        # 獲取預測概率
        probability = None
        if request.return_probability and hasattr(model, 'predict_proba'):
            prob_array = model.predict_proba(processed_data)[0]
            probability = float(prob_array[1])  # 詐騙的概率
        
        # 獲取特徵重要性
        feature_importance = None
        if request.include_feature_importance and hasattr(model, 'feature_importances_'):
            feature_names = processed_data.columns if hasattr(processed_data, 'columns') else [f'feature_{i}' for i in range(len(model.feature_importances_))]
            feature_importance = dict(zip(feature_names, model.feature_importances_))
            # 只返回前10個最重要的特徵
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # 構建響應
        response = PredictionResponse(
            transaction_id=request.transaction.TransactionID,
            prediction=int(prediction),
            probability=probability,
            confidence_level=get_prediction_confidence(probability) if probability else "未知",
            model_used=request.model_name,
            prediction_timestamp=datetime.now().isoformat(),
            feature_importance=feature_importance
        )
        
        return response
        
    except Exception as e:
        logger.error(f"預測過程中發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

@app.post("/predict/batch")
async def predict_fraud_batch(request: BatchPredictionRequest):
    """批量交易詐騙預測"""
    
    if request.model_name not in loaded_models:
        raise HTTPException(
            status_code=400, 
            detail=f"模型 {request.model_name} 不存在。可用模型: {list(loaded_models.keys())}"
        )
    
    try:
        predictions = []
        model = loaded_models[request.model_name]
        
        for transaction in request.transactions:
            # 預處理數據
            processed_data = preprocess_transaction(transaction)
            
            # 處理特徵維度
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                current_features = processed_data.shape[1]
                
                if current_features < expected_features:
                    for i in range(current_features, expected_features):
                        processed_data[f'feature_{i}'] = 0
                elif current_features > expected_features:
                    processed_data = processed_data.iloc[:, :expected_features]
            
            # 應用縮放器
            if request.model_name in scalers:
                processed_data = scalers[request.model_name].transform(processed_data)
            
            # 預測
            prediction = model.predict(processed_data)[0]
            probability = None
            
            if request.return_probability and hasattr(model, 'predict_proba'):
                prob_array = model.predict_proba(processed_data)[0]
                probability = float(prob_array[1])
            
            prediction_result = {
                "transaction_id": transaction.TransactionID,
                "prediction": int(prediction),
                "probability": probability,
                "confidence_level": get_prediction_confidence(probability) if probability else "未知"
            }
            
            predictions.append(prediction_result)
        
        return {
            "predictions": predictions,
            "model_used": request.model_name,
            "total_transactions": len(request.transactions),
            "prediction_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"批量預測過程中發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"批量預測失敗: {str(e)}")

@app.post("/models/reload")
async def reload_models():
    """重新載入模型"""
    try:
        global loaded_models, scalers
        loaded_models.clear()
        scalers.clear()
        
        load_models()
        
        return {
            "message": "模型重新載入成功",
            "models_loaded": len(loaded_models),
            "available_models": list(loaded_models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型重新載入失敗: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("啟動詐騙檢測API服務...")
    uvicorn.run(
        "prediction_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )