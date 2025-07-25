# 組件標準與介面規範

## 1. 程式碼組織標準

### 1.1 目錄結構
```
fraud_detection_project/
├── src/
│   ├── data/               # 資料處理模組
│   │   ├── __init__.py
│   │   ├── loader.py       # 資料載入
│   │   ├── preprocessor.py # 資料預處理
│   │   └── validator.py    # 資料驗證
│   ├── features/           # 特徵工程
│   │   ├── __init__.py
│   │   ├── engineering.py  # 特徵生成
│   │   ├── selection.py    # 特徵選擇
│   │   └── transformers.py # 自定義轉換器
│   ├── models/             # 模型相關
│   │   ├── __init__.py
│   │   ├── base.py         # 基礎模型類別
│   │   ├── ensemble.py     # 集成方法
│   │   └── training.py     # 訓練邏輯
│   ├── evaluation/         # 評估模組
│   │   ├── __init__.py
│   │   ├── metrics.py      # 評估指標
│   │   └── validation.py   # 交叉驗證
│   └── utils/              # 通用工具
│       ├── __init__.py
│       ├── config.py       # 配置管理
│       └── logger.py       # 日誌管理
├── notebooks/              # Jupyter筆記本
├── configs/                # 配置檔案
├── tests/                  # 單元測試
├── docs/                   # 文檔
└── requirements.txt        # 依賴管理
```

### 1.2 命名規範
- **檔案**: snake_case (如 `data_loader.py`)
- **類別**: PascalCase (如 `FraudDetector`)
- **函數/變數**: snake_case (如 `load_data()`)
- **常數**: UPPER_CASE (如 `MAX_FEATURES`)

## 2. 介面標準定義

### 2.1 資料處理介面
```python
from abc import ABC, abstractmethod
import pandas as pd

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        pass
```

### 2.2 特徵工程介面
```python
class FeatureEngineer(ABC):
    @abstractmethod
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def select_features(self, df: pd.DataFrame, target: str) -> list:
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
```

### 2.3 模型介面
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> dict:
        pass
```

## 3. 配置管理標準

### 3.1 配置檔案結構 (config.yaml)
```yaml
data:
  train_transaction: "data/train_transaction.csv"
  train_identity: "data/train_identity.csv"
  test_transaction: "data/test_transaction.csv"
  test_identity: "data/test_identity.csv"

preprocessing:
  missing_threshold: 0.9
  outlier_method: "iqr"
  imputation_strategy: "median"

feature_engineering:
  time_features: true
  aggregation_features: true
  categorical_encoding: "target"

model:
  algorithm: "lightgbm"
  cross_validation_folds: 5
  hyperparameter_trials: 100

evaluation:
  primary_metric: "auc"
  threshold_optimization: true
  
output:
  model_path: "models/"
  submission_path: "submissions/"
```

### 3.2 環境變數標準
- `FRAUD_DATA_PATH`: 資料目錄路徑
- `FRAUD_MODEL_PATH`: 模型儲存路徑
- `FRAUD_LOG_LEVEL`: 日誌等級
- `FRAUD_RANDOM_SEED`: 隨機種子

## 4. 錯誤處理標準

### 4.1 自定義異常類別
```python
class FraudDetectionError(Exception):
    pass

class DataValidationError(FraudDetectionError):
    pass

class FeatureEngineeringError(FraudDetectionError):
    pass

class ModelTrainingError(FraudDetectionError):
    pass
```

### 4.2 日誌標準
- **DEBUG**: 詳細執行資訊
- **INFO**: 一般流程資訊
- **WARNING**: 警告但可繼續執行
- **ERROR**: 錯誤需要處理
- **CRITICAL**: 嚴重錯誤需停止

## 5. 測試標準

### 5.1 測試覆蓋率
- 單元測試覆蓋率 > 80%
- 整合測試覆蓋核心流程
- 效能測試包含關鍵函數

### 5.2 測試命名規範
```python
def test_data_loader_valid_file():
    # 測試資料載入器處理有效檔案
    pass

def test_feature_engineer_missing_values():
    # 測試特徵工程處理缺失值
    pass
```

## 6. 文檔標準

### 6.1 Docstring格式 (Google Style)
```python
def preprocess_transaction_data(df: pd.DataFrame, 
                              missing_threshold: float = 0.9) -> pd.DataFrame:
    """預處理交易資料
    
    Args:
        df: 原始交易資料框
        missing_threshold: 缺失值閾值，超過此比例的欄位將被刪除
        
    Returns:
        處理後的資料框
        
    Raises:
        DataValidationError: 當資料格式不正確時
    """
    pass
```

### 6.2 README標準結構
- 專案描述與目標
- 安裝與設定指南
- 使用範例
- API文檔連結
- 貢獻指南
- 授權資訊