"""
數據處理模組 - IEEE-CIS 詐騙檢測項目
包含數據載入、清理、預處理等功能
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """數據處理器類別"""
    
    def __init__(self):
        self.missing_threshold = 0.9  # 缺失值超過90%的特徵將被移除
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self, transaction_path: str, identity_path: str) -> pd.DataFrame:
        """載入並合併交易和身份數據"""
        logger.info("載入交易數據...")
        transaction_df = pd.read_csv(transaction_path)
        
        logger.info("載入身份數據...")
        identity_df = pd.read_csv(identity_path)
        
        logger.info("合併數據...")
        merged_df = transaction_df.merge(identity_df, on='TransactionID', how='left')
        
        logger.info(f"合併後數據形狀: {merged_df.shape}")
        return merged_df
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析缺失值情況"""
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_table = pd.DataFrame({
            '缺失數量': missing_data,
            '缺失百分比': missing_percent
        })
        
        missing_table = missing_table[missing_table['缺失數量'] > 0].sort_values('缺失數量', ascending=False)
        return missing_table
    
    def remove_high_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """移除高缺失率特徵"""
        missing_percent = df.isnull().sum() / len(df)
        high_missing_features = missing_percent[missing_percent > self.missing_threshold].index.tolist()
        
        logger.info(f"移除 {len(high_missing_features)} 個高缺失率特徵 (>{self.missing_threshold*100}%)")
        return df.drop(columns=high_missing_features)
    
    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """識別數值型和類別型特徵"""
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # 移除目標變數和ID
        if 'isFraud' in numerical_features:
            numerical_features.remove('isFraud')
        if 'TransactionID' in numerical_features:
            numerical_features.remove('TransactionID')
            
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        logger.info(f"數值型特徵: {len(numerical_features)} 個")
        logger.info(f"類別型特徵: {len(categorical_features)} 個")
        
        return numerical_features, categorical_features
    
    def handle_outliers(self, df: pd.DataFrame, features: List[str], method: str = 'iqr') -> pd.DataFrame:
        """處理異常值"""
        df_cleaned = df.copy()
        
        for feature in features:
            if method == 'iqr':
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 將異常值設為邊界值
                df_cleaned[feature] = df_cleaned[feature].clip(lower_bound, upper_bound)
        
        logger.info(f"完成 {len(features)} 個特徵的異常值處理")
        return df_cleaned
    
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本預處理流程"""
        # 1. 移除高缺失率特徵
        df_processed = self.remove_high_missing_features(df)
        
        # 2. 識別特徵類型
        numerical_features, categorical_features = self.identify_feature_types(df_processed)
        
        # 3. 處理數值型特徵的異常值
        if numerical_features:
            df_processed = self.handle_outliers(df_processed, numerical_features)
        
        return df_processed

def load_and_preprocess_data(transaction_path: str, identity_path: str) -> pd.DataFrame:
    """便捷函數：載入並預處理數據"""
    processor = DataProcessor()
    
    # 載入數據
    df = processor.load_data(transaction_path, identity_path)
    
    # 基本預處理
    df_processed = processor.basic_preprocessing(df)
    
    return df_processed

if __name__ == "__main__":
    # 測試代碼
    processor = DataProcessor()
    print("數據處理模組已載入完成！") 