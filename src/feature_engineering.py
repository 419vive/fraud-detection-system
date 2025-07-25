"""
特徵工程模組 - IEEE-CIS 詐騙檢測項目
包含時間特徵、聚合特徵、交互特徵等高級特徵工程技術
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特徵工程器類別"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_time_features(self, df: pd.DataFrame, time_col: str = 'TransactionDT') -> pd.DataFrame:
        """從時間戳創建時間特徵"""
        df_time = df.copy()
        
        # TransactionDT 是從某個起始點開始的秒數
        # 我們需要將其轉換為更有意義的時間特徵
        
        # 基本時間特徵
        df_time['hour'] = (df_time[time_col] / 3600) % 24
        df_time['day'] = (df_time[time_col] / (3600 * 24)) % 7
        df_time['week'] = (df_time[time_col] / (3600 * 24 * 7))
        
        # 時間段分類
        df_time['time_of_day'] = pd.cut(df_time['hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['夜晚', '早晨', '下午', '晚上'])
        
        # 週末標識
        df_time['is_weekend'] = (df_time['day'] >= 5).astype(int)
        
        logger.info("時間特徵創建完成")
        return df_time
    
    def create_transaction_amount_features(self, df: pd.DataFrame, amount_col: str = 'TransactionAmt') -> pd.DataFrame:
        """創建交易金額相關特徵"""
        df_amt = df.copy()
        
        # 交易金額統計特徵
        df_amt['TransactionAmt_log'] = np.log1p(df_amt[amount_col])
        df_amt['TransactionAmt_sqrt'] = np.sqrt(df_amt[amount_col])
        
        # 交易金額分組
        df_amt['amt_range'] = pd.cut(df_amt[amount_col], 
                                    bins=[0, 50, 100, 500, 1000, float('inf')],
                                    labels=['小額', '中小額', '中額', '大額', '超大額'])
        
        # 是否為整數金額
        df_amt['is_round_amount'] = (df_amt[amount_col] % 1 == 0).astype(int)
        
        # 是否為常見金額 (以0或5結尾)
        df_amt['is_common_amount'] = (df_amt[amount_col] % 5 == 0).astype(int)
        
        logger.info("交易金額特徵創建完成")
        return df_amt
    
    def create_aggregation_features(self, df: pd.DataFrame, group_cols: List[str], 
                                  agg_cols: List[str], agg_funcs: List[str] = ['mean', 'std', 'count']) -> pd.DataFrame:
        """創建聚合特徵"""
        df_agg = df.copy()
        
        for group_col in group_cols:
            if group_col in df.columns:
                for agg_col in agg_cols:
                    if agg_col in df.columns:
                        for func in agg_funcs:
                            try:
                                # 計算聚合統計
                                agg_result = df.groupby(group_col)[agg_col].agg(func)
                                
                                # 創建特徵名稱
                                feature_name = f'{group_col}_{agg_col}_{func}'
                                
                                # 映射回原數據
                                df_agg[feature_name] = df_agg[group_col].map(agg_result)
                                
                            except Exception as e:
                                logger.warning(f"無法創建聚合特徵 {feature_name}: {e}")
        
        logger.info(f"聚合特徵創建完成，基於 {group_cols} 分組")
        return df_agg
    
    def create_card_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建基於卡片的聚合特徵"""
        df_card = df.copy()
        
        # 識別卡片相關欄位
        card_cols = [col for col in df.columns if 'card' in col.lower()]
        
        if not card_cols:
            logger.warning("未找到卡片相關欄位")
            return df_card
        
        logger.info(f"發現卡片欄位: {card_cols}")
        
        for card_col in card_cols:
            if df[card_col].dtype in ['object', 'int64', 'float64']:
                # 基於卡片的交易統計
                card_stats = df.groupby(card_col).agg({
                    'TransactionAmt': ['count', 'mean', 'std', 'sum', 'max', 'min'],
                    'isFraud': ['mean', 'sum'] if 'isFraud' in df.columns else ['mean']
                }).fillna(0)
                
                # 展平多層索引
                card_stats.columns = [f'{card_col}_{col[0]}_{col[1]}' for col in card_stats.columns]
                
                # 映射回原數據
                for stat_col in card_stats.columns:
                    df_card[stat_col] = df_card[card_col].map(card_stats[stat_col])
        
        logger.info("基於卡片的聚合特徵創建完成")
        return df_card
    
    def create_address_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建基於地址的聚合特徵"""
        df_addr = df.copy()
        
        # 識別地址相關欄位
        address_cols = [col for col in df.columns if 'addr' in col.lower()]
        
        if not address_cols:
            logger.warning("未找到地址相關欄位")
            return df_addr
        
        logger.info(f"發現地址欄位: {address_cols}")
        
        for addr_col in address_cols:
            if df[addr_col].dtype in ['object', 'int64', 'float64']:
                # 基於地址的交易統計
                addr_stats = df.groupby(addr_col).agg({
                    'TransactionAmt': ['count', 'mean', 'std', 'sum'],
                    'isFraud': ['mean', 'sum'] if 'isFraud' in df.columns else ['mean']
                }).fillna(0)
                
                # 展平多層索引
                addr_stats.columns = [f'{addr_col}_{col[0]}_{col[1]}' for col in addr_stats.columns]
                
                # 映射回原數據
                for stat_col in addr_stats.columns:
                    df_addr[stat_col] = df_addr[addr_col].map(addr_stats[stat_col])
        
        logger.info("基於地址的聚合特徵創建完成")
        return df_addr
    
    def create_device_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建基於設備的聚合特徵"""
        df_device = df.copy()
        
        # 識別設備相關欄位
        device_cols = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['device', 'id_', 'browser', 'os'])]
        
        if not device_cols:
            logger.warning("未找到設備相關欄位")
            return df_device
        
        logger.info(f"發現設備欄位: {device_cols[:10]}...")  # 只顯示前10個
        
        # 選擇主要的設備識別欄位
        primary_device_cols = [col for col in device_cols if col in 
                             ['DeviceType', 'DeviceInfo', 'id_30', 'id_31', 'id_33']]
        
        for device_col in primary_device_cols:
            if device_col in df.columns and df[device_col].dtype in ['object', 'int64', 'float64']:
                try:
                    # 基於設備的交易統計
                    device_stats = df.groupby(device_col).agg({
                        'TransactionAmt': ['count', 'mean', 'sum'],
                        'isFraud': ['mean'] if 'isFraud' in df.columns else ['mean']
                    }).fillna(0)
                    
                    # 展平多層索引
                    device_stats.columns = [f'{device_col}_{col[0]}_{col[1]}' for col in device_stats.columns]
                    
                    # 映射回原數據
                    for stat_col in device_stats.columns:
                        df_device[stat_col] = df_device[device_col].map(device_stats[stat_col])
                        
                except Exception as e:
                    logger.warning(f"創建 {device_col} 特徵時出錯: {e}")
        
        logger.info("基於設備的聚合特徵創建完成")
        return df_device
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """創建交互特徵"""
        df_interact = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # 數值型特徵交互
                if df[feat1].dtype in ['int64', 'float64'] and df[feat2].dtype in ['int64', 'float64']:
                    # 乘積
                    df_interact[f'{feat1}_x_{feat2}'] = df_interact[feat1] * df_interact[feat2]
                    
                    # 比值 (避免除零)
                    df_interact[f'{feat1}_div_{feat2}'] = df_interact[feat1] / (df_interact[feat2] + 1e-8)
                    
                    # 差值
                    df_interact[f'{feat1}_minus_{feat2}'] = df_interact[feat1] - df_interact[feat2]
        
        logger.info(f"交互特徵創建完成，共 {len(feature_pairs)} 對特徵")
        return df_interact
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_features: List[str], 
                                  method: str = 'label') -> pd.DataFrame:
        """編碼類別特徵"""
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature in df.columns:
                if method == 'label':
                    # Label Encoding
                    le = LabelEncoder()
                    df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                    self.label_encoders[feature] = le
                
                elif method == 'frequency':
                    # Frequency Encoding
                    freq_map = df[feature].value_counts().to_dict()
                    df_encoded[f'{feature}_freq'] = df_encoded[feature].map(freq_map)
        
        logger.info(f"類別特徵編碼完成，方法: {method}")
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """智能填補缺失值"""
        df_filled = df.copy()
        
        if strategy is None:
            strategy = {
                'numerical': 'median',
                'categorical': 'mode'
            }
        
        # 數值型特徵填補
        numerical_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_filled[col].isnull().sum() > 0:
                if strategy.get('numerical') == 'median':
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                elif strategy.get('numerical') == 'mean':
                    df_filled[col].fillna(df_filled[col].mean(), inplace=True)
                else:
                    df_filled[col].fillna(0, inplace=True)
        
        # 類別型特徵填補
        categorical_cols = df_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_filled[col].isnull().sum() > 0:
                if strategy.get('categorical') == 'mode':
                    mode_val = df_filled[col].mode()
                    if len(mode_val) > 0:
                        df_filled[col].fillna(mode_val[0], inplace=True)
                    else:
                        df_filled[col].fillna('Unknown', inplace=True)
                else:
                    df_filled[col].fillna('Unknown', inplace=True)
        
        logger.info("缺失值填補完成")
        return df_filled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 100) -> List[str]:
        """特徵選擇"""
        # 只對數值型特徵進行特徵選擇
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numerical = X[numerical_features]
        
        # 使用 SelectKBest 進行特徵選擇
        selector = SelectKBest(score_func=f_classif, k=min(k, len(numerical_features)))
        selector.fit(X_numerical, y)
        
        # 獲取選中的特徵
        selected_features = X_numerical.columns[selector.get_support()].tolist()
        
        # 保存特徵重要性
        feature_scores = dict(zip(numerical_features, selector.scores_))
        self.feature_importance = feature_scores
        
        logger.info(f"特徵選擇完成，從 {len(numerical_features)} 個特徵中選出 {len(selected_features)} 個")
        return selected_features
    
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote', 
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
        """處理類別不平衡問題"""
        logger.info(f"開始處理類別不平衡，方法: {method}")
        logger.info(f"原始數據分佈: {y.value_counts().to_dict()}")
        
        # 確保只使用數值型特徵
        X_numeric = X.select_dtypes(include=[np.number])
        
        if method == 'smote':
            # SMOTE 過採樣
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X_numeric, y)
            
        elif method == 'undersampling':
            # 隨機欠採樣
            undersampler = RandomUnderSampler(random_state=random_state)
            X_resampled, y_resampled = undersampler.fit_resample(X_numeric, y)
            
        elif method == 'smoteenn':
            # SMOTE + Edited Nearest Neighbours
            smoteenn = SMOTEENN(random_state=random_state)
            X_resampled, y_resampled = smoteenn.fit_resample(X_numeric, y)
            
        else:
            logger.warning(f"未知的平衡方法: {method}，返回原始數據")
            return X, y
        
        # 轉換回DataFrame
        X_resampled_df = pd.DataFrame(X_resampled, columns=X_numeric.columns)
        y_resampled_series = pd.Series(y_resampled)
        
        logger.info(f"平衡後數據分佈: {y_resampled_series.value_counts().to_dict()}")
        logger.info(f"數據形狀變化: {X.shape} -> {X_resampled_df.shape}")
        
        return X_resampled_df, y_resampled_series
    
    def full_feature_engineering_pipeline(self, df: pd.DataFrame, target_col: str = 'isFraud') -> pd.DataFrame:
        """完整的特徵工程流水線"""
        logger.info("開始執行完整特徵工程流水線...")
        
        df_processed = df.copy()
        
        # 1. 創建時間特徵
        if 'TransactionDT' in df_processed.columns:
            df_processed = self.create_time_features(df_processed)
        
        # 2. 創建交易金額特徵
        if 'TransactionAmt' in df_processed.columns:
            df_processed = self.create_transaction_amount_features(df_processed)
        
        # 3. 處理缺失值
        df_processed = self.handle_missing_values(df_processed)
        
        # 4. 創建聚合特徵
        df_processed = self.create_card_based_features(df_processed)
        df_processed = self.create_address_based_features(df_processed)
        df_processed = self.create_device_based_features(df_processed)
        
        # 5. 編碼類別特徵
        categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
        if target_col in categorical_features:
            categorical_features.remove(target_col)
        
        if categorical_features:
            df_processed = self.encode_categorical_features(df_processed, categorical_features)
        
        logger.info("特徵工程流水線執行完成")
        return df_processed

def engineer_features(df: pd.DataFrame, target_col: str = 'isFraud') -> pd.DataFrame:
    """便捷函數：執行特徵工程"""
    engineer = FeatureEngineer()
    return engineer.full_feature_engineering_pipeline(df, target_col)

if __name__ == "__main__":
    # 測試代碼
    engineer = FeatureEngineer()
    print("特徵工程模組已載入完成！") 