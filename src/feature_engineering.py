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
import time
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# 導入內存優化工具
from .memory_optimizer import MemoryProfiler, optimize_memory_usage, memory_monitor

from .config import get_config
from .exceptions import (
    FeatureEngineeringError, FeatureCreationError, FeatureSelectionError,
    DataValidationError, handle_exception
)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """優化的特徵工程器類別"""
    
    def __init__(self, config_manager=None, enable_parallel=True, enable_caching=True):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_importance = {}
        self.config = config_manager or get_config()
        self.feature_config = self.config.feature_config
        
        # 性能優化設置
        self.enable_parallel = enable_parallel
        self.enable_caching = enable_caching
        self.feature_cache = {}
        self.n_jobs = min(mp.cpu_count(), 8)  # 限制最大進程數
        
        # 統計信息
        self.processing_times = {}
        self.feature_counts = {}
        
    @handle_exception
    @memory_monitor()
    def create_time_features(self, df: pd.DataFrame, time_col: str = 'TransactionDT') -> pd.DataFrame:
        """優化的時間特徵創建"""
        logger.info("開始創建優化的時間特徵...")
        start_time = time.time()
        
        if time_col not in df.columns:
            raise FeatureCreationError(time_col, f"時間列 {time_col} 不存在於數據中")
        
        if df[time_col].isnull().all():
            raise FeatureCreationError(time_col, f"時間列 {time_col} 完全為空")
        
        # 使用緩存
        cache_key = f"time_features_{hash(df[time_col].values.tobytes())}"
        if self.enable_caching and cache_key in self.feature_cache:
            logger.info("使用緩存的時間特徵")
            return self.feature_cache[cache_key]
        
        try:
            # 避免複製整個DataFrame，只操作需要的列
            time_series = df[time_col].values
            
            # 檢查時間值是否合理
            if np.min(time_series) < 0:
                raise FeatureCreationError(time_col, "時間戳包含負值")
            
            # 向量化時間特徵計算（比循環快得多）
            seconds_in_hour = 3600
            seconds_in_day = 3600 * 24
            seconds_in_week = 3600 * 24 * 7
            
            # 使用numpy向量化操作
            hours = (time_series / seconds_in_hour) % 24
            days = (time_series / seconds_in_day) % 7
            weeks = time_series / seconds_in_week
            
            # 創建結果DataFrame
            df_time = df.copy()
            df_time['hour'] = hours.astype(np.float32)  # 使用float32節省內存
            df_time['day'] = days.astype(np.float32)
            df_time['week'] = weeks.astype(np.float32)
            
            # 時間段分類（優化版本）
            time_bins = self.feature_config.time_bins
            time_labels = self.feature_config.time_labels
            
            # 使用numpy的digitize進行更快的分組
            time_groups = np.digitize(hours, time_bins) - 1
            time_groups = np.clip(time_groups, 0, len(time_labels) - 1)
            df_time['time_of_day'] = time_groups.astype(np.int8)  # 使用int8節省內存
            
            # 週末標識（向量化）
            df_time['is_weekend'] = (days >= 5).astype(np.int8)
            
            # 額外的時間特徵
            df_time['hour_sin'] = np.sin(2 * np.pi * hours / 24).astype(np.float32)
            df_time['hour_cos'] = np.cos(2 * np.pi * hours / 24).astype(np.float32)
            df_time['day_sin'] = np.sin(2 * np.pi * days / 7).astype(np.float32)
            df_time['day_cos'] = np.cos(2 * np.pi * days / 7).astype(np.float32)
            
            # 緩存結果
            if self.enable_caching:
                self.feature_cache[cache_key] = df_time
            
            processing_time = time.time() - start_time
            self.processing_times['time_features'] = processing_time
            self.feature_counts['time_features'] = 8  # 新增特徵數量
            
            logger.info(f"時間特徵創建完成，耗時: {processing_time:.2f}秒，新增特徵: 8個")
            return df_time
            
        except Exception as e:
            raise FeatureCreationError("time_features", f"時間特徵創建失敗: {str(e)}")
    
    @handle_exception
    def create_transaction_amount_features(self, df: pd.DataFrame, amount_col: str = 'TransactionAmt') -> pd.DataFrame:
        """創建交易金額相關特徵"""
        if amount_col not in df.columns:
            raise FeatureCreationError(amount_col, f"金額列 {amount_col} 不存在於數據中")
        
        if df[amount_col].isnull().all():
            raise FeatureCreationError(amount_col, f"金額列 {amount_col} 完全為空")
        
        try:
            df_amt = df.copy()
            
            # 檢查金額是否合理
            if (df_amt[amount_col] < 0).any():
                raise FeatureCreationError(amount_col, "交易金額包含負值")
            
            # 使用配置中的金額分組參數
            amount_bins = self.feature_config.amount_bins
            amount_labels = self.feature_config.amount_labels
            
            # 交易金額統計特徵 - 處理零值
            df_amt['TransactionAmt_log'] = np.log1p(df_amt[amount_col])
            df_amt['TransactionAmt_sqrt'] = np.sqrt(np.maximum(df_amt[amount_col], 0))
            
            # 交易金額分組
            df_amt['amt_range'] = pd.cut(df_amt[amount_col], 
                                        bins=amount_bins,
                                        labels=amount_labels)
            
            # 是否為整數金額
            df_amt['is_round_amount'] = (df_amt[amount_col] % 1 == 0).astype(int)
            
            # 是否為常見金額 (以0或5結尾)
            df_amt['is_common_amount'] = (df_amt[amount_col] % 5 == 0).astype(int)
            
            logger.info("交易金額特徵創建完成")
            return df_amt
            
        except Exception as e:
            raise FeatureCreationError("transaction_amount_features", f"交易金額特徵創建失敗: {str(e)}")
    
    @memory_monitor()
    def create_aggregation_features(self, df: pd.DataFrame, group_cols: List[str], 
                                  agg_cols: List[str], agg_funcs: List[str] = ['mean', 'std', 'count']) -> pd.DataFrame:
        """優化的聚合特徵創建"""
        logger.info("開始創建優化的聚合特徵...")
        start_time = time.time()
        
        df_agg = df.copy()
        feature_count = 0
        
        # 預篩選有效的列組合
        valid_combinations = []
        for group_col in group_cols:
            if group_col in df.columns and df[group_col].nunique() < len(df) * 0.8:  # 避免過多唯一值
                for agg_col in agg_cols:
                    if agg_col in df.columns and agg_col != group_col:
                        for func in agg_funcs:
                            # 檢查聚合函數是否適用
                            if func in ['mean', 'std', 'sum'] and not pd.api.types.is_numeric_dtype(df[agg_col]):
                                continue
                            valid_combinations.append((group_col, agg_col, func))
        
        # 並行處理聚合特徵
        if self.enable_parallel and len(valid_combinations) > 10:
            logger.info(f"使用並行處理創建 {len(valid_combinations)} 個聚合特徵")
            
            def compute_aggregation(combo):
                group_col, agg_col, func = combo
                try:
                    # 計算聚合統計
                    agg_result = df.groupby(group_col)[agg_col].agg(func)
                    feature_name = f'{group_col}_{agg_col}_{func}'
                    
                    # 映射回原數據
                    mapped_values = df[group_col].map(agg_result)
                    
                    return feature_name, mapped_values
                except Exception as e:
                    logger.warning(f"聚合特徵計算失敗 {group_col}_{agg_col}_{func}: {e}")
                    return None, None
            
            # 使用線程池進行並行計算
            with ThreadPoolExecutor(max_workers=min(4, mp.cpu_count())) as executor:
                results = list(executor.map(compute_aggregation, valid_combinations))
            
            # 添加成功計算的特徵
            for feature_name, mapped_values in results:
                if feature_name is not None and mapped_values is not None:
                    df_agg[feature_name] = mapped_values.astype(np.float32)  # 統一使用float32
                    feature_count += 1
        
        else:
            # 順序處理
            for group_col, agg_col, func in valid_combinations:
                try:
                    # 計算聚合統計
                    agg_result = df.groupby(group_col)[agg_col].agg(func)
                    feature_name = f'{group_col}_{agg_col}_{func}'
                    
                    # 映射回原數據
                    df_agg[feature_name] = df_agg[group_col].map(agg_result).astype(np.float32)
                    feature_count += 1
                    
                except Exception as e:
                    logger.warning(f"無法創建聚合特徵 {feature_name}: {e}")
        
        # 內存優化
        df_agg = optimize_memory_usage(df_agg)
        
        processing_time = time.time() - start_time
        self.processing_times['aggregation_features'] = processing_time
        self.feature_counts['aggregation_features'] = feature_count
        
        logger.info(f"聚合特徵創建完成，耗時: {processing_time:.2f}秒，新增特徵: {feature_count}個")
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
    
    @memory_monitor()
    def full_feature_engineering_pipeline(self, df: pd.DataFrame, target_col: str = 'isFraud', 
                                        enable_advanced_features: bool = True) -> pd.DataFrame:
        """優化的完整特徵工程流水線"""
        logger.info("開始執行優化的特徵工程流水線...")
        total_start_time = time.time()
        
        # 初始內存使用量
        initial_memory = MemoryProfiler.estimate_dataframe_memory(df)
        logger.info(f"初始數據內存使用量: {initial_memory:.2f} GB")
        
        df_processed = df.copy()
        
        # 階段性內存優化
        df_processed = optimize_memory_usage(df_processed)
        
        pipeline_steps = []
        
        # 1. 創建時間特徵
        if 'TransactionDT' in df_processed.columns:
            logger.info("步驟 1: 創建時間特徵")
            df_processed = self.create_time_features(df_processed)
            pipeline_steps.append('time_features')
        
        # 2. 創建交易金額特徵
        if 'TransactionAmt' in df_processed.columns:
            logger.info("步驟 2: 創建交易金額特徵")
            df_processed = self.create_transaction_amount_features(df_processed)
            pipeline_steps.append('amount_features')
        
        # 3. 處理缺失值（優化版本）
        logger.info("步驟 3: 處理缺失值")
        df_processed = self._fast_missing_value_handling(df_processed)
        pipeline_steps.append('missing_values')
        
        # 4. 創建聚合特徵（選擇性執行）
        if enable_advanced_features:
            logger.info("步驟 4: 創建高級聚合特徵")
            
            # 並行創建不同類型的聚合特徵
            if self.enable_parallel:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_card = executor.submit(self._create_optimized_card_features, df_processed)
                    future_addr = executor.submit(self._create_optimized_address_features, df_processed)
                    future_device = executor.submit(self._create_optimized_device_features, df_processed)
                    
                    # 收集結果
                    try:
                        card_features = future_card.result(timeout=120)
                        addr_features = future_addr.result(timeout=120)
                        device_features = future_device.result(timeout=120)
                        
                        # 合併特徵
                        df_processed = self._merge_feature_dataframes(
                            df_processed, [card_features, addr_features, device_features]
                        )
                    except Exception as e:
                        logger.error(f"並行聚合特徵創建失敗: {e}")
                        # 退回到順序處理
                        df_processed = self._create_aggregation_features_sequential(df_processed)
            else:
                df_processed = self._create_aggregation_features_sequential(df_processed)
            
            pipeline_steps.append('aggregation_features')
        
        # 5. 智能類別特徵編碼
        logger.info("步驟 5: 智能類別特徵編碼")
        categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col in categorical_features:
            categorical_features.remove(target_col)
        
        if categorical_features:
            df_processed = self._intelligent_categorical_encoding(df_processed, categorical_features)
            pipeline_steps.append('categorical_encoding')
        
        # 6. 特徵交互（可選）
        if enable_advanced_features:
            logger.info("步驟 6: 創建特徵交互")
            df_processed = self._create_optimized_interactions(df_processed)
            pipeline_steps.append('feature_interactions')
        
        # 7. 最終優化和清理
        logger.info("步驟 7: 最終優化")
        df_processed = self._final_optimization(df_processed, target_col)
        
        # 統計信息
        total_time = time.time() - total_start_time
        final_memory = MemoryProfiler.estimate_dataframe_memory(df_processed)
        total_features = sum(self.feature_counts.values())
        
        logger.info("特徵工程流水線執行完成")
        logger.info(f"總耗時: {total_time:.2f}秒")
        logger.info(f"內存變化: {initial_memory:.2f} GB -> {final_memory:.2f} GB")
        logger.info(f"新增特徵總數: {total_features}個")
        logger.info(f"執行步驟: {', '.join(pipeline_steps)}")
        
        return df_processed
    
    def _fast_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """快速缺失值處理"""
        start_time = time.time()
        
        # 數值型特徵：使用median（更robust且更快）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # 類別型特徵：使用眾數或'Unknown'
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)
        
        processing_time = time.time() - start_time
        self.processing_times['missing_values'] = processing_time
        
        return df
    
    def _create_optimized_card_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """優化的卡片特徵創建"""
        card_cols = [col for col in df.columns if 'card' in col.lower()]
        if not card_cols:
            return df
        
        df_result = df.copy()
        
        # 選擇最重要的卡片列進行聚合
        important_card_cols = card_cols[:3]  # 限制數量以提高性能
        
        for card_col in important_card_cols:
            if df[card_col].nunique() < len(df) * 0.8:  # 避免過多唯一值
                # 基於卡片的交易統計
                card_stats = df.groupby(card_col).agg({
                    'TransactionAmt': ['count', 'mean', 'std'],
                }).fillna(0)
                
                # 展平列名
                card_stats.columns = [f'{card_col}_{col[1]}' for col in card_stats.columns]
                
                # 映射回原數據
                for stat_col in card_stats.columns:
                    df_result[stat_col] = df_result[card_col].map(card_stats[stat_col]).astype(np.float32)
        
        return df_result
    
    def _create_optimized_address_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """優化的地址特徵創建"""
        addr_cols = [col for col in df.columns if 'addr' in col.lower()]
        if not addr_cols:
            return df
        
        df_result = df.copy()
        
        for addr_col in addr_cols[:2]:  # 限制處理數量
            if df[addr_col].nunique() < len(df) * 0.8:
                addr_stats = df.groupby(addr_col).agg({
                    'TransactionAmt': ['count', 'mean'],
                }).fillna(0)
                
                addr_stats.columns = [f'{addr_col}_{col[1]}' for col in addr_stats.columns]
                
                for stat_col in addr_stats.columns:
                    df_result[stat_col] = df_result[addr_col].map(addr_stats[stat_col]).astype(np.float32)
        
        return df_result
    
    def _create_optimized_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """優化的設備特徵創建"""
        device_cols = [col for col in df.columns if any(keyword in col.lower() 
                      for keyword in ['device', 'id_'])]
        
        if not device_cols:
            return df
        
        df_result = df.copy()
        
        # 選擇最重要的設備列
        primary_device_cols = [col for col in device_cols if col in 
                             ['DeviceType', 'DeviceInfo', 'id_30', 'id_31']][:2]
        
        for device_col in primary_device_cols:
            if device_col in df.columns and df[device_col].nunique() < len(df) * 0.8:
                try:
                    device_stats = df.groupby(device_col).agg({
                        'TransactionAmt': ['count', 'mean'],
                    }).fillna(0)
                    
                    device_stats.columns = [f'{device_col}_{col[1]}' for col in device_stats.columns]
                    
                    for stat_col in device_stats.columns:
                        df_result[stat_col] = df_result[device_col].map(device_stats[stat_col]).astype(np.float32)
                        
                except Exception as e:
                    logger.warning(f"創建 {device_col} 特徵時出錯: {e}")
        
        return df_result
    
    def _merge_feature_dataframes(self, base_df: pd.DataFrame, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """合併特徵DataFrame"""
        result_df = base_df.copy()
        
        for feature_df in feature_dfs:
            if feature_df is not None:
                # 只添加新特徵列
                new_cols = [col for col in feature_df.columns if col not in result_df.columns]
                if new_cols:
                    result_df[new_cols] = feature_df[new_cols]
        
        return result_df
    
    def _create_aggregation_features_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """順序創建聚合特徵"""
        df = self.create_card_based_features(df)
        df = self.create_address_based_features(df)
        df = self.create_device_based_features(df)
        return df
    
    def _intelligent_categorical_encoding(self, df: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """智能類別特徵編碼"""
        start_time = time.time()
        
        for feature in categorical_features:
            unique_count = df[feature].nunique()
            
            if unique_count <= 10:
                # 少量類別：使用label encoding
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
            elif unique_count <= 100:
                # 中等數量類別：使用frequency encoding
                freq_map = df[feature].value_counts().to_dict()
                df[f'{feature}_freq'] = df[feature].map(freq_map).astype(np.float32)
                # 保留原始編碼
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature].astype(str))
                self.label_encoders[feature] = le
            else:
                # 大量類別：只使用frequency encoding
                freq_map = df[feature].value_counts().to_dict()
                df[f'{feature}_freq'] = df[feature].map(freq_map).astype(np.float32)
                # 移除原始列以節省內存
                df = df.drop(columns=[feature])
        
        processing_time = time.time() - start_time
        self.processing_times['categorical_encoding'] = processing_time
        
        return df
    
    def _create_optimized_interactions(self, df: pd.DataFrame, max_interactions: int = 50) -> pd.DataFrame:
        """創建優化的特徵交互"""
        start_time = time.time()
        
        # 選擇最重要的數值特徵進行交互
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除ID類型的列
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        # 限制特徵數量以控制計算複雜度
        if len(numeric_cols) > 10:
            # 可以基於方差或其他標準選擇重要特徵
            variances = df[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(10).index.tolist()
        
        interaction_count = 0
        
        # 創建有限的特徵交互
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if interaction_count >= max_interactions:
                    break
                
                # 乘積特徵
                df[f'{col1}_x_{col2}'] = (df[col1] * df[col2]).astype(np.float32)
                
                # 比值特徵（避免除零）
                df[f'{col1}_div_{col2}'] = (df[col1] / (df[col2] + 1e-8)).astype(np.float32)
                
                interaction_count += 2
        
        processing_time = time.time() - start_time
        self.processing_times['feature_interactions'] = processing_time
        self.feature_counts['feature_interactions'] = interaction_count
        
        return df
    
    def _final_optimization(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """最終優化和清理"""
        start_time = time.time()
        
        # 移除常數特徵
        constant_features = []
        for col in df.columns:
            if col != target_col and df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            df = df.drop(columns=constant_features)
            logger.info(f"移除常數特徵: {len(constant_features)}個")
        
        # 最終內存優化
        df = optimize_memory_usage(df)
        
        # 強制垃圾回收
        gc.collect()
        
        processing_time = time.time() - start_time
        self.processing_times['final_optimization'] = processing_time
        
        return df
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """獲取流水線執行摘要"""
        return {
            'processing_times': self.processing_times,
            'feature_counts': self.feature_counts,
            'total_time': sum(self.processing_times.values()),
            'total_features_created': sum(self.feature_counts.values()),
            'cache_size': len(self.feature_cache)
        }

def engineer_features(df: pd.DataFrame, target_col: str = 'isFraud', 
                     enable_parallel: bool = True, enable_advanced: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """便捷函數：執行優化的特徵工程"""
    engineer = FeatureEngineer(enable_parallel=enable_parallel)
    
    # 執行特徵工程
    result_df = engineer.full_feature_engineering_pipeline(
        df, target_col, enable_advanced_features=enable_advanced
    )
    
    # 獲取執行摘要
    summary = engineer.get_pipeline_summary()
    
    return result_df, summary

def fast_feature_engineering(df: pd.DataFrame, target_col: str = 'isFraud') -> pd.DataFrame:
    """快速特徵工程（生產環境優化版本）"""
    logger.info("執行快速特徵工程...")
    start_time = time.time()
    
    engineer = FeatureEngineer(enable_parallel=True, enable_caching=True)
    
    # 只執行最重要的特徵工程步驟
    df_processed = df.copy()
    
    # 1. 內存優化
    df_processed = optimize_memory_usage(df_processed)
    
    # 2. 基本時間特徵
    if 'TransactionDT' in df_processed.columns:
        df_processed = engineer.create_time_features(df_processed)
    
    # 3. 基本金額特徵
    if 'TransactionAmt' in df_processed.columns:
        df_processed = engineer.create_transaction_amount_features(df_processed)
    
    # 4. 快速缺失值處理
    df_processed = engineer._fast_missing_value_handling(df_processed)
    
    # 5. 基本類別編碼
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_features:
        categorical_features.remove(target_col)
    
    for feature in categorical_features[:5]:  # 只處理前5個類別特徵
        if df_processed[feature].nunique() <= 50:
            le = LabelEncoder()
            df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
    
    # 6. 最終優化
    df_processed = optimize_memory_usage(df_processed)
    
    total_time = time.time() - start_time
    logger.info(f"快速特徵工程完成，耗時: {total_time:.2f}秒")
    
    return df_processed

if __name__ == "__main__":
    # 測試代碼
    engineer = FeatureEngineer()
    print("特徵工程模組已載入完成！") 