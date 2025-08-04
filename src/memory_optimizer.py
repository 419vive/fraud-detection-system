"""
內存優化模組 - IEEE-CIS 詐騙檢測項目
提供大數據集的內存優化和分塊處理功能
"""

import pandas as pd
import numpy as np
import psutil
import gc
from typing import Iterator, Dict, Any, Optional, List, Callable
import logging
from functools import wraps
import warnings

from .config import get_config
from .exceptions import InsufficientMemoryError, DataTooLargeError, MemoryError

logger = logging.getLogger(__name__)

class MemoryProfiler:
    """內存分析器"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """獲取當前內存使用情況"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_gb': memory_info.rss / (1024**3),  # 物理內存
            'vms_gb': memory_info.vms / (1024**3),  # 虛擬內存
            'percent': process.memory_percent(),      # 內存使用百分比
            'available_gb': psutil.virtual_memory().available / (1024**3)
        }
    
    @staticmethod
    def estimate_dataframe_memory(df: pd.DataFrame) -> float:
        """估算DataFrame內存使用量（GB）"""
        return df.memory_usage(deep=True).sum() / (1024**3)
    
    @staticmethod
    def get_optimal_dtypes(df: pd.DataFrame) -> Dict[str, str]:
        """獲取最優數據類型建議"""
        optimal_dtypes = {}
        
        for col in df.columns:
            col_data = df[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # 數值型優化
                if pd.api.types.is_integer_dtype(col_data):
                    # 整數型優化
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    if pd.isna(min_val) or pd.isna(max_val):
                        continue
                    
                    if min_val >= 0:  # 無符號整數
                        if max_val < 256:
                            optimal_dtypes[col] = 'uint8'
                        elif max_val < 65536:
                            optimal_dtypes[col] = 'uint16'
                        elif max_val < 4294967296:
                            optimal_dtypes[col] = 'uint32'
                        else:
                            optimal_dtypes[col] = 'uint64'
                    else:  # 有符號整數
                        if min_val > -128 and max_val < 128:
                            optimal_dtypes[col] = 'int8'
                        elif min_val > -32768 and max_val < 32768:
                            optimal_dtypes[col] = 'int16'
                        elif min_val > -2147483648 and max_val < 2147483648:
                            optimal_dtypes[col] = 'int32'
                        else:
                            optimal_dtypes[col] = 'int64'
                
                elif pd.api.types.is_float_dtype(col_data):
                    # 浮點型優化
                    # 檢查是否可以使用float32而不損失精度
                    if col_data.max() < np.finfo(np.float32).max and \
                       col_data.min() > np.finfo(np.float32).min:
                        optimal_dtypes[col] = 'float32'
            
            elif pd.api.types.is_object_dtype(col_data):
                # 字符串型優化
                unique_count = col_data.nunique()
                total_count = len(col_data)
                
                # 如果唯一值較少，考慮使用category
                if unique_count / total_count < 0.5 and unique_count < 1000:
                    optimal_dtypes[col] = 'category'
        
        return optimal_dtypes
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """優化DataFrame內存使用"""
        logger.info("開始優化DataFrame內存使用...")
        
        original_memory = MemoryProfiler.estimate_dataframe_memory(df)
        optimal_dtypes = MemoryProfiler.get_optimal_dtypes(df)
        
        df_optimized = df.copy()
        
        for col, dtype in optimal_dtypes.items():
            try:
                if dtype == 'category':
                    df_optimized[col] = df_optimized[col].astype('category')
                else:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], errors='ignore')
                    df_optimized[col] = df_optimized[col].astype(dtype)
                    
            except (ValueError, OverflowError) as e:
                logger.warning(f"無法將列 {col} 轉換為 {dtype}: {e}")
                continue
        
        optimized_memory = MemoryProfiler.estimate_dataframe_memory(df_optimized)
        reduction = ((original_memory - optimized_memory) / original_memory) * 100
        
        logger.info(f"內存優化完成:")
        logger.info(f"  原始內存: {original_memory:.2f} GB")
        logger.info(f"  優化後內存: {optimized_memory:.2f} GB")
        logger.info(f"  減少: {reduction:.1f}%")
        
        return df_optimized

class ChunkProcessor:
    """分塊處理器"""
    
    def __init__(self, chunk_size: int = None, memory_limit_gb: float = None):
        config = get_config()
        self.chunk_size = chunk_size or config.system_config.chunk_size
        self.memory_limit_gb = memory_limit_gb or config.system_config.memory_limit_gb
    
    def read_csv_in_chunks(self, file_path: str, **kwargs) -> Iterator[pd.DataFrame]:
        """分塊讀取CSV文件"""
        logger.info(f"開始分塊讀取文件: {file_path}")
        
        try:
            chunk_iterator = pd.read_csv(file_path, chunksize=self.chunk_size, **kwargs)
            
            for i, chunk in enumerate(chunk_iterator):
                # 檢查內存使用
                current_memory = MemoryProfiler.get_memory_usage()
                if current_memory['rss_gb'] > self.memory_limit_gb:
                    logger.warning(f"內存使用超限: {current_memory['rss_gb']:.2f} GB")
                    gc.collect()  # 強制垃圾回收
                
                logger.debug(f"處理第 {i+1} 個chunk，大小: {len(chunk)}")
                yield chunk
                
        except Exception as e:
            raise MemoryError(f"分塊讀取失敗: {e}", "CHUNK_READ_ERROR")
    
    def process_dataframe_in_chunks(self, df: pd.DataFrame, 
                                   process_func: Callable, 
                                   **kwargs) -> pd.DataFrame:
        """分塊處理DataFrame"""
        logger.info(f"開始分塊處理DataFrame，總大小: {len(df)}")
        
        # 檢查是否需要分塊
        estimated_memory = MemoryProfiler.estimate_dataframe_memory(df)
        if estimated_memory < self.memory_limit_gb * 0.5:  # 如果小於50%限制，直接處理
            logger.info("數據較小，直接處理")
            return process_func(df, **kwargs)
        
        # 分塊處理
        chunks = np.array_split(df, max(1, len(df) // self.chunk_size))
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"處理第 {i+1}/{len(chunks)} 個chunk")
            
            try:
                processed_chunk = process_func(chunk, **kwargs)
                processed_chunks.append(processed_chunk)
                
                # 內存檢查和清理
                if i % 5 == 0:  # 每5個chunk檢查一次
                    gc.collect()
                    current_memory = MemoryProfiler.get_memory_usage()
                    logger.debug(f"內存使用: {current_memory['rss_gb']:.2f} GB")
                    
            except Exception as e:
                logger.error(f"處理chunk {i+1} 時出錯: {e}")
                raise
        
        # 合併結果
        logger.info("合併處理結果...")
        result = pd.concat(processed_chunks, ignore_index=True)
        
        # 清理
        del processed_chunks
        gc.collect()
        
        return result
    
    def aggregate_in_chunks(self, df: pd.DataFrame, 
                           group_cols: List[str], 
                           agg_dict: Dict[str, Any]) -> pd.DataFrame:
        """分塊聚合操作"""
        logger.info("開始分塊聚合操作...")
        
        def chunk_agg_func(chunk_df):
            return chunk_df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # 分塊聚合
        chunk_results = []
        chunks = np.array_split(df, max(1, len(df) // self.chunk_size))
        
        for chunk in chunks:
            chunk_result = chunk_agg_func(chunk)
            chunk_results.append(chunk_result)
        
        # 合併中間結果
        combined = pd.concat(chunk_results, ignore_index=True)
        
        # 最終聚合
        final_result = combined.groupby(group_cols).agg(agg_dict).reset_index()
        
        return final_result

class MemoryEfficientOperations:
    """內存高效操作"""
    
    @staticmethod
    def efficient_merge(left: pd.DataFrame, right: pd.DataFrame, 
                       on: str, how: str = 'left', 
                       chunk_size: int = None) -> pd.DataFrame:
        """內存高效的合併操作"""
        config = get_config()
        chunk_size = chunk_size or config.system_config.chunk_size
        
        # 檢查是否需要分塊合併
        estimated_memory = (MemoryProfiler.estimate_dataframe_memory(left) + 
                          MemoryProfiler.estimate_dataframe_memory(right))
        
        if estimated_memory < config.system_config.memory_limit_gb * 0.6:
            # 直接合併
            return left.merge(right, on=on, how=how)
        
        # 分塊合併
        logger.info("執行分塊合併操作...")
        left_chunks = np.array_split(left, max(1, len(left) // chunk_size))
        merged_chunks = []
        
        for chunk in left_chunks:
            merged_chunk = chunk.merge(right, on=on, how=how)
            merged_chunks.append(merged_chunk)
        
        result = pd.concat(merged_chunks, ignore_index=True)
        return result
    
    @staticmethod
    def efficient_groupby_apply(df: pd.DataFrame, group_cols: List[str], 
                               apply_func: Callable, 
                               chunk_size: int = None) -> pd.DataFrame:
        """內存高效的groupby apply操作"""
        config = get_config()
        chunk_size = chunk_size or config.system_config.chunk_size
        
        # 按組分塊處理
        unique_groups = df[group_cols].drop_duplicates()
        group_chunks = np.array_split(unique_groups, max(1, len(unique_groups) // 100))
        
        results = []
        for group_chunk in group_chunks:
            # 獲取當前組的數據
            mask = df.set_index(group_cols).index.isin(
                group_chunk.set_index(group_cols).index
            )
            subset = df[mask]
            
            # 應用函數
            result = subset.groupby(group_cols).apply(apply_func).reset_index()
            results.append(result)
            
            # 清理內存
            del subset, mask
            gc.collect()
        
        return pd.concat(results, ignore_index=True)

def memory_monitor(threshold_gb: float = None):
    """內存監控裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()
            threshold = threshold_gb or config.system_config.memory_limit_gb
            
            # 執行前檢查
            initial_memory = MemoryProfiler.get_memory_usage()
            if initial_memory['rss_gb'] > threshold:
                raise InsufficientMemoryError(
                    threshold, initial_memory['available_gb']
                )
            
            # 執行函數
            try:
                result = func(*args, **kwargs)
                
                # 執行後檢查
                final_memory = MemoryProfiler.get_memory_usage()
                memory_increase = final_memory['rss_gb'] - initial_memory['rss_gb']
                
                logger.info(f"函數 {func.__name__} 執行完成:")
                logger.info(f"  內存增長: {memory_increase:.2f} GB")
                logger.info(f"  當前內存: {final_memory['rss_gb']:.2f} GB")
                
                return result
                
            except MemoryError as e:
                logger.error(f"函數 {func.__name__} 內存不足: {e}")
                gc.collect()  # 嘗試釋放內存
                raise
                
        return wrapper
    return decorator

class DataFrameStreamer:
    """DataFrame流式處理器"""
    
    def __init__(self, chunk_size: int = None):
        config = get_config()
        self.chunk_size = chunk_size or config.system_config.chunk_size
    
    def stream_from_csv(self, file_path: str, **kwargs) -> Iterator[pd.DataFrame]:
        """從CSV文件流式讀取"""
        try:
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, **kwargs):
                # 內存優化
                chunk = MemoryProfiler.optimize_dataframe_memory(chunk)
                yield chunk
        except Exception as e:
            raise MemoryError(f"流式讀取失敗: {e}", "STREAM_READ_ERROR")
    
    def stream_process(self, df: pd.DataFrame, 
                      process_func: Callable,
                      output_path: str = None,
                      **kwargs) -> Optional[pd.DataFrame]:
        """流式處理大型DataFrame"""
        chunks = np.array_split(df, max(1, len(df) // self.chunk_size))
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"流式處理 chunk {i+1}/{len(chunks)}")
            
            # 處理chunk
            processed_chunk = process_func(chunk, **kwargs)
            
            if output_path:
                # 直接寫入文件，不保存在內存中
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            else:
                processed_chunks.append(processed_chunk)
            
            # 清理內存
            del chunk, processed_chunk
            if i % 5 == 0:
                gc.collect()
        
        if not output_path:
            return pd.concat(processed_chunks, ignore_index=True)
        else:
            logger.info(f"結果已保存至: {output_path}")
            return None

# 便捷函數
def optimize_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """便捷函數：優化DataFrame內存使用"""
    return MemoryProfiler.optimize_dataframe_memory(df)

def check_memory_requirements(df: pd.DataFrame, operation: str = "processing") -> bool:
    """檢查內存需求是否滿足"""
    config = get_config()
    required_memory = MemoryProfiler.estimate_dataframe_memory(df) * 2  # 估算需要2倍空間
    available_memory = MemoryProfiler.get_memory_usage()['available_gb']
    
    if required_memory > available_memory:
        logger.warning(f"{operation} 需要 {required_memory:.2f} GB，但只有 {available_memory:.2f} GB 可用")
        return False
    
    return True

def suggest_chunk_size(df: pd.DataFrame, target_memory_gb: float = 1.0) -> int:
    """建議最優分塊大小"""
    current_memory_per_row = MemoryProfiler.estimate_dataframe_memory(df) / len(df)
    suggested_chunk_size = int(target_memory_gb / current_memory_per_row)
    
    # 確保chunk_size在合理範圍內
    return max(1000, min(suggested_chunk_size, 100000))

if __name__ == "__main__":
    # 測試內存優化功能
    print("內存優化模組已載入完成！")
    
    # 顯示當前內存狀態
    memory_info = MemoryProfiler.get_memory_usage()
    print(f"當前內存使用: {memory_info['rss_gb']:.2f} GB ({memory_info['percent']:.1f}%)")
    print(f"可用內存: {memory_info['available_gb']:.2f} GB")