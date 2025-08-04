"""
內存優化模組測試 - IEEE-CIS 詐騙檢測項目
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from unittest.mock import patch, MagicMock
import psutil
import gc

sys.path.append('../src')

from src.memory_optimizer import (
    MemoryProfiler, ChunkProcessor, MemoryEfficientOperations,
    DataFrameStreamer, memory_monitor, optimize_memory_usage,
    check_memory_requirements, suggest_chunk_size
)
from src.exceptions import InsufficientMemoryError, DataTooLargeError, MemoryError
from src.config import ConfigManager


class TestMemoryProfiler:
    """測試內存分析器"""
    
    def test_get_memory_usage(self):
        """測試獲取內存使用情況"""
        memory_info = MemoryProfiler.get_memory_usage()
        
        # 檢查返回結果結構
        expected_keys = ['rss_gb', 'vms_gb', 'percent', 'available_gb']
        for key in expected_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
            assert memory_info[key] >= 0
        
        # 檢查值的合理性
        assert 0 <= memory_info['percent'] <= 100
        assert memory_info['rss_gb'] > 0  # 進程應該使用一些內存
        assert memory_info['available_gb'] > 0  # 應該有可用內存
    
    def test_estimate_dataframe_memory(self):
        """測試估算DataFrame內存使用量"""
        # 創建測試DataFrame
        test_df = pd.DataFrame({
            'int_col': range(1000),
            'float_col': np.random.randn(1000),
            'str_col': ['test_string'] * 1000
        })
        
        estimated_memory = MemoryProfiler.estimate_dataframe_memory(test_df)
        
        # 檢查估算結果
        assert isinstance(estimated_memory, float)
        assert estimated_memory > 0
        
        # 檢查估算是否合理（應該在合理範圍內）
        assert 0.001 < estimated_memory < 1.0  # 1000行數據應該在1MB到1GB之間
    
    def test_estimate_dataframe_memory_empty(self):
        """測試空DataFrame的內存估算"""
        empty_df = pd.DataFrame()
        estimated_memory = MemoryProfiler.estimate_dataframe_memory(empty_df)
        
        assert estimated_memory == 0 or estimated_memory < 0.001  # 空DataFrame內存很小
    
    def test_get_optimal_dtypes_integers(self):
        """測試整數型數據類型優化"""
        # 創建包含不同範圍整數的DataFrame
        test_df = pd.DataFrame({
            'small_positive': [1, 2, 3, 100],  # 適合uint8
            'small_negative': [-50, -10, 0, 50],  # 適合int8
            'medium_positive': [1000, 2000, 30000],  # 適合uint16
            'large_positive': [100000, 500000, 1000000],  # 適合uint32
            'mixed_large': [-1000000, 0, 1000000]  # 適合int32
        })
        
        optimal_dtypes = MemoryProfiler.get_optimal_dtypes(test_df)
        
        # 檢查優化建議
        assert optimal_dtypes['small_positive'] == 'uint8'
        assert optimal_dtypes['small_negative'] == 'int8'
        assert optimal_dtypes['medium_positive'] == 'uint16'
        assert optimal_dtypes['large_positive'] == 'uint32'
        assert optimal_dtypes['mixed_large'] == 'int32'
    
    def test_get_optimal_dtypes_floats(self):
        """測試浮點型數據類型優化"""
        # 創建適合float32的數據
        test_df = pd.DataFrame({
            'small_float': [1.1, 2.2, 3.3, 4.4],
            'large_float': [1e20, 2e20, 3e20]  # 需要float64
        })
        
        optimal_dtypes = MemoryProfiler.get_optimal_dtypes(test_df)
        
        # 小範圍浮點數應該被建議使用float32
        assert optimal_dtypes['small_float'] == 'float32'
        # 大範圍浮點數可能不會有優化建議（保持float64）
    
    def test_get_optimal_dtypes_categorical(self):
        """測試類別型數據類型優化"""
        # 創建適合category的字符串數據
        test_df = pd.DataFrame({
            'low_cardinality': ['A', 'B', 'C', 'A', 'B', 'C'] * 100,  # 低基數，適合category
            'high_cardinality': [f'unique_{i}' for i in range(500)],  # 高基數，不適合category
            'medium_cardinality': ['Type1', 'Type2', 'Type3', 'Type4'] * 50  # 中等基數
        })
        
        optimal_dtypes = MemoryProfiler.get_optimal_dtypes(test_df)
        
        # 低基數應該被建議使用category
        assert optimal_dtypes['low_cardinality'] == 'category'
        # 高基數不應該有優化建議
        assert 'high_cardinality' not in optimal_dtypes
        # 中等基數可能被建議使用category
        assert optimal_dtypes.get('medium_cardinality') == 'category'
    
    def test_get_optimal_dtypes_with_missing_values(self):
        """測試包含缺失值的數據類型優化"""
        test_df = pd.DataFrame({
            'with_nan': [1, 2, np.nan, 4, 5],
            'all_nan': [np.nan, np.nan, np.nan]
        })
        
        optimal_dtypes = MemoryProfiler.get_optimal_dtypes(test_df)
        
        # 包含NaN的列可能沒有優化建議
        # 全為NaN的列應該沒有優化建議
        assert 'all_nan' not in optimal_dtypes
    
    def test_optimize_dataframe_memory(self):
        """測試DataFrame內存優化"""
        # 創建可優化的測試數據
        test_df = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5] * 200,
            'category_data': ['A', 'B', 'C'] * 333 + ['A'],
            'float_data': [1.1, 2.2, 3.3] * 333 + [1.1],
            'large_int': range(1000)
        })
        
        original_memory = MemoryProfiler.estimate_dataframe_memory(test_df)
        optimized_df = MemoryProfiler.optimize_dataframe_memory(test_df)
        optimized_memory = MemoryProfiler.estimate_dataframe_memory(optimized_df)
        
        # 檢查優化結果
        assert len(optimized_df) == len(test_df)
        assert list(optimized_df.columns) == list(test_df.columns)
        
        # 內存使用應該減少（或至少不增加）
        assert optimized_memory <= original_memory
        
        # 檢查數據類型是否被優化
        assert optimized_df['small_int'].dtype in ['uint8', 'int8']
        assert optimized_df['category_data'].dtype.name == 'category'
        assert optimized_df['float_data'].dtype == 'float32'
    
    def test_optimize_dataframe_memory_with_conversion_errors(self):
        """測試內存優化中的轉換錯誤處理"""
        # 創建包含可能導致轉換錯誤的數據
        test_df = pd.DataFrame({
            'mixed_types': ['1', '2', 'not_a_number', '4'],
            'normal_int': [1, 2, 3, 4]
        })
        
        # 優化應該不會崩潰，並正確處理轉換錯誤
        optimized_df = MemoryProfiler.optimize_dataframe_memory(test_df)
        
        assert len(optimized_df) == len(test_df)
        # mixed_types可能保持原樣或轉換失敗
        # normal_int應該被優化
        assert optimized_df['normal_int'].dtype in ['uint8', 'int8']


class TestChunkProcessor:
    """測試分塊處理器"""
    
    def setup_method(self):
        """測試設置"""
        self.processor = ChunkProcessor(chunk_size=100, memory_limit_gb=1.0)
    
    def test_initialization(self):
        """測試初始化"""
        assert self.processor.chunk_size == 100
        assert self.processor.memory_limit_gb == 1.0
    
    def test_initialization_with_config(self):
        """測試使用配置初始化"""
        processor = ChunkProcessor()
        
        # 應該使用配置中的默認值
        assert processor.chunk_size > 0
        assert processor.memory_limit_gb > 0
    
    def test_read_csv_in_chunks(self):
        """測試分塊讀取CSV文件"""
        # 創建測試CSV文件
        test_data = pd.DataFrame({
            'col1': range(500),
            'col2': np.random.randn(500),
            'col3': ['test'] * 500
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            chunks = list(self.processor.read_csv_in_chunks(temp_path))
            
            # 檢查分塊結果
            assert len(chunks) > 1  # 應該被分成多個chunk
            
            # 檢查chunk大小
            for chunk in chunks[:-1]:  # 除了最後一個chunk
                assert len(chunk) == self.processor.chunk_size
            
            # 檢查最後一個chunk
            assert len(chunks[-1]) <= self.processor.chunk_size
            
            # 檢查總行數
            total_rows = sum(len(chunk) for chunk in chunks)
            assert total_rows == len(test_data)
            
            # 檢查列結構
            for chunk in chunks:
                assert list(chunk.columns) == list(test_data.columns)
        
        finally:
            os.unlink(temp_path)
    
    def test_read_csv_in_chunks_file_not_found(self):
        """測試讀取不存在的文件"""
        with pytest.raises(MemoryError, match="分塊讀取失敗"):
            list(self.processor.read_csv_in_chunks('nonexistent.csv'))
    
    def test_process_dataframe_in_chunks_small_data(self):
        """測試分塊處理小數據（直接處理）"""
        small_df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20)
        })
        
        def double_values(df):
            return df * 2
        
        result = self.processor.process_dataframe_in_chunks(small_df, double_values)
        
        # 檢查結果
        assert len(result) == len(small_df)
        assert (result['col1'] == small_df['col1'] * 2).all()
        assert (result['col2'] == small_df['col2'] * 2).all()
    
    def test_process_dataframe_in_chunks_large_data(self):
        """測試分塊處理大數據"""
        # 創建大一些的測試數據
        large_df = pd.DataFrame({
            'col1': range(1000),
            'col2': range(1000, 2000)
        })
        
        def add_one(df):
            return df + 1
        
        # 強制使用分塊處理
        processor = ChunkProcessor(chunk_size=100, memory_limit_gb=0.001)  # 很小的內存限制
        result = processor.process_dataframe_in_chunks(large_df, add_one)
        
        # 檢查結果
        assert len(result) == len(large_df)
        assert (result['col1'] == large_df['col1'] + 1).all()
        assert (result['col2'] == large_df['col2'] + 1).all()
    
    def test_process_dataframe_in_chunks_with_error(self):
        """測試分塊處理中的錯誤處理"""
        test_df = pd.DataFrame({'col1': range(200)})
        
        def failing_function(df):
            if len(df) > 50:  # 當chunk大小超過50時失敗
                raise ValueError("Processing failed")
            return df
        
        with pytest.raises(ValueError, match="Processing failed"):
            self.processor.process_dataframe_in_chunks(test_df, failing_function)
    
    def test_aggregate_in_chunks(self):
        """測試分塊聚合操作"""
        test_df = pd.DataFrame({
            'group': ['A', 'B', 'C'] * 100,
            'value1': range(300),
            'value2': range(300, 600)
        })
        
        agg_dict = {
            'value1': ['sum', 'mean'],
            'value2': ['sum', 'count']
        }
        
        result = self.processor.aggregate_in_chunks(test_df, ['group'], agg_dict)
        
        # 檢查結果結構
        assert len(result) == 3  # 3個組
        assert 'group' in result.columns
        
        # 檢查聚合結果的合理性
        assert result['group'].tolist() == ['A', 'B', 'C']
        
        # 驗證聚合計算的正確性
        expected_a = test_df[test_df['group'] == 'A']
        a_result = result[result['group'] == 'A'].iloc[0]
        
        # 檢查具體聚合值（需要根據實際列名調整）
        assert len(result.columns) > 3  # 應該有原始列加上聚合列


class TestMemoryEfficientOperations:
    """測試內存高效操作"""
    
    def test_efficient_merge_small_data(self):
        """測試小數據的高效合併"""
        left_df = pd.DataFrame({
            'key': ['A', 'B', 'C', 'D'],
            'value_left': [1, 2, 3, 4]
        })
        
        right_df = pd.DataFrame({
            'key': ['A', 'B', 'C', 'E'],
            'value_right': [10, 20, 30, 40]
        })
        
        result = MemoryEfficientOperations.efficient_merge(left_df, right_df, on='key', how='left')
        
        # 檢查合併結果
        assert len(result) == 4
        assert 'key' in result.columns
        assert 'value_left' in result.columns
        assert 'value_right' in result.columns
        
        # 檢查合併的正確性
        assert result[result['key'] == 'A']['value_right'].iloc[0] == 10
        assert pd.isna(result[result['key'] == 'D']['value_right'].iloc[0])  # 右表中沒有D
    
    def test_efficient_merge_large_data(self):
        """測試大數據的分塊合併"""
        # 創建較大的測試數據
        left_df = pd.DataFrame({
            'key': range(1000),
            'value_left': range(1000, 2000)
        })
        
        right_df = pd.DataFrame({
            'key': range(0, 1500, 2),  # 0, 2, 4, ..., 1498
            'value_right': range(2000, 2750)
        })
        
        result = MemoryEfficientOperations.efficient_merge(
            left_df, right_df, on='key', how='left', chunk_size=100
        )
        
        # 檢查合併結果
        assert len(result) == 1000
        assert 'key' in result.columns
        assert 'value_left' in result.columns
        assert 'value_right' in result.columns
    
    def test_efficient_groupby_apply(self):
        """測試內存高效的groupby apply操作"""
        test_df = pd.DataFrame({
            'group': ['A', 'B', 'C'] * 100,
            'value': range(300)
        })
        
        def custom_agg(group_df):
            return pd.Series({
                'sum_value': group_df['value'].sum(),
                'count': len(group_df),
                'mean_value': group_df['value'].mean()
            })
        
        result = MemoryEfficientOperations.efficient_groupby_apply(
            test_df, ['group'], custom_agg, chunk_size=50
        )
        
        # 檢查結果
        assert len(result) == 3  # 3個組
        assert 'group' in result.columns
        assert 'sum_value' in result.columns
        assert 'count' in result.columns
        assert 'mean_value' in result.columns
        
        # 驗證結果的正確性
        a_result = result[result['group'] == 'A'].iloc[0]
        expected_count = len(test_df[test_df['group'] == 'A'])
        assert a_result['count'] == expected_count


class TestDataFrameStreamer:
    """測試DataFrame流式處理器"""
    
    def setup_method(self):
        """測試設置"""
        self.streamer = DataFrameStreamer(chunk_size=100)
    
    def test_initialization(self):
        """測試初始化"""
        assert self.streamer.chunk_size == 100
    
    def test_stream_from_csv(self):
        """測試從CSV文件流式讀取"""
        # 創建測試CSV文件
        test_data = pd.DataFrame({
            'col1': range(500),
            'col2': np.random.randn(500)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            chunks = list(self.streamer.stream_from_csv(temp_path))
            
            # 檢查流式讀取結果
            assert len(chunks) > 1
            
            # 檢查總行數
            total_rows = sum(len(chunk) for chunk in chunks)
            assert total_rows == len(test_data)
            
            # 檢查數據完整性
            reconstructed = pd.concat(chunks, ignore_index=True)
            pd.testing.assert_frame_equal(reconstructed, test_data)
        
        finally:
            os.unlink(temp_path)
    
    def test_stream_from_csv_file_not_found(self):
        """測試流式讀取不存在的文件"""
        with pytest.raises(MemoryError, match="流式讀取失敗"):
            list(self.streamer.stream_from_csv('nonexistent.csv'))
    
    def test_stream_process_in_memory(self):
        """測試流式處理（內存模式）"""
        test_df = pd.DataFrame({
            'col1': range(500),
            'col2': range(500, 1000)
        })
        
        def multiply_by_two(df):
            return df * 2
        
        result = self.streamer.stream_process(test_df, multiply_by_two)
        
        # 檢查結果
        assert len(result) == len(test_df)
        assert (result['col1'] == test_df['col1'] * 2).all()
        assert (result['col2'] == test_df['col2'] * 2).all()
    
    def test_stream_process_to_file(self):
        """測試流式處理（文件輸出模式）"""
        test_df = pd.DataFrame({
            'col1': range(500),
            'col2': range(500, 1000)
        })
        
        def add_one(df):
            return df + 1
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            result = self.streamer.stream_process(test_df, add_one, output_path=output_path)
            
            # 檢查函數返回值
            assert result is None  # 文件模式應該返回None
            
            # 檢查輸出文件
            assert os.path.exists(output_path)
            
            # 驗證文件內容
            saved_data = pd.read_csv(output_path)
            assert len(saved_data) == len(test_df)
            assert (saved_data['col1'] == test_df['col1'] + 1).all()
            assert (saved_data['col2'] == test_df['col2'] + 1).all()
        
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMemoryMonitorDecorator:
    """測試內存監控裝飾器"""
    
    def test_memory_monitor_success(self):
        """測試內存監控成功情況"""
        @memory_monitor(threshold_gb=10.0)  # 設置較高的閾值
        def test_function(x, y):
            return x + y
        
        result = test_function(1, 2)
        assert result == 3
    
    def test_memory_monitor_insufficient_memory(self):
        """測試內存不足情況"""
        @memory_monitor(threshold_gb=0.001)  # 設置極低的閾值
        def test_function():
            return "should not execute"
        
        with pytest.raises(InsufficientMemoryError):
            test_function()
    
    @patch('src.memory_optimizer.MemoryProfiler.get_memory_usage')
    def test_memory_monitor_memory_increase_tracking(self, mock_memory):
        """測試內存增長追踪"""
        # 模擬內存使用情況
        mock_memory.side_effect = [
            {'rss_gb': 1.0, 'available_gb': 5.0},  # 執行前
            {'rss_gb': 1.5, 'available_gb': 4.5}   # 執行後
        ]
        
        @memory_monitor(threshold_gb=10.0)
        def test_function():
            return "executed"
        
        result = test_function()
        assert result == "executed"
        assert mock_memory.call_count == 2
    
    def test_memory_monitor_with_memory_error(self):
        """測試內存錯誤處理"""
        @memory_monitor(threshold_gb=10.0)
        def test_function():
            raise MemoryError("內存不足")
        
        with pytest.raises(MemoryError, match="內存不足"):
            test_function()
    
    def test_memory_monitor_default_threshold(self):
        """測試默認閾值"""
        @memory_monitor()  # 不指定閾值，使用配置中的默認值
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"


class TestUtilityFunctions:
    """測試工具函數"""
    
    def test_optimize_memory_usage_function(self):
        """測試便捷內存優化函數"""
        test_df = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],
            'category_data': ['A', 'B', 'C', 'A', 'B']
        })
        
        optimized_df = optimize_memory_usage(test_df)
        
        # 檢查優化結果
        assert len(optimized_df) == len(test_df)
        assert list(optimized_df.columns) == list(test_df.columns)
        
        # 檢查數據類型優化
        assert optimized_df['small_int'].dtype in ['uint8', 'int8']
        assert optimized_df['category_data'].dtype.name == 'category'
    
    def test_check_memory_requirements_sufficient(self):
        """測試內存需求檢查（充足）"""
        small_df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20)
        })
        
        result = check_memory_requirements(small_df, "test_operation")
        assert result == True
    
    @patch('src.memory_optimizer.MemoryProfiler.get_memory_usage')
    def test_check_memory_requirements_insufficient(self, mock_memory):
        """測試內存需求檢查（不足）"""
        # 模擬可用內存很少的情況
        mock_memory.return_value = {'available_gb': 0.001}
        
        test_df = pd.DataFrame({
            'col1': range(1000),
            'col2': range(1000, 2000)
        })
        
        result = check_memory_requirements(test_df, "large_operation")
        assert result == False
    
    def test_suggest_chunk_size_small_data(self):
        """測試建議分塊大小（小數據）"""
        small_df = pd.DataFrame({
            'col1': range(100),
            'col2': range(100, 200)
        })
        
        suggested_size = suggest_chunk_size(small_df, target_memory_gb=0.1)
        
        assert isinstance(suggested_size, int)
        assert 1000 <= suggested_size <= 100000  # 應該在合理範圍內
    
    def test_suggest_chunk_size_large_data(self):
        """測試建議分塊大小（大數據）"""
        # 創建內存使用較大的DataFrame
        large_df = pd.DataFrame({
            'col1': range(10000),
            'col2': np.random.randn(10000),
            'col3': ['long_string_data'] * 10000
        })
        
        suggested_size = suggest_chunk_size(large_df, target_memory_gb=0.1)
        
        assert isinstance(suggested_size, int)
        assert 1000 <= suggested_size <= 100000
        
        # 對於大數據，建議的chunk size應該相對較小
        assert suggested_size < len(large_df)


class TestMemoryOptimizerIntegration:
    """測試內存優化器集成功能"""
    
    def test_complete_memory_optimization_workflow(self):
        """測試完整的內存優化工作流程"""
        # 1. 創建測試數據
        test_df = pd.DataFrame({
            'id': range(1000),
            'category': ['A', 'B', 'C', 'D'] * 250,
            'value': np.random.randn(1000),
            'amount': np.random.randint(1, 1000, 1000),
            'flag': np.random.choice([0, 1], 1000)
        })
        
        # 2. 檢查初始內存使用
        initial_memory = MemoryProfiler.estimate_dataframe_memory(test_df)
        
        # 3. 優化內存使用
        optimized_df = optimize_memory_usage(test_df)
        optimized_memory = MemoryProfiler.estimate_dataframe_memory(optimized_df)
        
        # 4. 檢查內存需求
        memory_sufficient = check_memory_requirements(optimized_df, "processing")
        
        # 5. 如果需要，建議分塊大小
        chunk_size = suggest_chunk_size(optimized_df, target_memory_gb=0.1)
        
        # 驗證整個流程
        assert optimized_memory <= initial_memory  # 內存應該減少或持平
        assert isinstance(memory_sufficient, bool)
        assert isinstance(chunk_size, int)
        assert chunk_size > 0
        
        # 檢查數據完整性
        assert len(optimized_df) == len(test_df)
        assert list(optimized_df.columns) == list(test_df.columns)
    
    def test_large_dataset_processing_simulation(self):
        """測試大數據集處理模擬"""
        # 創建較大的測試數據
        large_df = pd.DataFrame({
            'feature1': np.random.randn(5000),
            'feature2': np.random.randint(0, 100, 5000),
            'category': np.random.choice(['A', 'B', 'C'], 5000),
            'target': np.random.choice([0, 1], 5000)
        })
        
        # 使用ChunkProcessor處理大數據
        processor = ChunkProcessor(chunk_size=1000)
        
        def processing_function(chunk_df):
            # 模擬一些處理操作
            chunk_df['feature1_squared'] = chunk_df['feature1'] ** 2
            chunk_df['feature2_normalized'] = chunk_df['feature2'] / 100.0
            return chunk_df
        
        processed_df = processor.process_dataframe_in_chunks(large_df, processing_function)
        
        # 驗證處理結果
        assert len(processed_df) == len(large_df)
        assert 'feature1_squared' in processed_df.columns
        assert 'feature2_normalized' in processed_df.columns
        
        # 檢查處理的正確性
        assert np.allclose(processed_df['feature1_squared'], processed_df['feature1'] ** 2)
        assert np.allclose(processed_df['feature2_normalized'], processed_df['feature2'] / 100.0)
    
    def test_streaming_csv_processing(self):
        """測試流式CSV處理"""
        # 創建測試CSV數據
        test_data = pd.DataFrame({
            'id': range(1000),
            'value': np.random.randn(1000),
            'category': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # 使用流式處理器
            streamer = DataFrameStreamer(chunk_size=200)
            
            def transform_function(chunk_df):
                chunk_df['value_doubled'] = chunk_df['value'] * 2
                return chunk_df
            
            # 從CSV讀取並處理，輸出到新CSV
            total_chunks = 0
            for chunk in streamer.stream_from_csv(input_path):
                # 處理chunk
                processed_chunk = transform_function(chunk)
                
                # 寫入輸出文件
                mode = 'w' if total_chunks == 0 else 'a'
                header = total_chunks == 0
                processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
                total_chunks += 1
            
            # 驗證結果
            assert total_chunks > 1  # 應該有多個chunk
            
            # 檢查輸出文件
            result_data = pd.read_csv(output_path)
            assert len(result_data) == len(test_data)
            assert 'value_doubled' in result_data.columns
            assert np.allclose(result_data['value_doubled'], result_data['value'] * 2)
        
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestMemoryOptimizerEdgeCases:
    """測試內存優化器邊界情況"""
    
    def test_empty_dataframe_optimization(self):
        """測試空DataFrame優化"""
        empty_df = pd.DataFrame()
        
        # 應該能處理空DataFrame而不崩潰
        optimized_df = optimize_memory_usage(empty_df)
        assert len(optimized_df) == 0
        assert len(optimized_df.columns) == 0
    
    def test_single_row_dataframe(self):
        """測試單行DataFrame"""
        single_row_df = pd.DataFrame({
            'col1': [1],
            'col2': ['test'],
            'col3': [1.5]
        })
        
        optimized_df = optimize_memory_usage(single_row_df)
        assert len(optimized_df) == 1
        assert list(optimized_df.columns) == list(single_row_df.columns)
    
    def test_all_null_dataframe(self):
        """測試全為空值的DataFrame"""
        null_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [None, None, None],
            'col3': [pd.NaType(), pd.NaType(), pd.NaType()] if hasattr(pd, 'NaType') else [np.nan, np.nan, np.nan]
        })
        
        # 應該能處理全空值DataFrame
        optimized_df = optimize_memory_usage(null_df)
        assert len(optimized_df) == len(null_df)
    
    def test_extreme_values_dataframe(self):
        """測試極端值DataFrame"""
        extreme_df = pd.DataFrame({
            'tiny_int': [0, 1, 2],
            'huge_int': [1e15, 2e15, 3e15],
            'tiny_float': [1e-10, 2e-10, 3e-10],
            'huge_float': [1e100, 2e100, 3e100]
        })
        
        optimized_df = optimize_memory_usage(extreme_df)
        
        # 檢查優化結果
        assert len(optimized_df) == len(extreme_df)
        
        # 小整數應該被優化
        assert optimized_df['tiny_int'].dtype in ['uint8', 'int8']
        
        # 大數值可能保持原樣
        # 這里不強制檢查具體類型，因為優化策略可能會根據實際值調整
    
    def test_chunk_processor_single_chunk(self):
        """測試分塊處理器處理單個chunk的情況"""
        small_df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20)
        })
        
        # 設置較大的chunk size，使數據成為單個chunk
        processor = ChunkProcessor(chunk_size=1000)
        
        def identity_function(df):
            return df
        
        result = processor.process_dataframe_in_chunks(small_df, identity_function)
        
        # 結果應該與原數據相同
        pd.testing.assert_frame_equal(result, small_df)
    
    @patch('psutil.virtual_memory')
    @patch('psutil.Process')
    def test_memory_profiler_with_system_errors(self, mock_process, mock_virtual_memory):
        """測試內存分析器處理系統錯誤"""
        # 模擬psutil錯誤
        mock_process.side_effect = Exception("System error")
        mock_virtual_memory.side_effect = Exception("Memory access error")
        
        # 函數應該能處理錯誤而不崩潰
        with pytest.raises(Exception):
            MemoryProfiler.get_memory_usage()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])