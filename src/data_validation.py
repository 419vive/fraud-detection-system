"""
數據驗證模組 - IEEE-CIS 詐騙檢測項目
包含數據品質檢查、完整性驗證、異常檢測等功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .config import get_config
from .exceptions import DataValidationError, handle_exception

logger = logging.getLogger(__name__)

class DataValidator:
    """數據驗證器類別"""
    
    def __init__(self, config_manager=None):
        self.validation_rules = {}
        self.validation_results = {}
        self.data_profile = {}
        self.config = config_manager or get_config()
        
        # 設置可視化樣式
        plt.style.use('default')
        sns.set_palette('husl')
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def validate_data_structure(self, df: pd.DataFrame, expected_columns: List[str] = None) -> Dict[str, Any]:
        """驗證數據結構"""
        logger.info("開始驗證數據結構...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_columns': [],
            'extra_columns': [],
            'structure_valid': True
        }
        
        # 檢查預期欄位
        if expected_columns:
            missing_columns = set(expected_columns) - set(df.columns)
            extra_columns = set(df.columns) - set(expected_columns)
            
            results['missing_columns'] = list(missing_columns)
            results['extra_columns'] = list(extra_columns)
            results['structure_valid'] = len(missing_columns) == 0
            
            if missing_columns:
                logger.warning(f"缺少預期欄位: {missing_columns}")
            if extra_columns:
                logger.info(f"額外欄位: {extra_columns}")
        
        logger.info(f"數據結構驗證完成 - 形狀: {df.shape}")
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """驗證數據品質"""
        logger.info("開始驗證數據品質...")
        
        quality_results = {
            'timestamp': datetime.now().isoformat(),
            'missing_values': {},
            'duplicate_rows': 0,
            'data_completeness': 0.0,
            'column_quality': {},
            'quality_score': 0.0
        }
        
        # 缺失值分析
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        quality_results['missing_values'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'high_missing_features': missing_percentages[missing_percentages > 50].index.tolist()
        }
        
        # 重複行檢查
        quality_results['duplicate_rows'] = df.duplicated().sum()
        
        # 數據完整性
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        quality_results['data_completeness'] = ((total_cells - missing_cells) / total_cells) * 100
        
        # 各欄位品質評估
        for col in df.columns:
            col_quality = self._assess_column_quality(df[col])
            quality_results['column_quality'][col] = col_quality
        
        # 總體品質分數 (0-100)
        completeness_score = quality_results['data_completeness']
        duplicate_penalty = min(quality_results['duplicate_rows'] / len(df) * 100, 20)
        quality_results['quality_score'] = max(0, completeness_score - duplicate_penalty)
        
        logger.info(f"數據品質驗證完成 - 品質分數: {quality_results['quality_score']:.2f}/100")
        return quality_results
    
    def _assess_column_quality(self, series: pd.Series) -> Dict[str, Any]:
        """評估單個欄位的品質"""
        quality = {
            'missing_count': series.isnull().sum(),
            'missing_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'unique_percentage': (series.nunique() / len(series)) * 100,
            'data_type': str(series.dtype),
            'quality_issues': []
        }
        
        # 檢查品質問題
        if quality['missing_percentage'] > 90:
            quality['quality_issues'].append('高缺失率')
        
        if quality['unique_count'] == 1:
            quality['quality_issues'].append('常數欄位')
        
        if series.dtype == 'object' and quality['unique_count'] > len(series) * 0.8:
            quality['quality_issues'].append('可能為ID欄位')
        
        # 數值型欄位的額外檢查
        if pd.api.types.is_numeric_dtype(series):
            quality.update(self._assess_numeric_column(series))
        
        return quality
    
    def _assess_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """評估數值型欄位"""
        numeric_quality = {
            'min_value': series.min(),
            'max_value': series.max(),
            'mean_value': series.mean(),
            'std_value': series.std(),
            'outlier_count': 0,
            'zero_count': (series == 0).sum(),
            'negative_count': (series < 0).sum()
        }
        
        # 異常值檢測 (IQR方法)
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            numeric_quality['outlier_count'] = len(outliers)
            numeric_quality['outlier_percentage'] = (len(outliers) / len(series)) * 100
        except:
            numeric_quality['outlier_count'] = 0
            numeric_quality['outlier_percentage'] = 0
        
        return numeric_quality
    
    def validate_business_rules(self, df: pd.DataFrame, rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """驗證業務規則"""
        logger.info("開始驗證業務規則...")
        
        if rules is None:
            rules = self._get_default_fraud_detection_rules()
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'rules_checked': len(rules),
            'rules_passed': 0,
            'rules_failed': 0,
            'rule_results': {},
            'overall_valid': True
        }
        
        for rule_name, rule_config in rules.items():
            try:
                rule_result = self._apply_business_rule(df, rule_name, rule_config)
                validation_results['rule_results'][rule_name] = rule_result
                
                if rule_result['passed']:
                    validation_results['rules_passed'] += 1
                else:
                    validation_results['rules_failed'] += 1
                    validation_results['overall_valid'] = False
                    
            except Exception as e:
                logger.error(f"規則 {rule_name} 執行失敗: {e}")
                validation_results['rule_results'][rule_name] = {
                    'passed': False,
                    'error': str(e),
                    'violation_count': 0
                }
                validation_results['rules_failed'] += 1
                validation_results['overall_valid'] = False
        
        logger.info(f"業務規則驗證完成 - 通過: {validation_results['rules_passed']}/{validation_results['rules_checked']}")
        return validation_results
    
    def _get_default_fraud_detection_rules(self) -> Dict[str, Any]:
        """獲取詐騙檢測的預設業務規則"""
        return {
            'transaction_amount_positive': {
                'type': 'numeric_range',
                'column': 'TransactionAmt',
                'min_value': 0,
                'description': '交易金額必須為正數'
            },
            'transaction_id_unique': {
                'type': 'uniqueness',
                'column': 'TransactionID',
                'description': '交易ID必須唯一'
            },
            'fraud_label_binary': {
                'type': 'categorical_values',
                'column': 'isFraud',
                'allowed_values': [0, 1],
                'description': '詐騙標籤必須為0或1'
            },
            'transaction_dt_reasonable': {
                'type': 'numeric_range',
                'column': 'TransactionDT',
                'min_value': 0,
                'description': '交易時間戳必須合理'
            }
        }
    
    def _apply_business_rule(self, df: pd.DataFrame, rule_name: str, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """應用單個業務規則"""
        rule_type = rule_config['type']
        column = rule_config['column']
        
        if column not in df.columns:
            return {
                'passed': False,
                'violation_count': 0,
                'error': f'欄位 {column} 不存在',
                'description': rule_config.get('description', '')
            }
        
        violations = []
        
        if rule_type == 'numeric_range':
            min_val = rule_config.get('min_value', float('-inf'))
            max_val = rule_config.get('max_value', float('inf'))
            violations = df[(df[column] < min_val) | (df[column] > max_val)].index.tolist()
            
        elif rule_type == 'uniqueness':
            duplicates = df[df.duplicated(subset=[column], keep=False)]
            violations = duplicates.index.tolist()
            
        elif rule_type == 'categorical_values':
            allowed_values = rule_config['allowed_values']
            violations = df[~df[column].isin(allowed_values)].index.tolist()
            
        elif rule_type == 'not_null':
            violations = df[df[column].isnull()].index.tolist()
        
        return {
            'passed': len(violations) == 0,
            'violation_count': len(violations),
            'violation_percentage': (len(violations) / len(df)) * 100,
            'violation_indices': violations[:100],  # 只返回前100個違規記錄
            'description': rule_config.get('description', '')
        }
    
    def validate_data_distribution(self, df: pd.DataFrame, reference_stats: Dict = None) -> Dict[str, Any]:
        """驗證數據分佈"""
        logger.info("開始驗證數據分佈...")
        
        current_stats = self._calculate_distribution_stats(df)
        
        distribution_results = {
            'timestamp': datetime.now().isoformat(),
            'current_stats': current_stats,
            'distribution_shifts': {},
            'overall_shift_detected': False
        }
        
        if reference_stats:
            for column in current_stats.keys():
                if column in reference_stats:
                    shift_detected = self._detect_distribution_shift(
                        current_stats[column], 
                        reference_stats[column]
                    )
                    distribution_results['distribution_shifts'][column] = shift_detected
                    
                    if shift_detected['significant_shift']:
                        distribution_results['overall_shift_detected'] = True
        
        logger.info("數據分佈驗證完成")
        return distribution_results
    
    def _calculate_distribution_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """計算分佈統計"""
        stats = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'q25': df[column].quantile(0.25),
                    'q50': df[column].quantile(0.50),
                    'q75': df[column].quantile(0.75),
                    'skewness': df[column].skew(),
                    'kurtosis': df[column].kurtosis()
                }
            else:
                value_counts = df[column].value_counts()
                stats[column] = {
                    'unique_count': df[column].nunique(),
                    'top_values': value_counts.head(10).to_dict(),
                    'entropy': self._calculate_entropy(value_counts)
                }
        
        return stats
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """計算熵值"""
        probabilities = value_counts / value_counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _detect_distribution_shift(self, current_stats: Dict, reference_stats: Dict) -> Dict[str, Any]:
        """檢測分佈偏移"""
        shift_result = {
            'significant_shift': False,
            'shift_metrics': {},
            'shift_threshold': 0.1  # 10% 變化閾值
        }
        
        # 對數值型統計進行比較
        numeric_metrics = ['mean', 'std', 'q25', 'q50', 'q75']
        
        for metric in numeric_metrics:
            if metric in current_stats and metric in reference_stats:
                current_val = current_stats[metric]
                reference_val = reference_stats[metric]
                
                if reference_val != 0:
                    relative_change = abs(current_val - reference_val) / abs(reference_val)
                    shift_result['shift_metrics'][metric] = relative_change
                    
                    if relative_change > shift_result['shift_threshold']:
                        shift_result['significant_shift'] = True
        
        return shift_result
    
    def generate_validation_report(self, df: pd.DataFrame, output_path: str = None) -> str:
        """生成完整的驗證報告"""
        logger.info("生成驗證報告...")
        
        # 執行所有驗證
        structure_results = self.validate_data_structure(df)
        quality_results = self.validate_data_quality(df)
        business_results = self.validate_business_rules(df)
        
        # 生成報告
        report = []
        report.append("# 數據驗證報告")
        report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"數據集大小: {df.shape[0]:,} 行 × {df.shape[1]} 列")
        report.append("")
        
        # 結構驗證結果
        report.append("## 數據結構驗證")
        report.append(f"- 總行數: {structure_results['total_rows']:,}")
        report.append(f"- 總列數: {structure_results['total_columns']}")
        report.append(f"- 記憶體使用: {structure_results['memory_usage'] / 1024 / 1024:.2f} MB")
        report.append(f"- 結構有效: {'✅ 是' if structure_results['structure_valid'] else '❌ 否'}")
        report.append("")
        
        # 品質驗證結果
        report.append("## 數據品質驗證")
        report.append(f"- 品質分數: {quality_results['quality_score']:.2f}/100")
        report.append(f"- 數據完整性: {quality_results['data_completeness']:.2f}%")
        report.append(f"- 重複行數: {quality_results['duplicate_rows']:,}")
        report.append(f"- 高缺失率特徵: {len(quality_results['missing_values']['high_missing_features'])}")
        report.append("")
        
        # 業務規則驗證結果
        report.append("## 業務規則驗證")
        report.append(f"- 通過規則: {business_results['rules_passed']}/{business_results['rules_checked']}")
        report.append(f"- 整體有效: {'✅ 是' if business_results['overall_valid'] else '❌ 否'}")
        
        for rule_name, rule_result in business_results['rule_results'].items():
            status = "✅ 通過" if rule_result['passed'] else "❌ 失敗"
            violation_count = rule_result.get('violation_count', 0)
            report.append(f"  - {rule_name}: {status} (違規: {violation_count})")
        
        report.append("")
        
        # 建議
        report.append("## 建議")
        if quality_results['quality_score'] < 80:
            report.append("- ⚠️  數據品質分數較低，建議進行清理")
        if quality_results['duplicate_rows'] > 0:
            report.append("- ⚠️  發現重複行，建議移除")
        if len(quality_results['missing_values']['high_missing_features']) > 0:
            report.append("- ⚠️  部分特徵缺失率過高，考慮移除或填補")
        if not business_results['overall_valid']:
            report.append("- ⚠️  業務規則驗證失敗，請檢查數據完整性")
        
        final_report = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
            logger.info(f"驗證報告已保存至: {output_path}")
        
        return final_report
    
    def create_data_quality_dashboard(self, df: pd.DataFrame, output_path: str = None) -> str:
        """創建數據品質視覺化儀表板"""
        logger.info("生成數據品質視覺化儀表板...")
        
        # 執行驗證
        structure_results = self.validate_data_structure(df)
        quality_results = self.validate_data_quality(df)
        
        # 創建子圖
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                '缺失值分佈', '數據類型分佈',
                '數值特徵分佈', '重複值分析',
                '數據完整性', '品質分數'
            ],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 1. 缺失值分佈
        missing_data = quality_results['missing_values']['percentages']
        top_missing = dict(sorted(missing_data.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig.add_trace(
            go.Bar(x=list(top_missing.keys()), y=list(top_missing.values()),
                   name="缺失百分比", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 2. 數據類型分佈
        dtype_counts = df.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values,
                   name="數據類型"),
            row=1, col=2
        )
        
        # 3. 數值特徵分佈（選擇前5個數值特徵）
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        for i, col in enumerate(numeric_cols):
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, opacity=0.7),
                row=2, col=1
            )
        
        # 4. 重複值分析
        duplicate_info = {
            '重複行': quality_results['duplicate_rows'],
            '唯一行': len(df) - quality_results['duplicate_rows']
        }
        fig.add_trace(
            go.Bar(x=list(duplicate_info.keys()), y=list(duplicate_info.values()),
                   marker_color=['red', 'green']),
            row=2, col=2
        )
        
        # 5. 數據完整性
        completeness_by_col = []
        completeness_vals = []
        for col in df.columns[:10]:  # 只顯示前10列
            completeness = (1 - df[col].isnull().sum() / len(df)) * 100
            completeness_by_col.append(col)
            completeness_vals.append(completeness)
        
        fig.add_trace(
            go.Bar(x=completeness_by_col, y=completeness_vals,
                   marker_color='lightblue'),
            row=3, col=1
        )
        
        # 6. 品質分數指示器
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_results['quality_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "數據品質分數"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=1000,
            title_text="數據品質儀表板",
            showlegend=False
        )
        
        # 保存或顯示
        if output_path:
            fig.write_html(output_path)
            logger.info(f"數據品質儀表板已保存至: {output_path}")
            return output_path
        else:
            fig.show()
            return "dashboard_displayed"
    
    def plot_missing_value_patterns(self, df: pd.DataFrame, top_n: int = 20):
        """繪製缺失值模式圖"""
        missing_data = df.isnull().sum().sort_values(ascending=False)[:top_n]
        missing_percent = (missing_data / len(df)) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 缺失值數量
        missing_data.plot(kind='barh', ax=ax1, color='coral')
        ax1.set_title('缺失值數量 Top 20')
        ax1.set_xlabel('缺失值數量')
        
        # 缺失值百分比
        missing_percent.plot(kind='barh', ax=ax2, color='lightblue')
        ax2.set_title('缺失值百分比 Top 20')
        ax2.set_xlabel('缺失百分比 (%)')
        
        plt.tight_layout()
        plt.show()
        
        # 缺失值熱力圖
        if len(df.columns) <= 50:  # 只有在列數不太多時才顯示熱力圖
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
            plt.title('缺失值模式熱力圖')
            plt.show()
    
    def plot_correlation_analysis(self, df: pd.DataFrame, target_col: str = 'isFraud'):
        """繪製相關性分析圖"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            # 相關性矩陣
            corr_matrix = numeric_df.corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
            plt.title('特徵相關性矩陣')
            plt.tight_layout()
            plt.show()
            
            # 與目標變數的相關性
            if target_col in numeric_df.columns:
                target_corr = numeric_df.corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)[1:21]
                
                plt.figure(figsize=(10, 8))
                target_corr.plot(kind='barh', color='green', alpha=0.7)
                plt.title(f'與 {target_col} 的相關性 (Top 20)')
                plt.xlabel('絕對相關係數')
                plt.tight_layout()
                plt.show()
    
    def plot_distribution_analysis(self, df: pd.DataFrame, target_col: str = 'isFraud', max_features: int = 12):
        """繪製分佈分析圖"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col][:max_features]
        
        if len(numeric_cols) == 0:
            logger.warning("沒有數值型特徵用於分佈分析")
            return
        
        # 計算子圖布局
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 繪製分佈圖
            if target_col in df.columns:
                # 按目標變數分組繪製
                for target_val in df[target_col].unique():
                    subset = df[df[target_col] == target_val][col].dropna()
                    if len(subset) > 0:
                        ax.hist(subset, alpha=0.6, label=f'{target_col}={target_val}', bins=30)
            else:
                ax.hist(df[col].dropna(), alpha=0.7, bins=30)
            
            ax.set_title(f'{col} 分佈')
            ax.set_xlabel(col)
            ax.set_ylabel('頻率')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隱藏多餘的子圖
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_outlier_analysis(self, df: pd.DataFrame, max_features: int = 9):
        """繪製異常值分析圖"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_features]
        
        if len(numeric_cols) == 0:
            logger.warning("沒有數值型特徵用於異常值分析")
            return
        
        # 計算子圖布局
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 箱線圖檢測異常值
            df[col].dropna().plot(kind='box', ax=ax)
            ax.set_title(f'{col} 異常值檢測')
            ax.set_ylabel(col)
        
        # 隱藏多餘的子圖
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_data_profile(self, df: pd.DataFrame, output_path: str = None):
        """創建交互式數據概覽"""
        # 基本統計信息
        basic_stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
            'duplicate_rows': df.duplicated().sum()
        }
        
        # 按數據類型分組
        dtype_info = {}
        for dtype in df.dtypes.unique():
            cols_of_type = df.select_dtypes(include=[dtype]).columns.tolist()
            dtype_info[str(dtype)] = {
                'count': len(cols_of_type),
                'columns': cols_of_type[:10]  # 只顯示前10個
            }
        
        # 缺失值統計
        missing_stats = {}
        missing_counts = df.isnull().sum()
        for col in missing_counts[missing_counts > 0].index:
            missing_stats[col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_counts[col] / len(df) * 100)
            }
        
        # 數值型特徵統計
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                numeric_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'unique_count': int(df[col].nunique())
                }
        
        # 類別型特徵統計
        categorical_stats = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else 'N/A',
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        # 組合所有信息
        data_profile = {
            'generated_at': datetime.now().isoformat(),
            'basic_statistics': basic_stats,
            'data_types': dtype_info,
            'missing_values': missing_stats,
            'numeric_features': numeric_stats,
            'categorical_features': categorical_stats
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_profile, f, ensure_ascii=False, indent=2)
            logger.info(f"數據概覽已保存至: {output_path}")
        
        return data_profile
    
    def comprehensive_data_validation_report(self, df: pd.DataFrame, 
                                           target_col: str = 'isFraud',
                                           output_dir: str = 'validation_reports') -> Dict[str, str]:
        """生成綜合數據驗證報告"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("生成綜合數據驗證報告...")
        
        # 1. 基本驗證報告
        text_report_path = os.path.join(output_dir, 'data_validation_report.md')
        self.generate_validation_report(df, text_report_path)
        
        # 2. 視覺化儀表板
        dashboard_path = os.path.join(output_dir, 'data_quality_dashboard.html')
        self.create_data_quality_dashboard(df, dashboard_path)
        
        # 3. 數據概覽JSON
        profile_path = os.path.join(output_dir, 'data_profile.json')
        self.create_interactive_data_profile(df, profile_path)
        
        # 4. 生成視覺化圖表
        logger.info("生成視覺化分析圖...")
        
        # 缺失值分析
        plt.ioff()  # 關閉交互模式以避免顯示圖表
        self.plot_missing_value_patterns(df)
        plt.savefig(os.path.join(output_dir, 'missing_value_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # 相關性分析
        if target_col in df.columns:
            self.plot_correlation_analysis(df, target_col)
            plt.savefig(os.path.join(output_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close('all')
        
        # 分佈分析
        self.plot_distribution_analysis(df, target_col)
        plt.savefig(os.path.join(output_dir, 'distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # 異常值分析
        self.plot_outlier_analysis(df)
        plt.savefig(os.path.join(output_dir, 'outlier_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close('all')
        
        plt.ion()  # 重新開啟交互模式
        
        report_files = {
            'text_report': text_report_path,
            'dashboard': dashboard_path,
            'data_profile': profile_path,
            'missing_analysis': os.path.join(output_dir, 'missing_value_analysis.png'),
            'correlation_analysis': os.path.join(output_dir, 'correlation_analysis.png'),
            'distribution_analysis': os.path.join(output_dir, 'distribution_analysis.png'),
            'outlier_analysis': os.path.join(output_dir, 'outlier_analysis.png')
        }
        
        logger.info(f"綜合驗證報告已生成，文件保存在: {output_dir}")
        return report_files

def validate_fraud_detection_data(df: pd.DataFrame) -> Dict[str, Any]:
    """便捷函數：驗證詐騙檢測數據"""
    validator = DataValidator()
    
    # 預期的詐騙檢測數據欄位
    expected_columns = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt']
    
    results = {
        'structure': validator.validate_data_structure(df, expected_columns),
        'quality': validator.validate_data_quality(df),
        'business_rules': validator.validate_business_rules(df)
    }
    
    return results

if __name__ == "__main__":
    print("數據驗證模組已載入完成！")