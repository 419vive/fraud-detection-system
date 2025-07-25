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

logger = logging.getLogger(__name__)

class DataValidator:
    """數據驗證器類別"""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_results = {}
        self.data_profile = {}
        
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