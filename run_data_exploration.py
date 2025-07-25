#!/usr/bin/env python3
"""
IEEE-CIS 詐騙檢測 - 數據探索執行腳本
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

def main():
    print("🚀 開始 IEEE-CIS 詐騙檢測數據探索")
    print("=" * 50)
    
    # 檢查數據文件
    print("\n📁 檢查數據文件...")
    files = ['train_transaction.csv', 'train_identity.csv']
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)
            print(f"✅ {file}: {size:.1f} MB")
        else:
            print(f"❌ {file}: 文件不存在")
            return
    
    try:
        # 載入數據
        print("\n🔄 載入數據...")
        train_transaction = pd.read_csv('train_transaction.csv')
        train_identity = pd.read_csv('train_identity.csv')
        df = train_transaction.merge(train_identity, on='TransactionID', how='left')
        
        # 基本分析
        fraud_count = df['isFraud'].sum()
        total_count = len(df)
        fraud_rate = fraud_count / total_count
        
        print(f"\n🎯 分析結果:")
        print(f"總交易數: {total_count:,}")
        print(f"詐騙交易: {fraud_count:,}")
        print(f"詐騙比例: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
        
        print("\n✅ 數據探索完成！")
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

if __name__ == "__main__":
    main() 