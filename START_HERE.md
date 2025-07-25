# 🚀 IEEE-CIS 詐騙檢測 - 快速啟動指南

## ✅ 當前狀態
- ✅ **Jupyter Lab**: 正在運行 (Port 8888)
- ✅ **依賴安裝**: 所有 Python 套件已安裝
- ✅ **數據文件**: 所有 CSV 文件已準備就緒
- ✅ **筆記本**: 01_data_exploration.ipynb 已創建並可運行

## 🌐 立即開始

### 方法 1：瀏覽器訪問
1. 打開瀏覽器
2. 訪問: **http://localhost:8888**
3. 打開 `notebooks/01_data_exploration.ipynb`
4. 開始運行代碼單元格 (Shift + Enter)

### 方法 2：如果需要 Token
```bash
# 停止當前的 Jupyter
ps aux | grep jupyter  # 找到 PID
kill <PID>             # 停止服務

# 重新啟動（會顯示 token）
jupyter lab
```

## 📊 筆記本內容
`notebooks/01_data_exploration.ipynb` 包含：
1. **庫導入**: pandas, numpy, matplotlib, seaborn
2. **數據載入**: train_transaction.csv + train_identity.csv
3. **基本分析**: 數據形狀、目標變數分析
4. **視覺化**: 詐騙率分佈圖表
5. **缺失值**: 完整的缺失值分析
6. **行動計劃**: 下一步分析方向

## 🎯 預期結果
運行完所有單元格後，您將看到：
- 📈 數據載入狀態和基本統計
- 🥧 詐騙 vs 正常交易比例圖
- 📊 缺失值分析表格
- 💡 數據集特徵洞察

## 🔧 故障排除
- **無法訪問 8888 端口**: 檢查 Jupyter 是否運行 `ps aux | grep jupyter`
- **導入錯誤**: 重新安裝依賴 `pip install -r requirements.txt`
- **數據載入失敗**: 確認 CSV 文件在正確位置

---
*開始您的 IEEE-CIS 詐騙檢測數據科學之旅！* 🎉 