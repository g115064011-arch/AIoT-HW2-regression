# AIoT HW2：洪水機率預測（多元線性回歸）

以 Kaggle 洪水預測資料集，依照 **CRISP-DM** 流程完成多元線性回歸分析，  
包含特徵選擇、模型評估、95% 信賴區間／預測區間，以及輕量化部署。

- **學號**：7115064011
- **主程式**：[`notebooks/7115064011_hw2.ipynb`](notebooks/7115064011_hw2.ipynb)
- **延伸自 HW1**：CWA 台灣天氣預報網站（HW1 建立天氣資料的取得與展示，HW2 進一步建立預測模型）

---

## 資料集

| 項目 | 內容 |
|------|------|
| 來源 | [Kaggle Playground Series S4E5 – Regression with a Flood Prediction Dataset](https://www.kaggle.com/competitions/playground-series-s4e5) |
| 訓練資料 | 1,117,957 筆 |
| 測試資料 | 745,305 筆 |
| 特徵 | 20 個整數評分特徵（如 MonsoonIntensity、Deforestation、Urbanization 等） |
| 目標 | `FloodProbability`（連續值，0.285 ~ 0.725） |

> **注意**：本資料為 Kaggle 合成資料，模型僅作方法示範，不代表真實地區的洪水風險。

---

## 主要成果

| 模型 | 特徵數 | Validation R² | RMSE | MAE |
|------|:------:|:-------------:|:----:|:---:|
| Baseline（單純線性回歸） | 1 | 0.0368 | 0.0500 | 0.0402 |
| OLS 多元線性回歸 | 20 | 0.8449 | 0.0201 | 0.0158 |
| RidgeCV | 20 | 0.8449 | 0.0201 | 0.0158 |
| LassoCV | 20 | 0.8449 | 0.0201 | 0.0158 |

- **95% 預測區間覆蓋率**：94.61%（理論值 95%）
- **Kaggle 分數（R²）**：Public **0.84458** / Private **0.84426**

---

## CRISP-DM 流程摘要

1. **Business Understanding**：建立洪水機率預測模型，作為災前風險評估的參考工具。
2. **Data Understanding**：無缺失值；20 個特徵彼此幾乎不相關（|r| < 0.02），與目標的相關係數皆為 0.178 ~ 0.190。
3. **Data Preparation**：80/20 切分、標準化（只用訓練集 fit）；VIF 介於 1.0017 ~ 1.0037，無多重共線性。比較 SelectKBest、RFE、LassoCV 三種特徵選擇方法，R² 隨特徵數穩定上升（k=10：0.377；k=15：0.596；k=20：0.845），因此保留全部 20 個特徵。
4. **Modeling**：statsmodels OLS 與 sklearn LinearRegression、RidgeCV、LassoCV；5-fold 交叉驗證結果穩定。20 個係數幾乎相同（0.0115 ~ 0.0118），模型近似等權重加總。
5. **Evaluation**：實際值 vs 預測值圖、95% 信賴區間與預測區間圖、殘差診斷。Durbin-Watson = 1.999（無自相關）；殘差右偏、厚尾，顯示存在線性模型無法捕捉的非線性關係。
6. **Deployment**：以全部訓練資料重新訓練，輸出 `model/model_params.json`；`predict_flood()` 只依賴 numpy 即可回傳預測值與 95% 預測區間，並產生 Kaggle 提交檔。

---

## 專案結構

```
├── index.html                       # 網頁平台主頁（整合 Problem 1, Problem 2, 自訂資料集）
├── style.css                        # 視覺主題樣式表（支援綠金雅致風與科技藍風格）
├── app.js                           # 核心統計運算、動態圖表與 OLS 引擎
├── notebooks/7115064011_hw2.ipynb   # 主程式（CRISP-DM 六階段）
├── model/model_params.json          # 部署用模型參數（執行後產生）
├── data/                            # 資料集（含 50_Startups.csv 與 Kaggle 資料）
└── submissions/                     # Kaggle 提交檔（未上傳）
```

---

## 🌐 互動式網頁平台功能說明 (Web Platform)

本專題提供完整的純前端互動式多元線性迴歸分析平台（免安裝複雜後端環境，點擊 [`index.html`](index.html) 或以任何靜態伺服器開啟即可直接使用）：

### 🎯 加分項目落實：
1. **加總分 1分：下拉視窗呈現兩種 Problem**
   - **Problem 1: 基礎多元線性迴歸分析 (Basic Multiple Linear Regression)**：呈現 50 Startups 全特徵基準模型，包含 $R^2$、RMSE、MAE、調整後 $R^2$、統計診斷圖表 (實際值 vs 預測值、殘差分佈圖)、特徵係數圖、迴歸方程式與係數顯著性統計表、測試集樣本預測表、即時獲利試算器、CRISP-DM 六大流程卡片與商業洞察。
   - **Problem 2: 特徵選擇與模型精簡 (Feature Selection & Model Comparison)**：比較 5 種機器學習特徵篩選方法（Lasso、Forward SFS、Backward SFS、RFE、ElasticNet、Exhaustive Search），提供 $k=1 \sim 5$ 動態切換、RMSE 與 $R^2$ 消長折線圖、鎖定 $k=2$ 最佳黃金平衡點（Optimal Sweet Spot，RMSE 降至全場最低 $8,198.80）以及 6 種特徵配置卡片。
2. **加總分 2分：可任意輸入 Dataset**
   - **支援任意公開 CSV URL 輸入**：預設帶入 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`，亦可載入任何外部 CSV 或本機檔案。
   - **自動型態辨識與動態欄位選擇**：自動區分數值與類別特徵，自由挑選預測目標 $y$ 與特徵矩陣 $X$。
   - **純前端 OLS 矩陣迴歸計算引擎**：即時求解 $\beta = (X^T X)^{-1} X^T y$，動態產生模型指標、殘差診斷圖、迴歸方程式與係數表。
   - **動態即時敏感度實驗室 (Live Sensitivity Lab)**：依據使用者選取之特徵自動生成互動滑桿與選單，滑動時即時推估預測值！
3. **加碼整合原 Kaggle 洪水機率預測專案**：在下拉選單中可切換至洪水機率預測專案，整合 20 項特徵與即時風險模擬器。
4. **雙風格外觀主題切換**：點擊右上角按鈕即可在「綠金雅致風格」與「科技藍風格」之間一鍵無縫切換。


---

## 執行方式

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn kaggle jupyterlab

# 下載資料（需先在 Kaggle 加入競賽並設定 API token）
kaggle competitions download -c playground-series-s4e5 -p data
tar -xf data/playground-series-s4e5.zip -C data

jupyter lab                        # 開啟 notebooks/7115064011_hw2.ipynb 並 Run All
```

---

## AI 協助

- **開發工具**：Antigravity（Gemini）輔助撰寫 notebook
- **ChatGPT**：分析策略與統計概念討論（對話紀錄見報告附錄）
- **NotebookLM**：彙整網路上同主題解法並撰寫摘要（見報告）