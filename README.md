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
├── notebooks/7115064011_hw2.ipynb   # 主程式（CRISP-DM 六階段）
├── model/model_params.json          # 部署用模型參數（執行後產生）
├── data/                            # Kaggle 資料（未上傳，見下方下載方式）
└── submissions/                     # Kaggle 提交檔（未上傳）
```

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