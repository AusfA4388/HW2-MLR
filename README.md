# HW2 · 多元線性回歸（Multiple Linear Regression）

**侯佳旻 7114056115**｜物聯網數據分析與應用

🔗 **網頁 Demo**：https://你的帳號.github.io/你的repo名稱/ （← 部署後改成實際網址）

以 Kaggle 上特徵數 > 13 的資料集，依 CRISP-DM 流程完成多元線性回歸分析：特徵選擇、模型評估、含 95% 信賴區間 / 預測區間的預測圖。整個分析以純 JavaScript 在瀏覽器即時計算，部署於 GitHub Pages。

## 加分項

| 項目 | 實作 |
|---|---|
| +1 下拉選單呈現兩種 problem | Problem 1 · Life Expectancy (WHO)／Problem 2 · Car Price |
| +2 可任意輸入 Dataset | 選「自訂 Dataset」→ 貼 CSV 網址（例如 [50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)）或上傳 CSV，可選 target 與是否取 log |

## 資料集

| Problem | Kaggle 來源 | 筆數 | 原始特徵數 | Target |
|---|---|---|---|---|
| 1 | [Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who) | 2,938 | 21 | Life expectancy（年） |
| 2 | [Car Price Prediction](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction) | 205 | 25 | price（美元） |

網頁內嵌的 CSV 取自 GitHub 鏡像（與 Kaggle 原始檔相同）：[Life Expectancy](https://raw.githubusercontent.com/Vipul97/life-expectancy-prediction/master/Life%20Expectancy%20Data.csv)、[Car Price](https://raw.githubusercontent.com/SampattKumar/Linear-Regression-Car-Dataset/master/CarPrice_Assignment.csv)。

## CRISP-DM

### 1. Business Understanding
- **Life Expectancy**：找出最能解釋各國平均壽命的少數因素，作為衛生資源分配參考。
- **Car Price**：車廠進入美國市場，想知道哪些規格決定車價。

### 2. Data Understanding
- Life Expectancy：19 個數值特徵＋1 個類別特徵（Status），另有 Country（193 類）。多個欄位有缺值，Population 缺 22%、Hepatitis B 缺 19%。
- Car Price：14 個數值＋9 個類別特徵，沒有缺值；price 右偏。

### 3. Data Preparation
1. 排除 ID 欄位與類別過多的文字欄位（Country、CarName）。
2. 有缺值的列整列刪除：Life Expectancy 從 2,938 筆剩 1,649 筆。
3. 類別欄位做 one-hot（drop first），編碼後 Life 有 20 個特徵、Car 有 43 個特徵。
4. Car Price 對 log(price) 建模，以改善右偏與異質變異。
5. 隨機切分 80/20（seed = 42）。所有排序、選擇與標準化都只用訓練集。

### 4. Modeling
四種特徵排序方法：**Forward SFS、RFE、Lasso path、ElasticNet path**。每種方法取前 k 個特徵做 OLS，用**訓練集 5-fold CV RMSE** 搭配 **1-SE rule** 選 k，測試集不參與任何選擇。

| Problem | 最終方法 | k | 選入特徵 |
|---|---|---|---|
| Life Expectancy | RFE | 8 / 20 | under-five deaths, infant deaths, Schooling, Adult Mortality, HIV/AIDS, Income composition of resources, percentage expenditure, BMI |
| Car Price | Forward SFS | 8 / 43 | curbweight, horsepower, enginelocation_rear, fuelsystem_mpfi, carbody_wagon, enginetype_ohc, cylindernumber_four, stroke |

### 5. Evaluation

| Problem | 模型 | Test RMSE | Test R² | 95% PI 覆蓋率 |
|---|---|---|---|---|
| Life Expectancy | 全特徵（20） | 3.739 | 0.8309 | – |
| | **選擇後（8）** | **3.774** | **0.8277** | **93.3%** |
| Car Price | 全特徵（43） | 2,223.7 | 0.8733 | – |
| | **選擇後（8）** | **2,386.7** | **0.8540** | **85.4%** |

**觀察：**
- Life Expectancy 只用 8 個特徵，就幾乎達到全特徵模型的表現。Train / Test R² 只差 0.005，沒有過擬合；測試集 93.3% 的樣本落在 95% PI 內，接近理論值。
- `infant deaths` 和 `under-five deaths` 的相關係數 r = 0.997，VIF 約 178，兩者係數一正一負互相抵銷。預測不受影響，但不能單獨解讀這兩個係數。
- 如果只看 41 筆的測試集，Car Price 全特徵模型的分數較高。但在 CV 上，8 特徵模型與最佳結果差距不到 1 個標準誤，所以選擇較精簡、較好解釋的模型。PI 覆蓋率 85%，低於 95%，代表少數高價車的殘差較大（厚尾）。
- 預測圖、偏效應圖（含 CI / PI）、殘差圖、Q-Q 圖都在網頁的 Evaluation 區。

### 6. Deployment
GitHub Pages 單一 `index.html`，頁面上有即時預測器，輸入特徵值會輸出預測值與 95% PI。所有計算（OLS、t / F 檢定、VIF、CI / PI、Lasso / ElasticNet 座標下降）都以 JavaScript 實作，並與 Python statsmodels 對照驗證，係數、p-value 與區間的相對誤差都小於 1e-12（見 `verify/`）。

## AI 協助

### GPT 輔助內容
本作業的 AI 輔助工具為 **Claude（Anthropic）**，對話紀錄 PDF 在 `ai_logs/`。AI 協助的部分：
- 比較並挑選資料集
- 設計 CV＋1-SE rule 的特徵選擇流程
- 撰寫網頁與統計程式碼，並與 statsmodels 交叉驗證
- 撰寫說明文字初稿

分析結果由本人確認。

### NotebookLM 摘要
> （待補：NotebookLM 對網路上同主題解法的研究摘要，100 字以上）

## Repo 結構
```
index.html        ← 網頁（由 build.py 產生，GitHub Pages 讀這個）
README.md
src/core.js       ← 回歸與特徵選擇核心（純函式）
src/template.html ← 網頁樣板
build.py          ← 把 core.js 與資料內嵌進 index.html
data/             ← 兩個資料集 CSV
verify/           ← 與 Python statsmodels 對照的驗證腳本
ai_logs/          ← AI 對話 PDF
```
