# HW2 · 多元線性回歸（Multiple Linear Regression）

**侯佳旻 7114056115**｜物聯網數據分析與應用

🔗 **網頁 Demo**：https://ausfa4388.github.io/HW2-MLR/

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

**Problem 1 · Life Expectancy (WHO)**

> 我透過 NotebookLM 比較了四份以 WHO 預期壽命資料集進行回歸分析的解法。
>
> **缺失值處理：**各解法的做法不同。有的先刪除沒有預期壽命數值的樣本，再用該國或全體資料的中位數補齊其他欄位；也有的改用裝袋樹（Bagged Trees）模型推估缺失值。
>
> **特徵選擇：**常見的工具包括前向／後向逐步回歸、相關係數矩陣、VIF 檢定，以及 PCA 與 Lasso 正則化。篩選之後，多數分析都保留了教育年限（Schooling）、HIV/AIDS、成人死亡率（Adult Mortality）、收入資源組合指數（Income composition of resources）和疫苗接種率這幾個關鍵變數。
>
> **模型與評估：**除了多元線性回歸，也有解法拿 Ridge、Lasso、KNN、SVM 和集成樹模型（Random Forest、Boosting）來比較。線性模型多以 R² 和 RMSE 評估，R² 約在 0.82 到 0.86 之間，RMSE 約 3.6 到 3.7 歲。
>
> **共同發現：**結論相當一致。HIV/AIDS 與成人死亡率越高，預期壽命越短；教育年限與經濟條件則與壽命呈強烈正相關。共同的難題是多重共線性，例如不同死亡率指標之間、不同經濟變數之間高度相關，通常要靠 VIF 篩選或刪除重複變數來處理。
>
> **與本作業的比較：**這些解法多半用 VIF 或逐步回歸決定保留哪些變數，但「要留幾個」常靠經驗判斷。本作業同樣比較了四種特徵選擇方法，再以 5-fold 交叉驗證搭配 1-SE rule 選出 RFE、保留 8 個特徵：在誤差與最佳值相差不到一個標準誤的模型中，挑特徵最少的一個。這讓特徵數的決定有客觀依據。測試集 R² 為 0.828，與上述線性模型 0.82–0.86 的表現相當。

**Problem 2 · Car Price**

> 我透過 NotebookLM 比較了 Kaggle 上幾份以 CarPrice_Assignment 資料集、用線性迴歸預測車價的解法。
>
> **前處理：**對 `CarName` 的看法分成兩派。jainnnnn、hellbuoy 與 SampattKumar 取車名第一個字當作品牌，並修正拼字錯誤（例如 maxda 改成 mazda）；AayushiWani 則視它為識別用的名稱，直接刪除。類別變數的編碼有 One-Hot Encoding、LabelEncoder，或把有順序的類別對應成數字。目標變數方面，jainnnnn 用 `log1p` 修正價格的右偏，並以 StandardScaler 標準化數值特徵；AayushiWani 則保留原始價格，也沒有做縮放。
>
> **特徵選擇：**hellbuoy 與 SampattKumar 先用 RFE 粗篩，再依 VIF 與 p-value 逐一剔除共線性高或不顯著的變數；AayushiWani 依相關係數手動刪掉資訊重複的欄位，例如保留 `highwaympg`、去掉與它高度相關的 `citympg`；jainnnnn 沒有另外篩選，24 個特徵全部保留。
>
> **模型表現：**AayushiWani 在原始價格上得到 R² 約 0.81、RMSE 約 3,383 美元；jainnnnn 在 log 價格尺度上得到 R² 0.918、RMSE 0.149。
>
> **共同發現：**引擎大小、車重、馬力等規格彼此高度相關，油耗相關變數之間也一樣，多重共線性是各解法都要處理的問題。另外，車價明顯右偏，取對數後模型擬合會明顯改善。
>
> **與本作業的比較：**本作業和 jainnnnn 一樣對 log(price) 建模，但多做了特徵選擇。比較四種方法後採用 Forward SFS，從 43 個特徵中選出 8 個，測試集 R² 在原始價格尺度為 0.854、log 尺度為 0.812，略低於 jainnnnn 的 0.918。主要差異在於本作業直接排除 `CarName`，沒有像其他解法一樣抽出品牌；品牌是影響車價的重要因素，這是本作業可以改進的地方。不過，與本作業自己的全特徵模型（R² 0.873）相比，只用不到五分之一的變數就達到相近的解釋力，也較不受多重共線性影響。

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
