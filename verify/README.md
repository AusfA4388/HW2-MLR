# 驗證：JS 與 statsmodels 對照

在 repo 根目錄執行：

```
node verify/run_core.js data/life.csv "Life expectancy" 0 && python verify/check.py
node verify/run_core.js data/car.csv price 1 && python verify/check.py
```
