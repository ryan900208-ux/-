# Quality Momentum Trader

這是一個基於**「基本面品質優選 (Quality)」**與**「技術面強勢動能 (Momentum)」**相結合的台股量化交易系統。

本專案是針對原始專案進行的全新重寫版，剔除了冗餘的機器學習步進測試，聚焦於最優的量化規則模型（即歷史回測中實現 **+917.64%** 總收益的策略），並將專案檔案數控制在 10 個核心檔案以內，便於 Git 管理。

---

## 策略邏輯摘要

1. **大盤環境濾網 (Regime Filter)**：
   - 僅在大盤（預設為 `0050.TW`）處於 **Bull (牛市)** 或 **Neutral (中性)** 時允許建立新倉。
   - 大盤轉為 **Bear (熊市)** 時，所有持倉立刻執行大盤防禦性出場。
2. **基本面 EVA 品質池 (EVA Pool)**：
   - 每季財報公告後，依據 ROIC、ROE、利潤率及負債比等指標，對全上市股票計算綜合 EVA 分數，篩選出**前 15 檔品質最優**的個股作為選股池。
3. **動能池內排序**：
   - 僅在 15 檔優質股中，計算 20 日及 60 日相對強弱 (Relative Strength)，挑選 RS 排名在前 40%~50% 的強勢整理標的。
4. **交易執行與資金管理**：
   - **T日收盤判斷，T+1日開盤價 (Open) 執行交易**。
   - 最大持倉 5 檔，每檔等權配置 20% 權重。包含手續費 (0.1425%)、滑價 (0.1%) 及證交稅 (0.3%)。
5. **風控出場機制**：
   - **硬性停損**：買入價下跌 12% 執行停損，觸發後個股進入 **30 天冷卻期**。
   - **持有到期**：最長持有 252 個交易日 (1年)。
   - **均線跌破**：股價跌破 `MA120` 時出場。

---

## 歷史回測表現 (2020-01-02 ~ 2026-07-02)

| 績效指標 | 表現數值 |
| :--- | :---: |
| **總報酬率 (Total Return)** | **+917.64%** (10.17倍) |
| **年化報酬率 (CAGR)** | **+42.92%** |
| **最大回撤 (Max Drawdown)** | **-25.09%** |
| **夏普值 (Sharpe Ratio)** | **1.54** |
| **交易筆數 (Total Trades)** | 89 筆 |
| **勝率 (Win Rate)** | 34.83% |
| **平均每筆交易報酬** | +16.61% |
| **期末權益 (Final Equity)** | 10,176,446 元 (起點100萬) |

---

## 專案結構

```text
quality_momentum_trader/
│
├── config.json                 # 交易參數與篩選閥值配置文件
├── requirements.txt            # 專案套件依賴項
├── README.md                   # 說明文件
│
├── data/                       # 數據目錄
│   ├── universe_twse_all.csv   # 全上市股票代號名單
│   └── fundamentals_finmind.csv# 歷史季度基本面資料
│
├── work/                       # 股價緩存目錄
│   └── price_cache/            # 股票歷史日 K 線資料 (.csv)
│
├── outputs/                    # 輸出目錄
│   ├── backtest/               # 歷史回測輸出 (summary, trades, equity_curve)
│   └── paper_trading/          # 模擬交易狀態、記錄及快照
│
├── src/                        # 核心源碼
│   ├── data_manager.py         # 數據讀取與快照快取模組
│   ├── indicators.py           # 技術指標與 Regime 計算模組
│   ├── strategy.py             # EVA 品質篩選與 RS 動能選股邏輯
│   ├── backtester.py           # T+1 Open 交易回測引擎
│   └── reporter.py             # 績效匯總報告生成器
│
├── docs/                       # 前端靜態網頁目錄
│   └── index.html              # 視覺化選股與交易看板網頁
│
├── api_app.py                  # API 伺服器 (提供前端所需 API 端點)
├── run_local_api.bat           # 快速啟動 API 伺服器的批次檔
├── run_backtest.py             # 歷史回測執行腳本
└── run_daily_update.py         # 每日收盤訊號更新與模擬交易自動化腳本
```

---

## 如何使用

### 1. 安裝套件
請確保您的 Python 環境已安裝 pandas 與 yfinance：
```powershell
pip install -r requirements.txt
```

### 2. 執行歷史回測
運行歷史回測，這將輸出分析報表並生成權益曲線與交易明細：
```powershell
python run_backtest.py
```
回測生成的詳細 CSV 資料將被保存在 `outputs/backtest/` 目錄中。

### 3. 每日自動化更新訊號與記錄交易
您可以使用工作排程器或手動在**每日台股收盤後**運行以下命令：
```powershell
python run_daily_update.py
```
該腳本會執行以下自動化流程：
1. **更新股價**：下載最新的日線數據，並刷新快取目錄。
2. **執行交易**：檢測昨日生成的掛單 (Pending Orders)，若今日 Open 價已載入，即在模擬帳戶中**以 Open 價完成交易**。
3. **風控監控**：計算今日收盤價，若持倉股觸發停損、跌破MA120或持有到期，產生 **Pending SELL** 訊號。
4. **生成新單**：若有空缺倉位，從今日的 Top-15 EVA 品質候選股中挑选最強個股，產生 **Pending BUY** 訊號。
5. **記錄日快照**：將今日帳戶權益與持倉價值存入 `outputs/paper_trading/daily_snapshots.csv`。
6. **更新狀態**：保存當前帳戶餘額、持倉股與掛單明細至 `outputs/paper_trading/state.json`。

### 4. 啟動 Web 視覺化看板 (FastAPI/Starlette Dashboard)
1. 雙擊執行 `run_local_api.bat`（或在終端機運行 `python -m uvicorn api_app:app --port 8000`）啟動 API 伺服器。
2. 在您的瀏覽器中直接開啟 `docs/index.html` 網頁。
3. 即可在精美的**暗色系玻璃擬物風格 (Glassmorphism)** 儀表板中：
   - 瀏覽實盤模擬的總收益率、帳戶權益與可用現金比率。
   - 實時監控持倉明細與未實現 PnL（點擊個股可顯示均線與停損點）。
   - 瀏覽次日掛單訊號 (Pending Orders) 與今日候選股 (EVA Candidates)。
   - 查看已完成的交易記錄日誌。
   - 點擊右上角的「更新訊號與帳戶」按鈕，前端將自動呼叫 API 執行 `run_daily_update.py` 腳本，實現每日一鍵全自動更新！
