### 使用 FinLab 資料 + 自訂 Gym 強化學習環境 + Stable-Baselines3 PPO，訓練一個能操作 多支股票投資組合（Portfolio） 的 RL 交易代理人。

#### -資料取得
從 Finlab 下載每日股價、成交量、三大法人、融資融券等資料。

#### -特徵工程
技術指標：MA、RSI、MACD、成交量波動 
籌碼指標：外資、投信、自營商近十日買賣超 
融資 / 融券變化 
報酬率與 lag features

#### -自訂 GYM 交易環境
環境名稱：StockTradingEnv-v0

#### -使用 PPO 訓練

#### -回測與績效圖示
<img width="877" height="470" alt="image" src="https://github.com/user-attachments/assets/d6950f92-1c9b-44ed-9ed5-591f7432eaad" />
