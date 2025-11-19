import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt

from finlab import data
from gymnasium import spaces
from stable_baselines3 import PPO
from gymnasium.envs.registration import register


# 從 finlab 取得資料
Closes = data.get('etl:adj_close')
Opens = data.get('etl:adj_open')
Highs = data.get('etl:adj_high')
Lows = data.get('etl:adj_low')
Volumes = data.get('after_market_odd_lot_trade:成交股數')
Foreign_Investors = data.get('institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
Investment_Trust = data.get('institutional_investors_trading_summary:投信買賣超股數')
Dealer = data.get('institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
Margin_Trading = data.get('margin_transactions:融資今日餘額')
Short_Selling = data.get('margin_transactions:融券今日餘額')


# 資料前處理一，將資料由寬表格改為窄表格
Closes = Closes.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Close')
Opens = Opens.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Open')
Highs = Highs.reset_index().melt(id_vars='date', var_name='stock_id', value_name='High')
Lows = Lows.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Low')
Volumes = Volumes.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Volume')
Foreign_Investors = Foreign_Investors.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Foreign_Investor')
Investment_Trust = Investment_Trust.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Investment_Trust')
Dealer = Dealer.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Dealer')
Margin_Trading = Margin_Trading.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Margin_Trading')
Short_Selling = Short_Selling.reset_index().melt(id_vars='date', var_name='stock_id', value_name='Short_Selling')


# 資料前處理二，將所有表格合併
df = Closes.merge(Opens, on=['date','stock_id'], how='left')
df = df.merge(Highs, on=['date','stock_id'], how='left')
df = df.merge(Lows, on=['date','stock_id'], how='left')
df = df.merge(Volumes, on=['date','stock_id'], how='left')
df = df.merge(Foreign_Investors, on=['date','stock_id'], how='left')
df = df.merge(Investment_Trust, on=['date','stock_id'], how='left')
df = df.merge(Dealer, on=['date','stock_id'], how='left')
df = df.merge(Margin_Trading, on=['date','stock_id'], how='left')
df = df.merge(Short_Selling, on=['date','stock_id'], how='left')


# 技術指標
def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


# 資料前處理三，加入技術指標
# 當日報酬率
df['ret'] = df['Close'].pct_change()

# 前幾日的報酬率
for l in [1,2,3,5]:
    df[f'lag_ret_{l}'] = df.groupby('stock_id')['ret'].shift(l)

# MA
df['ma5']  = df.groupby('stock_id')['Close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['ma10'] = df.groupby('stock_id')['Close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
df['ma20'] = df.groupby('stock_id')['Close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma_ratio_5_20'] = df['ma5'] / (df['ma20'] + 1e-9)

# 成交量波動
df['vol_10'] = df.groupby('stock_id')['Volume'].transform(lambda x: x.rolling(10, min_periods=1).std())
df['vol_20'] = df.groupby('stock_id')['Volume'].transform(lambda x: x.rolling(20, min_periods=1).std())

# RSI
df['rsi_14'] = df.groupby('stock_id')['Close'].transform(lambda x: rsi(x, period=14))

# MACD
df['macd'] = df.groupby('stock_id')['Close'].transform(lambda x: macd(x)[0])
df['macd_signal'] = df.groupby('stock_id')['Close'].transform(lambda x: macd(x)[1])
df['macd_hist'] = df.groupby('stock_id')['Close'].transform(lambda x: macd(x)[2])

# 三大法人近十日買賣超
df['foreign_10'] = df.groupby('stock_id')['Foreign_Investor'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
df['investment_10'] = df.groupby('stock_id')['Investment_Trust'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
df['dealer_10'] = df.groupby('stock_id')['Dealer'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())

# 融資融券近十日
df['margin_10'] = df.groupby('stock_id')['Margin_Trading'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())
df['short_10'] = df.groupby('stock_id')['Short_Selling'].transform(lambda x: x.rolling(window=10, min_periods=1).sum())

# 目標: 隔日漲跌 (1 = 漲, 0 = 平盤或跌)
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df.dropna(inplace=True)


'''GYM + PPO'''
# 自訂交易環境
class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=100000, embedding_dim=8, window=20):
        super(StockTradingEnv, self).__init__()
        
        # 原始資料處理
        df = df.copy().reset_index(drop=True)

        # 取得完整日期與股票清單
        all_dates = sorted(df["date"].unique())
        all_stocks = sorted(df["stock_id"].unique())

        # 建立完整日期×股票的 MultiIndex
        full_index = pd.MultiIndex.from_product([all_dates, all_stocks], names=["date", "stock_id"])

        # 將資料補齊，缺資料會 forward-fill
        df = (
            df.set_index(["date", "stock_id"])
              .reindex(full_index)
#               .groupby(level=1)
#               .ffill()   # 沿用前值
              .fillna(0) # 補0
              .reset_index()
        )
        
        # 基本設定
        self.df = df.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.embedding_dim = embedding_dim
        self.window = window
        self.feature_cols = [
            'ret', 'lag_ret_1', 'lag_ret_2', 'lag_ret_3', 'lag_ret_5',
            'ma5', 'ma10', 'ma20', 'ma_ratio_5_20',
            'vol_10', 'vol_20',
            'rsi_14',
            'macd', 'macd_signal', 'macd_hist',
            'foreign_10', 'investment_10', 'dealer_10',
            'margin_10', 'short_10'
        ]
        self.last_obs = {}
        self.fee_rate = 0.001425   # 0.1425% 手續費
        self.tax_rate = 0.003      # 0.3% 交易稅（賣出）

        # 整理日期與股票
        self.stock_list = sorted(self.df["stock_id"].unique())
        self.num_stocks = len(self.stock_list)
        self.dates = sorted(self.df["date"].unique())
        self.num_days = len(self.dates)

        # 建立 embedding
        self.embeddings = np.random.normal(0, 0.1, (self.num_stocks, embedding_dim)).astype(np.float32)

        # 定義 observation 與 action 空間
        # 每支股票：3個特徵 + embedding + 2個資金特徵
        obs_dim_per_stock = len(self.feature_cols) + embedding_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_stocks, obs_dim_per_stock), dtype=np.float32
        )

        # 動作空間：每支股票一個動作（買/賣/持有）
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32
        )

        # 交易參數 
        self.max_trade_ratio = 0.2
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_stocks, dtype=np.float32)
        self.net_worth = self.initial_balance
        return self._next_observation(), {}

    def _next_observation(self):
        date = self.dates[self.current_step]
        day_data = self.df[self.df["date"] == date].set_index("stock_id")

        obs_list = []
        for stock_id in self.stock_list:
            row = day_data.loc[stock_id]
                
            stock_idx = self.stock_list.index(stock_id)

            features = row[self.feature_cols].values.astype(np.float32)
            embed = self.embeddings[stock_idx]
            
            balance_ratio = np.float32(np.clip(self.balance / self.initial_balance, 0, 10))
            shares_ratio = np.float32(np.clip(self.shares_held[stock_idx] / 1000.0, 0, 10))

            obs = np.concatenate([features, embed, [balance_ratio, shares_ratio]])
            obs_list.append(obs)

        return np.stack(obs_list, axis=0)

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        date = self.dates[self.current_step]
        day_data = self.df[self.df["date"] == date].set_index("stock_id")

        prices = np.array([day_data.loc[s]["Close"] for s in self.stock_list], dtype=np.float32)
        
        # ✅ 記錄交易前的資產淨值
        prev_net_worth = self.net_worth

        # 每支股票計算交易金額
        for i, action in enumerate(actions):
            price = prices[i]
            
            # 跳過停牌或價格為0的股票
            if price <= 0 or np.isnan(price):
                continue
        
            trade_amount = self.balance * self.max_trade_ratio * abs(action)

            if action > 0:  # 買入
                shares_to_buy = trade_amount / price
                cost = shares_to_buy * price
                # 手續費
                fee = cost * self.fee_rate
                # 更新持股與資金
                self.shares_held[i] += shares_to_buy
                self.balance -= cost + fee
        
            elif action < 0:  # 賣出
                shares_to_sell = min(self.shares_held[i], trade_amount / price)
                revenue = shares_to_sell * price
                # 手續費 + 交易稅（台股賣出才收交易稅）
                fee = revenue * self.fee_rate
                tax = revenue * self.tax_rate
                # 更新持股與資金
                self.shares_held[i] -= shares_to_sell
                self.balance += revenue - fee - tax
        
        # 更新淨值
        self.net_worth = self.balance + np.sum(self.shares_held * prices)
        self.net_worth = np.clip(self.net_worth, 0, 1e9)

        # reward：日報酬
        reward = (self.net_worth - prev_net_worth) / (prev_net_worth + 1e-8)
        reward = float(np.tanh(reward * 10))
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0  # ✅ 防 NaN reward

        self.current_step += 1
        terminated = self.current_step >= self.num_days - 1

        return self._next_observation(), reward, terminated, False, {"net_worth": self.net_worth}

    def render(self):
        print(f"Day {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f}")


# 註冊環境
register(
    id='StockTradingEnv-v0',
    entry_point=lambda **kwargs: StockTradingEnv(**kwargs),
)

# 建立環境
env = gym.make('StockTradingEnv-v0', df=df)

# 訓練 PPO 模型
model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # 若要用 GPU 改 cuda
model.learn(total_timesteps=200000)

# 儲存模型
model.save("ppo_stock_trading")

# 模擬回測
obs, info = env.reset()
profits = []
for _ in range(len(df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    profits.append(info['net_worth'])
    if terminated or truncated:
        break

# 顯示績效
plt.figure(figsize=(10,5))
plt.plot(profits)
plt.title("Portfolio Net Worth Over Time")
plt.xlabel("Step")
plt.ylabel("Net Worth")
plt.grid(True)
plt.show()



