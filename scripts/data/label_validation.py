import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========== 配置 ==========
DATA_PATH = "../data_out/BTC_USDT_USDT-1h-futures.feather"   # 改成你的添加标签路径路径
WINDOW = 48
N_SAMPLES = 20
# ==========================

# 读取数据
if DATA_PATH.endswith(".feather"):
    df = pd.read_feather(DATA_PATH)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
else:
    df = pd.read_csv(DATA_PATH, parse_dates=['date'], index_col='date')

print("Regime distribution:")
print(df['regime'].value_counts(normalize=True))

# 随机抽样窗口
valid_indices = np.arange(WINDOW, len(df))
sample_ends = np.random.choice(valid_indices, size=N_SAMPLES, replace=False)

for i, end in enumerate(sample_ends):
    window_df = df.iloc[end-WINDOW:end]

    plt.figure(figsize=(10, 4))
    plt.plot(window_df.index, window_df['close'], label='Close')

    regime = int(window_df['regime'].iloc[-1])
    title_map = {0: 'Sideway', 1: 'Uptrend', 2: 'Downtrend', 3: 'Volatile'}
    plt.title(f"Sample {i+1} | Regime = {title_map.get(regime, regime)}")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # input("Press Enter to continue...")
