import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.data.dataset import OHLCVWindowDataset

import matplotlib

import os
print("[DEBUG] CWD =", os.getcwd())



# ================= 配置 =================
DATA_PATH = "other\data_out\BTC_USDT_USDT-1h-futures.feather"
WINDOW = 48
N_SAMPLES = 20          # 抽样窗口数量（建议 3–10）
SHOW_VOLUME = False   # 是否显示 volume
# =======================================


from pathlib import Path

def plot_window(x, y, idx):
    x = x.numpy()
    close = x[:, 3]

    title_map = {0: "Sideway", 1: "Uptrend", 2: "Downtrend"}

    fig = plt.figure(figsize=(8, 4))
    plt.plot(close, linewidth=2)
    plt.title(f"Sample {idx} | Label = {title_map.get(int(y), y)}")
    plt.xlabel("Time (bars)")
    plt.ylabel("Normalized Close")
    plt.grid(True)
    plt.tight_layout()

    # === 关键：绝对路径 + 显式创建目录 ===
    out_dir = Path(r"D:\project\trend-regime-transformer\_sanity_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"sample_{idx}.png"
    print(f"[DEBUG] Saving figure to: {out_path}")

    fig.savefig(out_path)
    plt.close(fig)




def main():
    ds = OHLCVWindowDataset(
        data_path=DATA_PATH,
        window=WINDOW,
        normalize=True
    )

    print(f"[INFO] Dataset size: {len(ds)}")

    sample_indices = random.sample(range(len(ds)), N_SAMPLES)

    for i, idx in enumerate(sample_indices):
        x, y = ds[idx]

        print(f"\n--- Sample {i+1} / {N_SAMPLES} ---")
        print(f"Index: {idx}")
        print(f"Label: {int(y)}")
        print(f"x shape: {x.shape}, dtype: {x.dtype}")

        plot_window(x, y, i + 1)


        # input("Press Enter to continue...")


if __name__ == "__main__":
    print("MATPLOTLIB BACKEND =", matplotlib.get_backend())
    main()
