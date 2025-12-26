# validation/error_analysis.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from research.data.dataset_future import FutureRegimeDataset
from research.model.model import RegimeTransformer


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    records = []

    for x, y, meta in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        for i in range(len(preds)):
            records.append({
                "y_true": int(y[i].item()),
                "y_pred": int(preds[i]),
                "future_rel_slope": float(meta["future_rel_slope"][i])
            })

    return pd.DataFrame(records)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 参数 =====
    window = 48
    batch_size = 128
    num_classes = 2

    val_path = "data/split/val.feather"
    ckpt_path = "research/checkpoints/best_model.pt"

    # ===== Dataset（Day 8 用：必须保留 future_rel_slope）=====
    val_dataset = FutureRegimeDataset(
        data_path=val_path,
        past_window=window,
        future_window=window,
        normalize=True,
        return_meta=True,     # ⭐关键：返回 future_rel_slope
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ===== 模型 =====
    model = RegimeTransformer(
        feature_dim=len(val_dataset.feature_cols),
        d_model=64,
        num_heads=4,
        num_layers=1,
        num_classes=num_classes,
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[INFO] Loaded model from {ckpt_path}")

    # ===== 收集预测 =====
    df = collect_predictions(model, val_loader, device)

    # ===== 分析 =====
    df["correct"] = df["y_true"] == df["y_pred"]
    df["abs_future_rel_slope"] = df["future_rel_slope"].abs()

    print("\n===== Sample Count =====")
    print(df["correct"].value_counts())

    print("\n===== abs(future_rel_slope) Statistics =====")
    print(df.groupby("correct")["abs_future_rel_slope"].describe())

    # ===== 分桶分析（关键）=====
    bins = [0, 0.0003, 0.0006, 0.001, 0.002, np.inf]
    df["slope_bin"] = pd.cut(df["abs_future_rel_slope"], bins=bins)

    print("\n===== Error Rate by Slope Bin =====")
    print(
        df.groupby("slope_bin")["correct"]
          .apply(lambda x: 1 - x.mean())
          .rename("error_rate")
    )


if __name__ == "__main__":
    main()
