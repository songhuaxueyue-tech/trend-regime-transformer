# scripts/data/label_generator_future.py

"""
label_generator_future.py

基于【已生成的当前 Regime 标签】，构造【未来预测标签】

职责：
- 不重新定义市场
- 只回答：站在 t 时刻，未来 window 内“整体更像 Up / Down / Sideway”

输出新增列：
- future_regime
- future_rel_slope（用于误差分析 / 置信度分析）
"""

import argparse
import numpy as np
import pandas as pd


def read_input(path: str) -> pd.DataFrame:
    if path.endswith(".feather") or path.endswith(".parquet"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    df = df.sort_index()
    df = df.rename(columns={c: c.lower() for c in df.columns})

    required = {"close", "regime"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    return df


def generate_future_labels(
    df: pd.DataFrame,
    future_window: int = 48,
    slope_quantile: float = 0.7,
):
    """
    使用未来 window 的价格变化，生成 future_regime
    """

    close = df["close"].values.astype(float)
    future_slope = np.full(len(df), np.nan)

    # --- 计算未来窗口 slope（仅用于聚合，不参与定义规则） ---
    for t in range(len(df) - future_window):
        y = np.log(close[t + 1 : t + 1 + future_window] + 1e-8)
        x = np.arange(len(y))
        coef = np.polyfit(x, y, 1)
        future_slope[t] = coef[0]

    df["future_slope"] = future_slope

    # 相对斜率（用于跨品种稳定）
    # future_rel_slope 本质是未来 window 的 log-price slope（per bar）
    df["future_rel_slope"] = df["future_slope"]      
    # --- 使用分位数确定趋势阈值 ---
    valid = df["future_rel_slope"].dropna().abs()
    threshold = np.nanquantile(valid, slope_quantile)

    # --- 生成 future_regime ---
    # future_regime:
    # 0 = future sideway / no-trade
    # 1 = future up
    # 2 = future down

    df["future_regime"] = 0  # Sideway

    df.loc[df["future_rel_slope"] > threshold, "future_regime"] = 1
    df.loc[df["future_rel_slope"] < -threshold, "future_regime"] = 2

    return df, threshold


def save_df(df: pd.DataFrame, out_path: str):
    if out_path.endswith(".feather") or out_path.endswith(".parquet"):
        df.reset_index().to_feather(out_path)
    else:
        df.to_csv(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    parser.add_argument("--future_window", type=int, default=48)
    parser.add_argument("--slope_quantile", type=float, default=0.7)

    args = parser.parse_args()

    df = read_input(args.infile)

    df_out, thresh = generate_future_labels(
        df,
        future_window=args.future_window,
        slope_quantile=args.slope_quantile,
    )

    save_df(df_out, args.outfile)

    print(f"[INFO] Saved future-labeled data to {args.outfile}")
    print(f"[INFO] future_rel_slope threshold (q={args.slope_quantile}): {thresh:.6e}")
    print("future_regime distribution:")
    print(df_out["future_regime"].value_counts())


if __name__ == "__main__":
    main()
