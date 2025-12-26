# scripts/split_dataset.py

"""
split_dataset.py

将【已打好 regime 标签】的时间序列数据
按时间顺序切分为 train / validation 数据集。

用途：
- 避免时间泄漏
- 明确区分训练 / 验证数据
- 与 Dataset / Model 解耦

示例：
python split_dataset.py \
    --in data/BTC_labeled.feather \
    --out_dir data/ \
    --train_ratio 0.8
"""

import argparse
import pandas as pd
import os


def read_input(path: str) -> pd.DataFrame:
    """读取 feather / csv，并确保按时间排序"""
    if path.endswith(".feather") or path.endswith(".parquet"):
        df = pd.read_feather(path)
    else:
        df = pd.read_csv(path)

    # 处理时间索引
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    df = df.sort_index()
    return df


def split_by_ratio(df: pd.DataFrame, train_ratio: float):
    """按时间顺序切分 DataFrame"""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    return train_df, val_df


def save_df(df: pd.DataFrame, path: str):
    """保存为 feather 或 csv"""
    if path.endswith(".feather") or path.endswith(".parquet"):
        df.reset_index().to_feather(path)
    else:
        df.to_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Split labeled dataset into train / val")
    parser.add_argument("--in", dest="input_path", required=True, help="labeled input file")
    parser.add_argument("--out_dir", default="data", help="output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="train split ratio")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_input(args.input_path)

    # sanity check
    if "regime" not in df.columns:
        raise ValueError("Input data must contain 'regime' column")

    train_df, val_df = split_by_ratio(df, args.train_ratio)

    train_path = os.path.join(args.out_dir, "train.feather")
    val_path = os.path.join(args.out_dir, "val.feather")

    save_df(train_df, train_path)
    save_df(val_df, val_path)

    print(f"[OK] Train samples: {len(train_df)}")
    print(f"[OK] Val samples:   {len(val_df)}")
    print(f"[OK] Saved train -> {train_path}")
    print(f"[OK] Saved val   -> {val_path}")


if __name__ == "__main__":
    main()
