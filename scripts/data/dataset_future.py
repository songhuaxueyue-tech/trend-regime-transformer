import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional


class FutureRegimeDataset(Dataset):
    """
    Predict FUTURE regime direction (Up / Down)
    using PAST OHLCV window.

    - Sideway samples are ignored
    """

    def __init__(
        self,
        data_path: str,
        past_window: int = 48,
        future_window: int = 48,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "regime",
        normalize: bool = True,
        return_meta: bool = False,
    ):
        self.past_window = past_window
        self.future_window = future_window
        self.normalize = normalize
        self.label_col = label_col
        self.return_meta = return_meta

        # -------- load data --------
        if data_path.endswith(".feather") or data_path.endswith(".parquet"):
            df = pd.read_feather(data_path)
        else:
            df = pd.read_csv(data_path)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # 排序
        df = df.sort_index()

        # 统一列名（非常关键）
        df = df.rename(columns={c: c.lower() for c in df.columns})


                # ===== feature columns =====
        if feature_cols is None:
            feature_cols = ["open", "high", "low", "close", "volume"]

        self.feature_cols = feature_cols


        # 检查 future_rel_slope 是否存在
        if "future_rel_slope" not in df.columns:
            raise ValueError(
                "future_rel_slope not found. "
                "Please run label_generator to generate future labels first."
            )

        self.df = df

        self.feature_cols = feature_cols

        # -------- build valid indices --------
        self.indices = []

        start = past_window - 1
        end = len(df) - future_window - 1

        for t in range(start, end + 1):

            future_label = int(df.iloc[t + future_window][label_col])

            if future_label == 0:      # Sideway → ignore
                continue

            self.indices.append(t)

        if len(self.indices) == 0:
            raise RuntimeError("No valid trend samples found.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        # ----- input: past window -----
        past_df = self.df.iloc[t - self.past_window + 1 : t + 1]
        X = past_df[self.feature_cols].values.astype(np.float32)

        if self.normalize:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mean) / std

        # ----- label: future regime -----
        raw_label = int(self.df.iloc[t + self.future_window][self.label_col])


        if raw_label == 1:      # Up
            y = 0
        elif raw_label == 2:    # Down
            y = 1
        else:
            raise RuntimeError("Sideway should not appear here")

        meta = {
            "future_rel_slope": self.df.iloc[t + self.future_window]["future_rel_slope"]
        }


        if self.return_meta:
            return X, y, meta
        else:
            return X, y


