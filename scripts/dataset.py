import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional


class OHLCVWindowDataset(Dataset):
    """
    PyTorch-friendly OHLCV window dataset.

    Each sample:
      X: Tensor of shape (window, feature_dim)
      y: int (regime label at window end)
    """

    def __init__(
        self,
        data_path: str,
        window: int = 48,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "regime",
        normalize: bool = True,
        drop_sideway: bool = False,
    ):
        """
        Args:
            data_path: path to labeled feather / csv
            window: rolling window length
            feature_cols: which columns to use as input features
            label_col: column name of regime label
            normalize: whether to normalize features per window
            drop_sideway: whether to drop regime == 0 samples
        """
        self.window = window
        self.label_col = label_col
        self.normalize = normalize
        self.drop_sideway = drop_sideway

        # -------- load data --------
        if data_path.endswith(".feather"):
            df = pd.read_feather(data_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
        else:
            df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

        df = df.sort_index()

        # default feature columns
        if feature_cols is None:
            feature_cols = ["open", "high", "low", "close", "volume"]

        self.feature_cols = feature_cols

        # basic checks
        for c in feature_cols + [label_col]:
            if c not in df.columns:
                raise ValueError(f"missing required column: {c}")

        # -------- build sample index --------
        self.df = df
        self.indices = []

        for end_idx in range(window - 1, len(df)):
            label = int(df.iloc[end_idx][label_col])
            if drop_sideway and label == 0:
                continue
            self.indices.append(end_idx)

        if len(self.indices) == 0:
            raise RuntimeError("No valid samples found, check your data or settings.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            X: Tensor (window, feature_dim)
            y: int
        """
        end_idx = self.indices[idx]
        start_idx = end_idx - self.window + 1

        window_df = self.df.iloc[start_idx:end_idx + 1]

        X = window_df[self.feature_cols].values.astype(np.float32)
        y = int(window_df[self.label_col].iloc[-1])

        # per-window normalization (important!)
        if self.normalize:
            mean = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            X = (X - mean) / std

        X = torch.from_numpy(X)
        y = torch.tensor(y, dtype=torch.long)

        return X, y
