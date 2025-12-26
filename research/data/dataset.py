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

            # 拿到原始数值 (numpy array)
            X_raw = window_df[self.feature_cols].values.astype(np.float32)
            
            # 获取标签
            raw_label = int(window_df[self.label_col].iloc[-1])

            # Re-map original labels {Up=1, Down=2} -> binary labels {Up=0, Down=1}
            if raw_label == 1:      # Up
                y = 0
            elif raw_label == 2:    # Down
                y = 1
            else:
                raise ValueError("Sideway label should not appear when drop_sideway=True")

            # ==========================================
            # 新的归一化逻辑：Percentage Change (保留波动幅度)
            # ==========================================
            
            # 假设 feature_cols 前4列是 open, high, low, close
            # 如果你的 feature_cols 顺序变了，这里需要调整
            price_cols = X_raw[:, :4]  
            
            # 获取窗口起点的价格作为基准 (base_price)
            base_price = price_cols[0, 3] # 使用第一根K线的 close 作为基准

            # 计算相对于基准价格的涨跌幅
            # 结果例如：0.0 (起点), 0.01 (涨1%), -0.02 (跌2%)...
            # 这样暴涨和微涨的数值大小就不同了，模型能看清“幅度”
            price_norm = (price_cols / (base_price + 1e-8)) - 1.0 

            # 对 Volume 单独归一化 (除以窗口内的平均值)
            # 假设第5列是 volume
            if X_raw.shape[1] > 4:
                vol_col = X_raw[:, 4:]
                vol_norm = vol_col / (vol_col.mean(axis=0, keepdims=True) + 1e-8)
                # 拼接回去
                X_final = np.concatenate([price_norm, vol_norm], axis=1)
            else:
                X_final = price_norm

            # 转换为 Tensor (只做一次)
            X = torch.from_numpy(X_final).float()
            y = torch.tensor(y, dtype=torch.long)

            return X, y

