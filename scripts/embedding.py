import math
import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """
    将原始 OHLCV 特征映射到 Transformer 的 d_model 维度
    """
    def __init__(self, feature_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return: (B, T, D)
        """
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """
    标准 Sinusoidal Positional Encoding
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 不作为参数，仅作为 buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len]



