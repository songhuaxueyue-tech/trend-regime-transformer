# scripts/model.py

import torch
import torch.nn as nn

from scripts.embedding import FeatureEmbedding, PositionalEncoding


class RegimeTransformer(nn.Module):
    """
    RegimeTransformer

    功能说明：
    - 输入：OHLCV 等时间序列特征 (B, T, F)
    - 表示层：FeatureEmbedding + PositionalEncoding
    - 建模层：Transformer Encoder（建模时间依赖）
    - 聚合层：时间维 mean pooling
    - 输出层：市场状态（regime）分类 logits

    用途：
    - 用于市场状态（震荡 / 上涨 / 下跌 等）分类
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ===== 表示层 =====
        self.embed = FeatureEmbedding(
            feature_dim=feature_dim,
            d_model=d_model
        )


        self.pos = PositionalEncoding(d_model)


        # ===== Transformer Encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,   # 轻量 FFN
            dropout=dropout,
            batch_first=True               # (B, T, D)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ===== 输出层 =====
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
        - x: Tensor, shape = (B, T, F)

        返回：
        - logits: Tensor, shape = (B, num_classes)
        """

        # ---- 表示映射 ----
        x = self.embed(x)     # (B, T, D)
        x = self.pos(x)       # (B, T, D)

        # ---- 时序建模 ----
        x = self.encoder(x)   # (B, T, D)

        # ---- 时间聚合 ----
        x = x.mean(dim=1)     # (B, D)

        # ---- 分类输出 ----
        logits = self.cls_head(x)  # (B, C)

        return logits
