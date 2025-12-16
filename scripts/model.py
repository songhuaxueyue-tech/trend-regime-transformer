import torch
import torch.nn as nn

from scripts.embedding import FeatureEmbedding, PositionalEncoding


class RegimeTransformer(nn.Module):
    """
    轻量级 Regime 分类 Transformer（Day 3 骨架）
    """
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        num_classes: int = 3,
    ):
        super().__init__()

        self.embed = FeatureEmbedding(feature_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        return logits: (B, num_classes)
        """
        x = self.embed(x)
        x = self.pos(x)
        x = self.encoder(x)

        # temporal pooling
        x = x.mean(dim=1)

        return self.cls_head(x)



