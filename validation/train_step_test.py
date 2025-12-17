import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.model import RegimeTransformer


def main():
    # ====== 模拟数据（替代真实 Dataset） ======
    batch_size = 8
    window = 48
    feature_dim = 6     # 例如 OHLCV + extra
    num_classes = 3

    x = torch.randn(batch_size, window, feature_dim)
    y = torch.randint(0, num_classes, (batch_size,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ====== 模型 ======
    model = RegimeTransformer(
        feature_dim=feature_dim,
        d_model=64,
        num_heads=4,
        num_layers=1,
        num_classes=num_classes
    )

    # ====== 训练组件 ======
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ====== 单步训练验证 ======
    model.train()
    for xb, yb in loader:
        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("[OK] Forward & Backward successful")
        print("logits shape:", logits.shape)
        print("loss:", float(loss))
        break


if __name__ == "__main__":
    main()
