# scripts/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from scripts.dataset import OHLCVWindowDataset
from scripts.model import RegimeTransformer


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 超参数 =====
    batch_size = 64
    lr = 1e-3
    epochs = 20
    window = 48
    num_classes = 3

    # ===== 数据路径（你真实的数据）=====
    train_path = "other/data_split/train.feather"           # 包含 date, open, high, low, close, volume, label 列前70% 的数据文件
    val_path = "other/data_split/val.feather"               # 包含 date, open, high, low, close, volume, label 列后30% 的数据文件

    train_dataset = OHLCVWindowDataset(
        data_path=train_path,
        window=window,
        normalize=True,
        drop_sideway=False,
    )

    val_dataset = OHLCVWindowDataset(
        data_path=val_path,
        window=window,
        normalize=True,
        drop_sideway=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ===== 模型 =====
    model = RegimeTransformer(
        feature_dim=len(train_dataset.feature_cols),
        d_model=64,
        num_heads=4,
        num_layers=1,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

    print("Training finished.")


if __name__ == "__main__":
    main()



