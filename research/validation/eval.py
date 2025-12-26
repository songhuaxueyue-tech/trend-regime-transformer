# scripts/eval.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from scripts.data.dataset_future import FutureRegimeDataset
from scripts.model.model import RegimeTransformer


@torch.no_grad()
def evaluate(model, loader, device):
    """
    推理整个数据集，返回 y_true, y_pred
    """
    model.eval()
    y_true = []
    y_pred = []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def naive_baseline_accuracy(y_true):
    """
    baseline：永远预测出现频率最高的类别
    """
    values, counts = np.unique(y_true, return_counts=True)
    majority_class = values[np.argmax(counts)]
    y_pred = np.full_like(y_true, fill_value=majority_class)
    return accuracy_score(y_true, y_pred), majority_class


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 参数 =====
    batch_size = 64
    window = 48
    num_classes = 2

    # ===== 数据路径 =====
    val_path = "other/data_split/val.feather"

    # ===== Dataset / Dataloader =====
    val_dataset = FutureRegimeDataset(
        data_path=val_path,
        past_window=window,
        future_window=window,
        normalize=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # ===== 模型 =====
    model = RegimeTransformer(
        feature_dim=len(val_dataset.feature_cols),
        d_model=64,
        num_heads=4,
        num_layers=1,
        num_classes=num_classes,
    ).to(device)

    # 加载最优 checkpoint
    ckpt_path = "checkpoints/best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    # ===== 推理 =====
    y_true, y_pred = evaluate(model, val_loader, device)

    # ===== 指标 =====
    print("\n===== Classification Report =====")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Up", "Down"],
            digits=4
        )
    )

    print("\n[Key Metric] Macro F1:",
      classification_report(y_true, y_pred, output_dict=True)["macro avg"]["f1-score"])


    # ===== 混淆矩阵 =====
    cm = confusion_matrix(y_true, y_pred)
    print("===== Confusion Matrix =====")
    print(cm)

    # ===== Baseline =====
    baseline_acc, majority_class = naive_baseline_accuracy(y_true)
    model_acc = accuracy_score(y_true, y_pred)

    print("\n===== Baseline Comparison =====")
    print(f"Majority class baseline accuracy: {baseline_acc:.4f}")
    print(f"Majority class label: {majority_class}")
    print(f"Model accuracy: {model_acc:.4f}")
    print(f"Improvement over baseline: {model_acc - baseline_acc:.4f}")


if __name__ == "__main__":
    main()
