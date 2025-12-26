import torch
import numpy as np
import pandas as pd
import os
from research.model.model import RegimeTransformer


class RegimePredictor:
    """
    RegimePredictor: 推理层封装
    """
    def __init__(self, model_path=None, device=None):
        self.window = 48
        self.feature_cols = ["open", "high", "low", "close", "volume"]
        
        # 1. 自动判定设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 2. 默认模型路径 (相对路径，基于根目录执行的假设)
        if model_path is None:
            # 假设用户在根目录运行，直接指向 research/checkpoints
            model_path = "research/checkpoints/best_model.pt"
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path} (请确保在项目根目录运行)")

        print(f"[INFO] Loading model from {model_path} on {self.device}...")
        
        # 3. 初始化模型
        self.model = RegimeTransformer(
            feature_dim=5, 
            d_model=64,
            num_heads=4, 
            num_layers=1, 
            num_classes=2,
            dropout=0.0
        ).to(self.device)
        
        # 4. 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[INFO] Model loaded successfully.")

    def preprocess(self, df: pd.DataFrame):
        # ... (这里保持之前的 Percentage Change 逻辑不变，代码省略以节省篇幅) ...
        # (同上一次回复的逻辑)
        if len(df) < self.window:
            return None
        window_df = df.iloc[-self.window:].copy()
        window_df.columns = [c.lower() for c in window_df.columns]
        vals = window_df[self.feature_cols].values.astype(np.float32)
        price_cols = vals[:, :4] 
        base_price = price_cols[0, 3] 
        if base_price == 0: base_price = 1e-8
        price_norm = (price_cols / base_price) - 1.0
        if vals.shape[1] > 4:
            vol_col = vals[:, 4:]
            vol_mean = vol_col.mean(axis=0, keepdims=True)
            if vol_mean == 0: vol_mean = 1.0
            vol_norm = vol_col / (vol_mean + 1e-8)
            x_norm = np.concatenate([price_norm, vol_norm], axis=1)
        else:
            x_norm = price_norm
        x_tensor = torch.from_numpy(x_norm).float().unsqueeze(0).to(self.device)
        return x_tensor

    def predict(self, df: pd.DataFrame):
        x = self.preprocess(df)
        if x is None: return -1
        with torch.no_grad():
            logits = self.model(x)
            pred = logits.argmax(dim=1).item()
        return pred