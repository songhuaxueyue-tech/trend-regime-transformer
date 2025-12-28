import torch
import numpy as np
import pandas as pd
import os
from research.model.model import RegimeTransformer
from pathlib import Path 
import time

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
            
            
        # 2. 智能锁定模型路径 (Path-based)
        if model_path is None:
            # 方案：以当前脚本 (predictor.py) 为锚点
            # 当前位置: .../inference/predictor.py
            current_dir = Path(__file__).resolve().parent
            # 项目根目录: .../inference/../ (即 ai_logic 目录)
            project_root = current_dir.parent
            # 拼接绝对路径: .../ai_logic/research/checkpoints/best_model.pt
            model_path_obj = project_root / "research" / "checkpoints" / "best_model.pt"
            
            model_path = str(model_path_obj) # 转为字符串以防万一
            
        if not os.path.exists(model_path):
            # 打印调试信息，让你知道它到底去哪找了
            print(f"[Debug] 尝试加载路径: {model_path}")
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

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
        """
        执行推理，并打印分段延迟数据
        """
        # [Timer 1] 开始计时
        t0 = time.perf_counter()
        
        # 1. 预处理
        x = self.preprocess(df)
        
        # [Timer 2] 预处理结束
        t1 = time.perf_counter()
        
        if x is None: return -1
        
        with torch.no_grad():
            # 2. 模型推理
            logits = self.model(x)
            
            # --- [专业细节] ---
            # 如果是 GPU 运行，强制等待计算完成再停止计时
            # 这能证明你懂 CPU-GPU 的异构协作机制
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            pred = logits.argmax(dim=1).item()
            
        # [Timer 3] 全部结束
        t2 = time.perf_counter()
        
        # 计算毫秒数
        prep_latency = (t1 - t0) * 1000  
        infer_latency = (t2 - t1) * 1000 
        total_latency = prep_latency + infer_latency
        
        # 打印日志
        print(f"[Profiling] Total: {total_latency:.2f}ms | Prep: {prep_latency:.2f}ms | Infer: {infer_latency:.2f}ms")
            
        return pred