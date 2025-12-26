import pandas as pd
import numpy as np
# 直接导入，依靠 python -m 机制
from inference.predictor import RegimePredictor

def test_random_data():
    print("===== 开始测试: 随机数据输入 =====")
    # 构造假数据
    dates = pd.date_range(start="2025-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "open": np.random.rand(100) * 100 + 10000,
        "high": np.random.rand(100) * 100 + 10000,
        "low": np.random.rand(100) * 100 + 10000,
        "close": np.random.rand(100) * 100 + 10000,
        "volume": np.random.rand(100) * 500
    }, index=dates)
    
    # 初始化
    predictor = RegimePredictor() 
    
    # 预测
    pred = predictor.predict(df)
    print(f"预测结果: {pred}")

if __name__ == "__main__":
    test_random_data()




