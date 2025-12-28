# 1. 基于 FreqAI 官方镜像 (作为底座)
FROM freqtradeorg/freqtrade:develop_freqai

# 2. 切换到 root 用户以安装软件
USER root

# 3. 强制安装 PyTorch 和其他可能缺失的核心库
# --no-cache-dir 减小体积，--upgrade 确保最新
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio numpy pandas scikit-learn joblib

# 4. 切回 freqtrade 用户 (安全最佳实践)
USER ftuser
