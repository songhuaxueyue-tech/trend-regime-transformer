import matplotlib.pyplot as plt
import numpy as np

# 1. 设置学术风格 (如果没有 seaborn 可以注释掉这一行，或者 pip install seaborn)
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
except ImportError:
    pass

# 2. 构造数据
# X轴：监控币种数量
assets = np.array([1, 10, 20, 50, 80, 100])

# Y轴 (Python)：模拟 GIL 锁导致的线性增长
# 假设单次 2.2ms (基于你的实测)
py_base_latency = 2.2 
py_latency = assets * py_base_latency 

# Y轴 (C++ Goal)：模拟多线程并行
# 假设基础开销 0.5ms，每增加一个币种只增加微小的上下文切换开销
cpp_base_latency = 0.5
cpp_latency = cpp_base_latency + (assets * 0.05) # 极平缓的增长

# 3. 绘图
plt.figure(figsize=(10, 6))

# 画 Python 线 (红色，表示危险/现状)
plt.plot(assets, py_latency, 'o-', color='#e74c3c', linewidth=2.5, label='Current (Python + GIL)')

# 画 C++ 线 (蓝色，表示理想/目标)
plt.plot(assets, cpp_latency, 's-', color='#2980b9', linewidth=2.5, label='Goal (C++ Multithreading)')

# 4. 添加"死亡线" (100ms Threshold)
plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
plt.text(5, 105, '100ms Latency Limit (High-Frequency Boundary)', color='gray', fontsize=10, style='italic')

# 5. 关键点标注 (Highlight the Gap)
# 标注 50 个币种时的巨大差距
plt.annotate(f'{py_latency[3]:.1f}ms\n(Unacceptable)', 
             xy=(50, py_latency[3]), xytext=(40, py_latency[3]+40),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)

plt.annotate(f'~{cpp_latency[3]:.1f}ms', 
             xy=(50, cpp_latency[3]), xytext=(55, cpp_latency[3]-15),
             color='#2980b9', fontsize=11)

# 6. 图表装饰
plt.title('System Latency Scaling Analysis: Python vs C++', fontsize=16, pad=20, weight='bold')
plt.xlabel('Number of Monitored Assets (Pairs)', fontsize=12)
plt.ylabel('Total Inference Latency (ms)', fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)

# 7. 保存
plt.tight_layout()
plt.savefig('latency_scaling.png', dpi=300)
print("图表已生成：latency_scaling.png")
plt.show()