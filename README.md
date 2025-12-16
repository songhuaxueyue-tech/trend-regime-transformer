# Trend Regime Transformer

> 基于轻量级 Transformer 的加密货币市场  
> **趋势 / 震荡 Regime 结构识别研究项目**

---

## 📌 项目背景

在加密货币市场中，**趋势行情（Trend）** 与 **震荡行情（Range / Sideway）**  
对量化交易策略的适用性存在本质差异。

现实中常见的问题是：

- 策略在历史回测中表现良好  
- 但在实盘中迅速失效  
- 核心原因之一：**未对市场结构状态（Market Regime）进行区分**

本项目关注的核心问题是：

> **是否可以通过深度学习模型，  
从 OHLCV 时间序列中自动学习市场的结构状态（Regime），  
并为后续量化策略提供稳定、可泛化的结构信号？**

---

## 🎯 项目目标

本项目的长期目标不是单纯“预测价格”，而是：

- 构建 **结构感知型市场表示**
- 输出 **低频、稳健、可解释的 Regime 信号**
- 服务于 **Regime-based 量化策略设计**

典型应用包括：

- 趋势过滤（只在趋势期启用趋势策略）
- 震荡过滤（避免趋势策略在区间内失效）
- 不同 Regime 下的差异化风控与仓位逻辑

---

## 🧠 设计理念（重要）

### 为什么优先做 Dataset / DataLoader，而不是模型？

在时间序列与金融数据任务中：

- **数据切片方式本身就是模型的一部分**
- 错误的窗口设计、对齐方式或标签逻辑  
  会导致模型“在错误结构上收敛”

因此本项目遵循以下原则：

> **先保证数据结构正确 → 再谈模型复杂度**

当前阶段重点放在：

- 数据工程可靠性  
- 标签逻辑可解释性  
- 工程结构可维护性  

---

## 🚧 当前阶段进展（已完成）

过去阶段，本项目聚焦在**工程与数据层的可靠性构建**：

### ✅ 已完成内容

#### 1. 数据集设计
- 基于 OHLCV 构建 **滑动窗口（window-based）样本**
- 支持固定时间尺度（如 1h）序列输入
- 明确时间对齐规则，避免未来信息泄露

#### 2. Dataset / DataLoader 实现
- 使用 PyTorch `Dataset` / `DataLoader`
- 支持：
  - batch
  - shuffle
  - drop_last
- 与后续 Transformer Encoder 输入格式完全对齐

#### 3. 数据完整性与形状验证（Sanity Check）
- 验证：
  - `x / y` 形状
  - 窗口长度
  - batch 维度
  - 特征维度
- 提供独立校验脚本，避免 silent bug

#### 4. 基础工程规范
- 明确区分：
  - 代码（tracked）
  - 原始数据 / 中间产物（ignored）
  - 运行与可视化产物（ignored）
- 合理配置 `.gitignore`
- 保证仓库 **可复现、可审阅**

---

## 📁 项目结构说明

```text
trend-regime-transformer/
├── scripts/                 # 核心代码（当前重点）
│   ├── dataset.py           # OHLCV 窗口化 Dataset 定义
│   ├── embedding.py         # 输入 embedding / 位置编码
│   ├── model.py             # Transformer Encoder（轻量版）
│   ├── label_generator.py   # Regime / 标签生成逻辑（初版）
│   ├── label_validation.py  # 标签合理性与分布验证
│   ├── train_step_test.py   # 单步 forward / backward 测试
│   └── __init__.py
│
├── validation/              # 数据与 DataLoader 校验脚本
│   ├── dataloader_check.py
│   └── dataset_sanity_check.py
│
├── data/                    # 原始市场数据（已忽略，不入库）
├── data_out/                # 中间处理结果（已忽略）
├── _sanity_output/          # 可视化 / 检查产物（已忽略）
│
├── .gitignore               # 忽略规则（数据、缓存、产物）
├── README.md
└── end                      # 项目阶段标记文件
```


## 🗺️ 项目计划（15 天阶段任务）
| 天数 | 任务（每日具体产出）                                                       |
| -- | ---------------------------------------------------------------- |
| 1  | 数据采集 + 初始清洗；编写标签生成器（slope / ATR 规则）并生成初版标签；样本索引器                 |
| 2  | 检查标签质量（分布统计、样本可视化）；修正阈值 α / β；准备 PyTorch Dataset / DataLoader    |
| 3  | 编写 embedding 层与位置编码；准备训练流水线（train / val / test split）            |
| 4  | 实现轻量 Transformer Encoder（2 层、4 head、小 FFN）；打通 forward；单步训练验证     |
| 5  | 模型训练（核心任务：3 类分类）；监控 loss / acc；保存 checkpoint                     |
| 6  | （可选）辅助任务：未来收益方向或序列排序恢复；联合训练实验                                    |
| 7  | 模型评估：混淆矩阵、Precision / Recall / F1、per-class accuracy；生成 baseline |
| 8  | 提取 embedding；t-SNE / UMAP 可视化；检查类别聚类与分离度                         |
| 9  | 模型打包与推理脚本（单序列 → regime）；准备模型导出（TorchScript 可选）                   |
| 10 | 集成至本地 Freqtrade 策略模板：`populate_indicators()` 返回 regime           |
| 11 | 设计 regime-based 策略逻辑（震荡过滤 / 趋势允许开仓 / 反转提前止盈）；回测初稿                |
| 12 | 回测对比与 ablation（无 regime vs 有 regime）；产出关键图表                      |
| 13 | 调参（模型 / 阈值 / 策略权重）；稳定性测试（不同币对、不同时间段）                             |
| 14 | 准备展示材料：PPT、口述稿、实验总结（亮点 / 局限 / 下一步）                               |
| 15 | 项目收尾与打包（README、运行说明、模型文件）；最终回测确认                                 |



## ⚠️ 当前研究边界与说明
- 本项目 不追求短期预测精度最大化
- 更关注：
  - 结构可解释性
  - 稳定性
  - 跨时间段泛化能力
- Regime 标签本身可能存在噪声
本项目目标是学习 “可操作的结构信号”，而非完美标签


## 🔮 后续可能扩展方向
- 引入多时间尺度（multi-scale）输入
- 使用 volatility-aware embedding
- 自监督 / 对比学习方式学习 market state
- 与真实交易绩效（PnL / 回撤）的弱监督对齐

