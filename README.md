# Trend Regime Transformer

> 基于轻量级 Transformer 的加密货币市场 **趋势 / 震荡 Regime 结构识别研究项目**

---

## 📌 项目背景

在加密货币市场中，**趋势行情**与**震荡行情**对交易策略的适用性有本质差异。  
许多量化策略在回测中表现良好，但在实盘中失效，核心原因之一是 **未对市场 Regime（结构状态）进行区分**。

本项目旨在探索：

> **能否通过深度学习模型，从 OHLCV 时间序列中自动学习市场的结构状态（Regime），并为后续量化策略提供稳定、可泛化的结构信号。**

---

## 🎯 当前阶段目标（已完成）

过去两天，本项目聚焦在**工程与数据层的可靠性构建**，而非直接建模：

### ✅ 已完成内容

- **数据集设计**
  - 基于 OHLCV 数据构建滑动窗口（window-based）样本
  - 支持按固定时间尺度（如 1h）生成序列输入
- **Dataset / DataLoader 实现**
  - 使用 PyTorch `Dataset` / `DataLoader`
  - 支持 batch、shuffle、drop_last
- **数据完整性与形状验证（Sanity Check）**
  - 验证 `x / y` 形状是否符合模型输入预期
  - 验证窗口长度、batch 维度、特征维度
- **基础工程规范**
  - 明确区分：
    - 代码（tracked）
    - 原始数据 / 中间产物（ignored）
    - 运行产物（ignored）
  - 合理配置 `.gitignore`

---

## 📁 项目结构说明

```text
trend-regime-transformer/
├── scripts/                 # 核心代码（当前重点）
│   ├── dataset.py           # OHLCV 窗口化 Dataset 定义
│   ├── label_generator.py   # Regime / 标签生成逻辑（初版）
│   ├── label_validation.py  # 标签合理性验证
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

## 🧠 设计理念（当前阶段）
为什么先做 Dataset / DataLoader，而不是模型？
- 在时间序列 + 金融数据任务中：
- 数据切片方式本身就是模型的一部分
- 错误的窗口、对齐或标签，会导致模型“学到错误规律”

因此本项目遵循：
> 先保证数据结构正确 → 再谈模型复杂度


