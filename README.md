# Trend Regime Transformer

> **Market Regime Representation Learning for Crypto Trading Systems**  
> 基于轻量级 Transformer 的加密货币市场结构状态（Regime）识别研究

---

## 1. 项目背景

在加密货币市场中，不同 **市场结构状态（Market Regime）**  
（如趋势、震荡）对量化策略的适用性存在显著差异。

实践中常见的问题包括：

- 策略在历史回测中表现良好  
- 实盘中绩效快速退化  
- 原因往往并非单一参数失效，而是 **策略未区分市场结构状态**

本项目关注的核心问题是：

> **是否可以从 OHLCV 时间序列中，  
> 自动学习稳定、可泛化的市场结构状态（Regime），  
> 并将其作为系统级信号供量化策略使用？**

---

## 2. 项目目标与定位

### 本项目**不**追求：

- 短期价格预测  
- 单模型 accuracy / Sharpe 最大化  
- 端到端自动交易  

### 本项目关注的是：

- **结构感知型时间序列表示（Structure-aware Representation）**  
- **低频、稳健、可解释的 Regime 信号**  
- **模型信号如何被交易系统正确消费**

典型应用场景包括：

- 趋势策略的 Regime 过滤  
- 震荡行情中的风险抑制  
- 不同 Regime 下的差异化仓位与风控逻辑  

---

## 3. 设计原则（关键理念）

### 为什么优先构建 Dataset / DataLoader，而不是堆模型？

在金融时间序列任务中：

- **数据切片方式本身就是模型的一部分**
- 窗口对齐、标签定义、时间因果性错误  
  会导致模型在“错误结构”上收敛

因此本项目遵循以下原则：

> **先保证数据结构与标签逻辑的可靠性，  
> 再讨论模型复杂度与性能。**

当前阶段重点放在：

- 数据工程的可验证性  
- 标签逻辑的可解释性  
- 工程结构的可维护性  

---

## 4. 当前阶段成果（已完成）

本项目已完成**工程与数据层的完整闭环**，并验证模型可成功接入真实交易框架。

### 已完成内容

#### 4.1 数据集与窗口化设计
- 基于 OHLCV 构建滑动窗口样本  
- 明确时间对齐规则，避免未来信息泄露  
- 支持固定时间尺度（如 1h）序列输入  

#### 4.2 Dataset / DataLoader 实现
- 基于 PyTorch `Dataset` / `DataLoader`  
- 支持 batch / shuffle / drop_last  
- 与 Transformer Encoder 输入格式完全对齐  

#### 4.3 数据与标签 Sanity Check
- 系统性校验：  
  - 输入 / 标签 shape  
  - 时间窗口长度  
  - batch 维度  
  - 特征维度  
- 提供独立校验脚本，避免 silent bug  

#### 4.4 轻量 Transformer Encoder
- Encoder-only 架构  
- 小规模参数设计（2 layers / 4 heads）  
- 支持 Regime 分类任务（多分类）  

#### 4.5 系统集成验证
- 模型推理结果可成功接入 **Freqtrade 策略框架**  
- 完成以下对比实验：  
  - 原始策略  
  - 引入 Regime 信号后的策略  
- 验证模型信号对系统行为产生真实影响  

---

## 5. 项目结构说明

```text
trend-regime-transformer/
├── scripts/                 # 核心实现
│   ├── dataset.py           # OHLCV 窗口化 Dataset
│   ├── embedding.py         # 特征映射与位置编码
│   ├── model.py             # 轻量 Transformer Encoder
│   ├── label_generator.py   # Regime 标签生成逻辑
│   ├── label_validation.py  # 标签分布与合理性验证
│   ├── train_step_test.py   # 单步训练 / 推理校验
│   └── __init__.py
│
├── validation/              # 数据完整性校验脚本
│   ├── dataloader_check.py
│   └── dataset_sanity_check.py
│
├── data/                    # 原始市场数据（忽略，不入库）
├── data_out/                # 中间产物（忽略）
├── _sanity_output/          # 可视化 / 检查输出（忽略）
│
├── .gitignore
├── README.md
└── end                      # 阶段完成标记
```

---

## 6. 实验结论与当前边界

### 当前观察到的关键现象

- 模型能够学习到一定的结构性信息
    
- Regime 信号在系统层面**可被正确调用**
    
- 但 **模型信号 ≠ 策略绩效提升**
    

这促使项目进一步聚焦于一个更底层的问题：

> **模型是否能被系统“精细表达和消费”，  
> 而不仅仅是被作为一个布尔条件使用。**

---

## 7. 当前研究边界说明

- 本项目**不追求短期交易绩效最优**
    
- Regime 标签本身可能存在噪声
    
- 当前结果更偏向 **系统认知与结构验证**
    

项目的核心价值在于：

> **明确模型、策略与系统之间的责任边界。**

---

## 8. 后续可能方向（未实现）

以下方向被明确记录，但**未在当前阶段展开**：

- 多时间尺度（multi-scale）结构建模
    
- Volatility-aware embedding
    
- 自监督 / 对比学习方式学习 market state
    
- Regime 与真实 PnL 的弱监督对齐
    
- 系统级决策表达机制设计
    

---

## 9. 项目状态

> **当前阶段已完成并封存。**

本仓库用于：

- 记录一次完整的工程与系统认知过程
    
- 支撑后续研究与复试讨论
    
- 不作为短期交易系统使用

---
