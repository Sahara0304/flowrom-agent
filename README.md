# FlowROM-Agent

**Autonomous Scientific Discovery Harness for Fluid Reduced-Order Modeling**

面向科学计算任务构建的自主科研发现 Agent Harness。项目以流体降阶建模（Reduced-Order Modeling, ROM）为具体应用场景，探索如何利用多 Agent、LLM、自动代码生成与实验验证机制，构建一个能够**提出研究假设 → 设计算法 → 生成实现 → 执行实验 → 验证结果 → 推进下一轮研究**的科研工作流。

> **Project Status:** Prototype / Research Project
> **Focus:** Agent Harness · Scientific Computing · Autonomous Research · LLM-based Workflow Automation

---

## Overview

传统科学机器学习工作流通常需要研究者手动完成：

```text
文献调研
   ↓
提出假设
   ↓
设计算法
   ↓
编写代码
   ↓
运行实验
   ↓
分析结果
   ↓
提出下一轮假设
```

FlowROM-Agent 尝试将这一过程组织成一个可执行的 Agent Harness：

```text
                    ┌──────────────────┐
                    │  Research Agent  │
                    │  研究假设生成    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Algorithm Agent  │
                    │ 算法设计与约束    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Code Agent    │
                    │ 自动代码生成      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     Verifier     │
                    │ 执行与科学验证    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Research State   │
                    │ 实验结果与失败诊断 │
                    └────────┬─────────┘
                             │
                             └──────────→ Research Agent
```

核心目标不是简单地“让 LLM 写代码”，而是建立一个能够管理 **状态、能力、工具、实验和失败恢复** 的科研执行环境。

---

## Key Features

### 1. Multi-Agent Scientific Loop

当前原型包含：

* **Research Agent**：根据研究状态、历史假设与实验诊断生成新的研究假设
* **Algorithm Agent**：将研究假设转化为结构化 Algorithm Contract
* **Code Agent**：根据 Contract 生成候选算法实现
* **Verifier**：执行 Smoke Test、实验与结果验证，并将结果反馈给 Research State

形成：

```text
Hypothesis
    ↓
Algorithm Contract
    ↓
Implementation
    ↓
Experiment
    ↓
Verification
    ↓
Research State
    ↓
Next Hypothesis
```

---

### 2. Algorithm Contract

为了避免 LLM 生成的研究假设与实际执行环境不匹配，项目引入结构化的 **Algorithm Contract**。

Contract 用于描述：

* 候选算法类型
* 数据表示
* latent dimension
* 所需输入
* 历史状态需求
* 动力学机制
* 实验类型
* 数据访问策略
* 实现要求

示意：

```text
Research Hypothesis
        ↓
Algorithm Contract
        ↓
Capability Check
        ↓
Executable / Not Implementable
```

这使得系统能够区分：

```text
科学假设不可执行
          ≠
代码执行失败
          ≠
实验结果否定假设
```

---

### 3. Capability-aware Execution

项目维护独立的 **Capability Profile**，描述当前实验基础设施能够支持什么。

当前基础表示主要包括：

```text
POD latent representation
latent dimension = 52
development train / validation
```

系统会根据候选假设所需能力进行检查，例如：

```text
required capability
        ↓
Capability Profile
        ↓
compatible?
   /          \
 yes            no
 ↓              ↓
execute       not_implementable
```

因此，无法由当前实验环境验证的高级研究假设不会被错误地伪装成实验成功。

---

### 4. Automated Code Generation & Verification

候选算法进入实现阶段后，自动执行：

```text
LLM Code Generation
        ↓
Static Validation
        ↓
Smoke Test
        ↓
Training / Validation
        ↓
Rollout Evaluation
        ↓
Verifier
```

当前支持的验证思路包括：

* Python 静态检查
* Candidate Model interface 检查
* latent dimension 检查
* 一步预测误差
* 多步 rollout error
* baseline comparison
* 实验结果归档

---

### 5. Data Isolation

科研发现阶段严格区分：

```text
Development Train
Development Validation
Final Test
```

其中：

```text
Development Train
Development Validation
        ↓
允许 Research / Algorithm / Experiment 使用

Final Test
        ↓
Discovery 阶段禁止访问
```

目标是避免在自动算法发现过程中产生测试集泄漏。

---

### 6. LLM Reliability & Recovery

LLM/API 被视为一个可能失败的外部依赖，而不是可靠的同步组件。

项目目前提供统一的 MiniMax API 调用层，包含：

* timeout
* retry
* transient failure handling
* API error handling
* failure state recording

例如：

```text
HTTP 529
   ↓
Retry
   ↓
Timeout
   ↓
Retry
   ↓
HTTP 200
```

单次 LLM/API 故障不会必然终止整个科研循环。

---

## Scientific Workflow

当前项目使用流体快照作为科学计算输入。

整体流程：

```text
VTU Snapshots
      ↓
Data Inspection
      ↓
POD
      ↓
52D Latent Representation
      ↓
Latent Trajectory
      ↓
Research Hypothesis
      ↓
Algorithm Contract
      ↓
Implementation
      ↓
Experiment
      ↓
Verifier
      ↓
Research State
      ↓
Next Research Hypothesis
```

当前基础动力学模型包括：

```text
Global Linear Dynamics

z_{t+1} = A z_t
```

以及围绕 latent dynamics 进一步探索的：

```text
Residual dynamics
State-dependent operators
Structured operators
Nonlinear latent dynamics
Diagnostic experiments
```

---

## Example Research Loop

Research Agent 可以提出类似：

```text
State-dependent operator interpolation
```

然后系统执行：

```text
Hypothesis
    ↓
Algorithm Contract
    ↓
Capability Check
    ↓
Code Generation
    ↓
Smoke Test
    ↓
Development Validation
    ↓
Rollout Evaluation
    ↓
Scientific Verdict
```

如果假设无法由当前基础设施执行，则记录：

```text
not_implementable
```

而不是将其错误归类为实验失败。

---

## Evaluation

项目当前主要关注：

### Predictive Evaluation

比较：

```text
Candidate Model
Global Linear Baseline
Persistence Baseline
```

并计算：

* one-step relative error
* multi-step rollout error

例如：

```text
H = 1
H = 5
H = 10
H = 20
H = 40
H = 80
```

用于观察长期预测稳定性与误差累积。

### Scientific Verification

Verifier 不只判断：

```text
程序是否运行成功
```

还记录：

```text
hypothesis
implementation
experiment
metrics
failure diagnostics
scientific verdict
```

使实验结果能够进入下一轮 Research State。

---

## Project Structure

```text
flowrom-agent/
│
├── agents/
│   ├── orchestrator.py
│   ├── research_agent.py
│   ├── algorithm_agent.py
│   ├── code_agent.py
│   └── verifier_agent.py
│
├── runtime/
│   ├── state.py
│   ├── checkpoint.py
│   ├── context.py
│   ├── budget.py
│   ├── sandbox.py
│   ├── algorithm_contract.py
│   └── capability_profile.json
│
├── tools/
│   ├── minimax_client.py
│   ├── vtu_tool.py
│   ├── train_tool.py
│   ├── evaluate_tool.py
│   ├── report_tool.py
│   ├── pod_analysis.py
│   ├── latent_analysis.py
│   ├── regime_analysis.py
│   └── ...
│
├── research/
│   ├── papers/
│   ├── gaps/
│   ├── candidates/
│   └── history/
│
├── algorithms/
│   ├── baseline/
│   └── discovered/
│
├── experiments/
│   └── runs/
│
├── scripts/
│   ├── run_discovery.py
│   ├── run_algorithm.py
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## LLM Backend

当前项目使用 MiniMax API 作为 LLM backend。

默认模型：

```text
MiniMax-M3
```

API endpoint：

```text
https://api.minimaxi.com/v1/text/chatcompletion_v2
```

推荐使用环境变量配置：

```powershell
$env:MINIMAX_API_KEY="YOUR_API_KEY"
$env:MINIMAX_BASE_URL="https://api.minimaxi.com"
$env:MINIMAX_MODEL="MiniMax-M3"
```

**不要将 API Key 写入代码或提交到 GitHub。**

---

## Quick Start

### 1. 创建虚拟环境

```powershell
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 配置 LLM

```powershell
$env:MINIMAX_API_KEY="YOUR_API_KEY"
$env:MINIMAX_BASE_URL="https://api.minimaxi.com"
$env:MINIMAX_MODEL="MiniMax-M3"
```

### 4. 测试 LLM Client

```powershell
python -m scripts.test_minimax
```

### 5. 运行 Autonomous Discovery

```powershell
python scripts/run_discovery.py --cycles 1 --hypotheses 3
```

示意：

```text
========================================
AUTONOMOUS RESEARCH CYCLE
========================================

[ResearchAgent] Generated 3 hypotheses.

[AlgorithmAgent] ...
[CodeAgent] ...
[Verifier] ...

FLOWROM AUTONOMOUS SCIENTIFIC DISCOVERY
```

---

## Data Policy

研究阶段使用：

```text
development_train
development_validation
```

最终测试集：

```text
final_test
```

在自主发现阶段禁止访问。

原始 `.vtu` 数据不建议直接提交到 GitHub 仓库。

---

## Current Status

当前项目仍处于 **Research Prototype** 阶段。

已经完成的核心能力包括：

* Multi-Agent scientific discovery loop
* Research / Algorithm / Code / Verification pipeline
* Algorithm Contract
* Capability-aware execution
* POD latent representation
* Automated code validation
* Smoke testing
* Rollout evaluation
* Research state management
* LLM retry / failure recovery
* Development / validation / final-test isolation

当前仍在持续完善：

* Literature retrieval and research-gap mining
* More general scientific experiment contracts
* Diagnostic experiment planning
* Richer Agent evaluation and regression testing
* More robust execution sandboxing
* Long-horizon autonomous research

---

## Motivation

这个项目的最终目标并不是训练一个更大的模型，而是探索：

> **如何让 Agent 从“回答科研问题”进一步走向“执行科研过程”。**

核心关注的问题包括：

```text
How should an agent represent a scientific hypothesis?

How should an agent decide whether a hypothesis is executable?

How should generated code be verified?

How should failed experiments update future research?

How should scientific agents remain reproducible and testable?

How can an LLM be embedded into a reliable execution loop?
```

FlowROM-Agent 以流体降阶建模为具体案例，尝试回答这些问题。

---

## Future Direction

下一阶段将重点研究：

```text
Scientific Experiment Planning
        ↓
Tool Selection
        ↓
Capability Upgrade
        ↓
Agent Evaluation
        ↓
Trajectory-level Verification
        ↓
Long-horizon Autonomous Discovery
```

最终希望形成一个更通用的：

**Agent Harness for Autonomous Scientific Discovery**

而不仅仅是针对单一 ROM 问题的自动建模工具。

---

## Author

**Li Han**

Tongji University
Mathematics · Computational Mathematics

Research interests:

* Agent Harness / AI Agents
* Scientific Machine Learning
* Autonomous Scientific Discovery
* Quantum Machine Learning
* Reduced-Order Modeling
* Scientific Computing
