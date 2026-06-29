# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个多智能体视觉-语言导航（VLN）系统，基于 DiscussNav、MSNav 和 Multi-agent Architecture Search via Agentic Supernet 论文的思路。该系统使用 Habitat 模拟环境实现了分层导航架构。

## 要求

使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。如果目标清晰但是路径不是最短，告诉我，并且建议更好的办法

1.不能写兼容性代码，除非我主动要求
2.需求模糊时，先提问澄清再写代码
3.每次被纠正后，反思并制定不再犯的计划
4.始终使用中文思考与回答


## 常用命令

### 安装
```bash
pip install -e .
```

### 测试
```bash
pytest tests/                          # 运行所有测试
pytest tests/test_agents.py            # 运行指定测试文件
pytest tests/ -v                       # 详细输出模式
```

### 训练
```bash
python scripts/train.py --config configs/default.yaml --episodes 1000
```

### 评估（实际主入口）
```bash
# 使用远程LLM服务器
python run_vln_experiment.py --use-remote-llm --llm-server http://localhost:8000 --episodes 5

# 紧急导航评估
python scripts/run_emergency_eval.py --exp baseline --episodes 10 --use-remote-llm --llm-server http://localhost:8000
```

### 评估（备用入口）
```bash
python scripts/evaluate.py --config configs/default.yaml --episodes 100 --scenes data/scene_datasets/
```

### 推理
```bash
python scripts/inference.py --instruction "turn left and go to the kitchen"
```

## 系统架构

### 当前实际架构（重要）

**主实验入口**: `run_vln_experiment.py`（根目录）直接管理所有agents，**不使用Supernet/VLNNavigator**。

```
run_vln_experiment.py (主实验)
    ↓ 直接创建
agents/InstructionAgent, PerceptionAgent, TrajectoryAgent, DecisionAgent
    ↓ 直接调用
strategies/ReAct, CoT, Debate, Reflection
```

**备用入口点**: scripts/inference.py等使用VLNNavigator架构（设计但实际实验未使用）。

```
scripts/inference.py (备用)
    ↓
VLNNavigator → Supernet → Agents + Strategies
```

### 分层结构
- **弱层（Weak level）**：本地小模型处理简单任务（Type-0）
- **强层（Strong level）**：多智能体协作处理复杂任务（Type-1 至 Type-4）

### 任务类型分类
系统根据复杂度将导航任务分为 5 类：
| 类型 | 描述 | 智能体 | 策略 |
|------|------|--------|------|
| Type-0 | 单步指令 | 无 | 无 |
| Type-1 | 走廊导航 | perception, decision | ReAct |
| Type-2 | 物体查找 | perception, trajectory, decision | ReAct, CoT |
| Type-3 | 跨房间导航 | 全部智能体 | CoT, Reflection |
| Type-4 | 模糊场景 | 全部智能体 | CoT, Debate, Reflection |

### 核心组件流程
1. `VLNNavigator` (core/navigator.py) - 主入口点，协调导航流程
2. `TaskTypeClassifier` (classifiers/task_classifier.py) - 确定任务复杂度
3. `Supernet` (supernet/supernet.py) - 选择并执行智能体-策略组合
4. `Agents` (agents/) - 专用智能体：InstructionAgent、PerceptionAgent、TrajectoryAgent、DecisionAgent
5. `Strategies` (strategies/) - 执行策略：ReAct、CoT、Debate、Reflection
6. `FailureHandler` (fallback/failure_handler.py) - 失败时的级联降级处理

### 主要模块结构
- `core/` - 上下文管理、动作定义、导航器（VLNNavigator备用）
- `agents/` - BaseAgent 抽象类及其专用实现
- `strategies/` - BaseStrategy 及策略实现（ReAct、CoT、Debate、Reflection）
- `supernet/` - 架构编排（备用，主实验不使用）
- `classifiers/` - 任务类型分类（规则-based + LLM 回退）
- `models/` - 模型实现（LocalModel、LLMModel、VisualEncoder）
- `environment/` - Habitat 环境封装器
- `fallback/` - 失败处理（备用，主实验不使用）
- `configs/` - YAML 和 Python 配置文件
- `emergency/` - 紧急导航处理（EmergencyDetector、PathReplanner）
- `evaluation/` - 评估脚本

## 配置说明

主配置文件：`configs/default.yaml`

主要配置项：
- `navigation` - 最大步数、停止距离、转向角度
- `classifier` - 任务分类阈值
- `supernet` - 架构搜索与自适应选择
- `agents` - 各智能体设置（超时时间、并行执行）
- `strategies` - 策略特定参数（最大迭代次数、共识阈值）
- `optimization` - 各任务类型的 Token 预算、上下文压缩
- `fallback` - 级联降级层级

## 环境设置

项目集成了位于 `src/habitat-lab-0-3-3/` 的 Habitat 模拟环境。Habitat 依赖为可选项，可单独安装：
```bash
# habitat-sim 和 habitat-lab 为可选依赖
# 安装说明请参考 src/habitat-lab-0-3-3/
```

## 命令行入口点

setup.py 中定义的命令行脚本：
- `vln-train` - 训练
- `vln-eval` - 评估
- `vln-infer` - 推理

## 测试方法

测试使用 pytest 配合组件 mock。测试文件遵循以下模式：
- `test_agents.py` - 智能体单元测试与集成测试
- `test_classifiers.py` - 任务分类器测试
- `test_strategies.py` - 策略执行测试
- `test_system_full.py` - 全系统集成测试
- `test_vln_mock.py` - Mock 导航测试