# CLAUDE.md

本文件为 Claude Code 在此代码库中工作时提供指导。

## 项目概述

Pipeline VLN 导航系统：基于 Habitat 模拟器的 6-Agent 流水线架构，使用三层异构模型部署。

## 常用命令

### 安装
```bash
pip install -e .
```

### 启动多模型 vLLM 服务
```bash
bash scripts/start_vllm_multi.sh          # 全部3个模型
bash scripts/start_vllm_multi.sh --no-35b # 跳过35B（节省显存）
```

### 运行实验
```bash
conda activate Habitat
python run_vln_experiment.py --use-remote-llm --episodes 10 --seed 42
```

## 系统架构

### Pipeline 流程（唯一架构）
```
Navigator (编排器 + 双层难度分级)
    │
    ├─ SubtaskDecompositionAgent (Qwen3.5-9B, 每episode 1次)
    │     指令 → 子任务列表 + 静态难度
    │
    └─ 导航循环 (每5步=1周期):
       ├─ ObservationAgent (Qwen3-VL-8B, GPU 0:8000)
       │     RGB+Depth → 结构化JSON
       ├─ 深度图障碍检测 (规则, 零LLM)
       ├─ 动态难度分类 (规则, 零LLM)
       ├─ EmergencyAgent (Qwen3.5-9B, GPU 1:8001)
       │     仅触发时调用LLM
       ├─ AnalysisAgent (Qwen3.6-35B, GPU 2+3:8002)
       │     CoT / Debate(Light/Standard/Deep) / Reflection
       ├─ PlanningAgent (Qwen3.6-35B, GPU 2+3:8002)
       │     LLM + Topology + A* → 5个动作
       └─ ReviewAgent (Qwen3.5-9B, GPU 1:8001)
             规则优先(4种条件) + LLM辅助
```

### 三层模型部署 (4x RTX 4090)
| GPU | 模型 | 端口 | 用途 |
|-----|------|------|------|
| 0 | Qwen3-VL-8B-Instruct | 8000 | VLM 结构化感知 |
| 1 | Qwen3.5-9B-AWQ | 8001 | 快速LLM (分解/审查/应急) |
| 2+3 | Qwen3.6-35B-A3B FP8 | 8002 | 强推理LLM (CoT/Debate/规划) |

### 难度分级策略
| 最终难度 | 策略 | LLM调用 | 模型 |
|---------|------|---------|------|
| easy | 规则 (跳过LLM) | 0 | 无 |
| medium | CoT | 1 | 35B |
| hard (首次) | Debate Light | 2 | 35B |
| hard (二次) | Debate Standard | 3 | 35B |
| hard (三次+) | Debate Deep | 4-5 | 35B |

## 项目结构
```
habitat_vln/
├── agents/pipeline/         # 6个SubAgent + 工具
│   ├── navigator.py         # 编排器 + 难度分级
│   ├── observation_agent.py # VLM 感知
│   ├── analysis_agent.py    # CoT/Debate/Reflection
│   ├── planning_agent.py    # LLM + 拓扑 + A*
│   ├── review_agent.py      # 完成验证
│   ├── emergency_agent.py   # 应急检测
│   └── subtask_decomposition_agent.py
├── core/                    # Action, Context
├── models/                  # ModelManager (多server路由)
├── environment/             # Habitat 集成
├── utils/                   # 工具函数
├── configs/                 # 配置文件
├── scripts/                 # 启动脚本
├── envs/                    # Conda 环境文件
├── run_vln_experiment.py    # 主实验入口
└── vllm_server.py           # vLLM 推理服务
```

## 测试
```bash
pytest tests/pipeline/
```
