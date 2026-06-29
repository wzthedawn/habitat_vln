# 多智能体视觉-语言导航系统

## 1. 项目概述

本系统是一个**分层多智能体协作导航框架**，基于Habitat模拟环境，实现智能体根据自然语言指令在3D场景中自主导航。系统核心由三部分组成：

- **算法结构**：分层多智能体架构，根据任务复杂度自适应调度
- **应急数据集**：基于R2R构建的动态障碍应急导航数据
- **模型微调**：QLoRA高效微调，实现应急场景快速适配

---

## 2. 算法结构

### 2.1 分层架构设计

系统采用**弱层-强层双层架构**，根据任务复杂度自动切换：

| 层级 | 处理能力 | 适用任务 | 特点 |
|------|----------|----------|------|
| **弱层** | 本地小模型 | Type-0（单步指令） | 快速响应，低资源消耗 |
| **强层** | 多智能体协作 | Type-1~Type-4 | LLM推理，复杂决策 |

### 2.2 任务复杂度分类

系统将导航任务分为5个复杂度等级：

| 类型 | 描述 | 智能体组合 | 推理策略 |
|------|------|------------|----------|
| Type-0 | 单步指令（如"前进"） | 无 | 无 |
| Type-1 | 走廊导航 | Perception + Decision | ReAct |
| Type-2 | 物体查找 | Perception + Trajectory + Decision | ReAct + CoT |
| Type-3 | 跨房间导航 | 全部4个智能体 | CoT + Reflection |
| Type-4 | 模糊/复杂场景 | 全部4个智能体 | CoT + Debate + Reflection |

**分类流程**：规则分类器（快速）→ 置信度<0.9 → LLM分类器（精准）

### 2.3 Supernet编排层

Supernet是系统核心编排层，负责：
- 任务类型识别
- 智能体池管理
- 策略链构建
- 执行流程协调

**执行流程**：
```
输入指令 → TaskClassifier分类 → Supernet选择配置 
         → 激活对应智能体 → 执行策略链 → 输出导航动作
```

### 2.4 四类专用智能体

| 智能体 | 职责 | 输入 | 输出 |
|--------|------|------|------|
| **InstructionAgent** | 指令理解、子任务分解 | 自然语言指令 | subtasks, completion_condition |
| **PerceptionAgent** | 环境感知、场景理解 | RGB + Depth图像 | room_type, objects, nav_hint, danger_info |
| **TrajectoryAgent** | 轨迹分析、历史记录 | 位置历史、动作序列 | distance, heading, stuck_state, history_summary |
| **DecisionAgent** | 最终决策输出 | 所有智能体信息 | action_sequence, reasoning, confidence |

### 2.5 四种推理策略

| 策略 | 适用场景 | 流程 | 响应时间 |
|------|----------|------|----------|
| **ReAct** | 简单任务 | 观察→思考→行动循环 | 2-3秒 |
| **CoT** | 中等复杂度 | 链式推理分解问题 | 3-5秒 |
| **Debate** | 存在冲突/不确定 | 多智能体讨论仲裁 | 10-15秒 |
| **Reflection** | 需要复盘调整 | 执行→反思→修正 | 5-8秒 |

### 2.6 应急响应机制

检测到应急信号时，系统切换至**快速路径模式**：
- 跳过LLM推理，直接调用PathReplanner算法
- 响应时间<1秒

---

## 3. 应急数据集

### 3.1 应急场景定义

传统VLN数据集（如R2R）仅包含静态环境导航指令。本项目扩展了**动态障碍应急场景**：

| 场景类型 | 描述 | 典型指令示例 |
|----------|------|--------------|
| **blocked_path** | 路径阻断 | "Walk forward, the path is blocked, turn left to reach the exit." |
| **dynamic_obstacle** | 动态障碍物 | "Proceed straight, obstacle appeared, find alternative route." |
| **emergency_evacuation** | 紧急撤离 | "Emergency: Navigate to the exit, obstacle detected, quickly turn right." |

### 3.2 数据集构建流程

基于R2R数据集改造生成应急指令：

1. 加载原始R2R episode（含场景、起点、终点、指令）
2. 选择应急场景类型
3. 应用指令模板生成应急指令
4. 配置障碍物触发时机（trigger_step）与位置
5. 评估指令难度等级

### 3.3 应急指令模板设计

三类应急指令模板：

```
模板A（blocked_path）：
"{normal_action}, the path is blocked, {reroute_action} to reach {goal}."

模板B（dynamic_obstacle）：
"{normal_action}, suddenly blocked, quickly {escape_action}."

模板C（emergency_evacuation）：
"Emergency: {normal_action}, route blocked, {emergency_action}."
```

### 3.4 数据集构成

| 数据子集 | 样本数量 | 用途 |
|----------|----------|------|
| 训练集 | 400 | QLoRA微调训练 |
| 验证集 | 80 | 微调过程验证 |
| 测试集 | 80 | 最终评估（不参与训练） |

### 3.5 难度评估标准

| 难度 | 评估规则 | 占比 |
|------|----------|------|
| **easy** | 无转弯指令，单步动作 | ~25% |
| **medium** | 1-2次转弯，复合动作 | ~50% |
| **hard** | 3次以上转弯，复杂路径 | ~25% |

---

## 4. 模型微调

### 4.1 微调方法选择

采用**QLoRA（Quantized LoRA）**进行高效微调：

| 优势 | 说明 |
|------|------|
| **低显存需求** | 4-bit量化，7B模型仅需~15GB显存 |
| **训练速度快** | 仅训练LoRA适配器参数（约0.3%原始参数） |
| **模型可复用** | 一个基座模型可加载多个LoRA适配器 |

### 4.2 基础模型配置

| 配置项 | 设置 |
|--------|------|
| 基础模型 | Qwen3.5-9B |
| 量化方式 | 4-bit NF4量化 |
| 计算精度 | BF16 |

### 4.3 LoRA参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA秩 | 16 | 适配器维度 |
| LoRA alpha | 32 | 缩放因子 |
| Dropout率 | 0.05 | 防止过拟合 |
| 目标模块 | in_proj_qkv, out_proj, gate_proj, up_proj, down_proj | 线性注意力层 + MLP层 |

### 4.4 训练配置

| 参数 | 值 |
|------|-----|
| 训练轮数 | 2 epochs |
| Batch size | 2 |
| 梯度累积 | 8 |
| 学习率 | 2e-4 |
| 最大序列长度 | 1024 |
| 训练时长 | ~39分钟 |

### 4.5 动态LoRA切换机制

仅对**决策任务**进行微调，系统采用动态切换机制：

| 调用场景 | 是否使用LoRA | 说明 |
|----------|--------------|------|
| PerceptionAgent（视觉感知） | 不使用 | 保留原始VLM能力 |
| DecisionAgent（导航决策） | 使用 | 获得微调增强效果 |
| 其他Agent | 不使用 | 未微调，保持原有能力 |

```
vLLM Server
├── Base Model: Qwen3.5-9B (完整VLM能力)
├── LoRA Adapter: qlora_mixed/decision (~126MB)
└── 推理时动态选择是否启用LoRA
```

### 4.6 微调产出

| 产出文件 | 说明 |
|----------|------|
| adapter_model.safetensors | LoRA适配器权重（126MB） |
| train_metrics.json | 训练指标记录 |
| checkpoint-{N}/ | 各阶段checkpoint |

**训练最终Loss**: 0.5014

---

## 5. 总结

本系统通过三个核心模块实现了VLN导航的智能化与应急能力扩展：

| 模块 | 技术亮点 |
|------|----------|
| 算法结构 | 自适应架构 + 多智能体协作 + 应急快速响应 |
| 应急数据集 | 基于R2R改造 + 三类场景模板 + 自动难度评估 |
| 模型微调 | QLoRA高效微调 + 动态LoRA切换机制 |

后续工作：大规模评估实验、性能指标验证、更多应急场景扩展。