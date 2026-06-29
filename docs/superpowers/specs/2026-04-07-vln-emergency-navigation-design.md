# VLN导航系统优化与应急场景扩展设计

## 概述

本文档描述VLN（视觉-语言导航）系统的优化方案，包括算法修复、多Agent协作架构、应急场景扩展、数据集构建和模型微调。目标将导航成功率从当前水平提升至50-70%，并扩展支持动态障碍应急场景。

---

## 一、整体路线图

| 阶段 | 目标 | 时间 | 产出 |
|------|------|------|------|
| **Phase 1** | 算法修复，JSON解析成功率>95%，导航成功率~50% | 1-2天 | 稳定运行的VLN系统 |
| **Phase 2a** | 应急场景扩展（动态障碍），导航成功率60-70% | 6天 | 论文1 - 动态障碍应急导航 |
| **Phase 2b** | 火灾视觉场景扩展（可选） | 12天 | 论文2/扩展章节 - 视觉应急导航 |

---

## 二、多Agent协作架构

### 2.1 设计原则

**核心原则**：
- **一个决策者**：DecisionAgent是唯一的决策中心
- **信息提供者不越权**：其他Agent只输出信息，不给动作指令
- **评估者不干预**：EvaluationAgent只做事后评估，不参与实时决策

### 2.2 Agent职责定义

```
┌─────────────────────────────────────────────────────────────────┐
│                    多Agent协作架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [信息提供层]                                                    │
│                                                                 │
│  PerceptionAgent ──────────────────────────────────────────────►│
│  ├─ 职责: 环境感知                                               │
│  ├─ 输出: room_type, objects, nav_hint, danger_info             │
│  ├─ 不输出: 动作建议（由DecisionAgent自己推理）                   │
│  └─ Phase 2a扩展: danger_detection                              │
│                                                                 │
│  InstructionAgent ─────────────────────────────────────────────►│
│  ├─ 职责: 指令理解、子任务分解                                   │
│  ├─ 输出: subtasks, completion_condition, task_level            │
│  ├─ 不输出: 动作建议                                             │
│  └─ Phase 1修复: 增强fallback规则                                │
│                                                                 │
│  TrajectoryAgent ──────────────────────────────────────────────►│
│  ├─ 职责: 轨迹分析、历史记录、stuck检测                          │
│  ├─ 输出: distance, heading, stuck_state, history_summary       │
│  ├─ 保留方法: get_history_summary, get_stuck_escape_opinion     │
│  ├─ 移除方法: build_debate_opinion等Debate相关                  │
│  └─ Phase 2a扩展: blocked_positions记录                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [决策层]                                                        │
│                                                                 │
│  DecisionAgent ◄────────────────────────────────────────────────│
│  ├─ 职责: 唯一的决策者                                           │
│  ├─ 输入: 所有Agent的信息输出                                    │
│  ├─ 内部流程:                                                    │
│  │   1. 策略选择（CoT / Debate / 应急快速路径）                  │
│  │   2. 冲突检测                                                 │
│  │   3. 动作序列生成                                             │
│  ├─ 输出: action_sequence, reasoning, confidence                │
│  └─ Phase 2a扩展: risk_score计算                                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [评估层 - 不参与实时决策]                                        │
│                                                                 │
│  EvaluationAgent                                                │
│  ├─ 职责: 事后评估、质量监控                                      │
│  ├─ 输入: DecisionAgent的决策结果                                │
│  ├─ 输出: score, feedback（供日志/分析）                         │
│  ├─ 不参与: 实时决策过程                                         │
│  └─ 用途:                                                        │
│      • 训练阶段：提供奖励信号                                    │
│      • 分析阶段：统计决策质量                                    │
│      • 调试阶段：记录决策评估                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 协作示例

**场景1：正常导航（无冲突）**
```
InstructionAgent输出: subtasks=[下楼, 右转, 走到rug]
                      completion_condition={Y下降1.5m}

PerceptionAgent输出:  room_type=stairs, nav_hint="楼梯在前方"
                      danger_info={danger_detected: false}

TrajectoryAgent输出:  distance=5.2m, stuck_state=false
                      history_summary="正常前进中"

DecisionAgent决策:
  → 策略选择: CoT
  → 综合信息生成动作序列
  → 输出: [forward, forward, forward...]
```

**场景2：检测到冲突**
```
InstructionAgent输出: subtasks="右转进入厨房"
                      completion_condition={rotation: right}

PerceptionAgent输出:  danger_info={danger_detected: true, 
                                   blocked_direction: "right"}
                      nav_hint="右侧有障碍"

TrajectoryAgent输出:  stuck_state=true
                      history="已多次尝试右转失败"

DecisionAgent决策:
  → 检测到冲突: 子任务要求右转 vs 感知检测到右侧障碍
  → 策略选择: 请求协作意见（可选）
  → 或直接决策: 先处理障碍再右转
```

**场景3：应急响应**
```
EmergencyDetector输出: trigger=true, type="obstacle_blocked"
                       position=(5.2, 3.1)

DecisionAgent决策:
  → 检测到应急信号
  → 策略选择: 快速路径（跳过CoT）
  → 直接调用PathReplanner
  → 输出: 新路径的动作序列
  → 响应时间: <1秒
```

---

## 三、策略选择机制

### 3.1 三种策略模式

| 模式 | 触发条件 | 流程 | 响应时间 |
|------|---------|------|---------|
| **CoT策略** | 正常导航，无冲突 | 信息汇聚 → LLM推理 → 动作序列 | 3-5秒 |
| **Debate策略** | 检测到冲突/不确定 | 收集Agent意见 → 冲突仲裁 → 动作序列 | 10-15秒 |
| **应急快速路径** | EmergencyDetector触发 | PathReplanner算法 → 动作序列 | <1秒 |

### 3.2 策略路由逻辑

```python
# DecisionAgent内部逻辑
def select_strategy(context):
    # 优先级1: 应急信号
    if context.metadata.get("emergency_signal", {}).get("trigger"):
        return "emergency_fast_path"
    
    # 优先级2: 检测冲突
    if has_conflict(context):
        return "debate"
    
    # 优先级3: 正常CoT
    return "cot"

def has_conflict(context):
    """检测Agent输出间是否存在冲突"""
    perception = context.metadata.get("perception_output", {})
    instruction = context.metadata.get("instruction_output", {})
    
    # 示例冲突检测
    if (instruction.get("current_subtask", {}).get("requires_turn") == "right"
        and perception.get("danger_info", {}).get("blocked_direction") == "right"):
        return True
    
    return False
```

### 3.3 Debate策略使用场景

**何时启用Debate**：
- ✅ 检测到Agent意见冲突
- ✅ 复杂场景需要多角度分析
- ❌ 应急场景（时间敏感）
- ❌ 简单场景（效率优先）

**Debate工作流程**：
```
1. DecisionAgent检测到冲突
2. 请求各Agent提供协作意见（可选，非必需）
3. TrajectoryAgent: 提供"避免重复动作"约束
4. PerceptionAgent: 提供"环境限制"约束
5. InstructionAgent: 提供"子任务要求"约束
6. DecisionAgent综合约束，做出最终决策
```

---

## 四、Phase 1: 算法修复

### 4.1 目标

- JSON解析成功率 > 95%
- 导航成功率 ~ 50%
- 验证架构可行性

### 4.2 已完成修复

| 问题 | 状态 | 效果 |
|------|------|------|
| 感知JSON解析失败 | ✅ 已修复 | nDTW: 0.617 → 0.819 |
| 楼梯方向检测 | ✅ 已修复 | Y坐标正确向下走 |

### 4.3 待修复问题（状态更新）

| 问题 | 优先级 | 状态 | 说明 |
|------|--------|------|------|
| 决策JSON解析失败 | P0 | ✅ 已完成 | 多层fallback解析已实现 |
| 子任务分解不稳定 | P1 | ⏳ 部分完成 | 有基础fallback，可优化 |
| TrajectoryAgent精简 | P2 | ⏳ 可选 | Debate方法保留但使用有限 |
| EvaluationAgent调整 | P3 | ✅ 已完成 | 改为事后评估模式 |

### 4.4 InstructionAgent Fallback增强

**当前问题**：LLM分解输出格式不稳定，fallback规则过于简单。

**增强方案**：
```python
def _split_instruction_enhanced(self, text: str) -> List[str]:
    """增强的指令分段规则"""
    segments = []
    
    # 1. 识别关键动作词
    action_keywords = ["turn", "walk", "go", "stop", "wait", "find", "enter"]
    
    # 2. 按动作词分段
    # "Walk down the stairs, turn right, and walk to the rug"
    # → ["Walk down the stairs", "turn right", "walk to the rug"]
    
    # 3. 提取每段的地标/目标
    # 用于生成completion_condition
    
    # 4. 自动生成completion_condition
    # "Walk down the stairs" → {type: "y_change", direction: "down", min_change: 1.5}
    
    return segments
```

### 4.5 TrajectoryAgent精简

**移除的方法**（约300行代码）：
- `build_debate_opinion()` - Debate相关
- `_build_trajectory_opinion_with_llm()` - Debate相关
- `_build_trajectory_opinion_fallback()` - Debate相关

**保留的方法**：
- `get_history_summary()` - CoT使用
- `record_action_sequence()` - 记录动作历史
- `get_stuck_escape_opinion()` - 应急场景使用
- `record_stuck_region()` - 记录卡住区域

### 4.6 EvaluationAgent调整

**改动**：
```python
# 原设计：参与决策
def _check_replan_needed(self) -> bool:
    return False  # 已禁用

# 新设计：事后评估
def process(self, context, decision):
    score = self._evaluate_decision(context, decision)
    
    # 记录到context，供日志使用
    context.metadata["evaluation_score"] = score
    
    # 写入实验结果文件
    log_decision_quality(context.step_count, score, decision)
    
    # 不返回任何影响决策的内容
    return AgentOutput.success_output(data={"score": score}, ...)
```

### 4.7 验收标准

- 连续运行100个episode无JSON解析失败
- 室内VLN导航成功率接近50%
- nDTW > 0.7
- 子任务分解成功率 > 90%

---

## 五、Phase 2a: 应急场景扩展

### 5.1 应急场景定义

**场景A: 动态障碍应对**
- 道路突然被阻断
- 障碍物移动
- 需要快速路径重规划

**场景B: 突发事件响应（Phase 2b）**
- 火灾逃生
- 地震避险
- 危险区域识别与规避

### 5.2 核心能力需求

1. **快速路径重规划** - 检测到障碍时秒级计算新路线
2. **风险评估与优先级** - 判断路径安全性，选择最优逃生路线
3. **动态环境感知** - 实时检测障碍物变化

### 5.3 新增模块设计

#### DynamicObstacleManager

```
功能: 管理障碍出现时机和位置
输入: 场景配置 (障碍触发步数、位置)
输出: 当前障碍状态
实现:
  • 在指定step触发障碍
  • 调用Habitat API修改navmesh可通行性
  • sim.pathfinder.set_nav_mesh_snapshot()
接口:
  • trigger_obstacle(position) → 修改地图
  • get_obstacle_state() → 返回当前障碍信息
  • reset_obstacles() → 恢复原始地图
特点: 实验控制用，不参与实时导航
```

#### PathReplanner

```
功能: 快速生成备选路径
输入:
  • current_pos: 当前位置
  • goal_pos: 目标位置
  • blocked_pos: 阻断点列表
  • navmesh: Habitat导航网格
输出:
  • primary_path: 主推荐路径
  • alternative_paths: 备选路径列表
  • path_scores: 各路径风险评分
实现:
  • A*算法 + 动态权重调整
  • blocked_pos区域设置高cost
  • 生成多条备选路径供决策选择
性能目标: <1秒响应时间
特点: 纯算法，不调用LLM
调用者: DecisionAgent（应急模式下）
```

#### EmergencyDetector

```
功能: 持续监控环境变化，触发应急响应
输入:
  • perception_history: PerceptionAgent连续输出
  • obstacle_state: DynamicObstacleManager状态
输出:
  • emergency_event: {type, position, timestamp}
  • trigger_signal: true/false
实现:
  • 对比前后perception输出检测变化
  • 监听obstacle_state更新
  • 触发时通知DecisionAgent切换模式
运行方式: 每step检查一次
```

### 5.4 DecisionAgent应急响应逻辑

```python
def generate_action_sequence(self, context, strategy_result, subtask):
    # 检查应急信号
    emergency_signal = context.metadata.get("emergency_signal", {})
    
    if emergency_signal.get("trigger"):
        return self._emergency_response(context, emergency_signal)
    else:
        return self._normal_decision(context, strategy_result, subtask)

def _emergency_response(self, context, signal):
    """应急响应：秒级"""
    # 1. 直接调用PathReplanner
    new_path = self.path_replanner.replan(
        current_pos=context.position,
        goal_pos=context.metadata["goal_position"],
        blocked_pos=signal.get("position")
    )
    
    # 2. 转换为动作序列
    actions = self._path_to_actions(new_path)
    
    return ActionSequenceResult(
        actions=actions,
        reasoning=f"Emergency replan: {signal.get('type')}",
        confidence=0.9
    )

def _normal_decision(self, context, strategy_result, subtask):
    """正常决策：CoT"""
    # 检测冲突
    if self._has_conflict(context):
        return self._debate_decision(context, strategy_result, subtask)
    
    # 使用CoT策略
    return self._cot_decision(context, strategy_result, subtask)
```

---

## 六、数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    完整数据流                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VLNNavigator (主控制器)                                         │
│      │                                                          │
│      ├─① DynamicObstacleManager.check_and_trigger()            │
│      │     → 修改navmesh，写入context.obstacle_state            │
│      │                                                          │
│      ├─② PerceptionAgent.process()                              │
│      │     → 输入: RGB+Depth图像                                 │
│      │     → 输出: context.perception_output (含danger_info)     │
│      │                                                          │
│      ├─③ TrajectoryAgent.process()                              │
│      │     → 输出: context.trajectory_output                     │
│      │                                                          │
│      ├─④ InstructionAgent.process()                             │
│      │     → 输出: context.subtasks                              │
│      │                                                          │
│      ├─⑤ EmergencyDetector.check()                              │
│      │     → 输入: perception_output, obstacle_state             │
│      │     → 输出: context.emergency_signal                      │
│      │                                                          │
│      ├─⑥ DecisionAgent.generate_action_sequence()               │
│      │     → 输入: 所有Agent输出 + emergency_signal              │
│      │     → 内部策略选择:                                       │
│      │         IF emergency: 快速路径(PathReplanner)             │
│      │         ELIF conflict: Debate策略                         │
│      │         ELSE: CoT策略                                     │
│      │     → 输出: action_sequence                               │
│      │                                                          │
│      ├─⑦ EvaluationAgent.process() [事后，可选]                 │
│      │     → 输入: 决策结果                                      │
│      │     → 输出: score (日志记录用)                            │
│      │                                                          │
│      └─⑧ 执行动作                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、数据集构建

### 7.1 混合方案

| 阶段 | 数据来源 | 规模 | 特点 |
|------|---------|------|------|
| Phase 2a前期 | 基于R2R改造应急指令 | 300-500样本 | 快速验证，标准格式 |
| Phase 2a后期 | 自建扩展数据 | 补充多样性 | 创新性，场景丰富 |

### 7.2 基于R2R改造流程

**指令改造示例**：
```
原R2R指令: "Walk down the stairs, turn right, and walk towards place with a rug."

改造为应急指令:
"Walk forward, the path is blocked, turn left and find another route to reach the exit."
"Walk along the corridor, avoid the obstacle ahead, turn right and continue towards the kitchen."
```

**句式模板**：
```
模板1: "[正常动作], [障碍提示], [重规划动作] to reach [目标]."
模板2: "Walk [方向], avoid [障碍描述], and [后续动作]."
模板3: "Navigate to [目标], re-route when [障碍触发], and [最终动作]."
```

### 7.3 数据集划分

| 子集 | 样本数 | 用途 |
|------|--------|------|
| 训练集 | 600-800 | LoRA微调 |
| 验证集 | 100 | 微调过程中验证 |
| 测试集 | 100-200 | 最终评估（不含微调） |

---

## 八、微调设计

### 8.1 微调任务划分

| 任务 | 输入 | 输出 | 数据量 | 显存 |
|------|------|------|--------|------|
| 感知模型 | RGB+Depth图像 | {room_type, objects, nav_hint, danger_info} | 300-400对 | 10-12GB |
| 决策模型 | perception_info + trajectory_info | {risk_score, action_sequence} | 500-600对 | 6-8GB |

### 8.2 LoRA配置

| 参数 | 感知模型 | 决策模型 |
|------|---------|---------|
| 目标模块 | q_proj, v_proj, k_proj, o_proj | q_proj, v_proj, k_proj, o_proj |
| LoRA秩 | 16 | 16 |
| 学习率 | 2e-4 | 2e-4 |
| Batch size | 4 | 8 |
| 训练轮数 | 3 epochs | 3 epochs |

### 8.3 微调目标

**注意**：微调针对内容质量提升，不针对格式问题。JSON格式稳定性应在Phase 1解决。

| 任务 | 微调目标 |
|------|---------|
| 感知模型 | danger_info准确性提升（正确识别障碍方向） |
| 决策模型 | risk_score合理性提升、动作质量提升 |

---

## 九、评估设计

### 9.1 评估维度

| 维度 | 指标 | Phase 1 | Phase 2a |
|------|------|---------|---------|
| 格式稳定性 | JSON解析成功率 | >95%验收 | — (已保证) |
| 基础导航 | SR, SPL, nDTW | ~50% | 60-70% |
| 应急响应 | 重规划成功率 | — | 核心指标 |
| 应急响应 | 响应时间 | — | <1秒 |
| 危险区域避开率 | — | — | >90% |

### 9.2 对比实验设计

| 实验组 | 配置 | 目的 |
|--------|------|------|
| Baseline | Phase 1修复后，纯LLM，无PathReplanner | 建立基准 |
| Exp-A | 启用PathReplanner，未微调模型 | 验证算法效果 |
| Exp-B | 启用PathReplanner，微调后模型 | 验证微调效果 |
| Exp-C | 未启用PathReplanner，微调后模型 | 对比LLM vs 算法规划 |

---

## 十、关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 成功率目标 | 50-70% | 平衡可行性与研究价值 |
| 决策架构 | DecisionAgent为唯一决策中心 | 职责清晰，避免混乱 |
| EvaluationAgent | 事后评估，不参与决策 | 减少冗余，提高效率 |
| TrajectoryAgent | 精简Debate代码，保留核心方法 | 移除未使用代码，保留有价值功能 |
| Debate策略 | 有条件使用（冲突时），应急时跳过 | 平衡决策质量和响应速度 |
| 应急响应 | 独立快速路径，纯算法 | 秒级响应，不依赖LLM |
| 数据集 | 基于R2R改造 + 自建 | 快速验证 + 创新性 |

---

## 十一、时间规划

| 阶段 | 任务 | 时间 |
|------|------|------|
| Phase 1 | 决策JSON解析修复 | 0.5天 |
| Phase 1 | InstructionAgent fallback增强 | 0.5天 |
| Phase 1 | TrajectoryAgent精简 | 0.5天 |
| Phase 1 | EvaluationAgent调整 | 0.5天 |
| **Phase 1总计** | | **1-2天** |
| Phase 2a | 架构扩展（新增模块） | 2天 |
| Phase 2a | 数据集构建 | 2天 |
| Phase 2a | 微调训练 | 1-2天 |
| Phase 2a | 评估实验 | 1天 |
| **Phase 2a总计** | | **6天** |
| Phase 2b | 火灾视觉场景（可选） | 12天 |

---

## 十二、后续扩展

Phase 2b（可选）将在Phase 2a基础上扩展：
- Python后处理叠加火焰纹理
- 火灾场景数据集（300-500样本）
- 增量微调（基于Phase 2a模型）
- 火灾逃生成功率评估

---

## 十三、LoRA模型集成设计

### 13.1 问题分析

微调的LoRA模型仅针对**决策任务**训练，未包含视觉感知数据。直接替换所有Agent会导致：

| Agent | 影响评估 |
|-------|---------|
| PerceptionAgent | ❌ **风险**：VLM能力可能被破坏 |
| DecisionAgent | ✅ **提升**：获得微调带来的决策增强 |
| 其他Agent | ⚠️ 中性：未微调，可能无变化 |

**结论**：需要保留PerceptionAgent的原始VLM能力，仅让DecisionAgent使用LoRA。

### 13.2 方案选择：动态LoRA切换

| 方案 | 显存 | 感知能力 | 决策能力 | 结论 |
|------|------|---------|---------|------|
| A. 双模型 | 36GB | ✅ | ✅ | ❌ 显存不足 |
| B. 单模型全部用LoRA | 20GB | ❌ 风险 | ✅ | ❌ 感知风险 |
| **C. 动态切换** | 20GB | ✅ | ✅ | **✅ 推荐** |

### 13.3 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Server                              │
├─────────────────────────────────────────────────────────────┤
│  Base Model: Qwen3.5-9B (FP16)                             │
│  ├── 路径: /data/WZ/Model/Qwen/Qwen3___5-9B                │
│  ├── VLM能力: ✅ 完整保留                                   │
│  └── 显存: ~18GB                                           │
│                                                             │
│  LoRA Adapter: qlora_mixed/decision (~126MB)               │
│  └── 动态加载，推理时选择是否使用                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    客户端调用逻辑                           │
├─────────────────────────────────────────────────────────────┤
│  PerceptionAgent  → generate_vision() → 不用LoRA           │
│  DecisionAgent    → generate() + lora_request → 使用LoRA    │
│  其他Agent        → generate() → 不用LoRA                   │
└─────────────────────────────────────────────────────────────┘
```

### 13.4 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `vllm_server.py` | 基础模型切换(AWQ→原始)，启用`enable_lora=True`，加载LoRA适配器 |
| `models/remote_client.py` | 添加`lora_request`参数支持 |
| `models/model_manager.py` | DecisionAgent调用时传入LoRA请求 |

### 13.5 关键代码变更

**vLLM服务器启动**:
```python
# 切换基础模型
model_path = "/data/WZ/Model/Qwen/Qwen3___5-9B"  # 原始模型，非AWQ

# 启用LoRA支持
engine = LLM(
    model=model_path,
    enable_lora=True,
    max_lora_rank=16,
    max_loras=1,
    gpu_memory_utilization=0.85,
)
```

**DecisionAgent调用**:
```python
# 使用LoRA
response = client.generate(
    model="qwen-9b-decision",
    prompt=prompt,
    lora_request=LoRARequest("decision-lora", 1, "outputs/qlora_mixed/decision")
)
```

### 13.6 显存估算

| 组件 | 显存 |
|------|------|
| Qwen3.5-9B (FP16) | ~18GB |
| LoRA Adapter | ~0.3GB |
| vLLM KV Cache | ~2GB |
| **总计** | ~20-21GB |

适配24GB显卡，预留3-4GB安全余量。

### 13.7 验收标准

1. vLLM服务器成功加载基础模型 + LoRA适配器
2. PerceptionAgent正常工作（VLM能力保留）
3. DecisionAgent使用LoRA推理成功
4. 显存不超过23GB，无OOM错误

---

## 进度跟踪

### Phase 1: 算法修复 ✅ 已完成

| 任务 | 状态 | 效果 |
|------|------|------|
| 感知JSON解析失败修复 | ✅ 完成 | nDTW: 0.617 → 0.819 |
| 楼梯方向检测修复 | ✅ 完成 | Y坐标正确向下走 |
| 决策JSON解析fallback | ✅ 完成 | 多层fallback解析 |
| EvaluationAgent调整 | ✅ 完成 | 事后评估模式 |

### Phase 2a: 应急场景扩展 ⏳ 进行中

| 任务 | 状态 | 产出 |
|------|------|------|
| 架构扩展（新增模块） | ✅ 完成 | `emergency/` 目录下3个模块 |
| 数据集构建 | ✅ 完成 | 560样本（400紧急+160正常） |
| 微调训练 | ✅ 完成 | `outputs/qlora_mixed/decision/` |
| **LoRA模型集成** | ✅ 完成 | 见第十三节 |
| **应急测试集实现** | ✅ 完成 | `scripts/run_emergency_eval.py` |
| 评估实验 | ⏳ 待执行 | 对比实验 |

### Phase 2a 微调训练详情

**训练配置**:
- 基础模型: `/data/WZ/Model/Qwen/Qwen3___5-9B`（非AWQ）
- LoRA rank: 16, alpha: 32, dropout: 0.05
- 训练轮数: 2 epochs
- 训练时长: ~39分钟
- 最终Loss: 0.5014
- 目标模块: `in_proj_qkv`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`

**验证结果**:
- 紧急导航成功率: 100%
- 正常导航成功率: 100%
- 输出文件: `outputs/qlora_mixed/decision/adapter_model.safetensors` (126MB)

---

## 下一步行动计划

### 已完成：应急场景测试集实现 ✅

**新增文件**: `scripts/run_emergency_eval.py`

**核心功能**:
- `EmergencyVLNEvaluator`: 应急场景评估器
- 集成`DynamicObstacleManager`: 动态障碍物管理
- 集成`EmergencyDetector`: 应急事件检测
- 支持四组对比实验: baseline, exp-a, exp-b, exp-c

**使用方法**:
```bash
# 单实验测试
python scripts/run_emergency_eval.py --exp baseline --episodes 10

# 全部实验对比
python scripts/run_emergency_eval.py --exp all --episodes 20
```

### 任务依赖关系

```
Baseline ──────────────────────────────────────► 可立即执行
Exp-A ─────────────────────────────────────────► 可立即执行
        │
        │  ┌─────────────────┐
        └─►│ LoRA模型集成    │────► Exp-B
           └─────────────────┘      Exp-C
```

### 执行顺序

| 优先级 | 任务 | 依赖 | 预计时间 |
|--------|------|------|---------|
| **1** | 运行Baseline评估实验 | 无 | 0.5天 |
| **2** | 运行Exp-A评估实验 | 无 | 0.5天 |
| **3** | LoRA模型集成实现 | 无 | 1天 |
| **4** | 运行Exp-B评估实验 | 任务3 | 0.5天 |
| **5** | 运行Exp-C评估实验 | 任务3 | 0.5天 |
| **6** | 结果分析与报告 | 任务1-5 | 0.5天 |

**说明**：
- 任务1-2可并行执行（不依赖LoRA集成）
- 任务3需要修改vLLM服务器和客户端代码
- 任务4-5依赖任务3完成

### 实验配置

| 实验组 | PathReplanner | LoRA模型 | 当前状态 |
|--------|--------------|---------|---------|
| Baseline | ❌ | ❌ | 可执行 |
| Exp-A | ✅ | ❌ | 可执行 |
| Exp-B | ✅ | ✅ | 待LoRA集成 |
| Exp-C | ❌ | ✅ | 待LoRA集成 |

---

*文档创建时间: 2026-04-07*
*文档更新时间: 2026-04-08*