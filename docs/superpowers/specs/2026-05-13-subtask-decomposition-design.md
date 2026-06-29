# SubtaskDecompositionAgent 设计文档

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将导航 instruction 分解为多个可验证的子任务，提升 Pipeline 导航精确度

**Architecture:** 新增 SubtaskDecompositionAgent（Pipeline SubAgent），Navigator 在初始化时调用，分解结果存入 `_subtasks` 列表，ReviewAgent 判断完成后切换

**Tech Stack:** Python, LLM (通过 ModelManager), dataclasses

---

## 问题背景

当前 Pipeline 将整条 instruction 作为单一子任务：
```
instruction = "Walk down the stairs, turn right, and walk towards place with a rug..."
current_subtask = {
    "id": 1,
    "description": instruction,  # 未分解
    "completion_condition": {"type": "distance_to_goal", "threshold": 3.0}
}
```

**问题**:
1. ObservationAgent 无法判断 `task_relevant`（任务描述太模糊）
2. AnalysisAgent 推理困难（不知道当前应该"下楼"还是"找地毯"）
3. ReviewAgent 只能检查最终距离，无法验证中间步骤

---

## 设计方案

### 1. 新增 SubtaskDecompositionAgent

**文件**: `agents/pipeline/subtask_decomposition_agent.py`

**继承**: `SubAgent`

**职责**: LLM 驱动，将 instruction 分解为多个子任务

**输入**:
```python
{
    "instruction": str,           # 完整导航指令
    "goal_position": List[float], # 最终目标位置 [x, y, z]
    "start_position": List[float] # 起始位置（用于判断是否需要下楼）
}
```

**输出**: `DecompositionOutput`
```python
@dataclass
class DecompositionOutput:
    subtasks: List[Dict]  # 子任务列表
    reasoning: str        # 分解推理过程
```

### 2. 子任务数据结构

每个 subtask:
```python
{
    "id": int,
    "description": str,           # 清晰描述（隐含动作方向）
    "completion_condition": dict  # 可验证条件
}
```

### 3. Completion Condition 类型

| type | 字段 | 示例 | ReviewAgent 验证 |
|------|------|------|-----------------|
| `y_change` | `direction: "up"/"down"`, `threshold: float` | `{type: "y_change", direction: "down", threshold: 1.0}` | dy >= threshold (down) 或 dy <= -threshold (up) |
| `rotation` | `direction: "left"/"right"`, `threshold: float` | `{type: "rotation", direction: "right", threshold: 45}` | rotation_change >= threshold |
| `near_object` | `target: str`, `distance: float` | `{type: "near_object", target: "rug", distance: 3.0}` | 物体可见 + 距离 < threshold |
| `distance_to_goal` | `threshold: float` | `{type: "distance_to_goal", threshold: 3.0}` | 距 goal_position < threshold |
| `combined` | `conditions: List[dict]` | `{type: "combined", conditions: [{type: "near_object", target: "piano"}, {type: "distance_to_goal", threshold: 3.0}]}` | 所有条件都满足 |

### 4. 分解示例

输入:
```
instruction = "Walk down the stairs, turn right, and walk towards place with a rug. Wait near the bench and piano."
goal_position = [11.63, -3.16, 2.0]
start_position = [5.58, -1.62, 2.8]
```

输出:
```json
{
  "subtasks": [
    {
      "id": 1,
      "description": "Find the stairs and go down",
      "completion_condition": {"type": "y_change", "direction": "down", "threshold": 1.0}
    },
    {
      "id": 2,
      "description": "Turn right at the bottom of stairs",
      "completion_condition": {"type": "rotation", "direction": "right", "threshold": 45}
    },
    {
      "id": 3,
      "description": "Walk towards the rug area",
      "completion_condition": {"type": "near_object", "target": "rug", "distance": 3.0}
    },
    {
      "id": 4,
      "description": "Wait near the bench and piano",
      "completion_condition": {"type": "combined", "conditions": [{"type": "near_object", "target": "piano"}, {"type": "distance_to_goal", "threshold": 3.0}]}
    }
  ],
  "reasoning": "Instruction has 4 clear stages: stairs descent, turn, approach rug, final position near piano."
}
```

### 5. LLM Prompt

```
You are decomposing a navigation instruction into subtasks.

## Input
- Instruction: {instruction}
- Start position: {start_position}
- Goal position: {goal_position}

## Task
Break the instruction into 2-6 subtasks. Each subtask should:
1. Have a clear description (with implicit action direction)
2. Have a verifiable completion_condition

## Completion Condition Types
- y_change: Elevation change (up/down stairs)
- rotation: Turning action (left/right)
- near_object: Reaching near an object
- distance_to_goal: Final goal distance
- combined: Multiple conditions for final subtask

## Output Format (JSON only)
{
  "subtasks": [
    {"id": 1, "description": "...", "completion_condition": {...}},
    {"id": 2, "description": "...", "completion_condition": {...}}
  ],
  "reasoning": "..."
}
```

---

## Navigator 集成修改

### 修改文件: `agents/pipeline/navigator.py`

### 1. 新增 `_decompose_instruction()`

```python
def _decompose_instruction(self, instruction: str, goal_position: List[float]) -> List[Dict]:
    """调用 SubtaskDecompositionAgent 分解指令"""
    if self._model_manager is None:
        # 回退：整条指令作为单一子任务
        return [{"id": 1, "description": instruction, "completion_condition": {"type": "distance_to_goal", "threshold": 3.0}}]

    decomposition_output = self._registry.call(
        "decomposition",
        instruction=instruction,
        goal_position=goal_position,
        start_position=self._position,
    )
    return decomposition_output.subtasks
```

### 2. 修改 `initialize_episode()`

原代码:
```python
# Simple version: whole instruction as single subtask
self._subtasks = [
    {"id": 1, "description": instruction, "completion_condition": None}
]
self._current_subtask = self._subtasks[0]
```

改为:
```python
# 调用 SubtaskDecompositionAgent 分解
self._subtasks = self._decompose_instruction(instruction, goal_position)
self._current_subtask_index = 0
self._current_subtask = self._subtasks[0]
```

新增字段:
```python
self._current_subtask_index: int = 0  # 当前子任务索引
```

### 3. 修改 `_run_navigation_cycle()` 添加切换逻辑

在 ReviewAgent 调用后:
```python
review_output = self._registry.call("review", subtask, execution_result, state_change, observation)

if review_output.completed:
    if self._current_subtask_index < len(self._subtasks) - 1:
        self._current_subtask_index += 1
        self._current_subtask = self._subtasks[self._current_subtask_index]
        self.logger.info(f"Switching to subtask {self._current_subtask_index + 1}: {self._current_subtask['description']}")
    else:
        # 所有子任务完成
        return [(ActionType.STOP, 1)]
```

### 4. 注册 SubtaskDecompositionAgent

修改 `register_subagents()`:
```python
def register_subagents(self) -> None:
    self._registry.register("decomposition", SubtaskDecompositionAgent())
    self._registry.register("observation", ObservationAgent())
    self._registry.register("analysis", AnalysisAgent())
    self._registry.register("planning", PlanningAgent())
    self._registry.register("review", ReviewAgent())
    self._registry.register("emergency", EmergencyAgent())
```

---

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| LLM 分解失败（返回空/格式错误） | 回退：整条 instruction 作为单一子任务，completion_condition = distance_to_goal |
| LLM 分解结果只有 1 个子任务 | 正常接受 |
| completion_condition 类型不支持 | ReviewAgent fallback 返回 completed=False |

---

## 数据流

```
Navigator.initialize_episode()
    ↓
SubtaskDecompositionAgent.process() → DecompositionOutput
    ↓
Navigator._subtasks = [{id:1, ...}, {id:2, ...}, ...]
    ↓
_run_navigation_cycle() 使用 _current_subtask
    ↓
ReviewAgent 判断 completed=True
    ↓
Navigator 切换 _current_subtask_index += 1
    ↓
下一轮使用新子任务
```

---

## 输出记录

agent_outputs.json 中新增字段:
```json
{
  "subtasks": [
    {"id": 1, "description": "...", "completion_condition": {...}},
    {"id": 2, "description": "...", "completion_condition": {...}}
  ],
  "step_outputs": [
    {
      "step": 1,
      "current_subtask": {"id": 1, ...},
      ...
    },
    {
      "step": 15,
      "current_subtask": {"id": 2, ...},  // 切换后
      ...
    }
  ]
}
```

---

## 文件清单

| 文件 | 操作 |
|------|------|
| `agents/pipeline/subtask_decomposition_agent.py` | 新增 |
| `agents/pipeline/base_pipeline_agent.py` | 修改（添加 DecompositionOutput dataclass） |
| `agents/pipeline/navigator.py` | 修改（集成分解和切换逻辑） |
| `run_vln_experiment.py` | 修改（传递 goal_position 给 Navigator） |