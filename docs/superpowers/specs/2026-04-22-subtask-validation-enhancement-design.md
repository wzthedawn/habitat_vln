# 子任务验证增强设计

## 概述

通过多重验证机制增强子任务完成判断，防止LLM过早停止导致导航失败。

**核心目标**：
- LLM判断 + 自动检测双重验证
- 冲突时保守策略（任一未完成就继续）
- 序列执行中途验证（提前检测完成）
- 详细进度展示（辅助LLM决策）

**预期效果**：
- 减少过早停止错误
- 提升复杂子任务完成准确率
- 防止LLM误判导致的失败

---

## 一、当前问题分析

### 1.1 现有机制

```python
# DecisionAgent._format_completion_check
# 自动检测条件类型：
- y_change: 垂直移动（上楼/下楼）
- rotation: 旋转角度
- distance: 移动距离
- near_object: 靠近物体

# 当前验证流程：
1. DecisionAgent调用前做一次自动检测
2. 如果auto_completed=True，强制覆盖LLM结果
```

### 1.2 存在问题

| 问题 | 描述 | 影响 |
|------|------|------|
| LLM误判 | LLM可能过早标记完成 | 导航失败 |
| 单次验证 | 只在调用LLM前验证一次 | 序列执行中途无法感知 |
| 强制覆盖 | auto_completed直接覆盖 | 缺乏双重确认 |
| 无进度展示 | 只显示完成/未完成 | LLM无决策辅助信息 |

---

## 二、多重验证机制

### 2.1 双重验证流程

```python
def validate_subtask_completion(
    context: NavContext,
    subtask: SubTask,
    llm_completed: bool,
    llm_confidence: float = 0.8
) -> Tuple[bool, str]:
    """双重验证子任务完成状态。

    Args:
        context: 导航上下文
        subtask: 当前子任务
        llm_completed: LLM判断的完成状态
        llm_confidence: LLM置信度

    Returns:
        (final_completed, reason): 最终完成状态和原因
    """

    # 1. 自动检测
    completion_condition = subtask.completion_condition
    auto_result = check_completion_condition(context, completion_condition)
    auto_completed = auto_result["completed"]
    auto_progress = auto_result["progress"]

    # 2. 冲突时保守策略
    # 任一判断未完成 → 继续导航
    if not auto_completed or not llm_completed:
        return False, f"继续执行 (自动:{auto_completed}, LLM:{llm_completed})"

    # 3. 两者都完成才标记完成
    return True, f"验证通过 (自动:{auto_completed}, LLM:{llm_completed}, 进度:{auto_progress:.0%})"
```

### 2.2 决策逻辑

| 自动检测 | LLM判断 | 最终结果 | 原因 |
|----------|---------|----------|------|
| False | False | False | 两方都认为未完成 |
| False | True | False | **保守策略**：自动检测未通过 |
| True | False | False | **保守策略**：LLM判断未通过 |
| True | True | True | 双重确认通过 |

---

## 三、序列执行中途验证

### 3.1 实现位置

在 `run_vln_experiment.py` 主循环中，每次执行action后检查。

```python
# run_vln_experiment.py 主循环

current_sequence = None
completion_checked = False

while not done and steps < max_steps:
    # 执行动作
    if current_sequence and current_sequence.current_index < len(current_sequence.actions):
        action = current_sequence.actions[current_sequence.current_index]
        execute_action(action)

        # === NEW: 中途验证 ===
        if not completion_checked and steps % 3 == 0:  # 每3步检查一次
            current_subtask = context.get_current_subtask()
            if current_subtask and current_subtask.completion_condition:
                auto_result = check_completion_condition(
                    context,
                    current_subtask.completion_condition
                )

                if auto_result["completed"]:
                    # 自动检测通过，等待LLM确认（下一个DecisionAgent调用）
                    self.logger.info(f"[中途验证] 检测到可能完成: {auto_result['reason']}")

                    # 可选：提前终止序列，等待LLM验证
                    if auto_result["confidence"] > 0.9:  # 高置信度
                        self.logger.info("[中途验证] 高置信度，提前终止序列")
                        current_sequence = None
                        completion_checked = True

        current_sequence.current_index += 1
    else:
        # 需要新序列 → 调用DecisionAgent
        ...
```

### 3.2 验证频率

- **默认频率**：每3步检查一次（平衡性能和及时性）
- **可配置**：`completion_check_interval` 参数

### 3.3 中途验证方法

```python
def check_completion_condition(
    context: NavContext,
    condition: Dict[str, Any]
) -> Dict[str, Any]:
    """检查子任务完成条件是否满足。

    Args:
        context: 导航上下文
        condition: 完成条件定义

    Returns:
        {
            "completed": bool,
            "progress": float,  # 0.0-1.0
            "confidence": float,  # 置信度
            "reason": str,
            "current_value": float,
            "threshold": float
        }
    """

    if not condition:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": "无条件"}

    cc_type = condition.get("type", "unknown")

    # 获取当前状态（从TrajectoryAgent或直接计算）
    trajectory_output = context.metadata.get("trajectory_output", {})
    subtask_delta = trajectory_output.get("subtask_delta", {})
    pos_delta = subtask_delta.get("position_delta", {})

    # 计算当前值和进度
    if cc_type == "y_change":
        dy = abs(pos_delta.get("dy", 0))
        threshold = condition.get("min_change", 1.5)
        progress = min(1.0, dy / threshold)
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress

        return {
            "completed": dy >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"|dy|={dy:.2f}m >= {threshold}m",
            "current_value": dy,
            "threshold": threshold
        }

    elif cc_type == "rotation":
        rot_delta = subtask_delta.get("rotation_delta", {})
        rot_change = abs(rot_delta.get("delta_deg", 0))
        threshold = condition.get("min_degrees", 70)
        progress = min(1.0, rot_change / threshold)

        return {
            "completed": rot_change >= threshold,
            "progress": progress,
            "confidence": 1.0 if progress > 0.9 else 0.7 + 0.3 * progress,
            "reason": f"rotation={rot_change:.0f}° >= {threshold}°",
            "current_value": rot_change,
            "threshold": threshold
        }

    elif cc_type == "distance":
        h_dist = pos_delta.get("horizontal_distance", 0)
        threshold = condition.get("min_meters", 5)
        progress = min(1.0, h_dist / threshold)

        return {
            "completed": h_dist >= threshold,
            "progress": progress,
            "confidence": 1.0 if progress > 0.9 else 0.7 + 0.3 * progress,
            "reason": f"distance={h_dist:.2f}m >= {threshold}m",
            "current_value": h_dist,
            "threshold": threshold
        }

    elif cc_type == "near_object":
        perception_output = context.metadata.get("perception_output", {})
        objects = perception_output.get("objects", [])
        target = condition.get("object", "").lower()

        found = any(target in o.get("object", "").lower() for o in objects)

        return {
            "completed": found,
            "progress": 1.0 if found else 0.0,
            "confidence": 1.0 if found else 0.3,
            "reason": f"object '{target}' {'FOUND' if found else 'NOT FOUND'}",
            "current_value": 1 if found else 0,
            "threshold": 1
        }

    else:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": f"未知条件类型: {cc_type}"}
```

---

## 四、详细进度展示

### 4.1 Prompt格式

在DecisionAgent prompt中展示详细进度：

```python
## 子任务进度
- 类型: {cc_type_name} ({description})
- 当前: {current_value:.2f} {unit}
- 目标: {threshold:.2f} {unit}
- 进度: {progress:.0%}
- 预估: 还需约{estimated_steps}步

**验证状态**: {auto_status} (自动检测) + {llm_status} (LLM判断)
```

### 4.2 示例

```python
## 子任务进度
- 类型: 下楼梯 (y_change down)
- 当前: |dy|=1.2m
- 目标: >=1.5m
- 进度: 80%
- 预估: 还需约2步

**验证状态**: 未完成 (自动检测) + 待确认 (LLM)
```

### 4.3 实现方法

```python
def format_completion_progress(
    condition: Dict[str, Any],
    auto_result: Dict[str, Any],
    llm_completed: bool = None
) -> str:
    """格式化子任务进度展示。"""

    if not condition:
        return "无条件"

    cc_type = condition.get("type", "unknown")
    type_names = {
        "y_change": "垂直移动",
        "rotation": "旋转",
        "distance": "移动距离",
        "near_object": "靠近物体"
    }
    cc_type_name = type_names.get(cc_type, cc_type)

    units = {
        "y_change": "m",
        "rotation": "°",
        "distance": "m",
        "near_object": ""
    }
    unit = units.get(cc_type, "")

    current = auto_result.get("current_value", 0)
    threshold = auto_result.get("threshold", 0)
    progress = auto_result.get("progress", 0)

    # 预估剩余步数
    remaining = threshold - current if threshold > current else 0
    estimated_steps = int(remaining / 0.25) + 2  # 每步约0.25m

    # 验证状态
    auto_status = "已完成" if auto_result["completed"] else "未完成"
    llm_status = "已完成" if llm_completed else "未完成" if llm_completed is not None else "待确认"

    return f"""## 子任务进度
- 类型: {cc_type_name} ({cc_type})
- 当前: {current:.2f}{unit}
- 目标: >={threshold:.2f}{unit}
- 进度: {progress:.0%}
- 预估: 还需约{estimated_steps}步

**验证状态**: {auto_status} (自动) + {llm_status} (LLM)"""
```

---

## 五、改动文件清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `run_vln_experiment.py` | 主循环 | 新增中途验证逻辑 |
| `run_vln_experiment.py` | 新增函数 | `check_completion_condition()` |
| `agents/decision_agent.py` | `_format_completion_check` | 返回详细进度信息 |
| `agents/decision_agent.py` | `_build_xxx_prompt` | 使用进度展示格式 |
| `agents/decision_agent.py` | `generate_sequence` | 应用双重验证逻辑 |

---

## 六、预期效果

| 指标 | 改动前 | 改动后 |
|------|--------|--------|
| 过早停止错误率 | ~15%（估计） | ~5%（预期） |
| 子任务完成准确率 | ~70% | ~85%（预期） |
| 中途完成检测 | 无 | 每3步检查 |
| LLM决策辅助信息 | 无进度 | 详细进度展示 |

---

## 七、设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 验证增强方向 | 多重验证机制（B） | LLM误判是主要失败原因 |
| 冲突处理策略 | 保守策略（D） | 防止过早停止 |
| 验证时机增加 | 序列执行中途验证（B） | 条件可能在中途满足 |
| 中途验证位置 | run_vln_experiment主循环（A） | 执行流程清晰 |
| 进度展示方式 | 详细进度（B） | 给LLM足够决策信息 |

---

## 八、验证方法

```bash
# 1. 单元测试
pytest tests/test_decision_agent.py -v -k completion

# 2. 集成测试（观察中途验证日志）
python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm

# 3. 检查日志输出
grep "中途验证" logs/eval_*.log
grep "验证状态" logs/eval_*.log
```

---

*文档创建时间: 2026-04-22*