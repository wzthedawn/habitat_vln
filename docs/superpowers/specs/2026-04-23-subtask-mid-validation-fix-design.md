---
name: 子任务中途验证数据修复
description: 修复中途验证使用旧trajectory_output数据导致子任务无法完成的问题
type: project
---

# 子任务中途验证数据修复设计

## 背景

### 问题
中途验证机制（第1328-1367行）每3步检查子任务完成条件，调用`check_completion_condition()`判断是否满足。但该函数从`trajectory_output.subtask_delta`读取移动距离，而`trajectory_output`只在序列生成时由TrajectoryAgent计算一次，后续action执行期间不更新。

### 后果
- `pos_delta.get("horizontal_distance", 0)` 返回0或序列生成时的旧值
- `obstacle_detected`条件（要求`min_distance_moved >= 1.0m`）永远无法满足
- 子任务无法完成，系统卡在第一个easy subtask，无法进入medium subtask
- CoT策略无法触发，CoT动作提取功能无法验证

### 日志证据
```
[中途验证] 进入验证块, step=6
[中途验证] current_subtask=SubTask(0: 检测到前方出现障碍物... [easy], comp={'type': 'obstacle_detected', 'min_distance_moved': 1.0})
[TrajectoryAgent] Distance traveled: 0.6m  # 实际已移动0.6m
[DEBUG] step=9, check_interval=0, completion_checked=None  # 仍未完成
```

## 解决方案

### 核心思路
在中途验证时，直接从`context.position`和`subtask.start_context`计算真实移动距离，不依赖旧的`trajectory_output`。

### 改动清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `run_vln_experiment.py` | 第1351-1356行 | 中途验证调用前计算真实距离 |
| `run_vln_experiment.py` | 第156-168行 | check_completion_condition使用传入距离 |

## 实现细节

### 1. 中途验证改动（第1351-1356行）

**现有代码：**
```python
else:
    try:
        auto_result = check_completion_condition(context, current_subtask.completion_condition)
    except Exception as e:
        self.logger.warning(f"[中途验证] check_completion_condition failed: {e}")
        auto_result = {"completed": False, "confidence": 0, "reason": str(e)}
```

**改为：**
```python
else:
    try:
        # 直接计算当前移动距离（不依赖旧的trajectory_output）
        current_pos = context.position
        start_context = current_subtask.start_context or {}
        start_pos = start_context.get("position", current_pos)
        dx = current_pos[0] - start_pos[0]
        dz = current_pos[2] - start_pos[2]
        actual_horizontal_dist = math.sqrt(dx*dx + dz*dz)

        # 更新condition传入真实距离
        updated_condition = current_subtask.completion_condition.copy()
        updated_condition["_actual_horizontal_distance"] = actual_horizontal_dist

        auto_result = check_completion_condition(context, updated_condition)
    except Exception as e:
        self.logger.warning(f"[中途验证] check_completion_condition failed: {e}")
        auto_result = {"completed": False, "confidence": 0, "reason": str(e)}
```

### 2. check_completion_condition改动（第156-168行）

**现有代码：**
```python
elif cc_type == "obstacle_detected":
    # Check if agent has moved at least min_distance_moved from subtask start
    threshold = condition.get("min_distance_moved", 1.0)
    h_dist = pos_delta.get("horizontal_distance", 0)
    progress = min(1.0, h_dist / threshold) if threshold > 0 else 0
    return {
        "completed": h_dist >= threshold,
        ...
    }
```

**改为：**
```python
elif cc_type == "obstacle_detected":
    # 优先使用传入的真实距离（中途验证时），否则从旧数据读取（序列生成时）
    actual_dist = condition.get("_actual_horizontal_distance")
    if actual_dist is not None:
        h_dist = actual_dist
    else:
        h_dist = pos_delta.get("horizontal_distance", 0)

    threshold = condition.get("min_distance_moved", 1.0)
    progress = min(1.0, h_dist / threshold) if threshold > 0 else 0
    return {
        "completed": h_dist >= threshold,
        "progress": progress,
        "confidence": 1.0 if progress > 0.9 else 0.7 + 0.3 * progress,
        "reason": f"moved={h_dist:.2f}m >= {threshold}m (obstacle reaction)",
        "current_value": h_dist,
        "threshold": threshold
    }
```

### 3. 同样处理其他依赖位置变化的条件类型

为保持一致性，对`y_change`、`distance`等类型也做类似处理：

```python
elif cc_type == "y_change":
    actual_dist = condition.get("_actual_dy")
    if actual_dist is not None:
        dy = abs(actual_dist)
    else:
        dy = abs(pos_delta.get("dy", 0))
    ...

elif cc_type == "distance":
    actual_dist = condition.get("_actual_horizontal_distance")
    if actual_dist is not None:
        h_dist = actual_dist
    else:
        h_dist = pos_delta.get("horizontal_distance", 0)
    ...
```

并在中途验证中同时计算dy：
```python
dy = current_pos[1] - start_pos[1]
updated_condition["_actual_dy"] = dy
```

## 测试验证

### 验证方法
运行评估脚本：
```bash
python scripts/run_emergency_eval.py --exp baseline --episodes 3 --use-remote-llm --llm-server http://localhost:8000
```

### 成功指标
日志中出现：
```
[中途验证] step=6: 检测到可能完成 - moved=1.2m >= 1.0m (obstacle reaction)
[SEQUENCE] Advanced to next subtask: 寻找替代路线避开障碍物...
[SEQUENCE] Using strategy: CoT  # 进入medium任务触发CoT
[Decision] 直接提取CoT动作: 3步，跳过LLM调用  # CoT提取功能生效
```

不再出现：
```
[DEBUG] completion_checked=None  # 子任务始终未完成
```

## 影响范围

- 只修改`run_vln_experiment.py`（中途验证调用逻辑 + check_completion_condition函数）
- 不改变TrajectoryAgent、DecisionAgent、NavContext
- 序列生成时的check_completion_condition调用仍使用原有逻辑（兼容）

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 计算逻辑与TrajectoryAgent不一致 | 使用相同公式：sqrt(dx² + dz²) |
| start_context可能未设置 | fallback到current_pos（距离为0） |
| 其他条件类型未处理 | 按需扩展，目前只处理obstacle_detected/y_change/distance |

## Why

中途验证使用旧数据导致子任务无法完成，系统卡在easy subtask，无法验证CoT提取功能。

## How to apply

在中途验证代码中，调用check_completion_condition前直接计算position变化，通过condition参数传入，函数优先使用传入值。