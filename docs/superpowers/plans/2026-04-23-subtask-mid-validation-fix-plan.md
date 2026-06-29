# 子任务中途验证数据修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复中途验证使用旧trajectory_output数据导致子任务无法完成的问题

**Architecture:** 在中途验证调用前直接从context.position计算真实移动距离，通过condition参数传入，check_completion_condition函数优先使用传入值

**Tech Stack:** Python, math.sqrt

---

## File Structure

| 文件 | 改动 |
|------|------|
| `run_vln_experiment.py` | 修改check_completion_condition函数（3处条件类型） |
| `run_vln_experiment.py` | 修改中途验证调用逻辑（第1351-1356行） |

---

### Task 1: 修改check_completion_condition函数 - y_change条件

**Files:**
- Modify: `run_vln_experiment.py:102-114`

- [ ] **Step 1: 读取现有代码**

现有代码（第102-114行）：
```python
    if cc_type == "y_change":
        dy = abs(pos_delta.get("dy", 0))
        threshold = condition.get("min_change", 1.5)
        progress = min(1.0, dy / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": dy >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"|dy|={dy:.2f}m >= {threshold}m",
            "current_value": dy,
            "threshold": threshold
        }
```

- [ ] **Step 2: 替换为支持传入真实距离的版本**

使用Edit工具，将第102-114行替换为：
```python
    if cc_type == "y_change":
        # 优先使用传入的真实距离（中途验证时），否则从旧数据读取（序列生成时）
        actual_dy = condition.get("_actual_dy")
        if actual_dy is not None:
            dy = abs(actual_dy)
        else:
            dy = abs(pos_delta.get("dy", 0))
        threshold = condition.get("min_change", 1.5)
        progress = min(1.0, dy / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": dy >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"|dy|={dy:.2f}m >= {threshold}m",
            "current_value": dy,
            "threshold": threshold
        }
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

---

### Task 2: 修改check_completion_condition函数 - distance条件

**Files:**
- Modify: `run_vln_experiment.py:129-140`

- [ ] **Step 1: 读取现有代码**

现有代码（第129-140行）：
```python
    elif cc_type == "distance":
        h_dist = pos_delta.get("horizontal_distance", 0)
        threshold = condition.get("min_meters", 5)
        progress = min(1.0, h_dist / threshold) if threshold > 0 else 0
        return {
            "completed": h_dist >= threshold,
            "progress": progress,
            "confidence": 1.0 if progress > 0.9 else 0.7 + 0.3 * progress,
            "reason": f"distance={h_dist:.2f}m >= {threshold}m",
            "current_value": h_dist,
            "threshold": threshold
        }
```

- [ ] **Step 2: 替换为支持传入真实距离的版本**

使用Edit工具，将第129-140行替换为：
```python
    elif cc_type == "distance":
        # 优先使用传入的真实距离（中途验证时），否则从旧数据读取（序列生成时）
        actual_dist = condition.get("_actual_horizontal_distance")
        if actual_dist is not None:
            h_dist = actual_dist
        else:
            h_dist = pos_delta.get("horizontal_distance", 0)
        threshold = condition.get("min_meters", 5)
        progress = min(1.0, h_dist / threshold) if threshold > 0 else 0
        return {
            "completed": h_dist >= threshold,
            "progress": progress,
            "confidence": 1.0 if progress > 0.9 else 0.7 + 0.3 * progress,
            "reason": f"distance={h_dist:.2f}m >= {threshold}m",
            "current_value": h_dist,
            "threshold": threshold
        }
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

---

### Task 3: 修改check_completion_condition函数 - obstacle_detected条件

**Files:**
- Modify: `run_vln_experiment.py:156-168`

- [ ] **Step 1: 读取现有代码**

现有代码（第156-168行）：
```python
    elif cc_type == "obstacle_detected":
        # Check if agent has moved at least min_distance_moved from subtask start
        threshold = condition.get("min_distance_moved", 1.0)
        h_dist = pos_delta.get("horizontal_distance", 0)
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

- [ ] **Step 2: 替换为支持传入真实距离的版本**

使用Edit工具，将第156-168行替换为：
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

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

---

### Task 4: 修改中途验证调用逻辑

**Files:**
- Modify: `run_vln_experiment.py:1351-1356`

- [ ] **Step 1: 读取现有代码**

现有代码（第1351-1356行）：
```python
                    else:
                        try:
                            auto_result = check_completion_condition(context, current_subtask.completion_condition)
                        except Exception as e:
                            self.logger.warning(f"[中途验证] check_completion_condition failed: {e}")
                            auto_result = {"completed": False, "confidence": 0, "reason": str(e)}
```

- [ ] **Step 2: 替换为计算真实距离的版本**

使用Edit工具，将第1351-1356行替换为：
```python
                    else:
                        try:
                            # 直接计算当前移动距离（不依赖旧的trajectory_output）
                            current_pos = context.position
                            start_context = current_subtask.start_context or {}
                            start_pos = start_context.get("position", current_pos)
                            dx = current_pos[0] - start_pos[0]
                            dz = current_pos[2] - start_pos[2]
                            dy = current_pos[1] - start_pos[1]
                            actual_horizontal_dist = math.sqrt(dx*dx + dz*dz)

                            # 更新condition传入真实距离
                            updated_condition = current_subtask.completion_condition.copy()
                            updated_condition["_actual_horizontal_distance"] = actual_horizontal_dist
                            updated_condition["_actual_dy"] = dy

                            auto_result = check_completion_condition(context, updated_condition)
                        except Exception as e:
                            self.logger.warning(f"[中途验证] check_completion_condition failed: {e}")
                            auto_result = {"completed": False, "confidence": 0, "reason": str(e)}
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

---

### Task 5: 运行测试验证

**Files:**
- Test: 运行评估脚本验证功能

- [ ] **Step 1: 运行单episode测试**

Run: `python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm --llm-server http://localhost:8000 2>&1 | tee /tmp/mid_validation_test.log`

Expected: 测试启动成功

- [ ] **Step 2: 检查日志确认子任务完成**

等待测试运行约60秒后检查日志：

Run: `grep -E "中途验证.*检测到可能完成|Advanced to next subtask|Using strategy: CoT|completion_checked=True" /tmp/mid_validation_test.log | head -10`

Expected:
- 出现 `[中途验证] step X: 检测到可能完成 - moved=X.XXm >= 1.0m`
- 出现 `[SEQUENCE] Advanced to next subtask`
- 出现 `[SEQUENCE] Using strategy: CoT`（如果进入medium任务）
- 不再出现 `completion_checked=None`（子任务一直未完成的情况）

- [ ] **Step 3: 确认无错误**

Run: `grep -E "Error|Exception|Traceback" /tmp/mid_validation_test.log`

Expected: 无严重错误（可能有一些Warning，不影响功能）

---

### Task 6: 提交代码

- [ ] **Step 1: 查看改动**

Run: `git diff run_vln_experiment.py | head -80`

Expected: 显示3处check_completion_condition修改 + 1处中途验证修改

- [ ] **Step 2: 提交**

Run: `git add run_vln_experiment.py && git commit -m "fix: mid-sequence validation uses actual position delta instead of stale trajectory_output

- Modified check_completion_condition to accept _actual_horizontal_distance and _actual_dy parameters
- y_change, distance, obstacle_detected conditions now use passed-in values when available
- Mid-sequence validation directly calculates position delta from context.position
- Fixes subtask completion detection that was blocked by stale trajectory_output data"`

---

## Spec Coverage Check

| Spec要求 | Task覆盖 |
|---------|---------|
| 中途验证调用前计算真实距离 | Task 4 ✅ |
| check_completion_condition使用传入距离 | Task 1, 2, 3 ✅ |
| y_change条件支持传入距离 | Task 1 ✅ |
| distance条件支持传入距离 | Task 2 ✅ |
| obstacle_detected条件支持传入距离 | Task 3 ✅ |
| 测试验证 | Task 5 ✅ |

---

## Self-Review

1. **Placeholder scan**: 无TBD/TODO，所有代码完整 ✅
2. **Type consistency**: `_actual_horizontal_distance` (float), `_actual_dy` (float) 在所有使用处一致 ✅
3. **Spec coverage**: 所有要求已覆盖 ✅

---

## 注意事项

- `math.sqrt` 已在文件顶部导入（第88行），无需额外导入
- `context.position` 返回tuple (x, y, z)
- `start_context.get("position", current_pos)` 确保未设置时fallback为当前位置（距离为0）