# 子任务中途验证修复实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复`check_completion_condition`函数，使其直接从context计算位置变化，不再依赖过期的trajectory_output。

**Architecture:** 重写函数内部数据获取逻辑，删除workaround代码，保持接口不变。

**Tech Stack:** Python, pytest, math

---

## Context

当前问题：序列执行期间不调用TrajectoryAgent，`trajectory_output.subtask_delta`只更新于序列生成时，导致中途验证使用过期数据。

解决方案：`check_completion_condition`直接从`context.position`和`current_subtask.start_context`计算状态变化。

---

## File Structure

| 文件 | 改动类型 | 职责 |
|------|----------|------|
| `tests/test_completion_condition.py` | 新建 | 单元测试 |
| `run_vln_experiment.py` | 修改 | 重写函数 + 删除workaround |

---

## Task 1: 编写单元测试

**Files:**
- Create: `tests/test_completion_condition.py`
- Modify: `run_vln_experiment.py` (no changes in this task)

- [ ] **Step 1: 创建测试文件骨架**

```python
"""Unit tests for check_completion_condition function."""

import pytest
import math
from run_vln_experiment import check_completion_condition
from core.context import NavContext, SubTask


class TestCheckCompletionCondition:
    """Tests for check_completion_condition function."""

    def _create_mock_context(self, position, rotation, subtask_start_pos=None, subtask_start_rot=None):
        """Helper to create mock NavContext."""
        context = NavContext()
        context.position = position
        context.rotation = rotation

        # Create mock subtask with start_context
        subtask = SubTask(
            id=0,
            description="test subtask",
            completion_condition={"type": "distance", "min_meters": 5}
        )
        if subtask_start_pos:
            subtask.start_context = {
                "position": subtask_start_pos,
                "rotation": subtask_start_rot if subtask_start_rot else rotation,
            }
        context.subtasks = [subtask]
        context.current_subtask_idx = 0

        return context
```

- [ ] **Step 2: 编写y_change条件测试**

```python
    def test_y_change_completed(self):
        """Test y_change condition when threshold met."""
        context = self._create_mock_context(
            position=(0.0, 2.0, 0.0),  # dy = 2.0
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "y_change", "min_change": 1.5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 1.5
        assert "dy" in result["reason"]

    def test_y_change_not_completed(self):
        """Test y_change condition when threshold not met."""
        context = self._create_mock_context(
            position=(0.0, 0.5, 0.0),  # dy = 0.5
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "y_change", "min_change": 1.5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["progress"] < 1.0
```

- [ ] **Step 3: 编写distance条件测试**

```python
    def test_distance_completed(self):
        """Test distance condition when threshold met."""
        context = self._create_mock_context(
            position=(5.0, 0.0, 0.0),  # horizontal_dist = 5.0
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 5.0

    def test_distance_not_completed(self):
        """Test distance condition when threshold not met."""
        context = self._create_mock_context(
            position=(2.0, 0.0, 0.0),  # horizontal_dist = 2.0
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["progress"] == 0.4  # 2.0/5.0
```

- [ ] **Step 4: 编写rotation条件测试**

```python
    def test_rotation_completed(self):
        """Test rotation condition when threshold met."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(90),  # 90 degrees
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 70

    def test_rotation_boundary_handling(self):
        """Test rotation handles -180/180 boundary."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(-170),  # -170 degrees
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=math.radians(170)  # 170 degrees
        )
        # delta should be -170 - 170 = -340 -> +20 after boundary fix
        condition = {"type": "rotation", "min_degrees": 70}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["current_value"] == 20  # boundary handled
```

- [ ] **Step 5: 编写边界情况测试**

```python
    def test_no_subtask(self):
        """Test when no current subtask."""
        context = NavContext()
        context.position = (1.0, 0.0, 0.0)
        context.subtasks = []

        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert "无子任务" in result["reason"]

    def test_no_start_context(self):
        """Test when subtask has no start_context."""
        context = self._create_mock_context(
            position=(5.0, 0.0, 0.0),
            rotation=0.0,
            subtask_start_pos=None  # No start_context
        )
        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        # Should use current position as start -> delta = 0
        assert result["completed"] == False
        assert result["current_value"] == 0

    def test_no_condition(self):
        """Test when condition is empty."""
        context = self._create_mock_context(
            position=(1.0, 0.0, 0.0),
            rotation=0.0
        )
        result = check_completion_condition(context, None)

        assert result["completed"] == False
        assert "无条件" in result["reason"]
```

- [ ] **Step 6: 运行测试确认失败**

Run: `pytest tests/test_completion_condition.py -v`
Expected: 所有测试FAIL（函数尚未改造）

---

## Task 2: 重写check_completion_condition函数

**Files:**
- Modify: `run_vln_experiment.py:68-204`

- [ ] **Step 1: 替换函数开头部分（数据获取逻辑）**

将第68-100行替换为：

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
            "progress": float (0.0-1.0),
            "confidence": float,
            "reason": str,
            "current_value": float,
            "threshold": float
        }
    """
    import math

    if not condition:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": "无条件", "current_value": 0, "threshold": 0}

    cc_type = condition.get("type", "unknown")

    # 获取当前子任务
    current_subtask = context.get_current_subtask()
    if not current_subtask:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": "无子任务", "current_value": 0, "threshold": 0}

    # 直接从context计算位置变化
    current_pos = context.position
    start_context = current_subtask.start_context or {}
    start_pos = start_context.get("position", current_pos)
    start_rot = start_context.get("rotation", context.rotation)

    # 计算位置delta
    dx = current_pos[0] - start_pos[0]
    dy = current_pos[1] - start_pos[1]
    dz = current_pos[2] - start_pos[2]
    horizontal_dist = math.sqrt(dx*dx + dz*dz)

    # 计算rotation变化（处理-180/180边界）
    current_deg = math.degrees(context.rotation)
    start_deg = math.degrees(start_rot)
    delta_deg = current_deg - start_deg
    if delta_deg > 180:
        delta_deg -= 360
    elif delta_deg < -180:
        delta_deg += 360
    abs_delta_deg = abs(delta_deg)

    # 处理不同条件类型
```

- [ ] **Step 2: 替换y_change条件处理**

将第102-119行替换为：

```python
    if cc_type == "y_change":
        threshold = condition.get("min_change", 1.5)
        progress = min(1.0, abs(dy) / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": abs(dy) >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"|dy|={abs(dy):.2f}m >= {threshold}m",
            "current_value": abs(dy),
            "threshold": threshold
        }
```

- [ ] **Step 3: 替换rotation条件处理**

将第121-132行替换为：

```python
    elif cc_type == "rotation":
        threshold = condition.get("min_degrees", 70)
        progress = min(1.0, abs_delta_deg / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": abs_delta_deg >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"rotation={abs_delta_deg:.0f}° >= {threshold}°",
            "current_value": abs_delta_deg,
            "threshold": threshold
        }
```

- [ ] **Step 4: 替换distance条件处理**

将第134-150行替换为：

```python
    elif cc_type == "distance":
        threshold = condition.get("min_meters", 5)
        progress = min(1.0, horizontal_dist / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": horizontal_dist >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"distance={horizontal_dist:.2f}m >= {threshold}m",
            "current_value": horizontal_dist,
            "threshold": threshold
        }
```

- [ ] **Step 5: 替换obstacle_detected条件处理**

将第166-182行替换为：

```python
    elif cc_type == "obstacle_detected":
        threshold = condition.get("min_distance_moved", 1.0)
        progress = min(1.0, horizontal_dist / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": horizontal_dist >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"moved={horizontal_dist:.2f}m >= {threshold}m (obstacle reaction)",
            "current_value": horizontal_dist,
            "threshold": threshold
        }
```

- [ ] **Step 6: 保持near_object和obstacle_cleared条件不变**

这两部分（第152-164行和第184-201行）保持不变，因为它们使用的是perception_output和blocked_info，不是position数据。

- [ ] **Step 7: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 8: 运行测试**

Run: `pytest tests/test_completion_condition.py -v`
Expected: 所有测试PASS

---

## Task 3: 删除workaround代码

**Files:**
- Modify: `run_vln_experiment.py:1414-1428`

- [ ] **Step 1: 简化中途验证逻辑**

将第1412-1428行替换为：

```python
                    else:
                        try:
                            auto_result = check_completion_condition(context, current_subtask.completion_condition)
                        except Exception as e:
                            self.logger.warning(f"[中途验证] check_completion_condition failed: {e}")
                            auto_result = {"completed": False, "confidence": 0, "reason": str(e)}
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

---

## Task 4: 集成测试验证

**Files:**
- Test: 现有测试文件

- [ ] **Step 1: 运行所有相关测试**

Run: `pytest tests/test_completion_condition.py tests/test_decision_sequence.py -v`
Expected: 所有测试PASS

- [ ] **Step 2: Commit**

```bash
git add tests/test_completion_condition.py run_vln_experiment.py
git commit -m "$(cat <<'EOF'
fix: subtask mid-validation uses stale data

Rewrite check_completion_condition to directly calculate
position/rotation delta from context.position and
subtask.start_context, eliminating dependency on
potentially outdated trajectory_output.

Changes:
- Rewrite check_completion_condition (~35 lines)
- Delete workaround code at line 1414-1428 (~15 lines)
- Add unit tests for all condition types

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**1. Spec coverage:**
- ✓ y_change: Task 1 Step 2 + Task 2 Step 2
- ✓ distance: Task 1 Step 3 + Task 2 Step 4
- ✓ rotation: Task 1 Step 4 + Task 2 Step 3
- ✓ obstacle_detected: Task 2 Step 5
- ✓ near_object: 保持不变（Task 2 Step 6）
- ✓ obstacle_cleared: 保持不变（Task 2 Step 6）
- ✓ 边界情况: Task 1 Step 5
- ✓ 删除workaround: Task 3

**2. Placeholder scan:**
- 无TBD、TODO
- 所有代码完整

**3. Type consistency:**
- 返回格式保持不变（completed, progress, confidence, reason, current_value, threshold）

---

*文档创建时间: 2026-04-25*