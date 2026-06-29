---
title: 导航子任务判定修复
date: 2026-04-27
type: design
status: draft
---

# 导航子任务判定修复设计文档

## 问题概述

实验 `episode-2026-0426-1550` 成功率仅 20%（1/5），分析发现核心问题：

| 问题 | 现象 | 根因 | 涉及 Episode |
|------|------|------|-------------|
| **A. y_change 判定失效** | y_change 条件用 `abs(dy)` 忽略方向 | 判定逻辑缺陷 | 1, 2, 3 |
| **B. Episode 2 异常终止** | 仅执行 20 步就停止 | 原因不明，需调试 | 2 |
| **C. 导航方向错误** | agent 向上走但目标是下楼 | A 的结果 | 1, 2, 3 |
| **D. success 不一致** | Episode 4 成功但子任务全未完成 | 两套独立判定 | 4 |
| **E. rotation 判定失效** | rotation 条件用 `abs(delta)` 忽略方向 | 判定逻辑缺陷（与 A 同类） | 4 |
| **F. Episode 5 导航失败** | 仅移动 1.49m，原地打转 | 导航决策问题 | 5 |

**优先级**：修复 A/E 是核心（同类 bug），C/F 是导航问题需单独分析。

## 核心问题分析

### 问题 A：y_change 判定忽略方向

**代码缺陷**：

```python
# run_vln_experiment.py:129
if cc_type == "y_change":
    threshold = condition.get("min_change", 1.5)
    completed = abs(dy) >= threshold  # ❌ 忽略 direction
```

**实验数据验证**：

| Episode | 条件 | 实际 dy | abs(dy) | 判定结果 | 正确结果 |
|---------|------|---------|---------|----------|----------|
| 1 | down, ≥1.5m | +1.66m（向上） | 1.66 | ✅ 完成 | ❌ 未完成 |
| 2 | down, ≥1.5m | +0.55m（向上） | 0.55 | ❌ 未完成 | ❌ 未完成 |
| 3 | down, ≥1.5m | +1.46m（向上） | 1.46 | ❌ 未完成 | ❌ 未完成 |

Episode 1 因 `abs(1.66) >= 1.5` 被错误判定为"下楼完成"，实际是向上走。

**因果链**：
```
abs(dy) 忽略方向 → 错误判定完成 → 切换子任务 → 导航继续向上
```

### 问题 E：rotation 判定忽略方向

**代码缺陷**：

```python
# run_vln_experiment.py:141-148
elif cc_type == "rotation":
    threshold = condition.get("min_degrees", 70)
    completed = abs_delta_deg >= threshold  # ❌ 忽略 direction
```

**实验数据验证**：

Episode 4 第一个子任务：
- 条件：`direction: left, min_degrees: 60`
- 实际：左转 60°（delta_deg = 60）
- 判定：`abs(60) >= 60` → 应该完成，但 completed = None

**问题**：与 y_change 同类 bug，rotation 也忽略了 direction 字段。

**因果链**：
```
abs(delta_deg) 忽略方向 → 子任务判定不准确 → completed 字段混乱
```

## 修复方案

### 修复 1：check_completion_condition（run_vln_experiment.py）

**位置**：第 124-135 行

**修复内容**：

```python
# 修复前
if cc_type == "y_change":
    threshold = condition.get("min_change", 1.5)
    completed = abs(dy) >= threshold
    reason = f"|dy|={abs(dy):.2f}m >= {threshold}m"

# 修复后
if cc_type == "y_change":
    threshold = condition.get("min_meters", condition.get("min_change", 1.5))
    direction = condition.get("direction", "")

    if direction == "down":
        # 下楼需要 dy 为负（y 减小）
        completed = dy <= -threshold
        current_value = -dy  # 用正值表示"下降距离"
        reason = f"dy={dy:.2f}m (down), need <=-{threshold}m"
    elif direction == "up":
        # 上楼需要 dy 为正（y 增加）
        completed = dy >= threshold
        current_value = dy
        reason = f"dy={dy:.2f}m (up), need >= {threshold}m"
    else:
        # 无方向要求，用绝对值
        completed = abs(dy) >= threshold
        current_value = abs(dy)
        reason = f"|dy|={abs(dy):.2f}m >= {threshold}m"

    progress = min(1.0, current_value / threshold) if threshold > 0 else 0
    confidence = 1.0 if completed else 0.7 + 0.3 * progress
```

### 修复 2：_format_completion_check（decision_agent.py）

**位置**：第 1444-1459 行

**修复内容**：同上逻辑，确保判定一致性。

```python
# 修复前（第 1448-1459 行）
if cc_type == "y_change":
    current_value = abs(dy)
    threshold = min_change
    if direction == "down":
        comparison = f"|dy| = {current_value:.2f}m >= {threshold}m"
        is_completed = current_value >= threshold
    ...

# 修复后
if cc_type == "y_change":
    threshold = min_change
    if direction == "down":
        current_value = -dy if dy < 0 else 0  # 只有向下才计入
        is_completed = dy <= -threshold
        comparison = f"dy={dy:.2f}m, need <=-{threshold}m (down)"
    elif direction == "up":
        current_value = dy if dy > 0 else 0
        is_completed = dy >= threshold
        comparison = f"dy={dy:.2f}m, need >= {threshold}m (up)"
    else:
        current_value = abs(dy)
        is_completed = abs(dy) >= threshold
        comparison = f"|dy|={abs(dy):.2f}m >= {threshold}m"
```

### 修复 3：rotation 判定方向检查（run_vln_experiment.py）

**位置**：第 137-148 行

**修复内容**：

```python
# 修复前
elif cc_type == "rotation":
    threshold = condition.get("min_degrees", 70)
    completed = abs_delta_deg >= threshold  # ❌ 忽略 direction

# 修复后
elif cc_type == "rotation":
    threshold = condition.get("min_degrees", 70)
    direction = condition.get("direction", "")

    if direction == "left":
        # 左转需要 delta_deg 为正（角度增加）
        completed = delta_deg >= threshold
        current_value = delta_deg if delta_deg > 0 else 0
        reason = f"rotation={delta_deg:.0f}° (left), need >= {threshold}°"
    elif direction == "right":
        # 右转需要 delta_deg 为负（角度减少）
        completed = delta_deg <= -threshold
        current_value = -delta_deg if delta_deg < 0 else 0
        reason = f"rotation={delta_deg:.0f}° (right), need <= -{threshold}°"
    else:
        # 无方向要求，用绝对值
        completed = abs_delta_deg >= threshold
        current_value = abs_delta_deg
        reason = f"rotation={abs_delta_deg:.0f}° >= {threshold}°"

    progress = min(1.0, current_value / threshold) if threshold > 0 else 0
    confidence = 1.0 if completed else 0.7 + 0.3 * progress
```

### 修复 4：rotation 判定方向检查（decision_agent.py）

**位置**：`_format_completion_check` 方法中的 rotation case

**修复内容**：同上逻辑，确保与 run_vln_experiment.py 一致。

### 修复 5：增加调试日志（问题 B）

**位置**：run_vln_experiment.py 主循环异常捕获

**修复内容**：

```python
# 在 while steps < max_steps 循环结束后添加
except Exception as e:
    self.logger.error(f"Episode terminated unexpectedly: {e}")
    self.logger.error(f"  Final state: steps={steps}, max_steps={max_steps}")
    self.logger.error(f"  last_action={action_name}, dist={dist:.2f}m")
    import traceback
    traceback.print_exc()
```

## 验证计划

修复后运行相同场景验证：

```bash
python run_vln_experiment.py \
  --use-remote-llm \
  --llm-server http://localhost:8000 \
  --episodes 5 \
  --max-steps 150 \
  --output-dir results/episode-verify-fix
```

**预期结果**：
- Episode 1/2/3 的 y_change 子任务正确判定方向
- Episode 4 的 rotation 子任务正确判定方向
- 导航决策能感知方向错误，调整策略
- 问题 B/F 的状态变化需观察

## 改动文件清单

| 文件 | 改动位置 | 改动类型 | 状态 |
|------|----------|----------|------|
| `run_vln_experiment.py` | 124-153 行 | 修复 y_change 方向判定 | ✅ 已完成 |
| `run_vln_experiment.py` | 155-184 行 | 修复 rotation 方向判定 | ✅ 已完成 |
| `run_vln_experiment.py` | 异常捕获 | 增加调试日志 | ✅ 已完成 |
| `agents/decision_agent.py` | 1444-1461 行 | 修复 y_change 方向判定 | ✅ 已完成 |
| `agents/decision_agent.py` | 1462-1481 行 | 修复 rotation 方向判定 | ✅ 已完成 |

## 风险评估

- **改动范围**：小，仅涉及判定条件
- **影响面**：所有涉及 y_change 的子任务
- **回归风险**：低，现有测试覆盖判定逻辑

---

*设计文档版本：2026-04-27*
*状态：待审核*