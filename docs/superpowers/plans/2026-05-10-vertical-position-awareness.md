# DecisionAgent垂直位置感知改进实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在DecisionAgent prompt中添加完整坐标信息和语义化垂直方向描述，让LLM能正确推理当前位置与goal的关系。

**Architecture:** 修改`_build_medium_prompt_with_analysis_v2`方法，扩展坐标提取并更新prompt模板。

**Tech Stack:** Python, 字符串格式化

---

## 文件结构

| 文件 | 改动类型 | 负责内容 |
|------|----------|----------|
| `agents/decision_agent.py:968-978` | 修改 | 扩展坐标提取，生成语义描述 |
| `agents/decision_agent.py:1007-1012` | 修改 | 更新prompt模板 |

---

## Task 1: 扩展坐标提取和语义描述生成

**文件:** `agents/decision_agent.py:968-978`

当前代码（第968-978行）：
```python
        # Vertical difference (y coordinate)
        goal_position = context.metadata.get("goal_position") if context else None
        current_position = context.position if context else None
        vertical_diff = 0.0
        vertical_ok = True
        if goal_position and current_position:
            goal_y = goal_position[1] if len(goal_position) > 1 else 0.0
            current_y = current_position[1] if len(current_position) > 1 else 0.0
            vertical_diff = current_y - goal_y
            vertical_ok = abs(vertical_diff) < 1.0
        vertical_status = "✓" if vertical_ok else f"❌ ({abs(vertical_diff):.1f}m)"
```

- [ ] **Step 1: 替换坐标提取代码**

将第968-978行替换为：

```python
        # Vertical difference (y coordinate) - 扩展为完整坐标提取
        goal_position = context.metadata.get("goal_position") if context else None
        current_position = context.position if context else None
        goal_x, goal_y, goal_z = 0.0, 0.0, 0.0
        curr_x, curr_y, curr_z = 0.0, 0.0, 0.0
        vertical_diff = 0.0
        vertical_ok = True
        vertical_direction = "位置未知"
        vertical_hint = ""

        if goal_position and len(goal_position) >= 3:
            goal_x, goal_y, goal_z = goal_position[0], goal_position[1], goal_position[2]
        if current_position and len(current_position) >= 3:
            curr_x, curr_y, curr_z = current_position[0], current_position[1], current_position[2]

        if goal_position and current_position:
            vertical_diff = curr_y - goal_y
            vertical_ok = abs(vertical_diff) < 1.0
            # 生成语义化方向描述
            if vertical_diff > 0.1:
                vertical_direction = "当前在目标上方"
                vertical_hint = "需要向下走"
            elif vertical_diff < -0.1:
                vertical_direction = "当前在目标下方"
                vertical_hint = "需要向上走"
            else:
                vertical_direction = "与目标同高度"
                vertical_hint = ""
        vertical_status = "✓" if vertical_ok else f"❌ ({abs(vertical_diff):.1f}m)"
```

- [ ] **Step 2: 验证语法正确**

运行: `python -c "from agents.decision_agent import DecisionAgent; print('OK')"`
预期: 输出 "OK"

---

## Task 2: 更新prompt模板

**文件:** `agents/decision_agent.py:1007-1012`

当前prompt（第1007-1012行）：
```python
## 最终目标 (Global Goal)
- 距离最终目标: {distance_to_goal:.1f}m (水平)
- 方向: {direction_hint}, 趋势: {distance_trend}
- 垂直: {vertical_status} (目标y vs 当前y)
- 成功条件: 水平 < 3m 且 垂直 < 0.5m
- 提示: 每步动作应使整体距离减小，而非只关注当前子任务
```

- [ ] **Step 1: 替换prompt模板中的最终目标段落**

将第1007-1012行替换为：

```python
## 最终目标 (Global Goal)
- 目标坐标: ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f})
- 当前坐标: ({curr_x:.2f}, {curr_y:.2f}, {curr_z:.2f})
- 距离: {distance_to_goal:.1f}m (水平), 方向: {direction_hint}
- 垂直: {vertical_direction} ({abs(vertical_diff):.1f}m) - {vertical_hint}
- 成功条件: 水平 < 3m 且 垂直 < 0.5m
```

注意：原第1013行的"- 提示"行保留不变。

- [ ] **Step 2: 验证语法正确**

运行: `python -c "from agents.decision_agent import DecisionAgent; print('OK')"`
预期: 输出 "OK"

---

## Task 3: Commit改动

- [ ] **Step 1: Git add并commit**

```bash
cd /home/WZ/MA_VLN/habitat_vln
git add agents/decision_agent.py
git commit -m "$(cat <<'EOF'
feat: DecisionAgent添加完整坐标和垂直方向语义描述

- 扩展goal_position和current_position提取为完整(x,y,z)
- 添加vertical_direction语义描述："当前在目标上方/下方"
- 添加vertical_hint导航提示："需要向下走/向上走"
- 更新prompt模板显示完整坐标信息
- 让LLM能正确推理楼梯场景中的位置关系

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: 验证commit成功**

运行: `git log -1 --oneline`
预期: 显示新commit

---

## Task 4: 验证测试

- [ ] **Step 1: 运行楼梯场景实验**

```bash
cd /home/WZ/MA_VLN/habitat_vln
bash scripts/run_vln_experiment_with_vllm.sh run_exp
```

预期: 实验运行完成

- [ ] **Step 2: 检查DecisionAgent prompt输出**

查看实验日志，确认prompt中包含：
- `目标坐标: (x, y, z)` 格式
- `当前坐标: (x, y, z)` 格式
- `垂直: 当前在目标上方/下方 (Xm) - 需要向X走`

- [ ] **Step 3: 验证Y坐标变化趋势**

查看results目录下的realtime_status.json：
- 期望Y坐标减小（向下走楼梯）
- subtask_completed可能为true或false（取决于整体导航效果）

---

## Self-Review

**1. Spec coverage:** ✅ 
- 完整坐标提取 → Task 1 ✓
- 语义化方向描述 → Task 1 ✓
- prompt模板更新 → Task 2 ✓
- 验证测试 → Task 4 ✓

**2. Placeholder scan:** ✅ 无TBD/TODO，所有代码块完整

**3. Type consistency:** ✅ 
- goal_x, goal_y, goal_z, curr_x, curr_y, curr_z 均为float
- vertical_direction, vertical_hint 均为str
- 与prompt模板中的{变量}引用一致

---

## 执行选项

**Plan complete and saved to `docs/superpowers/plans/2026-05-10-vertical-position-awareness.md`.**

**两种执行方式:**

**1. Inline Execution（推荐）** - 我在当前session直接执行，快速迭代验证

**2. Subagent-Driven** - 每个Task派发独立subagent，有review checkpoint

**选择哪种方式？**