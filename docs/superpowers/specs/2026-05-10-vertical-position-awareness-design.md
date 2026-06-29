# DecisionAgent垂直位置感知改进设计

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让DecisionAgent能正确理解当前位置与goal的垂直关系，结合VLM楼梯方向描述做出正确的导航决策。

**Architecture:** 在DecisionAgent prompt中添加完整坐标信息和语义化方向描述，让LLM能推理"我在楼梯顶部还是底部"。

**Tech Stack:** Python, LLM prompt修改

---

## 问题分析

### 当前问题

VLM描述楼梯方向时，输出的是**视觉形态**而非导航方向：
- `scene_description: "path ascends a staircase"` = 楼梯从观看者视角向上延伸
- `stairs.direction: "up"` = 楼梯物理形态向上

DecisionAgent误解为导航方向：
- 错误推理："我在楼梯顶部，向前走就是下楼"
- 实际：agent在楼梯底部（Y=-1.63），向前走=Y增加（上楼）

### 根本原因

DecisionAgent prompt中缺少关键信息：
1. **goal完整坐标**未传入LLM
2. **当前完整坐标**未传入LLM
3. **vertical_diff方向含义**不明确（只显示数值"1.5m"，没说"我在上方/下方"）

### 信息流现状

```
代码中已有:
  goal_position: (x, y, z) ✓
  current_position: (x, y, z) ✓
  vertical_diff = current_y - goal_y ✓

LLM看到的prompt:
  - 距离最终目标: 6.3m (水平)
  - 垂直: ❌ (1.5m) (目标y vs 当前y)  ← 方向含义不明确
```

---

## 设计方案

### 改动范围

**文件:** `agents/decision_agent.py`
**方法:** `_build_medium_prompt_with_analysis_v2`（约第949-1029行）

### 改动内容

#### 1. 提取完整坐标（第969-978行）

当前代码只提取Y坐标，改为提取X、Y、Z：

```python
# 改进后的坐标提取
goal_position = context.metadata.get("goal_position") if context else None
current_position = context.position if context else None
goal_x, goal_y, goal_z = 0.0, 0.0, 0.0
curr_x, curr_y, curr_z = 0.0, 0.0, 0.0
vertical_diff = 0.0
vertical_direction = "位置未知"
vertical_hint = ""

if goal_position and len(goal_position) >= 3:
    goal_x, goal_y, goal_z = goal_position[0], goal_position[1], goal_position[2]
if current_position and len(current_position) >= 3:
    curr_x, curr_y, curr_z = current_position[0], current_position[1], current_position[2]

# 计算垂直关系并生成语义描述
if goal_position and current_position:
    vertical_diff = curr_y - goal_y
    if vertical_diff > 0.1:  # 在上方超过0.1m
        vertical_direction = "当前在目标上方"
        vertical_hint = "需要向下走"
    elif vertical_diff < -0.1:  # 在下方超过0.1m
        vertical_direction = "当前在目标下方"
        vertical_hint = "需要向上走"
    else:
        vertical_direction = "与目标同高度"
        vertical_hint = ""
```

#### 2. 更新prompt模板（第1007-1012行）

当前：
```
## 最终目标 (Global Goal)
- 距离最终目标: {distance_to_goal:.1f}m (水平)
- 方向: {direction_hint}, 趋势: {distance_trend}
- 垂直: {vertical_status} (目标y vs 当前y)
- 成功条件: 水平 < 3m 且 垂直 < 0.5m
```

改进后：
```
## 最终目标 (Global Goal)
- 目标坐标: ({goal_x:.2f}, {goal_y:.2f}, {goal_z:.2f})
- 当前坐标: ({curr_x:.2f}, {curr_y:.2f}, {curr_z:.2f})
- 距离: {distance_to_goal:.1f}m (水平), 方向: {direction_hint}
- 垂直: {vertical_direction} ({abs(vertical_diff):.1f}m) - {vertical_hint}
- 成功条件: 水平 < 3m 且 垂直 < 0.5m
```

### 预期效果

LLM将能正确推理：

| 场景 | VLM输出 | LLM推理 |
|------|---------|---------|
| 楼梯底部，goal在下 | stairs.direction="up", Y=-1.63, goal Y=-3.13 | 当前在目标上方，楼梯向上延伸→向前走会爬楼（错误）→应找向下路径 |
| 楼梯顶部，goal在下 | stairs.direction="down", Y=0.5, goal Y=-3.13 | 当前在目标上方，楼梯向下延伸→向前走是下楼（正确） |
| 楼梯底部，goal在上 | stairs.direction="up", Y=-1.63, goal Y=0.5 | 当前在目标下方，楼梯向上延伸→向前走是上楼（正确） |

---

## 实现步骤

### Task 1: 修改坐标提取和语义描述生成

**文件:** `agents/decision_agent.py:969-978`

- [ ] **Step 1: 扩展坐标提取**
  提取goal和current的完整(x, y, z)坐标

- [ ] **Step 2: 生成语义化方向描述**
  根据vertical_diff生成"当前在目标上方/下方"+"需要向哪个方向走"

- [ ] **Step 3: Commit**
  ```bash
  git add agents/decision_agent.py
  git commit -m "feat: DecisionAgent添加完整坐标和垂直方向语义描述"
  ```

### Task 2: 更新prompt模板

**文件:** `agents/decision_agent.py:1007-1012`

- [ ] **Step 1: 修改最终目标段落**
  添加目标坐标、当前坐标、垂直方向描述

- [ ] **Step 2: Commit**
  ```bash
  git add agents/decision_agent.py
  git commit -m "feat: DecisionAgent prompt添加完整坐标信息"
  ```

### Task 3: 验证测试

- [ ] **Step 1: 运行楼梯场景实验**
  ```bash
  bash scripts/run_vln_experiment_with_vllm.sh run_exp
  ```

- [ ] **Step 2: 检查DecisionAgent推理输出**
  确认prompt中包含完整坐标和方向描述

- [ ] **Step 3: 验证Y坐标变化**
  期望：agent向下走楼梯（Y减小），而非向上（Y增大）

---

## Self-Review

**1. Placeholder scan:** ✅ 无TBD/TODO

**2. Internal consistency:** ✅ 设计描述与实现步骤一致

**3. Scope check:** ✅ 单文件改动，约15行代码，范围可控

**4. Ambiguity check:** ✅ 明确改动位置、变量名和prompt内容