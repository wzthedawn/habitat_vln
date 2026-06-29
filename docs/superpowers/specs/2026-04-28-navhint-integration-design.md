---
title: DecisionAgent集成PerceptionAgent导航提示
date: 2026-04-28
type: design
status: draft
---

# DecisionAgent信息传递全面修复设计文档

## 问题概述

深度分析发现**多个关键信息未被DecisionAgent使用**，导致决策质量下降。

### 问题总览

| # | 问题 | 影响模式 | 影响程度 | 根因 |
|---|------|----------|----------|------|
| 1 | nav_hint未使用 | easy/medium/hard | **严重** | 参数传入但prompt忽略 |
| 2 | walkable_analysis未使用 | easy | **严重** | open_dirs_str未传给easy模式 |
| 3 | distance_trend未使用 | easy | **中等** | easy模式无接近/远离趋势信息 |
| 4 | topology_summary未使用 | easy | **中等** | 参数传入但prompt忽略 |
| 5 | landmarks/goals未使用 | easy | **中等** | 参数传入但prompt忽略 |
| 6 | completion_condition未展示 | easy | **低** | 无完成条件状态展示 |
| 7 | pos_delta/rot_delta细节缺失 | easy | **低** | 未格式化状态变化 |

### 核心因果链

```
PerceptionAgent生成nav_hint + walkable → 存入context.metadata → DecisionAgent参数传入 → ❌ prompt未使用 → 决策与建议矛盾 → 导航失败（Episode 5原地打转）

TrajectoryAgent计算distance_trend → 存入context.metadata → ❌ easy模式未提取 → 无法判断接近/远离 → 决策方向盲目

TopologyGraph生成拓扑信息 → 存入trajectory_output → ❌ easy模式prompt忽略 → 无历史导航辅助 → 重复探索
```

### 实验数据验证

| Episode | nav_hint建议 | DecisionAgent决策 | distance_trend | topology可用 | 结果 |
|---------|-------------|------------------|----------------|-------------|------|
| 5 step30 | "Turn left and move backward" | TURN_LEFT + FORWARD（矛盾） | ❌未知 | ❌忽略 | 前进而非后退 |
| 5 step60 | "Turn left to face goal" | TURN_RIGHT, TURN_RIGHT, TURN_RIGHT | ❌未知 | ❌忽略 | 方向完全相反 |
| 5 step90 | "Turn around and move back" | TURN_RIGHT | ❌未知 | ❌忽略 | 错误转向 |

## 代码根因分析

### 问题1-5：nav_hint和相关信息未传入easy模式

**位置**: `agents/decision_agent.py` 第866-947行 `_build_simple_prompt_direct_v2`

```python
def _build_simple_prompt_direct_v2(
    self, seq_len: int, context, subtask, blocked, min_dist, dist_traveled,
    heading, distance_to_goal, directions, nav_hint, landmarks,  # ← nav_hint传入
    goals, completion_condition=None, ...
) -> str:
    perception_output = context.metadata.get("perception_output", {}) if context else {}
    room_type = perception_output.get("room_type", "unknown")
    objects = [o.get("name") for o in objects_raw[:3] if o.get("name")]

    # ❌ 关键问题1：nav_hint参数被完全忽略
    # ❌ 关键问题2：walkable_analysis未提取（medium/hard通过open_dirs_str使用）
    # ❌ 关键问题3：distance_trend未提取
    # ❌ 关键问题4：topology_summary参数未使用

    # 构建prompt时完全忽略这些关键信息
    return f"""## Subtask
    {subtask.description}
    ...
    ## State
    - Room: {room_type}
    - Objects: {objects if objects else "none"}
    # nav_hint / walkable / topology 完全没有出现！
```

### 对比：medium模式正确使用了部分信息

**位置**: `agents/decision_agent.py` 第949-1028行 `_build_medium_prompt_with_analysis_v2`

```python
# ✅ medium模式正确使用open_dirs_str
open_dirs_str = "/".join(open_dirs) if open_dirs else "unknown"

# ✅ medium模式使用distance_trend
distance_delta = context.metadata.get("distance_delta", 0)
distance_trend = "接近" if distance_delta < 0 else "远离"

# ✅ medium模式使用topology_str
topology_str = ""
if topology_summary:
    topology_str = f"拓扑: 当前{current_node}, 路径{path}, 已访问{visited}"

# ✅ medium模式使用pos_delta
state_str = f"位置({curr_pos[0]:.1f},{curr_pos[2]:.1f}), 移动{h_dist:.1f}m"

# ❌ 但nav_hint仍然未在prompt中使用！
```

### 对比：hard模式同样问题

**位置**: `agents/decision_agent.py` 第1031-1151行 `_build_hard_prompt`

```python
# ✅ hard模式使用open_dirs_str（line 1139）
# ✅ hard模式使用topology_section（line 1141）
# ✅ hard模式使用opinion_section（line 1130）
# ❌ 但nav_hint同样未在prompt中使用！
```

**关键发现**: easy模式信息缺失最严重，但nav_hint在所有模式都被忽略。

## 修复方案

### 修复1：_build_simple_prompt_direct_v2 全面集成关键信息

**位置**: `agents/decision_agent.py` 第866-947行

**修复内容**:

```python
def _build_simple_prompt_direct_v2(
    self, seq_len: int, context, subtask, blocked, min_dist, dist_traveled,
    heading, distance_to_goal, directions, nav_hint, landmarks,
    goals, completion_condition=None, action_history_summary="",
    pos_delta=None, rot_delta=None, navigation=None, spatial_memory_guidance="",
    topology_summary: Optional[dict] = None
) -> str:
    # Get perception info directly from context
    perception_output = context.metadata.get("perception_output", {}) if context else {}
    room_type = perception_output.get("room_type", "unknown")
    objects_raw = perception_output.get("objects", [])
    objects = [o.get("name") for o in objects_raw[:3] if o.get("name")]

    # ✅ 新增：提取nav_hint和walkable_analysis
    perception_nav_hint = perception_output.get("nav_hint", "")
    walkable_analysis = perception_output.get("walkable_analysis", {})

    # ✅ 新增：格式化walkable信息
    walkable_str = ""
    recommended_dir = "center"
    if walkable_analysis:
        recommended_dir = walkable_analysis.get("recommended", "center")
        clear_dirs = []
        for dir_name in ["left", "center", "right"]:
            dir_info = walkable_analysis.get(dir_name, {})
            if dir_info.get("clear", True):
                depth = dir_info.get("depth_m", 0)
                clear_dirs.append(f"{dir_name}({depth:.1f}m)")
        if clear_dirs:
            walkable_str = f"可行走: {', '.join(clear_dirs)} → 推荐{recommended_dir}"

    # ✅ 新增：提取distance_trend（接近/远离）
    distance_delta = context.metadata.get("distance_delta", 0) if context else 0
    distance_trend = "接近" if distance_delta < -0.1 else "远离" if distance_delta > 0.1 else "稳定"

    # Get goal direction info
    angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
    direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

    # Get vertical difference (y coordinate)
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

    # ✅ 新增：格式化拓扑信息（简化版）
    topology_str = ""
    if topology_summary:
        visited_rooms = topology_summary.get("visited_rooms", [])[:3]
        stuck_regions = topology_summary.get("stuck_regions", [])
        if visited_rooms:
            topology_str = f"已访问: {', '.join(visited_rooms)}"
        if stuck_regions:
            topology_str += f" | 避开区域: {len(stuck_regions)}个"

    # ✅ 新增：格式化landmarks/goals（简化）
    landmarks_str = ""
    if landmarks and len(landmarks) > 0:
        landmarks_str = f"地标: {', '.join(landmarks[:3])}"
    goals_str = ""
    if goals and len(goals) > 0:
        goals_str = f"目标: {goals[0] if isinstance(goals[0], str) else goals[0].get('name', '?')}"

    # Get obstacle info
    obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}
    obstacle_str = ""
    if obstacle_info:
        obstacle_str = f"障碍: {obstacle_info.get('direction', '')} {obstacle_info.get('distance', 0):.1f}m"

    # Get current step count
    current_step = context.metadata.get("step", 0) if context else 0
    max_steps = self.MAX_STEPS

    # Generate example actions
    if self.adaptive_sequence:
        example_actions = '{"action":"forward"}, {"action":"turn_right"}, {"action":"forward"}'
    else:
        example_actions = ', '.join(['{"action":"forward"}' if i % 2 == 0 else '{"action":"turn_right"}' for i in range(seq_len)])

    # Determine step constraint
    if self.adaptive_sequence:
        step_rule = f"输出 {self.min_sequence_length}-{self.max_sequence_length} 步"
    else:
        step_rule = f"必须输出 {seq_len} 步"

    # ✅ 新增：构建Navigation Guidance section
    nav_guidance_section = ""
    if perception_nav_hint:
        nav_guidance_section = f"""
## 导航建议 (PerceptionAgent)
{perception_nav_hint}
**重要**: 此建议基于视觉分析，优先考虑"""

    return f"""## 子任务
{subtask.description}

## 最终目标
- 距离: {distance_to_goal:.1f}m (水平), 趋势: {distance_trend}
- 方向: {direction_hint} ({angle_to_goal:.0f}°)
- 垂直: {vertical_status}
- 成功条件: 水平 < 3m 且 垂直 < 1m
{f"- {goals_str}" if goals_str else ""}
{f"- {landmarks_str}" if landmarks_str else ""}

## 状态
- 房间: {room_type}
- 可见物体: {objects if objects else "无"}
{f"- {walkable_str}" if walkable_str else ""}
{f"- {obstacle_str}" if obstacle_str else ""}
- 已行进: {dist_traveled:.1f}m
- 步数: {current_step}/{max_steps}
{f"- {topology_str}" if topology_str else ""}
{nav_guidance_section}

## 规则
1. {step_rule}
2. **优先遵循导航建议的方向**
3. 转向纠正方向偏差（>15°）
4. 停止仅当：距离<3m 且 垂直满足
5. 若趋势=远离，重新考虑方向
6. 避开已标记的卡住区域

## 输出 (JSON)
{"reasoning":"简述原因","subtask_completed":false,"actions":[{example_actions}]}"""
```

### 修复2：_build_medium_prompt_with_analysis_v2 添加nav_hint

**位置**: `agents/decision_agent.py` 第949-1028行

**修复内容**: 在现有基础上添加nav_hint section（与easy模式相同的nav_guidance_section）。

### 修复3：_build_hard_prompt 添加nav_hint

**位置**: `agents/decision_agent.py` 第1031-1151行

**修复内容**: 在现有基础上添加nav_hint section。

### 修复4：确保信息提取完整（_build_sequence_prompt_v2）

**位置**: `agents/decision_agent.py` 第680-808行

**验证**: 确保distance_delta、walkable_analysis等信息存入context.metadata。

## 验证计划

修复后运行相同场景验证：

```bash
python run_vln_experiment.py \
  --use-remote-llm \
  --llm-server http://localhost:8000 \
  --episodes 5 \
  --max-steps 150 \
  --output-dir results/episode-navhint-fix
```

**预期结果**:
- Episode 5的DecisionAgent决策应与nav_hint建议一致
- 导航不再原地打转
- 成功率从28.6%提升至50%+

## 改动文件清单

| 文件 | 改动位置 | 改动内容 | 状态 |
|------|----------|----------|------|
| `agents/decision_agent.py` | 866-1000行 | easy模式全面集成信息 | ✅ 已完成 |
| `agents/decision_agent.py` | 1001-1085行 | medium模式添加nav_hint+distance_trend | ✅ 已完成 |
| `agents/decision_agent.py` | 1086-1220行 | hard模式添加nav_hint+distance_trend | ✅ 已完成 |

## 已修复问题

| # | 问题 | 修复内容 | 影响模式 |
|---|------|----------|----------|
| 1 | nav_hint未使用 | 三种模式都添加了"导航建议"section | easy/medium/hard |
| 2 | walkable_analysis未使用 | easy模式提取并展示可行走方向 | easy |
| 3 | distance_trend未使用 | 三种模式都展示趋势（接近/远离） | easy/medium/hard |
| 4 | topology未使用 | easy模式添加拓扑摘要 | easy |
| 5 | landmarks未使用 | easy模式添加地标信息 | easy |
| 6 | spatial_memory_guidance未使用 | 三种模式都添加"空间记忆辅助"section | easy/medium/hard |
| 7 | action_history_summary未使用 | easy/medium模式添加"动作历史"section | easy/medium |
| 8 | y_direction/y_change未展示 | 三种模式都展示高度变化趋势 | easy/medium/hard |

## 修复后Prompt新增内容

### Easy模式新增
- 导航建议section
- 空间记忆辅助section
- 动作历史section
- 可行走方向（walkable）
- 高度变化趋势（y_trend）
- 拓扑摘要
- 地标信息

### Medium模式新增
- 导航建议section
- 空间记忆辅助section
- 动作历史section
- 高度变化趋势（y_trend）

### Hard模式新增
- 导航建议section
- 空间记忆辅助section
- 规则增加楼梯记忆提示

## 风险评估

- **改动范围**: 中等，涉及3个prompt构建函数
- **影响面**: 所有DecisionAgent决策
- **回归风险**: 低，仅添加信息，不改变核心决策逻辑
- **Token影响**: 每个prompt增加约50-100 tokens，但信息价值高

---

*设计文档版本：2026-04-28-v2*
*状态：待审核*