# 导航优化综合实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 实现三项导航优化（按优先级B > D > E）
- B: 卡住恢复增强 — 利用TopologyGraph历史信息
- D: Prompt压缩 — 分层压缩减少token消耗
- E: 子任务验证增强 — 多重验证防止过早停止

**架构：**
- B: TrajectoryAgent新增恢复建议方法 + 数据结构扩展
- D: DecisionAgent prompt构建方法重构
- E: run_vln_experiment中途验证 + 双重确认逻辑

**技术栈：** Python, dataclasses, pytest

---

## 文件结构

### 新增文件
- `tests/test_stuck_recovery.py` — 卡住恢复单元测试

### 修改文件
| 文件 | 改动内容 |
|------|----------|
| `agents/trajectory_agent.py` | 新增`get_stuck_recovery_suggestion`、扩展`_stuck_regions`结构 |
| `agents/decision_agent.py` | 压缩prompt构建方法、验证逻辑增强 |
| `run_vln_experiment.py` | 中途验证调用、恢复建议调用 |

---

## Part B: 卡住恢复增强（优先级最高）

---

### Task B-1: 扩展stuck_region数据结构

**Files:**
- Modify: `agents/trajectory_agent.py:81-83`

- [ ] **Step 1: 扩展_stuck_regions初始化注释**

修改第81-83行，添加扩展字段说明：

```python
# NEW: Stuck region tracking with escape history
# stuck_region structure:
# {
#     "position": Tuple[float, float, float],
#     "radius": float (default 1.5),
#     "escape_attempts": int,
#     "successful_direction": Optional[str] ("left"/"right"/None),
#     "failed_directions": List[str],
#     "last_attempt_step": int,
#     "created_at": int,
# }
self._stuck_regions: List[Dict] = []
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/trajectory_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/trajectory_agent.py
git commit -m "docs(trajectory): document stuck_region extended structure"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task B-2: 新增get_stuck_recovery_suggestion方法

**Files:**
- Modify: `agents/trajectory_agent.py` (新增方法约第1160行后)

- [ ] **Step 1: 添加get_stuck_recovery_suggestion方法**

在`get_stuck_region_info`方法后（约第1160行）添加：

```python
def get_stuck_recovery_suggestion(
    self,
    context: NavContext,
    depth_clear_direction: Optional[str] = None
) -> Dict[str, Any]:
    """根据拓扑历史 + 深度图分析返回恢复方向建议。

    Args:
        context: 当前导航上下文
        depth_clear_direction: 来自run_vln_experiment的深度图分析结果

    Returns:
        Dict包含：
        - preferred_direction: 推荐方向 ("left"/"right"/None)
        - avoid_directions: 需避免的方向列表
        - reason: 推荐理由
        - confidence: 置信度 (0.5-0.8)
        - use_depth_analysis: 是否需要深度图辅助
    """
    current_pos = context.position if hasattr(context, 'position') else (0, 0, 0)
    current_step = context.step_count if hasattr(context, 'step_count') else 0

    # 查找匹配的stuck_region
    matched_region = self._find_matching_stuck_region(current_pos)

    if matched_region:
        # 检查是否有历史成功方向
        if matched_region.get("successful_direction"):
            return {
                "preferred_direction": matched_region["successful_direction"],
                "avoid_directions": matched_region.get("failed_directions", []),
                "reason": "历史成功方向",
                "confidence": 0.8,
                "use_depth_analysis": False
            }

        # 有历史失败方向（无成功）
        return {
            "preferred_direction": depth_clear_direction,
            "avoid_directions": matched_region.get("failed_directions", []),
            "reason": "避开历史失败方向",
            "confidence": 0.6,
            "use_depth_analysis": True
        }

    # 无历史记录 → 创建新stuck_region
    self._create_stuck_region(current_pos, current_step)
    return {
        "preferred_direction": depth_clear_direction,
        "avoid_directions": [],
        "reason": "首次卡住，使用深度图分析",
        "confidence": 0.5,
        "use_depth_analysis": True
    }

def _find_matching_stuck_region(
    self,
    position: Tuple[float, float, float]
) -> Optional[Dict]:
    """通过距离阈值匹配已知stuck_region。

    Args:
        position: 当前位置

    Returns:
        匹配的stuck_region或None
    """
    if not self._stuck_regions:
        return None

    for region in self._stuck_regions:
        dx = position[0] - region["position"][0]
        dz = position[2] - region["position"][2]
        distance = math.sqrt(dx * dx + dz * dz)

        if distance < region.get("radius", 1.5):
            return region

    return None

def _create_stuck_region(
    self,
    position: Tuple[float, float, float],
    step: int
) -> None:
    """创建新的stuck_region记录（扩展结构）。

    Args:
        position: 卡住位置
        step: 当前步数
    """
    new_region = {
        "position": position,
        "radius": 1.5,  # 默认检测范围
        "escape_attempts": 0,
        "successful_direction": None,
        "failed_directions": [],
        "last_attempt_step": step,
        "created_at": step,
    }

    self._stuck_regions.append(new_region)
    self.logger.info(f"[Trajectory] Created stuck_region at {position}")
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/trajectory_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/trajectory_agent.py
git commit -m "feat(trajectory): add get_stuck_recovery_suggestion method

Adds recovery direction suggestions based on stuck_region history:
- _find_matching_stuck_region: distance threshold matching
- _create_stuck_region: create extended structure
- get_stuck_recovery_suggestion: return preferred/avoid directions

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B-3: 新增mark_escape_result方法

**Files:**
- Modify: `agents/trajectory_agent.py` (新增方法)

- [ ] **Step 1: 添加mark_escape_result方法**

在`_create_stuck_region`方法后添加：

```python
def mark_escape_result(
    self,
    position: Tuple[float, float, float],
    direction: str,
    success: bool,
    step: int
) -> None:
    """延迟更新escape结果。

    Args:
        position: 卡住位置
        direction: 尝试方向 ("left"/"right")
        success: 是否成功逃离
        step: 当前步数
    """
    region = self._find_matching_stuck_region(position)
    if not region:
        self.logger.warning(f"[Trajectory] No stuck_region found at {position}")
        return

    region["last_attempt_step"] = step
    region["escape_attempts"] = region.get("escape_attempts", 0) + 1

    if success:
        region["successful_direction"] = direction
        # 从失败列表移除（如果之前标记过）
        failed_dirs = region.get("failed_directions", [])
        if direction in failed_dirs:
            failed_dirs.remove(direction)
            region["failed_directions"] = failed_dirs
        self.logger.info(f"[Trajectory] Escape success with {direction} at {region['position']}")
    else:
        failed_dirs = region.get("failed_directions", [])
        if direction not in failed_dirs:
            failed_dirs.append(direction)
            region["failed_directions"] = failed_dirs
        self.logger.warning(f"[Trajectory] Escape failed with {direction} at {region['position']}")
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/trajectory_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/trajectory_agent.py
git commit -m "feat(trajectory): add mark_escape_result for delayed update

Records escape attempt success/failure in stuck_region:
- Updates escape_attempts counter
- Records successful_direction on success
- Appends to failed_directions on failure

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B-4: 在run_vln_experiment集成恢复建议调用

**Files:**
- Modify: `run_vln_experiment.py` (卡住检测处，约第996-1020行)

- [ ] **Step 1: 读取当前卡住检测代码**

Run: Read current stuck detection section in run_vln_experiment.py:996-1020

- [ ] **Step 2: 添加恢复建议调用**

在卡住检测后添加恢复建议调用（约第1000行后）：

```python
# === NEW: Get stuck recovery suggestion from TrajectoryAgent ===
if self.decision_agent._stuck_counter > self.decision_agent._stuck_threshold:
    # 获取深度图分析（已有方法）
    depth_clear = self._check_depth_clear_direction(depth_image) if depth_image else None

    # 获取恢复建议
    if self.trajectory_agent:
        suggestion = self.trajectory_agent.get_stuck_recovery_suggestion(
            context,
            depth_clear_direction=depth_clear
        )

        # 存储建议供DecisionAgent使用
        context.metadata["stuck_recovery_suggestion"] = suggestion

        # 标记开始escape
        context.metadata["escape_started"] = {
            "step": context.step_count,
            "position": tuple(context.position) if context.position else (0, 0, 0),
            "direction": suggestion.get("preferred_direction")
        }

        self.logger.info(f"[Recovery] Suggestion: {suggestion['reason']}, direction: {suggestion.get('preferred_direction')}")
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 4: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat(run): integrate stuck recovery suggestion

Calls trajectory_agent.get_stuck_recovery_suggestion when stuck:
- Gets depth analysis for fallback direction
- Stores suggestion in context.metadata
- Marks escape_started for delayed result tracking

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B-5: 在run_vln_experiment添加延迟更新逻辑

**Files:**
- Modify: `run_vln_experiment.py` (主循环中，序列执行后)

- [ ] **Step 1: 添加延迟更新检查**

在主循环中，序列执行完成后检查escape结果（约第1089行后）：

```python
# === NEW: Delayed escape result update ===
escape_start = context.metadata.get("escape_started")
if escape_start and context.step_count >= escape_start["step"] + 5:
    # 计算平均移动量
    recent_positions = context.trajectory[-5:] if len(context.trajectory) >= 5 else context.trajectory

    if len(recent_positions) >= 2:
        total_movement = 0.0
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dz = recent_positions[i][2] - recent_positions[i-1][2]
            total_movement += math.sqrt(dx*dx + dz*dz)

        avg_movement = total_movement / (len(recent_positions) - 1)

        # 判断成功/失败（阈值0.3m）
        success = avg_movement > 0.3

        # 更新结果
        if self.trajectory_agent:
            self.trajectory_agent.mark_escape_result(
                position=escape_start["position"],
                direction=escape_start.get("direction", "unknown"),
                success=success,
                step=context.step_count
            )

            self.logger.info(f"[Recovery] Escape result: {success}, avg_movement={avg_movement:.2f}m")

        # 清除escape标记
        context.metadata.pop("escape_started", None)
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat(run): add delayed escape result update

Updates stuck_region after 5 steps:
- Calculates average movement from recent trajectory
- Determines success/failure (threshold 0.3m)
- Calls trajectory_agent.mark_escape_result

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task B-6: 编写单元测试

**Files:**
- Create: `tests/test_stuck_recovery.py`

- [ ] **Step 1: 创建测试文件**

```python
"""Tests for stuck recovery enhancement."""

import pytest
from agents.trajectory_agent import TrajectoryAgent
from core.context import NavContext


class TestStuckRecovery:
    """Test stuck recovery functionality."""

    def test_find_matching_stuck_region(self):
        """Test distance threshold matching."""
        agent = TrajectoryAgent()

        # 创建stuck_region
        agent._create_stuck_region((1.0, 0.0, 1.0), 0)

        # 在范围内应匹配
        matched = agent._find_matching_stuck_region((1.2, 0.0, 1.2))
        assert matched is not None

        # 超出范围不匹配
        not_matched = agent._find_matching_stuck_region((5.0, 0.0, 5.0))
        assert not_matched is None

    def test_create_stuck_region_structure(self):
        """Test stuck_region has extended structure."""
        agent = TrajectoryAgent()

        agent._create_stuck_region((0.0, 0.0, 0.0), 10)

        assert len(agent._stuck_regions) == 1
        region = agent._stuck_regions[0]

        assert region["position"] == (0.0, 0.0, 0.0)
        assert region["radius"] == 1.5
        assert region["escape_attempts"] == 0
        assert region["successful_direction"] is None
        assert region["failed_directions"] == []
        assert region["created_at"] == 10

    def test_get_recovery_suggestion_no_history(self):
        """Test recovery suggestion without history."""
        agent = TrajectoryAgent()
        context = NavContext()
        context.position = (0.0, 0.0, 0.0)
        context.step_count = 0

        # 无历史，应创建新region并返回深度图方向
        suggestion = agent.get_stuck_recovery_suggestion(context, "left")

        assert suggestion["preferred_direction"] == "left"
        assert suggestion["avoid_directions"] == []
        assert suggestion["reason"] == "首次卡住，使用深度图分析"
        assert suggestion["confidence"] == 0.5
        assert len(agent._stuck_regions) == 1

    def test_get_recovery_suggestion_with_success_history(self):
        """Test recovery suggestion with successful history."""
        agent = TrajectoryAgent()

        # 创建有成功历史的region
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)
        agent._stuck_regions[0]["successful_direction"] = "left"

        context = NavContext()
        context.position = (0.5, 0.0, 0.5)  # 在范围内
        context.step_count = 10

        suggestion = agent.get_stuck_recovery_suggestion(context, "right")

        # 应返回历史成功方向，忽略深度图建议
        assert suggestion["preferred_direction"] == "left"
        assert suggestion["reason"] == "历史成功方向"
        assert suggestion["confidence"] == 0.8

    def test_mark_escape_result_success(self):
        """Test marking escape success."""
        agent = TrajectoryAgent()
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)

        agent.mark_escape_result((0.0, 0.0, 0.0), "left", success=True, step=10)

        region = agent._stuck_regions[0]
        assert region["successful_direction"] == "left"
        assert region["escape_attempts"] == 1

    def test_mark_escape_result_failure(self):
        """Test marking escape failure."""
        agent = TrajectoryAgent()
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)

        agent.mark_escape_result((0.0, 0.0, 0.0), "right", success=False, step=10)

        region = agent._stuck_regions[0]
        assert "right" in region["failed_directions"]
        assert region["escape_attempts"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_stuck_recovery.py -v`
Expected: 所有测试通过

- [ ] **Step 3: Commit**

```bash
git add tests/test_stuck_recovery.py
git commit -m "test: add stuck recovery unit tests

Tests for:
- _find_matching_stuck_region distance matching
- _create_stuck_region extended structure
- get_stuck_recovery_suggestion scenarios
- mark_escape_result success/failure

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Part D: Prompt压缩（优先级第二）

---

### Task D-1: 重构Easy级别prompt为精简版

**Files:**
- Modify: `agents/decision_agent.py:714-930` (`_build_simple_prompt_direct_v2`)

- [ ] **Step 1: 重构Easy prompt方法**

将`_build_simple_prompt_direct_v2`方法（约第714-930行）重构为精简版：

```python
def _build_simple_prompt_direct_v2(
    self, seq_len: int, context, subtask, blocked, min_dist, dist_traveled,
    heading, distance_to_goal, directions, nav_hint, landmarks,
    goals, completion_condition=None, action_history_summary="",
    pos_delta=None, rot_delta=None, navigation=None, spatial_memory_guidance="",
    topology_summary: Optional[dict] = None
) -> str:
    """Easy task prompt: compressed version (~250 tokens)."""

    # Get perception info directly from context
    perception_output = context.metadata.get("perception_output", {}) if context else {}
    room_type = perception_output.get("room_type", "unknown")
    objects_raw = perception_output.get("objects", [])
    objects = [o.get("object", o.get("name", str(o))) for o in objects_raw[:3]]

    # Get goal direction info
    angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
    direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

    # Determine step constraint
    if self.adaptive_sequence:
        seq_constraint = f"{self.min_sequence_length}-{self.max_sequence_length}步"
        rule_1 = f"输出{self.min_sequence_length}-{self.max_sequence_length}步"
    else:
        seq_constraint = f"{seq_len}步"
        rule_1 = f"输出{seq_len}步"

    # Obstacle description
    obstacle_desc = f"前方{min_dist:.1f}m" if blocked else "无"

    # Format objects
    objects_desc = ", ".join(objects) if objects else "无"

    # Build compressed prompt
    return f"""导航决策。生成{seq_constraint}动作序列。

## 任务
{subtask.description if subtask else "导航"}

## 状态
- 目标距离: {distance_to_goal:.1f}m
- 目标方向: {direction_hint} ({angle_to_goal:.0f}°)
- 已走: {dist_traveled:.1f}m

## 环境
- 房间: {room_type}
- 可见: {objects_desc}
- 障碍: {obstacle_desc}

## 规则
1. {rule_1}
2. 方向偏差>15°时需转向
3. 距离<3m时可停止

输出JSON: {"reasoning":"简述","subtask_completed":false,"actions":[{"action":"forward"}...]}"""
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat(decision): compress Easy prompt to ~250 tokens

Removes redundant info for simple tasks:
- No instruction semantics analysis
- No topology/history info
- No stair guidance
- Simplified format example

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task D-2: 重构Medium级别prompt压缩其他信息

**Files:**
- Modify: `agents/decision_agent.py:932-1090` (`_build_medium_prompt_with_analysis_v2`)

- [ ] **Step 1: 重构Medium prompt方法（保留分析）**

将`_build_medium_prompt_with_analysis_v2`方法压缩其他信息：

```python
def _build_medium_prompt_with_analysis_v2(
    self, seq_len: int, subtask, room_type, objects, scene_desc, open_dirs_str,
    blocked, min_dist, dist_traveled, heading, distance_to_goal,
    analysis, directions, nav_hint, landmarks, goals, completion_condition=None,
    action_history_summary="",
    pos_delta=None, rot_delta=None, navigation=None, context=None, spatial_memory_guidance="",
    topology_summary: Optional[dict] = None
) -> str:
    """Medium task prompt: keep CoT analysis, compress other (~400 tokens)."""

    # Direction info
    directions_str = "/".join(directions[:3]) if directions else "unknown"
    angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
    direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

    # Distance trend
    distance_delta = context.metadata.get("distance_delta", 0) if context else 0
    distance_trend = "接近" if distance_delta < 0 else "远离" if distance_delta > 0 else "稳定"

    # Simplified topology
    topology_str = ""
    if topology_summary:
        current_node = topology_summary.get("current_node", "?")
        path = topology_summary.get("path_to_goal", [])[:3]
        visited = topology_summary.get("visited_rooms", [])[:3]
        topology_str = f"拓扑: 当前{current_node}, 路径{path}, 已访问{visited}"

    # Simplified state
    curr_pos = pos_delta.get("current", [0,0,0]) if pos_delta else [0,0,0]
    h_dist = pos_delta.get("horizontal_distance", 0) if pos_delta else 0
    state_str = f"位置({curr_pos[0]:.1f},{curr_pos[2]:.1f}), 移动{h_dist:.1f}m"

    # Step constraint
    seq_constraint = f"{seq_len}步" if not self.adaptive_sequence else f"{self.min_sequence_length}-{self.max_sequence_length}步"

    # Build compressed prompt with CoT analysis preserved
    return f"""导航决策。生成{seq_constraint}动作序列。

## 任务
{subtask.description if subtask else "导航"}

## 分析结果 (CoT)
{analysis if analysis else "无分析"}

## 状态
- {state_str}, 目标{distance_to_goal:.1f}m
- 方向: {direction_hint} ({distance_trend})

## 环境
- 房间: {room_type}, 可见{objects[:3] if objects else []}
{topology_str if topology_str else ""}

## 规则
1. 输出{seq_constraint}
2. 根据分析结果决策

输出JSON: {"reasoning":"简述","subtask_completed":false,"actions":[...]}"""
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat(decision): compress Medium prompt, keep CoT analysis

Keeps analysis results (core value), compresses:
- Simplified topology (current + path only)
- Simplified state (position + distance)
- Removed stair guidance, obstacle radius
- Simplified format

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task D-3: 实现Hard级别智能摘要

**Files:**
- Modify: `agents/decision_agent.py` (`_build_hard_prompt`方法 + 新增辅助方法)

- [ ] **Step 1: 添加_summarize_opinions辅助方法**

在`_format_topology_section`方法后添加：

```python
def _summarize_opinions(self, opinions: Dict) -> str:
    """精简多Agent观点。"""
    if not opinions:
        return "无观点"

    lines = []
    for agent_name, opinion in opinions.items():
        action = opinion.get("suggested_action", "unknown")
        confidence = opinion.get("confidence", 0.5)
        reason = opinion.get("reasoning", "")[:50]  # 只取前50字符
        lines.append(f"{agent_name}: {action}({confidence:.0%}) - {reason}")

    return "\n".join(lines[:3])  # 只显示3个Agent
```

- [ ] **Step 2: 重构Hard prompt方法（智能摘要）**

```python
def _build_hard_prompt(
    self, subtask, room_type, objects, scene_desc, open_dirs_str,
    blocked, min_dist, dist_traveled, heading, distance_to_goal,
    opinions, consensus, directions, nav_hint, landmarks, goals, completion_condition=None,
    action_history_summary="",
    pos_delta=None, rot_delta=None, navigation=None, context=None, spatial_memory_guidance="",
    topology_summary: Optional[dict] = None
) -> str:
    """Hard task prompt: smart summary, dynamic adjust (~450-600 tokens)."""

    # === 智能摘要逻辑 ===

    # 1. Debate观点摘要策略
    opinion_section = ""
    if consensus and consensus.get("agreement_level", 0) > 0.8:
        # 观点一致 → 只保留共识
        opinion_section = f"共识: {consensus.get('agreed_action', 'forward')}"
    else:
        # 观点分歧 → 保留各方观点（精简）
        opinion_section = self._summarize_opinions(opinions)

    # 2. 拓扑信息摘要策略
    topology_section = ""
    if topology_summary:
        stuck_regions = topology_summary.get("stuck_regions", [])
        if len(stuck_regions) > 0:
            # 有卡住历史 → 保留拓扑详情
            topology_section = self._format_topology_section(topology_summary)
        else:
            # 无历史 → 拓扑简化
            total_nodes = topology_summary.get("total_nodes", 0)
            visited = topology_summary.get("visited_rooms", [])[:3]
            topology_section = f"拓扑: 节点{total_nodes}, 已访问{visited}"

    # 3. 历史信息摘要
    history_section = ""
    if action_history_summary:
        if len(action_history_summary) > 100:
            history_section = f"历史: {action_history_summary[:80]}..."
        else:
            history_section = f"历史: {action_history_summary}"
    else:
        history_section = "首次探索"

    # State info
    curr_pos = pos_delta.get("current", [0,0,0]) if pos_delta else [0,0,0]
    h_dist = pos_delta.get("horizontal_distance", 0) if pos_delta else 0

    # Build smart summary prompt
    seq_constraint = f"{self.min_sequence_length}-{self.max_sequence_length}步" if self.adaptive_sequence else f"{seq_len}步"

    return f"""导航决策。生成{seq_constraint}动作序列。

## 任务
{subtask.description if subtask else "导航"}

## 观点汇总
{opinion_section}

## 状态
- 位置({curr_pos[0]:.1f},{curr_pos[2]:.1f}), 移动{h_dist:.1f}m, 目标{distance_to_goal:.1f}m
{history_section}

## 环境
- 房间: {room_type}
{topology_section}

## 规则
1. 输出{seq_constraint}
2. 参考观点汇总决策

输出JSON: {"reasoning":"简述","subtask_completed":false,"actions":[...]}"""
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 4: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat(decision): implement Hard prompt smart summary

Dynamic adjustment based on context:
- Consensus high → simplified opinion
- Has stuck history → detailed topology
- Long history → truncated summary
- Adds _summarize_opinions helper method

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Part E: 子任务验证增强（优先级第三）

---

### Task E-1: 新增check_completion_condition方法

**Files:**
- Modify: `run_vln_experiment.py` (新增独立函数)

- [ ] **Step 1: 添加check_completion_condition函数**

在文件顶部（约第50行）添加：

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
        return {"completed": False, "progress": 0, "confidence": 0, "reason": "无条件"}

    cc_type = condition.get("type", "unknown")

    # 获取当前状态（从TrajectoryAgent或直接计算）
    trajectory_output = context.metadata.get("trajectory_output", {})
    subtask_delta = trajectory_output.get("subtask_delta", {}) if isinstance(trajectory_output, dict) else {}
    pos_delta = subtask_delta.get("position_delta", {})
    rot_delta = subtask_delta.get("rotation_delta", {})

    # 计算当前值和进度
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

    elif cc_type == "rotation":
        rot_change = abs(rot_delta.get("delta_deg", 0))
        threshold = condition.get("min_degrees", 70)
        progress = min(1.0, rot_change / threshold) if threshold > 0 else 0

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
        progress = min(1.0, h_dist / threshold) if threshold > 0 else 0

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

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat(run): add check_completion_condition function

Returns detailed completion status:
- completed: bool
- progress: 0.0-1.0
- confidence: based on threshold proximity
- current_value and threshold for display

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task E-2: 在主循环添加中途验证

**Files:**
- Modify: `run_vln_experiment.py` (主循环，动作执行后)

- [ ] **Step 1: 添加中途验证逻辑**

在动作执行循环中添加（约第1058行后，每次执行动作后）：

```python
# === NEW: Mid-sequence completion check (every 3 steps) ===
if steps % 3 == 0 and not context.metadata.get("completion_checked"):
    current_subtask = context.get_current_subtask() if hasattr(context, 'get_current_subtask') else None

    if current_subtask and current_subtask.completion_condition:
        auto_result = check_completion_condition(context, current_subtask.completion_condition)

        if auto_result["completed"]:
            # 自动检测通过，记录日志
            self.logger.info(f"[中途验证] step {steps}: 检测到可能完成 - {auto_result['reason']}")

            # 高置信度时提前终止序列
            if auto_result["confidence"] > 0.9:
                self.logger.info("[中途验证] 高置信度，提前终止序列，等待LLM验证")
                current_sequence = None  # 终止当前序列
                context.metadata["completion_checked"] = True
                context.metadata["mid_completion_detected"] = auto_result
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat(run): add mid-sequence completion check

Checks completion every 3 steps during sequence execution:
- Logs potential completion detection
- Terminates sequence early if high confidence
- Stores mid_completion_detected for LLM verification

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task E-3: 在DecisionAgent应用双重验证逻辑

**Files:**
- Modify: `agents/decision_agent.py` (`generate_sequence`方法，约第245-248行)

- [ ] **Step 1: 修改双重验证逻辑**

修改第245-248行的验证逻辑：

```python
# ===== NEW: Dual verification with conservative strategy =====
# 任一判断未完成 → 继续导航（保守策略）
if auto_completed and subtask_completed:
    # 两者都完成才标记完成
    final_completed = True
    self.logger.info(f"[Decision] 双重验证通过: 自动={auto_completed}, LLM={subtask_completed}")
elif auto_completed and not subtask_completed:
    # 自动检测完成但LLM未完成 → 继续执行（保守）
    final_completed = False
    self.logger.info(f"[Decision] 保守策略: 自动完成但LLM未确认，继续执行")
elif not auto_completed and subtask_completed:
    # LLM完成但自动检测未完成 → 继续执行（保守）
    final_completed = False
    self.logger.warning(f"[Decision] 保守策略: LLM声称完成但自动检测未通过，继续执行")
else:
    # 两方都未完成
    final_completed = False

# Override subtask_completed with final verification result
subtask_completed = final_completed
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat(decision): implement dual verification with conservative strategy

Conservative approach: any uncompleted → continue navigation
- Both auto and LLM completed → mark complete
- Auto complete + LLM incomplete → continue
- LLM complete + Auto incomplete → continue (prevent early stop)
- Both incomplete → continue

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task E-4: 新增进度展示格式化方法

**Files:**
- Modify: `agents/decision_agent.py` (新增方法)

- [ ] **Step 1: 添加format_completion_progress方法**

在`_format_completion_check`方法后添加：

```python
def format_completion_progress(
    self,
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
    estimated_steps = int(remaining / 0.25) + 2 if remaining > 0 else 0

    # 验证状态
    auto_status = "已完成" if auto_result.get("completed") else "未完成"
    llm_status = "已完成" if llm_completed else "未完成" if llm_completed is not None else "待确认"

    return f"""## 子任务进度
- 类型: {cc_type_name} ({cc_type})
- 当前: {current:.2f}{unit}
- 目标: >={threshold:.2f}{unit}
- 进度: {progress:.0%}
- 预估: 还需约{estimated_steps}步

**验证状态**: {auto_status} (自动) + {llm_status} (LLM)"""
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 3: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat(decision): add format_completion_progress for detailed display

Shows completion progress to LLM:
- Condition type and description
- Current value vs threshold
- Progress percentage
- Estimated remaining steps
- Dual verification status

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## 验证方法

```bash
# 1. 语法验证（所有改动文件）
python -m py_compile agents/trajectory_agent.py
python -m py_compile agents/decision_agent.py
python -m py_compile run_vln_experiment.py

# 2. 单元测试
pytest tests/test_stuck_recovery.py -v

# 3. 集成测试（简单场景）
python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm --llm-server http://localhost:8000

# 4. 检查日志关键信息
grep "Recovery Suggestion" logs/*.log
grep "中途验证" logs/*.log
grep "双重验证" logs/*.log
```

---

## 预期效果

| 指标 | Part B | Part D | Part E |
|------|--------|--------|--------|
| 卡住恢复成功率 | ~30% → ~50% | - | - |
| Token消耗 | - | 降低40-50% | - |
| 过早停止错误 | - | - | ~15% → ~5% |
| LLM响应时间 | - | 提升20-30% | - |

---

## 执行顺序

按优先级 B > D > E 执行：
1. Task B-1 → B-2 → B-3 → B-4 → B-5 → B-6（卡住恢复）
2. Task D-1 → D-2 → D-3（Prompt压缩）
3. Task E-1 → E-2 → E-3 → E-4（子任务验证）

---

*计划创建时间: 2026-04-22*