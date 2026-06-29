# 卡住恢复增强设计

## 概述

利用TopologyGraph的stuck_regions历史信息，增强卡住恢复策略，避免在相同死胡同反复尝试相同失败策略。

**核心目标**：
- 记录历史escape尝试结果（成功/失败方向）
- 复用成功方向，避免失败方向
- 结合深度图分析做最终决策

---

## 一、数据结构扩展

### 1.1 stuck_region扩展字段

当前TopologyGraph的stuck_region只记录位置，扩展为：

```python
stuck_region = {
    "position": Tuple[float, float, float],  # 卡住区域中心
    "radius": float,                          # 检测范围（默认1.5m）
    "escape_attempts": int,                   # 尝试次数
    "successful_direction": Optional[str],    # 成功方向: "left"/"right"/None
    "failed_directions": List[str],           # 失败方向列表: ["left", "right"]
    "last_attempt_step": int,                 # 最后尝试步数（防频繁重试）
    "created_at": int,                        # 创建时的导航步数
}
```

### 1.2 更新时机

- **创建**：首次在位置卡住时
- **更新**：执行escape序列后延迟判断结果（观察3-5步）

---

## 二、TrajectoryAgent新增方法

### 2.1 get_stuck_recovery_suggestion

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
        - preferred_direction: 推荐方向
        - avoid_directions: 需避免的方向
        - reason: 推荐理由
        - confidence: 置信度
        - use_depth_analysis: 是否需要深度图辅助
    """

    current_pos = context.position
    matched_region = self._find_matching_stuck_region(current_pos)

    if matched_region:
        # 有历史成功方向
        if matched_region["successful_direction"]:
            return {
                "preferred_direction": matched_region["successful_direction"],
                "avoid_directions": matched_region["failed_directions"],
                "reason": "历史成功方向",
                "confidence": 0.8,
                "use_depth_analysis": False
            }

        # 有历史失败方向（无成功）
        return {
            "preferred_direction": depth_clear_direction,
            "avoid_directions": matched_region["failed_directions"],
            "reason": "避开历史失败方向",
            "confidence": 0.6,
            "use_depth_analysis": True
        }

    # 无历史记录 → 创建新stuck_region
    self._create_stuck_region(current_pos, context.step_count)
    return {
        "preferred_direction": depth_clear_direction,
        "avoid_directions": [],
        "reason": "首次卡住，使用深度图分析",
        "confidence": 0.5,
        "use_depth_analysis": True
    }
```

### 2.2 _find_matching_stuck_region

```python
def _find_matching_stuck_region(self, position: Tuple[float, float, float]) -> Optional[Dict]:
    """通过距离阈值匹配已知stuck_region。

    Args:
        position: 当前位置

    Returns:
        匹配的stuck_region或None
    """
    if not self.topology_graph.stuck_regions:
        return None

    for region in self.topology_graph.stuck_regions:
        dx = position[0] - region["position"][0]
        dz = position[2] - region["position"][2]
        distance = math.sqrt(dx * dx + dz * dz)

        if distance < region["radius"]:
            return region

    return None
```

### 2.3 _create_stuck_region

```python
def _create_stuck_region(self, position: Tuple[float, float, float], step: int) -> None:
    """创建新的stuck_region记录。

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

    self.topology_graph.stuck_regions.append(new_region)
    self.logger.info(f"[Topology] Created stuck_region at {position}")
```

### 2.4 mark_escape_result

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
        direction: 尝试方向
        success: 是否成功逃离
        step: 当前步数
    """
    region = self._find_matching_stuck_region(position)
    if not region:
        return

    region["last_attempt_step"] = step
    region["escape_attempts"] += 1

    if success:
        region["successful_direction"] = direction
        # 从失败列表移除（如果之前标记过）
        if direction in region["failed_directions"]:
            region["failed_directions"].remove(direction)
        self.logger.info(f"[Topology] Escape success with {direction} at region {region['position']}")
    else:
        if direction not in region["failed_directions"]:
            region["failed_directions"].append(direction)
        self.logger.warning(f"[Topology] Escape failed with {direction} at region {region['position']}")
```

---

## 三、调用链

### 3.1 run_vln_experiment集成

```python
# 检测卡住
if stuck_counter > STUCK_THRESHOLD:
    # 获取深度图分析（已有方法）
    depth_clear = self._check_depth_clear_direction(depth_image)

    # 获取恢复建议
    suggestion = self.trajectory_agent.get_stuck_recovery_suggestion(
        context,
        depth_clear_direction=depth_clear
    )

    # 存储建议供DecisionAgent使用
    context.metadata["stuck_recovery_suggestion"] = suggestion

    # 标记开始escape
    context.metadata["escape_started"] = {
        "step": context.step_count,
        "position": context.position,
        "direction": suggestion["preferred_direction"]
    }
```

### 3.2 DecisionAgent使用建议

```python
# DecisionAgent在生成动作序列时
suggestion = context.metadata.get("stuck_recovery_suggestion")

if suggestion:
    preferred = suggestion.get("preferred_direction")
    avoid = suggestion.get("avoid_directions", [])

    # 根据偏好生成序列
    if preferred == "left":
        actions = ["turn_left", "forward", "forward", "forward"]
    elif preferred == "right":
        actions = ["turn_right", "forward", "forward", "forward"]
    else:
        # 无偏好或深度图结果
        actions = self._generate_default_escape()
```

### 3.3 延迟更新触发

```python
# 执行escape序列后（约3-5步）
escape_start = context.metadata.get("escape_started")
if escape_start and context.step_count >= escape_start["step"] + 5:
    # 计算平均移动量
    recent_positions = context.trajectory[-5:]
    avg_movement = calculate_avg_movement(recent_positions)

    # 判断成功/失败
    success = avg_movement > 0.3  # 成功阈值

    # 更新结果
    self.trajectory_agent.mark_escape_result(
        position=escape_start["position"],
        direction=escape_start["direction"],
        success=success,
        step=context.step_count
    )

    # 清除escape标记
    context.metadata.pop("escape_started", None)
```

---

## 四、改动文件清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `agents/topology_graph.py` | stuck_regions结构 | 扩展字段定义 |
| `agents/trajectory_agent.py` | 新增方法 | `get_stuck_recovery_suggestion`, `_find_matching_stuck_region`, `_create_stuck_region`, `mark_escape_result` |
| `run_vln_experiment.py` | 卡住检测处 | 调用恢复建议 + 延迟更新逻辑 |

---

## 五、预期效果

| 指标 | 改动前 | 改动后 |
|------|--------|--------|
| 重复死胡同处理 | 随机尝试 | 复用成功方向 |
| escape效率 | 低（多次失败尝试） | 高（首次或历史成功） |
| 卡住恢复成功率 | ~30%（估计） | ~50%（预期提升） |

---

## 六、设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 方案架构 | TrajectoryAgent主导 | TopologyGraph在此Agent，改动最小 |
| stuck_region字段 | 中等扩展 | 记录尝试历史，不过度复杂 |
| 区域匹配方式 | 距离阈值 | 简单可靠，语义可能不稳定 |
| 结果更新时机 | 延迟更新（3-5步后） | 单步波动大，序列结果更准确 |
| 建议输出格式 | 方向偏好（C方案） | 保持职责分离，DecisionAgent生成序列 |
| 深度图获取方式 | run_vln_experiment传入 | 已有方法，避免重复分析 |

---

*文档创建时间: 2026-04-22*