# 拓扑信息实时传递修复设计

## 概述

修复序列执行期间拓扑信息不更新的问题，让TopologyGraph在序列执行中实时记录关键节点（stairs/junction）。

**核心问题**：序列执行期间TrajectoryAgent不被调用，TopologyGraph不更新，visited_rooms、stuck_regions等信息过期。

**解决方案**：TrajectoryAgent新增 `update_topology_only` 方法，run_vln_experiment每3步调用。

---

## 一、问题诊断

### 1.1 当前信息流

```
序列生成时 → TrajectoryAgent.process() → topology_graph更新 → trajectory_output.topology_summary
序列执行时 → TrajectoryAgent不调用 → topology_graph不更新 → topology_summary过期
```

### 1.2 问题根源

| 问题点 | 说明 |
|--------|------|
| TrajectoryAgent不调用 | 序列执行期间（5-10步）agent不执行 |
| topology_summary过期 | visited_rooms、stuck_regions不更新 |
| 关键节点遗漏 | 穿过楼梯入口不会被记录 |

### 1.3 实际影响

- stairs检测：序列执行中穿过楼梯 → 不记录为"stairs"节点 → 但perception的nav_hint仍能感知
- junction检测：序列执行中转向 → 不记录为转向点 → 拓扑图缺失路径分支信息
- room_entrance检测：room_type过期 → 无法检测 → 延迟到下一序列生成时处理

---

## 二、解决方案

### 2.1 核心改动

**TrajectoryAgent新增方法**：`update_topology_only`

**检测范围**：
- stairs：y变化 > 0.5m（实时可用）
- junction：转向 > 28°（实时可用）
- room_entrance：**跳过**（room_type过期，延迟到序列生成）

### 2.2 方法实现

```python
def update_topology_only(
    self,
    current_pos: tuple,
    prev_pos: tuple,
    current_rot: float,
    prev_rot: float,
    step_count: int
) -> Optional[str]:
    """轻量级拓扑更新（序列执行期间调用，仅检测stairs/junction）。
    
    Args:
        current_pos: 当前位置 (x, y, z)
        prev_pos: 前一步位置
        current_rot: 当前旋转角度（弧度）
        prev_rot: 前一步旋转角度
        step_count: 当前步数
        
    Returns:
        新节点ID（如果添加了节点），否则None
    """
    # 检测stairs（高度变化 > 0.5m）
    y_diff = current_pos[1] - prev_pos[1]
    if abs(y_diff) > 0.5:
        semantic_info = "楼梯上升" if y_diff > 0 else "楼梯下降"
        new_node_id = self.topology_graph.add_node(
            position=current_pos,
            node_type="stairs",
            semantic_info=semantic_info,
            timestamp=step_count
        )
        if self._last_topology_node_id:
            self.topology_graph.add_edge(
                source_id=self._last_topology_node_id,
                target_id=new_node_id
            )
        self._last_topology_node_id = new_node_id
        self.logger.debug(f"[Topology-lite] Added stairs node: {semantic_info}")
        return new_node_id
    
    # 检测junction（转向 > 28°，约0.5弧度）
    rot_diff = current_rot - prev_rot
    # 处理-π/π边界
    if rot_diff > math.pi:
        rot_diff -= 2 * math.pi
    elif rot_diff < -math.pi:
        rot_diff += 2 * math.pi
    
    if abs(rot_diff) > 0.5:
        new_node_id = self.topology_graph.add_node(
            position=current_pos,
            node_type="junction",
            semantic_info="转向点",
            timestamp=step_count
        )
        if self._last_topology_node_id:
            self.topology_graph.add_edge(
                source_id=self._last_topology_node_id,
                target_id=new_node_id
            )
        self._last_topology_node_id = new_node_id
        self.logger.debug(f"[Topology-lite] Added junction node")
        return new_node_id
    
    # 无关键位置变化，更新当前节点
    self.topology_graph.update_current_node(current_pos)
    return None
```

---

## 三、调用逻辑

### 3.1 调用时机与顺序

run_vln_experiment序列执行循环，每3步调用一次。

**调用位置**：rotation更新之后（第1456行 `context.rotation = yaw` 之后）

**完整顺序**：
```
1. context.position = tuple(pos)        # 第1447行
2. context.add_trajectory_point(...)    # 第1448行
3. context.rotation = yaw               # 第1456行
4. 调用拓扑更新（每3步）                 # 新增，在此处
5. 更新trajectory_output.topology_summary
```

### 3.2 调用代码

```python
# 在context.rotation更新后调用（第1456行之后）
if steps % 3 == 0 and self.trajectory_agent:
    # 从context获取前一步位置
    prev_pos = context.trajectory[-2] if len(context.trajectory) > 1 else tuple(pos)
    
    # 获取rotation（从context._prev_rotation获取上一帧）
    current_rot = context.rotation
    prev_rot = getattr(context, '_prev_rotation', current_rot)  # 第一帧默认=current_rot
    
    new_node_id = self.trajectory_agent.update_topology_only(
        current_pos=tuple(pos),
        prev_pos=prev_pos,
        current_rot=current_rot,
        prev_rot=prev_rot,
        step_count=steps
    )
    
    if new_node_id:
        self.logger.info(f"[拓扑] 序列执行中新增节点: {new_node_id}")
    
    # 更新trajectory_output的topology_summary
    if "trajectory_output" not in context.metadata:
        context.metadata["trajectory_output"] = {}
    context.metadata["trajectory_output"]["topology_summary"] = \
        self.trajectory_agent.topology_graph.get_summary()
    
    # 保存当前rotation用于下次比较
    context._prev_rotation = current_rot
```

### 3.3 prev_rotation初始化

第一帧调用时，`context._prev_rotation`不存在，默认使用`current_rot`，导致`rot_diff=0`。

**这是预期行为**：第一帧无法检测转向，从第二帧开始正常检测。

### 3.4 rotation更新位置

确认rotation在第1456行更新：

```python
# 第1456行
context.rotation = yaw
# 第1457行之后插入拓扑更新代码
```

---

## 四、文件改动清单

| 文件 | 改动内容 | 改动量 |
|------|----------|--------|
| `agents/trajectory_agent.py` | 新增 `update_topology_only` 方法 | ~25行 |
| `run_vln_experiment.py` | 序列执行循环调用拓扑更新 + prev_rot存储 | ~20行 |

**总改动量**：约45行

---

## 五、预期效果

| 指标 | 当前 | 修复后 |
|------|------|--------|
| stairs节点记录 | 延迟5-10步 | 实时记录 |
| junction节点记录 | 延迟或不记录 | 实时记录 |
| topology_summary更新频率 | 每序列生成 | 每3步 |

---

## 六、不处理的部分

**room_entrance检测**：序列执行期间room_type过期，保持延迟检测（序列生成时处理）。

---

## 七、设计自检

### 7.1 Placeholder检查
- 无TBD、TODO占位符
- 所有代码完整

### 7.2 Scope检查
- 仅涉及TrajectoryAgent和run_vln_experiment
- 不影响其他Agent

### 7.3 边界处理
- rotation边界（-π/π）已处理
- 第一帧junction检测失效是预期行为（prev_rot=current_rot导致rot_diff=0）

### 7.4 顺序检查
- 拓扑更新在rotation更新之后调用
- prev_rot在调用后保存（不是调用前）

---

*文档创建时间: 2026-04-25*