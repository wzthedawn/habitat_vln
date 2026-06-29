# 拓扑信息实时传递修复实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现拓扑信息实时传递，让TopologyGraph在序列执行期间实时记录stairs/junction节点。

**Architecture:** TrajectoryAgent新增`update_topology_only`方法，run_vln_experiment每3步调用，实时更新topology_summary。

**Tech Stack:** Python, pytest, math

---

## Context

当前问题：序列执行期间TrajectoryAgent不被调用，TopologyGraph不更新，导致拓扑信息过期。

解决方案：新增轻量级拓扑更新方法，仅检测stairs和junction（room_entrance延迟检测）。

---

## File Structure

| 文件 | 改动类型 | 职责 |
|------|----------|------|
| `tests/test_topology_graph.py` | 修改 | 添加update_topology_only单元测试 |
| `agents/trajectory_agent.py` | 修改 | 新增update_topology_only方法 |
| `run_vln_experiment.py` | 修改 | 序列执行循环调用拓扑更新 |

---

## Task 1: 添加单元测试

**Files:**
- Modify: `tests/test_topology_graph.py`

- [ ] **Step 1: 添加测试类骨架**

在`test_topology_graph.py`末尾添加：

```python
class TestUpdateTopologyOnly:
    """Tests for TrajectoryAgent.update_topology_only method."""
    
    @pytest.fixture
    def trajectory_agent(self):
        """Create TrajectoryAgent instance for testing."""
        from agents.trajectory_agent import TrajectoryAgent
        return TrajectoryAgent(config={})
```

- [ ] **Step 2: 添加stairs上升检测测试**

```python
    def test_stairs_up_detection(self, trajectory_agent):
        """Test stairs detection when y increases > 0.5m."""
        current_pos = (0.0, 1.0, 0.0)  # y increased by 1.0
        prev_pos = (0.0, 0.0, 0.0)
        
        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=0.0,
            prev_rot=0.0,
            step_count=10
        )
        
        assert new_node_id is not None
        assert new_node_id in trajectory_agent.topology_graph.nodes
        node = trajectory_agent.topology_graph.nodes[new_node_id]
        assert node.node_type == "stairs"
        assert "上升" in node.semantic_info
```

- [ ] **Step 3: 添加stairs下降检测测试**

```python
    def test_stairs_down_detection(self, trajectory_agent):
        """Test stairs detection when y decreases > 0.5m."""
        current_pos = (0.0, 0.0, 0.0)  # y decreased by 1.0
        prev_pos = (0.0, 1.0, 0.0)
        
        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=0.0,
            prev_rot=0.0,
            step_count=10
        )
        
        assert new_node_id is not None
        node = trajectory_agent.topology_graph.nodes[new_node_id]
        assert node.node_type == "stairs"
        assert "下降" in node.semantic_info
```

- [ ] **Step 4: 添加junction检测测试**

```python
    def test_junction_detection(self, trajectory_agent):
        """Test junction detection when rotation > 28 degrees."""
        import math
        current_pos = (1.0, 0.0, 1.0)
        prev_pos = (0.0, 0.0, 0.0)
        current_rot = math.pi / 2  # 90 degrees
        prev_rot = 0.0
        
        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=current_rot,
            prev_rot=prev_rot,
            step_count=10
        )
        
        assert new_node_id is not None
        node = trajectory_agent.topology_graph.nodes[new_node_id]
        assert node.node_type == "junction"
```

- [ ] **Step 5: 添加rotation边界处理测试**

```python
    def test_rotation_boundary_handling(self, trajectory_agent):
        """Test rotation handles -pi/pi boundary correctly."""
        import math
        current_pos = (1.0, 0.0, 0.0)
        prev_pos = (0.0, 0.0, 0.0)
        # Near boundary: -170 deg to 170 deg = 20 deg actual difference
        current_rot = -170 * math.pi / 180
        prev_rot = 170 * math.pi / 180
        
        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=current_rot,
            prev_rot=prev_rot,
            step_count=10
        )
        
        # Should NOT detect junction (20 deg < 28 deg threshold)
        assert new_node_id is None
```

- [ ] **Step 6: 添加无关键位置变化测试**

```python
    def test_no_key_position(self, trajectory_agent):
        """Test returns None when no key position change."""
        current_pos = (0.5, 0.0, 0.5)  # Small movement, no y change
        prev_pos = (0.0, 0.0, 0.0)
        
        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=0.0,
            prev_rot=0.0,
            step_count=10
        )
        
        assert new_node_id is None
```

- [ ] **Step 7: 运行测试确认失败**

Run: `pytest tests/test_topology_graph.py::TestUpdateTopologyOnly -v`
Expected: 所有测试FAIL（方法尚未实现）

---

## Task 2: 新增update_topology_only方法

**Files:**
- Modify: `agents/trajectory_agent.py`

- [ ] **Step 1: 找到插入位置**

找到`get_spatial_memory_guidance`方法（约第610行），在其后添加新方法。

- [ ] **Step 2: 添加update_topology_only方法**

```python
    def update_topology_only(
        self,
        current_pos: Tuple[float, float, float],
        prev_pos: Tuple[float, float, float],
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

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile agents/trajectory_agent.py`
Expected: 无错误输出

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_topology_graph.py::TestUpdateTopologyOnly -v`
Expected: 所有测试PASS

- [ ] **Step 5: Commit**

```bash
git add agents/trajectory_agent.py tests/test_topology_graph.py
git commit -m "$(cat <<'EOF'
feat(trajectory): add update_topology_only for realtime topology

Add lightweight topology update method for sequence execution phase.
Detects stairs (y change > 0.5m) and junction (rotation > 28 deg).
Room entrance detection delayed to sequence generation.

Tests: 6 unit tests added (stairs/junction/boundary/no-change)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 修改run_vln_experiment调用拓扑更新

**Files:**
- Modify: `run_vln_experiment.py:1456`

- [ ] **Step 1: 找到rotation更新位置**

找到第1456行：`context.rotation = yaw`

- [ ] **Step 2: 在rotation更新后添加拓扑更新代码**

在第1458行（`except Exception as e:`之前）插入：

```python
                        context.rotation = yaw

                        # === 拓扑轻量更新（每3步）===
                        if steps % 3 == 0 and self.trajectory_agent:
                            # 从context获取前一步位置
                            prev_pos_topo = context.trajectory[-2] if len(context.trajectory) > 1 else tuple(pos)
                            
                            # 获取rotation
                            current_rot_topo = yaw
                            prev_rot_topo = getattr(context, '_prev_rotation', current_rot_topo)
                            
                            new_node_id = self.trajectory_agent.update_topology_only(
                                current_pos=tuple(pos),
                                prev_pos=prev_pos_topo,
                                current_rot=current_rot_topo,
                                prev_rot=prev_rot_topo,
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
                            context._prev_rotation = current_rot_topo

                    except Exception as e:
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 4: Commit**

```bash
git add run_vln_experiment.py
git commit -m "$(cat <<'EOF'
feat: call topology update every 3 steps in sequence execution

Add realtime topology update during sequence execution phase.
- Update TopologyGraph every 3 steps
- Detect stairs/junction nodes in real-time
- Update trajectory_output.topology_summary for DecisionAgent

Position: after context.rotation = yaw (line 1456)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 集成测试验证

**Files:**
- Test: 现有测试文件

- [ ] **Step 1: 运行拓扑相关测试**

Run: `pytest tests/test_topology_graph.py -v`
Expected: 所有测试PASS

- [ ] **Step 2: 验证无语法错误**

Run: `python -m py_compile run_vln_experiment.py agents/trajectory_agent.py`
Expected: 无错误输出

- [ ] **Step 3: 确认git状态**

Run: `git status --short`
Expected: 无未提交改动

---

## Self-Review

**1. Spec coverage:**
- ✓ stairs检测（y变化>0.5m）: Task 1 Step 2-3 + Task 2
- ✓ junction检测（转向>28°）: Task 1 Step 4-5 + Task 2
- ✓ rotation边界处理: Task 1 Step 5 + Task 2
- ✓ 无关键位置: Task 1 Step 6 + Task 2
- ✓ 调用时机（每3步）: Task 3
- ✓ prev_rot存储: Task 3

**2. Placeholder scan:**
- 无TBD、TODO
- 所有代码完整

**3. Type consistency:**
- current_pos/prev_pos类型一致：tuple[float, float, float]
- current_rot/prev_rot类型一致：float（弧度）

---

*文档创建时间: 2026-04-25*