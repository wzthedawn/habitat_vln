# 策略链拓扑信息集成设计

## 概述

修复ReAct/CoT策略链不使用拓扑信息的问题，让策略链从Agent输出获取数据并包含拓扑信息。

**问题诊断**：
- ReAct/CoT策略prompt**不包含**topology_summary
- 策略链从context原始属性取数据，不从metadata取Agent处理后的信息
- TopologyGraph被构建但未被策略链利用

---

## 一、改动范围

| 文件 | 改动内容 | 改动行数 |
|------|----------|----------|
| `strategies/react.py` | 修改数据获取方法、添加拓扑prompt段落 | ~30行 |
| `strategies/cot.py` | 修改轨迹收集方法、添加拓扑prompt段落 | ~20行 |

---

## 二、ReAct策略改动

### 2.1 修改`_get_perception_info`

**当前问题**：从`context.visual_features`取原始数据

**修复方案**：从`context.metadata["perception_output"]["data"]`取Agent处理后的数据

**注意**：Supernet存储的是`output.to_dict()`，所以metadata中的Agent输出是dict而非AgentOutput对象。访问方式应为`metadata.get("agent_output", {}).get("data", {})`。

```python
def _get_perception_info(self, context: NavContext) -> str:
    """Get perception information from Agent output."""
    parts = []

    # 从metadata获取PerceptionAgent输出（dict格式）
    # 注意：Supernet存储的是output.to_dict()
    perception_output = context.metadata.get("perception_output", {})
    perception_data = perception_output.get("data", {}) if isinstance(perception_output, dict) else {}
    
    if perception_data:
        # 使用Agent处理后的信息
        room_type = perception_data.get("room_type", "unknown")
        if room_type and room_type != "unknown":
            parts.append(f"房间类型: {room_type}")
        
        scene_desc = perception_data.get("scene_description", "")
        if scene_desc:
            parts.append(f"场景描述: {scene_desc[:100]}")
        
        objects = perception_data.get("objects", [])
        if objects:
            obj_names = [o.get("object", o.get("name", str(o))) for o in objects[:5]]
            parts.append(f"可见物体: {', '.join(obj_names)}")
        
        nav_hint = perception_data.get("nav_hint", "")
        if nav_hint:
            parts.append(f"导航提示: {nav_hint[:80]}")
    
    # Fallback: 如果无Agent输出，使用原始visual_features
    if not parts and hasattr(context, 'visual_features'):
        if context.visual_features.scene_description:
            parts.append(f"场景: {context.visual_features.scene_description[:100]}")
    
    return "\n".join(parts) if parts else "无感知信息"
```

### 2.2 修改`_get_trajectory_info`

**新增拓扑信息获取**：

```python
def _get_trajectory_info(self, context: NavContext) -> str:
    """Get trajectory information including topology from Agent output."""
    parts = []
    
    # 从metadata获取TrajectoryAgent输出（dict格式）
    trajectory_output = context.metadata.get("trajectory_output", {})
    trajectory_data = trajectory_output.get("data", {}) if isinstance(trajectory_output, dict) else {}
    
    if trajectory_data:
        parts.append(f"已走距离: {trajectory_data.get('distance_traveled', 0):.1f}m")
        parts.append(f"进度: {trajectory_data.get('progress_percentage', 0):.1f}%")
    
    # 原有轨迹计算
    if len(context.trajectory) >= 2:
        start = context.trajectory[0]
        current = context.trajectory[-1]
        dist = ((current[0] - start[0])**2 + (current[2] - start[2])**2)**0.5
        parts.append(f"总距离: {dist:.1f}m")
    
    return "\n".join(parts) if parts else f"步数: {context.step_count}"
```

**注意**：拓扑信息单独通过`_get_topology_info`方法获取，不在此方法中处理。

### 2.3 修改`_build_react_prompt`

**添加拓扑信息段落**：

```python
def _build_react_prompt(self, context: NavContext, previous_steps: List[Dict]) -> str:
    """Build the ReAct prompt for LLM."""
    instruction = context.instruction
    
    # 获取子任务
    current_subtask = context.get_current_subtask()
    subtask_desc = current_subtask.description if current_subtask else "无"
    
    # 使用修改后的数据获取方法
    perception_info = self._get_perception_info(context)
    trajectory_info = self._get_trajectory_info(context)
    
    # 新增：获取拓扑信息
    topology_info = self._get_topology_info(context)
    
    # 获取最近动作
    recent_actions = []
    if context.action_history:
        for a in context.action_history[-5:]:
            recent_actions.append(a.action_type.name)
    
    prompt = f"""你是一个导航推理专家。请使用ReAct模式进行推理和决策。

## 导航指令
{instruction}

## 当前子任务
{subtask_desc}

## 当前状态
- 步数: {context.step_count}
- 位置: ({context.position[0]:.1f}, {context.position[1]:.1f}, {context.position[2]:.1f})
- 房间类型: {context.room_type}

## 感知信息
{perception_info}

## 轨迹信息
{trajectory_info}

## 拓扑信息
{topology_info}

## 最近动作历史
{', '.join(recent_actions) if recent_actions else '无'}

## 可选动作
- forward: 向前移动
- turn_left: 向左转
- turn_right: 向右转
- stop: 停止导航

## 要求
请按ReAct模式思考:
1. Thought: 观察当前状态，分析应该做什么
2. Action: 选择一个动作执行

## 输出格式
请严格按照以下JSON格式输出:
```json
{{
  "thought": "对当前状态的观察和思考...",
  "action": "forward/turn_left/turn_right/stop",
  "confidence": 0.0-1.0,
  "reasoning": "选择该动作的简要理由"
}}
```
"""
    return prompt
```

### 2.4 新增`_get_topology_info`方法

```python
def _get_topology_info(self, context: NavContext) -> str:
    """Get topology information from TrajectoryAgent output."""
    # 从metadata获取TrajectoryAgent输出（dict格式）
    trajectory_output = context.metadata.get("trajectory_output", {})
    trajectory_data = trajectory_output.get("data", {}) if isinstance(trajectory_output, dict) else {}
    topology_summary = trajectory_data.get("topology_summary", {})
    
    if not topology_summary:
        return "无拓扑信息"
    
    total_nodes = topology_summary.get("total_nodes", 0)
    key_nodes = topology_summary.get("key_nodes", [])
    current_node = topology_summary.get("current_node", "unknown")
    visited_rooms = topology_summary.get("visited_rooms", [])
    stuck_regions = topology_summary.get("stuck_regions", [])
    path_to_goal = topology_summary.get("path_to_goal", [])
    
    key_nodes_str = ", ".join([n.get("type", str(n)) for n in key_nodes[:5]]) if key_nodes else "无"
    visited_rooms_str = ", ".join(visited_rooms[:5]) if visited_rooms else "无"
    path_str = " -> ".join(path_to_goal[:5]) if path_to_goal else "未知"
    
    return f"""- 总节点数: {total_nodes}
- 关键节点: {key_nodes_str}
- 当前位置节点: {current_node}
- 已访问房间: {visited_rooms_str}
- 目标路径: {path_str}
- 卡住区域: {len(stuck_regions)}个

**拓扑提示**: 使用已访问房间信息避免重复探索，参考关键节点做路径规划。"""
```

---

## 三、CoT策略改动

### 3.1 修改`_collect_trajectory`

**新增topology_summary字段**：

```python
def _collect_trajectory(self, context: NavContext) -> Dict[str, Any]:
    """Collect trajectory information including topology from context."""
    # 从metadata获取TrajectoryAgent输出（dict格式）
    trajectory_output = context.metadata.get("trajectory_output", {})
    trajectory_data = trajectory_output.get("data", {}) if isinstance(trajectory_output, dict) else {}
    
    # 获取历史摘要
    history_summary = "无历史"
    if self.trajectory_agent:
        try:
            current_pos = tuple(context.position) if hasattr(context, "position") else (0, 0, 0)
            history_summary = self.trajectory_agent.get_history_summary(current_pos)
        except Exception as e:
            self.logger.warning(f"[CoT] Failed to get history: {e}")
    
    # 新增：获取topology_summary
    topology_summary = trajectory_data.get("topology_summary", {})
    
    return {
        "distance_traveled": trajectory_data.get("distance_traveled", 0),
        "heading": trajectory_data.get("heading", "unknown"),
        "progress_percentage": trajectory_data.get("progress_percentage", 0),
        "stuck_counter": getattr(context, "stuck_counter", 0),
        "step_count": context.step_count,
        "position": context.position,
        "rotation": context.rotation,
        "history_summary": history_summary,
        "topology_summary": topology_summary,  # 新增
    }
```

### 3.2 修改`_build_analysis_prompt`

**添加拓扑信息段落**：

在prompt中添加：

```python
## 拓扑信息
- 总节点数: {trajectory_info['topology_summary'].get('total_nodes', 0)}
- 关键节点: {[n.get('type', str(n)) for n in trajectory_info['topology_summary'].get('key_nodes', [])[:5]]}
- 已访问房间: {trajectory_info['topology_summary'].get('visited_rooms', [])[:5]}
- 目标路径: {trajectory_info['topology_summary'].get('path_to_goal', [])[:5]}
- 卡住区域: {len(trajectory_info['topology_summary'].get('stuck_regions', []))}个

**拓扑提示**: 使用拓扑信息辅助路径规划，避免重复探索已访问区域。
```

---

## 四、验证方法

```bash
# 1. 语法验证
python -m py_compile strategies/react.py
python -m py_compile strategies/cot.py

# 2. 测试验证（如有测试）
pytest tests/test_strategies.py -v

# 3. 集成测试
python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm --llm-server http://localhost:8000

# 4. 检查日志中拓扑信息是否出现在策略prompt中
```

---

## 五、预期效果

| 指标 | 改动前 | 改动后 |
|------|--------|--------|
| ReAct拓扑信息 | 无 | ✓ 包含 |
| CoT拓扑信息 | 无 | ✓ 包含 |
| 数据来源 | 原始visual_features | Agent处理后metadata |
| 策略决策质量 | 低（无拓扑） | 提升（有拓扑） |

---

## 六、设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 改动方式 | 修改现有方法 | 改动最小，复用现有代码 |
| 数据来源 | metadata Agent输出 | 使用Agent智能处理后的信息 |
| topology_info格式 | 类似DecisionAgent | 保持一致性 |

---

*文档创建时间: 2026-04-21*