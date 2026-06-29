# Prompt压缩设计

## 概述

通过分层压缩策略减少DecisionAgent prompt冗余，降低token消耗，加快LLM响应速度。

**核心目标**：
- Easy级别：精简到~250 tokens（当前~600）
- Medium级别：保留分析结果，其他压缩到~400 tokens（当前~800）
- Hard级别：智能摘要，动态调整~450-600 tokens（当前~1000）
- 输出格式：简化说明，~20 tokens（当前~60）

**预期效果**：
- Token消耗降低40-50%
- LLM响应速度提升20-30%
- 决策质量不受影响（保留关键信息）

---

## 一、压缩策略概览

| 级别 | 当前Token | 目标Token | 核心策略 |
|------|-----------|-----------|----------|
| Easy | ~600 | ~250 | 只保留核心决策信息 |
| Medium | ~800 | ~400 | 保留CoT分析，压缩其他 |
| Hard | ~1000 | ~450-600 | 智能摘要，动态调整 |
| 格式说明 | ~60 | ~20 | 只给示例，去掉完整说明 |

---

## 二、Easy级别压缩（~250 tokens）

### 2.1 保留信息

```python
# 核心决策信息
prompt = f"""导航决策。生成{seq_len}步动作序列。

## 任务
{subtask.description}

## 状态
- 目标距离: {distance_to_goal:.1f}m
- 目标方向: {direction_hint} ({angle_to_goal:.0f}°)
- 已走: {dist_traveled:.1f}m

## 环境
- 房间: {room_type}
- 可见: {objects if objects else "无"}
- 障碍: {f"前方{min_dist:.1f}m" if blocked else "无"}

## 规则
1. 输出{seq_len}步动作
2. 方向偏差>15°时需转向
3. 距离<3m时可停止

输出JSON: {"reasoning":"...","subtask_completed":false,"actions":[...]}"""
```

### 2.2 去掉信息

| 去掉项 | 原因 | Token节省 |
|--------|------|-----------|
| 指令语义分析 | Easy任务简单，指令直接 | ~50 |
| 拓扑信息 | 无需历史记忆 | ~80 |
| 行动历史 | 场景简单 | ~60 |
| 楼梯指导 | Easy任务少涉及楼梯 | ~100 |
| 状态变化详情 | 只需当前状态 | ~40 |
| 中文标签 | 用符号代替 | ~30 |

### 2.3 实现位置

修改 `_build_simple_prompt_direct_v2()` 方法，直接重构。

---

## 三、Medium级别压缩（~400 tokens）

### 3.1 保留信息（完整）

- **CoT分析结果**：这是核心价值，完整保留
- **拓扑信息**（简化）：当前节点 + 路径（去掉历史详情）
- **环境分析**：房间 + 关键物体

### 3.2 压缩信息

```python
# 环境分析（简化）
环境: {room_type}, 可见{objects[:3]}, {scene_desc[:50]}

# 拓扑信息（简化）
拓扑: 当前{current_node}, 路径{path_to_goal[:3]}, 已访问{visited_rooms[:3]}

# 状态变化（精简）
状态: 位置({curr_pos}), 移动{h_dist:.1f}m, 目标{dist_to_goal:.1f}m

# 导航状态（精简）
距离: {distance_to_goal:.1f}m ({distance_trend}), 方向: {direction_hint}
```

### 3.3 去掉信息

| 去掉项 | 原因 | Token节省 |
|--------|------|-----------|
| 楼梯指导（完整版） | 简化为单行提示 | ~80 |
| 行动历史详情 | 保留summary，去掉详情 | ~60 |
| 障碍物半径等细节 | 只需是否阻塞 | ~30 |
| 中文标签 | 用符号代替 | ~40 |
| 格式完整说明 | 只给示例 | ~40 |

### 3.4 实现位置

修改 `_build_medium_prompt_with_analysis_v2()` 方法。

---

## 四、Hard级别智能摘要（~450-600 tokens）

### 4.1 动态摘要逻辑

```python
def _build_hard_prompt_compressed(self, ...):
    """Hard级别智能摘要"""
    
    # === 动态调整信息量 ===
    
    # 1. Debate观点摘要策略
    opinions = strategy_data.get("opinions", {})
    consensus = strategy_data.get("consensus", {})
    
    if consensus.get("agreement_level", 0) > 0.8:
        # 观点一致 → 只保留共识
        opinion_section = f"共识: {consensus.get('agreed_action', 'forward')}"
        opinion_token_cost = ~50
    else:
        # 观点分歧 → 保留各方观点（精简）
        opinion_section = self._summarize_opinions(opinions)
        opinion_token_cost = ~150
    
    # 2. 拓扑信息摘要策略
    topology_summary = trajectory_output.get("topology_summary", {})
    stuck_regions = topology_summary.get("stuck_regions", [])
    
    if len(stuck_regions) > 0:
        # 有卡住历史 → 保留拓扑详情
        topology_section = self._format_topology_section(topology_summary)
        topology_token_cost = ~80
    else:
        # 无历史 → 拓扑简化
        topology_section = f"拓扑: 节点{total_nodes}, 已访问{visited_rooms[:3]}"
        topology_token_cost = ~30
    
    # 3. 历史信息摘要策略
    if action_history_summary and len(action_history_summary) > 100:
        # 历史长 → 只取关键摘要
        history_section = f"历史: {action_history_summary[:80]}..."
        history_token_cost = ~40
    else:
        history_section = action_history_summary or "首次探索"
        history_token_cost = ~20
    
    # 总Token控制
    total_estimated = opinion_token_cost + topology_token_cost + history_token_cost + base_cost
    assert total_estimated <= 600
```

### 4.2 观点摘要方法

```python
def _summarize_opinions(self, opinions: Dict) -> str:
    """精简多Agent观点"""
    lines = []
    for agent, opinion in opinions.items():
        action = opinion.get("suggested_action", "unknown")
        confidence = opinion.get("confidence", 0.5)
        reason = opinion.get("reasoning", "")[:50]  # 只取前50字符
        lines.append(f"{agent}: {action}({confidence:.0%}) - {reason}")
    return "\n".join(lines[:3])  # 只显示3个Agent
```

### 4.3 实现位置

修改 `_build_hard_prompt()` 方法，新增 `_summarize_opinions()` 辅助方法。

---

## 五、输出格式压缩（~20 tokens）

### 5.1 当前格式（~60 tokens）

```python
## Output Format (JSON)
{"reasoning":"brief reasoning in 1-2 sentences","subtask_completed":false,"actions":[{"action":"forward"},{"action":"turn_right"},{"action":"forward"}]}

IMPORTANT: Keep reasoning brief (1-2 sentences). Output JSON directly:
```

### 5.2 简化格式（~20 tokens）

```python
输出JSON: {"reasoning":"简述","subtask_completed":false,"actions":[{"action":"forward"}...]}
```

### 5.3 实现位置

修改所有 `_build_xxx_prompt` 方法的末尾格式部分。

---

## 六、标签简化

### 6.1 当前标签（中文+英文）

```python
## 拓扑信息 (Topology Memory)
## 环境分析 (PerceptionAgent)
## 导航状态 (Navigation State)
```

### 6.2 简化标签（符号）

```python
## 拓扑
## 环境
## 状态
```

节省约30 tokens/级别。

---

## 七、改动文件清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `agents/decision_agent.py` | `_build_simple_prompt_direct_v2` | 重构为精简版 |
| `agents/decision_agent.py` | `_build_medium_prompt_with_analysis_v2` | 保留分析，压缩其他 |
| `agents/decision_agent.py` | `_build_hard_prompt` | 新增智能摘要逻辑 |
| `agents/decision_agent.py` | `_summarize_opinions` | 新增辅助方法 |
| `agents/decision_agent.py` | `_format_topology_section` | 简化标签 |

---

## 八、预期效果

| 指标 | 改动前 | 改动后 |
|------|--------|--------|
| Easy prompt token | ~600 | ~250 |
| Medium prompt token | ~800 | ~400 |
| Hard prompt token | ~1000 | ~450-600 |
| 总Token消耗 | 高 | 降低40-50% |
| LLM响应时间 | 基线 | 提升20-30% |
| 决策质量 | 基线 | 不受影响 |

---

## 九、设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 压缩策略 | 分层压缩（C） | 已有级别区分，针对性压缩 |
| Easy目标 | 精简版（~250, B） | 简单场景无需复杂信息 |
| Medium目标 | 保留分析（A） | CoT分析是核心价值 |
| Hard目标 | 智能摘要（C） | 复杂场景需求动态变化 |
| 格式说明 | 简化示例（B） | 示例足够，去掉完整说明 |
| 实现方式 | 修改现有方法（A） | 改动集中，便于测试 |

---

## 十、验证方法

```bash
# 1. Token统计验证
python -c "
from agents.decision_agent import DecisionAgent
# 构造测试context
prompt_easy = agent._build_simple_prompt_direct_v2(...)
print(f'Easy prompt: {len(prompt_easy)} chars')

# 2. 功能验证
pytest tests/test_decision_agent.py -v

# 3. 集成验证
python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm
```

---

*文档创建时间: 2026-04-22*