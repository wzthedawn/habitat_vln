---
name: CoT动作直接提取
description: 从CoT分析文本直接提取动作建议，跳过二次LLM调用，解决medium任务动作解析失败问题
type: project
---

# CoT动作直接提取设计

## 背景

### 问题
CoT策略为medium难度任务生成分析文本（包含"Step 4 - Action suggestion"），DecisionAgent收到后再次调用LLM生成动作序列。LLM倾向于继续推理模式而非输出简洁JSON，导致：

- 返回长推理文本（5000+字符）
- regex fallback仅提取少量动作关键词（1-2个）
- 低于min_actions阈值（3），触发解析失败

### 日志证据
```
INFO:DecisionAgent:[Decision] LLM response length: 5156 chars
INFO:DecisionAgent:[Decision] Attempting regex fallback extraction...
WARNING:DecisionAgent:[Decision] All parsing failed, generating default actions
ERROR:MultiAgentVLNEvaluator:[SEQUENCE] Parse failed, got 2 actions, need at least 3
```

## 解决方案

### 核心思路
新增专用函数从CoT分析文本直接提取"Action suggestion"部分，转换为动作序列，跳过二次LLM调用。

### 优势
1. 避免二次LLM调用的推理倾向问题
2. 直接利用CoT已有的推理结果
3. 改动集中，不影响CoT策略稳定性
4. fallback机制保留，LLM调用作为后备

## 改动清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `agents/decision_agent.py` | 新增函数（约第1200行后） | `_extract_actions_from_cot_analysis` |
| `agents/decision_agent.py` | `generate_action_sequence`（第212行） | 添加调用逻辑 |

## 实现细节

### 1. 新增函数：`_extract_actions_from_cot_analysis`

```python
def _extract_actions_from_cot_analysis(self, analysis: str) -> List[Tuple[ActionType, int]]:
    """从CoT分析文本中提取动作建议。
    
    CoT分析格式示例：
    Step 4 - Action suggestion: turn left 2-3 times, then move forward 3 times
    
    Args:
        analysis: CoT策略生成的分析文本
        
    Returns:
        List of (ActionType, count) tuples
        返回空列表如果无法提取
    """
    actions = []
    
    # 1. 定位"Action suggestion"或"动作建议"行
    suggestion_patterns = [
        r"Step 4.*Action suggestion[:\s]+(.+)",
        r"动作建议[:\s]+(.+)",
        r"Action suggestion[:\s]+(.+)",
    ]
    
    suggestion_text = None
    for pattern in suggestion_patterns:
        match = re.search(pattern, analysis, re.IGNORECASE)
        if match:
            suggestion_text = match.group(1).strip()
            break
    
    if not suggestion_text:
        return []
    
    # 2. 提取动作
    # 模式：turn left/right N-M times, forward/straight N times
    turn_pattern = r"turn\s+(left|right)\s+(\d+)(?:-\d+)?\s*times"
    forward_pattern = r"(?:move\s+forward|go\s+(?:straight\s+)?forward|forward)\s+(\d+)(?:-\d+)?\s*times"
    
    # 匹配转向
    for match in re.finditer(turn_pattern, suggestion_text, re.IGNORECASE):
        direction = match.group(1).lower()
        count = int(match.group(2))  # 取范围最小值
        action = ActionType.TURN_LEFT if direction == "left" else ActionType.TURN_RIGHT
        actions.append((action, count))
    
    # 匹配前进
    for match in re.finditer(forward_pattern, suggestion_text, re.IGNORECASE):
        count = int(match.group(1))
        actions.append((ActionType.MOVE_FORWARD, count))
    
    # 3. 按出现顺序组合
    # 如果没有找到模式化的动作，尝试简单关键词匹配
    if not actions:
        if "left" in suggestion_text.lower():
            actions.append((ActionType.TURN_LEFT, 2))
        if "right" in suggestion_text.lower():
            actions.append((ActionType.TURN_RIGHT, 2))
        if "forward" in suggestion_text.lower() or "straight" in suggestion_text.lower():
            actions.append((ActionType.MOVE_FORWARD, 3))
    
    self.logger.info(f"[Decision] CoT提取: {len(actions)}个动作组合")
    return actions
```

### 2. 调用逻辑：`generate_action_sequence`

在现有LLM调用之前（约第212行）添加：

```python
# === Skip LLM for medium tasks when CoT analysis has clear actions ===
if level == "medium" and strategy_data.get("analysis"):
    direct_actions = self._extract_actions_from_cot_analysis(strategy_data.get("analysis", ""))
    min_actions = 3  # medium任务最小动作数
    
    if len(direct_actions) >= min_actions:
        self.logger.info(f"[Decision] 直接提取CoT动作: {len(direct_actions)}步，跳过LLM调用")
        reasoning = strategy_data.get("analysis", "")[:200]
        return ActionSequence(
            subtask_id=subtask.id if subtask else 0,
            subtask_description=subtask.description if subtask else "",
            actions=direct_actions,
            estimated_steps=sum(a[1] for a in direct_actions),
            reasoning=f"CoT直接提取: {reasoning[:100]}",
            confidence=0.7,
            abort_conditions={"stuck_for_steps": 5},
            subtask_completed=False,
        )
    else:
        self.logger.info(f"[Decision] CoT提取动作不足({len(direct_actions)}步)，继续LLM调用")

# === 现有LLM调用流程继续 ===
prompt = self._build_sequence_prompt_v2(...)
```

### 3. 支持的输入格式

| 输入 | 解析结果 |
|------|---------|
| `"turn left 2-3 times"` | `[(TURN_LEFT, 2)]` |
| `"turn right 1-2 times, then forward 3 times"` | `[(TURN_RIGHT, 1), (MOVE_FORWARD, 3)]` |
| `"go straight forward 5 times"` | `[(MOVE_FORWARD, 5)]` |
| `"turn left 2 times, forward 2 times, turn right 1 time"` | `[(TURN_LEFT, 2), (MOVE_FORWARD, 2), (TURN_RIGHT, 1)]` |

### 4. Fallback机制

提取失败时：
- 日志：`[Decision] CoT提取动作不足(X步)，继续LLM调用`
- 继续现有LLM调用流程
- 保持原有JSON解析 + regex fallback机制

## 测试验证

### 验证方法
运行评估脚本：
```bash
python scripts/run_emergency_eval.py --exp baseline --episodes 3 --use-remote-llm --llm-server http://localhost:8000
```

### 成功指标
日志中出现：
```
[Decision] 直接提取CoT动作: X步，跳过LLM调用
```

medium任务不再出现：
```
Parse failed, got 2 actions, need at least 3
```

### 预期效果
- medium任务成功率提升
- 减少LLM调用次数（每episode约减少1次）
- 响应速度提升

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| CoT格式变化导致解析失败 | 多种pattern匹配 + fallback到LLM |
| 提取动作数不足 | 阈值检查（>=3），不足时继续LLM |
| 动作顺序错误 | 按文本出现顺序提取 |

## Why

二次LLM调用产生长推理文本，现有解析器无法正确提取动作，导致medium任务失败率高。

## How to apply

在`generate_action_sequence`中，level=="medium"时优先尝试直接提取CoT动作，成功则跳过LLM，失败则fallback。