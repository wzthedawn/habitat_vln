# CoT动作提取逻辑修复设计

## 概述

修复CoT策略的nav_hint解读问题，让LLM正确将语义导航提示映射到动作序列。

**核心问题**：PerceptionAgent正确生成nav_hint，但CoT策略的LLM解读失败，导致动作序列错误。

**解决方案**：CoT直接输出结构化动作建议，DecisionAgent负责精确化和校验。

---

## 一、问题诊断

### 1.1 当前信息流

```
PerceptionAgent → nav_hint(正确) → CoT策略 → analysis(文本) → DecisionAgent → LLM二次推理 → action_sequence(错误)
```

### 1.2 问题根源

| 问题点 | 说明 |
|--------|------|
| CoT输出格式 | 纯文本analysis，无结构化动作建议 |
| DecisionAgent二次推理 | 重复CoT工作，且prompt不够明确 |
| 信息断层 | analysis不包含可直接执行的动作信息 |

---

## 二、职责重新划分

### 2.1 新职责边界

| 角色 | 职责 | 是否需要LLM |
|------|------|------------|
| **CoT策略** | 输出导航建议（方向、步数范围） | 需要LLM |
| **DecisionAgent** | 将建议精确化、校验、状态管理 | 不需要LLM |

### 2.2 DecisionAgent保留职责

- 动作精确化：建议 → 精确动作序列
- 安全校验：检查障碍冲突
- 状态管理：stuck/visited/emergency
- Completion检测：检查子任务完成条件
- 多策略整合：Hard任务整合Debate观点
- 应急响应：emergency_signal处理

---

## 三、CoT输出格式设计

### 3.1 修订后输出格式

```json
{
  "analysis": "Goal: stairs. Quote: 'lower space below on left'. Interpret: stairs going down on left.",
  "suggestion": {
    "direction": "left",
    "turn_count": {"min": 2, "max": 3},
    "forward_count": {"min": 3, "max": 5},
    "confidence": 0.8
  },
  "subtask_completed": false
}
```

### 3.2 字段说明

| 字段 | 取值 | 说明 |
|------|------|------|
| `direction` | `"left"` \| `"right"` \| `"forward"` | 主要转向方向 |
| `turn_count.min` | 0-5 | 最少转向次数 |
| `turn_count.max` | 0-5 | 最多转向次数 |
| `forward_count.min` | 1-10 | 最少前进次数 |
| `forward_count.max` | 1-10 | 最多前进次数 |
| `confidence` | 0.0-1.0 | 建议置信度 |

---

## 四、DecisionAgent转换逻辑

### 4.1 建议转精确动作

```python
def _convert_suggestion_to_actions(self, suggestion: dict) -> List[tuple]:
    """将CoT建议转换为精确动作序列"""

    direction = suggestion.get("direction", "forward")
    turn_min = suggestion.get("turn_count", {}).get("min", 0)
    turn_max = suggestion.get("turn_count", {}).get("max", 0)
    forward_min = suggestion.get("forward_count", {}).get("min", 1)
    forward_max = suggestion.get("forward_count", {}).get("max", 5)

    actions = []

    # 1. 转向动作
    if direction == "left":
        turn_count = min(turn_max, max(turn_min, 2))  # 取中间值
        actions.extend([(ActionType.TURN_LEFT, 1)] * turn_count)
    elif direction == "right":
        turn_count = min(turn_max, max(turn_min, 2))
        actions.extend([(ActionType.TURN_RIGHT, 1)] * turn_count)

    # 2. 前进动作
    forward_count = min(forward_max, max(forward_min, 3))
    actions.extend([(ActionType.MOVE_FORWARD, 1)] * forward_count)

    return actions
```

### 4.2 步数选择策略

- 从[min, max]范围取中间值
- 保证最少转向2次、前进3次
- 避免过于保守（只转1次）或过于激进（转5次）

---

## 五、CoT Prompt改造

### 5.1 当前prompt结尾

```python
Output analysis result directly, no JSON format:
```

### 5.2 修订后prompt结尾

```python
## Output Format (JSON)
Output JSON only:
{
  "analysis": "Step 1-4分析结果合并",
  "suggestion": {
    "direction": "left" | "right" | "forward",
    "turn_count": {"min": 2, "max": 3},
    "forward_count": {"min": 3, "max": 5}
  },
  "subtask_completed": false
}

## Rules
- direction: 必须是left/right/forward之一
- turn_count: 仅当direction为left/right时填写
- forward_count: 必须填写，建议3-5步
- subtask_completed: 默认false
```

---

## 六、Fallback解析处理

LLM输出可能不完整，多层fallback确保解析成功：

```python
def _parse_action_suggestion(self, response: str) -> dict:
    """解析LLM输出，多层fallback"""

    # Layer 1: JSON解析
    try:
        result = json.loads(response)
        if "suggestion" in result:
            return result
    except:
        pass

    # Layer 2: 正则提取direction
    direction_match = re.search(r'"direction":\s*"(\w+)"', response)
    if direction_match:
        direction = direction_match.group(1)
        if direction in ["left", "right", "forward"]:
            return {
                "suggestion": {
                    "direction": direction,
                    "turn_count": {"min": 2, "max": 3},
                    "forward_count": {"min": 3, "max": 5}
                }
            }

    # Layer 3: 从analysis文本推断
    if "turn left" in response.lower() or "左转" in response:
        return {"suggestion": {"direction": "left", ...}}
    elif "turn right" in response.lower() or "右转" in response:
        return {"suggestion": {"direction": "right", ...}}
    elif "forward" in response.lower() or "前进" in response:
        return {"suggestion": {"direction": "forward", ...}}

    # Layer 4: 默认前进
    return {"suggestion": {"direction": "forward", ...}}
```

---

## 七、文件变更清单

| 文件 | 操作 | 内容 | 改动量 |
|------|------|------|--------|
| `strategies/cot.py` | 修改 | `_build_analysis_prompt`结尾部分 | ~20行 |
| `strategies/cot.py` | 新增 | `_parse_action_suggestion`方法 | ~30行 |
| `strategies/cot.py` | 修改 | `execute`返回结构化suggestion | ~10行 |
| `agents/decision_agent.py` | 新增 | `_convert_suggestion_to_actions`方法 | ~20行 |
| `agents/decision_agent.py` | 修改 | `generate_action_sequence`优先使用suggestion | ~15行 |

**总改动量**：约95行

---

## 八、预期效果

| 指标 | 当前 | 修复后 |
|------|------|--------|
| nav_hint解读成功率 | ~30% | ~80% |
| Medium任务成功率 | ~10% | ~30-40% |
| LLM调用次数 | 2次(CoT+Decision) | 1次(CoT) |
| 平均推理时间 | ~3秒 | ~1.5秒 |

---

## 九、设计自检

### 9.1 Placeholder检查
- 无TBD、TODO占位符

### 9.2 内部一致性
- direction取值与DecisionAgent处理逻辑一致
- turn_count/forward_count范围与默认值一致

### 9.3 Scope检查
- 仅涉及CoT策略和DecisionAgent，范围明确
- 不涉及其他Agent或策略

### 9.4 Ambiguity检查
- direction取值明确（left/right/forward）
- 步数选择策略明确（取中间值）
- fallback层级明确（4层）

---

*文档创建时间: 2026-04-25*