# CoT动作直接提取 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从CoT分析文本直接提取动作建议，跳过二次LLM调用，解决medium任务动作解析失败问题

**Architecture:** 在DecisionAgent中新增 `_extract_actions_from_cot_analysis` 函数，用regex从CoT分析文本提取"Action suggestion"部分，转换为动作序列。在 `generate_action_sequence` 中添加调用逻辑，提取成功时直接返回，失败时fallback到现有LLM流程。

**Tech Stack:** Python, regex, ActionType

---

## File Structure

| 文件 | 改动 |
|------|------|
| `agents/decision_agent.py` | 新增 `_extract_actions_from_cot_analysis` 函数（约第1200行后） |
| `agents/decision_agent.py` | 在 `generate_action_sequence` 中添加调用逻辑（约第212行） |

---

### Task 1: 新增 `_extract_actions_from_cot_analysis` 函数

**Files:**
- Modify: `agents/decision_agent.py:1200` (在 `_regex_fallback_extract` 函数后)

- [ ] **Step 1: 在 `_regex_fallback_extract` 函数后添加新函数**

在 `agents/decision_agent.py` 第1200行左右（`_regex_fallback_extract` 函数之后），添加：

```python
def _extract_actions_from_cot_analysis(self, analysis: str) -> List[tuple]:
    """从CoT分析文本中提取动作建议。
    
    CoT分析格式示例：
    Step 4 - Action suggestion: turn left 2-3 times, then move forward 3 times
    
    Args:
        analysis: CoT策略生成的分析文本
        
    Returns:
        List of (ActionType, count) tuples
        返回空列表如果无法提取
    """
    if not analysis:
        return []
    
    actions = []
    
    # 1. 定位"Action suggestion"行
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
            self.logger.info(f"[Decision] 找到Action suggestion: {suggestion_text[:80]}")
            break
    
    if not suggestion_text:
        self.logger.info("[Decision] 未找到Action suggestion行")
        return []
    
    # 2. 提取动作 - 使用finditer保持顺序
    # 转向模式：turn left/right N-M times
    turn_pattern = r"turn\s+(left|right)\s+(\d+)(?:-\d+)?\s*times?"
    # 前进模式：forward/straight N times
    forward_pattern = r"(?:move\s+forward|go\s+(?:straight\s+)?forward|forward)\s+(\d+)(?:-\d+)?\s*times?"
    
    # 收集所有匹配及其位置，保持顺序
    matches_with_pos = []
    
    for match in re.finditer(turn_pattern, suggestion_text, re.IGNORECASE):
        direction = match.group(1).lower()
        count = int(match.group(2))
        action = ActionType.TURN_LEFT if direction == "left" else ActionType.TURN_RIGHT
        matches_with_pos.append((match.start(), action, count))
    
    for match in re.finditer(forward_pattern, suggestion_text, re.IGNORECASE):
        count = int(match.group(1))
        matches_with_pos.append((match.start(), ActionType.MOVE_FORWARD, count))
    
    # 按位置排序，保持文本中的顺序
    matches_with_pos.sort(key=lambda x: x[0])
    actions = [(m[1], m[2]) for m in matches_with_pos]
    
    # 3. 如果没有找到模式化的动作，尝试简单关键词匹配
    if not actions:
        simple_patterns = [
            ("left", ActionType.TURN_LEFT, 2),
            ("right", ActionType.TURN_RIGHT, 2),
            ("forward", ActionType.MOVE_FORWARD, 3),
            ("straight", ActionType.MOVE_FORWARD, 3),
        ]
        for keyword, action_type, default_count in simple_patterns:
            if keyword in suggestion_text.lower():
                actions.append((action_type, default_count))
    
    self.logger.info(f"[Decision] CoT提取: {len(actions)}个动作组合")
    return actions
```

- [ ] **Step 2: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

---

### Task 2: 在 `generate_action_sequence` 中添加调用逻辑

**Files:**
- Modify: `agents/decision_agent.py:212` (在LLM调用之前)

- [ ] **Step 1: 读取现有代码结构**

先读取 `generate_action_sequence` 函数中第205-230行代码，确认插入位置：

Run: `head -n 250 agents/decision_agent.py | tail -n 50`

- [ ] **Step 2: 在LLM调用之前添加CoT直接提取逻辑**

在第212行左右（`prompt = self._build_sequence_prompt_v2(...)` 之前），添加：

```python
        # === Skip LLM for medium tasks when CoT analysis has clear actions ===
        if level == "medium" and strategy_data and strategy_data.get("analysis"):
            direct_actions = self._extract_actions_from_cot_analysis(strategy_data.get("analysis", ""))
            min_actions_for_medium = 3
            
            if len(direct_actions) >= min_actions_for_medium:
                self.logger.info(f"[Decision] 直接提取CoT动作: {len(direct_actions)}步，跳过LLM调用")
                reasoning_preview = strategy_data.get("analysis", "")[:200]
                
                from core.action import ActionSequence
                return ActionSequence(
                    subtask_id=subtask.id if subtask else 0,
                    subtask_description=subtask.description if subtask else "",
                    actions=direct_actions,
                    estimated_steps=sum(a[1] for a in direct_actions),
                    reasoning=f"CoT直接提取: {reasoning_preview[:100]}",
                    confidence=0.7,
                    abort_conditions={"stuck_for_steps": 5},
                    subtask_completed=False,
                )
            else:
                self.logger.info(f"[Decision] CoT提取动作不足({len(direct_actions)}步)，继续LLM调用")

        # === 现有LLM调用流程 ===
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

---

### Task 3: 运行测试验证

**Files:**
- Test: 运行评估脚本验证

- [ ] **Step 1: 运行单episode测试**

Run: `python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm --llm-server http://localhost:8000 2>&1 | tee /tmp/cot_extraction_test.log`

Expected: 测试启动成功

- [ ] **Step 2: 检查日志确认功能生效**

等待测试运行一段时间（约60秒），检查日志：

Run: `grep -E "直接提取CoT动作|CoT提取动作不足|找到Action suggestion|未找到Action suggestion" /tmp/cot_extraction_test.log | head -10`

Expected: 
- 如果出现 `[Decision] 直接提取CoT动作: X步，跳过LLM调用` → 功能成功
- 如果出现 `[Decision] CoT提取动作不足(X步)，继续LLM调用` → fallback正常工作
- 如果没有相关日志 → 可能medium任务没有触发，检查其他episode

- [ ] **Step 3: 确认无Parse failed错误**

Run: `grep "Parse failed" /tmp/cot_extraction_test.log`

Expected: 无输出（或输出比之前明显减少）

---

### Task 4: 提交代码

- [ ] **Step 1: 查看改动**

Run: `git diff agents/decision_agent.py | head -100`

Expected: 显示新增函数和调用逻辑

- [ ] **Step 2: 提交**

Run: `git add agents/decision_agent.py && git commit -m "feat: add _extract_actions_from_cot_analysis for direct CoT action extraction

- New function: _extract_actions_from_cot_analysis extracts actions from CoT analysis text
- Skip secondary LLM call when CoT provides clear action suggestions (>=3 steps)
- Fallback to existing LLM flow when extraction fails
- Fixes medium task parsing failures caused by long LLM reasoning responses"`

---

## Spec Coverage Check

| Spec要求 | Task覆盖 |
|---------|---------|
| 新增 `_extract_actions_from_cot_analysis` 函数 | Task 1 ✅ |
| 支持多种输入格式 | Task 1 regex patterns ✅ |
| 在 `generate_action_sequence` 添加调用逻辑 | Task 2 ✅ |
| Fallback机制 | Task 2 条件判断 ✅ |
| 测试验证 | Task 3 ✅ |

---

## Self-Review

1. **Placeholder scan**: 无TBD/TODO，所有代码完整 ✅
2. **Type consistency**: 返回 `List[tuple]` (ActionType, count)，与 ActionSequence.actions 一致 ✅
3. **Spec coverage**: 所有要求已覆盖 ✅