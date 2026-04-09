# 应急指令解析优化设计方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 修复应急指令解析失败导致子任务破损的问题

**架构：** 关键词检测 + LLM智能分解（改进prompt）

**技术栈：** Python, regex关键词检测, LLM prompt优化

---

## 问题分析

### 当前问题

应急指令有多种格式，但当前正则只匹配一种：

| 格式 | 示例 | 当前匹配状态 |
|------|------|-------------|
| 标准模板 | `Go straight, the path is blocked, ...to reach...` | ✅ 匹配 |
| 变体1 | `Walk right, avoid the obstacle ahead, and reach your target.` | ❌ 失败 |
| 变体2 | `Emergency: Go straight, route blocked, quickly turn...` | ❌ 失败 |
| 变体3 | `Move forward, suddenly blocked, quickly navigate around...` | ❌ 失败 |
| 变体4 | `Urgent: Navigate to the door, obstacle detected...` | ❌ 失败 |

### 后果

匹配失败的指令回退到LLM分解，LLM输出不稳定JSON，最终按逗号分割产生破损子任务如：
- "Continue ahead suddenly blocked..."
- "Proceed right obstacle appeared..."

---

## 解决方案设计

### 方案：关键词检测 + LLM智能分解

**核心思路：** 放弃精确正则匹配，改用关键词检测识别应急指令，然后用改进的LLM prompt进行智能分解。

### 改动点

#### 1. 新增 `_is_emergency_instruction` 方法

在 `agents/instruction_agent.py` 中添加关键词检测方法：

```python
EMERGENCY_KEYWORDS = [
    "blocked", "obstacle", "emergency", "urgent", 
    "suddenly", "path is blocked", "route blocked",
    "obstacle detected", "avoid the obstacle"
]

def _is_emergency_instruction(self, instruction: str) -> bool:
    """检测是否为应急指令"""
    instruction_lower = instruction.lower()
    return any(kw in instruction_lower for kw in self.EMERGENCY_KEYWORDS)
```

#### 2. 修改 `_split_instruction` 方法

在方法开头添加应急指令检测和处理：

```python
def _split_instruction(self, text: str) -> List[str]:
    """Split instruction into subtask segments."""
    
    # 首先检测是否为应急指令
    if self._is_emergency_instruction(text):
        self.logger.info(f"[Instruction] Detected emergency instruction: {text[:50]}...")
        # 尝试LLM分解
        llm_subtasks = self._emergency_decompose_with_llm(text)
        if llm_subtasks:
            return llm_subtasks
    
    # 原有逻辑...
```

#### 3. 新增 `_emergency_decompose_with_llm` 方法

专门用于应急指令的LLM分解，使用改进的prompt：

```python
def _emergency_decompose_with_llm(self, instruction: str) -> Optional[List[str]]:
    """使用LLM分解应急指令"""
    
    prompt = f"""/no_think
You are a navigation instruction analyzer. Break down this emergency instruction into 2-3 clear subtasks.

## Instruction
{instruction}

## Rules
1. Each subtask must be a complete sentence (verb + direction + object)
2. Typical emergency pattern: detect obstacle → find alternative route → continue to goal
3. Output strict JSON only, no other text

## Output Format
{{"subtasks":["Complete sentence 1","Complete sentence 2","Complete sentence 3"]}}

## Example
Input: "Move forward, suddenly blocked, quickly navigate around the obstacle."
Output: {{"subtasks":["First attempt to move forward along the planned path","Obstacle detected on the path, find an alternative route around it","Continue navigating toward the goal destination"]}}

Output JSON only:"""
    
    response = self._model_manager.generate(
        "qwen-9b-instruction",
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.1,
    )
    
    if response:
        # 解析JSON
        try:
            import json
            data = json.loads(response)
            subtasks = data.get("subtasks", [])
            if subtasks and len(subtasks) >= 2:
                self.logger.info(f"[Instruction] Emergency decomposition: {len(subtasks)} subtasks")
                return subtasks
        except json.JSONDecodeError:
            pass
    
    return None
```

---

## 文件修改清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `agents/instruction_agent.py` | 约第26行 | 添加 `EMERGENCY_KEYWORDS` 常量 |
| `agents/instruction_agent.py` | 新增方法 | `_is_emergency_instruction` |
| `agents/instruction_agent.py` | 新增方法 | `_emergency_decompose_with_llm` |
| `agents/instruction_agent.py` | 约第482行 | 修改 `_split_instruction` 开头添加检测 |

---

## 验收标准

| 指标 | 修改前 | 修改后预期 |
|------|--------|-----------|
| 应急指令匹配率 | ~20%（仅标准模板） | ~95%（关键词检测覆盖） |
| 子任务描述 | 破损碎片化 | 完整句子 |
| LLM分解成功率 | 不稳定 | >80% |

---

## 测试用例

```python
test_cases = [
    "Go straight, the path is blocked, go back and try different path to reach the table.",
    "Walk right, avoid the obstacle ahead, and reach your target.",
    "Emergency: Go straight, route blocked, quickly turn and find exit.",
    "Move forward, suddenly blocked, quickly navigate around the obstacle.",
    "Urgent: Navigate to the door, obstacle detected, turn right and look for alternative route."
]

# 预期：所有测试用例都被识别为应急指令并正确分解
```

---

## 预期效果

修改后，应急指令将产生类似以下的完整子任务：

```
Input: "Move forward, suddenly blocked, quickly navigate around the obstacle."

Output:
1. "First attempt to move forward along the planned path"
2. "Obstacle detected on the path, find an alternative route around it"
3. "Continue navigating toward the goal destination"
```

而非当前的破损碎片：
```
1. "Continue ahead suddenly blocked"
2. "quickly navigate around the obstacle"
```