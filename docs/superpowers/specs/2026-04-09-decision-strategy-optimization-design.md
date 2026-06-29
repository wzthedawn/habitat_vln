# 导航决策策略优化设计方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将导航成功率从10%提升到45-50%（接近MSNav Zero-Shot水平）

**架构：** 修复已知错误 + 增强决策信息 + 优化Prompt + 添加简单记忆

**技术栈：** Python, LLM Prompt优化, Context Metadata传递

---

## 问题分析

### 当前状态

| 指标 | 当前值 | MSNav ZS | 差距 |
|------|--------|----------|------|
| 成功率 (SR) | 10% | 50.9% | -40.9% |
| SPL | 0.079 | 0.426 | -0.347 |
| 平均终点距离 | 5.4m | 5.02m | +0.4m |

### 失败模式分析

| 模式 | 数量 | 占比 | 示例 |
|------|------|------|------|
| 快速失败 | 5/9 | 56% | Episode 47: 5步后停止 |
| 长距离失败 | 4/9 | 44% | Episode 191: 150步，距离7.87m |

### 根因分析

1. **方法缺失错误**：`_extract_action_and_condition`方法不存在，导致LLM指令分解失败
2. **决策信息不足**：DecisionAgent不知道距离变化趋势
3. **过早放弃**：Prompt没有引导Agent坚持导航
4. **原地打转**：没有记忆机制，可能重复走已经走过的路

---

## 解决方案设计

### 改进1：修复InstructionAgent方法缺失

**文件：** `agents/instruction_agent.py`

**问题：**
```python
# 错误日志
AttributeError: 'InstructionAgent' object has no attribute '_extract_action_and_condition'
```

**解决方案：** 添加缺失的方法

```python
def _extract_action_and_condition(self, description: str) -> Tuple[str, str]:
    """从子任务描述中提取动作和条件。

    Args:
        description: 子任务描述文本

    Returns:
        (action, condition) 元组
    """
    action_verbs = ["go", "move", "turn", "walk", "proceed", "continue", "find", "avoid", "navigate"]

    # 提取动词作为动作
    action = "navigate"
    desc_lower = description.lower()
    for verb in action_verbs:
        if verb in desc_lower:
            action = verb
            break

    # 条件使用完整描述
    condition = description.strip()

    return action, condition
```

**位置：** 在 `_parse_json_format` 方法前（约第1160行）

---

### 改进2：添加距离变化趋势反馈

**文件1：** `run_vln_experiment.py`（导航循环）

**位置：** 在 `dist` 计算后（约第1131行）

**代码：**
```python
dist = self._distance(pos, episode.goal_position)

# === 计算距离变化趋势 ===
last_distance = context.metadata.get("last_distance", dist)
distance_delta = last_distance - dist  # 正数=靠近，负数=远离

if distance_delta > 0.1:
    distance_trend = "正在靠近目标"
elif distance_delta < -0.1:
    distance_trend = "正在远离目标"
else:
    distance_trend = "距离稳定"

# 更新记录
context.metadata["last_distance"] = dist
context.metadata["distance_delta"] = distance_delta
context.metadata["distance_trend"] = distance_trend
```

**文件2：** `agents/decision_agent.py`（Prompt构建）

**位置：** 在 `_build_easy_prompt` 方法中

**代码：**
```python
# 获取距离变化信息
distance_delta = context.metadata.get("distance_delta", 0) if context else 0
distance_trend = context.metadata.get("distance_trend", "未知") if context else "未知"

# 在prompt中添加
- 目标距离: {distance_to_goal:.1f}m
- 距离变化: {distance_trend} ({distance_delta:+.2f}m)
```

---

### 改进3：优化DecisionAgent决策Prompt

**文件：** `agents/decision_agent.py`

**位置：** `_build_easy_prompt`, `_build_medium_prompt`, `_build_hard_prompt` 方法

**改进内容：**

```python
## 导航状态
- 当前目标距离: {distance:.1f}m (成功阈值: <3m)
- 距离变化: {trend} ({delta:+.2f}m)
- 目标方向: {direction_hint} ({angle_to_goal:.0f}°)
- 当前步数: {step}/150

## 导航规则
1. 如果正在靠近目标，继续当前方向前进
2. 如果正在远离目标，根据目标方向转向
3. 距离 < 3m 时，执行 stop 动作表示到达
4. 不要过早放弃，最大步数150步，坚持导航
5. 结合PerceptionAgent建议决定行动方向

## 重要提示
- 任务是到达目标位置，不要中途停止
- 每次决策都要考虑如何靠近目标
- 遇到障碍物时，绕行后继续向目标前进
```

---

### 改进4：添加简单记忆和打转检测

**文件1：** `run_vln_experiment.py`（导航循环）

**位置：** 在距离变化计算后

**代码：**
```python
# === 记录位置历史 ===
if "position_history" not in context.metadata:
    context.metadata["position_history"] = []

# 只记录x,z坐标（忽略y/高度）
pos_tuple = (round(pos[0], 1), round(pos[2], 1))
context.metadata["position_history"].append(pos_tuple)

# 限制历史长度，避免内存过大
if len(context.metadata["position_history"]) > 20:
    context.metadata["position_history"] = context.metadata["position_history"][-20:]

# === 检测原地打转 ===
recent_positions = context.metadata["position_history"][-5:]
if len(recent_positions) >= 5:
    unique_positions = set(recent_positions)
    if len(unique_positions) <= 2:
        context.metadata["stuck_detected"] = True
        context.metadata["stuck_message"] = "检测到原地打转，建议转向探索新方向"
    else:
        context.metadata["stuck_detected"] = False

# 记录已探索的不同位置数
context.metadata["explored_positions"] = len(set(context.metadata["position_history"]))
```

**文件2：** `agents/decision_agent.py`（Prompt构建）

**代码：**
```python
# 获取记忆信息
stuck_detected = context.metadata.get("stuck_detected", False) if context else False
stuck_message = context.metadata.get("stuck_message", "") if context else ""
explored_positions = context.metadata.get("explored_positions", 0) if context else 0

# 在prompt中添加
- 已探索位置: {explored_positions}个
{f"- ⚠️ {stuck_message}" if stuck_detected else ""}
```

---

## 文件修改清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|----------|
| `agents/instruction_agent.py` | 约第1160行 | 添加 `_extract_action_and_condition` 方法 |
| `run_vln_experiment.py` | 约第1131行 | 计算距离变化趋势 |
| `run_vln_experiment.py` | 距离变化后 | 添加位置历史和打转检测 |
| `agents/decision_agent.py` | `_build_easy_prompt` | 显示距离变化、优化规则、添加记忆信息 |
| `agents/decision_agent.py` | `_build_medium_prompt` | 同上 |
| `agents/decision_agent.py` | `_build_hard_prompt` | 同上 |

---

## 验收标准

| 指标 | 修改前 | 修改后预期 |
|------|--------|-----------|
| 成功率 (SR) | 10% | 30-50% |
| 快速失败率 | 50% | <20% |
| 平均终点距离 | 5.4m | <4m |
| SPL | 0.079 | >0.2 |

---

## 测试方法

```bash
# 1. 语法验证
python -m py_compile agents/instruction_agent.py run_vln_experiment.py agents/decision_agent.py

# 2. 单episode测试
python scripts/run_emergency_eval.py --exp baseline --episodes 1 --use-remote-llm --llm-server http://localhost:8000

# 3. 并行评估（10 episodes）
python scripts/run_emergency_eval.py --exp baseline --episodes 10 --parallel 3 --gpus 1,2,3 --use-remote-llm --llm-server http://localhost:8000

# 4. 检查日志验证改进
grep -E "距离变化|打转|_extract_action" /tmp/parallel_eval.log
```

---

## 预期效果

修改后，DecisionAgent将获得更丰富的决策信息：

**修改前Prompt示例：**
```
- Distance to goal: 5.2m
```

**修改后Prompt示例：**
```
- 目标距离: 5.2m (成功阈值: <3m)
- 距离变化: 正在靠近目标 (+0.15m)
- 目标方向: 右前方 (45°)
- 已探索位置: 8个
- 当前步数: 25/150

## 导航规则
1. 如果正在靠近目标，继续当前方向前进
2. 如果正在远离目标，根据目标方向转向
3. 距离 < 3m 时，执行 stop 动作表示到达
4. 不要过早放弃，最大步数150步，坚持导航
```

这将帮助Agent做出更明智的导航决策。