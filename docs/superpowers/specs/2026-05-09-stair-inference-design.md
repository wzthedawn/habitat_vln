# 楼梯方向推断改进设计

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让DecisionAgent从scene_description文本中自主推断楼梯方向，不依赖VLM输出结构化stair_direction字段

**Architecture:** 信息流调整 - 将scene_description直接传入DecisionAgent prompt，利用LLM推理能力从模糊文本中提取导航线索

**Tech Stack:** Python, LLM prompt修改

---

## 问题分析

### 当前信息流

```
VLM输出:
  room_type: "hallway" (错误分类)
  scene_description: "A narrow indoor hallway..."
  stair_direction: "none" (未检测)

→ CoT分析（scene_description传入，但无楼梯关键词）
→ DecisionAgent（只看到room_type + objects，缺失scene_description）
```

### 根本问题

1. **VLM无法识别楼梯** - Qwen-9b-perception从depth图像无法推断高度变化
2. **DecisionAgent缺失关键信息** - prompt中没有scene_description字段
3. **过度依赖结构化字段** - stair_direction字段为none就放弃楼梯检测

### 解决思路

LLM具有更强的推理能力，可以从模糊文本中推断意图。将scene_description直接传入DecisionAgent，让LLM自己判断。

---

## 设计方案

### 改动点

**文件: `agents/decision_agent.py`**

修改 `_build_medium_prompt_with_analysis_v2` 方法（约第1019行）：

```python
# 当前状态信息（缺失scene_description）：
## 状态
- {state_str}
- 房间: {room_type}, 可见{[o.get("name") for o in objects[:3]]}

# 改为（添加scene_description）：
## 状态
- {state_str}
- 房间: {room_type}
- 场景描述: {scene_desc[:100]}
- 可见物体: {[o.get("name") for o in objects[:3]]}
```

### 信息流变化

```
改动后：
VLM输出:
  scene_description: "A narrow indoor hallway with light-colored walls..."

→ CoT分析（scene_description传入）
→ DecisionAgent（现在能看到scene_description）
   → LLM推理：场景描述中是否有楼梯线索？
   → 结合指令"Walk down the stairs"做出判断
```

### 为什么能解决

即使scene_description不完美，LLM可以：
- 从模糊描述中推断意图（"path leads away" → 可能是楼梯）
- 结合用户指令做出推理（指令说"stairs" → 寻找楼梯线索）
- 观察Y坐标变化趋势（持续上升 → 可能走错方向）

---

## 实现步骤

### Task 1: 修改DecisionAgent prompt

**文件:** `agents/decision_agent.py:1019`

- [ ] **Step 1: 在状态信息中添加scene_description字段**

修改 `_build_medium_prompt_with_analysis_v2` 方法的prompt模板：

```python
# 在第1018-1019行之间添加scene_description
## 状态
- {state_str}
- 房间: {room_type}
- 场景描述: {scene_desc[:100]}
- 可见物体: {[o.get("name") for o in objects[:3]]}
{topology_str}
```

- [ ] **Step 2: 验证参数传递**

确认 `scene_desc` 参数已从方法入口传入（第950行已有参数）。

- [ ] **Step 3: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat: DecisionAgent prompt添加scene_description字段

- 让LLM从场景描述文本中推断楼梯方向
- 不依赖VLM结构化stair_direction字段
- 提升楼梯场景导航准确性"
```

---

### Task 2: 验证测试

- [ ] **Step 1: 运行楼梯场景实验**

```bash
bash scripts/run_vln_experiment_with_vllm.sh run_exp
```

- [ ] **Step 2: 检查DecisionAgent推理输出**

确认prompt中包含场景描述，LLM推理是否更准确。

- [ ] **Step 3: 验证Y坐标变化**

期望：agent向下走楼梯（Y减小），而非向上（Y增大）。

---

## Self-Review

**1. Placeholder scan:** ✅ 无TBD/TODO

**2. Internal consistency:** ✅ 设计描述与实现步骤一致

**3. Scope check:** ✅ 单文件改动，范围可控

**4. Ambiguity check:** ✅ 明确改动位置和内容