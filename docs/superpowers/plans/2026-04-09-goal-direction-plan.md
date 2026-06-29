# 目标方向感知实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 添加目标方向角度让Agent知道目标在哪个方向，提高导航成功率。

**Architecture:** 在导航循环中计算目标相对角度，存入context.metadata，DecisionAgent在prompt中显示"目标在X方向"。

**Tech Stack:** Python, math.atan2, Habitat simulator

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `run_vln_experiment.py:1129-1145` | 计算angle_to_goal，存入context |
| `agents/decision_agent.py:541,790-815` | 读取angle_to_goal，添加到prompt |

---

### Task 1: 在导航循环中计算目标方向角度

**Files:**
- Modify: `run_vln_experiment.py:1129-1145`

- [ ] **Step 1: 找到计算distance_to_goal的位置**

读取 `run_vln_experiment.py` 第1125-1145行，找到更新context.rotation和dist的代码块。

- [ ] **Step 2: 在dist计算后添加角度计算代码**

在第1129行 `dist = self._distance(pos, episode.goal_position)` 之后添加：

```python
# === 计算目标方向角度 ===
dx = episode.goal_position[0] - pos[0]
dz = episode.goal_position[2] - pos[2]
angle_to_goal = math.atan2(dx, dz)
relative_angle = angle_to_goal - context.rotation
relative_angle_deg = math.degrees(relative_angle)
# 归一化到 [-180, 180]
relative_angle_deg = ((relative_angle_deg + 180) % 360) - 180

# 生成人类可读方向提示
if -30 < relative_angle_deg < 30:
    direction_hint = "前方"
elif 30 <= relative_angle_deg < 60:
    direction_hint = "右前方"
elif 60 <= relative_angle_deg < 120:
    direction_hint = "右边"
elif 120 <= relative_angle_deg <= 180:
    direction_hint = "右后方"
elif -60 <= relative_angle_deg < -30:
    direction_hint = "左前方"
elif -120 <= relative_angle_deg < -60:
    direction_hint = "左边"
else:
    direction_hint = "左后方"

# 存入 context metadata
context.metadata["angle_to_goal"] = relative_angle_deg
context.metadata["direction_hint"] = direction_hint
```

- [ ] **Step 3: 验证语法**

Run: `python -m py_compile run_vln_experiment.py`
Expected: 无错误输出

- [ ] **Step 4: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat: calculate goal direction angle in navigation loop"
```

---

### Task 2: 在DecisionAgent prompt中显示目标方向

**Files:**
- Modify: `agents/decision_agent.py:541,790-815`

- [ ] **Step 1: 找到_build_easy_prompt方法中distance_to_goal的位置**

读取 `agents/decision_agent.py` 第790-820行，找到构建prompt的代码。

- [ ] **Step 2: 在distance_to_goal行后添加目标方向信息**

找到类似 `- Distance to goal: {distance_to_goal:.1f}m` 的行，在其后添加：

```python
# 获取目标方向信息
angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

# 在prompt中添加（找到现有的distance_to_goal行后面加）
prompt += f"\n- Goal direction: 目标在{direction_hint} (相对角度: {angle_to_goal:.0f}°)"
```

需要修改 `_build_easy_prompt` 方法，在约第811行添加：

```python
- Heading: {heading}
- Distance to goal: {distance_to_goal:.1f}m
- Goal direction: 目标在{direction_hint} (相对角度: {angle_to_goal:.0f}°)
```

- [ ] **Step 3: 同样修改_build_medium_prompt和_build_hard_prompt方法**

在相同位置添加目标方向信息。

- [ ] **Step 4: 验证语法**

Run: `python -m py_compile agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 5: Commit**

```bash
git add agents/decision_agent.py
git commit -m "feat: show goal direction in DecisionAgent prompt"
```

---

### Task 3: 集成测试

**Files:**
- None (测试运行)

- [ ] **Step 1: 启动vLLM服务器**

```bash
CUDA_VISIBLE_DEVICES=0 /home/WZ/.conda/envs/vllm_env/bin/python vllm_server.py --port 8000 &
```

- [ ] **Step 2: 运行单个episode测试**

Run: `python scripts/run_emergency_eval.py --exp baseline --episodes 1 --no-video --no-trajectory --use-remote-llm --llm-server http://localhost:8000`

Expected: 日志中显示 `Goal direction: 目标在X方 (相对角度: X°)`

- [ ] **Step 3: 检查导航行为改善**

对比最终距离是否比之前的6.43m更小。

---

## 验收清单

- [ ] angle_to_goal 在导航循环中正确计算
- [ ] direction_hint 正确转换为中文方向
- [ ] DecisionAgent prompt 中显示目标方向
- [ ] 导航成功率提升（从0%提升）

## 预期效果

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 导航成功率 | 0% | 预计 30-50% |
| 平均距离 | 6.5m | 预计 < 3m |