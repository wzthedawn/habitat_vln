# 导航准确率提升设计方案

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 解决导航失败问题，提升成功率从0%到预期30-50%

**架构：** 四个独立改进模块：子任务分解修复、感知-目标对齐、障碍物位置增强、距离反馈

**技术栈：** Python, LLM prompt优化, context metadata传递

---

## 问题根因分析

### 问题1：子任务描述损坏

**现象：** 日志显示 `Proceed right obstacle appeared...` 这样的破损子任务描述

**原因：** 应急指令模板 `"Go straight, the path is blocked, go back and try different path to reach the table."` 按逗号分割后产生碎片化子任务。LLM语义分解失败后回退到简单分割，无法正确理解指令结构。

### 问题3：感知与目标方向冲突

**现象：** PerceptionAgent说"Continue straight"但目标实际在右前方48°

**原因：** PerceptionAgent生成nav_hint时不知道目标方向，产生与目标冲突的建议，导致Agent困惑。

### 问题4：障碍物处理效率低

**现象：** Agent执行 turn_right + forward 循环，绕障碍物打转浪费步数

**原因：** Agent只知道障碍物"存在"，不知道具体位置/大小，无法规划高效绕行路径。

### 问题5：无导航进度感知

**现象：** Agent不知道是否正在靠近/远离目标

**原因：** 每次决策只显示当前距离，没有距离变化趋势，无法评估导航效果。

---

## 解决方案设计

### 方案1A：修复LLM子任务分解

**文件：** `agents/instruction_agent.py`

**改动点：**

1. **改进 `_semantic_decompose_with_llm` 方法（约第735行）**

   - 强化JSON格式约束，使用更明确的输出模板
   - 添加应急指令识别逻辑
   - 确保每个子任务描述是完整句子

```python
prompt = f"""/no_think
你是导航指令分析专家。将指令分解为可执行的子任务。

## 指令
{instruction}

## 分解规则
1. 每个子任务必须是完整的句子（动词+方向+目标）
2. 应急指令（含"blocked"、"obstacle"、"path"）分解为：绕行准备 + 寻找替代路线 + 继续前进
3. 不要按逗号机械分割，要理解语义

## 输出格式（严格JSON）
{{
  "subtasks": [
    {{
      "id": 0,
      "description": "检测到路径阻塞，准备向右绕行",
      "level": "easy"
    }},
    {{
      "id": 1,
      "description": "寻找替代路线避开障碍物",
      "level": "medium"
    }},
    {{
      "id": 2,
      "description": "继续向目标前进",
      "level": "easy"
    }}
  ]
}}

只输出JSON，无其他内容。"""
```

2. **优化 `_split_instruction` 方法（约第444行）**

   - 添加应急指令模板识别：`{normal_action}, the path is blocked, {reroute_action} to reach {goal}`
   - 对应急指令使用专用分割逻辑，而非按逗号分割

3. **添加应急指令模板识别函数**

```python
EMERGENCY_TEMPLATE_PATTERN = r"(.+?), the path is blocked, (.+?) to reach (.+)"

def _parse_emergency_instruction(self, instruction: str) -> List[str]:
    """解析应急指令模板格式"""
    match = re.match(self.EMERGENCY_TEMPLATE_PATTERN, instruction)
    if match:
        normal_action = match.group(1)  # "Go straight"
        reroute_action = match.group(2)  # "go back and try different path"
        goal = match.group(3)  # "the table"

        return [
            f"首先尝试{normal_action}",
            f"检测到路径阻塞，执行{reroute_action}",
            f"继续前进到达{goal}"
        ]
    return None  # 不匹配模板，使用默认分割
```

---

### 方案3A：PerceptionAgent目标方向对齐

**文件：** `agents/perception_agent.py`

**改动点：**

1. **修改 `_build_vlm_prompt` 方法接收目标方向参数**

   - 在方法签名添加 `goal_direction` 参数
   - 在prompt中包含目标方向信息

```python
def _build_vlm_prompt(self, rgb_desc: str, depth_desc: str, goal_direction: dict = None) -> str:
    """构建VLM感知prompt，包含目标方向"""
    goal_hint = ""
    if goal_direction:
        angle = goal_direction.get("angle", 0)
        hint = goal_direction.get("hint", "前方")
        goal_hint = f"\n目标位置：{hint}（相对角度{angle:.0f}°）。导航建议应与目标方向一致。"

    prompt = f"""分析场景并提供导航建议。

## RGB观察
{rgb_desc}

## 深度观察
{depth_desc}

## 目标信息{goal_hint}

请提供：
1. room_type: 房间类型
2. scene_brief: 场景简述
3. objects: 可见物体列表
4. nav_hint: 导航建议（需与目标方向一致）

输出JSON格式。"""
    return prompt
```

2. **修改 `process` 方法从context获取目标方向**

```python
def process(self, context: NavContext, ...) -> AgentOutput:
    # 获取目标方向信息
    goal_direction = {
        "angle": context.metadata.get("angle_to_goal", 0),
        "hint": context.metadata.get("direction_hint", "前方")
    }

    # 构建prompt时传入
    prompt = self._build_vlm_prompt(rgb_desc, depth_desc, goal_direction)
```

---

### 方案4A：障碍物位置信息增强

**文件：** `run_vln_experiment.py`（导航循环部分）

**改动点：**

1. **在障碍物触发后计算相对位置（约第711行后）**

```python
if triggered:
    # 计算障碍物相对位置
    obstacle_pos = obstacle_state.get("obstacles", [{}])[0].get("position", (0,0,0))
    dx = obstacle_pos[0] - pos[0]
    dz = obstacle_pos[2] - pos[2]
    obstacle_angle = math.atan2(dx, dz)
    obstacle_relative_angle = obstacle_angle - context.rotation
    obstacle_relative_angle_deg = math.degrees(obstacle_relative_angle)
    obstacle_relative_angle_deg = ((obstacle_relative_angle_deg + 180) % 360) - 180

    obstacle_radius = obstacle_state.get("obstacles", [{}])[0].get("radius", 1.0)
    obstacle_dist = math.sqrt(dx*dx + dz*dz)

    # 障碍物方向提示
    if -60 < obstacle_relative_angle_deg < 60:
        obstacle_direction = "前方"
    elif 60 <= obstacle_relative_angle_deg < 120:
        obstacle_direction = "右前方"
    elif -120 <= obstacle_relative_angle_deg < -60:
        obstacle_direction = "左前方"
    else:
        obstacle_direction = "侧后方"

    # 存入context
    context.metadata["obstacle_info"] = {
        "position": obstacle_pos,
        "radius": obstacle_radius,
        "distance": obstacle_dist,
        "relative_angle": obstacle_relative_angle_deg,
        "direction": obstacle_direction,
        "description": obstacle_state.get("obstacles", [{}])[0].get("description", "障碍物")
    }
```

**文件：** `agents/decision_agent.py`

**改动点：**

2. **在 `_build_easy_prompt` 等方法中显示障碍物信息**

```python
# 获取障碍物信息
obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}

# 在prompt中添加
obstacle_section = ""
if obstacle_info:
    obstacle_section = f"""
- 障碍物位置: {obstacle_info.get('direction', '未知')} {obstacle_info.get('relative_angle', 0):.0f}°
- 障碍物距离: {obstacle_info.get('distance', 0):.1f}m，半径: {obstacle_info.get('radius', 1.0):.1f}m
- 绕行建议: 向{'左' if obstacle_info.get('relative_angle', 0) > 0 else '右'}绕行"""
```

---

### 方案5A：距离变化反馈

**文件：** `run_vln_experiment.py`

**改动点：**

1. **在导航循环中计算距离变化（约第1131行附近）**

```python
# 记录上一帧距离
if "last_distance" not in context.metadata:
    context.metadata["last_distance"] = dist

last_distance = context.metadata.get("last_distance", dist)
distance_delta = last_distance - dist  # 正数=靠近，负数=远离

# 判断趋势
if distance_delta > 0.1:
    distance_trend = "正在靠近目标"
elif distance_delta < -0.1:
    distance_trend = "正在远离目标"
else:
    distance_trend = "距离基本稳定"

# 更新记录
context.metadata["last_distance"] = dist
context.metadata["distance_delta"] = distance_delta
context.metadata["distance_trend"] = distance_trend
```

**文件：** `agents/decision_agent.py`

**改动点：**

2. **在prompt中显示距离变化**

```python
# 获取距离变化信息
distance_delta = context.metadata.get("distance_delta", 0) if context else 0
distance_trend = context.metadata.get("distance_trend", "未知") if context else "未知"

# 在prompt中（Distance to goal行后）
- Distance to goal: {distance_to_goal:.1f}m
- 距离变化: {distance_trend} ({distance_delta:+.2f}m)
```

---

## 文件修改清单

| 文件 | 改动位置 | 改动内容 |
|------|----------|---------|
| `agents/instruction_agent.py` | 第735行 `_semantic_decompose_with_llm` | 改进prompt，强化JSON约束 |
| `agents/instruction_agent.py` | 第444行 `_split_instruction` | 添加应急指令模板识别 |
| `agents/instruction_agent.py` | 新增函数 | `_parse_emergency_instruction` |
| `agents/perception_agent.py` | `_build_vlm_prompt` | 添加目标方向参数 |
| `agents/perception_agent.py` | `process` | 从context获取目标方向 |
| `run_vln_experiment.py` | 第711行后 | 计算障碍物相对位置 |
| `run_vln_experiment.py` | 第1131行附近 | 计算距离变化 |
| `agents/decision_agent.py` | `_build_easy_prompt` 等 | 显示障碍物位置、距离变化 |

---

## 验收标准

| 指标 | 修改前 | 修改后预期 |
|------|--------|-----------|
| 子任务描述 | 破损碎片化 | 完整句子 |
| nav_hint一致性 | 与目标冲突 | 与目标方向一致 |
| 障碍物绕行 | 打转浪费步数 | 高效绕行 |
| 导航成功率 | 0% | 30-50% |
| 平均终点距离 | 6.5m | < 3m |

---

## 测试方法

```bash
# 1. 单episode验证
python scripts/run_emergency_eval.py --exp baseline --episodes 1 --no-video --no-trajectory --use-remote-llm --llm-server http://localhost:8000

# 2. 检查日志验证改进
# - 子任务描述完整（无碎片）
# - nav_hint与目标方向一致
# - 障碍物位置正确显示
# - 距离变化趋势正确

# 3. 批量评估
python scripts/run_emergency_eval.py --exp all --episodes 10 --use-remote-llm --llm-server http://localhost:8000
```