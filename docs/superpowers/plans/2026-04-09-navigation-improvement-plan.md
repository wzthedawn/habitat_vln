# 导航准确率提升实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 解决导航失败问题，提升成功率从0%到预期30-50%

**架构：** 四个独立改进模块：子任务分解修复、感知-目标对齐、障碍物位置增强、距离反馈

**技术栈：** Python, LLM prompt优化, context metadata传递

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `agents/instruction_agent.py` | 子任务分解 - 改进LLM prompt，添加应急指令识别 |
| `agents/perception_agent.py` | 感知对齐 - 添加目标方向参数到VLM prompt |
| `run_vln_experiment.py` | 障碍物位置计算、距离变化计算 |
| `agents/decision_agent.py` | 显示障碍物信息、距离变化反馈 |

---

### Task 1: 修复LLM子任务分解

**Files:**
- Modify: `agents/instruction_agent.py:735-810`
- Modify: `agents/instruction_agent.py:444-520`

- [ ] **Step 1: 添加应急指令模板识别常量**

在 `InstructionAgent` 类中，约第26行后添加：

```python
# 应急指令模板正则
EMERGENCY_TEMPLATE_PATTERN = r"(.+?), the path is blocked, (.+?) to reach (.+)"
```

- [ ] **Step 2: 添加 `_parse_emergency_instruction` 方法**

在 `_split_instruction` 方法前（约第444行），添加新方法：

```python
def _parse_emergency_instruction(self, instruction: str) -> Optional[List[str]]:
    """解析应急指令模板格式。

    应急指令模板: "{normal_action}, the path is blocked, {reroute_action} to reach {goal}"

    Args:
        instruction: 导航指令

    Returns:
        解析后的子任务列表，如果不匹配则返回None
    """
    match = re.match(self.EMERGENCY_TEMPLATE_PATTERN, instruction, re.IGNORECASE)
    if match:
        normal_action = match.group(1).strip()  # "Go straight"
        reroute_action = match.group(2).strip()  # "go back and try different path"
        goal = match.group(3).strip()  # "the table"

        self.logger.info(f"[Instruction] Matched emergency template: action={normal_action}, reroute={reroute_action}, goal={goal}")

        return [
            f"首先尝试{normal_action}",
            f"检测到路径阻塞，执行{reroute_action}",
            f"继续前进到达{goal}"
        ]
    return None
```

- [ ] **Step 3: 修改 `_split_instruction` 方法优先检测应急指令**

修改 `_split_instruction` 方法开头（第444-460行），添加应急指令检测：

```python
def _split_instruction(self, text: str) -> List[str]:
    """Split instruction into subtask segments with enhanced recognition.

    Enhanced rules:
    1. Check for emergency instruction template first
    2. Identify key action keywords as segment boundaries
    3. Handle conjunction patterns more accurately
    4. Preserve landmark context with each segment
    """
    # 首先检测应急指令模板
    emergency_segments = self._parse_emergency_instruction(text)
    if emergency_segments:
        return emergency_segments

    # Key action keywords that typically start new subtasks
    action_keywords = ["turn", "walk", "go", "move", "stop", "wait", "find", "enter", "exit", "head", "pass", "cross"]
    # ... 后续代码保持不变
```

- [ ] **Step 4: 改进 `_semantic_decompose_with_llm` 方法的prompt**

修改 `_semantic_decompose_with_llm` 方法中的prompt（约第749-784行）：

```python
prompt = f"""/no_think
你是导航指令分析专家。将指令分解为可执行的子任务。

## 指令
{instruction}

## 分解规则
1. 每个子任务必须是完整的句子（动词+方向+目标）
2. 应急指令（含"blocked"、"obstacle"、"path"）分解为：检测障碍 → 寻找替代路线 → 继续前进
3. 不要按逗号机械分割，要理解语义
4. 子任务数量控制在2-4个

## 输出格式（严格JSON，无其他内容）
{{
  "subtasks": [
    {{
      "id": 0,
      "description": "检测到路径阻塞，准备绕行",
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

- [ ] **Step 5: 验证语法**

Run: `python -m py_compile agents/instruction_agent.py`
Expected: 无错误输出

- [ ] **Step 6: Commit**

```bash
git add agents/instruction_agent.py
git commit -m "fix: improve subtask decomposition for emergency instructions"
```

---

### Task 2: PerceptionAgent目标方向对齐

**Files:**
- Modify: `agents/perception_agent.py:238-300`
- Modify: `agents/perception_agent.py:110-140`

- [ ] **Step 1: 修改 `_analyze_with_vlm` 方法签名添加目标方向参数**

修改 `_analyze_with_vlm` 方法签名（第238-242行）：

```python
def _analyze_with_vlm(
    self,
    rgb_image: Optional[np.ndarray],
    depth_image: Optional[np.ndarray],
    goal_direction: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze RGB + Depth images using VLM with goal direction awareness.

    Args:
        rgb_image: RGB图像
        depth_image: 深度图像
        goal_direction: 目标方向信息 {"angle": 角度, "hint": 方向提示}

    Returns:
        Dictionary with room_type, objects, scene_description, nav_hint, walkable_analysis, obstacle_ahead
    """
```

- [ ] **Step 2: 修改VLM prompt添加目标方向信息**

在 `_analyze_with_vlm` 方法中，修改prompt构建部分（约第256-287行），添加目标方向：

```python
has_depth = depth_image is not None

# 构建目标方向提示
goal_hint = ""
if goal_direction:
    angle = goal_direction.get("angle", 0)
    hint = goal_direction.get("hint", "前方")
    goal_hint = f"""
## 目标方向
- 目标在{hint}（相对角度{angle:.0f}°）
- 导航建议应与目标方向一致"""

if has_depth:
    # RGB + Depth: Simplified prompt with stair direction detection
    prompt = f"""/no_think
You are a navigation robot. Analyze the images and output JSON.

## Images
- First: RGB image
- Second: Depth image (red=near, blue=far)
{goal_hint}

## Required Output (JSON only)
{{"room_type":"hallway/bedroom/living_room/kitchen/bathroom/stairs",
"scene_brief":"short description in 10-15 words",
"objects":[{{"name":"object name","distance":1.0}}],
"stair_direction":"up/down/none",
"nav_hint":"navigation hint in 20-30 words"}}

## Guidelines
1. room_type: Use simple names (hallway, bedroom, living_room, kitchen, bathroom, stairs)
2. scene_brief: Brief description of what you see
3. objects: List 3-5 visible objects with distance in meters
4. stair_direction: ONLY when stairs are visible:
   - "up" if stairs ascend away from you
   - "down" if stairs descend away from you
   - "none" if no stairs visible
5. nav_hint: Key navigation guidance aligned with goal direction{f" (目标在{goal_direction.get('hint', '前方')})" if goal_direction else ""}. If stairs visible, mention direction.

Output JSON only, no explanation:"""
```

- [ ] **Step 3: 修改 `process` 方法获取并传递目标方向**

修改 `process` 方法中调用 `_analyze_with_vlm` 的部分（约第138-140行）：

```python
# Use VLM for all perception (RGB + Depth fusion)
vlm_result = None
if self._model_manager:
    # 获取目标方向信息
    goal_direction = None
    if context and hasattr(context, 'metadata'):
        angle = context.metadata.get("angle_to_goal", 0)
        hint = context.metadata.get("direction_hint", "前方")
        if angle != 0 or hint != "前方":
            goal_direction = {"angle": angle, "hint": hint}

    vlm_result = self._analyze_with_vlm(rgb_image, depth_image, goal_direction)
```

- [ ] **Step 4: 验证语法**

Run: `python -m py_compile agents/perception_agent.py`
Expected: 无错误输出

- [ ] **Step 5: Commit**

```bash
git add agents/perception_agent.py
git commit -m "feat: add goal direction awareness to PerceptionAgent"
```

---

### Task 3: 障碍物位置信息增强

**Files:**
- Modify: `run_vln_experiment.py:710-734`
- Modify: `agents/decision_agent.py:650-670`

- [ ] **Step 1: 在障碍物触发后计算相对位置**

在 `run_vln_experiment.py` 第711-713行后添加障碍物位置计算：

找到这段代码：
```python
if triggered:
    self.logger.warning(f"[EMERGENCY] Obstacle triggered at step {steps}!")
    obstacle_state = obstacle_manager.get_obstacle_state()
    context.metadata["obstacle_state"] = obstacle_state
```

在其后添加：
```python
if triggered:
    self.logger.warning(f"[EMERGENCY] Obstacle triggered at step {steps}!")
    obstacle_state = obstacle_manager.get_obstacle_state()
    context.metadata["obstacle_state"] = obstacle_state

    # === 计算障碍物相对位置 ===
    obstacles = obstacle_state.get("obstacles", [])
    if obstacles:
        obstacle = obstacles[0]  # 取第一个障碍物
        obstacle_pos = obstacle.get("position", (0, 0, 0))

        # 计算相对位置
        dx = obstacle_pos[0] - pos[0]
        dz = obstacle_pos[2] - pos[2]
        obstacle_angle = math.atan2(dx, dz)
        obstacle_relative_angle = obstacle_angle - context.rotation
        obstacle_relative_angle_deg = math.degrees(obstacle_relative_angle)
        obstacle_relative_angle_deg = ((obstacle_relative_angle_deg + 180) % 360) - 180

        obstacle_radius = obstacle.get("radius", 1.0)
        obstacle_dist = math.sqrt(dx*dx + dz*dz)

        # 障碍物方向提示
        if -60 < obstacle_relative_angle_deg < 60:
            obstacle_direction = "前方"
        elif 60 <= obstacle_relative_angle_deg < 120:
            obstacle_direction = "右前方"
        elif -120 <= obstacle_relative_angle_deg < -60:
            obstacle_direction = "左前方"
        elif 120 <= obstacle_relative_angle_deg <= 180 or -180 <= obstacle_relative_angle_deg < -120:
            obstacle_direction = "侧后方"
        else:
            obstacle_direction = "前方"

        # 绕行建议
        bypass_direction = "左" if obstacle_relative_angle_deg > 0 else "右"

        # 存入context
        context.metadata["obstacle_info"] = {
            "position": obstacle_pos,
            "radius": obstacle_radius,
            "distance": obstacle_dist,
            "relative_angle": obstacle_relative_angle_deg,
            "direction": obstacle_direction,
            "bypass_direction": bypass_direction,
            "description": obstacle.get("description", "障碍物")
        }

        self.logger.info(f"[EMERGENCY] Obstacle at {obstacle_direction}, distance {obstacle_dist:.1f}m, bypass {bypass_direction}")
```

- [ ] **Step 2: 在DecisionAgent prompt中显示障碍物信息**

修改 `agents/decision_agent.py` 的 `_build_easy_prompt` 方法（约第654行后），在获取目标方向信息后添加：

```python
# Get goal direction info
angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

# Get obstacle info
obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}
```

- [ ] **Step 3: 在prompt中添加障碍物信息显示**

在 `_build_easy_prompt` 方法中，找到构建prompt的位置（约第810行），在 `- Goal direction` 行后添加：

```python
- Goal direction: 目标在{direction_hint} (相对角度: {angle_to_goal:.0f}°){f"""
- 障碍物: {obstacle_info.get('direction', '')} {obstacle_info.get('distance', 0):.1f}m, 半径 {obstacle_info.get('radius', 1.0):.1f}m
- 绕行建议: 向{obstacle_info.get('bypass_direction', '右')}绕行""" if obstacle_info else ""}
```

- [ ] **Step 4: 同样修改 `_build_medium_prompt` 和 `_build_hard_prompt`**

在相同的position添加障碍物信息显示。

- [ ] **Step 5: 验证语法**

Run: `python -m py_compile run_vln_experiment.py agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 6: Commit**

```bash
git add run_vln_experiment.py agents/decision_agent.py
git commit -m "feat: add obstacle position info to context and DecisionAgent prompt"
```

---

### Task 4: 距离变化反馈

**Files:**
- Modify: `run_vln_experiment.py:1129-1160`
- Modify: `agents/decision_agent.py:810-850`

- [ ] **Step 1: 在导航循环中计算距离变化**

在 `run_vln_experiment.py` 第1129行后，`dist` 计算后添加距离变化计算：

找到这段代码：
```python
dist = self._distance(pos, episode.goal_position)

# === 计算目标方向角度 ===
```

在 `dist` 计算后、目标方向计算前添加：

```python
dist = self._distance(pos, episode.goal_position)

# === 计算距离变化 ===
last_distance = context.metadata.get("last_distance", dist)
distance_delta = last_distance - dist  # 正数=靠近，负数=远离

# 判断趋势
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

# === 计算目标方向角度 ===
```

- [ ] **Step 2: 在DecisionAgent prompt中显示距离变化**

修改 `agents/decision_agent.py` 的 `_build_easy_prompt` 方法，在获取障碍物信息后添加：

```python
# Get obstacle info
obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}

# Get distance change info
distance_delta = context.metadata.get("distance_delta", 0) if context else 0
distance_trend = context.metadata.get("distance_trend", "未知") if context else "未知"
```

- [ ] **Step 3: 在prompt中添加距离变化显示**

在 `_build_easy_prompt` 方法的prompt中，在 `- Distance to goal` 行后添加：

```python
- Distance to goal: {distance_to_goal:.1f}m
- 距离变化: {distance_trend} ({distance_delta:+.2f}m)
```

- [ ] **Step 4: 同样修改 `_build_medium_prompt` 和 `_build_hard_prompt`**

在相同的position添加距离变化显示。

- [ ] **Step 5: 验证语法**

Run: `python -m py_compile run_vln_experiment.py agents/decision_agent.py`
Expected: 无错误输出

- [ ] **Step 6: Commit**

```bash
git add run_vln_experiment.py agents/decision_agent.py
git commit -m "feat: add distance change feedback to navigation"
```

---

### Task 5: 集成测试

**Files:**
- None (测试运行)

- [ ] **Step 1: 确认vLLM服务器运行**

Run: `curl -s http://localhost:8000/health`
Expected: 返回 `{"status":"healthy",...}`

如果未运行：
```bash
CUDA_VISIBLE_DEVICES=0 /home/WZ/.conda/envs/vllm_env/bin/python vllm_server.py --port 8000 &
```

- [ ] **Step 2: 运行单episode测试**

Run: `python scripts/run_emergency_eval.py --exp baseline --episodes 1 --no-video --no-trajectory --use-remote-llm --llm-server http://localhost:8000 2>&1 | tee /tmp/nav_test.log`

Expected: 测试完成无错误

- [ ] **Step 3: 检查日志验证改进**

Run: `grep -E "Subtask|障碍物|距离变化|nav_hint|Goal direction" /tmp/nav_test.log | head -30`

验证：
- 子任务描述完整（无"Proceed right obstacle appeared..."碎片）
- 障碍物位置正确显示（如"障碍物: 前方 3.5m"）
- 距离变化正确显示（如"距离变化: 正在靠近目标 (+0.15m)"）
- nav_hint与目标方向一致

- [ ] **Step 4: 批量评估（可选）**

Run: `python scripts/run_emergency_eval.py --exp baseline --episodes 10 --use-remote-llm --llm-server http://localhost:8000`

Expected: 成功率 > 0%，平均终点距离 < 6m

---

## 验收清单

- [ ] 应急指令正确分解为完整子任务
- [ ] PerceptionAgent nav_hint与目标方向一致
- [ ] 障碍物位置信息正确显示在DecisionAgent prompt
- [ ] 距离变化反馈正确显示
- [ ] 导航成功率提升（从0%提升）
- [ ] 所有commit已提交

## 预期效果

| 指标 | 修改前 | 修改后 |
|------|--------|--------|
| 子任务描述 | 破损碎片化 | 完整句子 |
| nav_hint一致性 | 与目标冲突 | 与目标方向一致 |
| 障碍物信息 | 仅知道存在 | 位置+距离+绕行建议 |
| 导航反馈 | 仅当前距离 | 距离+变化趋势 |
| 导航成功率 | 0% | 预期 30-50% |
| 平均终点距离 | 6.5m | 预期 < 3m |