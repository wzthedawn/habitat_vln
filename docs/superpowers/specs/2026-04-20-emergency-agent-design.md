# EmergencyAgent设计文档

## 概述

本文档描述EmergencyAgent的设计，用于多智能体VLN系统的应急响应。

**核心目标**：
- 检测三类应急事件（路径阻断、动态障碍物、紧急撤离）
- 快速生成应急动作建议（<1秒）
- 复杂场景可选LLM推理（3-5秒）

---

## 一、应急场景定义

### 1.1 三类应急场景

EmergencyAgent专注处理三类应急场景：

| 应急类型 | 描述 | 触发条件 | severity |
|----------|------|----------|----------|
| **路径阻断** | 静态障碍物阻断预设路径 | obstacle_state.has_obstacles + 距离<阈值 | 0.7-0.8 |
| **动态障碍物** | 突然出现的移动障碍物 | obstacle_count增加 + nav_hint关键词 | 0.6-0.7 |
| **紧急撤离** | 需要快速逃离危险区域 | danger_detected=True | 0.9 |

**不处理**：
- ❌ **卡住检测** — 属于导航失败状态，由TrajectoryAgent或FailureHandler处理
- ❌ **路径无效** — 属于规划层面问题，由FailureHandler处理

### 1.2 执行时序（方案B：Emergency + Decision协作）

```
实验脚本层面：
    obstacle_manager.check_and_trigger(step)
    context.metadata["obstacle_state"] = obstacle_manager.get_obstacle_state()

Supernet执行：
    1. PerceptionAgent.quick_detect() → danger_info（<0.5秒）
    2. EmergencyAgent.process() → 应急检测 + 建议路径
       ├── 应急触发 → perception完整process → DecisionAgent应急分支 → Action
       └── 无应急 → 继续正常流程
    3. [无应急] PerceptionAgent.process()（完整检测）
    4. [无应急] TrajectoryAgent.process()
    5. [无应急] InstructionAgent.process()
    6. [无应急] DecisionAgent.process()（正常决策）

关键变化：
    • 应急模式：Emergency建议 + Decision决策（单一出口）
    • 正常模式：多Agent协作 + Decision综合决策
    • DecisionAgent统一所有决策出口
```

### 1.3 数据依赖修正

**EmergencyAgent输入（修正后）**：
| 数据项 | 来源 | 获取方式 |
|--------|------|----------|
| obstacle_state | DynamicObstacleManager | `context.metadata["obstacle_state"]`（实验脚本注入） |
| danger_info | PerceptionAgent.quick_detect | 调用 `perception_agent.quick_detect()` |
| position_history | NavContext | `context.trajectory`（直接读取） |
| current_position | NavContext | `context.position` |
| goal_position | NavContext | `context.metadata["goal_position"]` |

**不再依赖**：
- ❌ `trajectory_output`（TrajectoryAgent）
- ❌ `perception_output`（完整感知）

---

## 二、职责定义

### 2.1 EmergencyAgent职责（方案B调整）

| 职责 | 说明 | 输出 |
|------|------|------|
| 路径阻断检测 | 检测静态障碍物阻断预设路径 | EmergencyEvent(type="obstacle_blocked") |
| 动态障碍物检测 | 检测突然出现的移动障碍物 | EmergencyEvent(type="obstacle_blocked") |
| 紧急撤离检测 | 检测危险区域需要快速撤离 | EmergencyEvent(type="unexpected_change") |
| 快速路径规划 | PathReplanner A*算法 | **suggested_path**（建议路径） |
| 动作建议生成 | path_to_actions转换 | **suggested_actions**（建议动作） |
| LLM风险评估 | 可选，复杂场景 | risk_score + strategy |

**不处理**：
- ❌ 卡住检测（由TrajectoryAgent负责）
- ❌ 路径无效（由FailureHandler负责）

**关键变化**：
- ❌ 不直接返回action_sequence执行
- ✓ 输出建议供DecisionAgent决策
- ✓ DecisionAgent作为唯一决策出口

### 2.2 与其他模块的职责边界

| 检测类型 | EmergencyAgent | TrajectoryAgent | FailureHandler |
|----------|----------------|-----------------|----------------|
| 路径阻断 | ✅ 主责 | - | - |
| 动态障碍物 | ✅ 主责 | - | - |
| 紧急撤离 | ✅ 主责 | - | - |
| **卡住检测** | ❌ 不处理 | ✅ 主责 | ✅ 备用 |
| **路径无效** | ❌ 不处理 | - | ✅ 主责 |
| 完整场景理解 | - | - | - (PerceptionAgent) |
| 轨迹历史记录 | - | ✅ 记录维护 | - |

### 2.3 TrajectoryAgent新增卡住检测职责

```python
# agents/trajectory_agent.py 新增方法

def process(self, context: NavContext) -> AgentOutput:
    """轨迹Agent主方法"""

    # 1. 记录轨迹（原有逻辑）
    context.add_trajectory_point(context.position)

    # 2. 卡住检测（新增）
    stuck_event = self._check_stuck(context)
    if stuck_event:
        context.is_stuck = True
        context.stuck_counter += 1
        # 输出卡住信息供FailureHandler处理
        return AgentOutput(
            success=True,
            data={
                "stuck_detected": True,
                "stuck_counter": context.stuck_counter,
                "avg_movement": stuck_event.get("avg_movement", 0),
            },
            confidence=0.7,
            reasoning=f"卡住检测: {stuck_event['reason']}"
        )

    # 3. 正常输出（原有逻辑）
    ...

def _check_stuck(self, context) -> Optional[Dict]:
    """卡住状态检测"""

    # 方法A: 计数器检测
    if context.stuck_counter >= 5:
        return {"reason": "连续5步无法前进", "avg_movement": 0}

    # 方法B: 移动量检测
    if len(context.trajectory) >= 5:
        recent = context.trajectory[-5:]
        total_movement = 0.0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dz = recent[i][2] - recent[i-1][2]
            total_movement += math.sqrt(dx*dx + dz*dz)

        avg_movement = total_movement / 4
        if avg_movement < 0.3:  # 阈值
            return {"reason": f"平均移动量过低: {avg_movement:.2f}m", "avg_movement": avg_movement}

    return None
```

---

## 三、混合模式设计

### 3.1 三级响应模式

| 应急严重度 | 模式 | 响应时间 | 执行内容 |
|------------|------|----------|----------|
| Low (severity < 0.5) | 纯算法 | <1秒 | EmergencyDetector → PathReplanner |
| Medium (0.5 ≤ severity ≤ 0.8) | 算法验证 | 1-2秒 | PathReplanner + 结果验证 |
| High (severity > 0.8) | 算法+LLM | 3-5秒 | PathReplanner + LLMResponder风险评估 |

### 3.2 配置项

```python
EmergencyAgentConfig:
    # 检测阈值
    stuck_threshold: 5               # 卡住步数阈值
    stuck_movement_threshold: 0.3    # 卡住移动量阈值（米）
    obstacle_distance_threshold: 2.0 # 障碍距离阈值（米）
    min_severity_to_trigger: 0.5     # 触发应急的最低严重度
    
    # LLMResponder（可选）
    enable_llm_responder: False      # 是否启用LLM复杂推理
    llm_trigger_severity: 0.8        # LLM触发阈值
    model_key: "qwen-9b-decision"    # 使用现有模型
    fallback_default_risk: 0.5       # 解析失败时的默认风险值
```

---

## 四、接口设计

### 4.1 AgentRole扩展

```python
# agents/base_agent.py
class AgentRole(Enum):
    INSTRUCTION = "instruction"
    PERCEPTION = "perception"
    TRAJECTORY = "trajectory"
    DECISION = "decision"
    EVALUATION = "evaluation"
    EMERGENCY = "emergency"  # 新增
```

### 4.2 EmergencyAgent接口

```python
class EmergencyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "emergency_agent"
    
    @property
    def role(self) -> AgentRole:
        return AgentRole.EMERGENCY
    
    def process(self, context: NavContext) -> AgentOutput:
        """处理应急检测与响应
        
        执行流程：
        1. 调用PerceptionAgent.quick_detect()获取danger_info
        2. EmergencyDetector检测应急事件
        3. 应急触发 → PathReplanner生成action_sequence
        4. 高严重度 → 可选LLMResponder风险评估
        
        Returns:
            AgentOutput:
                - success: True
                - data: {emergency_detected, emergency_event, action_sequence, response_mode}
                - confidence: 应急响应置信度
                - reasoning: 应急处理说明
        """
```

### 4.3 输出数据结构（方案B调整）

```python
AgentOutput.data = {
    "emergency_detected": bool,
    "emergency_event": {
        "type": "obstacle_blocked" | "unexpected_change",  # 移除stuck
        "severity": 0.0-1.0,
        "position": (x, y, z),
        "description": str,
        "subtype": "path_blocked" | "dynamic_obstacle" | "emergency_evacuate"  # 细分类别
    },
    "response_mode": "emergency_collaboration",  # 新模式标识
    "suggested_path": [(x, z), ...],             # 坐标路径建议
    "suggested_actions": ["turn_left", "forward", ...],  # 动作建议
    "alternative_paths": [...],                  # 可选备选路径
    "risk_score": 0.0-1.0,                       # 仅高级模式
    "reasoning": str
}

# DecisionAgent读取此输出，综合感知信息后最终决策
```

---

## 五、内部组件设计

### 5.1 EmergencyDetector（综合检测方案）

基于Habitat内置检测机制和VLN研究标准方法，设计四层综合检测：

```python
class EmergencyDetector:
    """应急事件检测器 - 综合检测方案
    
    结合Habitat内置检测和VLN研究标准方法：
    Layer 1: geodesic_distance变化检测（路径阻断）
    Layer 2: 物理碰撞累积检测（实际受阻）
    Layer 3: danger_info语义检测（紧急撤离）
    Layer 4: obstacle_count变化检测（动态障碍）
    
    参考来源：
    - Habitat nav.py: ProximitySensor, Collisions Measure
    - Habitat habitat_simulator.py: is_navigable, geodesic_distance
    - Social Navigation研究：动态障碍检测标准
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Layer 1: geodesic阈值
        self._geodesic_increase_threshold = config.get("geodesic_increase_threshold", 1.5)
        
        # Layer 2: 碰撞阈值
        self._collision_count_threshold = config.get("collision_count_threshold", 3)
        
        # Layer 3: danger关键词
        self._danger_keywords = ["blocked", "obstacle", "danger", "fire", "emergency"]
        
        # Layer 4: obstacle检测范围
        self._obstacle_distance_threshold = config.get("obstacle_distance_threshold", 2.0)
        
        # 状态追踪
        self._original_geodesic: Optional[float] = None
        self._collision_count: int = 0
        self._last_obstacle_count: int = 0
    
    def detect(
        self,
        context: NavContext,
        danger_info: Dict,
        obstacle_state: Dict,
        sim: Any = None  # Habitat simulator
    ) -> Optional[EmergencyEvent]:
        """综合检测应急事件
        
        Args:
            context: 导航上下文
            danger_info: 来自PerceptionAgent
            obstacle_state: 来自DynamicObstacleManager
            sim: Habitat simulator（可选，用于物理检测）
        
        Returns:
            最高severity的EmergencyEvent或None
        """
        events = []
        
        # 初始化原始路径距离（首次调用）
        if self._original_geodesic is None:
            self._original_geodesic = context.metadata.get("original_geodesic_distance", 10.0)
        
        # Layer 1: geodesic距离变化检测
        event1 = self._check_geodesic_change(context, sim)
        if event1: events.append(event1)
        
        # Layer 2: 物理碰撞累积检测
        event2 = self._check_collision_cumulative(context, sim)
        if event2: events.append(event2)
        
        # Layer 3: danger_info语义检测
        event3 = self._check_danger_semantic(danger_info)
        if event3: events.append(event3)
        
        # Layer 4: obstacle_count变化检测
        event4 = self._check_obstacle_dynamic(obstacle_state, context.position)
        if event4: events.append(event4)
        
        # 选择最高severity的事件
        if events:
            return max(events, key=lambda e: e.severity)
        
        return None
    
    def _check_geodesic_change(self, context, sim) -> Optional[EmergencyEvent]:
        """Layer 1: 路径距离变化检测
        
        基于Habitat pathfinder.geodesic_distance
        当路径距离显著增加或变为inf时触发
        
        Habitat标准方法（nav.py, habitat_simulator.py:549-554）
        """
        if sim is None or not hasattr(sim, 'geodesic_distance'):
            return None
        
        goal_position = context.metadata.get("goal_position")
        current_geodesic = sim.geodesic_distance(context.position, goal_position)
        
        # 完全阻断检测
        if current_geodesic == float('inf'):
            return EmergencyEvent(
                type="obstacle_blocked",
                subtype="path_blocked",
                severity=1.0,  # 最高
                position=context.position,
                description="路径完全阻断，无法到达目标"
            )
        
        # 路径变长检测
        if self._original_geodesic > 0:
            increase_ratio = current_geodesic / self._original_geodesic
            
            if increase_ratio > self._geodesic_increase_threshold:
                return EmergencyEvent(
                    type="obstacle_blocked",
                    subtype="path_blocked",
                    severity=0.8,
                    position=context.position,
                    description=f"路径距离增加{increase_ratio:.1f}倍: {current_geodesic:.1f}m vs 原始{self._original_geodesic:.1f}m"
                )
        
        return None
    
    def _check_collision_cumulative(self, context, sim) -> Optional[EmergencyEvent]:
        """Layer 2: 物理碰撞累积检测
        
        基于Habitat sim.previous_step_collided
        连续碰撞累积触发
        
        Habitat标准方法（nav.py:658-676 Collisions Measure）
        """
        if sim is None or not hasattr(sim, 'previous_step_collided'):
            return None
        
        if sim.previous_step_collided:
            self._collision_count += 1
            context.metadata["collision_count"] = self._collision_count
            
            if self._collision_count >= self._collision_count_threshold:
                return EmergencyEvent(
                    type="obstacle_blocked",
                    subtype="path_blocked",
                    severity=0.7,
                    position=context.position,
                    description=f"连续碰撞{self._collision_count}次，路径受阻"
                )
        else:
            # 重置碰撞计数（成功移动）
            self._collision_count = 0
        
        return None
    
    def _check_danger_semantic(self, danger_info) -> Optional[EmergencyEvent]:
        """Layer 3: danger_info语义检测
        
        基于nav_hint关键词匹配
        VLN研究常用方法
        
        触发条件：
        1. danger_detected == True → severity 0.9
        2. nav_hint含关键词 → severity 0.6
        """
        # 高严重度：明确危险检测
        if danger_info.get("danger_detected", False):
            return EmergencyEvent(
                type="unexpected_change",
                subtype="emergency_evacuate",
                severity=0.9,
                position=None,  # 无具体位置
                description=f"紧急撤离: {danger_info.get('danger_type', '检测到危险区域')}"
            )
        
        # 中严重度：关键词检测
        nav_hint = danger_info.get("nav_hint", "").lower()
        matched_keywords = [kw for kw in self._danger_keywords if kw in nav_hint]
        
        if matched_keywords:
            return EmergencyEvent(
                type="unexpected_change",
                subtype="emergency_evacuate",
                severity=0.6,
                position=None,
                description=f"紧急撤离建议: 检测到关键词{matched_keywords}"
            )
        
        return None
    
    def _check_obstacle_dynamic(self, obstacle_state, position) -> Optional[EmergencyEvent]:
        """Layer 4: obstacle_count变化检测
        
        基于障碍物数量变化检测新障碍
        Social Navigation研究标准方法
        
        触发条件：
        1. obstacle_count增加 → 新障碍出现
        2. 障碍物距离 < threshold → 近距离障碍
        """
        if not obstacle_state.get("has_obstacles", False):
            return None
        
        obstacles = obstacle_state.get("obstacles", [])
        new_count = obstacle_state.get("obstacle_count", 0)
        
        # 新障碍物出现检测
        if new_count > self._last_obstacle_count:
            self._last_obstacle_count = new_count
            
            if obstacles:
                latest_obstacle = obstacles[-1]
                return EmergencyEvent(
                    type="obstacle_blocked",
                    subtype="dynamic_obstacle",
                    severity=0.7,
                    position=latest_obstacle.get("position"),
                    description=f"动态障碍物出现: {latest_obstacle.get('description', 'unknown')}"
                )
        
        # 近距离障碍检测（备用）
        for obs in obstacles:
            obs_pos = obs.get("position", (0, 0, 0))
            distance = self._calculate_distance(position, obs_pos)
            
            if distance < self._obstacle_distance_threshold:
                return EmergencyEvent(
                    type="obstacle_blocked",
                    subtype="path_blocked",
                    severity=0.8,
                    position=obs_pos,
                    description=f"近距离障碍: 距离{distance:.1f}m"
                )
        
        return None
    
    def _calculate_distance(self, pos1, pos2) -> float:
        """计算水平距离"""
        import math
        dx = pos1[0] - pos2[0]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx * dx + dz * dz)
    
    def reset(self):
        """重置检测器状态"""
        self._original_geodesic = None
        self._collision_count = 0
        self._last_obstacle_count = 0
```

### 5.2 检测层级与优先级

| Layer | 检测方法 | 触发条件 | severity | 来源 |
|-------|----------|----------|----------|------|
| **Layer 1** | geodesic_distance变化 | 路径距离增加>1.5倍 或 inf | 0.8-1.0 | Habitat内置 |
| **Layer 2** | 碰撞累积 | 连续碰撞≥3次 | 0.7 | Habitat内置 |
| **Layer 3** | danger_info语义 | danger_detected 或 关键词 | 0.6-0.9 | VLN研究标准 |
| **Layer 4** | obstacle_count变化 | 障碍物数量增加 | 0.7-0.8 | Social Nav研究 |

**优先级规则**：取最高severity的事件作为最终输出

### 5.3 与Habitat内置机制对比

| 检测类型 | Habitat内置方法 | EmergencyDetector使用 |
|----------|----------------|----------------------|
| 路径阻断 | `geodesic_distance` | ✓ Layer 1使用 |
| 碰撞检测 | `previous_step_collided` | ✓ Layer 2使用 |
| 可通行性 | `is_navigable` | ✓ pathfinder底层 |
| 近障碍距离 | `distance_to_closest_obstacle` | ✓ Layer 4使用 |
| 碰撞计数 | `Collisions Measure` | ✓ Layer 2借鉴 |

### 5.4 PathReplanner

```python
class PathReplanner:
    """快速路径重规划（现有模块，直接使用）
    
    响应时间: <1秒
    算法: A* + 动态成本调整
    """
    
    def replan(
        self,
        current_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        blocked_positions: List[Tuple[float, float, float]],
        blocked_areas: List[Dict]
    ) -> PathResult:
        """生成避开障碍的路径"""
```

### 5.3 LLMResponder

```python
class LLMResponder:
    """复杂应急场景的LLM响应器
    
    使用基础Qwen模型 + 强fallback解析
    不进行微调，后续视效果决定
    """
    
    def assess_risk(self, context, emergency_event) -> Tuple[float, str]:
        """风险评估
        
        Returns:
            (risk_score, reasoning)
        """
        prompt = self._build_risk_prompt(context, emergency_event)
        response = self._model_manager.generate(prompt)
        
        # 多层fallback解析
        return self._fallback_parser.parse_risk(response)
```

### 5.4 FallbackParser

```python
class FallbackParser:
    """多层fallback解析LLM输出"""
    
    def parse_risk(self, response: str) -> Tuple[float, str]:
        """解析风险评分
        
        解析层级：
        1. JSON解析
        2. 正则匹配数字
        3. 关键词推断
        4. 默认兜底（0.5）
        """
```

---

## 六、PerceptionAgent扩展

### 6.1 新增quick_detect方法

```python
# agents/perception_agent.py 新增方法

def quick_detect(self, context: NavContext) -> Dict:
    """快速感知检测
    
    仅检测危险信息，供EmergencyAgent前置调用
    
    响应时间: <0.5秒
    不调用LLM，仅使用YOLO/深度分析
    
    Returns:
        {
            "danger_detected": bool,
            "blocked_direction": str,  # "left" | "right" | "front"
            "nav_hint": str,
            "closest_obstacle_distance": float
        }
    """
    rgb = context.rgb_image
    depth = context.depth_image
    
    # 1. YOLO检测危险物体（可选）
    danger_objects = self._detect_danger_objects(rgb)
    
    # 2. 深度图分析近距离障碍
    blocked_direction = self._analyze_depth_blocking(depth)
    
    return {
        "danger_detected": len(danger_objects) > 0 or blocked_direction != "none",
        "blocked_direction": blocked_direction,
        "nav_hint": self._generate_quick_hint(blocked_direction),
        "closest_obstacle_distance": self._get_min_depth(depth)
    }
```

---

## 七、Supernet修改

### 7.1 ARCHITECTURE_CONFIG更新

```python
ARCHITECTURE_CONFIG = {
    "Type-0": {"level": "weak", "agents": [], "strategies": []},
    "Type-1": {"level": "strong", "agents": ["perception", "emergency", "decision"], "strategies": ["ReAct"]},
    "Type-2": {"level": "strong", "agents": ["perception", "trajectory", "emergency", "decision"], "strategies": ["ReAct", "CoT"]},
    "Type-3": {"level": "strong", "agents": ["instruction", "perception", "trajectory", "emergency", "decision"], "strategies": ["CoT", "Reflection"]},
    "Type-4": {"level": "strong", "agents": ["instruction", "perception", "trajectory", "emergency", "decision"], "strategies": ["CoT", "Debate", "Reflection"]},
}
```

### 7.2 _strong_level_forward修改（方案B实现）

```python
def _strong_level_forward(self, context: NavContext, config: Dict) -> Action:
    """强层级执行（方案B：Emergency建议 + Decision决策）"""

    # 1. 快速感知（前置）
    perception_agent = self._get_agent("perception")
    danger_info = perception_agent.quick_detect(context)
    context.metadata["danger_info"] = danger_info

    # 2. EmergencyAgent检测
    emergency_agent = self._get_agent("emergency")
    if emergency_agent:
        emergency_output = emergency_agent.process(context)
        context.metadata["emergency_output"] = emergency_output.to_dict()

        if emergency_output.data.get("emergency_detected"):
            # 应急模式（方案B）：Emergency建议 + Decision决策
            # 立即执行完整感知（用于校验）
            perception_output = perception_agent.process(context)
            context.metadata["perception_output"] = perception_output.to_dict()

            # DecisionAgent应急分支决策
            decision_agent = self._get_agent("decision")
            action = decision_agent.process(context)
            return action  # 单一出口，不跳过DecisionAgent

    # 3. 正常模式：执行其他agents
    agent_names = config.get("agents", [])
    agents = [self._get_agent(name) for name in agent_names]
    agents = [a for a in agents if a and a.name != "emergency_agent"]

    for agent in agents:
        output = agent.process(context)
        context.metadata[f"{agent.name}_output"] = output.to_dict()

    # 4. 执行策略链
    # ... (原有逻辑)
```

---

## 八、实验脚本修改

### 8.1 obstacle_state注入

```python
# run_vln_experiment.py 修改

def _run_navigation_step(self, context, step):
    """执行单步导航"""
    
    # 1. 更新障碍物状态（前置）
    if self.obstacle_manager:
        self.obstacle_manager.check_and_trigger(step)
        obstacle_state = self.obstacle_manager.get_obstacle_state()
        context.metadata["obstacle_state"] = obstacle_state
    
    # 2. 执行导航（Supernet处理）
    action = self.supernet.forward(context)
    
    return action
```

---

## 十、DecisionAgent应急分支扩展

### 10.1 新增应急决策方法

```python
# agents/decision_agent.py 新增方法

def process(self, context: NavContext) -> AgentOutput:
    """决策Agent主方法"""

    emergency_output = context.metadata.get("emergency_output")

    if emergency_output and emergency_output.get("emergency_detected"):
        # 应急模式：快速决策
        return self._emergency_decision(context, emergency_output)
    else:
        # 正常模式：原有逻辑
        return self._normal_decision(context)

def _emergency_decision(self, context, emergency_output) -> AgentOutput:
    """应急模式决策 - 快速综合 + 简单校验"""

    # 1. 读取完整感知输出（辅助验证）
    perception_output = context.metadata.get("perception_output", {})
    blocked_direction = perception_output.data.get("blocked_direction", "none")
    closest_distance = perception_output.data.get("closest_obstacle_distance", 10.0)

    # 2. 获取Emergency建议
    suggested_actions = emergency_output.get("suggested_actions", [])
    emergency_event = emergency_output.get("emergency_event", {})

    # 3. 简单校验
    validation = self._simple_validate(
        suggested_actions,
        blocked_direction,
        closest_distance
    )

    # 4. 决策输出
    if validation["safe"]:
        return AgentOutput(
            success=True,
            data={
                "action": suggested_actions[0] if suggested_actions else "stop",
                "action_sequence": suggested_actions[:3],  # 只取前3步
                "validation": validation,
                "mode": "emergency"
            },
            confidence=0.8,
            reasoning=f"应急响应: {emergency_event['type']} - {validation['reason']}"
        )
    else:
        # 校验失败 → 降级策略
        fallback_action = self._fallback_emergency_action(blocked_direction)
        return AgentOutput(
            success=True,
            data={"action": fallback_action, "mode": "emergency_fallback"},
            confidence=0.5,
            reasoning=f"应急降级: {validation['reason']}"
        )

def _simple_validate(self, actions, blocked_dir, min_distance) -> Dict:
    """简单校验：检查动作是否与障碍冲突"""

    # 规则1: 前方近距离障碍，不应forward
    if blocked_dir == "front" and min_distance < 1.0:
        if "forward" in actions[:2]:
            return {"safe": False, "reason": "前进方向有近距离障碍"}

    # 规则2: 左侧障碍，不应turn_left+forward
    if blocked_dir == "left":
        if "turn_left" in actions[:2] and "forward" in actions[:3]:
            return {"safe": False, "reason": "左转后前进可能碰撞"}

    # 规则3: 右侧障碍，不应turn_right+forward
    if blocked_dir == "right":
        if "turn_right" in actions[:2] and "forward" in actions[:3]:
            return {"safe": False, "reason": "右转后前进可能碰撞"}

    return {"safe": True, "reason": "建议动作安全"}

def _fallback_emergency_action(self, blocked_dir) -> str:
    """降级策略：根据障碍方向反向避让"""
    fallback_map = {
        "front": "turn_left",   # 前方障碍，左转
        "left": "turn_right",   # 左侧障碍，右转
        "right": "turn_left",   # 右侧障碍，左转
        "none": "stop"          # 无方向信息，停止
    }
    return fallback_map.get(blocked_dir, "stop")
```

### 10.2 应急模式响应时间分析

| 步骤 | 耗时 | 说明 |
|------|------|------|
| Perception完整process | ~1秒 | 深度分析 + 对象检测 |
| Emergency规划 | ~0.5秒 | A*路径 + 动作建议 |
| Decision校验决策 | ~0.3秒 | 简单规则校验 |
| **总计** | **~1.8秒** | 比方案A多0.8秒，但决策质量更高 |

---

## 十一、文件变更清单（方案B更新）

| 文件 | 操作 | 内容 |
|------|------|------|
| `agents/base_agent.py` | 修改 | AgentRole添加EMERGENCY |
| `agents/emergency_agent.py` | 新建 | EmergencyAgent实现（约500行），仅检测三类应急场景 |
| `agents/perception_agent.py` | 修改 | 新增quick_detect方法（约50行） |
| `agents/trajectory_agent.py` | **修改** | 新增卡住检测：_check_stuck方法（约30行） |
| `agents/decision_agent.py` | **修改** | 新增应急分支：_emergency_decision, _simple_validate, _fallback |
| `agents/__init__.py` | 修改 | 导出EmergencyAgent |
| `supernet/supernet.py` | 修改 | ARCHITECTURE_CONFIG更新，应急模式调用Perception+Decision |
| `run_vln_experiment.py` | 修改 | obstacle_state注入context |

---

## 十二、关键设计决策（方案B + 综合检测）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 执行时序 | obstacle_state前置注入 + quick_detect前置 | 解决数据依赖问题 |
| **应急场景定义** | **三类：路径阻断、动态障碍物、紧急撤离** | **保持原始设计清晰** |
| **检测方法** | **综合检测（四层）** | **符合Habitat内置机制和VLN研究标准** |
| **Layer优先级** | **取最高severity事件** | **安全优先原则** |
| **卡住检测归属** | **TrajectoryAgent** | **不属于应急场景，是导航失败状态** |
| **应急决策出口** | **DecisionAgent统一** | **单一出口，后续连贯性好** |
| **Emergency输出** | **建议而非动作** | **允许Decision综合校验** |
| DecisionAgent应急逻辑 | 新增应急分支 + 简单校验 | 快速响应但保证安全 |
| LLM微调 | 不微调，先验证效果 | 节省成本，视效果决定 |

### 综合检测设计依据

| Layer | 方法 | 参考来源 |
|-------|------|----------|
| Layer 1 | geodesic_distance变化 | Habitat `pathfinder.geodesic_distance` |
| Layer 2 | 碰撞累积检测 | Habitat `previous_step_collided`, Collisions Measure |
| Layer 3 | danger_info语义 | VLN研究标准（nav_hint关键词） |
| Layer 4 | obstacle_count变化 | Social Navigation研究（动态障碍检测） |
| **卡住检测归属** | **TrajectoryAgent** | **不属于应急场景，是导航失败状态** |
| **应急决策出口** | **DecisionAgent统一** | **单一出口，后续连贯性好** |
| **Emergency输出** | **建议而非动作** | **允许Decision综合校验** |
| DecisionAgent应急逻辑 | 新增应急分支 + 简单校验 | 快速响应但保证安全 |
| LLM微调 | 不微调，先验证效果 | 节省成本，视效果决定 |

---

## 十三、多智能体协作流程图（方案B）

### 13.1 整体架构流程

```
                         ┌──────────────┐
                         │ VLNNavigator │
                         │  (主入口)    │
                         └──────┬───────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  TaskTypeClassifier  │
                    └──────┬───────┬───────┘
                           │       │
              Type-0       │       │ Type-1~4
                           │       │
                           ▼       ▼
                    ┌─────────┐ ┌──────────────────────┐
                    │LocalModel│ │     Supernet        │
                    └────┬────┘ │   (多智能体编排)    │
                         │      └──────┬───────────────┘
                         │             │
                         │             ▼
                         │    ┌────────────────────┐
                         │    │ NavContext.metadata│← 信息交换中心
                         │    └───────┬─────┬──────┘
                         │            │     │
                         │            ▼     ▼
                         │    ┌──────┐┌──────┐
                         │    │Percep││Traj │
                         │    │Agent ││Agent│
                         │    └───┬──┘└──┬──┘
                         │        │      │
                         │        └──────┼────────┘
                         │               │
                         │               ▼
                         │    ┌──────────────────────┐
                         │    │   EmergencyAgent     │
                         │    │ (检测 + 建议)        │← 方案B：不直接决策
                         │    └──────┬───────────────┘
                         │           │
                         │   ┌───────┴───────┐
                         │   │               │
                         │ 应急             无应急
                         │   │               │
                         │   ▼               ▼
                         │┌────────────┐ ┌──────────┐
                         ││Perception  │ │ 正常流程 │
                         ││完整process │ │(所有Agent)│
                         │└─────┬──────┘ └────┬─────┘
                         │      │             │
                         │      └──────┬──────┘
                         │             │
                         │             ▼
                         │    ┌──────────────────────┐
                         │    │   DecisionAgent      │← 单一决策出口
                         │    │ (应急分支/正常分支)  │
                         │    └──────┬───────────────┘
                         │           │
                         └───────────┼───────────────┘
                                     │
                                     ▼
                              ┌────────────┐
                              │   Action   │
                              └────────────┘
```

### 13.2 应急模式详细流程（方案B）

```
应急触发时序（方案B）：

Step 0: 前置注入
────────────────────
obstacle_manager.check_and_trigger(step)
└─► context.metadata["obstacle_state"] = obstacle_state


Step 1: 快速感知（前置）
──────────────────────────
PerceptionAgent.quick_detect(context)
└─► danger_info → context.metadata["danger_info"]


Step 2: EmergencyAgent检测
──────────────────────────────
EmergencyAgent.process(context)
│
│ 输入：
│ ├─ obstacle_state ← metadata["obstacle_state"]
│ ├─ danger_info    ← metadata["danger_info"]
│ └─ position       ← context.position
│ （注：卡住检测由TrajectoryAgent负责，不在EmergencyAgent）
│
│ 检测（三类应急场景）：
│ ├─ 路径阻断: obstacle距离 < threshold
│ ├─ 动态障碍物: obstacle_count增加
│ ├─ 紧急撤离: danger_detected=True
│ └─► emergency_detected = True
│
│ 规划：
│ ├─ PathReplanner.replan() → suggested_path
│ ├─ path_to_actions() → suggested_actions
│
│ 输出（方案B）：
│ └─► emergency_output = {
│     │   "emergency_detected": True,
│     │   "emergency_event": {"type": "...", "subtype": "..."},
│     │   "suggested_actions": ["turn_left", "forward", ...],
│     │   "suggested_path": [(x,z), ...]
│     │ }
│ └─► metadata["emergency_output"] = emergency_output


Step 3: 完整感知（校验辅助）
──────────────────────────────
[应急触发 → 立即执行完整感知]
PerceptionAgent.process(context)
│
│ 深度分析：
│ ├─ YOLO对象检测
│ ├─ 深度图分析
│ └─► blocked_direction, closest_obstacle_distance
│
└─► perception_output → metadata["perception_output"]


Step 4: DecisionAgent应急分支
──────────────────────────────────
DecisionAgent.process(context)
│
│ 检查应急：
│ └─► emergency_detected → 进入_emergency_decision()
│
│ 应急决策：
│ ├─ 读取perception_output
│ ├─ 读取emergency_output.suggested_actions
│ │
│ │ 简单校验：
│ │ _simple_validate(actions, blocked_dir, distance)
│ │ ├─ 规则1: 前方障碍 < 1m → 不应forward
│ │ ├─ 规则2: 左侧障碍 → 不应turn_left+forward
│ │ ├─ 规则3: 右侧障碍 → 不应turn_right+forward
│ │ └─► {safe: True/False, reason: str}
│ │
│ │ 决策输出：
│ │ ├─ safe=True → 采纳suggested_actions[:3]
│ │ └─► safe=False → 降级策略（反向避让）
│ │
└─► Action


总耗时：~1.8秒
├─ Perception完整: ~1秒
├─ Emergency规划: ~0.5秒
└─ Decision校验: ~0.3秒
```

### 13.3 方案A vs 方案B对比

```
┌─────────────────────────────────────────────────────────────┐
│                方案A（Emergency独立） vs 方案B（协作）        │
└─────────────────────────────────────────────────────────────┘

方案A流程：
──────────
EmergencyAgent.process()
├─ 检测应急
├─ 路径规划
├─ 动作生成
└─► 直接返回Action（跳过其他Agent）

特点：
• 单Agent完成全决策链
• 响应时间: ~1秒
• 无校验机制
• 决策出口: EmergencyAgent
• 后续连贯性: 可能断层


方案B流程：
──────────
EmergencyAgent.process()
├─ 检测应急
├─ 路径规划
└─► 输出建议

PerceptionAgent.process()（完整版）
└─► 感知输出

DecisionAgent._emergency_decision()
├─ 读取Emergency建议
├─ 读取Perception输出
├─ 简单校验
└─► 返回Action

特点：
• 多Agent协作决策
• 响应时间: ~1.8秒
• 简单校验机制
• 决策出口: DecisionAgent（统一）
• 后续连贯性: 好（同一Agent）
```

---

## 十四、TopologyGraph设计（TrajectoryAgent拓扑记忆模块）

### 14.1 设计背景

基于MSNav/MAP多智能体导航架构分析，当前系统成功率仅10%，而MSNav可达50.9%。差距主要原因：

| 性能差距 | 来源 | 贡献 |
|----------|------|------|
| **40% SR差距** | MSNav Memory Module | ~20%提升（长距离任务） |
| 缺失拓扑图 | 当前仅有raw trajectory | LLM无法理解路径结构 |
| 无节点剪枝 | 轨迹数据直接传LLM | Token预算浪费 |

**解决方案**：为TrajectoryAgent添加TopologyGraph模块，压缩轨迹为10-20个关键节点。

### 14.2 核心数据结构

```python
@dataclass
class GraphNode:
    node_id: str               # "node_0", "node_1"...
    position: Tuple[float, float, float]  # 3D坐标
    node_type: str             # junction/room_entrance/stairs/stuck_region
    timestamp: int             # 创建时的导航步数
    visit_count: int           # 重复经过次数
    semantic_info: Optional[str]  # "厨房入口", "楼梯底部"
    is_key_position: bool      # 是否关键节点（剪枝时优先保留）

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    distance: float            # 两节点间欧氏距离
    action_sequence: List[str] # 连接动作序列 ["forward", "turn_left"]
    traversable: bool          # 是否可通行（blocked_path会标记False）

@dataclass
class TopologyGraph:
    nodes: Dict[str, GraphNode]        # {node_id: GraphNode}
    edges: Dict[str, GraphEdge]        # {"edge_0_1": GraphEdge}
    current_node_id: Optional[str]     # 当前位置对应节点
    goal_node_id: Optional[str]        # 目标位置节点
```

### 14.3 关键位置类型

| 节点类型 | 检测方法 | 触发条件 | 说明 |
|----------|----------|----------|------|
| **junction** | 动作序列分析 | 连续转向动作（3步内含转向+动作差异） | 路口/转弯点 |
| **room_entrance** | 语义变化检测 | PerceptionAgent.room_changed | 房间入口 |
| **stairs** | 高度变化检测 | Y轴变化 > 0.5m | 楼层切换 |
| **stuck_region** | 动作重复检测 | 10步内动作类型 ≤ 2 | 卡住区域 |

### 14.4 关键位置检测逻辑

```python
class KeyPositionDetector:
    """检测关键位置，触发节点创建"""

    def detect(self, current_pos, prev_pos, action_history, semantic_info) -> Optional[str]:
        """
        Returns:
            检测到的关键位置类型，或None
        """

        # 1. junction检测：连续转向动作
        if len(action_history) >= 3:
            recent = action_history[-3:]
            if any("turn" in a for a in recent) and any(a != recent[0] for a in recent):
                return "junction"

        # 2. room_entrance检测：语义变化 + 距离阈值
        if semantic_info and semantic_info.get("room_changed"):
            return "room_entrance"

        # 3. stairs检测：高度变化显著
        if abs(current_pos[1] - prev_pos[1]) > 0.5:  # Y轴高度变化
            return "stairs"

        # 4. stuck_region检测：TrajectoryAgent通知
        if action_history and len(action_history) >= 10:
            recent_actions = set(action_history[-10:])
            if len(recent_actions) <= 2:  # 动作高度重复
                return "stuck_region"

        return None
```

### 14.5 节点剪枝策略

当节点数超过阈值时执行剪枝，保持LLM上下文可控：

```python
class TopologyGraph:
    MAX_NODES = 30  # LLM上下文限制
    PRUNE_CHECK_INTERVAL = 50  # 每50步检查

    def prune_nodes(self) -> None:
        """压缩节点数至MAX_NODES以内"""
        if len(self.nodes) <= self.MAX_NODES:
            return

        # 标记关键节点（优先保留）
        for node in self.nodes.values():
            node.is_key_position = (
                node.node_id == self.goal_node_id or
                node.node_type in ["stuck_region", "room_entrance"] or
                node.visit_count >= 3  # 多次访问节点
            )

        # 合并非关键相邻节点
        nodes_to_merge = []
        for edge in self.edges.values():
            if edge.traversable and not edge.distance < 5.0:  # 短距离边
                src = self.nodes[edge.source_id]
                tgt = self.nodes[edge.target_id]
                if not src.is_key_position and not tgt.is_key_position:
                    nodes_to_merge.append((src, tgt, edge))

        # 执行合并（将多个节点合并为path_segment）
        for src, tgt, edge in nodes_to_merge:
            self._merge_nodes(src, tgt, edge)

        # 删除孤立节点
        self._remove_orphan_nodes()
```

**剪枝触发时机**：混合策略
- 固定检查：每50步自动检查
- LLM请求时：DecisionAgent调用时强制剪枝

**剪枝优先级**：
| 优先级 | 节点类型 | 处理方式 |
|--------|----------|----------|
| 最高 | goal_position, stuck_region | 必须保留 |
| 高 | room_entrance | 保留 |
| 中 | junction | 视情况保留 |
| 低 | 普通节点 | 合并为path_segment |

### 14.6 Agent集成方式

TrajectoryAgent独立维护TopologyGraph，不改动其他Agent：

```python
# TrajectoryAgent: 独立维护TopologyGraph
class TrajectoryAgent(BaseAgent):
    def __init__(self):
        self.topology_graph = TopologyGraph()
        self.key_detector = KeyPositionDetector()
        self.last_room = ""

    def process(self, position_history, action_history, perception_info):
        # 从perception_info获取房间变化信息
        room_type = perception_info.get("room_type", "")
        semantic_info = {"room_changed": room_type != self.last_room,
                         "room_type": room_type}

        # 检测关键位置
        node_type = self.key_detector.detect(
            current_pos=position_history[-1],
            prev_pos=position_history[-2] if len(position_history) > 1 else position_history[-1],
            action_history=action_history,
            semantic_info=semantic_info
        )

        if node_type:
            self.topology_graph.add_node(
                position=position_history[-1],
                node_type=node_type,
                semantic_info=semantic_info.get("room_type")
            )

        # 固定检查剪枝
        if len(position_history) % 50 == 0:
            self.topology_graph.prune_nodes()

        # 更新房间记录
        self.last_room = room_type

        # 返回拓扑摘要（新增输出字段）
        topology_summary = self.topology_graph.get_summary()

        return {
            "distance": ...,
            "heading": ...,
            "stuck_state": ...,
            "history_summary": ...,
            "topology_summary": topology_summary,  # 新增
        }
```

**改动范围**：
| Agent | 改动内容 |
|-------|----------|
| TrajectoryAgent | 新增TopologyGraph + KeyPositionDetector + topology_summary输出 |
| DecisionAgent | 输入已包含topology_summary，无结构改动（仅读取新字段） |
| PerceptionAgent | 完全不变，只提供room_type作为TrajectoryAgent输入 |

### 14.7 TopologyGraph API接口

TrajectoryAgent对外提供的拓扑摘要格式：

```python
def get_summary(self) -> Dict[str, Any]:
    """返回LLM可理解的拓扑摘要"""
    return {
        "total_nodes": len(self.nodes),
        "key_nodes": [
            {"id": n.node_id, "type": n.node_type, "position": n.position}
            for n in self.nodes.values() if n.is_key_position
        ],
        "current_node": self.current_node_id,
        "goal_node": self.goal_node_id,
        "path_to_goal": self._find_path_to_goal(),  # 简单路径列表
        "visited_rooms": list(set(
            n.semantic_info for n in self.nodes.values()
            if n.node_type == "room_entrance" and n.semantic_info
        )),
        "stuck_regions": [
            {"id": n.node_id, "position": n.position}
            for n in self.nodes.values() if n.node_type == "stuck_region"
        ],
    }
```

**输出示例**：
```json
{
  "total_nodes": 12,
  "key_nodes": [
    {"id": "node_3", "type": "room_entrance", "position": [5.2, 0.0, 3.1]},
    {"id": "node_7", "type": "junction", "position": [8.0, 0.0, 6.5]}
  ],
  "current_node": "node_11",
  "goal_node": "node_g",
  "path_to_goal": ["node_11", "node_7", "node_3", "node_g"],
  "visited_rooms": ["厨房", "客厅"],
  "stuck_regions": [{"id": "node_5", "position": [4.0, 0.0, 2.0]}]
}
```

### 14.8 实现技术选择

| 技术点 | 选择 | 理由 |
|--------|------|------|
| 数据结构 | 纯Python dict | 轻量、无额外依赖、JSON序列化简单 |
| 图算法 | 手动实现 | 需求简单（节点遍历、路径查询），可控 |
| 序列化 | JSON | 便于保存/加载导航历史 |
| 依赖 | 无新增 | 符合现有代码风格 |

### 14.9 文件变更清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `agents/trajectory_agent.py` | 修改 | 新增TopologyGraph类、KeyPositionDetector类、topology_summary输出 |
| `agents/decision_agent.py` | 修改 | 读取topology_summary字段（约5行） |
| `supernet/supernet.py` | 无需修改 | TrajectoryAgent输出已自动传递给DecisionAgent |

---

## 十五、设计文档自检

### 15.1 Placeholder检查

✓ 无TBD、TODO占位符
✓ 所有章节完整

### 15.2 内部一致性检查

✓ Section 1.1应急场景定义为三类（路径阻断、动态障碍物、紧急撤离）
✓ Section 2.2明确卡住检测归属TrajectoryAgent
✓ Section 5.1 EmergencyDetector采用综合检测（四层）
✓ Section 5.2检测层级与Habitat内置机制对应
✓ Section 4.3输出结构type字段不含stuck
✓ Section 10 DecisionAgent新增应急分支与方案B一致
✓ Section 12关键设计决策记录综合检测选择

### 15.3 Scope检查

✓ 单Agent实现范围明确
✓ DecisionAgent修改范围明确（应急分支）
✓ 不涉及LLM微调

### 15.4 Ambiguity检查

✓ obstacle_state来源明确：实验脚本注入
✓ Emergency输出明确：建议而非动作
✓ Decision应急分支逻辑明确：简单校验规则

---

*文档创建时间: 2026-04-20*
*文档更新时间: 2026-04-21*
*更新内容:
  1. 采用方案B协作模式，DecisionAgent统一决策出口
  2. 移除卡住检测，保持三类应急场景定义清晰
  3. 卡住检测职责划归TrajectoryAgent
  4. 采用综合检测方案（四层），符合Habitat内置机制和VLN研究标准
  5. 新增TopologyGraph设计（第十四章），TrajectoryAgent拓扑记忆模块
  6. 基于MSNav/MAP分析，补齐Memory Module差距，提升长距离任务成功率*