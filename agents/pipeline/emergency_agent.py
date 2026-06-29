"""EmergencyAgent - LLM核心，障碍检测+处理+撤离。

职责：
- 应急检测与处理
- LLM角色：核心 - LLM判断绕行方向、撤离路径

触发条件：
| 触发源 | 条件 |
|-------|------|
| 环境触发 | 碰撞检测、路径受阻（ObservationAgent提示） |
| 用户触发 | 紧急撤离指令 |

处理策略（LLM决策）：
| 应急类型 | LLM决策 |
|---------|---------|
| obstacle | 绕行方向（左/右）或回退 |
| evacuate | 撤离路径规划 |
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from agents.pipeline.base_pipeline_agent import SubAgent, EmergencyEvent, ObservationOutput
from agents.base_agent import AgentRole


class EmergencyAgent(SubAgent):
    """应急Agent - LLM核心。

    应急检测与处理Agent，负责：
    1. 规则检测应急事件（碰撞、路径受阻、用户撤离指令）
    2. LLM评估严重程度
    3. LLM决策绕行/撤离方案

    优先级：
    - 用户撤离指令 > 碰撞检测 > 路径受阻
    """

    name = "emergency_agent"

    # Default fallback actions for different scenarios
    DEFAULT_OBSTACLE_ACTIONS = ["turn_left", "forward", "forward"]
    DEFAULT_EVACUATE_ACTIONS = ["turn_left", "turn_left", "forward", "forward", "forward"]

    def _get_smart_escape_actions(self, depth_info: Dict[str, Any]) -> List[str]:
        """Generate smart escape actions based on depth clearance analysis.

        Turns toward the side with more open space, not blindly left.

        Args:
            depth_info: Dict from Navigator._check_depth_obstacle() with
                       escape_direction, left/right_clearance, center_depth

        Returns:
            Action sequence list
        """
        direction = depth_info.get("escape_direction", "left")
        left = depth_info.get("left_clearance", 0)
        right = depth_info.get("right_clearance", 0)

        if direction == "right":
            self.logger.info(f"[Emergency] Smart escape: turning RIGHT (L={left:.1f}m, R={right:.1f}m)")
            return ["turn_right", "forward", "forward"]
        else:
            self.logger.info(f"[Emergency] Smart escape: turning LEFT (L={left:.1f}m, R={right:.1f}m)")
            return ["turn_left", "forward", "forward"]

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize EmergencyAgent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("EmergencyAgent")

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def process(
        self,
        context: dict,
    ) -> Optional[EmergencyEvent]:
        """检测应急事件。

        Args:
            context: 包含 observation, history, collision_status, user_command 等

        Returns:
            EmergencyEvent 或 None（无应急）
        """
        # 规则检测应急
        event = self._detect_by_rules(context)

        if event is None:
            return None

        # LLM评估严重程度
        severity = self._assess_severity(event, context)
        event.severity = severity

        self.logger.info(f"[EmergencyAgent] Detected emergency: {event.type}, severity: {event.severity}")

        return event

    def handle(
        self,
        event: EmergencyEvent,
        context: dict,
    ) -> List[str]:
        """处理应急事件。

        Args:
            event: EmergencyEvent
            context: 上下文信息

        Returns:
            动作序列 ["turn_left", "forward", ...]
        """
        if event.type == "obstacle":
            return self._handle_obstacle(event, context)
        elif event.type == "evacuate":
            return self._handle_evacuate(event, context)

        # Unknown event type
        self.logger.warning(f"[EmergencyAgent] Unknown event type: {event.type}")
        return []

    def _detect_by_rules(self, context: dict) -> Optional[EmergencyEvent]:
        """规则检测应急事件。

        检测优先级：
        1. 用户紧急撤离指令（最高优先级）
        2. 碰撞检测
        3. 路径受阻（ObservationAgent提示）

        Args:
            context: 包含各种触发源信息

        Returns:
            EmergencyEvent 或 None
        """
        user_command = context.get("user_command")
        collision = context.get("collision_status", False)
        observation = context.get("observation")

        # 1. 用户紧急撤离指令（最高优先级）
        if user_command and user_command.get("type") == "emergency":
            self.logger.warning("[EmergencyAgent] User emergency evacuate command detected")
            return EmergencyEvent(
                type="evacuate",
                severity="high",  # User command is always high priority
                details={"command": user_command, "source": "user"},
            )

        # 2. 碰撞检测（需要实际碰撞才触发）
        if collision:
            self.logger.warning("[EmergencyAgent] Collision detected")
            return EmergencyEvent(
                type="obstacle",
                severity="medium",  # Initial severity, will be assessed by LLM
                details={"collision": True, "source": "collision"},
            )

        # 3. 路径受阻检测 - 深度图规则 + VLM交叉验证
        # 深度图检测：前方区域深度值持续低于阈值 → 有障碍物
        # 结合VLM的path_blocked信号（需连续两帧确认，避免地毯误判）
        depth_blocked = context.get("depth_blocked", False)
        vlm_blocked = False
        if observation and isinstance(observation, ObservationOutput):
            vlm_blocked = observation.path_blocked

        # VLM path_blocked needs 2 consecutive confirmations to rule out false positives
        if vlm_blocked:
            if not hasattr(self, '_vlm_blocked_count'):
                self._vlm_blocked_count = 0
            self._vlm_blocked_count += 1
            if self._vlm_blocked_count < 2:
                self.logger.debug("[EmergencyAgent] VLM path_blocked: waiting for confirmation")
                vlm_blocked = False  # Not confirmed yet
        else:
            self._vlm_blocked_count = 0

        # Trigger if depth confirms obstacle OR VLM confirms twice
        if depth_blocked or vlm_blocked:
            source = "depth" if depth_blocked else "vlm_2frame"
            self.logger.info(f"[EmergencyAgent] Path blocked detected from {source}")
            return EmergencyEvent(
                type="obstacle",
                severity="medium",
                details={"path_blocked": True, "source": source,
                         "depth_blocked": depth_blocked, "vlm_blocked": vlm_blocked},
            )

        # 无应急
        return None

    def _assess_severity(self, event: EmergencyEvent, context: dict) -> str:
        """LLM评估严重程度。

        Args:
            event: EmergencyEvent
            context: 上下文

        Returns:
            严重程度: "high" / "medium" / "low"
        """
        # 规则优先：用户撤离指令总是high
        if event.details.get("source") == "user":
            return "high"

        # 碰撞总是high（实际碰撞）
        if event.details.get("collision"):
            return "high"

        # 有ModelManager时，使用LLM评估
        if self._model_manager is not None:
            try:
                prompt = self._build_severity_prompt(event, context)
                response = self._call_llm(prompt, max_tokens=100, temperature=0.3)
                severity = self._parse_severity(response)
                if severity in ["high", "medium", "low"]:
                    return severity
            except Exception as e:
                self.logger.warning(f"[EmergencyAgent] LLM severity assessment failed: {e}")

        # 规则回退：根据初始值判断
        if event.severity == "low":
            return "medium"  # 升级为medium（保守处理）
        return event.severity

    def _build_severity_prompt(self, event: EmergencyEvent, context: dict) -> str:
        """构建严重程度评估prompt。

        Args:
            event: EmergencyEvent
            context: 上下文

        Returns:
            Prompt字符串
        """
        event_type = event.type
        details = event.details

        # 获取相关上下文信息
        observation = context.get("observation")
        history = context.get("history", [])

        history_summary = ""
        if history and len(history) > 0:
            recent = history[-3:]
            history_summary = f"Recent positions: {[h.get('position', 'unknown') for h in recent]}"

        observation_info = ""
        if observation and isinstance(observation, ObservationOutput):
            # Format objects - handle both dict and string formats
            obj_names = []
            for obj in (observation.objects or []):
                if isinstance(obj, dict):
                    obj_names.append(obj.get('name', str(obj)))
                else:
                    obj_names.append(str(obj))
            objects_str = ', '.join(obj_names) if obj_names else 'none'
            cues_str = ', '.join(observation.navigation_cues) if observation.navigation_cues else 'none'
            observation_info = f"""
- Scene: {observation.scene_description}
- Path blocked: {observation.path_blocked}
- Objects visible: {objects_str}
- Navigation cues: {cues_str}
"""

        prompt = f"""Assess the severity of this emergency event for a navigation robot.

## Event Type
{event_type}

## Event Details
- Source: {details.get('source', 'unknown')}
- Collision: {details.get('collision', False)}
- Path blocked: {details.get('path_blocked', False)}

## Context
{observation_info}
{history_summary}

## Severity Levels
- high: Immediate danger, requires urgent response (e.g., actual collision, user emergency command)
- medium: Significant obstacle, requires careful handling (e.g., path blocked but not immediate danger)
- low: Minor issue, can be handled with simple adjustment

## Output
Output ONLY a JSON object with severity level:
{{"severity": "high" | "medium" | "low"}}"""

        return prompt

    def _parse_severity(self, response: str) -> str:
        """解析严重程度响应。

        Args:
            response: LLM响应

        Returns:
            严重程度字符串
        """
        if not response:
            return "medium"

        # 尝试JSON解析
        try:
            # 提取JSON
            json_str = response.strip()
            if "```json" in json_str:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", json_str)
                if match:
                    json_str = match.group(1).strip()
            elif "```" in json_str:
                match = re.search(r"```\s*([\s\S]*?)\s*```", json_str)
                if match:
                    json_str = match.group(1).strip()

            # 找到JSON对象
            start = json_str.find("{")
            if start != -1:
                brace_count = 0
                end = -1
                for i in range(start, len(json_str)):
                    if json_str[i] == "{":
                        brace_count += 1
                    elif json_str[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break

                if end != -1:
                    data = json.loads(json_str[start:end+1])
                    return data.get("severity", "medium")
        except (json.JSONDecodeError, ValueError):
            pass

        # 关键词匹配
        response_lower = response.lower()
        if "high" in response_lower:
            return "high"
        elif "low" in response_lower:
            return "low"

        return "medium"

    def _handle_obstacle(self, event: EmergencyEvent, context: dict) -> List[str]:
        """障碍绕行决策（深度智能选向 + LLM辅助）。

        Args:
            event: EmergencyEvent (type=obstacle)
            context: 上下文

        Returns:
            动作序列
        """
        # Priority 1: Use depth-based smart escape direction (zero LLM, fast)
        depth_info = context.get("depth_info", {})
        if depth_info.get("escape_direction"):
            return self._get_smart_escape_actions(depth_info)

        # Priority 2: LLM决策（depth info不可用时）
        if self._model_manager is not None:
            try:
                prompt = self._build_obstacle_prompt(event, context)
                response = self._call_llm(prompt, max_tokens=200, temperature=0.3)
                actions = self._parse_actions(response)
                if actions:
                    return actions
            except Exception as e:
                self.logger.warning(f"[EmergencyAgent] LLM obstacle handling failed: {e}")

        # 默认回退：左转+前进
        self.logger.info("[EmergencyAgent] Using default obstacle fallback: turn_left + forward")
        return self.DEFAULT_OBSTACLE_ACTIONS.copy()

    def _build_obstacle_prompt(self, event: EmergencyEvent, context: dict) -> str:
        """构建障碍处理prompt。

        Args:
            event: EmergencyEvent
            context: 上下文

        Returns:
            Prompt字符串
        """
        details = event.details
        severity = event.severity

        # 获取观察信息
        observation = context.get("observation")
        observation_info = ""
        if observation and isinstance(observation, ObservationOutput):
            observation_info = f"""
## Current Observation
- Scene: {observation.scene_description}
- Objects visible: {', '.join(obj.get('name', str(obj)) if isinstance(obj, dict) else str(obj) for obj in observation.objects) if observation.objects else 'none'}
- Target direction: {observation.target_direction}
- Target distance: {observation.target_distance}
- Navigation cues: {', '.join(observation.navigation_cues) if observation.navigation_cues else 'none'}
"""

        prompt = f"""You are a navigation robot encountering an obstacle. Decide the best bypass strategy.

## Emergency Type
obstacle

## Severity
{severity}

## Obstacle Details
- Collision: {details.get('collision', False)}
- Path blocked: {details.get('path_blocked', False)}

{observation_info}

## Available Actions
- turn_left: Turn 90 degrees left
- turn_right: Turn 90 degrees right
- forward: Move forward one step

## Strategy Options
1. Turn left then forward: Bypass obstacle on the left side
2. Turn right then forward: Bypass obstacle on the right side
3. Turn around (turn_left twice): Retreat and find alternative path

## Decision Criteria
- If collision detected (high severity): Quick turn and retreat
- If path blocked (medium/low): Choose direction based on navigation cues
- Prefer turning toward target direction if visible

## Output
Output ONLY a JSON object with action sequence (2-5 actions):
{{"actions": ["turn_left", "forward", "forward"]}}
or
{{"actions": ["turn_right", "forward", "turn_left", "forward"]}}"""

        return prompt

    def _handle_evacuate(self, event: EmergencyEvent, context: dict) -> List[str]:
        """LLM规划撤离路径。

        Args:
            event: EmergencyEvent (type=evacuate)
            context: 上下文

        Returns:
            动作序列
        """
        topology = context.get("topology")
        position = context.get("position")

        # 尝试使用拓扑信息找出口
        if topology and position:
            try:
                # 如果拓扑模块有find_nearest_exit方法
                if hasattr(topology, "find_nearest_exit"):
                    exit_node = topology.find_nearest_exit(position)
                    if exit_node:
                        self.logger.info(f"[EmergencyAgent] Found exit node: {exit_node}")

                        # 有ModelManager时，使用LLM规划路径
                        if self._model_manager is not None:
                            prompt = self._build_evacuate_prompt(position, exit_node, context)
                            response = self._call_llm(prompt, max_tokens=300, temperature=0.3)
                            actions = self._parse_actions(response)
                            if actions:
                                return actions
            except Exception as e:
                self.logger.warning(f"[EmergencyAgent] Topology-based evacuation failed: {e}")

        # 有ModelManager但无拓扑，使用LLM简单规划
        if self._model_manager is not None:
            try:
                prompt = self._build_simple_evacuate_prompt(event, context)
                response = self._call_llm(prompt, max_tokens=200, temperature=0.3)
                actions = self._parse_actions(response)
                if actions:
                    return actions
            except Exception as e:
                self.logger.warning(f"[EmergencyAgent] LLM evacuate planning failed: {e}")

        # 默认回退：转向后退
        self.logger.info("[EmergencyAgent] Using default evacuate fallback: turn around + forward")
        return self.DEFAULT_EVACUATE_ACTIONS.copy()

    def _build_evacuate_prompt(
        self,
        position: List[float],
        exit_node,
        context: dict,
    ) -> str:
        """构建撤离路径规划prompt（有拓扑信息）。

        Args:
            position: 当前位置
            exit_node: 出口节点
            context: 上下文

        Returns:
            Prompt字符串
        """
        # 获取观察信息
        observation = context.get("observation")
        observation_info = ""
        if observation and isinstance(observation, ObservationOutput):
            observation_info = f"""
## Current Observation
- Scene: {observation.scene_description}
- Objects visible: {', '.join(obj.get('name', str(obj)) if isinstance(obj, dict) else str(obj) for obj in observation.objects) if observation.objects else 'none'}
- Target direction: {observation.target_direction}
"""

        # 获取出口信息
        exit_info = str(exit_node) if exit_node else "unknown"
        if hasattr(exit_node, "position"):
            exit_info = f"Position: {exit_node.position}"
        elif isinstance(exit_node, dict):
            exit_info = f"Position: {exit_node.get('position', 'unknown')}"

        prompt = f"""You are a navigation robot needing to evacuate. Plan the path to exit.

## Current Position
{position}

## Exit Location
{exit_info}

{observation_info}

## Available Actions
- turn_left: Turn 90 degrees left
- turn_right: Turn 90 degrees right
- forward: Move forward one step

## Task
Plan a sequence of actions to reach the exit safely and quickly.

## Output
Output ONLY a JSON object with action sequence (3-10 actions):
{{"actions": ["turn_left", "forward", "forward", "turn_right", "forward"]}}"""

        return prompt

    def _build_simple_evacuate_prompt(
        self,
        event: EmergencyEvent,
        context: dict,
    ) -> str:
        """构建简单撤离prompt（无拓扑信息）。

        Args:
            event: EmergencyEvent
            context: 上下文

        Returns:
            Prompt字符串
        """
        observation = context.get("observation")
        observation_info = ""
        if observation and isinstance(observation, ObservationOutput):
            observation_info = f"""
## Current Observation
- Scene: {observation.scene_description}
- Objects visible: {', '.join(obj.get('name', str(obj)) if isinstance(obj, dict) else str(obj) for obj in observation.objects) if observation.objects else 'none'}
- Navigation cues: {', '.join(observation.navigation_cues) if observation.navigation_cues else 'none'}
"""

        prompt = f"""You are a navigation robot receiving an emergency evacuate command.

## Event
- Type: {event.type}
- Severity: {event.severity}
- Details: {event.details}

{observation_info}

## Task
Plan actions to retreat from current position safely. Since no exit location is known, focus on:
1. Turning around (180 degrees)
2. Moving away from current area
3. Finding a safe path

## Available Actions
- turn_left: Turn 90 degrees left
- turn_right: Turn 90 degrees right
- forward: Move forward one step

## Output
Output ONLY a JSON object with action sequence (3-5 actions):
{{"actions": ["turn_left", "turn_left", "forward", "forward", "forward"]}}"""

        return prompt

    def _parse_actions(self, response: str) -> List[str]:
        """解析动作序列响应。

        Args:
            response: LLM响应

        Returns:
            动作列表
        """
        if not response:
            return []

        valid_actions = ["turn_left", "turn_right", "forward"]

        try:
            # 提取JSON
            json_str = response.strip()
            if "```json" in json_str:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", json_str)
                if match:
                    json_str = match.group(1).strip()
            elif "```" in json_str:
                match = re.search(r"```\s*([\s\S]*?)\s*```", json_str)
                if match:
                    json_str = match.group(1).strip()

            # 找到JSON对象
            start = json_str.find("{")
            if start != -1:
                brace_count = 0
                end = -1
                for i in range(start, len(json_str)):
                    if json_str[i] == "{":
                        brace_count += 1
                    elif json_str[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end = i
                            break

                if end != -1:
                    data = json.loads(json_str[start:end+1])
                    actions = data.get("actions", [])
                    # 验证动作有效性
                    validated = [a for a in actions if a in valid_actions]
                    if validated:
                        return validated
        except (json.JSONDecodeError, ValueError):
            pass

        # 关键词匹配
        actions = []
        response_lower = response.lower()
        for action in valid_actions:
            if action in response_lower:
                actions.append(action)

        # 按出现顺序排序
        ordered_actions = []
        for action in valid_actions:
            idx = response_lower.find(action)
            if idx != -1:
                ordered_actions.append((idx, action))

        ordered_actions.sort()
        return [a[1] for a in ordered_actions] if ordered_actions else []