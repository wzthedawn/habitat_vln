"""ReAct (Reasoning + Acting) strategy implementation.

This strategy uses LLM-based interleaved reasoning and action selection.
The LLM generates thoughts and decides actions in an iterative cycle.
"""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class ReActStrategy(BaseStrategy):
    """
    ReAct strategy: LLM-based interleaved reasoning and action.

    Pattern: LLM Thought → LLM Action → Observation → LLM Thought → ...

    This strategy uses LLM to generate intelligent thoughts and
    decide actions step by step.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Configuration
        self.max_iterations = self.config.get("max_iterations", 10)

    @property
    def name(self) -> str:
        return "ReAct"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.REACT

    def execute(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        prev_result: Optional[StrategyResult] = None,
    ) -> StrategyResult:
        """
        Execute ReAct strategy with LLM-based reasoning.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with action
        """
        self.initialize()

        steps = []

        # Initialize with previous result if available
        if prev_result:
            steps.append({
                "type": "context",
                "from": prev_result.reasoning if prev_result else "",
            })

        try:
            # Single iteration ReAct (LLM generates thought and action together)
            thought, action, confidence = self._generate_llm_thought_and_action(
                context, steps
            )

            steps.append({"type": "thought", "content": thought})
            steps.append({
                "type": "action",
                "action": action.to_habitat_action(),
                "confidence": confidence,
            })

            return StrategyResult(
                success=True,
                action=action,
                reasoning=thought,
                steps=steps,
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"[ReAct] Execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"ReAct failed: {str(e)}",
                steps=steps,
            )

    def _generate_llm_thought_and_action(
        self,
        context: NavContext,
        previous_steps: List[Dict],
    ) -> tuple:
        """
        Use LLM to generate thought and decide action in one call.

        Returns:
            tuple: (thought, action, confidence)
        """
        # Build LLM prompt
        prompt = self._build_react_prompt(context, previous_steps)

        # Call LLM
        response = self._call_llm(
            prompt,
            max_tokens=250,
            temperature=0.2
        )

        # Parse LLM response
        thought, action_str, confidence = self._parse_llm_response(response)

        # Convert to Action object
        action = self._action_from_string(action_str)

        return thought, action, confidence

    def _build_react_prompt(
        self,
        context: NavContext,
        previous_steps: List[Dict],
    ) -> str:
        """Build the ReAct prompt for LLM."""
        instruction = context.instruction

        # Get current subtask
        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description if current_subtask else "无"

        # Get perception info
        perception_info = self._get_perception_info(context)

        # Get trajectory info
        trajectory_info = self._get_trajectory_info(context)

        # Get topology info
        topology_info = self._get_topology_info(context)

        # Get recent actions
        recent_actions = []
        if context.action_history:
            for a in context.action_history[-5:]:
                recent_actions.append(a.action_type.name)

        prompt = f"""你是一个导航推理专家。请使用ReAct模式进行推理和决策。

## 导航指令
{instruction}

## 当前子任务
{subtask_desc}

## 当前状态
- 步数: {context.step_count}
- 位置: ({context.position[0]:.1f}, {context.position[1]:.1f}, {context.position[2]:.1f})
- 房间类型: {context.room_type}

## 感知信息
{perception_info}

## 轨迹信息
{trajectory_info}

## 拓扑信息
{topology_info}

## 最近动作历史
{', '.join(recent_actions) if recent_actions else '无'}

## 可选动作
- forward: 向前移动
- turn_left: 向左转
- turn_right: 向右转
- stop: 停止导航

## 要求
请按ReAct模式思考:
1. Thought: 观察当前状态，分析应该做什么
2. Action: 选择一个动作执行

## 输出格式
请严格按照以下JSON格式输出:
```json
{{
  "thought": "对当前状态的观察和思考...",
  "action": "forward/turn_left/turn_right/stop",
  "confidence": 0.0-1.0,
  "reasoning": "选择该动作的简要理由"
}}
```
"""
        return prompt

    def _get_perception_info(self, context: NavContext) -> str:
        """Get perception information from Agent output."""
        parts = []

        # 从metadata获取PerceptionAgent输出（dict格式）
        # 注意：run_vln_experiment.py存储的是agent_result.data（直接dict）
        perception_data = context.metadata.get("perception_output", {})
        if not isinstance(perception_data, dict):
            perception_data = {}

        if perception_data:
            # 使用Agent处理后的信息
            room_type = perception_data.get("room_type", "unknown")
            if room_type and room_type != "unknown":
                parts.append(f"房间类型: {room_type}")

            scene_desc = perception_data.get("scene_description", "")
            if scene_desc:
                parts.append(f"场景描述: {scene_desc[:100]}")

            objects = perception_data.get("objects", [])
            if objects:
                obj_names = [o.get("object", o.get("name", str(o))) for o in objects[:5]]
                parts.append(f"可见物体: {', '.join(obj_names)}")

            scene_description = perception_data.get("scene_description", "")
            if scene_description:
                parts.append(f"场景描述: {scene_description[:100]}")

        # Fallback: 如果无Agent输出，使用原始visual_features
        if not parts and context.visual_features.scene_description:
            parts.append(f"场景: {context.visual_features.scene_description[:100]}")

        return "\n".join(parts) if parts else "无感知信息"

    def _get_trajectory_info(self, context: NavContext) -> str:
        """Get trajectory information from Agent output."""
        parts = []

        # 从metadata获取TrajectoryAgent输出（dict格式）
        # 注意：run_vln_experiment.py存储的是agent_result.data（直接dict）
        trajectory_data = context.metadata.get("trajectory_output", {})
        if not isinstance(trajectory_data, dict):
            trajectory_data = {}

        if trajectory_data:
            distance = trajectory_data.get("distance_traveled", 0)
            if distance:
                parts.append(f"已走距离: {distance:.1f}m")

            heading = trajectory_data.get("heading", "unknown")
            if heading and heading != "unknown":
                parts.append(f"当前朝向: {heading}")

            progress = trajectory_data.get("progress_percentage", 0)
            if progress:
                parts.append(f"进度: {progress:.1f}%")

        # Fallback: 如果无Agent输出，从原始trajectory计算
        if not parts and len(context.trajectory) >= 2:
            start = context.trajectory[0]
            current = context.trajectory[-1]
            dist = ((current[0] - start[0])**2 + (current[2] - start[2])**2)**0.5
            parts.append(f"总距离: {dist:.1f}m")
            parts.append(f"步数: {context.step_count}")

        return "\n".join(parts) if parts else f"步数: {context.step_count}"

    def _get_topology_info(self, context: NavContext) -> str:
        """Get topology information from TrajectoryAgent output.

        Returns formatted topology summary for ReAct prompt.
        """
        # 从metadata获取TrajectoryAgent输出（dict格式）
        # 注意：run_vln_experiment.py存储的是agent_result.data（直接dict）
        trajectory_data = context.metadata.get("trajectory_output", {})
        if not isinstance(trajectory_data, dict):
            trajectory_data = {}
        topology_summary = trajectory_data.get("topology_summary", {})

        if not topology_summary:
            return "无拓扑信息"

        total_nodes = topology_summary.get("total_nodes", 0)
        key_nodes = topology_summary.get("key_nodes", [])
        current_node = topology_summary.get("current_node", "unknown")
        visited_rooms = topology_summary.get("visited_rooms", [])
        stuck_regions = topology_summary.get("stuck_regions", [])
        path_to_goal = topology_summary.get("path_to_goal", [])

        # 格式化关键节点（显示类型）
        key_nodes_str = ", ".join([n.get("type", str(n)) for n in key_nodes[:5]]) if key_nodes else "无"
        visited_rooms_str = ", ".join(visited_rooms[:5]) if visited_rooms else "无"
        path_str = " -> ".join(path_to_goal[:5]) if path_to_goal else "未知"

        return f"""- 总节点数: {total_nodes}
- 关键节点: {key_nodes_str}
- 当前位置节点: {current_node}
- 已访问房间: {visited_rooms_str}
- 目标路径: {path_str}
- 卡住区域: {len(stuck_regions)}个

**拓扑提示**: 使用已访问房间信息避免重复探索，参考关键节点做路径规划。"""

    def _parse_llm_response(self, response: str) -> tuple:
        """
        Parse LLM response to extract thought, action, and confidence.

        Returns:
            tuple: (thought, action_str, confidence)
        """
        import json
        import re

        # Default values
        thought = "基于当前状态继续导航"
        action_str = "forward"
        confidence = 0.5

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # Build thought from components
                thought_parts = []
                if data.get("thought"):
                    thought_parts.append(f"思考: {data['thought']}")
                if data.get("reasoning"):
                    thought_parts.append(f"理由: {data['reasoning']}")

                thought = " | ".join(thought_parts) if thought_parts else data.get("thought", thought)

                action_str = data.get("action", "forward").lower()
                confidence = float(data.get("confidence", 0.5))

                # Clamp confidence
                confidence = max(0.0, min(1.0, confidence))

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"[ReAct] Failed to parse JSON response: {e}")

            # Fallback: try to extract action from text
            if "turn_left" in response.lower():
                action_str = "turn_left"
            elif "turn_right" in response.lower():
                action_str = "turn_right"
            elif "stop" in response.lower():
                action_str = "stop"

            thought = f"LLM推理: {response[:200]}"

        return thought, action_str, confidence

    def _action_from_string(self, action_str: str) -> Action:
        """Convert action string to Action object."""
        action_map = {
            "stop": Action.stop(),
            "move_forward": Action.forward(),
            "forward": Action.forward(),
            "turn_left": Action.turn_left(),
            "turn_right": Action.turn_right(),
            "look_up": Action.look_up(),
            "look_down": Action.look_down(),
        }
        return action_map.get(action_str.lower(), Action.forward())