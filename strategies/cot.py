"""Chain of Thought (CoT) strategy implementation.

This strategy collects information from agents and generates analysis
for action sequence generation.
"""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class CoTStrategy(BaseStrategy):
    """
    Chain of Thought strategy for simple tasks.

    Collects information from Perception, Trajectory, and Instruction agents,
    then generates analysis for action sequence generation.

    Pattern: Collect Info → Analyze → Provide to DecisionAgent
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "CoT"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.COT

    def execute(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        prev_result: Optional[StrategyResult] = None,
    ) -> StrategyResult:
        """
        Execute CoT strategy - collect information and generate analysis.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with collected information and analysis
        """
        self.initialize()

        # 保存 trajectory_agent 引用
        self.trajectory_agent = None
        for agent in agents:
            if agent.name == "trajectory_agent":
                self.trajectory_agent = agent
                break

        steps = []

        try:
            # Step 1: Collect information from agents
            perception_info = self._collect_perception(context)
            steps.append({"type": "perception", "data": perception_info})

            trajectory_info = self._collect_trajectory(context)
            steps.append({"type": "trajectory", "data": trajectory_info})

            instruction_info = self._collect_instruction(context)
            steps.append({"type": "instruction", "data": instruction_info})

            # 从上次策略结果中提取历史信息
            history_info = self._collect_history(prev_result)

            # Step 2: Generate analysis using LLM
            analysis_raw = self._generate_analysis(
                context, perception_info, trajectory_info, instruction_info, history_info
            )
            print(f"\n[CoT] LLM Raw Response:\n{analysis_raw}")

            # Parse LLM output to structured suggestion
            parsed_result = self._parse_action_suggestion(analysis_raw)
            analysis = parsed_result.get("analysis", analysis_raw)
            suggestion = parsed_result.get("suggestion", {})
            steps.append({"type": "analysis", "content": analysis})

            return StrategyResult(
                success=True,
                action=None,  # No single action - will be used for sequence generation
                reasoning=analysis,
                steps=steps,
                confidence=parsed_result.get("suggestion", {}).get("confidence", 0.7),
                metadata={
                    "perception": perception_info,
                    "trajectory": trajectory_info,
                    "instruction": instruction_info,
                    "analysis": analysis,
                    "history": history_info,
                    "suggestion": suggestion,
                    "subtask_completed": parsed_result.get("subtask_completed", False),
                },
            )

        except Exception as e:
            self.logger.error(f"[CoT] Execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"CoT failed: {str(e)}",
                steps=steps,
            )

    def _collect_perception(self, context: NavContext) -> Dict[str, Any]:
        """Collect perception information from context.

        perception_output is now a natural language string from PerceptionAgent.
        No structured extraction needed - DecisionAgent will interpret directly.
        """
        perception_output = context.metadata.get("perception_output", "")
        scene_description = perception_output if isinstance(perception_output, str) else str(perception_output)

        return {
            "scene_description": scene_description,
        }

    def _collect_trajectory(self, context: NavContext) -> Dict[str, Any]:
        """Collect trajectory information including topology from context."""
        # 从metadata获取TrajectoryAgent输出（dict格式）
        # 注意：run_vln_experiment.py存储的是agent_result.data（直接dict）
        trajectory_data = context.metadata.get("trajectory_output", {})
        if not isinstance(trajectory_data, dict):
            trajectory_data = {}

        # 获取历史摘要
        history_summary = "无历史"
        if self.trajectory_agent:
            try:
                current_pos = tuple(context.position) if hasattr(context, "position") else (0, 0, 0)
                history_summary = self.trajectory_agent.get_history_summary(current_pos)
            except Exception as e:
                self.logger.warning(f"[CoT] Failed to get history: {e}")

        # 获取topology_summary
        topology_summary = trajectory_data.get("topology_summary", {})

        return {
            "distance_traveled": trajectory_data.get("distance_traveled", 0),
            "heading": trajectory_data.get("heading", "unknown"),
            "progress_percentage": trajectory_data.get("progress_percentage", 0),
            "stuck_counter": getattr(context, "stuck_counter", 0),
            "step_count": context.step_count,
            "position": context.position,
            "rotation": context.rotation,
            "history_summary": history_summary,
            "topology_summary": topology_summary,  # 新增
            # FIX: Include navigation dict for distance_to_goal
            "navigation": trajectory_data.get("navigation", {}),
            "subtask_delta": trajectory_data.get("subtask_delta", {}),
            "state": trajectory_data.get("state", {}),
        }

    def _collect_history(self, prev_result: Optional[StrategyResult]) -> Dict[str, Any]:
        """从上次策略结果中提取历史信息。

        包括：
        - 上次的推理信息
        - 上次的场景描述 (scene_description)
        - 上次的动作效果（来自 TrajectoryAgent 的 history_summary）
        """
        if not prev_result:
            return {
                "available": False,
                "message": "No history (first execution)",
            }

        try:
            # 从上次策略结果中提取信息
            prev_metadata = prev_result.metadata if hasattr(prev_result, 'metadata') and prev_result.metadata else {}

            # Previous reasoning
            prev_reasoning = prev_result.reasoning if hasattr(prev_result, 'reasoning') else "none"

            # Previous Perception info (includes scene_description)
            prev_perception = prev_metadata.get("perception", {})
            prev_scene_desc = prev_perception.get("scene_description", "none")

            # Previous Trajectory info (includes history_summary)
            prev_trajectory = prev_metadata.get("trajectory", {})
            prev_history_summary = prev_trajectory.get("history_summary", "none")

            return {
                "available": True,
                "prev_reasoning": prev_reasoning,
                "prev_scene_desc": prev_scene_desc,
                "prev_history_summary": prev_history_summary,  # Action effect history
            }
        except Exception as e:
            self.logger.warning(f"[CoT] Failed to collect history: {e}")
            return {
                "available": False,
                "message": f"History parse failed: {str(e)}",
            }

    def _collect_instruction(self, context: NavContext) -> Dict[str, Any]:
        """Collect instruction information from context."""
        instruction_output = context.metadata.get("instruction_output", {})

        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description if current_subtask else ""

        return {
            "full_instruction": context.instruction,
            "current_subtask": subtask_desc,
            "subtask_level": current_subtask.level if current_subtask else "medium",
            # Semantic reasoning info
            "directions": instruction_output.get("directions", []),
            "complexity": instruction_output.get("complexity", 0.0),
            "instruction_analysis": instruction_output.get("instruction_analysis", {}),
        }

    def _generate_analysis(
        self,
        context: NavContext,
        perception_info: Dict,
        trajectory_info: Dict,
        instruction_info: Dict,
        history_info: Dict = None,
    ) -> str:
        """
        Use LLM to generate analysis for sequence generation.

        Returns:
            Analysis string for sequence generation
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            context, perception_info, trajectory_info, instruction_info, history_info
        )

        # Call LLM
        response = self._call_llm(
            prompt,
            max_tokens=300,  # Increased from 150 to allow complete JSON output
            temperature=0.2
        )

        return response

    def _parse_action_suggestion(self, response: str) -> dict:
        """解析LLM输出，多层fallback确保解析成功."""
        import json
        import re

        # Layer 1: JSON解析尝试
        try:
            result = json.loads(response)
            if "suggestion" in result:
                suggestion = result.get("suggestion", {})
                if "direction" not in suggestion:
                    suggestion["direction"] = "forward"
                if "turn_count" not in suggestion:
                    suggestion["turn_count"] = {"min": 2, "max": 3}
                if "forward_count" not in suggestion:
                    suggestion["forward_count"] = {"min": 3, "max": 5}
                result["suggestion"] = suggestion
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Layer 2: 正则提取direction
        direction_match = re.search(r'"direction":\s*"(\w+)"', response)
        if direction_match:
            direction = direction_match.group(1)
            if direction in ["left", "right", "forward"]:
                return {
                    "analysis": response[:100],
                    "suggestion": {
                        "direction": direction,
                        "turn_count": {"min": 2, "max": 3},
                        "forward_count": {"min": 3, "max": 5}
                    },
                    "subtask_completed": False
                }

        # Layer 3: 从文本关键词推断
        response_lower = response.lower()
        if "turn left" in response_lower or "左转" in response:
            return {
                "analysis": response[:100],
                "suggestion": {"direction": "left", "turn_count": {"min": 2, "max": 3}, "forward_count": {"min": 3, "max": 5}},
                "subtask_completed": False
            }
        elif "turn right" in response_lower or "右转" in response:
            return {
                "analysis": response[:100],
                "suggestion": {"direction": "right", "turn_count": {"min": 2, "max": 3}, "forward_count": {"min": 3, "max": 5}},
                "subtask_completed": False
            }
        elif "forward" in response_lower or "前进" in response or "straight" in response_lower:
            return {
                "analysis": response[:100],
                "suggestion": {"direction": "forward", "turn_count": {"min": 0, "max": 0}, "forward_count": {"min": 3, "max": 5}},
                "subtask_completed": False
            }

        # Layer 4: 默认前进
        return {
            "analysis": response[:100],
            "suggestion": {"direction": "forward", "turn_count": {"min": 0, "max": 0}, "forward_count": {"min": 3, "max": 5}},
            "subtask_completed": False
        }

    def _build_analysis_prompt(
        self,
        context: NavContext,
        perception_info: Dict,
        trajectory_info: Dict,
        instruction_info: Dict,
        history_info: Dict = None,
    ) -> str:
        """Build the analysis prompt for LLM."""
        # Get completion condition for current subtask
        current_subtask = context.get_current_subtask()
        completion_condition = current_subtask.completion_condition if current_subtask else None
        condition_str = str(completion_condition) if completion_condition else "none"

        # Build history section
        history_section = ""
        if history_info and history_info.get("available", False):
            history_section = f"""
## Previous Execution History (Reference)
- Previous scene: {history_info.get('prev_scene_desc', 'none')[:80]}
- Previous reasoning: {history_info.get('prev_reasoning', 'none')[:80]}
- Previous action effect: {history_info.get('prev_history_summary', 'none')}
"""

        prompt = f"""You are a navigation analysis expert. Analyze current navigation state to guide path planning.

## Navigation Instruction
{instruction_info['full_instruction']}

## Current Subtask
{instruction_info['current_subtask']} (level: {instruction_info['subtask_level']})

## Subtask Completion Condition
{condition_str}

## Perception Info
Scene description: {perception_info['scene_description']}

## Trajectory State
- Distance traveled: {trajectory_info['distance_traveled']:.1f}m
- Current heading: {trajectory_info['heading']}
- Steps: {trajectory_info['step_count']}
- Position: ({trajectory_info['position'][0]:.1f}, {trajectory_info['position'][1]:.1f}, {trajectory_info['position'][2]:.1f})
- Action history reference: {trajectory_info.get('history_summary', 'No history')}

## Topology Info
- Total nodes: {trajectory_info['topology_summary'].get('total_nodes', 0)}
- Key nodes: {[n.get('type', str(n)) for n in trajectory_info['topology_summary'].get('key_nodes', [])[:5]]}
- Visited rooms: {trajectory_info['topology_summary'].get('visited_rooms', [])[:5]}
- Path to goal: {trajectory_info['topology_summary'].get('path_to_goal', [])[:5]}
- Stuck regions: {len(trajectory_info['topology_summary'].get('stuck_regions', []))}

**Topology hint**: Use topology info for path planning, avoid re-exploring visited areas.
{history_section}

## Spatial Reasoning Guide
Core task: Compare **perception results** with **subtask goal** to infer target position

## Historical Comparison Analysis (if history available)
If "Previous Execution History" is provided, compare:
- Environment change: previous room/scene vs current, determine if entered new area
- Reasoning validation: was previous reasoning validated by subsequent perception, need strategy adjustment?
- Action effect: did previous action move closer or farther from target, guide this action selection

## MANDATORY Analysis Format (follow exactly)
Step 1 - Goal keyword: Identify 1-2 key words from subtask (e.g., "down", "turn left", "rug")

Step 2 - Scene analysis: Analyze scene description for navigation cues
- Identify key features: doors, corridors, open spaces, stairs
- Example: "door on the right" → navigation cue: turn right toward door
- If no relevant information found, write: "No specific navigation cues in scene"

Step 3 - Interpret: Explain what the scene features mean for navigation
- Interpret ONLY observed features
- Do NOT add information not observed

Step 4 - Action suggestion: Give specific action suggestion
- Format: "turn [direction] [N] times, then move forward [M] times"
- Example: "turn left 2-3 times, then move forward 3 times"
- Example: "go straight forward 5 times"
- Example: "turn right 1-2 times, then move forward"

## CRITICAL RULES
- Every claim MUST be supported by direct observation from scene_description
- Do NOT invent facts or make logical leaps

## Action Mapping Guide
- "on left" / "left side" / "door on left" → turn left 2-3 times, then forward
- "on right" / "right side" / "door on right" → turn right 2-3 times, then forward
- "ahead" / "forward" / "straight" / "open space ahead" → go straight forward
- "behind" / "backward" → turn around (6 turns), then forward

## Elevation/Direction Navigation Guide
If subtask mentions direction-related words (stairs, up, down, ascend, descend, elevation):
- Read scene description for ANY elevation or direction-related information
- Match with subtask: "Walk down" requires finding path going DOWN
- Position Y change: If Y increased but subtask requires "down", you went WRONG direction!

## Output Format (JSON)
Output JSON only, no other text:
{{
  "analysis": "Step 1-4 analysis merged into one sentence",
  "suggestion": {{
    "direction": "left" | "right" | "forward",
    "turn_count": {{"min": 2, "max": 3}},
    "forward_count": {{"min": 3, "max": 5}}
  }},
  "subtask_completed": false
}}

## Rules
- direction: MUST be one of "left", "right", "forward"
- turn_count: Only when direction is left/right, default {{"min": 2, "max": 3}}
- forward_count: Always required, suggest 3-5 steps
- subtask_completed: Default false, set true only when goal clearly reached
- Output ONLY the JSON, no explanation before or after"""

        return prompt
