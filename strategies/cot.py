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

        steps = []

        try:
            # Step 1: Collect information from agents
            perception_info = self._collect_perception(context)
            steps.append({"type": "perception", "data": perception_info})

            trajectory_info = self._collect_trajectory(context)
            steps.append({"type": "trajectory", "data": trajectory_info})

            instruction_info = self._collect_instruction(context)
            steps.append({"type": "instruction", "data": instruction_info})

            # Step 2: Generate analysis using LLM
            analysis = self._generate_analysis(
                context, perception_info, trajectory_info, instruction_info
            )
            steps.append({"type": "analysis", "content": analysis})

            return StrategyResult(
                success=True,
                action=None,  # No single action - will be used for sequence generation
                reasoning=analysis,
                steps=steps,
                confidence=0.7,
                metadata={
                    "perception": perception_info,
                    "trajectory": trajectory_info,
                    "instruction": instruction_info,
                    "analysis": analysis,
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
        """Collect perception information from context."""
        perception_output = context.metadata.get("perception_output", {})

        return {
            "room_type": perception_output.get("room_type", "unknown"),
            "scene_description": perception_output.get("scene_description", ""),
            "objects": perception_output.get("objects", [])[:5],
            "landmarks": perception_output.get("landmarks", []),
            "nav_hint": perception_output.get("nav_hint", ""),
            "walkable_analysis": perception_output.get("walkable_analysis", {}),
            "obstacle_ahead": perception_output.get("obstacle_ahead", {}),
        }

    def _collect_trajectory(self, context: NavContext) -> Dict[str, Any]:
        """Collect trajectory information from context."""
        trajectory_output = context.metadata.get("trajectory_output", {})

        return {
            "distance_traveled": trajectory_output.get("distance_traveled", 0),
            "heading": trajectory_output.get("heading", "unknown"),
            "progress_percentage": trajectory_output.get("progress_percentage", 0),
            "stuck_counter": getattr(context, "stuck_counter", 0),
            "step_count": context.step_count,
            "position": context.position,
            "rotation": context.rotation,
        }

    def _collect_instruction(self, context: NavContext) -> Dict[str, Any]:
        """Collect instruction information from context."""
        instruction_output = context.metadata.get("instruction_output", {})

        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description if current_subtask else ""

        return {
            "full_instruction": context.instruction,
            "current_subtask": subtask_desc,
            "subtask_level": current_subtask.level if current_subtask else "中等",
            # 语义推理信息
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
    ) -> str:
        """
        Use LLM to generate analysis for sequence generation.

        Returns:
            Analysis string for sequence generation
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            context, perception_info, trajectory_info, instruction_info
        )

        # Call LLM
        response = self._call_llm(
            prompt,
            max_tokens=150,
            temperature=0.2
        )

        return response

    def _build_analysis_prompt(
        self,
        context: NavContext,
        perception_info: Dict,
        trajectory_info: Dict,
        instruction_info: Dict,
    ) -> str:
        """Build the analysis prompt for LLM."""
        prompt = f"""你是一个导航分析专家。请分析当前导航状态，为后续路径规划提供指导。

## 导航指令
{instruction_info['full_instruction']}

## 当前子任务
{instruction_info['current_subtask']} (难度: {instruction_info['subtask_level']})

## 感知信息
- 房间类型: {perception_info['room_type']}
- 场景描述: {perception_info['scene_description'][:100]}
- 可见物体: {[o.get('物体', o.get('name', '')) for o in perception_info['objects'][:3]]}
- 导航提示: {perception_info['nav_hint']}

## 轨迹状态
- 已走距离: {trajectory_info['distance_traveled']:.1f}m
- 当前朝向: {trajectory_info['heading']}
- 步数: {trajectory_info['step_count']}
- 位置: ({trajectory_info['position'][0]:.1f}, {trajectory_info['position'][1]:.1f}, {trajectory_info['position'][2]:.1f})

## 分析要求
1. 分析当前子任务的关键要素
2. 评估当前环境对导航的影响
3. 识别潜在障碍和机会
4. 给出导航建议（30字以内）

直接输出分析结果，不要JSON格式："""

        return prompt