"""Reflection strategy implementation.

This strategy collects information, reflects on history, and generates
analysis for action sequence generation.
"""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class ReflectionStrategy(BaseStrategy):
    """
    Reflection strategy for medium difficulty tasks.

    Collects information from agents, reflects on action history,
    and generates analysis with improvement suggestions.

    Pattern: Collect Info → Reflect on History → Provide to DecisionAgent
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Configuration
        self.history_window = self.config.get("history_window", 10)

        # Learning storage
        self._lessons_learned: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "Reflection"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.REFLECTION

    def execute(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        prev_result: Optional[StrategyResult] = None,
    ) -> StrategyResult:
        """
        Execute Reflection strategy - collect info, reflect, and generate analysis.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with collected information, reflection, and lessons
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

            # Step 2: Review recent actions
            action_review = self._review_recent_actions(context)
            steps.append({"type": "action_review", "content": action_review})

            # Step 3: Reflect on history
            reflection = self._reflect_on_history(
                context, perception_info, trajectory_info, instruction_info, action_review
            )
            steps.append({"type": "reflection", "content": reflection})

            # Step 4: Store lesson
            self._store_lesson(context, reflection)

            return StrategyResult(
                success=True,
                action=None,  # No single action - will be used for sequence generation
                reasoning=reflection,
                steps=steps,
                confidence=0.75,
                metadata={
                    "perception": perception_info,
                    "trajectory": trajectory_info,
                    "instruction": instruction_info,
                    "reflection": reflection,
                    "lessons": self._lessons_learned[-3:],
                    "action_review": action_review,
                },
            )

        except Exception as e:
            self.logger.error(f"[Reflection] Execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"Reflection failed: {str(e)}",
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
            "subtask_level": current_subtask.level if current_subtask else "medium",
            # Semantic reasoning info
            "directions": instruction_output.get("directions", []),
            "complexity": instruction_output.get("complexity", 0.0),
            "instruction_analysis": instruction_output.get("instruction_analysis", {}),
        }

    def _review_recent_actions(self, context: NavContext) -> str:
        """Review recent navigation actions."""
        if not context.action_history:
            return "No action history"

        recent = context.action_history[-self.history_window:]

        # Summarize actions
        action_counts = {}
        for action in recent:
            action_type = action.action_type.name
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        summary = f"Recent {len(recent)} actions: "
        summary += ", ".join(f"{k}({v})" for k, v in action_counts.items())

        return summary

    def _reflect_on_history(
        self,
        context: NavContext,
        perception_info: Dict,
        trajectory_info: Dict,
        instruction_info: Dict,
        action_review: str,
    ) -> str:
        """
        Use LLM to reflect on history and generate suggestions.

        Returns:
            Reflection string with improvement suggestions
        """
        # Build reflection prompt
        prompt = self._build_reflection_prompt(
            context, perception_info, trajectory_info, instruction_info, action_review
        )

        # Call LLM
        response = self._call_llm(
            prompt,
            max_tokens=200,
            temperature=0.3
        )

        return response

    def _build_reflection_prompt(
        self,
        context: NavContext,
        perception_info: Dict,
        trajectory_info: Dict,
        instruction_info: Dict,
        action_review: str,
    ) -> str:
        """Build the reflection prompt for LLM."""
        # Format lessons
        lessons_str = "No historical experience"
        if self._lessons_learned:
            recent = self._lessons_learned[-3:]
            lessons_str = "\n".join([
                f"{i+1}. {l.get('context', 'unknown')}: {l.get('insight', '')[:50]}"
                for i, l in enumerate(recent)
            ])

        prompt = f"""/no_think
You are a navigation reflection expert. Analyze current state, reflect on action history, and provide improvement suggestions.

## Navigation Instruction
{instruction_info['full_instruction']}

## Current Subtask
{instruction_info['current_subtask']} (level: {instruction_info['subtask_level']})

## Perception Info
- Room type: {perception_info['room_type']}
- Scene description: {perception_info['scene_description'][:100]}
- Visible objects: {[o.get('object', o.get('物体', o.get('name', ''))) for o in perception_info['objects'][:3]]}

## Trajectory State
- Distance traveled: {trajectory_info['distance_traveled']:.1f}m
- Steps: {trajectory_info['step_count']}
- Position: ({trajectory_info['position'][0]:.1f}, {trajectory_info['position'][1]:.1f}, {trajectory_info['position'][2]:.1f})

## Action Review
{action_review}

## Historical Experience
{lessons_str}

## Analysis Requirements
1. Analyze whether current strategy is correct
2. Identify potential issues (e.g., repeated actions, inefficient paths)
3. Provide concrete improvement suggestions
4. Give next step navigation focus (within 30 words)

Output reflection result directly, no JSON format:"""

        return prompt

    def _store_lesson(self, context: NavContext, reflection: str) -> None:
        """Store lesson for future reference."""
        lesson = {
            "context": context.room_type,
            "subtask": context.get_current_subtask().description if context.get_current_subtask() else "",
            "insight": reflection[:150],
            "step": context.step_count,
        }
        self._lessons_learned.append(lesson)

        # Keep only recent lessons
        max_lessons = self.config.get("max_lessons", 100)
        if len(self._lessons_learned) > max_lessons:
            self._lessons_learned = self._lessons_learned[-max_lessons:]

    def get_lessons(self) -> List[Dict[str, Any]]:
        """Get stored lessons."""
        return self._lessons_learned.copy()

    def clear_lessons(self) -> None:
        """Clear stored lessons."""
        self._lessons_learned.clear()