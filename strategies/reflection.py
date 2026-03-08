"""Reflection strategy implementation."""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class ReflectionStrategy(BaseStrategy):
    """
    Reflection strategy: Self-evaluation and improvement.

    Pattern: Action → Observation → Reflection → Adjustment

    This strategy enables learning from past actions through
    explicit reflection and course correction.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("ReflectionStrategy")

        # Configuration
        self.reflection_depth = self.config.get("reflection_depth", 3)
        self.include_lessons = self.config.get("include_lessons", True)
        self.history_window = self.config.get("history_window", 5)

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
        Execute Reflection strategy.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with reflected action
        """
        self.initialize()

        steps = []

        try:
            # Step 1: Review recent actions
            action_review = self._review_recent_actions(context)
            steps.append({"type": "action_review", "content": action_review})

            # Step 2: Analyze outcomes
            outcome_analysis = self._analyze_outcomes(context)
            steps.append({"type": "outcome_analysis", "content": outcome_analysis})

            # Step 3: Generate reflection
            reflection = self._generate_reflection(context, action_review, outcome_analysis)
            steps.append({"type": "reflection", "content": reflection})

            # Step 4: Identify improvements
            improvements = self._identify_improvements(context, reflection)
            steps.append({"type": "improvements", "content": improvements})

            # Step 5: Adjust decision
            action, confidence, reasoning = self._adjusted_decision(
                context, agents, reflection, improvements
            )
            steps.append({
                "type": "adjusted_decision",
                "action": action.to_habitat_action(),
                "confidence": confidence,
            })

            # Step 6: Store lessons (optional)
            if self.include_lessons:
                self._store_lesson(context, action, reflection)

            return StrategyResult(
                success=True,
                action=action,
                reasoning=reasoning,
                steps=steps,
                confidence=confidence,
                metadata={"lessons_count": len(self._lessons_learned)},
            )

        except Exception as e:
            self.logger.error(f"Reflection execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"Reflection failed: {str(e)}",
                steps=steps,
            )

    def _review_recent_actions(self, context: NavContext) -> str:
        """Review recent navigation actions."""
        if not context.action_history:
            return "No previous actions to review."

        recent = context.action_history[-self.history_window:]

        # Summarize actions
        action_counts = {}
        for action in recent:
            action_type = action.action_type.name
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        summary = f"Recent {len(recent)} actions: "
        summary += ", ".join(f"{k}({v})" for k, v in action_counts.items())

        return summary

    def _analyze_outcomes(self, context: NavContext) -> str:
        """Analyze outcomes of recent actions."""
        outcomes = []

        # Position progress
        if len(context.trajectory) >= 2:
            start = context.trajectory[0]
            current = context.trajectory[-1]
            distance = (
                (current[0] - start[0]) ** 2 +
                (current[2] - start[2]) ** 2
            ) ** 0.5
            outcomes.append(f"Distance from start: {distance:.2f}m")

        # Room transitions
        if context.room_type != "unknown":
            outcomes.append(f"Currently in: {context.room_type}")

        # Subtask progress
        if context.subtasks:
            completed = sum(1 for s in context.subtasks if s.status == "completed")
            outcomes.append(f"Subtasks: {completed}/{len(context.subtasks)} completed")

        # Step efficiency
        if context.step_count > 0:
            outcomes.append(f"Steps taken: {context.step_count}")

        return " | ".join(outcomes) if outcomes else "No outcome data available"

    def _generate_reflection(
        self,
        context: NavContext,
        action_review: str,
        outcome_analysis: str,
    ) -> str:
        """Generate reflection on navigation progress."""
        reflection_parts = []

        # Reflect on direction
        if context.action_history:
            recent_actions = [a.action_type.name for a in context.action_history[-3:]]
            if "TURN_LEFT" in recent_actions and "TURN_RIGHT" in recent_actions:
                reflection_parts.append(
                    "Notice: Recent actions include both left and right turns - "
                    "may indicate uncertainty"
                )

            if all(a == "MOVE_FORWARD" for a in recent_actions):
                reflection_parts.append(
                    "Notice: Consistent forward movement - good progress trajectory"
                )

        # Reflect on instruction alignment
        instruction_lower = context.instruction.lower()
        if "turn" in instruction_lower:
            if "TURN_LEFT" not in [a.action_type.name for a in context.action_history]:
                reflection_parts.append(
                    "Note: Instruction mentions turning but no turn executed yet"
                )

        # Reflect on progress
        if context.step_count > 10 and not context.subtasks:
            reflection_parts.append(
                "Consider: Many steps taken without explicit subtask progress"
            )

        # Add lessons from past
        if self._lessons_learned:
            relevant_lessons = [
                l for l in self._lessons_learned[-3:]
                if l.get("context_type") == context.room_type
            ]
            if relevant_lessons:
                reflection_parts.append(
                    f"Recalled {len(relevant_lessons)} relevant past lesson(s)"
                )

        if reflection_parts:
            return "; ".join(reflection_parts)
        else:
            return "Navigation appears on track with no significant issues to address."

    def _identify_improvements(
        self, context: NavContext, reflection: str
    ) -> str:
        """Identify potential improvements based on reflection."""
        improvements = []

        # Check for uncertainty patterns
        if "uncertainty" in reflection.lower():
            improvements.append("Consider: Pausing to reassess environment before continuing")

        # Check for missed actions
        if "no turn executed" in reflection.lower():
            improvements.append("Consider: Executing required turn action")

        # Check for efficiency
        if "many steps" in reflection.lower():
            improvements.append("Consider: More direct path toward goal")

        # Check for lessons
        if "relevant past lesson" in reflection.lower():
            improvements.append("Apply: Lessons learned from similar situations")

        if improvements:
            return " | ".join(improvements)
        else:
            return "No immediate improvements identified - continue current strategy"

    def _adjusted_decision(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        reflection: str,
        improvements: str,
    ) -> tuple:
        """Make adjusted decision based on reflection."""
        # Get base decision from decision agent
        decision_agent = self.get_agent_by_role(agents, "decision")

        base_action = None
        base_confidence = 0.5

        if decision_agent:
            output = decision_agent.process(context)
            if output.success:
                action_str = output.data.get("action", "forward")
                base_action = self._action_from_string(action_str)
                base_confidence = output.confidence

        if base_action is None:
            base_action = Action.forward()

        # Apply adjustments based on reflection
        adjusted_action = base_action
        adjusted_confidence = base_confidence
        adjustment_reasoning = []

        # Check if improvement suggests different action
        if "required turn action" in improvements.lower():
            instruction_lower = context.instruction.lower()
            if "left" in instruction_lower:
                adjusted_action = Action.turn_left()
                adjustment_reasoning.append("Adjusted to turn left per instruction")
            elif "right" in instruction_lower:
                adjusted_action = Action.turn_right()
                adjustment_reasoning.append("Adjusted to turn right per instruction")
            adjusted_confidence = min(base_confidence + 0.1, 1.0)

        # Check for uncertainty adjustment
        if "uncertainty" in reflection.lower():
            adjusted_confidence = max(adjusted_confidence - 0.1, 0.3)
            adjustment_reasoning.append("Lowered confidence due to detected uncertainty")

        # Check for lesson application
        if "lesson" in reflection.lower():
            adjusted_confidence = min(adjusted_confidence + 0.05, 1.0)
            adjustment_reasoning.append("Boosted confidence with lesson application")

        # Build final reasoning
        reasoning = f"Base: {base_action.action_type.name}"
        if adjustment_reasoning:
            reasoning += f" | Adjustments: {'; '.join(adjustment_reasoning)}"
        reasoning += f" | Final: {adjusted_action.action_type.name} (conf: {adjusted_confidence:.2f})"

        return adjusted_action, adjusted_confidence, reasoning

    def _store_lesson(
        self, context: NavContext, action: Action, reflection: str
    ) -> None:
        """Store lesson for future reference."""
        lesson = {
            "context_type": context.room_type,
            "action_taken": action.action_type.name,
            "reflection_summary": reflection[:100],
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

    def _action_from_string(self, action_str: str) -> Action:
        """Convert action string to Action object."""
        action_map = {
            "stop": Action.stop(),
            "move_forward": Action.forward(),
            "forward": Action.forward(),
            "turn_left": Action.turn_left(),
            "turn_right": Action.turn_right(),
        }
        return action_map.get(action_str.lower(), Action.forward())