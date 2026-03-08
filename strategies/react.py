"""ReAct (Reasoning + Acting) strategy implementation."""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class ReActStrategy(BaseStrategy):
    """
    ReAct strategy: Interleaves reasoning and action.

    Pattern: Thought → Action → Observation → Thought → ...

    This strategy enables step-by-step reasoning with explicit
    thought-action-observation cycles.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("ReActStrategy")

        # Configuration
        self.max_iterations = self.config.get("max_iterations", 10)
        self.thought_template = self.config.get(
            "thought_template",
            "Current state: {state}\nTask: {task}\nThought: {thought}",
        )

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
        Execute ReAct strategy.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with action
        """
        self.initialize()

        steps = []
        iteration = 0

        # Initialize with previous result if available
        if prev_result:
            steps.append({
                "type": "context",
                "from": prev_result.reasoning if prev_result else "",
            })

        try:
            # Main ReAct loop
            while iteration < self.max_iterations:
                # Step 1: Thought - Analyze current situation
                thought = self._generate_thought(context, steps, agents)
                steps.append({"type": "thought", "content": thought})

                # Step 2: Action - Decide what to do
                action, confidence = self._decide_action(context, thought, agents)
                steps.append({
                    "type": "action",
                    "action": action.to_habitat_action(),
                    "confidence": confidence,
                })

                # Step 3: Check termination conditions
                if action.action_type.value == 0:  # STOP
                    return StrategyResult(
                        success=True,
                        action=action,
                        reasoning=thought,
                        steps=steps,
                        confidence=confidence,
                    )

                # For single-step ReAct, return after first action decision
                if iteration == 0:
                    return StrategyResult(
                        success=True,
                        action=action,
                        reasoning=thought,
                        steps=steps,
                        confidence=confidence,
                    )

                iteration += 1

            # Max iterations reached
            return StrategyResult(
                success=True,
                action=Action.stop(confidence=0.5),
                reasoning="Max iterations reached",
                steps=steps,
                confidence=0.5,
            )

        except Exception as e:
            self.logger.error(f"ReAct execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"ReAct failed: {str(e)}",
                steps=steps,
            )

    def _generate_thought(
        self,
        context: NavContext,
        previous_steps: List[Dict],
        agents: List[BaseAgent],
    ) -> str:
        """Generate thought about current situation."""
        # Gather information from agents
        instruction_thought = self._get_instruction_thought(context, agents)
        perception_thought = self._get_perception_thought(context, agents)
        trajectory_thought = self._get_trajectory_thought(context, agents)

        # Combine thoughts
        thoughts = []

        if instruction_thought:
            thoughts.append(instruction_thought)

        if perception_thought:
            thoughts.append(perception_thought)

        if trajectory_thought:
            thoughts.append(trajectory_thought)

        # Add history consideration
        if context.action_history:
            recent_actions = context.get_action_summary(3)
            thoughts.append(f"Recent actions: {recent_actions}")

        # Generate final thought
        if thoughts:
            thought = " | ".join(thoughts)
        else:
            thought = f"Need to navigate according to: {context.instruction[:50]}..."

        return thought

    def _get_instruction_thought(
        self, context: NavContext, agents: List[BaseAgent]
    ) -> str:
        """Get thought from instruction analysis."""
        if context.subtasks:
            current = context.get_current_subtask()
            if current:
                return f"Current subtask: {current.description}"

        return f"Instruction: {context.instruction[:60]}..."

    def _get_perception_thought(
        self, context: NavContext, agents: List[BaseAgent]
    ) -> str:
        """Get thought from perception."""
        if context.visual_features.scene_description:
            return context.visual_features.scene_description[:100]

        if context.room_type and context.room_type != "unknown":
            return f"In {context.room_type}"

        return ""

    def _get_trajectory_thought(
        self, context: NavContext, agents: List[BaseAgent]
    ) -> str:
        """Get thought from trajectory analysis."""
        if context.metadata.get("trajectory"):
            progress = context.metadata["trajectory"].get("progress", 0)
            return f"Progress: {progress*100:.0f}%"

        return f"Step {context.step_count}"

    def _decide_action(
        self,
        context: NavContext,
        thought: str,
        agents: List[BaseAgent],
    ) -> tuple:
        """Decide action based on thought."""
        # Get decision agent
        decision_agent = self.get_agent_by_role(agents, "decision")

        if decision_agent:
            # Let decision agent decide
            output = decision_agent.process(context)
            if output.success:
                action_str = output.data.get("action", "stop")
                confidence = output.confidence
                action = self._action_from_string(action_str)
                return action, confidence

        # Fallback: keyword-based decision
        return self._keyword_decision(context, thought)

    def _keyword_decision(self, context: NavContext, thought: str) -> tuple:
        """Make decision based on keywords."""
        instruction_lower = context.instruction.lower()
        thought_lower = thought.lower()

        # Check for direction keywords
        if "turn left" in instruction_lower or "left" in thought_lower:
            return Action.turn_left(confidence=0.7), 0.7
        elif "turn right" in instruction_lower or "right" in thought_lower:
            return Action.turn_right(confidence=0.7), 0.7
        elif "stop" in instruction_lower or "goal reached" in thought_lower:
            return Action.stop(confidence=0.8), 0.8
        else:
            return Action.forward(confidence=0.6), 0.6

    def _action_from_string(self, action_str: str) -> Action:
        """Convert action string to Action object."""
        from core.action import Action

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