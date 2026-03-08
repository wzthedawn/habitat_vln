"""Chain of Thought (CoT) strategy implementation."""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class CoTStrategy(BaseStrategy):
    """
    Chain of Thought strategy: Explicit multi-step reasoning.

    Pattern: Question → Analysis → Reasoning Steps → Answer

    This strategy enables complex reasoning by breaking down
    navigation problems into explicit reasoning steps.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("CoTStrategy")

        # Configuration
        self.max_reasoning_steps = self.config.get("max_reasoning_steps", 5)
        self.include_verification = self.config.get("include_verification", True)

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
        Execute CoT strategy.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with reasoned action
        """
        self.initialize()

        steps = []

        try:
            # Step 1: Define the question/problem
            question = self._define_question(context)
            steps.append({"type": "question", "content": question})

            # Step 2: Gather relevant information
            analysis = self._analyze_situation(context, agents)
            steps.append({"type": "analysis", "content": analysis})

            # Step 3: Generate reasoning steps
            reasoning_steps = self._generate_reasoning_steps(context, analysis, agents)
            for i, step in enumerate(reasoning_steps):
                steps.append({"type": "reasoning", "step": i + 1, "content": step})

            # Step 4: Derive conclusion and action
            conclusion, action, confidence = self._derive_conclusion(
                context, reasoning_steps, agents
            )
            steps.append({"type": "conclusion", "content": conclusion})

            # Step 5: Verification (optional)
            if self.include_verification:
                verification = self._verify_action(context, action, reasoning_steps)
                steps.append({"type": "verification", "content": verification})
                if not verification.get("valid", True):
                    # Re-evaluate if verification fails
                    action, confidence = self._re_evaluate(context, reasoning_steps)

            return StrategyResult(
                success=True,
                action=action,
                reasoning=conclusion,
                steps=steps,
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"CoT execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"CoT failed: {str(e)}",
                steps=steps,
            )

    def _define_question(self, context: NavContext) -> str:
        """Define the navigation question."""
        instruction = context.instruction

        # Identify key question components
        if context.subtasks:
            current = context.get_current_subtask()
            if current:
                return f"What is the best action to accomplish: '{current.description}'?"

        return f"How should I navigate to accomplish: '{instruction}'?"

    def _analyze_situation(self, context: NavContext, agents: List[BaseAgent]) -> str:
        """Analyze current navigation situation."""
        parts = []

        # Current state
        parts.append(f"Current step: {context.step_count}")
        parts.append(f"Position: {context.position}")

        # Room context
        if context.room_type != "unknown":
            parts.append(f"Location: {context.room_type}")

        # Visual context
        if context.visual_features.scene_description:
            parts.append(f"Scene: {context.visual_features.scene_description[:100]}")

        # History context
        if context.action_history:
            recent = context.get_action_summary(3)
            parts.append(f"Recent: {recent}")

        return " | ".join(parts)

    def _generate_reasoning_steps(
        self,
        context: NavContext,
        analysis: str,
        agents: List[BaseAgent],
    ) -> List[str]:
        """Generate explicit reasoning steps."""
        steps = []

        # Step 1: Understand instruction
        instruction_step = self._reason_about_instruction(context)
        steps.append(instruction_step)

        # Step 2: Assess environment
        environment_step = self._reason_about_environment(context)
        steps.append(environment_step)

        # Step 3: Consider constraints
        constraint_step = self._reason_about_constraints(context)
        steps.append(constraint_step)

        # Step 4: Evaluate options
        options_step = self._evaluate_options(context)
        steps.append(options_step)

        # Step 5: Select action
        selection_step = self._select_action(context)
        steps.append(selection_step)

        return steps[:self.max_reasoning_steps]

    def _reason_about_instruction(self, context: NavContext) -> str:
        """Reason about the instruction requirements."""
        instruction = context.instruction.lower()

        # Extract key elements
        has_turn = "turn" in instruction
        has_forward = any(w in instruction for w in ["go", "walk", "move", "forward"])
        has_stop = "stop" in instruction
        has_landmark = any(w in instruction for w in ["find", "look", "near", "next"])

        reasoning = "Analyzing instruction: "

        if has_turn:
            direction = "left" if "left" in instruction else "right" if "right" in instruction else "unspecified"
            reasoning += f"Requires turning {direction}. "

        if has_forward:
            reasoning += "Requires forward movement. "

        if has_landmark:
            reasoning += "Involves finding landmarks. "

        if has_stop:
            reasoning += "Has stop condition. "

        if not any([has_turn, has_forward, has_stop, has_landmark]):
            reasoning += "Standard navigation required. "

        return reasoning.strip()

    def _reason_about_environment(self, context: NavContext) -> str:
        """Reason about the current environment."""
        reasoning = "Environment assessment: "

        if context.room_type != "unknown":
            reasoning += f"In {context.room_type}. "

        if context.visual_features.object_detections:
            objects = [o.get("name", "") for o in context.visual_features.object_detections[:3]]
            reasoning += f"Visible objects: {', '.join(objects)}. "

        if context.metadata.get("trajectory"):
            progress = context.metadata["trajectory"].get("progress", 0)
            reasoning += f"Progress: {progress*100:.0f}%. "

        return reasoning.strip()

    def _reason_about_constraints(self, context: NavContext) -> str:
        """Reason about navigation constraints."""
        reasoning = "Constraints: "

        # Step limit
        max_steps = self.config.get("max_steps", 500)
        remaining = max_steps - context.step_count
        reasoning += f"Steps remaining: {remaining}. "

        # Check for obstacles
        if context.visual_features.object_detections:
            # Placeholder for obstacle consideration
            pass

        return reasoning.strip()

    def _evaluate_options(self, context: NavContext) -> str:
        """Evaluate navigation options."""
        options = []

        # Forward option
        options.append("forward: continue current trajectory")

        # Turn options
        if "left" in context.instruction.lower():
            options.append("turn_left: follow instruction")
        if "right" in context.instruction.lower():
            options.append("turn_right: follow instruction")

        # Stop option
        if context.subtasks and all(s.status == "completed" for s in context.subtasks):
            options.append("stop: all subtasks completed")

        return "Options: " + ", ".join(options) if options else "Options: forward, turn_left, turn_right, stop"

    def _select_action(self, context: NavContext) -> str:
        """Select the best action."""
        instruction_lower = context.instruction.lower()

        if "turn left" in instruction_lower:
            return "Selected: turn_left - aligns with instruction to turn left"
        elif "turn right" in instruction_lower:
            return "Selected: turn_right - aligns with instruction to turn right"
        elif "stop" in instruction_lower:
            return "Selected: stop - instruction indicates stopping"
        else:
            return "Selected: forward - continue navigation toward goal"

    def _derive_conclusion(
        self,
        context: NavContext,
        reasoning_steps: List[str],
        agents: List[BaseAgent],
    ) -> tuple:
        """Derive final conclusion and action."""
        # Get decision agent recommendation
        decision_agent = self.get_agent_by_role(agents, "decision")

        if decision_agent:
            output = decision_agent.process(context)
            if output.success:
                action_str = output.data.get("action", "forward")
                action = self._action_from_string(action_str)
                confidence = output.confidence

                conclusion = f"Based on reasoning chain, the optimal action is {action_str}"
                return conclusion, action, confidence

        # Fallback: derive from reasoning steps
        last_step = reasoning_steps[-1] if reasoning_steps else ""

        if "turn_left" in last_step:
            return last_step, Action.turn_left(), 0.7
        elif "turn_right" in last_step:
            return last_step, Action.turn_right(), 0.7
        elif "stop" in last_step:
            return last_step, Action.stop(), 0.8
        else:
            return last_step, Action.forward(), 0.6

    def _verify_action(
        self, context: NavContext, action: Action, reasoning_steps: List[str]
    ) -> Dict[str, Any]:
        """Verify the selected action is appropriate."""
        verification = {
            "valid": True,
            "concerns": [],
        }

        # Check if action makes sense given instruction
        instruction_lower = context.instruction.lower()
        action_type = action.action_type.name.lower()

        # Verify turn actions
        if "turn" in action_type:
            if "turn" not in instruction_lower and "left" not in instruction_lower and "right" not in instruction_lower:
                verification["concerns"].append("Turning without explicit turn instruction")

        # Verify stop action
        if action_type == "stop":
            if context.step_count < 5:
                verification["concerns"].append("Stopping very early in navigation")

        if verification["concerns"]:
            verification["valid"] = len(verification["concerns"]) == 0

        return verification

    def _re_evaluate(
        self, context: NavContext, reasoning_steps: List[str]
    ) -> tuple:
        """Re-evaluate decision after failed verification."""
        # Simple fallback: return forward action
        return Action.forward(confidence=0.5), 0.5

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