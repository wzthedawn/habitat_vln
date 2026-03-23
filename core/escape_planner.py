"""Escape Planner for VLN multi-agent system.

This module provides multi-step escape sequence generation
for stuck situations.

Phase 3 of Debate Strategy Redesign.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import logging
import math

from .debate_types import DebateOpinion, ACTION_NAMES


@dataclass
class EscapeStep:
    """A single step in an escape sequence."""
    action: str                      # Action to execute
    purpose: str                     # Why this action
    condition: Optional[str] = None  # Execution condition (e.g., "obstacle_forward")
    abort_if_fail: bool = True       # Whether to abort sequence on failure


@dataclass
class EscapePlan:
    """Multi-step escape plan.

    Updated 2026-03-19: Extended max_steps from 5 to 15 for better escape sequences.
    """
    trigger_stuck_steps: int                            # How many steps stuck when triggered
    steps: List[EscapeStep] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 15  # Increased from 5 to 15 for extended escape sequences
    current_step: int = 0
    start_position: Optional[Tuple[float, float, float]] = None

    def get_next_action(self) -> Optional[str]:
        """Get the next action to execute."""
        if self.current_step >= len(self.steps):
            return None
        return self.steps[self.current_step].action

    def advance(self) -> bool:
        """Advance to the next step.

        Returns:
            True if there are more steps, False if complete
        """
        self.current_step += 1
        return self.current_step < len(self.steps)

    def is_complete(self) -> bool:
        """Check if the plan is complete."""
        return self.current_step >= len(self.steps)

    def get_current_step(self) -> Optional[EscapeStep]:
        """Get the current step."""
        if self.current_step >= len(self.steps):
            return None
        return self.steps[self.current_step]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trigger_stuck_steps": self.trigger_stuck_steps,
            "steps": [
                {
                    "action": s.action,
                    "purpose": s.purpose,
                    "condition": s.condition,
                    "abort_if_fail": s.abort_if_fail,
                }
                for s in self.steps
            ],
            "success_criteria": self.success_criteria,
            "max_steps": self.max_steps,
            "current_step": self.current_step,
            "start_position": self.start_position,
        }


class EscapePlanner:
    """Generator for multi-step escape sequences.

    Analyzes debate opinions and context to create effective
    escape plans when the agent is stuck.

    Updated 2026-03-19: Extended max_steps to 15 with multi-phase escape sequences.

    Sequence phases (7-15 steps):
    - Phase 1 (Steps 1-3): Multiple turns to find open direction
    - Phase 2 (Steps 4-6): Continuous forward movement
    - Phase 3 (Steps 7-9): Direction adjustment
    - Phase 4 (Steps 10-12): Forward towards goal
    - Phase 5 (Steps 13-15): Final adjustment
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the escape planner.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("EscapePlanner")

        # Configuration parameters - updated defaults
        self.max_steps = self.config.get("max_escape_steps", 15)  # Increased from 5
        self.success_distance = self.config.get("success_distance", 1.0)

    def generate_plan(
        self,
        stuck_steps: int,
        opinions: List[DebateOpinion],
        context: Any,  # NavContext
        failed_directions: List[str] = None
    ) -> EscapePlan:
        """Generate an escape plan based on debate opinions.

        Updated 2026-03-19: Extended to support up to 15 steps with multi-phase sequences.

        Args:
            stuck_steps: Number of steps the agent has been stuck
            opinions: List of DebateOpinions from agents
            context: Navigation context
            failed_directions: Previously failed escape directions

        Returns:
            EscapePlan with sequence of steps (up to 15 steps)
        """
        failed_directions = failed_directions or []

        # Extract information from opinions
        obstacle_info = self._extract_obstacle_info(opinions)
        open_directions = self._extract_open_directions(opinions)
        goal_direction = self._extract_goal_direction(opinions)
        suggested_turn = self._get_suggested_turn(opinions, context, failed_directions)

        # Build simple forward-only escape steps
        steps = []

        # Only forward movements
        for i in range(self.max_steps):
            steps.append(EscapeStep(
                action="move_forward",
                purpose=f"前进({i+1})",
                abort_if_fail=(i == 0),
            ))

        # Limit to max steps
        steps = steps[:self.max_steps]

        self.logger.info(
            f"[ESCAPE PLAN] Generated {len(steps)}-step plan "
            f"(triggered at {stuck_steps} stuck steps)"
        )

        return EscapePlan(
            trigger_stuck_steps=stuck_steps,
            steps=steps,
            success_criteria={
                "position_change": f"> {self.success_distance}m",
                "or_stuck_resolved": True,
            },
            max_steps=self.max_steps,
            start_position=context.position if hasattr(context, 'position') else None,
        )

    def generate_simple_plan(
        self,
        stuck_steps: int,
        suggested_action: str,
        context: Any
    ) -> EscapePlan:
        """Generate a simple escape plan from a single action.

        Updated 2026-03-19: Extended to support more steps for better escape.

        Args:
            stuck_steps: Number of steps stuck
            suggested_action: Primary suggested action
            context: Navigation context

        Returns:
            Simple EscapePlan
        """
        steps = []

        # Build simple forward-only escape steps
        steps = []

        # Only forward movements
        for i in range(self.max_steps):
            steps.append(EscapeStep(
                action="move_forward",
                purpose=f"前进({i+1})",
                abort_if_fail=(i == 0),
            ))

        # Limit to max steps
        steps = steps[:self.max_steps]

        return EscapePlan(
            trigger_stuck_steps=stuck_steps,
            steps=steps,
            success_criteria={"position_change": f"> {self.success_distance}m"},
            max_steps=self.max_steps,
            start_position=context.position if hasattr(context, 'position') else None,
        )

    def _extract_obstacle_info(self, opinions: List[DebateOpinion]) -> List[Dict[str, Any]]:
        """Extract obstacle information from opinions."""
        for opinion in opinions:
            if opinion.agent == "perception":
                return opinion.evidence.get("obstacles", [])
        return []

    def _extract_open_directions(self, opinions: List[DebateOpinion]) -> List[str]:
        """Extract open directions from opinions."""
        for opinion in opinions:
            if opinion.agent == "perception":
                return opinion.evidence.get("open_directions", ["right"])
        return ["right"]

    def _extract_goal_direction(self, opinions: List[DebateOpinion]) -> Optional[str]:
        """Extract goal direction from instruction opinion."""
        for opinion in opinions:
            if opinion.agent == "instruction":
                return opinion.evidence.get("target_direction")
        return None

    def _get_suggested_turn(
        self,
        opinions: List[DebateOpinion],
        context: Any,
        failed_directions: List[str]
    ) -> Optional[str]:
        """Get the best suggested turn direction.

        Considers:
        1. Perception agent's suggestion
        2. Failed directions to avoid
        3. Trajectory analysis
        """
        # First check perception opinion
        for opinion in opinions:
            if opinion.agent == "perception":
                suggested = opinion.evidence.get("suggested_turn")
                if suggested and f"turn_{suggested}" not in failed_directions:
                    return f"turn_{suggested}"

        # Check trajectory opinion
        for opinion in opinions:
            if opinion.agent == "trajectory":
                action = opinion.primary_action
                if action in ["turn_left", "turn_right"] and action not in failed_directions:
                    return action

        # Default based on rotation
        if hasattr(context, 'rotation'):
            # Try a different direction based on recent turns
            if "turn_left" in failed_directions:
                return "turn_right"
            elif "turn_right" in failed_directions:
                return "turn_left"

        return "turn_right"  # Default


class EscapeVerifier:
    """Verifies escape sequence results."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the verifier."""
        self.config = config or {}
        self.logger = logging.getLogger("EscapeVerifier")
        self.success_distance = self.config.get("success_distance", 1.0)

    def verify(
        self,
        escape_plan: EscapePlan,
        context: Any,
        trajectory_output: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Verify if escape was successful.

        Args:
            escape_plan: The escape plan that was executed
            context: Current navigation context
            trajectory_output: Output from trajectory agent

        Returns:
            Verification result dictionary
        """
        if escape_plan.start_position is None:
            return {
                "success": True,
                "position_change": 0.0,
                "still_stuck": False,
                "visited_new_area": True,
                "action": "continue_subtask",
            }

        # Calculate position change
        current_pos = context.position if hasattr(context, 'position') else (0, 0, 0)
        dx = current_pos[0] - escape_plan.start_position[0]
        dz = current_pos[2] - escape_plan.start_position[2]
        position_change = math.sqrt(dx * dx + dz * dz)

        # Check if still stuck (use trajectory output if available)
        still_stuck = self._check_still_stuck(context, trajectory_output)

        # Check if visited new area
        visited_new_area = self._check_new_area(context, trajectory_output)

        # Determine success
        success = (
            position_change > self.success_distance or
            (not still_stuck and position_change > 0.5) or
            visited_new_area
        )

        # Determine next action
        if success:
            action = "continue_subtask"
        else:
            action = "redebate"

        return {
            "success": success,
            "position_change": position_change,
            "still_stuck": still_stuck,
            "visited_new_area": visited_new_area,
            "action": action,
        }

    def _check_still_stuck(self, context: Any, trajectory_output: Dict[str, Any]) -> bool:
        """Check if agent is still stuck."""
        # Check context stuck state
        if hasattr(context, 'is_stuck') and context.is_stuck:
            return True

        # Check trajectory corrections
        if trajectory_output:
            corrections = trajectory_output.get("corrections", [])
            if any(c.get("type") == "stuck" for c in corrections):
                return True

        return False

    def _check_new_area(self, context: Any, trajectory_output: Dict[str, Any]) -> bool:
        """Check if agent visited a new area."""
        if trajectory_output:
            return trajectory_output.get("visited_new_area", False)

        # Check trajectory length change
        if hasattr(context, 'trajectory') and len(context.trajectory) > 1:
            # Simple heuristic: if we've moved, we're in a new area
            recent = context.trajectory[-5:] if len(context.trajectory) >= 5 else context.trajectory
            if len(recent) >= 2:
                dx = recent[-1][0] - recent[0][0]
                dz = recent[-1][2] - recent[0][2]
                return math.sqrt(dx * dx + dz * dz) > 0.5

        return False