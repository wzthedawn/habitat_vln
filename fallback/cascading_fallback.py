"""Cascading fallback for multi-level degradation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from core.context import NavContext
from core.action import Action


@dataclass
class FallbackLevel:
    """Configuration for a fallback level."""
    name: str
    agents: List[str]
    strategies: List[str]
    model_tier: str
    description: str


class CascadingFallback:
    """
    Cascading Fallback for multi-level degradation.

    Implements progressive degradation from complex multi-agent
    systems to simple rule-based navigation.
    """

    # Default fallback levels
    DEFAULT_LEVELS = [
        FallbackLevel(
            name="strong_full",
            agents=["instruction", "perception", "trajectory", "decision"],
            strategies=["CoT", "Debate", "Reflection"],
            model_tier="cloud_large",
            description="Full multi-agent system with complex strategies",
        ),
        FallbackLevel(
            name="strong_simplified",
            agents=["perception", "decision"],
            strategies=["ReAct"],
            model_tier="local_medium",
            description="Simplified multi-agent with basic strategy",
        ),
        FallbackLevel(
            name="strong_minimal",
            agents=["decision"],
            strategies=["ReAct"],
            model_tier="local_small",
            description="Minimal agent configuration",
        ),
        FallbackLevel(
            name="weak",
            agents=[],
            strategies=[],
            model_tier="local_small",
            description="Local model only",
        ),
        FallbackLevel(
            name="safe",
            agents=[],
            strategies=[],
            model_tier="none",
            description="Safe fallback action",
        ),
    ]

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cascading fallback.

        Args:
            config: Fallback configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("CascadingFallback")

        # Initialize levels
        self._levels = self._initialize_levels()

        # Current level index
        self._current_level = 0

        # Failure tracking
        self._level_failures: Dict[str, int] = {level.name: 0 for level in self._levels}

    def _initialize_levels(self) -> List[FallbackLevel]:
        """Initialize fallback levels from config."""
        custom_levels = self.config.get("levels", [])

        if custom_levels:
            levels = []
            for level_config in custom_levels:
                level = FallbackLevel(
                    name=level_config.get("name", "custom"),
                    agents=level_config.get("agents", []),
                    strategies=level_config.get("strategies", []),
                    model_tier=level_config.get("model_tier", "local_small"),
                    description=level_config.get("description", ""),
                )
                levels.append(level)
            return levels

        return self.DEFAULT_LEVELS.copy()

    def handle_failure(
        self,
        context: NavContext,
        failure_info: Any,
    ) -> Action:
        """
        Handle failure with cascading fallback.

        Args:
            context: Navigation context
            failure_info: Failure information

        Returns:
            Action from fallback level
        """
        # Record failure at current level
        current_level = self._levels[self._current_level]
        self._level_failures[current_level.name] += 1

        # Decide whether to degrade
        if self._should_degrade(current_level):
            self._degrade_level()
            self.logger.info(f"Degraded to level: {self._levels[self._current_level].name}")

        # Execute at current level
        return self._execute_level(context, self._levels[self._current_level])

    def _should_degrade(self, current_level: FallbackLevel) -> bool:
        """
        Determine if should degrade to lower level.

        Args:
            current_level: Current fallback level

        Returns:
            True if should degrade
        """
        # Check if at lowest level
        if self._current_level >= len(self._levels) - 1:
            return False

        # Check failure count
        failure_count = self._level_failures[current_level.name]
        degrade_threshold = self.config.get("degrade_threshold", 2)

        return failure_count >= degrade_threshold

    def _degrade_level(self) -> None:
        """Degrade to next lower level."""
        if self._current_level < len(self._levels) - 1:
            self._current_level += 1

    def _execute_level(
        self,
        context: NavContext,
        level: FallbackLevel,
    ) -> Action:
        """
        Execute navigation at a specific fallback level.

        Args:
            context: Navigation context
            level: Fallback level configuration

        Returns:
            Navigation action
        """
        if level.name == "safe":
            return self._safe_action(context)

        if level.name == "weak":
            return self._weak_level_action(context)

        return self._strong_level_action(context, level)

    def _safe_action(self, context: NavContext) -> Action:
        """
        Generate a safe fallback action.

        Args:
            context: Navigation context

        Returns:
            Safe action (stop or observe)
        """
        # Return stop as safest action
        return Action.stop(
            confidence=0.3,
            reasoning="Safe fallback: stopping due to repeated failures"
        )

    def _weak_level_action(self, context: NavContext) -> Action:
        """
        Execute weak level (local model only).

        Args:
            context: Navigation context

        Returns:
            Action from local model
        """
        try:
            from models.local_model import LocalModel
            model = LocalModel(self.config.get("local_model", {}))
            action_str = model.predict(context)
            return self._action_from_string(action_str)
        except Exception as e:
            self.logger.error(f"Weak level failed: {e}")
            return self._safe_action(context)

    def _strong_level_action(
        self,
        context: NavContext,
        level: FallbackLevel,
    ) -> Action:
        """
        Execute strong level with specified agents.

        Args:
            context: Navigation context
            level: Fallback level configuration

        Returns:
            Action from multi-agent system
        """
        try:
            # Create agent instances
            agents = self._create_agents(level.agents)

            # Use decision agent if available
            for agent in agents:
                if "decision" in agent.name.lower():
                    output = agent.process(context)
                    if output.success:
                        action_str = output.data.get("action", "forward")
                        return self._action_from_string(action_str)

            # Fallback to keyword-based
            return self._keyword_action(context)

        except Exception as e:
            self.logger.error(f"Strong level failed: {e}")
            return self._weak_level_action(context)

    def _create_agents(self, agent_names: List[str]) -> List[Any]:
        """Create agent instances."""
        agents = []

        agent_classes = {
            "instruction": ("agents.instruction_agent", "InstructionAgent"),
            "perception": ("agents.perception_agent", "PerceptionAgent"),
            "trajectory": ("agents.trajectory_agent", "TrajectoryAgent"),
            "decision": ("agents.decision_agent", "DecisionAgent"),
        }

        for name in agent_names:
            if name in agent_classes:
                try:
                    import importlib
                    module_path, class_name = agent_classes[name]
                    module = importlib.import_module(module_path)
                    agent_class = getattr(module, class_name)
                    agents.append(agent_class())
                except Exception as e:
                    self.logger.warning(f"Failed to create agent {name}: {e}")

        return agents

    def _keyword_action(self, context: NavContext) -> Action:
        """Generate action based on keywords."""
        instruction_lower = context.instruction.lower()

        if "left" in instruction_lower:
            return Action.turn_left(confidence=0.5)
        elif "right" in instruction_lower:
            return Action.turn_right(confidence=0.5)
        elif "stop" in instruction_lower:
            return Action.stop(confidence=0.5)
        else:
            return Action.forward(confidence=0.5)

    def _action_from_string(self, action_str: str) -> Action:
        """Convert action string to Action object."""
        action_map = {
            "stop": Action.stop(),
            "forward": Action.forward(),
            "move_forward": Action.forward(),
            "turn_left": Action.turn_left(),
            "turn_right": Action.turn_right(),
        }
        return action_map.get(action_str.lower(), Action.forward())

    def get_current_level(self) -> FallbackLevel:
        """Get current fallback level."""
        return self._levels[self._current_level]

    def get_level_name(self) -> str:
        """Get name of current level."""
        return self._levels[self._current_level].name

    def reset(self) -> None:
        """Reset to highest level."""
        self._current_level = 0
        for name in self._level_failures:
            self._level_failures[name] = 0

    def get_failure_stats(self) -> Dict[str, int]:
        """Get failure statistics per level."""
        return self._level_failures.copy()


class CascadingFallbackBuilder:
    """Builder for CascadingFallback instances."""

    def __init__(self):
        self._config: Dict[str, Any] = {}

    def with_max_level(self, max_level: int) -> "CascadingFallbackBuilder":
        """Set maximum fallback level."""
        self._config["max_level"] = max_level
        return self

    def with_degrade_threshold(self, threshold: int) -> "CascadingFallbackBuilder":
        """Set degrade threshold."""
        self._config["degrade_threshold"] = threshold
        return self

    def with_custom_levels(self, levels: List[Dict]) -> "CascadingFallbackBuilder":
        """Set custom fallback levels."""
        self._config["levels"] = levels
        return self

    def build(self) -> CascadingFallback:
        """Build cascading fallback instance."""
        return CascadingFallback(self._config)