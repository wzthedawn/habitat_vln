"""Supernet main class for agent and strategy orchestration."""

from typing import Dict, Any, List, Optional
import logging

from core.context import NavContext, TaskType
from core.action import Action
from configs.architecture_config import get_architecture_for_task, get_ordered_strategies


class Supernet:
    """
    Supernet: The main orchestration layer for multi-agent navigation.

    Based on the "Multi-agent Architecture Search via Agentic Supernet" paper,
    this class manages the dynamic selection and execution of agent-strategy
    combinations based on task type.
    """

    # Architecture configuration mapping
    ARCHITECTURE_CONFIG = {
        "Type-0": {"level": "weak", "agents": [], "strategies": []},
        "Type-1": {"level": "strong", "agents": ["perception", "decision"], "strategies": ["ReAct"]},
        "Type-2": {"level": "strong", "agents": ["perception", "trajectory", "decision"], "strategies": ["ReAct", "CoT"]},
        "Type-3": {"level": "strong", "agents": ["instruction", "perception", "trajectory", "decision"], "strategies": ["CoT", "Reflection"]},
        "Type-4": {"level": "strong", "agents": ["instruction", "perception", "trajectory", "decision"], "strategies": ["CoT", "Debate", "Reflection"]},
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Supernet.

        Args:
            config: Supernet configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("Supernet")

        # Agent and strategy pools
        self._agents: Dict[str, Any] = {}
        self._strategies: Dict[str, Any] = {}

        # Initialize pools
        self._initialize_pools()

        # Statistics
        self._stats = {
            "total_forwards": 0,
            "weak_level_calls": 0,
            "strong_level_calls": 0,
            "by_task_type": {f"Type-{i}": 0 for i in range(5)},
        }

    def _initialize_pools(self) -> None:
        """Initialize agent and strategy pools."""
        # Lazy initialization - agents and strategies are loaded on demand
        pass

    def _get_agent(self, name: str) -> Any:
        """Get or create an agent by name."""
        if name not in self._agents:
            self._agents[name] = self._create_agent(name)
        return self._agents[name]

    def _create_agent(self, name: str) -> Any:
        """Create an agent instance."""
        agent_classes = {
            "instruction": "agents.instruction_agent.InstructionAgent",
            "perception": "agents.perception_agent.PerceptionAgent",
            "trajectory": "agents.trajectory_agent.TrajectoryAgent",
            "decision": "agents.decision_agent.DecisionAgent",
        }

        class_path = agent_classes.get(name)
        if class_path:
            module_path, class_name = class_path.rsplit(".", 1)
            try:
                import importlib
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
                return agent_class(self.config.get(f"{name}_agent", {}))
            except Exception as e:
                self.logger.warning(f"Failed to load agent {name}: {e}")

        # Return None if agent cannot be created
        return None

    def _get_strategy(self, name: str) -> Any:
        """Get or create a strategy by name."""
        if name not in self._strategies:
            self._strategies[name] = self._create_strategy(name)
        return self._strategies[name]

    def _create_strategy(self, name: str) -> Any:
        """Create a strategy instance."""
        strategy_classes = {
            "ReAct": "strategies.react.ReActStrategy",
            "CoT": "strategies.cot.CoTStrategy",
            "Debate": "strategies.debate.DebateStrategy",
            "Reflection": "strategies.reflection.ReflectionStrategy",
        }

        class_path = strategy_classes.get(name)
        if class_path:
            module_path, class_name = class_path.rsplit(".", 1)
            try:
                import importlib
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                return strategy_class(self.config.get(f"{name.lower()}_strategy", {}))
            except Exception as e:
                self.logger.warning(f"Failed to load strategy {name}: {e}")

        return None

    def forward(self, context: NavContext) -> Action:
        """
        Execute forward pass through the supernet.

        Args:
            context: Navigation context with task type already classified

        Returns:
            Navigation action
        """
        self._stats["total_forwards"] += 1

        task_type = context.task_type.value
        self._stats["by_task_type"][task_type] = (
            self._stats["by_task_type"].get(task_type, 0) + 1
        )

        config = self.ARCHITECTURE_CONFIG.get(task_type, self.ARCHITECTURE_CONFIG["Type-0"])

        if config["level"] == "weak":
            return self._weak_level_forward(context)
        else:
            return self._strong_level_forward(context, config)

    def _weak_level_forward(self, context: NavContext) -> Action:
        """
        Execute weak level navigation (local small model only).

        Args:
            context: Navigation context

        Returns:
            Action from local model
        """
        self._stats["weak_level_calls"] += 1

        try:
            from models.local_model import LocalModel
            model = LocalModel(self.config.get("local_model", {}))
            action_str = model.predict(context)
            return self._action_from_string(action_str)
        except Exception as e:
            self.logger.error(f"Weak level forward failed: {e}")
            return self._fallback_action(context)

    def _strong_level_forward(
        self, context: NavContext, config: Dict[str, Any]
    ) -> Action:
        """
        Execute strong level navigation (multi-agent collaboration).

        Args:
            context: Navigation context
            config: Architecture configuration

        Returns:
            Action from multi-agent system
        """
        self._stats["strong_level_calls"] += 1

        # Get required agents
        agent_names = config.get("agents", [])
        agents = [self._get_agent(name) for name in agent_names]
        agents = [a for a in agents if a is not None]  # Filter None

        # Get required strategies
        strategy_names = config.get("strategies", [])
        strategy_names = get_ordered_strategies(strategy_names)

        # Execute agents first
        agent_outputs = {}
        for agent in agents:
            if agent:
                try:
                    output = agent.process(context)
                    agent_outputs[agent.name] = output
                    context.metadata[f"{agent.name}_output"] = output.to_dict()
                except Exception as e:
                    self.logger.warning(f"Agent {agent.name} failed: {e}")

        # Execute strategy chain
        prev_result = None
        final_result = None

        for strategy_name in strategy_names:
            strategy = self._get_strategy(strategy_name)
            if strategy:
                try:
                    result = strategy.execute(context, agents, prev_result)
                    if result.success:
                        prev_result = result
                        final_result = result
                    else:
                        self.logger.warning(f"Strategy {strategy_name} failed")
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_name} error: {e}")

        # Return final action
        if final_result and final_result.action:
            return final_result.action

        # Fallback: use decision agent directly
        if agents:
            decision_agent = agents[-1]  # Decision agent is usually last
            if decision_agent and "decision" in decision_agent.name:
                output = decision_agent.process(context)
                if output.success:
                    action_str = output.data.get("action", "forward")
                    return self._action_from_string(action_str)

        return self._fallback_action(context)

    def _fallback_action(self, context: NavContext) -> Action:
        """Generate fallback action when normal flow fails."""
        # Simple rule-based fallback
        instruction_lower = context.instruction.lower()

        if "left" in instruction_lower:
            return Action.turn_left(confidence=0.5, reasoning="Fallback: keyword-based")
        elif "right" in instruction_lower:
            return Action.turn_right(confidence=0.5, reasoning="Fallback: keyword-based")
        elif "stop" in instruction_lower:
            return Action.stop(confidence=0.5, reasoning="Fallback: keyword-based")
        else:
            return Action.forward(confidence=0.5, reasoning="Fallback: default forward")

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

    def get_statistics(self) -> Dict[str, Any]:
        """Get supernet statistics."""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics."""
        for key in self._stats:
            if isinstance(self._stats[key], dict):
                for k in self._stats[key]:
                    self._stats[key][k] = 0
            else:
                self._stats[key] = 0


class SupernetBuilder:
    """Builder for Supernet instances."""

    def __init__(self):
        self._config: Dict[str, Any] = {}

    def with_agent_config(self, agent_name: str, config: Dict[str, Any]) -> "SupernetBuilder":
        """Set agent configuration."""
        self._config[f"{agent_name}_agent"] = config
        return self

    def with_strategy_config(self, strategy_name: str, config: Dict[str, Any]) -> "SupernetBuilder":
        """Set strategy configuration."""
        self._config[f"{strategy_name.lower()}_strategy"] = config
        return self

    def with_local_model_config(self, config: Dict[str, Any]) -> "SupernetBuilder":
        """Set local model configuration."""
        self._config["local_model"] = config
        return self

    def build(self) -> Supernet:
        """Build Supernet instance."""
        return Supernet(self._config)