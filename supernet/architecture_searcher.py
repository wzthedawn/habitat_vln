"""Architecture searcher for optimal agent-strategy configuration."""

from typing import Dict, Any, List, Optional, Tuple
import logging
import random

from core.context import NavContext, TaskType
from core.action import Action


class ArchitectureSearcher:
    """
    Architecture Searcher for finding optimal agent-strategy combinations.

    Based on "Multi-agent Architecture Search via Agentic Supernet" paper,
    this class implements differentiable architecture search for VLN.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize architecture searcher.

        Args:
            config: Search configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("ArchitectureSearcher")

        # Search space
        self.agent_space = ["instruction", "perception", "trajectory", "decision"]
        self.strategy_space = ["ReAct", "CoT", "Debate", "Reflection"]

        # Architecture parameters (learnable)
        self._agent_weights: Dict[str, Dict[str, float]] = {}
        self._strategy_weights: Dict[str, Dict[str, float]] = {}

        # Search history
        self._search_history: List[Dict[str, Any]] = []

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize architecture weights uniformly."""
        for task_type in ["Type-0", "Type-1", "Type-2", "Type-3", "Type-4"]:
            # Agent weights
            self._agent_weights[task_type] = {
                agent: 1.0 / len(self.agent_space) for agent in self.agent_space
            }
            # Strategy weights
            self._strategy_weights[task_type] = {
                strategy: 1.0 / len(self.strategy_space)
                for strategy in self.strategy_space
            }

    def search(
        self,
        context: NavContext,
        evaluate_fn: callable,
        num_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Search for optimal architecture.

        Args:
            context: Navigation context
            evaluate_fn: Function to evaluate architecture performance
            num_samples: Number of architectures to sample

        Returns:
            Best architecture found
        """
        task_type = context.task_type.value

        best_architecture = None
        best_score = -float("inf")

        for _ in range(num_samples):
            # Sample architecture
            architecture = self._sample_architecture(task_type)

            # Evaluate
            try:
                score = evaluate_fn(context, architecture)

                # Record search
                self._search_history.append({
                    "task_type": task_type,
                    "architecture": architecture,
                    "score": score,
                })

                # Update best
                if score > best_score:
                    best_score = score
                    best_architecture = architecture

            except Exception as e:
                self.logger.warning(f"Architecture evaluation failed: {e}")

        # Update weights based on search
        if best_architecture:
            self._update_weights(task_type, best_architecture, best_score)

        return best_architecture or self._get_default_architecture(task_type)

    def _sample_architecture(self, task_type: str) -> Dict[str, Any]:
        """Sample an architecture based on current weights."""
        weights = self._agent_weights[task_type]
        strategy_weights = self._strategy_weights[task_type]

        # Sample agents (keep decision agent always)
        selected_agents = ["decision"]
        for agent in ["instruction", "perception", "trajectory"]:
            if random.random() < weights[agent]:
                selected_agents.append(agent)

        # Sample strategies
        selected_strategies = []
        for strategy in self.strategy_space:
            if random.random() < strategy_weights[strategy]:
                selected_strategies.append(strategy)

        return {
            "agents": list(set(selected_agents)),
            "strategies": selected_strategies or ["ReAct"],
        }

    def _get_default_architecture(self, task_type: str) -> Dict[str, Any]:
        """Get default architecture for task type."""
        defaults = {
            "Type-0": {"agents": [], "strategies": []},
            "Type-1": {"agents": ["perception", "decision"], "strategies": ["ReAct"]},
            "Type-2": {"agents": ["perception", "trajectory", "decision"], "strategies": ["ReAct", "CoT"]},
            "Type-3": {"agents": ["instruction", "perception", "trajectory", "decision"], "strategies": ["CoT", "Reflection"]},
            "Type-4": {"agents": ["instruction", "perception", "trajectory", "decision"], "strategies": ["CoT", "Debate", "Reflection"]},
        }
        return defaults.get(task_type, defaults["Type-1"])

    def _update_weights(
        self, task_type: str, architecture: Dict[str, Any], score: float
    ) -> None:
        """Update architecture weights based on performance."""
        learning_rate = self.config.get("learning_rate", 0.1)

        # Normalize score to [0, 1]
        normalized_score = max(0, min(1, score))

        # Update agent weights
        for agent in self.agent_space:
            if agent in architecture["agents"]:
                # Increase weight for selected agents
                delta = learning_rate * normalized_score
            else:
                # Decrease weight for non-selected agents
                delta = -learning_rate * (1 - normalized_score) / (len(self.agent_space) - 1)

            new_weight = self._agent_weights[task_type][agent] + delta
            self._agent_weights[task_type][agent] = max(0.05, min(0.95, new_weight))

        # Update strategy weights
        for strategy in self.strategy_space:
            if strategy in architecture["strategies"]:
                delta = learning_rate * normalized_score
            else:
                delta = -learning_rate * (1 - normalized_score) / (len(self.strategy_space) - 1)

            new_weight = self._strategy_weights[task_type][strategy] + delta
            self._strategy_weights[task_type][strategy] = max(0.05, min(0.95, new_weight))

        # Normalize weights
        self._normalize_weights(task_type)

    def _normalize_weights(self, task_type: str) -> None:
        """Normalize weights to sum to 1."""
        # Normalize agent weights
        total = sum(self._agent_weights[task_type].values())
        if total > 0:
            for agent in self._agent_weights[task_type]:
                self._agent_weights[task_type][agent] /= total

        # Normalize strategy weights
        total = sum(self._strategy_weights[task_type].values())
        if total > 0:
            for strategy in self._strategy_weights[task_type]:
                self._strategy_weights[task_type][strategy] /= total

    def get_architecture_weights(self, task_type: str) -> Dict[str, Any]:
        """Get current architecture weights for a task type."""
        return {
            "agents": self._agent_weights.get(task_type, {}),
            "strategies": self._strategy_weights.get(task_type, {}),
        }

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history."""
        return self._search_history.copy()

    def reset(self) -> None:
        """Reset searcher state."""
        self._initialize_weights()
        self._search_history.clear()


class DifferentiableArchitectureSearcher(ArchitectureSearcher):
    """
    Differentiable Architecture Search (DARTS) for VLN.

    Implements continuous relaxation of the architecture space
    for gradient-based optimization.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Continuous architecture parameters
        self._alpha_agents: Dict[str, Dict[str, float]] = {}
        self._alpha_strategies: Dict[str, Dict[str, float]] = {}

        self._initialize_alpha()

    def _initialize_alpha(self) -> None:
        """Initialize architecture parameters."""
        import random
        for task_type in ["Type-0", "Type-1", "Type-2", "Type-3", "Type-4"]:
            self._alpha_agents[task_type] = {
                agent: random.uniform(-0.1, 0.1) for agent in self.agent_space
            }
            self._alpha_strategies[task_type] = {
                strategy: random.uniform(-0.1, 0.1)
                for strategy in self.strategy_space
            }

    def get_architecture_probs(self, task_type: str) -> Dict[str, Dict[str, float]]:
        """Get softmax probabilities from architecture parameters."""
        import math

        def softmax(d: Dict[str, float]) -> Dict[str, float]:
            max_val = max(d.values())
            exp_vals = {k: math.exp(v - max_val) for k, v in d.items()}
            total = sum(exp_vals.values())
            return {k: v / total for k, v in exp_vals.items()}

        return {
            "agents": softmax(self._alpha_agents[task_type]),
            "strategies": softmax(self._alpha_strategies[task_type]),
        }

    def update_alpha(
        self, task_type: str, gradients: Dict[str, Dict[str, float]]
    ) -> None:
        """Update architecture parameters with gradients."""
        lr = self.config.get("architecture_lr", 0.01)

        for agent in self._alpha_agents[task_type]:
            if agent in gradients.get("agents", {}):
                self._alpha_agents[task_type][agent] += lr * gradients["agents"][agent]

        for strategy in self._alpha_strategies[task_type]:
            if strategy in gradients.get("strategies", {}):
                self._alpha_strategies[task_type][strategy] += lr * gradients["strategies"][strategy]