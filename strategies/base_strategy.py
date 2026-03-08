"""Base strategy class for VLN system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class StrategyType(Enum):
    """Strategy type enumeration."""
    REACT = "ReAct"
    COT = "CoT"
    DEBATE = "Debate"
    REFLECTION = "Reflection"


@dataclass
class StrategyResult:
    """Result from strategy execution."""

    success: bool
    action: Optional[Action] = None
    reasoning: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "action": self.action.to_habitat_action() if self.action else None,
            "reasoning": self.reasoning,
            "steps": self.steps,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Strategies define how agents collaborate to produce navigation decisions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy-specific configuration
        """
        self.config = config or {}
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass

    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Get strategy type."""
        pass

    @abstractmethod
    def execute(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        prev_result: Optional[StrategyResult] = None,
    ) -> StrategyResult:
        """
        Execute the strategy.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional result from previous strategy

        Returns:
            StrategyResult with action and reasoning
        """
        pass

    def initialize(self) -> None:
        """Initialize strategy resources."""
        self._initialized = True

    def get_agent_by_role(
        self, agents: List[BaseAgent], role: str
    ) -> Optional[BaseAgent]:
        """Get agent by role name."""
        for agent in agents:
            if agent.name.startswith(role) or agent.role.value == role:
                return agent
        return None

    def get_ordered_agents(self, agents: List[BaseAgent]) -> List[BaseAgent]:
        """Get agents in execution order."""
        from configs.architecture_config import get_ordered_agents

        agent_names = [a.name.replace("_agent", "") for a in agents]
        ordered_names = get_ordered_agents(agent_names)

        ordered_agents = []
        for name in ordered_names:
            for agent in agents:
                if agent.name.startswith(name):
                    ordered_agents.append(agent)
                    break

        return ordered_agents

    def validate_agents(self, agents: List[BaseAgent]) -> List[str]:
        """
        Validate that required agents are available.

        Returns:
            List of missing agent names
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class StrategyChain:
    """
    Chain of strategies to execute sequentially.
    """

    def __init__(self, strategies: List[BaseStrategy] = None):
        """
        Initialize strategy chain.

        Args:
            strategies: List of strategies to chain
        """
        self.strategies = strategies or []
        self._results: List[StrategyResult] = []

    def add_strategy(self, strategy: BaseStrategy) -> "StrategyChain":
        """Add strategy to chain."""
        self.strategies.append(strategy)
        return self

    def execute(self, context: NavContext, agents: List[BaseAgent]) -> StrategyResult:
        """
        Execute the strategy chain.

        Args:
            context: Navigation context
            agents: List of available agents

        Returns:
            Final strategy result
        """
        prev_result = None

        for strategy in self.strategies:
            result = strategy.execute(context, agents, prev_result)
            self._results.append(result)

            if not result.success:
                # Strategy failed, stop chain
                return result

            prev_result = result

        return self._results[-1] if self._results else StrategyResult(success=False)

    def get_all_results(self) -> List[StrategyResult]:
        """Get all strategy results."""
        return self._results.copy()

    def reset(self) -> None:
        """Reset chain state."""
        self._results.clear()