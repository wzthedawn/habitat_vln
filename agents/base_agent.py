"""Base agent class for VLN system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from core.context import NavContext


class AgentRole(Enum):
    """Agent role enumeration."""
    INSTRUCTION = "instruction"
    PERCEPTION = "perception"
    TRAJECTORY = "trajectory"
    DECISION = "decision"


@dataclass
class AgentOutput:
    """Output from an agent's processing."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reasoning: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def success_output(
        cls, data: Dict[str, Any], confidence: float = 1.0, reasoning: str = ""
    ) -> "AgentOutput":
        """Create a successful output."""
        return cls(success=True, data=data, confidence=confidence, reasoning=reasoning)

    @classmethod
    def failure_output(cls, errors: List[str], reasoning: str = "") -> "AgentOutput":
        """Create a failure output."""
        return cls(success=False, errors=errors, reasoning=reasoning)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    All specialized agents (Instruction, Perception, Trajectory, Decision)
    inherit from this class and implement the process method.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize agent.

        Args:
            config: Agent-specific configuration
        """
        self.config = config or {}
        self._model = None
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Get agent name."""
        pass

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Get agent role."""
        pass

    @abstractmethod
    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Process navigation context and produce output.

        Args:
            context: Navigation context
            strategy_result: Optional output from previous strategy

        Returns:
            AgentOutput with processing results
        """
        pass

    def initialize(self) -> None:
        """Initialize agent resources (lazy loading)."""
        if not self._initialized:
            self._model = self.get_model()
            self._initialized = True

    def get_model(self, task_type: str = None) -> Any:
        """
        Get appropriate model for the task.

        Args:
            task_type: Optional task type for model selection

        Returns:
            Model instance
        """
        from models.model_selector import ModelSelector

        selector = ModelSelector(self.config.get("model_selector", {}))
        return selector.select_model(task_type or "Type-1")

    def validate_context(self, context: NavContext) -> List[str]:
        """
        Validate that context has required data.

        Args:
            context: Navigation context

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not context.instruction:
            errors.append("Missing instruction")

        return errors

    def get_required_inputs(self) -> List[str]:
        """Get list of required input keys."""
        return []

    def get_output_keys(self) -> List[str]:
        """Get list of output keys this agent produces."""
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class AgentRegistry:
    """Registry for agent instances."""

    _agents: Dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """Register an agent."""
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return cls._agents.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return cls._agents.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered agents."""
        cls._agents.clear()