"""SubAgentRegistry - Registry for SubAgent registration and invocation."""

from typing import Any, Dict, Optional


class SubAgentRegistry:
    """Registry for managing SubAgent instances.

    Provides methods to register, retrieve, call, and list SubAgents.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._agents: Dict[str, Any] = {}

    def register(self, name: str, agent: Any) -> None:
        """Register a SubAgent with a given name.

        Args:
            name: Unique identifier for the agent.
            agent: The SubAgent instance to register.
        """
        self._agents[name] = agent

    def get(self, name: str) -> Optional[Any]:
        """Get a SubAgent by name.

        Args:
            name: The identifier of the SubAgent to retrieve.

        Returns:
            The SubAgent instance if found, None otherwise.
        """
        return self._agents.get(name)

    def call(self, name: str, *args, **kwargs) -> Any:
        """Call a SubAgent's process method.

        Args:
            name: The identifier of the SubAgent to call.
            *args: Positional arguments to pass to process.
            **kwargs: Keyword arguments to pass to process.

        Returns:
            The result of the SubAgent's process method.

        Raises:
            KeyError: If no agent is registered with the given name.
        """
        agent = self._agents.get(name)
        if agent is None:
            raise KeyError(f"No SubAgent registered with name '{name}'")
        return agent.process(*args, **kwargs)

    def list_all(self) -> Dict[str, Any]:
        """List all registered SubAgents.

        Returns:
            A dictionary mapping agent names to their instances.
        """
        return dict(self._agents)

    def clear(self) -> None:
        """Clear all registered SubAgents."""
        self._agents.clear()