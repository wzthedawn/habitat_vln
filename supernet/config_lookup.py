"""Configuration lookup table for architecture mappings."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from configs.model_config import get_model_for_task, ModelTier
from configs.architecture_config import (
    get_architecture_for_task,
    get_ordered_agents,
    get_ordered_strategies,
)


@dataclass
class ArchitectureEntry:
    """Entry in the configuration lookup table."""
    task_type: str
    level: str
    agents: List[str]
    strategies: List[str]
    model_tier: str
    max_tokens_budget: int
    description: str


class ConfigLookup:
    """
    Configuration Lookup Table.

    Provides fast access to architecture configurations, model mappings,
    and token budgets for each task type.
    """

    # Pre-computed lookup table
    LOOKUP_TABLE: Dict[str, ArchitectureEntry] = {}

    @classmethod
    def initialize(cls) -> None:
        """Initialize the lookup table."""
        from configs.architecture_config import ARCHITECTURE_CONFIG

        for task_type, config in ARCHITECTURE_CONFIG.items():
            model_tier = get_model_for_task(task_type)

            entry = ArchitectureEntry(
                task_type=task_type,
                level=config.get("level", "strong").value if hasattr(config.get("level", "strong"), "value") else config.get("level", "strong"),
                agents=config.get("agents", []),
                strategies=config.get("strategies", []),
                model_tier=model_tier.value if hasattr(model_tier, "value") else str(model_tier),
                max_tokens_budget=config.get("max_tokens_budget", 500),
                description=config.get("description", ""),
            )
            cls.LOOKUP_TABLE[task_type] = entry

    @classmethod
    def get_architecture(cls, task_type: str) -> Optional[ArchitectureEntry]:
        """
        Get architecture configuration for a task type.

        Args:
            task_type: Task type (Type-0 to Type-4)

        Returns:
            ArchitectureEntry or None if not found
        """
        if not cls.LOOKUP_TABLE:
            cls.initialize()

        return cls.LOOKUP_TABLE.get(task_type)

    @classmethod
    def get_agents(cls, task_type: str) -> List[str]:
        """
        Get agent list for a task type.

        Args:
            task_type: Task type

        Returns:
            List of agent names (ordered by priority)
        """
        entry = cls.get_architecture(task_type)
        if entry:
            return get_ordered_agents(entry.agents)
        return []

    @classmethod
    def get_strategies(cls, task_type: str) -> List[str]:
        """
        Get strategy list for a task type.

        Args:
            task_type: Task type

        Returns:
            List of strategy names (ordered by execution)
        """
        entry = cls.get_architecture(task_type)
        if entry:
            return get_ordered_strategies(entry.strategies)
        return []

    @classmethod
    def get_model_tier(cls, task_type: str) -> str:
        """
        Get model tier for a task type.

        Args:
            task_type: Task type

        Returns:
            Model tier name
        """
        entry = cls.get_architecture(task_type)
        if entry:
            return entry.model_tier
        return "local_small"

    @classmethod
    def get_token_budget(cls, task_type: str) -> int:
        """
        Get token budget for a task type.

        Args:
            task_type: Task type

        Returns:
            Maximum token budget
        """
        entry = cls.get_architecture(task_type)
        if entry:
            return entry.max_tokens_budget
        return 500

    @classmethod
    def is_weak_level(cls, task_type: str) -> bool:
        """
        Check if task type uses weak level (local model only).

        Args:
            task_type: Task type

        Returns:
            True if weak level
        """
        entry = cls.get_architecture(task_type)
        if entry:
            return entry.level == "weak"
        return False

    @classmethod
    def get_all_configurations(cls) -> Dict[str, ArchitectureEntry]:
        """
        Get all configurations.

        Returns:
            Dictionary of all architecture configurations
        """
        if not cls.LOOKUP_TABLE:
            cls.initialize()

        return cls.LOOKUP_TABLE.copy()

    @classmethod
    def get_summary(cls) -> str:
        """
        Get summary of all configurations.

        Returns:
            Human-readable summary string
        """
        if not cls.LOOKUP_TABLE:
            cls.initialize()

        lines = ["Configuration Lookup Summary:", "-" * 50]

        for task_type, entry in cls.LOOKUP_TABLE.items():
            lines.append(f"\n{task_type}:")
            lines.append(f"  Level: {entry.level}")
            lines.append(f"  Agents: {', '.join(entry.agents) or 'none'}")
            lines.append(f"  Strategies: {', '.join(entry.strategies) or 'none'}")
            lines.append(f"  Model: {entry.model_tier}")
            lines.append(f"  Token Budget: {entry.max_tokens_budget}")
            lines.append(f"  Description: {entry.description}")

        return "\n".join(lines)


# Convenience functions
def get_config(task_type: str) -> Optional[ArchitectureEntry]:
    """Get architecture configuration for a task type."""
    return ConfigLookup.get_architecture(task_type)


def get_agents_for_task(task_type: str) -> List[str]:
    """Get agent list for a task type."""
    return ConfigLookup.get_agents(task_type)


def get_strategies_for_task(task_type: str) -> List[str]:
    """Get strategy list for a task type."""
    return ConfigLookup.get_strategies(task_type)