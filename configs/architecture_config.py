"""Architecture configuration for task-type to Agent/Strategy mapping."""

from typing import Dict, Any, List
from enum import Enum


class ArchitectureLevel(Enum):
    """Architecture complexity level."""
    WEAK = "weak"        # Local small model only
    STRONG = "strong"    # Multi-agent collaboration


# Architecture configuration for each task type
ARCHITECTURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "Type-0": {
        "level": ArchitectureLevel.WEAK,
        "agents": [],
        "strategies": [],
        "model_tier": "local_small",
        "description": "Simple navigation - single step instruction",
        "max_tokens_budget": 100,
    },

    "Type-1": {
        "level": ArchitectureLevel.STRONG,
        "agents": ["perception", "decision"],
        "strategies": ["ReAct"],
        "model_tier": "local_medium",
        "description": "Path following - corridor navigation",
        "max_tokens_budget": 300,
    },

    "Type-2": {
        "level": ArchitectureLevel.STRONG,
        "agents": ["perception", "trajectory", "decision"],
        "strategies": ["ReAct", "CoT"],
        "model_tier": "local_medium",
        "description": "Target search - object finding",
        "max_tokens_budget": 500,
    },

    "Type-3": {
        "level": ArchitectureLevel.STRONG,
        "agents": ["instruction", "perception", "trajectory", "decision"],
        "strategies": ["CoT", "Reflection"],
        "model_tier": "cloud_small",
        "description": "Spatial reasoning - cross-room navigation",
        "max_tokens_budget": 1000,
    },

    "Type-4": {
        "level": ArchitectureLevel.STRONG,
        "agents": ["instruction", "perception", "trajectory", "decision"],
        "strategies": ["CoT", "Debate", "Reflection"],
        "model_tier": "cloud_large",
        "description": "Complex decision - ambiguous scenes",
        "max_tokens_budget": 2000,
    },
}


# Strategy execution order configuration
STRATEGY_ORDER = {
    "ReAct": 1,
    "CoT": 2,
    "Debate": 3,
    "Reflection": 4,
}


# Agent role definitions
AGENT_ROLES = {
    "instruction": {
        "name": "InstructionAgent",
        "description": "Decomposes and interprets navigation instructions",
        "inputs": ["instruction"],
        "outputs": ["subtasks", "landmarks", "goals"],
        "priority": 1,
    },

    "perception": {
        "name": "PerceptionAgent",
        "description": "Analyzes visual observations and environment state",
        "inputs": ["visual_features", "observation"],
        "outputs": ["room_type", "objects", "landmarks", "obstacles"],
        "priority": 2,
    },

    "trajectory": {
        "name": "TrajectoryAgent",
        "description": "Plans and evaluates navigation trajectories",
        "inputs": ["position", "goal", "history"],
        "outputs": ["waypoints", "path_confidence", "progress"],
        "priority": 3,
    },

    "decision": {
        "name": "DecisionAgent",
        "description": "Makes final navigation decisions",
        "inputs": ["all_agent_outputs", "context"],
        "outputs": ["action", "confidence", "reasoning"],
        "priority": 4,
    },
}


def get_architecture_config() -> Dict[str, Any]:
    """
    Get complete architecture configuration.

    Returns:
        Architecture configuration dictionary
    """
    return {
        "architectures": ARCHITECTURE_CONFIG,
        "strategy_order": STRATEGY_ORDER,
        "agent_roles": AGENT_ROLES,
    }


def get_architecture_for_task(task_type: str) -> Dict[str, Any]:
    """
    Get architecture configuration for a specific task type.

    Args:
        task_type: Task type (Type-0 to Type-4)

    Returns:
        Architecture configuration
    """
    return ARCHITECTURE_CONFIG.get(task_type, ARCHITECTURE_CONFIG["Type-0"])


def get_ordered_strategies(strategies: List[str]) -> List[str]:
    """
    Sort strategies by execution order.

    Args:
        strategies: List of strategy names

    Returns:
        Sorted list of strategies
    """
    return sorted(strategies, key=lambda s: STRATEGY_ORDER.get(s, 999))


def get_ordered_agents(agents: List[str]) -> List[str]:
    """
    Sort agents by priority.

    Args:
        agents: List of agent names

    Returns:
        Sorted list of agents
    """
    return sorted(
        agents,
        key=lambda a: AGENT_ROLES.get(a, {}).get("priority", 999)
    )