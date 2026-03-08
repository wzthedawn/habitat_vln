"""Model configuration for VLN system."""

from typing import Dict, Any, List
from enum import Enum


class ModelTier(Enum):
    """Model tier classification."""
    LOCAL_SMALL = "local_small"      # Local small model (CLIP, etc.)
    LOCAL_MEDIUM = "local_medium"    # Local medium model (7B LLM)
    CLOUD_SMALL = "cloud_small"      # Cloud small model (GPT-3.5)
    CLOUD_LARGE = "cloud_large"      # Cloud large model (GPT-4)


MODEL_CONFIGS = {
    # Local small models - fast, cheap, limited capability
    ModelTier.LOCAL_SMALL: {
        "models": ["clip-vit-base", "vit-gpt2"],
        "max_tokens": 512,
        "latency_ms": 50,
        "cost_per_1k_tokens": 0.0,
        "capabilities": ["visual_encoding", "simple_reasoning"],
        "task_types": ["Type-0"],
    },

    # Local medium models - balanced
    ModelTier.LOCAL_MEDIUM: {
        "models": ["llama-7b", "mistral-7b"],
        "max_tokens": 2048,
        "latency_ms": 200,
        "cost_per_1k_tokens": 0.0,
        "capabilities": ["reasoning", "instruction_parsing", "planning"],
        "task_types": ["Type-0", "Type-1", "Type-2"],
    },

    # Cloud small models - capable, affordable
    ModelTier.CLOUD_SMALL: {
        "models": ["gpt-3.5-turbo", "claude-instant"],
        "max_tokens": 4096,
        "latency_ms": 500,
        "cost_per_1k_tokens": 0.002,
        "capabilities": ["reasoning", "instruction_parsing", "planning", "debate"],
        "task_types": ["Type-1", "Type-2", "Type-3"],
    },

    # Cloud large models - most capable, expensive
    ModelTier.CLOUD_LARGE: {
        "models": ["gpt-4", "claude-opus"],
        "max_tokens": 8192,
        "latency_ms": 2000,
        "cost_per_1k_tokens": 0.03,
        "capabilities": ["complex_reasoning", "debate", "reflection", "spatial_reasoning"],
        "task_types": ["Type-3", "Type-4"],
    },
}


def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration.

    Returns:
        Model configuration dictionary
    """
    return {
        "tiers": {tier.value: config for tier, config in MODEL_CONFIGS.items()},
        "selection": {
            "Type-0": ModelTier.LOCAL_SMALL,
            "Type-1": ModelTier.LOCAL_MEDIUM,
            "Type-2": ModelTier.LOCAL_MEDIUM,
            "Type-3": ModelTier.CLOUD_SMALL,
            "Type-4": ModelTier.CLOUD_LARGE,
        },
        "fallback_chain": [
            ModelTier.CLOUD_LARGE,
            ModelTier.CLOUD_SMALL,
            ModelTier.LOCAL_MEDIUM,
            ModelTier.LOCAL_SMALL,
        ],
    }


def get_model_for_task(task_type: str) -> ModelTier:
    """
    Get recommended model tier for a task type.

    Args:
        task_type: Task type (Type-0 to Type-4)

    Returns:
        Model tier
    """
    config = get_model_config()
    return config["selection"].get(task_type, ModelTier.LOCAL_MEDIUM)


def get_model_capabilities(tier: ModelTier) -> List[str]:
    """
    Get capabilities for a model tier.

    Args:
        tier: Model tier

    Returns:
        List of capabilities
    """
    return MODEL_CONFIGS.get(tier, {}).get("capabilities", [])