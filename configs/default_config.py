"""Default configuration for VLN system."""

from typing import Dict, Any
import yaml
from pathlib import Path


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the VLN system.

    Returns:
        Configuration dictionary
    """
    return {
        # System settings
        "system": {
            "name": "Multi-Agent VLN Navigator",
            "version": "1.0.0",
            "log_level": "INFO",
        },

        # Navigation settings
        "navigation": {
            "max_steps": 500,
            "stop_distance": 0.2,
            "turn_angle": 15.0,  # degrees
            "collision_threshold": 0.1,
        },

        # Classifier settings
        "classifier": {
            "rule_threshold": 0.9,
            "use_llm_fallback": True,
            "cache_results": True,
        },

        # Supernet settings
        "supernet": {
            "architecture_search": False,
            "adaptive_selection": True,
        },

        # Agent settings
        "agents": {
            "default_model": "local",
            "fallback_model": "llm",
            "parallel_execution": False,
        },

        # Strategy settings
        "strategies": {
            "default_strategy": "ReAct",
            "max_iterations": 10,
            "timeout": 30.0,
        },

        # Optimization settings
        "optimization": {
            "context_compression": True,
            "compression_level": "standard",
            "cache_prompts": True,
            "max_history_length": 50,
        },

        # Fallback settings
        "fallback": {
            "enabled": True,
            "max_retries": 3,
            "cascading_levels": 5,
        },

        # Model settings
        "model": {
            "local_model": {
                "type": "clip",
                "device": "cuda",
            },
            "llm_model": {
                "type": "gpt-4",
                "max_tokens": 2000,
                "temperature": 0.7,
            },
        },
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        return get_default_config()

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Merge with defaults
    default = get_default_config()
    return deep_merge(default, config)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result