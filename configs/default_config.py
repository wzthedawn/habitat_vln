"""Default configuration for VLN Pipeline system."""

from typing import Dict, Any
import yaml
from pathlib import Path


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the VLN Pipeline system.

    Returns:
        Configuration dictionary
    """
    return {
        # System settings
        "system": {
            "name": "Pipeline VLN Navigator",
            "version": "2.0.0",
            "log_level": "INFO",
        },

        # Navigation settings
        "navigation": {
            "max_steps": 150,
            "stop_distance": 0.2,
            "turn_angle": 15.0,
            "success_distance": 3.0,
        },

        # Pipeline Agent configuration
        "pipeline": {
            "max_steps": 150,
            "report_interval": 10,
            "debate_mode": "standard",  # light | standard | deep
            # Model key allocation per SubAgent
            "model_configs": {
                "decomposition": "qwen3.5-9b-fast",
                "observation": "qwen3-vl-8b",
                "analysis": "qwen3.5-9b-fast",
                "analysis_strong": "qwen3.6-35b-strong",
                "planning": "qwen3.6-35b-strong",
                "review": "qwen3.5-9b-fast",
                "emergency": "qwen3.5-9b-fast",
            },
        },

        # Difficulty grading thresholds
        "difficulty": {
            "static_hard_threshold": 6,
            "static_medium_threshold": 3,
            "dynamic_hard_threshold": 5,
            "dynamic_medium_threshold": 2,
        },

        # Remote LLM settings
        "remote_llm": {
            "server_url": "http://localhost:8000",
            "timeout": 120.0,
            "use_siliconflow": False,
        },

        # Datasets
        "datasets": {
            "mp3d_path": "/data/WZ/Dataset/mp3d_dataset/mp3d",
            "r2r_path": "/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json",
        },

        # Output
        "output": {
            "dir": "results",
            "enable_video": True,
            "enable_trajectory": True,
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
