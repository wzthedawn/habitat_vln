"""Model selector for choosing appropriate models based on task requirements."""

from typing import Dict, Any, Optional
import logging

from configs.model_config import (
    get_model_config,
    get_model_for_task,
    get_model_capabilities,
    ModelTier,
)


class ModelSelector:
    """
    Model Selector for choosing appropriate models.

    Selects models based on:
    1. Task type requirements
    2. Available capabilities
    3. Cost/latency constraints
    4. Fallback chain
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model selector.

        Args:
            config: Selector configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("ModelSelector")

        # Load model configurations
        self._model_config = get_model_config()

        # Capability requirements by task type
        self._capability_requirements = {
            "Type-0": ["visual_encoding"],
            "Type-1": ["visual_encoding", "simple_reasoning"],
            "Type-2": ["reasoning", "instruction_parsing"],
            "Type-3": ["spatial_reasoning", "planning"],
            "Type-4": ["complex_reasoning", "debate", "reflection"],
        }

        # Cache for selected models
        self._model_cache: Dict[str, Any] = {}

    def select_model(self, task_type: str) -> Any:
        """
        Select the best model for a task type.

        Args:
            task_type: Task type (Type-0 to Type-4)

        Returns:
            Model instance
        """
        # Check cache first
        if task_type in self._model_cache:
            return self._model_cache[task_type]

        # Get recommended tier
        tier = get_model_for_task(task_type)

        # Create model instance
        model = self._create_model(tier)

        # Cache it
        self._model_cache[task_type] = model

        return model

    def select_model_by_capability(self, required_capabilities: list) -> ModelTier:
        """
        Select model tier based on required capabilities.

        Args:
            required_capabilities: List of required capabilities

        Returns:
            ModelTier that satisfies all requirements
        """
        # Check tiers from smallest to largest
        tier_order = [
            ModelTier.LOCAL_SMALL,
            ModelTier.LOCAL_MEDIUM,
            ModelTier.CLOUD_SMALL,
            ModelTier.CLOUD_LARGE,
        ]

        for tier in tier_order:
            capabilities = get_model_capabilities(tier)
            if all(cap in capabilities for cap in required_capabilities):
                return tier

        # Default to largest if none found
        return ModelTier.CLOUD_LARGE

    def _create_model(self, tier: ModelTier) -> Any:
        """Create a model instance for the given tier."""
        config = self._model_config["tiers"].get(tier.value, {})
        models = config.get("models", [])

        if not models:
            self.logger.warning(f"No models configured for tier {tier}")
            return self._create_mock_model()

        model_name = models[0]  # Use first available model

        if tier in [ModelTier.LOCAL_SMALL, ModelTier.LOCAL_MEDIUM]:
            try:
                from models.local_model import LocalModel
                return LocalModel({"model_name": model_name, "tier": tier.value})
            except ImportError:
                self.logger.warning("LocalModel not available, using mock")
                return self._create_mock_model()
        else:
            try:
                from models.llm_model import LLMModel
                # Get model-specific config
                model_configs = self._model_config.get("model_configs", {})

                # Try to find matching config
                model_config = None
                for config_key, cfg in model_configs.items():
                    if cfg.get("model") == model_name or config_key in model_name.lower():
                        model_config = cfg.copy()
                        break

                if model_config:
                    model_config["model_name"] = model_name
                    model_config["tier"] = tier.value
                else:
                    model_config = {"model_name": model_name, "tier": tier.value}

                return LLMModel(model_config)
            except ImportError:
                self.logger.warning("LLMModel not available, using mock")
                return self._create_mock_model()

    def _create_mock_model(self):
        """Create a mock model for testing."""
        class MockModel:
            def __init__(self):
                self.name = "mock_model"

            def predict(self, context):
                return "forward"

            def generate(self, prompt):
                return "Action: forward"

        return MockModel()

    def get_fallback_chain(self, task_type: str) -> list:
        """
        Get fallback model chain for a task type.

        Args:
            task_type: Task type

        Returns:
            List of model tiers to try in order
        """
        return self._model_config.get("fallback_chain", [])

    def get_model_info(self, tier: ModelTier) -> Dict[str, Any]:
        """
        Get information about a model tier.

        Args:
            tier: Model tier

        Returns:
            Model information dictionary
        """
        return self._model_config["tiers"].get(tier.value, {})

    def estimate_cost(self, task_type: str, estimated_tokens: int) -> float:
        """
        Estimate cost for a task.

        Args:
            task_type: Task type
            estimated_tokens: Estimated token usage

        Returns:
            Estimated cost in dollars
        """
        tier = get_model_for_task(task_type)
        config = self._model_config["tiers"].get(tier.value, {})
        cost_per_1k = config.get("cost_per_1k_tokens", 0)

        return (estimated_tokens / 1000) * cost_per_1k

    def estimate_latency(self, task_type: str) -> float:
        """
        Estimate latency for a task.

        Args:
            task_type: Task type

        Returns:
            Estimated latency in milliseconds
        """
        tier = get_model_for_task(task_type)
        config = self._model_config["tiers"].get(tier.value, {})
        return config.get("latency_ms", 500)

    def clear_cache(self) -> None:
        """Clear model cache."""
        self._model_cache.clear()