"""LLM-based task classifier."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging

from core.context import NavContext, TaskType


@dataclass
class LLMClassificationResult:
    """Result of LLM classification."""
    task_type: TaskType
    confidence: float
    reasoning: str
    subtasks: list
    complexity_factors: Dict[str, Any]


class LLMClassifier:
    """
    LLM-based task classifier.

    Uses an LLM for fine-grained task classification when rule-based
    classification has low confidence.
    """

    CLASSIFICATION_PROMPT = """Analyze the following navigation instruction and classify its complexity level.

Instruction: "{instruction}"

Current context:
- Position: {position}
- Room type: {room_type}
- Steps taken: {steps}

Classification criteria:
- Type-0: Simple single-step commands (turn, move forward, stop)
- Type-1: Path following in a single area (walk down hallway, go straight)
- Type-2: Object/landmark search (find the chair, locate the door)
- Type-3: Multi-room spatial reasoning (go to kitchen via living room)
- Type-4: Complex decisions with conditions (if you see X, turn left, otherwise go straight)

Respond in JSON format:
{{
    "task_type": "Type-0" | "Type-1" | "Type-2" | "Type-3" | "Type-4",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "subtasks": ["list of subtasks if applicable"],
    "complexity_factors": {{
        "multi_room": true/false,
        "conditional_logic": true/false,
        "object_recognition_needed": true/false,
        "spatial_reasoning_needed": true/false,
        "estimated_steps": number
    }}
}}"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LLM classifier.

        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config or {}
        self.logger = logging.getLogger("LLMClassifier")

        # Model configuration
        self.model_type = self.config.get("model_type", "gpt-3.5-turbo")
        self.max_tokens = self.config.get("max_tokens", 500)
        self.temperature = self.config.get("temperature", 0.3)

        # Lazy-loaded model
        self._model = None

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            # Import model based on type
            if "gpt" in self.model_type.lower() or "qwen" in self.model_type.lower():
                try:
                    from models.llm_model import LLMModel
                    import yaml

                    # Load model config
                    with open("configs/model_config.yaml", 'r') as f:
                        config = yaml.safe_load(f)

                    model_configs = config.get('model_configs', {})
                    qwen_cfg = model_configs.get('qwen3.5-plus', {})

                    self._model = LLMModel({
                        "model_name": qwen_cfg.get('model', self.model_type),
                        "api_key": qwen_cfg.get('api_key', ''),
                        "api_base": qwen_cfg.get('base_url', ''),
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    })
                except Exception as e:
                    self.logger.warning(f"LLMModel not available: {e}, using mock")
                    self._model = self._create_mock_model()
            else:
                from models.local_model import LocalModel
                self._model = LocalModel({"model_name": self.model_type})
        return self._model

    def _create_mock_model(self):
        """Create a mock model for testing."""
        class MockModel:
            def generate(self, prompt: str) -> str:
                # Simple heuristic for mock responses
                if any(w in prompt.lower() for w in ["turn", "move", "stop"]):
                    return '{"task_type": "Type-0", "confidence": 0.8, "reasoning": "Simple command", "subtasks": [], "complexity_factors": {"multi_room": false, "conditional_logic": false, "object_recognition_needed": false, "spatial_reasoning_needed": false, "estimated_steps": 1}}'
                elif any(w in prompt.lower() for w in ["find", "search", "look for"]):
                    return '{"task_type": "Type-2", "confidence": 0.85, "reasoning": "Object search task", "subtasks": ["scan environment", "identify object", "navigate to object"], "complexity_factors": {"multi_room": false, "conditional_logic": false, "object_recognition_needed": true, "spatial_reasoning_needed": false, "estimated_steps": 10}}'
                elif any(w in prompt.lower() for w in ["room", "bedroom", "kitchen"]):
                    return '{"task_type": "Type-3", "confidence": 0.8, "reasoning": "Multi-room navigation", "subtasks": ["exit current room", "navigate to target room", "locate goal"], "complexity_factors": {"multi_room": true, "conditional_logic": false, "object_recognition_needed": false, "spatial_reasoning_needed": true, "estimated_steps": 20}}'
                elif any(w in prompt.lower() for w in ["if", "unless", "either"]):
                    return '{"task_type": "Type-4", "confidence": 0.9, "reasoning": "Conditional navigation", "subtasks": ["evaluate condition", "choose path", "execute navigation"], "complexity_factors": {"multi_room": true, "conditional_logic": true, "object_recognition_needed": true, "spatial_reasoning_needed": true, "estimated_steps": 30}}'
                else:
                    return '{"task_type": "Type-1", "confidence": 0.7, "reasoning": "Standard navigation", "subtasks": ["follow path"], "complexity_factors": {"multi_room": false, "conditional_logic": false, "object_recognition_needed": false, "spatial_reasoning_needed": false, "estimated_steps": 5}}'
        return MockModel()

    def classify(self, context: NavContext) -> LLMClassificationResult:
        """
        Classify task using LLM.

        Args:
            context: Navigation context

        Returns:
            LLMClassificationResult with detailed analysis
        """
        model = self._get_model()

        # Build prompt
        prompt = self.CLASSIFICATION_PROMPT.format(
            instruction=context.instruction,
            position=context.position,
            room_type=context.room_type,
            steps=context.step_count,
        )

        try:
            # Get LLM response
            response = model.generate(prompt)

            # Parse response
            result = self._parse_response(response)

            return result

        except Exception as e:
            self.logger.error(f"LLM classification error: {e}")
            return self._default_result()

    def _parse_response(self, response: str) -> LLMClassificationResult:
        """Parse LLM response into structured result."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return LLMClassificationResult(
                    task_type=TaskType(data.get("task_type", "Type-1")),
                    confidence=float(data.get("confidence", 0.7)),
                    reasoning=data.get("reasoning", ""),
                    subtasks=data.get("subtasks", []),
                    complexity_factors=data.get("complexity_factors", {}),
                )
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")

        return self._default_result()

    def _default_result(self) -> LLMClassificationResult:
        """Return default classification result."""
        return LLMClassificationResult(
            task_type=TaskType.TYPE_1,
            confidence=0.5,
            reasoning="Default classification due to parsing error",
            subtasks=[],
            complexity_factors={},
        )

    def classify_with_context(
        self, context: NavContext, additional_info: Dict[str, Any] = None
    ) -> LLMClassificationResult:
        """
        Classify with additional context information.

        Args:
            context: Navigation context
            additional_info: Extra information for classification

        Returns:
            Detailed classification result
        """
        # Enhance prompt with additional info
        if additional_info:
            context.metadata.update(additional_info)

        return self.classify(context)