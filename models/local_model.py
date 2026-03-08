"""Local model for lightweight inference."""

from typing import Dict, Any, Optional
import logging

from core.context import NavContext


class LocalModel:
    """
    Local Model for fast, lightweight inference.

    Uses local models (CLIP, ViT, small LLMs) for:
    - Visual encoding
    - Simple action prediction
    - Quick decision making
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize local model.

        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("LocalModel")

        # Model settings
        self.model_name = self.config.get("model_name", "clip-vit-base")
        self.device = self.config.get("device", "cuda")
        self.tier = self.config.get("tier", "local_small")

        # Model instance (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the model."""
        if self._initialized:
            return

        try:
            # Try to load actual model
            if "clip" in self.model_name.lower():
                self._initialize_clip()
            elif "vit" in self.model_name.lower():
                self._initialize_vit()
            else:
                self.logger.info(f"Using mock model for {self.model_name}")
                self._initialized = True

        except Exception as e:
            self.logger.warning(f"Model initialization failed: {e}, using mock")
            self._initialized = True

    def _initialize_clip(self) -> None:
        """Initialize CLIP model."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to(self.device)

            self._initialized = True
            self.logger.info("CLIP model initialized")

        except ImportError:
            self.logger.warning("Transformers not available for CLIP")

    def _initialize_vit(self) -> None:
        """Initialize ViT model."""
        try:
            import torch
            from transformers import ViTModel, ViTImageProcessor

            self._model = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self._processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to(self.device)

            self._initialized = True
            self.logger.info("ViT model initialized")

        except ImportError:
            self.logger.warning("Transformers not available for ViT")

    def predict(self, context: NavContext) -> str:
        """
        Predict action from context.

        Args:
            context: Navigation context

        Returns:
            Action string (forward, turn_left, turn_right, stop)
        """
        self.initialize()

        # If we have a real model, use it
        if self._model is not None and "clip" in self.model_name.lower():
            return self._predict_with_clip(context)

        # Otherwise, use simple heuristic prediction
        return self._heuristic_predict(context)

    def _predict_with_clip(self, context: NavContext) -> str:
        """Predict using CLIP model."""
        try:
            # Define action labels
            action_labels = [
                "move forward",
                "turn left",
                "turn right",
                "stop",
            ]

            # Get visual features if available
            if context.visual_features.rgb_embedding is not None:
                # Use pre-computed embedding
                pass

            # Simple text-based prediction for now
            instruction_lower = context.instruction.lower()

            if "left" in instruction_lower:
                return "turn_left"
            elif "right" in instruction_lower:
                return "turn_right"
            elif "stop" in instruction_lower:
                return "stop"
            else:
                return "forward"

        except Exception as e:
            self.logger.warning(f"CLIP prediction failed: {e}")
            return "forward"

    def _heuristic_predict(self, context: NavContext) -> str:
        """Heuristic-based action prediction."""
        instruction_lower = context.instruction.lower()

        # Check for explicit direction commands
        if "turn left" in instruction_lower:
            return "turn_left"
        elif "turn right" in instruction_lower:
            return "turn_right"
        elif "stop" in instruction_lower:
            return "stop"

        # Check for movement commands
        if any(word in instruction_lower for word in ["go", "walk", "move", "forward"]):
            return "forward"

        # Check for search/find commands
        if any(word in instruction_lower for word in ["find", "search", "look"]):
            # For search tasks, suggest exploration
            return "forward"

        # Check action history for patterns
        if context.action_history:
            recent = [a.action_type.name for a in context.action_history[-3:]]
            if "TURN_LEFT" in recent and "TURN_RIGHT" in recent:
                # Oscillating, try forward
                return "forward"

        # Default to forward
        return "forward"

    def encode_text(self, text: str) -> Any:
        """
        Encode text using the model.

        Args:
            text: Input text

        Returns:
            Text embedding
        """
        self.initialize()

        if self._model is not None and "clip" in self.model_name.lower():
            try:
                import torch
                inputs = self._processor(text=[text], return_tensors="pt", padding=True)
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_features = self._model.get_text_features(**inputs)

                return text_features.cpu().numpy()
            except Exception as e:
                self.logger.warning(f"Text encoding failed: {e}")

        return None

    def encode_image(self, image: Any) -> Any:
        """
        Encode image using the model.

        Args:
            image: Input image

        Returns:
            Image embedding
        """
        self.initialize()

        if self._model is not None and "clip" in self.model_name.lower():
            try:
                import torch
                inputs = self._processor(images=image, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    image_features = self._model.get_image_features(**inputs)

                return image_features.cpu().numpy()
            except Exception as e:
                self.logger.warning(f"Image encoding failed: {e}")

        return None

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text (for local LLM models).

        Args:
            prompt: Input prompt
            max_length: Maximum output length

        Returns:
            Generated text
        """
        self.initialize()

        # Local models typically don't generate complex text
        # Return simple action prediction
        if "action" in prompt.lower():
            if "left" in prompt.lower():
                return "turn_left"
            elif "right" in prompt.lower():
                return "turn_right"
            elif "stop" in prompt.lower():
                return "stop"

        return "forward"