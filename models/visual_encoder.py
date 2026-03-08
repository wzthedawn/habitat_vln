"""Visual encoder for processing observations."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np


class VisualEncoder:
    """
    Visual Encoder for processing RGB-D observations.

    Handles:
    - RGB image encoding
    - Depth map processing
    - Panorama feature extraction
    - Object detection integration
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visual encoder.

        Args:
            config: Encoder configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("VisualEncoder")

        # Encoder settings
        self.encoder_type = self.config.get("encoder_type", "clip")
        self.image_size = self.config.get("image_size", 224)
        self.device = self.config.get("device", "cuda")

        # Model components (lazy loaded)
        self._encoder = None
        self._preprocessor = None
        self._object_detector = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the encoder."""
        if self._initialized:
            return

        try:
            if self.encoder_type == "clip":
                self._initialize_clip_encoder()
            elif self.encoder_type == "vit":
                self._initialize_vit_encoder()
            else:
                self.logger.info(f"Using default encoder for {self.encoder_type}")

            self._initialized = True

        except Exception as e:
            self.logger.warning(f"Encoder initialization failed: {e}")
            self._initialized = True

    def _initialize_clip_encoder(self) -> None:
        """Initialize CLIP-based encoder."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            if self.device == "cuda" and torch.cuda.is_available():
                self._encoder = self._encoder.to(self.device)

            self.logger.info("CLIP encoder initialized")

        except ImportError:
            self.logger.warning("Transformers not available for CLIP encoder")

    def _initialize_vit_encoder(self) -> None:
        """Initialize ViT-based encoder."""
        try:
            import torch
            from transformers import ViTModel, ViTImageProcessor

            self._encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self._preprocessor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

            if self.device == "cuda" and torch.cuda.is_available():
                self._encoder = self._encoder.to(self.device)

            self.logger.info("ViT encoder initialized")

        except ImportError:
            self.logger.warning("Transformers not available for ViT encoder")

    def encode(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode visual observation.

        Args:
            observation: Observation dictionary with RGB, depth, etc.

        Returns:
            Encoded features dictionary
        """
        self.initialize()

        features = {
            "rgb_embedding": None,
            "depth_features": None,
            "object_detections": [],
        }

        # Encode RGB
        if "rgb" in observation:
            features["rgb_embedding"] = self.encode_image(observation["rgb"])

        # Process depth
        if "depth" in observation:
            features["depth_features"] = self.process_depth(observation["depth"])

        # Detect objects
        if self._object_detector and "rgb" in observation:
            features["object_detections"] = self.detect_objects(observation["rgb"])

        return features

    def encode_image(self, image: Any) -> Optional[np.ndarray]:
        """
        Encode a single image.

        Args:
            image: Input image (PIL, numpy, or tensor)

        Returns:
            Image embedding
        """
        if self._encoder is None:
            return None

        try:
            import torch

            # Preprocess image
            inputs = self._preprocessor(images=image, return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get features
            with torch.no_grad():
                if hasattr(self._encoder, "get_image_features"):
                    features = self._encoder.get_image_features(**inputs)
                else:
                    output = self._encoder(**inputs)
                    features = output.last_hidden_state[:, 0, :]  # CLS token

            return features.cpu().numpy()

        except Exception as e:
            self.logger.warning(f"Image encoding failed: {e}")
            return None

    def process_depth(self, depth: Any) -> Optional[Dict[str, Any]]:
        """
        Process depth map.

        Args:
            depth: Depth map

        Returns:
            Depth features
        """
        try:
            depth = np.array(depth)

            features = {
                "mean_depth": float(np.mean(depth)),
                "min_depth": float(np.min(depth)),
                "max_depth": float(np.max(depth)),
                "obstacle_map": self._compute_obstacle_map(depth),
            }

            return features

        except Exception as e:
            self.logger.warning(f"Depth processing failed: {e}")
            return None

    def _compute_obstacle_map(self, depth: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        """Compute binary obstacle map from depth."""
        return (depth < threshold).astype(np.uint8)

    def detect_objects(self, image: Any) -> List[Dict[str, Any]]:
        """
        Detect objects in image.

        Args:
            image: Input image

        Returns:
            List of detected objects
        """
        # Placeholder for object detection
        # In practice, integrate with models like YOLO, DETR, etc.

        if self._object_detector is not None:
            # Use actual detector
            pass

        return []

    def encode_panorama(
        self, images: List[Any], headings: List[float]
    ) -> Dict[str, Any]:
        """
        Encode panorama from multiple views.

        Args:
            images: List of images at different headings
            headings: List of heading angles

        Returns:
            Panorama features
        """
        embeddings = []

        for image in images:
            emb = self.encode_image(image)
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)
            return {
                "embeddings": embeddings,
                "headings": headings,
                "num_views": len(images),
            }

        return {"embeddings": None, "headings": headings, "num_views": 0}

    def classify_room(
        self, observation: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Classify room type from observation.

        Args:
            observation: Observation dictionary

        Returns:
            Tuple of (room_type, confidence)
        """
        # Room type labels
        room_types = [
            "bedroom", "bathroom", "kitchen", "living_room",
            "dining_room", "office", "hallway", "garage",
        ]

        if self._encoder is not None and "rgb" in observation:
            try:
                # Use CLIP for zero-shot classification
                import torch

                image = observation["rgb"]
                inputs = self._preprocessor(
                    text=[f"a photo of a {room}" for room in room_types],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                )

                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._encoder(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                probs = probs.cpu().numpy()[0]
                best_idx = np.argmax(probs)

                return room_types[best_idx], float(probs[best_idx])

            except Exception as e:
                self.logger.warning(f"Room classification failed: {e}")

        return "unknown", 0.0

    def get_model(self) -> Any:
        """Get the underlying encoder model."""
        return self._encoder

    def set_object_detector(self, detector: Any) -> None:
        """Set object detector model."""
        self._object_detector = detector