"""Model Manager for local and remote model management.

Manages Qwen3.5 series models with INT8 quantization:
- Qwen3.5-2B (perception): For PerceptionAgent visual descriptions
- Qwen3.5-2B (trajectory): For TrajectoryAgent path summarization
- Qwen3.5-4B: For DecisionAgent action selection
- Qwen3.5-9B: For EvaluationAgent decision assessment
- YOLOv5s: Object detection (compatible with numpy 2.x)

Supports dual-environment IPC architecture:
- Local mode: Load models directly (requires Python 3.10+)
- Remote mode: Use HTTP API to communicate with LLM server

Total VRAM (INT8): ~17GB
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import gc
import threading
import numpy as np

# Monkey patch torch.from_numpy for numpy 2.x / torch compatibility
# This fixes the "expected np.ndarray (got numpy.ndarray)" error
try:
    import torch
    _original_from_numpy = torch.from_numpy

    def _patched_from_numpy(ndarray):
        try:
            return _original_from_numpy(ndarray)
        except TypeError:
            # Fallback to torch.as_tensor which doesn't have the type check issue
            return torch.as_tensor(ndarray)

    torch.from_numpy = _patched_from_numpy
except ImportError:
    pass  # torch not available


class ModelManager:
    """
    Manager for loading and managing local models.

    Manages model lifecycle:
    - YOLOv5s: Object detection (~0.5GB) - compatible with numpy 2.x
    - Qwen3.5-2B (perception): Visual descriptions (~2.1GB INT8)
    - Qwen3.5-2B (trajectory): Path summarization (~2.1GB INT8)
    - Qwen3.5-4B: Navigation decisions (~4GB INT8)
    - Qwen3.5-9B: Decision evaluation (~9GB INT8)

    Total VRAM: ~17GB with INT8 quantization
    """

    _instance = None
    _lock = threading.Lock()

    # Model configurations with INT8 quantization
    MODEL_CONFIGS = {
        "yolov5s": {
            "type": "yolo",
            "model_name": "yolov5su.pt",  # YOLOv5 small ultra - compatible with numpy 2.x
            "vram_gb": 0.5,
            "load_time": 1.0,
        },
        "qwen-2b-perception": {
            "type": "llm",
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B",
            "vram_gb": 2.1,
            "load_time": 10.0,
            "max_new_tokens": 256,
            "temperature": 0.3,
        },
        "qwen-2b-trajectory": {
            "type": "llm",
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B",
            "vram_gb": 2.1,
            "load_time": 10.0,
            "max_new_tokens": 200,
            "temperature": 0.2,
        },
        "qwen-4b": {
            "type": "llm",
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "vram_gb": 4.0,
            "load_time": 15.0,
            "max_new_tokens": 150,
            "temperature": 0.1,
        },
        "qwen-9b": {
            "type": "llm",
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-9B",
            "vram_gb": 9.0,
            "load_time": 20.0,
            "max_new_tokens": 400,
            "temperature": 0.3,
        },
    }

    def __new__(cls, config: Dict[str, Any] = None):
        """Singleton pattern for model manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ModelManager."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or {}
        self.logger = logging.getLogger("ModelManager")

        # Device configuration
        self.device = self.config.get("device", "cuda")
        self.use_int8 = self.config.get("use_int8", True)

        # Remote LLM configuration (for dual-environment IPC)
        self.use_remote = self.config.get("use_remote", False)
        self.remote_server_url = self.config.get("remote_server_url", "http://localhost:8000")
        self._remote_client = None
        self._remote_healthy = False

        # Model storage
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

        # Loading state
        self._initialized = True
        self._models_loaded = False
        self._llm_loaded = False

        # Initialize remote client if needed
        if self.use_remote:
            self._init_remote_client()

        mode = "remote" if self.use_remote else "local"
        self.logger.info(f"ModelManager initialized (device={self.device}, int8={self.use_int8}, mode={mode})")

    def _init_remote_client(self) -> bool:
        """Initialize remote LLM client."""
        try:
            from models.remote_client import RemoteLLMClient
            self._remote_client = RemoteLLMClient(
                server_url=self.remote_server_url,
                timeout=self.config.get("remote_timeout", 60.0),
            )

            # Check server health
            self._remote_healthy = self._remote_client.health_check()
            if self._remote_healthy:
                self.logger.info(f"Remote LLM server connected: {self.remote_server_url}")
            else:
                self.logger.warning(f"Remote LLM server not responding: {self.remote_server_url}")

            return self._remote_healthy

        except ImportError as e:
            self.logger.error(f"Failed to import RemoteLLMClient: {e}")
            self.logger.error("Install with: pip install aiohttp or pip install requests")
            self.use_remote = False
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize remote client: {e}")
            self.use_remote = False
            return False

    def load_all_models(self, load_llms: bool = False) -> bool:
        """
        Load all models at once.

        Args:
            load_llms: Whether to load LLM models (default: False for lazy loading)

        Returns:
            True if all models loaded successfully
        """
        if self._models_loaded:
            self.logger.info("[load_all_models] Models already loaded, skipping")
            return True

        self.logger.info("=" * 60)
        self.logger.info("[load_all_models] Loading all models...")
        self.logger.info(f"[load_all_models] use_remote={self.use_remote}, device={self.device}")

        try:
            # Check CUDA availability
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("[load_all_models] CUDA not available, falling back to CPU")
                self.device = "cpu"

            # Load YOLO
            self.logger.info("[load_all_models] Loading YOLO...")
            if not self._load_yolo():
                self.logger.warning("[load_all_models] YOLO loading failed, using fallback")
            else:
                self.logger.info("[load_all_models] YOLO loaded successfully")

            # Load LLMs if requested
            if load_llms:
                self.logger.info("[load_all_models] Loading LLMs...")
                self.load_all_llms()

            self._models_loaded = True
            self.logger.info(f"[load_all_models] Complete. Loaded models: {list(self._models.keys())}")
            self.logger.info("=" * 60)
            return True

        except Exception as e:
            self.logger.error(f"[load_all_models] Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_yolo(self) -> bool:
        """Load YOLOv5 model."""
        model_key = "yolov5s"
        self.logger.info(f"[_load_yolo] Attempting to load {model_key}...")

        if model_key in self._models:
            self.logger.info(f"[_load_yolo] {model_key} already loaded")
            return True

        try:
            from ultralytics import YOLO

            model_name = self.MODEL_CONFIGS[model_key]["model_name"]
            self.logger.info(f"[_load_yolo] Loading {model_name} (YOLOv5 for numpy 2.x compatibility)...")

            self._models[model_key] = YOLO(model_name)
            self.logger.info(f"[_load_yolo] YOLOv5 loaded successfully, model type: {type(self._models[model_key])}")
            return True

        except ImportError as e:
            self.logger.warning(f"[_load_yolo] ultralytics not installed, YOLO unavailable: {e}")
            return False
        except Exception as e:
            self.logger.error(f"[_load_yolo] Failed to load YOLO: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_llm(self, model_key: str) -> bool:
        """
        Load a Qwen LLM model with INT8 quantization.

        Args:
            model_key: Model identifier (qwen-2b-perception, qwen-2b-trajectory, qwen-4b, qwen-9b)

        Returns:
            True if loaded successfully
        """
        if model_key in self._models:
            return True

        if model_key not in self.MODEL_CONFIGS:
            self.logger.error(f"Unknown model key: {model_key}")
            return False

        config = self.MODEL_CONFIGS[model_key]
        if config["type"] != "llm":
            self.logger.error(f"Model {model_key} is not an LLM")
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            model_path = config["model_name"]
            self.logger.info(f"Loading {model_key} from {model_path}...")

            # Check if path exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model path does not exist: {model_path}")
                return False

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )

            # Configure INT8 quantization
            if self.use_int8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                # For INT8, we don't specify device_map as it's handled by bitsandbytes
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            else:
                # FP16 without quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
                if self.device == "cuda":
                    model = model.to(self.device)

            model.eval()

            self._models[model_key] = model
            self._tokenizers[model_key] = tokenizer

            vram = config.get("vram_gb", 0)
            self.logger.info(f"{model_key} loaded successfully (VRAM: ~{vram}GB)")
            return True

        except ImportError as e:
            self.logger.error(f"Missing dependencies for LLM loading: {e}")
            self.logger.error("Install with: pip install transformers bitsandbytes accelerate")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_all_llms(self) -> bool:
        """
        Load all Qwen LLM models.

        Returns:
            True if all loaded successfully
        """
        if self._llm_loaded:
            return True

        self.logger.info("Loading all LLM models...")

        llm_keys = ["qwen-2b-perception", "qwen-2b-trajectory", "qwen-4b", "qwen-9b"]
        success = True

        for key in llm_keys:
            if not self.load_llm(key):
                self.logger.warning(f"Failed to load {key}")
                success = False

        self._llm_loaded = success
        return success

    def generate(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """
        Generate text using a Qwen LLM model.

        Supports both local and remote generation based on configuration.
        When use_remote is True, uses HTTP API to communicate with
        Python 3.10 LLM server.

        Args:
            model_key: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation kwargs

        Returns:
            Generated text
        """
        # Use remote generation if configured
        if self.use_remote and self._remote_client:
            return self._generate_remote(
                model_key=model_key,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )

        # Local generation
        return self._generate_local(
            model_key=model_key,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )

    def _generate_remote(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate text using remote LLM server."""
        if not self._remote_client:
            self.logger.error("Remote client not initialized")
            return ""

        # Map model keys to available remote models
        # If the requested model is not available, fallback to qwen-4b
        remote_model_key = model_key
        available_models = self._get_available_remote_models()

        if model_key not in available_models:
            # Fallback to qwen-4b for any unavailable model
            if "qwen-4b" in available_models:
                self.logger.debug(f"Model {model_key} not available, using qwen-4b")
                remote_model_key = "qwen-4b"
            elif available_models:
                remote_model_key = available_models[0]
                self.logger.debug(f"Model {model_key} not available, using {remote_model_key}")

        config = self.MODEL_CONFIGS.get(model_key, {})
        if max_new_tokens is None:
            max_new_tokens = config.get("max_new_tokens", 256)
        if temperature is None:
            temperature = config.get("temperature", 0.3)

        try:
            result = self._remote_client.generate(
                model=remote_model_key,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                conversation_id=kwargs.get("conversation_id"),
                keep_context=kwargs.get("keep_context", False),
            )
            return result
        except Exception as e:
            self.logger.error(f"Remote generation failed: {e}")
            return ""

    def _get_available_remote_models(self) -> List[str]:
        """Get list of available models from remote server."""
        if not self._remote_client:
            return []

        try:
            # Use health check to get available models
            health = self._remote_client.health_check_sync()
            if isinstance(health, dict) and "models_loaded" in health:
                return health.get("models_loaded", [])
        except:
            pass

        return []

    def _generate_local(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> str:
        """Generate text using locally loaded models."""
        model = self._models.get(model_key)
        tokenizer = self._tokenizers.get(model_key)

        if model is None or tokenizer is None:
            self.logger.warning(f"Model {model_key} not loaded, attempting to load...")
            if not self.load_llm(model_key):
                return ""
            model = self._models.get(model_key)
            tokenizer = self._tokenizers.get(model_key)

        if model is None or tokenizer is None:
            self.logger.error(f"Failed to get model {model_key}")
            return ""

        # Get default config values
        config = self.MODEL_CONFIGS.get(model_key, {})
        if max_new_tokens is None:
            max_new_tokens = config.get("max_new_tokens", 256)
        if temperature is None:
            temperature = config.get("temperature", 0.3)

        try:
            import torch

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")

            # Move to device
            if self.use_int8:
                # For INT8, model handles device placement
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=kwargs.get("top_p", 0.9),
                    top_k=kwargs.get("top_k", 50),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode output
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"Generation failed for {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def get_tokenizer(self, model_key: str) -> Optional[Any]:
        """Get tokenizer for a model."""
        return self._tokenizers.get(model_key)

    def get_model(self, model_key: str) -> Optional[Any]:
        """
        Get model by key.

        Args:
            model_key: Model identifier (yolov5s, qwen-2b-perception, qwen-2b-trajectory, qwen-4b, qwen-9b)

        Returns:
            Model instance or None
        """
        return self._models.get(model_key)

    def detect_objects(self, image: Any, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detect objects using YOLO.

        Args:
            image: RGB image (numpy array or PIL Image)
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of detection dictionaries
        """
        # Use very low threshold for indoor scenes to detect more objects
        confidence_threshold = min(confidence_threshold, 0.05)

        self.logger.info(f"[detect_objects] Called with image type: {type(image)}, threshold: {confidence_threshold}")

        yolo = self.get_model("yolov5s")
        self.logger.info(f"[detect_objects] YOLO model: {yolo is not None}")

        if yolo is None:
            self.logger.warning("[detect_objects] YOLO model not loaded - returning empty list")
            return []

        if image is None:
            self.logger.warning("[detect_objects] YOLO input image is None - returning empty list")
            return []

        # Debug logging for input image (INFO level for visibility)
        if isinstance(image, np.ndarray):
            self.logger.info(f"[detect_objects] YOLO input: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")

        try:
            import tempfile
            import os
            from PIL import Image as PILImage

            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                # Ensure uint8 type
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image

            # Convert RGBA to RGB if necessary
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Save to temp file and reload (workaround for ultralytics bug)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                pil_image.save(tmp_path)

            try:
                # Run inference on file path with specified confidence threshold
                results = yolo(tmp_path, conf=confidence_threshold, verbose=False)
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    if confidence < confidence_threshold:
                        continue

                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detections.append({
                        "name": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "class_id": class_id,
                    })

            # Log detection results (INFO level for visibility)
            self.logger.info(f"YOLO detected {len(detections)} objects with conf >= {confidence_threshold}")
            for det in detections[:5]:
                self.logger.info(f"  - {det['name']}: {det['confidence']:.2f}")

            # Save debug image if no objects detected
            if len(detections) == 0:
                self.logger.warning("[YOLO DEBUG] No objects detected! Saving debug image...")
                try:
                    import cv2
                    debug_path = "/tmp/yolo_debug.jpg"
                    if isinstance(image, np.ndarray):
                        # Ensure proper format for saving
                        if image.dtype != np.uint8:
                            if image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)
                        cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        self.logger.warning(f"[YOLO DEBUG] Image saved to {debug_path}")
                except Exception as e:
                    self.logger.warning(f"[YOLO DEBUG] Failed to save debug image: {e}")

            return detections

        except TypeError as e:
            # Handle numpy/ultralytics compatibility issue gracefully
            # Return empty detections but log only once
            if not hasattr(self, '_yolo_error_logged'):
                self.logger.warning(f"YOLO numpy compatibility issue - object detection disabled: {e}")
                self._yolo_error_logged = True
            return []
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return []

    def estimate_distance(
        self,
        depth_image: Any,
        bbox: List[float],
        depth_scale: float = 1.0
    ) -> float:
        """
        Estimate distance to object using depth image.

        Args:
            depth_image: Depth image (H, W)
            bbox: Bounding box [x1, y1, x2, y2]
            depth_scale: Scale factor for depth values

        Returns:
            Estimated distance in meters
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure valid bounds
            h, w = depth_image.shape[:2]
            x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
            y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                return 0.0

            # Get depth values in bbox region
            region = depth_image[y1:y2, x1:x2]

            # Use median to handle noise
            valid_depths = region[region > 0]
            if len(valid_depths) == 0:
                return 0.0

            distance = float(np.median(valid_depths)) * depth_scale
            return distance

        except Exception as e:
            self.logger.error(f"Distance estimation failed: {e}")
            return 0.0

    def estimate_angle(self, bbox: List[float], image_width: int, fov: float = 90.0) -> float:
        """
        Estimate angle to object from center of view.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            image_width: Image width in pixels
            fov: Field of view in degrees

        Returns:
            Angle in degrees (positive = right, negative = left)
        """
        try:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            image_center = image_width / 2

            # Calculate angle
            offset = (center_x - image_center) / image_width
            angle = offset * (fov / 2)

            return angle

        except Exception as e:
            self.logger.error(f"Angle estimation failed: {e}")
            return 0.0

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
        except:
            pass

        return {"error": "CUDA not available"}

    def clear_model(self, model_key: str) -> None:
        """Clear specific model from memory."""
        if model_key in self._models:
            del self._models[model_key]

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        self.logger.info(f"Cleared model: {model_key}")

    def clear_all(self) -> None:
        """Clear all models from memory."""
        self._models.clear()
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        self._models_loaded = False
        self.logger.info("Cleared all models")


# Global instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(config: Dict[str, Any] = None) -> ModelManager:
    """Get global ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(config)
    return _model_manager