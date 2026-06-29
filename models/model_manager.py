"""Model Manager for local and remote model management.

Manages Qwen3.5 series models with INT8 quantization:
- Qwen3.5-4B (perception): For PerceptionAgent visual descriptions
- Qwen3.5-2B (trajectory): For TrajectoryAgent path summarization
- Qwen3.5-4B: For DecisionAgent action selection
- Qwen3.5-4B (evaluation): For EvaluationAgent decision assessment
- YOLOv5s: Object detection (compatible with numpy 2.x)

Supports dual-environment IPC architecture:
- Local mode: Load models directly (requires Python 3.10+)
- Remote mode: Use HTTP API to communicate with LLM server

Total VRAM (INT8, 方案二): ~14.1GB + Habitat ~2GB = ~18GB
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
    - Qwen3.5-9B-AWQ (perception): Visual descriptions (~9GB AWQ)
    - Qwen3.5-2B (trajectory): Path summarization (~2.1GB INT8)
    - Qwen3.5-9B-AWQ: Navigation decisions (~9GB AWQ)
    - Qwen3.5-9B-AWQ (evaluation): Decision evaluation (~9GB AWQ)

    Total VRAM: ~18GB with AWQ quantization
    """

    _instance = None
    _lock = threading.Lock()

    # Multi-tier model configuration for heterogeneous agent allocation.
    # Tier 1 (VLM):  Qwen3-VL-8B-Instruct - dedicated vision-language model for perception
    # Tier 2 (Fast): Qwen3.5-9B-AWQ - lightweight LLM for latency-sensitive operations
    # Tier 3 (Strong): Qwen3.6-35B-A3B - MoE LLM for complex reasoning (debate/reflection/planning)
    MODEL_CONFIGS = {
        # === Tier 1: Dedicated VLM for perception ===
        "qwen3-vl-8b": {
            "type": "vlm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3-VL-8B-Instruct",
            "vram_gb": 16.0,
            "load_time": 20.0,
            "max_new_tokens": 400,
            "temperature": 0.2,
            "description": "Qwen3-VL-8B: dedicated VLM for structured scene perception",
        },
        # === Tier 2: Fast LLM for simple/low-latency operations ===
        "qwen3.5-9b-fast": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "load_time": 15.0,
            "max_new_tokens": 256,
            "temperature": 0.2,
            "description": "Qwen3.5-9B: fast LLM for subtask decomposition, emergency, review",
        },
        # === Tier 3: Strong MoE LLM for complex reasoning ===
        "qwen3.6-35b-strong": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B",
            "vram_gb": 20.0,
            "load_time": 25.0,
            "max_new_tokens": 400,
            "temperature": 0.3,
            "description": "Qwen3.6-35B-A3B: strong MoE LLM for debate, reflection, planning",
        },
        # === Backward-compatible aliases (map to new tier-2 model) ===
        "qwen-9b-perception": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "load_time": 15.0,
            "max_new_tokens": 256,
            "temperature": 0.3,
            "description": "Backward compat: maps to qwen3.5-9b-fast",
        },
        "qwen-9b": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "load_time": 15.0,
            "max_new_tokens": 150,
            "temperature": 0.1,
            "description": "Backward compat: default pipeline model",
        },
        "qwen-9b-decision": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "load_time": 15.0,
            "max_new_tokens": 300,
            "temperature": 0.1,
            "description": "Backward compat: maps to qwen3.5-9b-fast",
        },
        "qwen-9b-instruction": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "load_time": 15.0,
            "max_new_tokens": 500,
            "temperature": 0.1,
            "description": "Backward compat: maps to qwen3.5-9b-fast",
        },
        "qwen-9b-evaluation": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "load_time": 15.0,
            "max_new_tokens": 150,
            "temperature": 0.2,
            "description": "Backward compat: maps to qwen3.5-9b-fast",
        },
        "qwen-9b-trajectory": {
            "type": "llm",
            "model_name": "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ",
            "vram_gb": 8.0,
            "max_new_tokens": 200,
            "temperature": 0.2,
            "description": "Backward compat: maps to qwen3.5-9b-fast",
        },
        "qwen2-vl-2b": {
            "type": "vlm",
            "model_name": "/data/WZ/Model/Qwen/Qwen2-VL-2B-Instruct",
            "vram_gb": 4.0,
            "load_time": 15.0,
            "max_new_tokens": 256,
            "temperature": 0.3,
            "optional": True,
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
        self.use_remote = self.config.get("use_remote", self.config.get("use_remote_llm", False))
        self.remote_server_url = self.config.get("remote_server_url", self.config.get("llm_server_url", "http://localhost:8000"))
        self.model_path = self.config.get("model_path", None)  # Dynamic model path for vLLM

        # SiliconFlow API configuration
        self.use_siliconflow = self.config.get("use_siliconflow", False)
        self.siliconflow_api_key = self.config.get("siliconflow_api_key")
        self.siliconflow_model = self.config.get("siliconflow_model")
        # Local VLM server for vision tasks (can be different from SiliconFlow)
        self.vlm_server_url = self.config.get("vlm_server_url", "http://localhost:8000")

        self._remote_client = None
        self._vlm_client = None  # Separate client for VLM
        self._remote_healthy = False

        # Model storage
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}

        # Loading state
        self._initialized = True
        self._models_loaded = False
        self._llm_loaded = False

        # Initialize remote client if needed
        if self.use_remote or self.use_siliconflow:
            self._init_remote_client()

        if self.use_siliconflow:
            mode = "siliconflow+local_vlm"
        elif self.use_remote:
            mode = "remote"
        else:
            mode = "local"
        self.logger.info(f"ModelManager initialized (device={self.device}, int8={self.use_int8}, mode={mode})")

    def _init_remote_client(self) -> bool:
        """Initialize remote LLM client."""
        try:
            from models.remote_client import RemoteLLMClient

            # Initialize SiliconFlow client for text LLM if enabled
            if self.use_siliconflow:
                # SiliconFlow needs longer timeout for large models
                siliconflow_timeout = self.config.get("remote_timeout", 180.0)
                self._remote_client = RemoteLLMClient(
                    server_url="https://api.siliconflow.cn",
                    timeout=siliconflow_timeout,
                    use_openai=False,
                    use_siliconflow=True,
                    siliconflow_api_key=self.siliconflow_api_key,
                    siliconflow_model=self.siliconflow_model,
                )
                self.logger.info(f"Using SiliconFlow API for text LLM (timeout={siliconflow_timeout}s)")
                self._remote_healthy = True

                # Initialize separate VLM client for local vLLM server (HTTP mode, not OpenAI mode)
                vlm_timeout = self.config.get("vlm_timeout", 60.0)
                self._vlm_client = RemoteLLMClient(
                    server_url=self.vlm_server_url,
                    timeout=vlm_timeout,
                    use_openai=False,  # Use HTTP mode for custom vllm_server.py
                    use_siliconflow=False,
                    model_path=self.model_path,  # Dynamic model path override
                )
                self.logger.info(f"Using local VLM server (HTTP mode): {self.vlm_server_url}")
                return True
            else:
                # Use vLLM server for all LLM (text + VLM)
                # use_openai=True: use vLLM's native OpenAI-compatible API (/v1/chat/completions)
                self._remote_client = RemoteLLMClient(
                    server_url=self.remote_server_url,
                    timeout=self.config.get("remote_timeout", 60.0),
                    use_openai=True,  # Use vLLM's OpenAI-compatible API
                    model_path=self.model_path,  # Dynamic model path override
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

    def load_llm(self, model_key: str) -> bool:
        """
        Load a Qwen LLM model with INT8 quantization.

        Models with the same model_path are shared to save VRAM.
        For example: qwen-9b-instruction and qwen-9b-trajectory share the same Qwen3.5-2B model.

        Args:
            model_key: Model identifier (qwen-9b-instruction, qwen-9b-trajectory, qwen-9b, etc.)

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

        model_path = config["model_name"]

        # Check if a model with the same path is already loaded (model sharing)
        for existing_key, existing_config in self.MODEL_CONFIGS.items():
            if existing_key in self._models and existing_config["model_name"] == model_path and existing_key != model_key:
                self.logger.info(f"[load_llm] Sharing model {existing_key} -> {model_key} (same path: {model_path})")
                self._models[model_key] = self._models[existing_key]
                self._tokenizers[model_key] = self._tokenizers[existing_key]
                return True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

        # 方案二: 9B AWQ perception + 2B trajectory + 9B AWQ decision + 4B instruction + 9B AWQ evaluation
        llm_keys = ["qwen-9b-perception", "qwen-9b-trajectory", "qwen-9b-decision", "qwen-9b-instruction", "qwen-9b-evaluation"]
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
        lora_name: Optional[str] = None,
        seed: Optional[int] = None,
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
            lora_name: Optional LoRA adapter name (e.g., "decision-lora")
            seed: Random seed for deterministic output (default: 42)
            **kwargs: Additional generation kwargs

        Returns:
            Generated text
        """
        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        # Use remote generation if configured
        if self.use_remote and self._remote_client:
            return self._generate_remote(
                model_key=model_key,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lora_name=lora_name,
                seed=seed_value,
                **kwargs
            )

        # Local generation (LoRA not supported in local mode yet)
        return self._generate_local(
            model_key=model_key,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )

    def generate_sync(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        lora_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Synchronous generation using remote LLM server.

        This is a convenience method that wraps the remote client's generate_sync.
        Primarily used by pipeline agents for synchronous LLM calls.

        Args:
            model_key: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            lora_name: Optional LoRA adapter name
            seed: Random seed for deterministic output (default: 42)
            **kwargs: Additional generation kwargs

        Returns:
            Generated text string (empty string on failure)
        """
        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        if not self.use_remote or not self._remote_client:
            self.logger.warning("generate_sync requires remote LLM mode")
            # Fall back to regular generate for local mode
            return self.generate(
                model_key=model_key,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lora_name=lora_name,
                seed=seed_value,
                **kwargs
            )

        config = self.MODEL_CONFIGS.get(model_key, {})
        if max_new_tokens is None:
            max_new_tokens = config.get("max_new_tokens", 256)
        if temperature is None:
            temperature = config.get("temperature", 0.3)

        try:
            result = self._remote_client.generate_sync(
                model=model_key,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lora_name=lora_name,
            )
            return result.response if result and hasattr(result, 'response') else ""
        except Exception as e:
            self.logger.error(f"generate_sync failed: {e}")
            return ""

    def generate_vision(
        self,
        image: Any,
        prompt: str,
        model_key: str = "qwen-9b-perception",
        max_new_tokens: int = None,
        temperature: float = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text from image using VLM.

        Uses remote VLM server for inference.

        Args:
            image: PIL Image or numpy array
            prompt: Input prompt for generation
            model_key: VLM model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed for deterministic output (default: 42)

        Returns:
            Dictionary with 'response', 'objects', 'scene_description', 'nav_hint'
        """
        # For VLM, use local vLLM server (not SiliconFlow)
        vlm_client = self._vlm_client if self.use_siliconflow else self._remote_client
        if not (self.use_remote or self.use_siliconflow) or not vlm_client:
            self.logger.warning("VLM requires remote server mode or local VLM server")
            return self._get_vlm_fallback(prompt)

        config = self.MODEL_CONFIGS.get(model_key, {})
        if max_new_tokens is None:
            max_new_tokens = config.get("max_new_tokens", 256)
        if temperature is None:
            temperature = config.get("temperature", 0.3)

        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        try:
            result = vlm_client.generate_vision(
                image=image,
                prompt=prompt,
                model=model_key,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed_value,
            )

            if result.error:
                self.logger.warning(f"VLM generation error: {result.error}")
                return self._get_vlm_fallback(prompt)

            # Parse VLM response
            return self._parse_vlm_response(result.response)

        except Exception as e:
            self.logger.error(f"VLM generation failed: {e}")
            return self._get_vlm_fallback(prompt)

    def generate_vision_dual(
        self,
        rgb_image: Any,
        depth_image: Any,
        prompt: str,
        model_key: str = "qwen-9b-perception",
        max_new_tokens: int = None,
        temperature: float = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text from RGB + Depth images using VLM.

        Depth image is converted to JET colormap (red=near, blue=far)
        for better visualization by the VLM.

        Args:
            rgb_image: PIL Image or numpy array for RGB
            depth_image: numpy array for depth (in meters)
            prompt: Input prompt for generation
            model_key: VLM model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed for deterministic output (default: 42)

        Returns:
            Dictionary with 'response', 'objects', 'scene_description', 'nav_hint'
        """
        self.logger.debug(f"[generate_vision_dual] Called with model_key={model_key}, max_tokens={max_new_tokens}, temp={temperature}")
        self.logger.debug(f"[generate_vision_dual] Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"[generate_vision_dual] Prompt: {prompt}")

        # For VLM, use local vLLM server (not SiliconFlow)
        vlm_client = self._vlm_client if self.use_siliconflow else self._remote_client
        if not (self.use_remote or self.use_siliconflow) or not vlm_client:
            self.logger.warning("VLM requires remote server mode or local VLM server")
            return self._get_vlm_fallback(prompt)

        config = self.MODEL_CONFIGS.get(model_key, {})
        if max_new_tokens is None:
            max_new_tokens = config.get("max_new_tokens", 256)
        if temperature is None:
            temperature = config.get("temperature", 0.3)

        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        try:
            self.logger.debug(f"[generate_vision_dual] Calling VLM client...")
            result = vlm_client.generate_vision_dual(
                rgb_image=rgb_image,
                depth_image=depth_image,
                prompt=prompt,
                model=model_key,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed_value,
            )

            if result.error:
                self.logger.warning(f"VLM dual-vision generation error: {result.error}")
                return self._get_vlm_fallback(prompt)

            self.logger.debug(f"[generate_vision_dual] Raw response length: {len(result.response) if result.response else 0}")
            self.logger.debug(f"[generate_vision_dual] Response preview: {result.response[:300] if result.response else 'None'}...")

            # Parse VLM response
            parsed = self._parse_vlm_response(result.response)
            self.logger.debug(f"[generate_vision_dual] Parsed: room_type={parsed.get('room_type')}, objects={len(parsed.get('objects', []))}, nav_hint={parsed.get('nav_hint', '')[:50]}")
            return parsed

        except Exception as e:
            self.logger.error(f"VLM dual-vision generation failed: {e}")
            return self._get_vlm_fallback(prompt)

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into structured format.

        Supports both JSON-formatted responses (preferred) and plain text.

        Args:
            response: Raw VLM response text

        Returns:
            Dictionary with parsed fields
        """
        import json
        import re

        result = {
            "response": response,
            "objects": [],
            "scene_description": "",
            "nav_hint": "",
        }

        if not response:
            return result

        # Try to extract JSON from response
        json_str = response
        if '```json' in response:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
        elif '```' in response:
            json_match = re.search(r'```\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()

        # Try to parse JSON - handle multiple JSON objects in response
        # Strategy: find the first complete valid JSON object
        start = json_str.find('{')
        if start != -1:
            # Try to find matching closing brace using stack
            brace_count = 0
            end = -1
            for i in range(start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break

            if end != -1:
                json_candidate = json_str[start:end+1]
                self.logger.debug(f"JSON candidate length: {len(json_candidate)}")
                try:
                    data = json.loads(json_candidate)
                    self.logger.debug(f"JSON parsed: room_type={data.get('room_type')}, objects={len(data.get('objects', []))}")

                    # Extract room_type for scene_description
                    room_type = data.get("room_type", "")
                    scene_brief = data.get("scene_brief", data.get("scene_description", ""))
                    if room_type and scene_brief:
                        result["scene_description"] = f"{room_type}: {scene_brief}"
                    elif scene_brief:
                        result["scene_description"] = scene_brief
                    elif room_type:
                        result["scene_description"] = room_type

                    # Extract nav_hint
                    result["nav_hint"] = data.get("nav_hint", "")

                    # Extract objects - normalize field names to "name"
                    # VLM may output objects with different field names
                    if "objects" in data and isinstance(data["objects"], list):
                        for obj in data["objects"]:
                            if isinstance(obj, dict):
                                # Try multiple possible field names for object name
                                name = (obj.get("name") or
                                        obj.get("object") or
                                        obj.get("object_name") or
                                        obj.get("type") or "")
                                if name:
                                    # Output unified format: {"name": "object_name"}
                                    # Optionally preserve distance if present
                                    normalized_obj = {"name": name}
                                    if "distance" in obj:
                                        normalized_obj["distance"] = obj["distance"]
                                    result["objects"].append(normalized_obj)
                                    self.logger.debug(f"[VLM parse] Normalized object: {normalized_obj}")
                        self.logger.debug(f"[VLM parse] Objects count: {len(result['objects'])}, format: List[Dict] with 'name' field")

                    return result
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.debug(f"JSON parsing failed: {e}")

        # Fallback: line-by-line parsing for non-JSON responses
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for object mentions - output unified dict format
            if '物体' in line or 'object' in line.lower() or '检测到' in line or 'detected' in line.lower():
                result["objects"].append({"name": line})

            # Look for navigation hints
            if '导航' in line or '建议' in line or '方向' in line or 'navigation' in line.lower() or 'suggest' in line.lower() or 'direction' in line.lower():
                if not result["nav_hint"]:  # Only set if not already set from JSON
                    result["nav_hint"] = line

        # First meaningful line as scene description
        if not result["scene_description"] and lines:
            result["scene_description"] = lines[0].strip()

        return result

    def _get_vlm_fallback(self, prompt: str) -> Dict[str, Any]:
        """Get fallback VLM response.

        Args:
            prompt: Original prompt

        Returns:
            Fallback response dictionary
        """
        return {
            "response": "视觉分析暂时不可用。",
            "objects": [],
            "scene_description": "视觉分析暂时不可用，请继续探索。",
            "nav_hint": "继续前进探索环境。",
        }

    def _generate_remote(
        self,
        model_key: str,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = None,
        lora_name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using remote LLM server.

        Args:
            model_key: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            lora_name: Optional LoRA adapter name
            seed: Random seed for deterministic output
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        if not self._remote_client:
            self.logger.error("Remote client not initialized")
            return ""

        # For unified model server, use model_key directly
        # Server has all qwen-9b-* aliases registered to the same engine
        remote_model_key = model_key

        config = self.MODEL_CONFIGS.get(model_key, {})
        if max_new_tokens is None:
            max_new_tokens = config.get("max_new_tokens", 256)
        if temperature is None:
            temperature = config.get("temperature", 0.3)

        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        try:
            result = self._remote_client.generate(
                model=remote_model_key,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                conversation_id=kwargs.get("conversation_id"),
                keep_context=kwargs.get("keep_context", False),
                lora_name=lora_name,
                seed=seed_value,
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
        except Exception as e:
            self.logger.debug(f"Failed to get available models: {e}")

        # Return default 4B path as fallback
        return [self.MODEL_CONFIGS["qwen-9b-perception"]["model_name"]]

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
            model_key: Model identifier (yolov5s, qwen-9b-perception, qwen-9b-trajectory, qwen-9b, qwen-9b)

        Returns:
            Model instance or None
        """
        return self._models.get(model_key)

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