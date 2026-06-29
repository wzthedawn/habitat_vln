"""Remote LLM Client for communicating with Python 3.10 LLM server.

This module provides an HTTP client for the LLM inference server,
enabling the Python 3.9 VLN environment to use Qwen3.5 models
running in a separate Python 3.10 process.

Usage:
    from models.remote_client import RemoteLLMClient

    client = RemoteLLMClient("http://localhost:8000")

    # Generate text
    response = await client.generate(
        model="qwen-9b-perception",
        prompt="描述当前场景...",
        max_new_tokens=200
    )
"""

import asyncio
import base64
import logging
import time
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# SiliconFlow API configuration
SILICONFLOW_API_KEY = "sk-fjoebpuejranwmbdywtoilezpypfsodztntpfpfpfskqekgj"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "Qwen/Qwen3.5-397B-A17B"


@dataclass
class GenerateResult:
    """Result from text generation."""
    response: str
    model: str
    tokens_generated: int
    latency_ms: float
    conversation_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.response)


@dataclass
class GenerateVisionResult:
    """Result from vision-language generation."""
    response: str
    model: str
    tokens_generated: int
    latency_ms: float
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.response)


class RemoteLLMClient:
    """HTTP client for LLM inference server.

    Provides both async and sync methods for generating text
    using remote Qwen3.5 models.

    Supports three modes:
    - OpenAI mode: Uses vLLM's OpenAI-compatible server
    - SiliconFlow mode: Uses SiliconFlow API (GLM-5 model)
    - HTTP mode (fallback): Uses custom FastAPI server

    Attributes:
        server_url: Base URL of the LLM server
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_enabled: Whether to use fallback responses on error
        use_openai: Whether to use OpenAI SDK format
        use_siliconflow: Whether to use SiliconFlow API
    """

    # Model path mapping for vLLM OpenAI server (all use same Qwen3.6-35B-A3B model)
    MODEL_PATHS = {
        "qwen-9b-perception": "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B",
        "qwen-9b-instruction": "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B",
        "qwen-9b-decision": "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B",
        "qwen-9b-evaluation": "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B",
        "qwen-9b-trajectory": "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B",
    }

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 30.0,  # Lower timeout for faster failure
        max_retries: int = 2,   # Reduced retries
        retry_delay: float = 1.0,
        fallback_enabled: bool = True,  # Enable fallback by default
        use_openai: bool = False,  # Use HTTP mode by default (custom endpoints)
        use_siliconflow: bool = False,  # Use SiliconFlow API
        siliconflow_api_key: Optional[str] = None,  # Optional custom API key
        siliconflow_model: Optional[str] = None,  # Optional custom model
        model_path: Optional[str] = None,  # Dynamic model path override
    ):
        """Initialize the remote LLM client.

        Args:
            server_url: Base URL of the LLM server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            fallback_enabled: Whether to use fallback responses on error
            use_openai: Whether to use OpenAI SDK format (recommended for vLLM)
            use_siliconflow: Whether to use SiliconFlow API
            siliconflow_api_key: Optional custom API key (default: built-in key)
            siliconflow_model: Optional custom model (default: Pro/zai-org/GLM-5)
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_enabled = fallback_enabled
        self.logger = logging.getLogger("RemoteLLMClient")
        self.model_path = model_path  # Dynamic model path override

        # SiliconFlow configuration
        self.use_siliconflow = use_siliconflow
        self.siliconflow_api_key = siliconflow_api_key or SILICONFLOW_API_KEY
        self.siliconflow_model = siliconflow_model or SILICONFLOW_MODEL

        # Check available libraries
        if not AIOHTTP_AVAILABLE and not REQUESTS_AVAILABLE:
            raise ImportError(
                "Neither aiohttp nor requests is installed. "
                "Install with: pip install aiohttp or pip install requests"
            )

        # Initialize OpenAI client for SiliconFlow or vLLM
        self.use_openai = False
        self.openai_client = None

        if self.use_siliconflow and OPENAI_AVAILABLE:
            # Use SiliconFlow API
            self.openai_client = OpenAI(
                api_key=self.siliconflow_api_key,
                base_url=SILICONFLOW_BASE_URL,
                timeout=timeout
            )
            self.logger.info(f"Using SiliconFlow API with model: {self.siliconflow_model}")
        elif use_openai and OPENAI_AVAILABLE:
            # Use vLLM OpenAI-compatible server
            self.use_openai = True
            self.openai_client = OpenAI(
                api_key="EMPTY",
                base_url=f"{self.server_url}/v1",
                timeout=timeout
            )
            self.logger.info("Using OpenAI SDK mode for vLLM server")

        # Cache for health check
        self._last_health_check: Optional[float] = None
        self._healthy: bool = False

    async def generate_async(
        self,
        model: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        conversation_id: Optional[str] = None,
        keep_context: bool = False,
        lora_name: Optional[str] = None,
    ) -> GenerateResult:
        """Generate text using remote model (async).

        Args:
            model: Model identifier (e.g., "qwen-9b-perception")
            prompt: Input prompt for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            conversation_id: Optional ID for multi-turn conversations
            keep_context: Whether to keep conversation context
            lora_name: Optional LoRA adapter name (e.g., "decision-lora")

        Returns:
            GenerateResult with response and metadata
        """
        if not AIOHTTP_AVAILABLE:
            # Fallback to sync if aiohttp not available
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.generate_sync(
                    model=model,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    conversation_id=conversation_id,
                    keep_context=keep_context,
                    lora_name=lora_name,
                )
            )

        payload = {
            "model": model,
            "prompt": prompt,
        }

        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if lora_name is not None:
            payload["lora_name"] = lora_name
        payload["keep_context"] = keep_context

        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_url}/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return GenerateResult(
                                response=data.get("response", ""),
                                model=data.get("model", model),
                                tokens_generated=data.get("tokens_generated", 0),
                                latency_ms=data.get("latency_ms", 0),
                                conversation_id=data.get("conversation_id"),
                                error=data.get("error"),
                            )
                        else:
                            error_text = await response.text()
                            last_error = f"HTTP {response.status}: {error_text}"

                            # Don't retry on client errors
                            if response.status < 500:
                                break

            except asyncio.TimeoutError:
                last_error = f"Request timed out after {self.timeout}s"
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
            except aiohttp.ClientError as e:
                last_error = f"Connection error: {e}"
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay)

        return GenerateResult(
            response="",
            model=model,
            tokens_generated=0,
            latency_ms=0,
            error=last_error or "Unknown error",
        )

    def generate_sync(
        self,
        model: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        conversation_id: Optional[str] = None,
        keep_context: bool = False,
        lora_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        """Generate text using remote model (sync).

        Args:
            model: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            conversation_id: Optional ID for multi-turn conversations
            keep_context: Whether to keep conversation context
            lora_name: Optional LoRA adapter name
            seed: Random seed for deterministic output (default: 42)

        Returns:
            GenerateResult with response and metadata
        """
        # Use OpenAI mode if available
        if self.use_openai and self.openai_client:
            return self.generate_openai(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lora_name=lora_name,
                seed=seed,
            )

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is not installed. Install with: pip install requests")

        payload = {
            "model": model,
            "prompt": prompt,
        }

        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if lora_name is not None:
            payload["lora_name"] = lora_name
        payload["keep_context"] = keep_context

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/generate",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    return GenerateResult(
                        response=data.get("response", ""),
                        model=data.get("model", model),
                        tokens_generated=data.get("tokens_generated", 0),
                        latency_ms=data.get("latency_ms", 0),
                        conversation_id=data.get("conversation_id"),
                        error=data.get("error"),
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if response.status_code < 500:
                        break

            except requests.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.RequestException as e:
                last_error = f"Connection error: {e}"
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return GenerateResult(
            response="",
            model=model,
            tokens_generated=0,
            latency_ms=0,
            error=last_error or "Unknown error",
        )

    def generate(
        self,
        model: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        conversation_id: Optional[str] = None,
        keep_context: bool = False,
        lora_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text using remote model (convenience method).

        This is a sync method that returns just the response string.
        Uses fallback responses when LLM fails and fallback_enabled=True.

        Args:
            model: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            conversation_id: Optional ID for multi-turn conversations
            keep_context: Whether to keep conversation context
            lora_name: Optional LoRA adapter name (e.g., "decision-lora")
            seed: Random seed for deterministic output (default: 42)

        Returns:
            Generated text, or fallback response on error
        """
        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        # Use SiliconFlow API if enabled
        if self.use_siliconflow:
            result = self.generate_siliconflow(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        # Use OpenAI mode if available
        elif self.use_openai:
            result = self.generate_openai(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                lora_name=lora_name,
                seed=seed_value,
            )
        else:
            result = self.generate_sync(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                conversation_id=conversation_id,
                keep_context=keep_context,
                lora_name=lora_name,
            )

        if result.error:
            # LLM 不可用时直接报错，停止实验
            raise RuntimeError(f"LLM 服务不可用 [{model}]: {result.error}")

        return result.response

    def _get_fallback_response(self, model: str, prompt: str) -> str:
        """Get fallback response when LLM fails.

        Args:
            model: Model identifier
            prompt: Original prompt

        Returns:
            Fallback response string
        """
        # Perception fallback (4B model)
        if "perception" in model:
            return "前方视野开阔，未检测到明显障碍物。当前场景需要进一步探索。"

        # Trajectory fallback
        elif "trajectory" in model:
            return "导航进行中，继续前进探索环境。"

        # Decision fallback
        elif "decision" in model or "4b" in model:
            # Default to forward motion
            return '{"reasoning":"LLM 服务不可用，使用默认探索策略","subtask_completed":false,"actions":[{"action":"forward"},{"action":"forward"},{"action":"turn_left"},{"action":"forward"},{"action":"forward"},{"action":"turn_right"},{"action":"forward"},{"action":"forward"},{"action":"turn_left"},{"action":"forward"}]}'

        # Evaluation fallback
        elif "evaluation" in model:
            return '{"score": 0.5, "feedback": "评估完成", "suggestions": []}'

        # Generic fallback
        return "继续执行。"

    def _get_model_path(self, model_key: str) -> str:
        """Get model path for vLLM server.

        Args:
            model_key: Model identifier (e.g., qwen-9b-perception)

        Returns:
            Actual model name for vLLM server (e.g., qwen-9b)
        """
        # Priority: 1. Dynamic model_path override, 2. Use unified alias
        if self.model_path:
            return self.model_path
        # All qwen-9b-* aliases map to unified qwen-9b (vLLM --served-model-name)
        if model_key.startswith("qwen-9b"):
            return "qwen-9b"
        return self.MODEL_PATHS.get(model_key, model_key)

    def generate_siliconflow(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerateResult:
        """Generate text using SiliconFlow API.

        This method uses SiliconFlow's OpenAI-compatible API with GLM-5 model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            GenerateResult with response and metadata
        """
        if not self.openai_client:
            return GenerateResult(
                response="",
                model=self.siliconflow_model,
                tokens_generated=0,
                latency_ms=0,
                error="SiliconFlow client not initialized"
            )

        start_time = time.time()

        try:
            response = self.openai_client.chat.completions.create(
                model=self.siliconflow_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens or 1500,
                temperature=temperature if temperature is not None else 0.7,
            )

            latency = (time.time() - start_time) * 1000
            self.logger.info(f"[SiliconFlow] Generated in {latency:.0f}ms")

            return GenerateResult(
                response=response.choices[0].message.content or "",
                model=self.siliconflow_model,
                tokens_generated=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
            )

        except Exception as e:
            self.logger.error(f"SiliconFlow generation failed: {e}")
            return GenerateResult(
                response="",
                model=self.siliconflow_model,
                tokens_generated=0,
                latency_ms=0,
                error=str(e)
            )

    def generate_vision_siliconflow(
        self,
        image: Any,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth_image: Optional[Any] = None,
    ) -> GenerateVisionResult:
        """Generate text from image using SiliconFlow VLM API.

        Args:
            image: PIL Image or numpy array
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            depth_image: Optional depth image (will be added as second image)

        Returns:
            GenerateVisionResult with response and metadata
        """
        if not self.openai_client:
            return GenerateVisionResult(
                response="",
                model=self.siliconflow_model,
                tokens_generated=0,
                latency_ms=0,
                error="SiliconFlow client not initialized"
            )

        start_time = time.time()

        try:
            # Convert RGB image to base64 data URL
            image_base64 = self._image_to_base64(image)
            image_url = f"data:image/jpeg;base64,{image_base64}"

            # Build content list
            content = []

            # Add RGB image
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

            # Add depth image if provided
            if depth_image is not None:
                depth_colored = self._depth_to_colormap(depth_image)
                depth_base64 = self._image_to_base64(depth_colored)
                depth_url = f"data:image/jpeg;base64,{depth_base64}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": depth_url}
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt
            })

            response = self.openai_client.chat.completions.create(
                model=self.siliconflow_model,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                max_tokens=max_new_tokens or 1500,
                temperature=temperature if temperature is not None else 0.3,
            )

            latency = (time.time() - start_time) * 1000

            self.logger.info(f"[SiliconFlow-VLM] Generated in {latency:.0f}ms")

            return GenerateVisionResult(
                response=response.choices[0].message.content or "",
                model=self.siliconflow_model,
                tokens_generated=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
            )

        except Exception as e:
            self.logger.error(f"SiliconFlow VLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            return GenerateVisionResult(
                response="",
                model=self.siliconflow_model,
                tokens_generated=0,
                latency_ms=0,
                error=str(e)
            )

    def generate_openai(
        self,
        model: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        lora_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GenerateResult:
        """Generate text using OpenAI SDK format.

        This method uses vLLM's OpenAI-compatible server.

        Args:
            model: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            lora_name: Optional LoRA adapter name

        Returns:
            GenerateResult with response and metadata
        """
        if not self.openai_client:
            return GenerateResult(
                response="",
                model=model,
                tokens_generated=0,
                latency_ms=0,
                error="OpenAI client not initialized"
            )

        start_time = time.time()
        model_path = self._get_model_path(model)

        # Build extra_body for vLLM
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        if lora_name:
            # vLLM supports LoRA via extra_body
            extra_body["lora_name"] = lora_name

        try:
            # Use fixed seed for deterministic output
            seed_value = seed if seed is not None else 42
            response = self.openai_client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens or 1500,
                temperature=temperature if temperature is not None else 0.2,
                seed=seed_value,
                extra_body=extra_body
            )

            latency = (time.time() - start_time) * 1000

            return GenerateResult(
                response=response.choices[0].message.content or "",
                model=model,
                tokens_generated=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
            )

        except Exception as e:
            self.logger.error(f"OpenAI generation failed: {e}")
            return GenerateResult(
                response="",
                model=model,
                tokens_generated=0,
                latency_ms=0,
                error=str(e)
            )

    def generate_vision_openai(
        self,
        image: Any,
        prompt: str,
        model: str = "qwen-9b-perception",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        depth_image: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> GenerateVisionResult:
        """Generate text from image using OpenAI SDK format.

        This method uses vLLM's OpenAI-compatible server with multimodal support.

        Args:
            image: PIL Image or numpy array
            prompt: Input prompt
            model: VLM model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            depth_image: Optional depth image (will be added as second image)

        Returns:
            GenerateVisionResult with response and metadata
        """
        if not self.openai_client:
            return GenerateVisionResult(
                response="",
                model=model,
                tokens_generated=0,
                latency_ms=0,
                error="OpenAI client not initialized"
            )

        start_time = time.time()
        model_path = self._get_model_path(model)

        try:
            # Convert RGB image to base64 data URL
            image_base64 = self._image_to_base64(image)
            image_url = f"data:image/jpeg;base64,{image_base64}"

            # Build content list
            content = []

            # Add RGB image
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

            # Add depth image if provided
            if depth_image is not None:
                depth_colored = self._depth_to_colormap(depth_image)
                depth_base64 = self._image_to_base64(depth_colored)
                depth_url = f"data:image/jpeg;base64,{depth_base64}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": depth_url}
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt
            })

            # Use dynamic seed based on time to avoid KV-cache collision
            import random
            seed_value = seed if seed is not None else random.randint(1, 999999)
            response = self.openai_client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                max_tokens=max_new_tokens or 1500,
                temperature=temperature if temperature is not None else 0.3,
                seed=seed_value,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}  # Disable Qwen3.5 thinking mode
            )

            latency = (time.time() - start_time) * 1000

            self.logger.info(f"[OpenAI-VLM] {model}: generated in {latency:.0f}ms")

            return GenerateVisionResult(
                response=response.choices[0].message.content or "",
                model=model,
                tokens_generated=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
            )

        except Exception as e:
            self.logger.error(f"OpenAI vision generation failed: {e}")
            import traceback
            traceback.print_exc()
            return GenerateVisionResult(
                response="",
                model=model,
                tokens_generated=0,
                latency_ms=0,
                error=str(e)
            )

    def generate_vision(
        self,
        image: Any,
        prompt: str,
        model: str = "qwen-9b-perception",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> GenerateVisionResult:
        """Generate text from image using VLM (sync).

        Automatically uses SiliconFlow or OpenAI mode if available, otherwise falls back to HTTP.

        Args:
            image: PIL Image or numpy array
            prompt: Input prompt
            model: VLM model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed for deterministic output (default: 42)

        Returns:
            GenerateVisionResult with response and metadata
        """
        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        # Use SiliconFlow mode if enabled
        if self.use_siliconflow:
            return self.generate_vision_siliconflow(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        # Use OpenAI mode if available
        if self.use_openai:
            return self.generate_vision_openai(
                image=image,
                prompt=prompt,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed_value,
            )

        # Fallback to HTTP mode
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is not installed. Install with: pip install requests")

        # Convert image to base64
        image_base64 = self._image_to_base64(image)


        payload = {
            "model": model,
            "prompt": prompt,
            "image_base64": image_base64,
        }

        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/generate_vision",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    return GenerateVisionResult(
                        response=data.get("response", ""),
                        model=data.get("model", model),
                        tokens_generated=data.get("tokens_generated", 0),
                        latency_ms=data.get("latency_ms", 0),
                        error=data.get("error"),
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if response.status_code < 500:
                        break

            except requests.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.RequestException as e:
                last_error = f"Connection error: {e}"
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return GenerateVisionResult(
            response="",
            model=model,
            tokens_generated=0,
            latency_ms=0,
            error=last_error or "Unknown error",
        )

    def generate_vision_dual(
        self,
        rgb_image: Any,
        depth_image: Any,
        prompt: str,
        model: str = "qwen-9b-perception",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> GenerateVisionResult:
        """Generate text from RGB + Depth images using VLM (sync).

        Automatically uses OpenAI mode if available, otherwise falls back to HTTP.
        Depth image will be converted to pseudocolor (JET colormap) for
        better visualization by the VLM.

        Args:
            rgb_image: PIL Image or numpy array for RGB
            depth_image: numpy array for depth (in meters)
            prompt: Input prompt
            model: VLM model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed for deterministic output (default: 42)

        Returns:
            GenerateVisionResult with response and metadata
        """
        self.logger.debug(f"[generate_vision_dual] Called: model={model}, max_tokens={max_new_tokens}, temp={temperature}, use_openai={self.use_openai}, use_siliconflow={self.use_siliconflow}")

        # Use fixed seed for deterministic output
        seed_value = seed if seed is not None else 42

        # Use SiliconFlow mode if enabled
        if self.use_siliconflow:
            self.logger.debug("[generate_vision_dual] Using SiliconFlow mode")
            return self.generate_vision_siliconflow(
                image=rgb_image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                depth_image=depth_image,
            )

        # Use OpenAI mode if available
        if self.use_openai:
            self.logger.debug("[generate_vision_dual] Using OpenAI mode")
            return self.generate_vision_openai(
                image=rgb_image,
                prompt=prompt,
                model=model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                depth_image=depth_image,
                seed=seed_value,
            )

        # Fallback to HTTP mode
        self.logger.debug("[generate_vision_dual] Using HTTP mode")
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is not installed. Install with: pip install requests")

        # Convert RGB image to base64
        rgb_base64 = self._image_to_base64(rgb_image)

        # Convert depth image to pseudocolor and then base64
        depth_colored = self._depth_to_colormap(depth_image)
        depth_base64 = self._image_to_base64(depth_colored)

        payload = {
            "model": model,
            "prompt": prompt,
            "image_base64": rgb_base64,
            "depth_base64": depth_base64,
        }

        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        self.logger.debug(f"[generate_vision_dual] Sending request to {self.server_url}/generate_vision")

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/generate_vision",
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    self.logger.debug(f"[generate_vision_dual] Response received: latency={data.get('latency_ms', 0)}ms, tokens={data.get('tokens_generated', 0)}")
                    self.logger.debug(f"[generate_vision_dual] Response text: {str(data.get('response', ''))[:200]}...")
                    return GenerateVisionResult(
                        response=data.get("response", ""),
                        model=data.get("model", model),
                        tokens_generated=data.get("tokens_generated", 0),
                        latency_ms=data.get("latency_ms", 0),
                        error=data.get("error"),
                    )
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    if response.status_code < 500:
                        break

            except requests.Timeout:
                last_error = f"Request timed out after {self.timeout}s"
                self.logger.warning(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
            except requests.RequestException as e:
                last_error = f"Connection error: {e}"
                self.logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return GenerateVisionResult(
            response="",
            model=model,
            tokens_generated=0,
            latency_ms=0,
            error=last_error or "Unknown error",
        )

    def _depth_to_colormap(self, depth_image: Any, max_depth: float = 10.0) -> "np.ndarray":
        """Convert depth image to JET colormap for visualization.

        Args:
            depth_image: Depth array in meters
            max_depth: Maximum depth for normalization

        Returns:
            RGB image with JET colormap applied
        """
        import numpy as np
        from PIL import Image as PILImage

        # Handle None or invalid input
        if depth_image is None:
            # Return a blank image
            return np.zeros((224, 224, 3), dtype=np.uint8)

        # Ensure numpy array
        if not hasattr(depth_image, 'shape'):
            depth_image = np.array(depth_image)

        # IMPORTANT: JET colormap 0=blue(far), 255=red(near)
        # depth值=距离，需要反转：小距离(近处)→大像素值→红色
        valid_mask = depth_image > 0
        normalized = np.zeros_like(depth_image, dtype=np.float32)
        if valid_mask.any():
            # 反转映射：近处(小depth)→255(红色)，远处(大depth)→0(蓝色)
            normalized[valid_mask] = 255 - np.clip(depth_image[valid_mask] / max_depth, 0, 1) * 255
        normalized_uint8 = normalized.astype(np.uint8)

        # Apply JET colormap: now 255(near)=red, 0(far)=blue
        try:
            import cv2
            colored = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
            # Convert BGR to RGB
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        except ImportError:
            # Fallback: simple grayscale to RGB
            colored = np.stack([normalized_uint8] * 3, axis=-1)

        return colored

    def _image_to_base64(self, image: Any) -> str:
        """Convert image to base64 string.

        Args:
            image: PIL Image, numpy array, or file path

        Returns:
            Base64 encoded image string
        """
        from io import BytesIO

        # Handle numpy array
        if hasattr(image, 'shape'):  # numpy array
            import numpy as np
            from PIL import Image as PILImage

            # Ensure uint8 type
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            pil_image = PILImage.fromarray(image)

        # Handle PIL Image
        elif hasattr(image, 'save'):
            pil_image = image

        # Handle file path
        elif isinstance(image, str):
            from PIL import Image as PILImage
            pil_image = PILImage.open(image)

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Save to buffer
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return image_base64

    async def health_check_async(self) -> Dict[str, Any]:
        """Check server health (async).

        Returns:
            Health status dictionary
        """
        if not AIOHTTP_AVAILABLE:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.health_check_sync
            )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        self._healthy = True
                        self._last_health_check = time.time()
                        return await response.json()
                    else:
                        self._healthy = False
                        return {"status": "unhealthy", "code": response.status}
        except Exception as e:
            self._healthy = False
            return {"status": "error", "error": str(e)}

    def health_check_sync(self) -> Dict[str, Any]:
        """Check server health (sync).

        Returns:
            Health status dictionary
        """
        if not REQUESTS_AVAILABLE:
            return {"status": "error", "error": "requests not installed"}

        try:
            # Use /v1/models endpoint for vLLM OpenAI-compatible server
            response = requests.get(
                f"{self.server_url}/v1/models",
                timeout=5.0
            )
            if response.status_code == 200:
                self._healthy = True
                self._last_health_check = time.time()
                data = response.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "") for m in models]
                return {
                    "status": "healthy",
                    "models_loaded": model_ids,
                }
            else:
                self._healthy = False
                return {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            self._healthy = False
            return {"status": "error", "error": str(e)}

    def health_check(self) -> bool:
        """Check if server is healthy (convenience method).

        Returns:
            True if server is healthy
        """
        result = self.health_check_sync()
        return result.get("status") == "healthy"

    @property
    def is_healthy(self) -> bool:
        """Check if server was recently healthy.

        Uses cached result if recent (within 30 seconds).

        Returns:
            True if server is believed to be healthy
        """
        if self._last_health_check is None:
            return False

        if time.time() - self._last_health_check > 30:
            return False

        return self._healthy

    async def list_models_async(self) -> Dict[str, Any]:
        """List available models (async).

        Returns:
            Dictionary of model configurations
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/models",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return {}
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return {}

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation context on server.

        Args:
            conversation_id: Conversation ID to clear

        Returns:
            True if cleared successfully
        """
        if not REQUESTS_AVAILABLE:
            return False

        try:
            response = requests.post(
                f"{self.server_url}/clear_conversation",
                json={"conversation_id": conversation_id},
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to clear conversation: {e}")
            return False


# Convenience function for quick usage
def generate(
    prompt: str,
    model: str = "qwen-9b",
    server_url: str = "http://localhost:8000",
    **kwargs
) -> str:
    """Quick generation function.

    Args:
        prompt: Input prompt
        model: Model identifier
        server_url: LLM server URL
        **kwargs: Additional arguments for generation

    Returns:
        Generated text
    """
    client = RemoteLLMClient(server_url)
    return client.generate(model=model, prompt=prompt, **kwargs)