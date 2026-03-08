"""Remote LLM Client for communicating with Python 3.10 LLM server.

This module provides an HTTP client for the LLM inference server,
enabling the Python 3.9 VLN environment to use Qwen3.5 models
running in a separate Python 3.10 process.

Usage:
    from models.remote_client import RemoteLLMClient

    client = RemoteLLMClient("http://localhost:8000")

    # Generate text
    response = await client.generate(
        model="qwen-2b-perception",
        prompt="描述当前场景...",
        max_new_tokens=200
    )
"""

import asyncio
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


class RemoteLLMClient:
    """HTTP client for LLM inference server.

    Provides both async and sync methods for generating text
    using remote Qwen3.5 models.

    Attributes:
        server_url: Base URL of the LLM server
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the remote LLM client.

        Args:
            server_url: Base URL of the LLM server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger("RemoteLLMClient")

        # Check available libraries
        if not AIOHTTP_AVAILABLE and not REQUESTS_AVAILABLE:
            raise ImportError(
                "Neither aiohttp nor requests is installed. "
                "Install with: pip install aiohttp or pip install requests"
            )

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
    ) -> GenerateResult:
        """Generate text using remote model (async).

        Args:
            model: Model identifier (e.g., "qwen-2b-perception")
            prompt: Input prompt for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            conversation_id: Optional ID for multi-turn conversations
            keep_context: Whether to keep conversation context

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
    ) -> GenerateResult:
        """Generate text using remote model (sync).

        Args:
            model: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            conversation_id: Optional ID for multi-turn conversations
            keep_context: Whether to keep conversation context

        Returns:
            GenerateResult with response and metadata
        """
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
    ) -> str:
        """Generate text using remote model (convenience method).

        This is a sync method that returns just the response string.

        Args:
            model: Model identifier
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            conversation_id: Optional ID for multi-turn conversations
            keep_context: Whether to keep conversation context

        Returns:
            Generated text, or empty string on error
        """
        result = self.generate_sync(
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            conversation_id=conversation_id,
            keep_context=keep_context,
        )

        if result.error:
            self.logger.error(f"Generation error: {result.error}")

        return result.response

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
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5.0
            )
            if response.status_code == 200:
                self._healthy = True
                self._last_health_check = time.time()
                return response.json()
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
    model: str = "qwen-4b",
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