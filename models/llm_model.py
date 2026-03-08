"""LLM model wrapper for cloud-based language models."""

from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import time

from core.context import NavContext
from utils.token_tracker import get_token_tracker
from utils.timeout_fallback import TimeoutError, timeout


class LLMModel:
    """
    LLM Model wrapper for cloud-based language models.

    Supports GPT-4, Claude, and other cloud LLMs for:
    - Complex reasoning
    - Instruction understanding
    - Multi-step planning
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LLM model.

        Args:
            config: Model configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("LLMModel")

        # Model settings
        self.model_name = self.config.get("model_name", "gpt-4")
        self.tier = self.config.get("tier", "cloud_large")
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.temperature = self.config.get("temperature", 0.7)

        # Timeout settings
        self.api_timeout = self.config.get("api_timeout", 60)  # Default 60 seconds

        # API settings
        self.api_key = self.config.get("api_key")
        self.api_base = self.config.get("api_base")

        # Client (lazy loaded)
        self._client = None
        self._initialized = False

        # Token tracking
        self._token_tracker = get_token_tracker()
        self._agent_name = "LLMModel"
        self._last_token_usage: Tuple[int, int] = (0, 0)  # (input, output)

    def initialize(self) -> None:
        """Initialize the LLM client."""
        if self._initialized:
            return

        try:
            # Support for Qwen/Dashscope (OpenAI compatible)
            if "qwen" in self.model_name.lower() or "dashscope" in str(self.api_base).lower():
                self._initialize_openai_compatible()
            elif "claude" in self.model_name.lower():
                self._initialize_anthropic()
            else:
                self._initialize_openai_compatible()

        except Exception as e:
            self.logger.warning(f"LLM initialization failed: {e}, using mock")
            self._initialized = True

    def _initialize_openai_compatible(self) -> None:
        """Initialize OpenAI-compatible client (supports OpenAI, Qwen, etc.)."""
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            self._initialized = True
            self.logger.info(f"OpenAI-compatible client initialized for {self.model_name}")

        except ImportError:
            self.logger.warning("OpenAI package not available, trying legacy init")
            self._initialize_openai_legacy()

    def _initialize_openai_legacy(self) -> None:
        """Initialize OpenAI client using legacy method."""
        try:
            import openai

            if self.api_key:
                openai.api_key = self.api_key
            if self.api_base:
                openai.api_base = self.api_base

            self._client = openai
            self._initialized = True
            self.logger.info("OpenAI client initialized (legacy)")

        except ImportError:
            self.logger.warning("OpenAI package not available")

    def _initialize_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._initialized = True
            self.logger.info("Anthropic client initialized")

        except ImportError:
            self.logger.warning("Anthropic package not available")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        self.initialize()

        if self._client is None:
            return self._mock_generate(prompt)

        try:
            return self._generate_openai_compatible(prompt, system_prompt)
        except TimeoutError as e:
            self.logger.error(f"LLM API timeout: {e}")
            return self._mock_generate(prompt)
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return self._mock_generate(prompt)

    def _generate_openai_compatible(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using OpenAI-compatible API (OpenAI, Qwen, etc.)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Check if using new OpenAI client or legacy
        if hasattr(self._client, 'chat'):
            # New OpenAI client (>= 1.0)
            try:
                start_time = time.time()
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.api_timeout,  # Add timeout parameter
                )
                elapsed = time.time() - start_time
                self.logger.debug(f"API call completed in {elapsed:.2f}s")

                # Extract token usage from response
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                # Record token usage
                self._last_token_usage = (input_tokens, output_tokens)
                self._token_tracker.record_usage(
                    agent_name=self._agent_name,
                    model_name=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    prompt_type="",
                )

                return response.choices[0].message.content

            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    raise TimeoutError(f"API call timed out after {self.api_timeout}s")
                raise
        else:
            # Legacy OpenAI client
            try:
                start_time = time.time()
                response = self._client.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    request_timeout=self.api_timeout,  # Legacy uses request_timeout
                )
                elapsed = time.time() - start_time
                self.logger.debug(f"API call completed in {elapsed:.2f}s")

                # Extract token usage from response
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

                # Record token usage
                self._last_token_usage = (input_tokens, output_tokens)
                self._token_tracker.record_usage(
                    agent_name=self._agent_name,
                    model_name=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    prompt_type="",
                )

                return response.choices[0].message.content

            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    raise TimeoutError(f"API call timed out after {self.api_timeout}s")
                raise

    def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing."""
        prompt_lower = prompt.lower()

        # Simple heuristic responses
        if "classify" in prompt_lower or "type" in prompt_lower:
            if any(word in prompt_lower for word in ["left", "right", "turn"]):
                return '{"task_type": "Type-0", "confidence": 0.8}'
            elif any(word in prompt_lower for word in ["find", "search", "look"]):
                return '{"task_type": "Type-2", "confidence": 0.85}'
            elif any(word in prompt_lower for word in ["room", "kitchen", "bedroom"]):
                return '{"task_type": "Type-3", "confidence": 0.8}'
            elif any(word in prompt_lower for word in ["if", "unless", "either"]):
                return '{"task_type": "Type-4", "confidence": 0.9}'
            else:
                return '{"task_type": "Type-1", "confidence": 0.7}'

        if "action" in prompt_lower:
            if "left" in prompt_lower:
                return "turn_left"
            elif "right" in prompt_lower:
                return "turn_right"
            elif "stop" in prompt_lower:
                return "stop"
            else:
                return "forward"

        # Default response
        return "forward"

    def predict(self, context: NavContext) -> str:
        """
        Predict action using LLM.

        Args:
            context: Navigation context

        Returns:
            Action string
        """
        # Build prompt
        prompt = self._build_navigation_prompt(context)

        # Generate response
        response = self.generate(prompt)

        # Parse action
        return self._parse_action(response)

    def _build_navigation_prompt(self, context: NavContext) -> str:
        """Build navigation prompt for LLM."""
        prompt_parts = [
            "You are a navigation assistant. Given the current state, decide the next action.",
            "",
            f"Instruction: {context.instruction}",
            f"Current step: {context.step_count}",
            f"Room: {context.room_type}",
        ]

        if context.visual_features.scene_description:
            prompt_parts.append(f"Scene: {context.visual_features.scene_description}")

        if context.action_history:
            recent = context.get_action_summary(3)
            prompt_parts.append(f"Recent actions: {recent}")

        prompt_parts.extend([
            "",
            "Available actions: forward, turn_left, turn_right, stop",
            "Respond with ONLY the action name.",
        ])

        return "\n".join(prompt_parts)

    def _parse_action(self, response: str) -> str:
        """Parse action from LLM response."""
        response_lower = response.lower().strip()

        # Direct action matches
        if "forward" in response_lower or "move_forward" in response_lower:
            return "forward"
        elif "turn_left" in response_lower or "left" in response_lower:
            return "turn_left"
        elif "turn_right" in response_lower or "right" in response_lower:
            return "turn_right"
        elif "stop" in response_lower:
            return "stop"

        # Default to forward
        return "forward"

    def generate_with_reasoning(
        self,
        prompt: str,
        reasoning_steps: int = 3,
        system_prompt: str = None,
    ) -> Dict[str, Any]:
        """
        Generate response with explicit reasoning steps.

        Args:
            prompt: User prompt
            reasoning_steps: Number of reasoning steps
            system_prompt: Optional system prompt

        Returns:
            Dictionary with reasoning and conclusion
        """
        # Build chain-of-thought prompt
        cot_prompt = f"{prompt}\n\nLet's think step by step:"

        response = self.generate(cot_prompt, system_prompt)

        return {
            "reasoning": response,
            "conclusion": self._extract_conclusion(response),
        }

    def _extract_conclusion(self, response: str) -> str:
        """Extract conclusion from reasoning response."""
        lines = response.strip().split("\n")

        # Look for conclusion markers
        for line in reversed(lines):
            line = line.strip().lower()
            if any(marker in line for marker in ["therefore", "thus", "so", "conclusion", "answer"]):
                return line

        # Return last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return response

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts

        Returns:
            List of responses
        """
        return [self.generate(prompt) for prompt in prompts]

    def set_agent_name(self, name: str) -> None:
        """
        Set the agent name for token tracking.

        Args:
            name: Agent name
        """
        self._agent_name = name

    def get_last_token_usage(self) -> Tuple[int, int]:
        """
        Get the token usage from the last API call.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        return self._last_token_usage

    def generate_with_usage(
        self, prompt: str, system_prompt: str = None
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Generate response and return token usage.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Tuple of (response, (input_tokens, output_tokens))
        """
        response = self.generate(prompt, system_prompt)
        return response, self._last_token_usage