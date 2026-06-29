"""SubAgent base class and output data structures for pipeline architecture.

This module defines:
1. Output data structures (ObservationOutput, AnalysisOutput, PlanningOutput, ReviewOutput, EmergencyEvent, DecompositionOutput)
2. SubAgent base class that all pipeline agents inherit from
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent


@dataclass
class ObservationOutput:
    """Observation Agent output - visual scene analysis result.

    Contains information about what the agent observes in the environment,
    including subtask-relevant objects, instruction-related objects, and navigation cues.
    """

    # Boolean flags
    subtask_relevant: bool  # Whether subtask target objects are visible (renamed from task_relevant)
    instruction_relevant: bool  # Whether original instruction objects are visible (fallback)
    fallback_mode: bool  # Whether currently in fallback mode
    path_blocked: bool  # Whether forward path is blocked

    # String fields
    exploration_hint: str  # Exploration direction suggestion
    target_direction: str  # Target direction
    target_distance: str  # Target distance
    scene_description: str = ""  # Scene description text

    # Complex types (with defaults due to field ordering)
    objects: List[Dict[str, Any]] = field(default_factory=list)  # List of objects with features
    navigation_cues: List[str] = field(default_factory=list)  # Navigation cues for decision making

    # Compatibility alias for legacy code
    @property
    def task_relevant(self) -> bool:
        """Alias for backward compatibility."""
        return self.subtask_relevant


@dataclass
class AnalysisOutput:
    """Analysis Agent output - goal and situation analysis.

    Contains the agent's analysis of the current navigation goal,
    the gap between current state and target, and recommended action.
    """

    goal_summary: str  # Goal summary in one sentence
    current_gap: str  # Current gap analysis (what's missing)
    recommended_action: str  # Recommended action to take
    reasoning: str  # Reasoning process explanation
    confidence: float  # Confidence level (0-1)
    strategy_used: str  # Strategy used for analysis (cot/react/debate)


@dataclass
class PlanningOutput:
    """Planning Agent output - action sequence planning.

    Contains the planned action sequence and expected result,
    with optional path information for topology-based planning.
    """

    actions: List[str]  # Action sequence (typically 5 actions)
    expected_result: str  # Expected result after executing actions
    algorithm_used: str  # Algorithm used (llm/topology/hybrid)
    path: Optional[List] = None  # Path nodes (for topology planning)


@dataclass
class ReviewOutput:
    """Review Agent output - progress review and completion check.

    Contains review results including completion status, progress,
    and current value vs threshold comparison.
    """

    completed: bool  # Whether the task is completed
    reason: str  # Review conclusion/reasoning
    progress: float  # Progress ratio (0-1)
    current_value: float  # Current metric value
    threshold: float  # Threshold for completion


@dataclass
class EmergencyEvent:
    """Emergency event data structure.

    Contains information about detected emergency situations
    like obstacles or evacuation needs.
    """

    type: str  # Event type (obstacle/evacuate)
    severity: str  # Severity level (high/medium/low)
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed information


@dataclass
class DecompositionOutput:
    """Subtask Decomposition Agent output - instruction decomposition result.

    Contains the list of decomposed subtasks, reasoning, and static difficulty assessment.
    """

    subtasks: List[Dict[str, Any]]  # List of subtask dicts
    reasoning: str = ""  # Decomposition reasoning explanation
    static_difficulty: str = "medium"  # Static difficulty: easy/medium/hard
    difficulty_factors: Dict[str, Any] = field(default_factory=dict)  # Difficulty factors


class SubAgent(BaseAgent):
    """SubAgent base class for pipeline agents.

    All pipeline agents (ObservationAgent, AnalysisAgent, PlanningAgent, etc.)
    inherit from this base class. Each SubAgent must use LLM for core decision making.

    Key features:
    - Inherits from BaseAgent for basic agent functionality
    - Provides _call_llm and _call_vlm methods for model access
    - Requires ModelManager to be set before processing
    - Abstract process() method must be implemented by subclasses
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SubAgent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self._model_manager = None

    def set_model_manager(self, model_manager) -> None:
        """Set ModelManager for LLM/VLM access.

        Args:
            model_manager: ModelManager instance for generating text/vision
        """
        self._model_manager = model_manager

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process method - each SubAgent implements specific processing logic.

        Args:
            *args: Positional arguments specific to agent
            **kwargs: Keyword arguments specific to agent

        Returns:
            Agent-specific output (one of the Output dataclasses or dict)
        """
        pass

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.3,
        model_key: str = None,
        **kwargs,
    ) -> str:
        """Call LLM for text generation.

        Supports multi-model allocation: each SubAgent can specify its
        model_key via config or override it per call.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model_key: Optional model key override (falls back to config["model_key"],
                       then to "qwen3.5-9b-fast")
            **kwargs: Additional generation parameters

        Returns:
            Generated text response

        Raises:
            RuntimeError: If ModelManager is not set
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager not set")

        # Resolve model_key: call override > config > default
        resolved_key = model_key or self.config.get("model_key", "qwen3.5-9b-fast")

        return self._model_manager.generate_sync(
            model_key=resolved_key,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def _call_vlm(
        self,
        prompt: str,
        images: List,
        max_tokens: int = 300,
        temperature: float = 0.2,
        **kwargs,
    ) -> Dict[str, Any]:
        """Call VLM for vision-language generation.

        Args:
            prompt: Input prompt text
            images: List of images (PIL Image or numpy array)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with response, objects, scene_description, etc.

        Raises:
            RuntimeError: If ModelManager is not set
        """
        if self._model_manager is None:
            raise RuntimeError("ModelManager not set")

        # TEMPORARY: Use single RGB image only (skip depth) to fix VLM recognition
        if len(images) >= 1:
            return self._model_manager.generate_vision(
                image=images[0],
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        elif len(images) == 2 and hasattr(self._model_manager, "generate_vision_dual"):
            return self._model_manager.generate_vision_dual(
                rgb_image=images[0],
                depth_image=images[1],
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
        else:
            # Fallback to single image processing
            return self._model_manager.generate_vision(
                image=images[0],
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )