"""Main task classifier combining rule and LLM classifiers."""

from typing import Dict, Any, Optional
import logging

from core.context import NavContext, TaskType
from .rule_classifier import RuleClassifier, ClassificationResult
from .llm_classifier import LLMClassifier, LLMClassificationResult


class TaskTypeClassifier:
    """
    Main task type classifier.

    Uses a two-stage classification approach:
    1. Fast rule-based classification for clear cases
    2. LLM-based classification for ambiguous cases
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize task classifier.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("TaskTypeClassifier")

        # Classification thresholds
        self.rule_threshold = self.config.get("rule_threshold", 0.9)
        self.use_llm_fallback = self.config.get("use_llm_fallback", True)
        self.cache_results = self.config.get("cache_results", True)

        # Initialize classifiers
        self.rule_classifier = RuleClassifier(self.config.get("rule", {}))
        self.llm_classifier = LLMClassifier(self.config.get("llm", {})) if self.use_llm_fallback else None

        # Cache
        self._cache: Dict[str, TaskType] = {}

        # Statistics
        self._stats = {
            "total_classifications": 0,
            "rule_classifications": 0,
            "llm_classifications": 0,
            "cache_hits": 0,
        }

    def classify(self, context: NavContext) -> TaskType:
        """
        Classify the navigation task type.

        Args:
            context: Navigation context containing instruction and state

        Returns:
            TaskType enum value (Type-0 to Type-4)
        """
        self._stats["total_classifications"] += 1

        # Check cache first
        cache_key = self._get_cache_key(context)
        if self.cache_results and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        # Stage 1: Rule-based classification
        rule_result = self.rule_classifier.classify(context)

        if rule_result.confidence >= self.rule_threshold:
            self._stats["rule_classifications"] += 1
            self.logger.debug(
                f"Rule classification: {rule_result.task_type.value} "
                f"(confidence: {rule_result.confidence:.2f})"
            )
            task_type = rule_result.task_type
        else:
            # Stage 2: LLM-based classification
            if self.use_llm_fallback and self.llm_classifier:
                self._stats["llm_classifications"] += 1
                llm_result = self.llm_classifier.classify(context)
                task_type = llm_result.task_type

                # Update context with LLM insights
                self._update_context_with_llm_result(context, llm_result)

                self.logger.debug(
                    f"LLM classification: {task_type.value} "
                    f"(confidence: {llm_result.confidence:.2f})"
                )
            else:
                # Fall back to rule result even with low confidence
                task_type = rule_result.task_type

        # Cache result
        if self.cache_results:
            self._cache[cache_key] = task_type

        return task_type

    def classify_with_details(self, context: NavContext) -> Dict[str, Any]:
        """
        Classify with detailed results including confidence and reasoning.

        Args:
            context: Navigation context

        Returns:
            Dictionary with detailed classification info
        """
        # Get rule classification
        rule_result = self.rule_classifier.classify(context)

        result = {
            "task_type": None,
            "confidence": 0.0,
            "method": None,
            "rule_result": {
                "task_type": rule_result.task_type.value,
                "confidence": rule_result.confidence,
                "reasoning": rule_result.reasoning,
            },
            "llm_result": None,
        }

        if rule_result.confidence >= self.rule_threshold:
            result["task_type"] = rule_result.task_type
            result["confidence"] = rule_result.confidence
            result["method"] = "rule"
        elif self.use_llm_fallback and self.llm_classifier:
            llm_result = self.llm_classifier.classify(context)
            result["task_type"] = llm_result.task_type
            result["confidence"] = llm_result.confidence
            result["method"] = "llm"
            result["llm_result"] = {
                "task_type": llm_result.task_type.value,
                "confidence": llm_result.confidence,
                "reasoning": llm_result.reasoning,
                "subtasks": llm_result.subtasks,
                "complexity_factors": llm_result.complexity_factors,
            }

        return result

    def _get_cache_key(self, context: NavContext) -> str:
        """Generate cache key from context."""
        # Use instruction and key state as cache key
        return f"{context.instruction}_{context.step_count}"

    def _update_context_with_llm_result(
        self, context: NavContext, llm_result: LLMClassificationResult
    ) -> None:
        """Update context with LLM classification insights."""
        from core.context import SubTask

        # Add subtasks if present
        if llm_result.subtasks:
            for i, subtask_desc in enumerate(llm_result.subtasks):
                subtask = SubTask(
                    id=i,
                    description=subtask_desc,
                    status="pending",
                )
                context.subtasks.append(subtask)

        # Store complexity factors in metadata
        if llm_result.complexity_factors:
            context.metadata["complexity_factors"] = llm_result.complexity_factors

    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return self._stats.copy()

    def clear_cache(self) -> None:
        """Clear classification cache."""
        self._cache.clear()

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0


class TaskTypeClassifierBuilder:
    """Builder for TaskTypeClassifier instances."""

    def __init__(self):
        self._config: Dict[str, Any] = {}

    def with_rule_threshold(self, threshold: float) -> "TaskTypeClassifierBuilder":
        """Set rule classification threshold."""
        self._config["rule_threshold"] = threshold
        return self

    def with_llm_fallback(self, enable: bool) -> "TaskTypeClassifierBuilder":
        """Enable/disable LLM fallback."""
        self._config["use_llm_fallback"] = enable
        return self

    def with_caching(self, enable: bool) -> "TaskTypeClassifierBuilder":
        """Enable/disable result caching."""
        self._config["cache_results"] = enable
        return self

    def with_llm_config(self, llm_config: Dict[str, Any]) -> "TaskTypeClassifierBuilder":
        """Set LLM configuration."""
        if "llm" not in self._config:
            self._config["llm"] = {}
        self._config["llm"].update(llm_config)
        return self

    def build(self) -> TaskTypeClassifier:
        """Build classifier instance."""
        return TaskTypeClassifier(self._config)