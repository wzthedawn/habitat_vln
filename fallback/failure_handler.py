"""Failure handler for managing navigation failures."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

from core.context import NavContext
from core.action import Action


@dataclass
class FailureInfo:
    """Information about a navigation failure."""
    error_type: str
    message: str
    step: int
    context_snapshot: Dict[str, Any]
    attempted_actions: List[str]


class FailureHandler:
    """
    Failure Handler for managing navigation failures.

    Handles:
    - Error detection and classification
    - Failure logging and analysis
    - Graceful degradation
    - Recovery initiation
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize failure handler.

        Args:
            config: Handler configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("FailureHandler")

        # Settings
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 0.1)

        # Failure tracking
        self._failure_history: List[FailureInfo] = []
        self._retry_count = 0

        # Cascading fallback
        self._cascading_fallback = None

    def handle(
        self,
        context: NavContext,
        error_message: str,
        error_type: str = "unknown",
    ) -> Action:
        """
        Handle a navigation failure.

        Args:
            context: Navigation context
            error_message: Error description
            error_type: Type of error

        Returns:
            Recovery action
        """
        self.logger.warning(f"Handling failure: {error_type} - {error_message}")

        # Create failure info
        failure_info = FailureInfo(
            error_type=error_type,
            message=error_message,
            step=context.step_count,
            context_snapshot=self._snapshot_context(context),
            attempted_actions=[a.action_type.name for a in context.action_history[-5:]],
        )

        # Record failure
        self._failure_history.append(failure_info)

        # Determine recovery strategy
        action = self._determine_recovery_action(context, failure_info)

        return action

    def _determine_recovery_action(
        self,
        context: NavContext,
        failure_info: FailureInfo,
    ) -> Action:
        """
        Determine the best recovery action.

        Args:
            context: Navigation context
            failure_info: Failure information

        Returns:
            Recovery action
        """
        # Classify error severity
        severity = self._classify_severity(failure_info)

        if severity == "low":
            # Simple retry or alternative action
            return self._simple_recovery(context)

        elif severity == "medium":
            # Use cascading fallback
            return self._cascading_recovery(context, failure_info)

        else:
            # High severity - stop safely
            return Action.stop(
                confidence=0.1,
                reasoning=f"High severity failure: {failure_info.error_type}"
            )

    def _classify_severity(self, failure_info: FailureInfo) -> str:
        """
        Classify failure severity.

        Args:
            failure_info: Failure information

        Returns:
            Severity level (low, medium, high)
        """
        error_type = failure_info.error_type.lower()

        # High severity errors
        high_severity = ["critical", "fatal", "unrecoverable"]
        if any(e in error_type for e in high_severity):
            return "high"

        # Medium severity errors
        medium_severity = ["model_error", "agent_failure", "strategy_failure"]
        if any(e in error_type for e in medium_severity):
            return "medium"

        # Check retry count
        if self._retry_count >= self.max_retries:
            return "high"

        return "low"

    def _simple_recovery(self, context: NavContext) -> Action:
        """
        Simple recovery - try alternative action.

        Args:
            context: Navigation context

        Returns:
            Alternative action
        """
        # Get last action and try alternative
        if context.action_history:
            last_action = context.action_history[-1].action_type.name

            # Try opposite or different action
            if last_action == "MOVE_FORWARD":
                # Maybe turn instead
                return Action.turn_left(
                    confidence=0.5,
                    reasoning="Recovery: trying turn instead of forward"
                )
            elif "TURN" in last_action:
                # Try forward
                return Action.forward(
                    confidence=0.5,
                    reasoning="Recovery: trying forward after turn"
                )

        # Default: move forward
        return Action.forward(
            confidence=0.4,
            reasoning="Recovery: default forward"
        )

    def _cascading_recovery(
        self,
        context: NavContext,
        failure_info: FailureInfo,
    ) -> Action:
        """
        Use cascading fallback for recovery.

        Args:
            context: Navigation context
            failure_info: Failure information

        Returns:
            Recovery action from cascade
        """
        # Initialize cascading fallback if needed
        if self._cascading_fallback is None:
            from fallback.cascading_fallback import CascadingFallback
            self._cascading_fallback = CascadingFallback(
                self.config.get("cascading", {})
            )

        return self._cascading_fallback.handle_failure(context, failure_info)

    def _snapshot_context(self, context: NavContext) -> Dict[str, Any]:
        """Create snapshot of context for debugging."""
        return {
            "instruction": context.instruction[:100],
            "step": context.step_count,
            "position": context.position,
            "room_type": context.room_type,
            "task_type": context.task_type.value,
        }

    def get_failure_history(self) -> List[FailureInfo]:
        """Get list of recorded failures."""
        return self._failure_history.copy()

    def get_failure_count(self) -> int:
        """Get total failure count."""
        return len(self._failure_history)

    def reset(self) -> None:
        """Reset failure handler state."""
        self._failure_history.clear()
        self._retry_count = 0

    def should_abort(self, context: NavContext) -> bool:
        """
        Check if navigation should abort.

        Args:
            context: Navigation context

        Returns:
            True if should abort
        """
        # Check consecutive failures
        if len(self._failure_history) >= 3:
            recent = self._failure_history[-3:]
            if all(f.step >= context.step_count - 3 for f in recent):
                return True

        # Check max steps
        max_steps = self.config.get("max_steps", 500)
        if context.step_count >= max_steps:
            return True

        return False


class ErrorClassifier:
    """Classifier for navigation errors."""

    ERROR_TYPES = {
        "model_error": ["model", "inference", "timeout", "api"],
        "agent_error": ["agent", "process", "execution"],
        "strategy_error": ["strategy", "planning", "reasoning"],
        "environment_error": ["environment", "habitat", "simulation"],
        "input_error": ["input", "invalid", "missing"],
    }

    @classmethod
    def classify(cls, error_message: str) -> str:
        """
        Classify error from message.

        Args:
            error_message: Error message

        Returns:
            Error type
        """
        error_lower = error_message.lower()

        for error_type, keywords in cls.ERROR_TYPES.items():
            if any(kw in error_lower for kw in keywords):
                return error_type

        return "unknown_error"