"""Main navigation controller for VLN system."""

from typing import Optional, Dict, Any
import logging

from .context import NavContext, NavContextBuilder, TaskType
from .action import Action, ActionType


class VLNNavigator:
    """
    Main navigation controller.

    This is the entry point for the multi-agent VLN navigation system.
    It orchestrates task classification, agent selection, strategy execution,
    and fallback handling.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_fallback: bool = True,
        log_level: str = "INFO",
    ):
        """
        Initialize the navigator.

        Args:
            config: Configuration dictionary
            enable_fallback: Whether to enable fallback handling
            log_level: Logging level
        """
        self.config = config or {}
        self.enable_fallback = enable_fallback

        # Setup logging
        self.logger = logging.getLogger("VLNNavigator")
        self.logger.setLevel(getattr(logging, log_level))

        # Lazy-loaded components
        self._task_classifier = None
        self._supernet = None
        self._failure_handler = None
        self._context_builder = NavContextBuilder()

        # State
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        self.logger.info("Initializing VLNNavigator...")

        # Import here to avoid circular imports
        from classifiers import TaskTypeClassifier
        from supernet import Supernet
        from fallback import FailureHandler

        self._task_classifier = TaskTypeClassifier(self.config.get("classifier", {}))
        self._supernet = Supernet(self.config.get("supernet", {}))

        if self.enable_fallback:
            self._failure_handler = FailureHandler(self.config.get("fallback", {}))

        self._initialized = True
        self.logger.info("VLNNavigator initialized successfully")

    def reset(self) -> None:
        """Reset navigator state for new episode."""
        self._context_builder = NavContextBuilder()

    def set_instruction(self, instruction: str) -> "VLNNavigator":
        """Set navigation instruction."""
        self._context_builder.with_instruction(instruction)
        return self

    def set_position(
        self, position: tuple, rotation: float = 0.0
    ) -> "VLNNavigator":
        """Set current position and rotation."""
        self._context_builder.with_position(position).with_rotation(rotation)
        return self

    def set_visual_features(self, features: Any) -> "VLNNavigator":
        """Set visual features from observation."""
        from .context import VisualFeatures
        if isinstance(features, VisualFeatures):
            self._context_builder.with_visual_features(features)
        else:
            # Assume it's raw observation, create VisualFeatures
            vf = VisualFeatures(rgb_embedding=features)
            self._context_builder.with_visual_features(vf)
        return self

    def navigate(self, context: Optional[NavContext] = None) -> Action:
        """
        Main navigation method.

        Args:
            context: Optional pre-built context. If None, uses builder.

        Returns:
            Action to execute
        """
        if not self._initialized:
            self.initialize()

        # Build context if not provided
        if context is None:
            context = self._context_builder.build()

        try:
            # Step 1: Classify task type
            context = self._classify_task(context)

            # Step 2: Execute through supernet
            action = self._execute_supernet(context)

            # Step 3: Update context
            context.current_action = action
            context.add_action(action)

            return action

        except Exception as e:
            self.logger.error(f"Navigation error: {e}")
            if self.enable_fallback and self._failure_handler:
                return self._failure_handler.handle(context, str(e))
            return Action.stop(confidence=0.0, reasoning=f"Error: {str(e)}")

    def _classify_task(self, context: NavContext) -> NavContext:
        """Classify task type and update context."""
        task_type = self._task_classifier.classify(context)
        context.task_type = task_type
        self.logger.debug(f"Task classified as: {task_type.value}")
        return context

    def _execute_supernet(self, context: NavContext) -> Action:
        """Execute navigation through supernet."""
        return self._supernet.forward(context)

    def get_context_for_step(self) -> NavContext:
        """Get current context for the step."""
        return self._context_builder.build()

    @property
    def task_classifier(self):
        """Get task classifier instance."""
        return self._task_classifier

    @property
    def supernet(self):
        """Get supernet instance."""
        return self._supernet

    @property
    def failure_handler(self):
        """Get failure handler instance."""
        return self._failure_handler


class VLNNavigatorBuilder:
    """Builder for VLNNavigator instances."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._enable_fallback: bool = True
        self._log_level: str = "INFO"

    def with_config(self, config: Dict[str, Any]) -> "VLNNavigatorBuilder":
        """Set configuration."""
        self._config.update(config)
        return self

    def with_fallback(self, enable: bool) -> "VLNNavigatorBuilder":
        """Enable/disable fallback."""
        self._enable_fallback = enable
        return self

    def with_log_level(self, level: str) -> "VLNNavigatorBuilder":
        """Set log level."""
        self._log_level = level
        return self

    def build(self) -> VLNNavigator:
        """Build navigator instance."""
        navigator = VLNNavigator(
            config=self._config,
            enable_fallback=self._enable_fallback,
            log_level=self._log_level,
        )
        return navigator