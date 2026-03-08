"""Logger utilities for the VLN system."""

from typing import Dict, Any, Optional
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "VLN",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Optional custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "VLN") -> logging.Logger:
    """
    Get existing logger or create default one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        return setup_logger(name)

    return logger


class NavigationLogger:
    """
    Specialized logger for navigation events.
    """

    def __init__(self, name: str = "Navigation", log_file: Optional[str] = None):
        """
        Initialize navigation logger.

        Args:
            name: Logger name
            log_file: Optional log file path
        """
        self.logger = setup_logger(name, log_file=log_file)
        self._episode_log: list = []

    def log_episode_start(
        self,
        episode_id: int,
        instruction: str,
        scene: str = "unknown",
    ) -> None:
        """Log episode start."""
        self.logger.info(f"Episode {episode_id} started in {scene}")
        self.logger.info(f"Instruction: {instruction[:100]}...")

        self._episode_log.append({
            "type": "episode_start",
            "episode_id": episode_id,
            "scene": scene,
        })

    def log_action(
        self,
        step: int,
        action: str,
        confidence: float,
        reasoning: str = "",
    ) -> None:
        """Log navigation action."""
        self.logger.debug(
            f"Step {step}: {action} (confidence: {confidence:.2f})"
        )

        self._episode_log.append({
            "type": "action",
            "step": step,
            "action": action,
            "confidence": confidence,
        })

    def log_task_classification(
        self,
        task_type: str,
        confidence: float,
        method: str,
    ) -> None:
        """Log task classification."""
        self.logger.info(
            f"Task classified as {task_type} (confidence: {confidence:.2f}, method: {method})"
        )

        self._episode_log.append({
            "type": "classification",
            "task_type": task_type,
            "confidence": confidence,
            "method": method,
        })

    def log_failure(
        self,
        step: int,
        error_type: str,
        message: str,
    ) -> None:
        """Log navigation failure."""
        self.logger.warning(f"Failure at step {step}: {error_type} - {message}")

        self._episode_log.append({
            "type": "failure",
            "step": step,
            "error_type": error_type,
            "message": message,
        })

    def log_episode_end(
        self,
        episode_id: int,
        success: bool,
        steps: int,
        distance: float = 0.0,
    ) -> None:
        """Log episode end."""
        status = "SUCCESS" if success else "FAILURE"
        self.logger.info(
            f"Episode {episode_id} ended: {status} "
            f"(steps: {steps}, distance: {distance:.2f}m)"
        )

        self._episode_log.append({
            "type": "episode_end",
            "episode_id": episode_id,
            "success": success,
            "steps": steps,
            "distance": distance,
        })

    def get_episode_log(self) -> list:
        """Get episode log."""
        return self._episode_log.copy()

    def clear_episode_log(self) -> None:
        """Clear episode log."""
        self._episode_log.clear()


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, name: str = "Performance"):
        """Initialize performance logger."""
        self.logger = get_logger(name)
        self._metrics: Dict[str, list] = {}

    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self._metrics:
            self._metrics[metric_name] = []
        self._metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> float:
        """Get average of a metric."""
        values = self._metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}

        for name, values in self._metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return summary

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()