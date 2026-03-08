"""Timeout handling and fallback mechanisms for VLN system."""

import signal
import logging
import threading
from functools import wraps
from typing import Callable, Any, Optional, TypeVar
from contextlib import contextmanager

logger = logging.getLogger("TimeoutFallback")

T = TypeVar('T')


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


class TimeoutContext:
    """Thread-safe timeout context manager."""

    def __init__(self, seconds: int, error_message: str = "Operation timed out"):
        self.seconds = seconds
        self.error_message = error_message
        self._timer = None
        self._cancelled = False

    def _timeout_handler(self):
        if not self._cancelled:
            raise TimeoutError(self.error_message)

    def __enter__(self):
        # Use signal-based timeout for main thread
        # For subprocesses or threads, use threading timer
        if threading.current_thread() is threading.main_thread():
            def signal_handler(signum, frame):
                raise TimeoutError(self.error_message)

            self._old_handler = signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.seconds)
        else:
            # Threading-based timeout for non-main threads
            self._timer = threading.Timer(self.seconds, self._timeout_handler)
            self._timer.daemon = True
            self._timer.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cancelled = True

        if threading.current_thread() is threading.main_thread():
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)
        elif self._timer is not None:
            self._timer.cancel()

        return False  # Don't suppress exceptions


@contextmanager
def timeout(seconds: int, error_message: str = "Operation timed out"):
    """
    Context manager for timeout.

    Usage:
        try:
            with timeout(60, "Scene loading timeout"):
                load_scene()
        except TimeoutError:
            handle_timeout()

    Args:
        seconds: Timeout duration in seconds
        error_message: Error message for timeout exception
    """
    def signal_handler(signum, frame):
        raise TimeoutError(error_message)

    # Only works in main thread
    if threading.current_thread() is not threading.main_thread():
        logger.warning("timeout() context manager only works in main thread, skipping")
        yield
        return

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def with_timeout(seconds: int, fallback: Optional[Callable[..., T]] = None):
    """
    Decorator for timeout with optional fallback.

    Usage:
        @with_timeout(60, fallback=lambda: "default_result")
        def slow_function():
            ...

    Args:
        seconds: Timeout duration in seconds
        fallback: Optional fallback function to call on timeout

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if threading.current_thread() is not threading.main_thread():
                # Can't use signal in non-main threads
                logger.warning(f"with_timeout only works in main thread, running {func.__name__} without timeout")
                return func(*args, **kwargs)

            def handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            except TimeoutError as e:
                logger.warning(str(e))
                if fallback is not None:
                    logger.info(f"Using fallback for {func.__name__}")
                    return fallback(*args, **kwargs)
                raise
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper

    return decorator


def timeout_wrapper(
    func: Callable[..., T],
    seconds: int,
    fallback: Optional[Callable[..., T]] = None,
    *args,
    **kwargs
) -> T:
    """
    Wrap a function call with timeout.

    Usage:
        result = timeout_wrapper(load_scene, 60, fallback_scene, scene_id)

    Args:
        func: Function to call
        seconds: Timeout duration in seconds
        fallback: Optional fallback function
        *args: Arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Function result or fallback result
    """
    if threading.current_thread() is not threading.main_thread():
        logger.warning("timeout_wrapper only works in main thread, running without timeout")
        return func(*args, **kwargs)

    def handler(signum, frame):
        raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)

    try:
        return func(*args, **kwargs)
    except TimeoutError as e:
        logger.warning(str(e))
        if fallback is not None:
            logger.info(f"Using fallback for {func.__name__}")
            return fallback(*args, **kwargs)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class StepTimeout:
    """
    Step-level timeout tracker for navigation loops.

    Tracks time per step and provides timeout detection.
    """

    def __init__(self, step_timeout: int = 30, episode_timeout: int = 300):
        """
        Initialize step timeout tracker.

        Args:
            step_timeout: Maximum seconds per step
            episode_timeout: Maximum seconds per episode
        """
        self.step_timeout = step_timeout
        self.episode_timeout = episode_timeout
        self._episode_start: Optional[float] = None
        self._step_start: Optional[float] = None
        self._step_count = 0

    def start_episode(self) -> None:
        """Start episode timer."""
        import time
        self._episode_start = time.time()
        self._step_count = 0

    def start_step(self) -> None:
        """Start step timer."""
        import time
        self._step_start = time.time()
        self._step_count += 1

    def check_step_timeout(self) -> bool:
        """Check if current step has timed out."""
        if self._step_start is None:
            return False
        import time
        return (time.time() - self._step_start) > self.step_timeout

    def check_episode_timeout(self) -> bool:
        """Check if episode has timed out."""
        if self._episode_start is None:
            return False
        import time
        return (time.time() - self._episode_start) > self.episode_timeout

    def get_remaining_step_time(self) -> float:
        """Get remaining time for current step."""
        if self._step_start is None:
            return float(self.step_timeout)
        import time
        elapsed = time.time() - self._step_start
        return max(0, self.step_timeout - elapsed)

    def get_remaining_episode_time(self) -> float:
        """Get remaining time for episode."""
        if self._episode_start is None:
            return float(self.episode_timeout)
        import time
        elapsed = time.time() - self._episode_start
        return max(0, self.episode_timeout - elapsed)

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count


# Default timeout configuration
DEFAULT_TIMEOUTS = {
    "scene_load": 60,      # Scene loading timeout (seconds)
    "api_call": 60,        # LLM API call timeout (seconds)
    "step": 30,            # Navigation step timeout (seconds)
    "episode": 300,        # Episode total timeout (seconds)
    "navigation": 180,     # Navigation loop timeout (seconds)
}