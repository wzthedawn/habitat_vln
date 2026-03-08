"""Action definitions for VLN navigation system."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class ActionType(Enum):
    """Navigation action types."""
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


@dataclass
class Action:
    """Represents a navigation action with metadata."""

    action_type: ActionType
    confidence: float = 1.0
    reasoning: Optional[str] = None
    predicted_position: Optional[Tuple[float, float, float]] = None
    predicted_rotation: Optional[float] = None

    def __str__(self) -> str:
        return f"Action({self.action_type.name}, conf={self.confidence:.2f})"

    def to_habitat_action(self) -> str:
        """Convert to Habitat-compatible action string."""
        action_map = {
            ActionType.STOP: "stop",
            ActionType.MOVE_FORWARD: "move_forward",
            ActionType.TURN_LEFT: "turn_left",
            ActionType.TURN_RIGHT: "turn_right",
            ActionType.LOOK_UP: "look_up",
            ActionType.LOOK_DOWN: "look_down",
        }
        return action_map[self.action_type]

    @classmethod
    def stop(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a stop action."""
        return cls(ActionType.STOP, confidence=confidence, reasoning=reasoning)

    @classmethod
    def forward(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a forward action."""
        return cls(ActionType.MOVE_FORWARD, confidence=confidence, reasoning=reasoning)

    @classmethod
    def turn_left(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a turn left action."""
        return cls(ActionType.TURN_LEFT, confidence=confidence, reasoning=reasoning)

    @classmethod
    def turn_right(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a turn right action."""
        return cls(ActionType.TURN_RIGHT, confidence=confidence, reasoning=reasoning)

    @classmethod
    def look_up(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a look up action."""
        return cls(ActionType.LOOK_UP, confidence=confidence, reasoning=reasoning)

    @classmethod
    def look_down(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a look down action."""
        return cls(ActionType.LOOK_DOWN, confidence=confidence, reasoning=reasoning)