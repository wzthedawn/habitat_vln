"""History manager for tracking navigation history."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import json

from core.context import NavContext
from core.action import Action


@dataclass
class HistoryEntry:
    """Entry in navigation history."""
    step: int
    position: Tuple[float, float, float]
    rotation: float
    action: str
    confidence: float
    reasoning: str = ""
    room_type: str = "unknown"
    timestamp: float = 0.0


class HistoryManager:
    """
    History Manager for tracking navigation history.

    Maintains:
    - Action history
    - Trajectory history
    - Decision history
    - Performance metrics
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize history manager.

        Args:
            config: Manager configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("HistoryManager")

        # History limits
        self.max_actions = self.config.get("max_actions", 500)
        self.max_trajectory = self.config.get("max_trajectory", 500)
        self.max_decisions = self.config.get("max_decisions", 100)

        # Storage
        self._action_history: deque = deque(maxlen=self.max_actions)
        self._trajectory: deque = deque(maxlen=self.max_trajectory)
        self._decision_history: deque = deque(maxlen=self.max_decisions)

        # Episode tracking
        self._episode_start_time = 0.0
        self._episode_steps = 0

        # Statistics
        self._stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "total_distance": 0.0,
            "action_distribution": {},
        }

    def record_action(
        self,
        action: Action,
        position: Tuple[float, float, float] = None,
        rotation: float = 0.0,
        room_type: str = "unknown",
    ) -> None:
        """
        Record an action in history.

        Args:
            action: Action taken
            position: Current position
            rotation: Current rotation
            room_type: Current room type
        """
        import time

        entry = HistoryEntry(
            step=self._episode_steps,
            position=position or (0.0, 0.0, 0.0),
            rotation=rotation,
            action=action.to_habitat_action(),
            confidence=action.confidence,
            reasoning=action.reasoning or "",
            room_type=room_type,
            timestamp=time.time(),
        )

        self._action_history.append(entry)
        self._episode_steps += 1

        # Update action distribution
        action_name = action.action_type.name
        self._stats["action_distribution"][action_name] = (
            self._stats["action_distribution"].get(action_name, 0) + 1
        )

    def record_position(
        self, position: Tuple[float, float, float]
    ) -> None:
        """
        Record position in trajectory.

        Args:
            position: Current position
        """
        self._trajectory.append(position)

        # Update distance traveled
        if len(self._trajectory) >= 2:
            prev = self._trajectory[-2]
            distance = (
                (position[0] - prev[0]) ** 2 +
                (position[2] - prev[2]) ** 2
            ) ** 0.5
            self._stats["total_distance"] += distance

    def record_decision(
        self,
        decision: Dict[str, Any],
        context: NavContext = None,
    ) -> None:
        """
        Record a navigation decision.

        Args:
            decision: Decision dictionary
            context: Optional context for additional info
        """
        entry = {
            "step": self._episode_steps,
            "decision": decision,
            "timestamp": self._get_timestamp(),
        }

        if context:
            entry["context"] = {
                "instruction": context.instruction[:100],
                "room_type": context.room_type,
                "position": context.position,
            }

        self._decision_history.append(entry)

    def get_recent_actions(self, n: int = 5) -> List[HistoryEntry]:
        """
        Get recent actions.

        Args:
            n: Number of recent actions

        Returns:
            List of recent history entries
        """
        return list(self._action_history)[-n:]

    def get_recent_trajectory(self, n: int = 10) -> List[Tuple]:
        """
        Get recent trajectory points.

        Args:
            n: Number of recent points

        Returns:
            List of position tuples
        """
        return list(self._trajectory)[-n:]

    def get_action_distribution(self) -> Dict[str, int]:
        """
        Get action distribution.

        Returns:
            Dictionary mapping action names to counts
        """
        return self._stats["action_distribution"].copy()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get history summary.

        Returns:
            Summary dictionary
        """
        return {
            "total_steps": len(self._action_history),
            "trajectory_length": len(self._trajectory),
            "decisions_recorded": len(self._decision_history),
            "action_distribution": self.get_action_distribution(),
            "total_distance": self._stats["total_distance"],
        }

    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get current episode statistics.

        Returns:
            Episode statistics
        """
        import time

        duration = time.time() - self._episode_start_time if self._episode_start_time else 0

        return {
            "steps": self._episode_steps,
            "duration_seconds": duration,
            "steps_per_second": self._episode_steps / duration if duration > 0 else 0,
            "distance": self._stats["total_distance"],
            "actions": self.get_action_distribution(),
        }

    def start_episode(self) -> None:
        """Start a new episode."""
        import time

        self._episode_start_time = time.time()
        self._episode_steps = 0
        self._stats["total_episodes"] += 1

    def end_episode(self) -> Dict[str, Any]:
        """
        End current episode.

        Returns:
            Episode summary
        """
        stats = self.get_episode_stats()
        self._stats["total_steps"] += self._episode_steps
        return stats

    def clear(self) -> None:
        """Clear all history."""
        self._action_history.clear()
        self._trajectory.clear()
        self._decision_history.clear()
        self._episode_steps = 0

    def export_history(self, format: str = "json") -> str:
        """
        Export history to string format.

        Args:
            format: Export format (json, csv)

        Returns:
            Exported history string
        """
        if format == "json":
            data = {
                "actions": [
                    {
                        "step": e.step,
                        "action": e.action,
                        "position": e.position,
                        "confidence": e.confidence,
                    }
                    for e in self._action_history
                ],
                "trajectory": list(self._trajectory),
                "decisions": list(self._decision_history),
            }
            return json.dumps(data, indent=2)

        elif format == "csv":
            lines = ["step,action,position_x,position_y,position_z,confidence"]
            for e in self._action_history:
                lines.append(
                    f"{e.step},{e.action},{e.position[0]},{e.position[1]},{e.position[2]},{e.confidence}"
                )
            return "\n".join(lines)

        return ""

    def import_history(self, data: str, format: str = "json") -> None:
        """
        Import history from string.

        Args:
            data: History data string
            format: Import format
        """
        if format == "json":
            parsed = json.loads(data)

            for action_data in parsed.get("actions", []):
                entry = HistoryEntry(
                    step=action_data["step"],
                    position=tuple(action_data["position"]),
                    rotation=0.0,
                    action=action_data["action"],
                    confidence=action_data["confidence"],
                )
                self._action_history.append(entry)

            for pos in parsed.get("trajectory", []):
                self._trajectory.append(tuple(pos))

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()

    def get_backtrack_count(self, window: int = 10) -> int:
        """
        Count recent backtracking behavior.

        Args:
            window: Window size to check

        Returns:
            Number of detected backtracks
        """
        if len(self._trajectory) < 3:
            return 0

        recent = list(self._trajectory)[-window:]
        backtracks = 0

        for i in range(1, len(recent) - 1):
            prev = recent[i - 1]
            curr = recent[i]
            next_pos = recent[i + 1]

            # Check if position changed direction significantly
            v1 = (curr[0] - prev[0], curr[2] - prev[2])
            v2 = (next_pos[0] - curr[0], next_pos[2] - curr[2])

            if v1 != (0, 0) and v2 != (0, 0):
                import math
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                if dot < 0:  # Opposite direction
                    backtracks += 1

        return backtracks

    def get_efficiency_score(self) -> float:
        """
        Calculate navigation efficiency score.

        Returns:
            Efficiency score (0-1)
        """
        if len(self._trajectory) < 2:
            return 1.0

        # Direct distance
        start = self._trajectory[0]
        end = self._trajectory[-1]
        direct = (
            (end[0] - start[0]) ** 2 + (end[2] - start[2]) ** 2
        ) ** 0.5

        # Actual distance
        actual = self._stats["total_distance"]

        if actual == 0:
            return 1.0

        return min(direct / actual, 1.0)