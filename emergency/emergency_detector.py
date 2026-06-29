"""Emergency Detector for navigation scenarios.

This module monitors environment changes and triggers emergency responses.

Key features:
1. Detect sudden obstacles
2. Monitor perception changes
3. Track stuck situations
4. Trigger emergency replanning

Runs on every navigation step.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import time
import math


@dataclass
class EmergencyEvent:
    """Represents an emergency event."""
    event_type: str                    # Type: obstacle_blocked, stuck, path_invalid
    position: Tuple[float, float, float]  # Event location
    timestamp: float                   # When it occurred
    severity: float                    # 0.0-1.0
    description: str = ""              # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergencyDetector:
    """Monitors environment and triggers emergency responses.

    This class:
    - Monitors perception output changes
    - Tracks obstacle state updates
    - Detects stuck situations
    - Triggers emergency mode for DecisionAgent

    Usage:
        detector = EmergencyDetector()
        detector.update(perception_output, obstacle_state, current_pos)

        if detector.should_trigger_emergency():
            event = detector.get_emergency_event()
            # Switch to emergency mode
    """

    # Emergency types
    EMERGENCY_OBSTACLE_BLOCKED = "obstacle_blocked"
    EMERGENCY_STUCK = "stuck"
    EMERGENCY_PATH_INVALID = "path_invalid"
    EMERGENCY_UNEXPECTED_CHANGE = "unexpected_change"

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the emergency detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("EmergencyDetector")

        # Configuration
        self._stuck_threshold = self.config.get("stuck_threshold", 5)  # steps
        self._position_change_threshold = self.config.get("position_change_threshold", 0.3)  # meters
        self._obstacle_distance_threshold = self.config.get("obstacle_distance_threshold", 2.0)  # meters
        self._min_severity = self.config.get("min_severity_to_trigger", 0.5)

        # State tracking
        self._perception_history: List[Dict] = []
        self._position_history: List[Tuple[float, float, float]] = []
        self._last_obstacle_state: Dict = {}
        self._current_position: Tuple[float, float, float] = (0, 0, 0)
        self._last_emergency: Optional[EmergencyEvent] = None
        self._consecutive_stuck_count = 0

        # Emergency state
        self._emergency_active = False
        self._current_event: Optional[EmergencyEvent] = None

    def update(
        self,
        perception_output: Dict[str, Any],
        obstacle_state: Dict[str, Any],
        current_position: Tuple[float, float, float],
        stuck_counter: int = 0,
        trajectory_output: Dict[str, Any] = None
    ) -> Optional[EmergencyEvent]:
        """Update detector state and check for emergencies.

        Args:
            perception_output: Latest perception agent output
            obstacle_state: Current obstacle state from DynamicObstacleManager
            current_position: Current agent position
            stuck_counter: Current stuck counter value
            trajectory_output: Optional trajectory agent output

        Returns:
            EmergencyEvent if emergency detected, None otherwise
        """
        # Store state
        self._current_position = current_position
        self._perception_history.append(perception_output)
        self._position_history.append(current_position)

        # Keep limited history
        max_history = 10
        if len(self._perception_history) > max_history:
            self._perception_history = self._perception_history[-max_history:]
        if len(self._position_history) > max_history:
            self._position_history = self._position_history[-max_history:]

        # Update stuck tracking
        if stuck_counter > 0:
            self._consecutive_stuck_count += 1
        else:
            self._consecutive_stuck_count = 0

        # Check for various emergency conditions
        event = None

        # 1. Check for obstacle blocking
        event = self._check_obstacle_blocking(obstacle_state)
        if event:
            return self._trigger_emergency(event)

        # 2. Check for stuck situation
        event = self._check_stuck_situation(stuck_counter, trajectory_output)
        if event:
            return self._trigger_emergency(event)

        # 3. Check for unexpected environment change
        event = self._check_environment_change(perception_output)
        if event:
            return self._trigger_emergency(event)

        # No emergency
        self._emergency_active = False
        self._current_event = None

        return None

    def _check_obstacle_blocking(self, obstacle_state: Dict[str, Any]) -> Optional[EmergencyEvent]:
        """Check if obstacle is blocking current path.

        Args:
            obstacle_state: Current obstacle state

        Returns:
            EmergencyEvent if obstacle is blocking, None otherwise
        """
        if not obstacle_state.get("has_obstacles", False):
            return None

        obstacles = obstacle_state.get("obstacles", [])
        if not obstacles:
            return None

        # Check if any obstacle is close to current position
        for obs in obstacles:
            obs_pos = obs.get("position", (0, 0, 0))
            distance = self._calculate_distance(self._current_position, obs_pos)

            if distance < self._obstacle_distance_threshold:
                return EmergencyEvent(
                    event_type=self.EMERGENCY_OBSTACLE_BLOCKED,
                    position=obs_pos,
                    timestamp=time.time(),
                    severity=0.8,
                    description=f"Obstacle detected at {obs_pos}, distance: {distance:.1f}m",
                    metadata={"obstacle": obs, "distance": distance}
                )

        # Check if obstacle state changed (new obstacle appeared)
        if self._last_obstacle_state:
            old_count = self._last_obstacle_state.get("obstacle_count", 0)
            new_count = obstacle_state.get("obstacle_count", 0)

            if new_count > old_count:
                # New obstacle appeared
                new_obstacles = obstacles[old_count:]  # Get new obstacles
                if new_obstacles:
                    latest = new_obstacles[-1]
                    return EmergencyEvent(
                        event_type=self.EMERGENCY_OBSTACLE_BLOCKED,
                        position=latest.get("position", (0, 0, 0)),
                        timestamp=time.time(),
                        severity=0.7,
                        description=f"New obstacle appeared: {latest.get('description', 'unknown')}",
                        metadata={"obstacle": latest}
                    )

        self._last_obstacle_state = obstacle_state
        return None

    def _check_stuck_situation(
        self,
        stuck_counter: int,
        trajectory_output: Dict[str, Any] = None
    ) -> Optional[EmergencyEvent]:
        """Check if agent is stuck.

        Args:
            stuck_counter: Current stuck counter
            trajectory_output: Trajectory agent output

        Returns:
            EmergencyEvent if stuck, None otherwise
        """
        if stuck_counter >= self._stuck_threshold:
            return EmergencyEvent(
                event_type=self.EMERGENCY_STUCK,
                position=self._current_position,
                timestamp=time.time(),
                severity=0.6,
                description=f"Agent stuck for {stuck_counter} steps",
                metadata={"stuck_counter": stuck_counter}
            )

        # Also check position history for lack of movement
        if len(self._position_history) >= 5:
            recent = self._position_history[-5:]
            total_movement = 0.0

            for i in range(1, len(recent)):
                total_movement += self._calculate_distance(recent[i-1], recent[i])

            avg_movement = total_movement / (len(recent) - 1)

            if avg_movement < self._position_change_threshold:
                self._consecutive_stuck_count += 1

                if self._consecutive_stuck_count >= 3:
                    return EmergencyEvent(
                        event_type=self.EMERGENCY_STUCK,
                        position=self._current_position,
                        timestamp=time.time(),
                        severity=0.5,
                        description=f"Minimal movement detected: {avg_movement:.2f}m avg",
                        metadata={"avg_movement": avg_movement}
                    )
            else:
                self._consecutive_stuck_count = 0

        return None

    def _check_environment_change(
        self,
        perception_output: Dict[str, Any]
    ) -> Optional[EmergencyEvent]:
        """Check for unexpected environment changes.

        Args:
            perception_output: Latest perception output

        Returns:
            EmergencyEvent if significant change detected, None otherwise
        """
        if len(self._perception_history) < 2:
            return None

        prev = self._perception_history[-2]
        curr = perception_output

        # Check for room type change (indicates transition to new area)
        prev_room = prev.get("room_type", "unknown")
        curr_room = curr.get("room_type", "unknown")

        # Room change is not necessarily emergency, but worth noting
        if prev_room != curr_room and prev_room != "unknown":
            self.logger.info(f"Room transition: {prev_room} -> {curr_room}")

        # Check for sudden obstacle in nav_hint
        curr_hint = curr.get("nav_hint", "").lower()
        if any(keyword in curr_hint for keyword in ["blocked", "obstacle", "cannot", "impassable"]):
            return EmergencyEvent(
                event_type=self.EMERGENCY_UNEXPECTED_CHANGE,
                position=self._current_position,
                timestamp=time.time(),
                severity=0.6,
                description=f"Obstacle mentioned in nav_hint",
                metadata={"nav_hint": curr_hint}
            )

        # Check for danger info
        danger_info = curr.get("danger_info", {})
        if danger_info.get("danger_detected", False):
            return EmergencyEvent(
                event_type=self.EMERGENCY_UNEXPECTED_CHANGE,
                position=self._current_position,
                timestamp=time.time(),
                severity=0.9,
                description=f"Danger detected: {danger_info.get('type', 'unknown')}",
                metadata={"danger_info": danger_info}
            )

        return None

    def _trigger_emergency(self, event: EmergencyEvent) -> EmergencyEvent:
        """Trigger emergency mode.

        Args:
            event: The emergency event

        Returns:
            The triggered event
        """
        # Only trigger if severity is above threshold
        if event.severity < self._min_severity:
            self.logger.debug(f"Emergency severity {event.severity} below threshold, ignoring")
            return None

        self._emergency_active = True
        self._current_event = event
        self._last_emergency = event

        self.logger.warning(f"[EMERGENCY] {event.event_type}: {event.description}")
        print(f"\n[EMERGENCY DETECTOR] {event.event_type.upper()}")
        print(f"  Position: {event.position}")
        print(f"  Severity: {event.severity:.1f}")
        print(f"  Description: {event.description}")

        return event

    def _calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float]
    ) -> float:
        """Calculate horizontal distance between two positions."""
        dx = pos1[0] - pos2[0]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx * dx + dz * dz)

    def is_emergency_active(self) -> bool:
        """Check if emergency mode is active."""
        return self._emergency_active

    def get_current_event(self) -> Optional[EmergencyEvent]:
        """Get the current emergency event."""
        return self._current_event

    def get_last_emergency(self) -> Optional[EmergencyEvent]:
        """Get the last emergency event."""
        return self._last_emergency

    def clear_emergency(self) -> None:
        """Clear emergency state."""
        self._emergency_active = False
        self._current_event = None
        self.logger.info("Emergency cleared")

    def reset(self) -> None:
        """Reset detector state for new episode."""
        self._perception_history = []
        self._position_history = []
        self._last_obstacle_state = {}
        self._current_position = (0, 0, 0)
        self._last_emergency = None
        self._consecutive_stuck_count = 0
        self._emergency_active = False
        self._current_event = None

        self.logger.info("Emergency detector reset")

    def get_emergency_signal(self) -> Dict[str, Any]:
        """Get emergency signal for context metadata.

        Returns:
            Dictionary with emergency signal information
        """
        return {
            "trigger": self._emergency_active,
            "event_type": self._current_event.event_type if self._current_event else None,
            "position": self._current_event.position if self._current_event else None,
            "severity": self._current_event.severity if self._current_event else 0.0,
            "description": self._current_event.description if self._current_event else "",
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get detector summary."""
        return {
            "emergency_active": self._emergency_active,
            "current_event": self._current_event.event_type if self._current_event else None,
            "last_emergency": self._last_emergency.event_type if self._last_emergency else None,
            "perception_history_size": len(self._perception_history),
            "position_history_size": len(self._position_history),
            "consecutive_stuck_count": self._consecutive_stuck_count,
        }