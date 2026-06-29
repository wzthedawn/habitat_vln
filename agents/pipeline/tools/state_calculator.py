"""State calculator for position, distance, and angle computations."""

import math
from typing import Dict, List, Union


class StateCalculator:
    """Calculator for position changes, distances, and angles."""

    def compute_position_change(
        self,
        start_pos: List[float],
        current_pos: List[float]
    ) -> Dict[str, float]:
        """Calculate position change between two points.

        Args:
            start_pos: Starting position [x, y, z]
            current_pos: Current position [x, y, z]

        Returns:
            Dictionary with dx, dy, dz, and horizontal_dist
        """
        dx = current_pos[0] - start_pos[0]
        dy = current_pos[1] - start_pos[1]
        dz = current_pos[2] - start_pos[2]
        horizontal_dist = math.sqrt(dx * dx + dz * dz)

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "horizontal_dist": horizontal_dist
        }

    def compute_rotation_change(
        self,
        start_rot: float,
        current_rot: float
    ) -> float:
        """Calculate rotation change in degrees.

        Handles -180/180 degree boundary crossing.
        Positive value = left turn, negative value = right turn.

        Args:
            start_rot: Starting rotation in radians
            current_rot: Current rotation in radians

        Returns:
            Rotation change in degrees
        """
        # Convert to degrees
        start_deg = math.degrees(start_rot)
        current_deg = math.degrees(current_rot)

        # Calculate delta
        delta = current_deg - start_deg

        # Normalize to -180 to 180 range
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360

        return delta

    def compute_distance(
        self,
        position1: List[float],
        position2: List[float]
    ) -> float:
        """Calculate Euclidean distance between two points.

        Args:
            position1: First position [x, y, z]
            position2: Second position [x, y, z]

        Returns:
            Euclidean distance
        """
        dx = position2[0] - position1[0]
        dy = position2[1] - position1[1]
        dz = position2[2] - position1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def is_y_down(self, dy: float, threshold: float) -> bool:
        """Check if Y change indicates going down (e.g., downstairs).

        Args:
            dy: Y-axis change
            threshold: Minimum threshold for significant change

        Returns:
            True if dy <= -threshold (going down)
        """
        return dy <= -threshold

    def is_y_up(self, dy: float, threshold: float) -> bool:
        """Check if Y change indicates going up (e.g., upstairs).

        Args:
            dy: Y-axis change
            threshold: Minimum threshold for significant change

        Returns:
            True if dy >= threshold (going up)
        """
        return dy >= threshold

    def compute_distance_to_goal(
        self,
        current_pos: List[float],
        goal_pos: List[float]
    ) -> float:
        """Calculate distance from current position to goal.

        Args:
            current_pos: Current position [x, y, z]
            goal_pos: Goal position [x, y, z]

        Returns:
            Euclidean distance to goal
        """
        return self.compute_distance(current_pos, goal_pos)