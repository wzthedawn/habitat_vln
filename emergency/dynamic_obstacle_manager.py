"""Dynamic Obstacle Manager for emergency navigation scenarios.

This module manages dynamic obstacles that appear during navigation,
allowing simulation of emergency scenarios like blocked paths.

Key features:
1. Trigger obstacles at specific steps
2. Modify navmesh walkability
3. Track obstacle state for replanning
4. Reset obstacles for new episodes

Note: This is for experimental control, not real-time navigation.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import time
import numpy as np


@dataclass
class ObstacleConfig:
    """Configuration for a dynamic obstacle."""
    trigger_step: int                    # Step at which obstacle appears
    position: Tuple[float, float, float] # World position (x, y, z)
    radius: float = 1.0                  # Obstacle radius in meters
    obstacle_type: str = "blocked_path"  # Type: blocked_path, fallen_object, etc.
    description: str = ""                # Human-readable description
    duration: int = -1                   # Duration in steps (-1 = permanent)
    active: bool = False                 # Currently active
    activated_at_step: int = 0           # Step when activated


class DynamicObstacleManager:
    """Manages dynamic obstacles for emergency navigation experiments.

    This class handles:
    - Triggering obstacles at specified steps
    - Tracking obstacle states
    - Providing obstacle information to navigation system
    - Resetting obstacles for new episodes

    Usage:
        manager = DynamicObstacleManager(config)
        manager.add_obstacle(trigger_step=10, position=(5.0, 0.0, 3.0), radius=1.5)

        # In navigation loop:
        manager.check_and_trigger(current_step)
        obstacle_state = manager.get_obstacle_state()
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the obstacle manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger("DynamicObstacleManager")

        # Obstacle storage
        self._obstacles: List[ObstacleConfig] = []
        self._active_obstacles: List[ObstacleConfig] = []

        # State tracking
        self._current_step = 0
        self._episode_id = 0
        self._sim = None  # Habitat simulator reference

        # Configuration
        self._auto_trigger = self.config.get("auto_trigger_obstacles", True)
        self._log_events = self.config.get("log_obstacle_events", True)

    def set_simulator(self, sim) -> None:
        """Set the Habitat simulator reference for navmesh modification.

        Args:
            sim: Habitat simulator instance
        """
        self._sim = sim
        self.logger.info("Simulator reference set for obstacle manager")

    def add_obstacle(
        self,
        trigger_step: int,
        position: Tuple[float, float, float],
        radius: float = 1.0,
        obstacle_type: str = "blocked_path",
        description: str = "",
        duration: int = -1
    ) -> int:
        """Add a dynamic obstacle configuration.

        Args:
            trigger_step: Step at which obstacle appears
            position: World position (x, y, z)
            radius: Obstacle radius in meters
            obstacle_type: Type of obstacle
            description: Human-readable description
            duration: Duration in steps (-1 = permanent)

        Returns:
            Obstacle ID (index in list)
        """
        obstacle = ObstacleConfig(
            trigger_step=trigger_step,
            position=position,
            radius=radius,
            obstacle_type=obstacle_type,
            description=description,
            duration=duration,
        )
        self._obstacles.append(obstacle)
        self.logger.info(f"Added obstacle at step {trigger_step}, position {position}")
        return len(self._obstacles) - 1

    def add_obstacles_from_config(self, obstacle_configs: List[Dict]) -> None:
        """Add multiple obstacles from configuration list.

        Args:
            obstacle_configs: List of obstacle configuration dictionaries
        """
        for i, cfg in enumerate(obstacle_configs):
            self.add_obstacle(
                trigger_step=cfg.get("trigger_step", 10),
                position=tuple(cfg.get("position", [0, 0, 0])),
                radius=cfg.get("radius", 1.0),
                obstacle_type=cfg.get("type", "blocked_path"),
                description=cfg.get("description", f"Obstacle {i}"),
                duration=cfg.get("duration", -1)
            )

    def check_and_trigger(self, current_step: int) -> List[ObstacleConfig]:
        """Check for obstacles to trigger at current step.

        Args:
            current_step: Current navigation step

        Returns:
            List of newly triggered obstacles
        """
        self._current_step = current_step
        newly_triggered = []

        for obstacle in self._obstacles:
            if obstacle.trigger_step == current_step and not obstacle.active:
                self._activate_obstacle(obstacle)
                newly_triggered.append(obstacle)

        if newly_triggered and self._log_events:
            self.logger.info(f"Step {current_step}: Triggered {len(newly_triggered)} obstacle(s)")

        return newly_triggered

    def _activate_obstacle(self, obstacle: ObstacleConfig) -> bool:
        """Activate an obstacle.

        Args:
            obstacle: Obstacle to activate

        Returns:
            True if successfully activated
        """
        try:
            obstacle.active = True
            obstacle.activated_at_step = self._current_step
            self._active_obstacles.append(obstacle)

            # Try to modify navmesh if simulator is available
            if self._sim is not None:
                self._modify_navmesh(obstacle, block=True)

            if self._log_events:
                print(f"\n[OBSTACLE] Triggered at step {self._current_step}: {obstacle.description}")
                print(f"  Position: {obstacle.position}, Radius: {obstacle.radius}m")

            return True

        except Exception as e:
            self.logger.error(f"Failed to activate obstacle: {e}")
            return False

    def _modify_navmesh(self, obstacle: ObstacleConfig, block: bool = True) -> bool:
        """Modify navmesh to block/unblock obstacle area.

        Note: This is a simplified implementation. Full implementation
        would use Habitat's navmesh modification API.

        Args:
            obstacle: Obstacle configuration
            block: True to block, False to unblock

        Returns:
            True if successful
        """
        # TODO: Implement actual navmesh modification
        # This would involve:
        # 1. Getting the navmesh from simulator
        # 2. Setting the obstacle area as non-navigable
        # 3. Updating the pathfinder

        # For now, just log the action
        action = "blocked" if block else "unblocked"
        self.logger.debug(f"Navmesh {action} at {obstacle.position} (radius: {obstacle.radius}m)")

        return True

    def deactivate_obstacle(self, obstacle_id: int) -> bool:
        """Deactivate an obstacle by ID.

        Args:
            obstacle_id: Index of obstacle in list

        Returns:
            True if successfully deactivated
        """
        if obstacle_id < 0 or obstacle_id >= len(self._obstacles):
            return False

        obstacle = self._obstacles[obstacle_id]
        if not obstacle.active:
            return False

        try:
            obstacle.active = False

            # Remove from active list
            self._active_obstacles = [o for o in self._active_obstacles if o != obstacle]

            # Restore navmesh
            if self._sim is not None:
                self._modify_navmesh(obstacle, block=False)

            self.logger.info(f"Deactivated obstacle at {obstacle.position}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deactivate obstacle: {e}")
            return False

    def get_obstacle_state(self) -> Dict[str, Any]:
        """Get current obstacle state for navigation context.

        Returns:
            Dictionary with obstacle information
        """
        return {
            "has_obstacles": len(self._active_obstacles) > 0,
            "obstacle_count": len(self._active_obstacles),
            "obstacles": [
                {
                    "position": obs.position,
                    "radius": obs.radius,
                    "type": obs.obstacle_type,
                    "description": obs.description,
                    "active_for_steps": self._current_step - obs.activated_at_step,
                }
                for obs in self._active_obstacles
            ],
            "blocked_positions": [obs.position for obs in self._active_obstacles],
            "blocked_areas": [
                {
                    "center": obs.position,
                    "radius": obs.radius,
                }
                for obs in self._active_obstacles
            ],
        }

    def get_blocked_positions(self) -> List[Tuple[float, float, float]]:
        """Get list of blocked positions for path replanning.

        Returns:
            List of blocked world positions
        """
        return [obs.position for obs in self._active_obstacles]

    def is_position_blocked(self, position: Tuple[float, float, float], margin: float = 0.5) -> bool:
        """Check if a position is within any blocked area.

        Args:
            position: Position to check (x, y, z)
            margin: Additional margin to consider

        Returns:
            True if position is blocked
        """
        for obs in self._active_obstacles:
            dx = position[0] - obs.position[0]
            dz = position[2] - obs.position[2]
            distance = (dx * dx + dz * dz) ** 0.5

            if distance < obs.radius + margin:
                return True

        return False

    def get_nearest_obstacle(self, position: Tuple[float, float, float]) -> Optional[ObstacleConfig]:
        """Get the nearest active obstacle to a position.

        Args:
            position: Reference position (x, y, z)

        Returns:
            Nearest obstacle or None if no active obstacles
        """
        if not self._active_obstacles:
            return None

        nearest = None
        min_distance = float('inf')

        for obs in self._active_obstacles:
            dx = position[0] - obs.position[0]
            dz = position[2] - obs.position[2]
            distance = (dx * dx + dz * dz) ** 0.5

            if distance < min_distance:
                min_distance = distance
                nearest = obs

        return nearest

    def reset(self) -> None:
        """Reset all obstacles for a new episode."""
        for obstacle in self._obstacles:
            obstacle.active = False
            obstacle.activated_at_step = 0

        self._active_obstacles = []
        self._current_step = 0

        self.logger.info("Obstacle manager reset")

    def start_episode(self, episode_id: int, obstacle_configs: List[Dict] = None) -> None:
        """Start a new episode with optional obstacle configurations.

        Args:
            episode_id: Episode identifier
            obstacle_configs: Optional list of obstacle configurations for this episode
        """
        self.reset()
        self._episode_id = episode_id

        if obstacle_configs:
            self._obstacles = []
            self.add_obstacles_from_config(obstacle_configs)

        self.logger.info(f"Started episode {episode_id} with {len(self._obstacles)} configured obstacles")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of obstacle manager state.

        Returns:
            Summary dictionary
        """
        return {
            "episode_id": self._episode_id,
            "current_step": self._current_step,
            "total_obstacles": len(self._obstacles),
            "active_obstacles": len(self._active_obstacles),
            "obstacles": [
                {
                    "trigger_step": obs.trigger_step,
                    "position": obs.position,
                    "active": obs.active,
                }
                for obs in self._obstacles
            ],
        }

    def create_default_obstacle_config(
        self,
        scene_id: str,
        path_block_position: Tuple[float, float, float] = None
    ) -> List[Dict]:
        """Create a default obstacle configuration for testing.

        Args:
            scene_id: Scene identifier
            path_block_position: Optional position to block

        Returns:
            List of obstacle configurations
        """
        # Default: block path after some exploration
        if path_block_position is None:
            path_block_position = (5.0, 0.0, 5.0)

        return [
            {
                "trigger_step": 15,  # Block after initial navigation
                "position": list(path_block_position),
                "radius": 1.5,
                "type": "blocked_path",
                "description": "Dynamic obstacle blocking path",
                "duration": -1,
            }
        ]