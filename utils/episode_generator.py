"""Episode generator for VLN evaluation."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import random
import logging
import numpy as np


@dataclass
class VLNEpisode:
    """Represents a single VLN navigation episode."""

    episode_id: int
    scene_id: str
    start_position: Tuple[float, float, float]
    start_rotation: float  # Yaw angle in degrees
    goal_position: Tuple[float, float, float]
    instruction: str
    geodesic_distance: float = 0.0
    difficulty: str = "medium"  # easy, medium, hard
    task_type: str = "Type-1"  # Type-0 to Type-4

    # Optional fields
    reference_path: List[Tuple[float, float, float]] = field(default_factory=list)
    landmarks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary."""
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "start_position": self.start_position,
            "start_rotation": self.start_rotation,
            "goal_position": self.goal_position,
            "instruction": self.instruction,
            "geodesic_distance": self.geodesic_distance,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "reference_path": self.reference_path,
            "landmarks": self.landmarks,
            "metadata": self.metadata,
        }


class EpisodeGenerator:
    """
    Generate VLN navigation episodes.

    Creates navigation tasks with:
    - Random start positions on navigable areas
    - Goal positions at specified distance ranges
    - Simulated navigation instructions
    """

    # Instruction templates by task type
    INSTRUCTION_TEMPLATES = {
        "Type-0": [
            "Move forward.",
            "Go straight ahead.",
            "Walk forward.",
        ],
        "Type-1": [
            "Go straight and turn {direction} at the end of the hallway.",
            "Walk forward until you see {landmark}, then turn {direction}.",
            "Follow the corridor and stop when you reach {landmark}.",
        ],
        "Type-2": [
            "Find the {landmark} in this room.",
            "Locate the {landmark} and move towards it.",
            "Search for the {landmark} and stop nearby.",
        ],
        "Type-3": [
            "Leave this room and find the {landmark} in the next room.",
            "Go through the door, walk down the hallway, and locate the {landmark}.",
            "Exit the current room, turn {direction}, and proceed to the {landmark}.",
        ],
        "Type-4": [
            "Navigate to the {landmark1}, then find the {landmark2} nearby.",
            "Go past the {landmark1}, continue until you see {landmark2}, then stop.",
            "Find the {landmark1} first, then proceed to locate {landmark2}.",
        ],
    }

    LANDMARKS = [
        "table", "chair", "door", "window", "sofa", "bed",
        "kitchen counter", "desk", "bookshelf", "fireplace",
        "stairs", "painting", "lamp", "couch", "cabinet",
    ]

    DIRECTIONS = ["left", "right"]

    def __init__(
        self,
        min_distance: float = 5.0,
        max_distance: float = 20.0,
        success_distance: float = 3.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize episode generator.

        Args:
            min_distance: Minimum geodesic distance to goal
            max_distance: Maximum geodesic distance to goal
            success_distance: Distance threshold for success
            random_seed: Random seed for reproducibility
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.success_distance = success_distance

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.logger = logging.getLogger("EpisodeGenerator")
        self._episode_counter = 0

    def generate_episodes(
        self,
        scene_path: str,
        num_episodes: int,
        task_type_distribution: Optional[Dict[str, float]] = None,
        sim=None,
    ) -> List[VLNEpisode]:
        """
        Generate navigation episodes for a scene.

        Args:
            scene_path: Path to scene file
            num_episodes: Number of episodes to generate
            task_type_distribution: Distribution of task types (e.g., {"Type-1": 0.5, ...})
            sim: Optional Habitat simulator instance

        Returns:
            List of VLNEpisode objects
        """
        episodes = []

        # Default task type distribution
        if task_type_distribution is None:
            task_type_distribution = {
                "Type-0": 0.1,
                "Type-1": 0.3,
                "Type-2": 0.25,
                "Type-3": 0.2,
                "Type-4": 0.15,
            }

        # Get navigable points
        nav_points = self._get_navigable_points(scene_path, sim)

        if not nav_points:
            self.logger.warning(f"No navigable points found for scene: {scene_path}")
            # Generate mock episodes
            return self._generate_mock_episodes(scene_path, num_episodes)

        for i in range(num_episodes):
            # Select task type based on distribution
            task_type = self._sample_task_type(task_type_distribution)

            # Generate episode
            episode = self._generate_single_episode(
                episode_id=self._episode_counter,
                scene_id=scene_path,
                nav_points=nav_points,
                task_type=task_type,
                sim=sim,
            )

            if episode:
                episodes.append(episode)
                self._episode_counter += 1

        self.logger.info(f"Generated {len(episodes)} episodes for {scene_path}")
        return episodes

    def _get_navigable_points(
        self,
        scene_path: str,
        sim,
    ) -> List[Tuple[float, float, float]]:
        """Get navigable points from scene."""
        if sim is not None:
            try:
                nav_points = sim.pathfinder.get_navigable_points()
                return [tuple(p.tolist()) for p in nav_points[:1000]]  # Limit points
            except Exception as e:
                self.logger.warning(f"Could not get navigable points: {e}")

        # Return empty list - will fall back to mock episodes
        return []

    def _generate_single_episode(
        self,
        episode_id: int,
        scene_id: str,
        nav_points: List[Tuple[float, float, float]],
        task_type: str,
        sim,
    ) -> Optional[VLNEpisode]:
        """Generate a single navigation episode."""
        if len(nav_points) < 2:
            return None

        # Random start position
        start_idx = random.randint(0, len(nav_points) - 1)
        start_position = nav_points[start_idx]

        # Random rotation (yaw)
        start_rotation = random.uniform(0, 360)

        # Find goal position within distance range
        goal_position = None
        geodesic_distance = 0.0

        # Try to find a suitable goal
        for _ in range(100):  # Max attempts
            goal_idx = random.randint(0, len(nav_points) - 1)
            if goal_idx == start_idx:
                continue

            candidate_goal = nav_points[goal_idx]

            # Calculate distance
            if sim is not None:
                dist = sim.geodesic_distance(start_position, candidate_goal)
            else:
                dist = self._euclidean_distance(start_position, candidate_goal)

            # Check if within range
            if self.min_distance <= dist <= self.max_distance:
                goal_position = candidate_goal
                geodesic_distance = dist
                break

        if goal_position is None:
            # Use a random goal if no suitable one found
            goal_idx = (start_idx + len(nav_points) // 2) % len(nav_points)
            goal_position = nav_points[goal_idx]
            geodesic_distance = self._euclidean_distance(start_position, goal_position)

        # Generate instruction
        instruction = self._generate_instruction(task_type)

        # Determine difficulty based on distance
        if geodesic_distance < 8:
            difficulty = "easy"
        elif geodesic_distance < 15:
            difficulty = "medium"
        else:
            difficulty = "hard"

        return VLNEpisode(
            episode_id=episode_id,
            scene_id=scene_id,
            start_position=start_position,
            start_rotation=start_rotation,
            goal_position=goal_position,
            instruction=instruction,
            geodesic_distance=geodesic_distance,
            difficulty=difficulty,
            task_type=task_type,
        )

    def _generate_instruction(self, task_type: str) -> str:
        """Generate a navigation instruction for the given task type."""
        templates = self.INSTRUCTION_TEMPLATES.get(task_type, self.INSTRUCTION_TEMPLATES["Type-1"])
        template = random.choice(templates)

        # Fill in template placeholders
        instruction = template.format(
            direction=random.choice(self.DIRECTIONS),
            landmark=random.choice(self.LANDMARKS),
            landmark1=random.choice(self.LANDMARKS),
            landmark2=random.choice(self.LANDMARKS),
        )

        return instruction

    def _sample_task_type(self, distribution: Dict[str, float]) -> str:
        """Sample a task type from the distribution."""
        task_types = list(distribution.keys())
        probabilities = list(distribution.values())

        return np.random.choice(task_types, p=probabilities)

    def _euclidean_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def _generate_mock_episodes(
        self,
        scene_path: str,
        num_episodes: int,
    ) -> List[VLNEpisode]:
        """Generate mock episodes when no navigable points available."""
        episodes = []

        for i in range(num_episodes):
            task_type = self._sample_task_type({
                "Type-0": 0.1,
                "Type-1": 0.3,
                "Type-2": 0.25,
                "Type-3": 0.2,
                "Type-4": 0.15,
            })

            # Mock positions
            start_position = (
                random.uniform(-5, 5),
                0.0,
                random.uniform(-5, 5),
            )
            goal_position = (
                start_position[0] + random.uniform(-10, 10),
                0.0,
                start_position[2] + random.uniform(-10, 10),
            )

            episode = VLNEpisode(
                episode_id=self._episode_counter,
                scene_id=scene_path,
                start_position=start_position,
                start_rotation=random.uniform(0, 360),
                goal_position=goal_position,
                instruction=self._generate_instruction(task_type),
                geodesic_distance=self._euclidean_distance(start_position, goal_position),
                task_type=task_type,
            )

            episodes.append(episode)
            self._episode_counter += 1

        return episodes


class EpisodeDataset:
    """Dataset for managing VLN episodes."""

    def __init__(self, episodes: Optional[List[VLNEpisode]] = None):
        """Initialize dataset with optional episodes."""
        self._episodes: List[VLNEpisode] = episodes or []

    def add_episodes(self, episodes: List[VLNEpisode]) -> None:
        """Add episodes to dataset."""
        self._episodes.extend(episodes)

    def get_episode(self, episode_id: int) -> Optional[VLNEpisode]:
        """Get episode by ID."""
        for episode in self._episodes:
            if episode.episode_id == episode_id:
                return episode
        return None

    def get_episodes_by_scene(self, scene_id: str) -> List[VLNEpisode]:
        """Get all episodes for a scene."""
        return [e for e in self._episodes if e.scene_id == scene_id]

    def get_episodes_by_task_type(self, task_type: str) -> List[VLNEpisode]:
        """Get all episodes of a task type."""
        return [e for e in self._episodes if e.task_type == task_type]

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self._episodes:
            return {"total_episodes": 0}

        # Task type distribution
        task_type_counts = {}
        for episode in self._episodes:
            task_type_counts[episode.task_type] = task_type_counts.get(episode.task_type, 0) + 1

        # Distance statistics
        distances = [e.geodesic_distance for e in self._episodes]

        return {
            "total_episodes": len(self._episodes),
            "task_type_distribution": task_type_counts,
            "avg_geodesic_distance": np.mean(distances) if distances else 0,
            "min_geodesic_distance": min(distances) if distances else 0,
            "max_geodesic_distance": max(distances) if distances else 0,
        }

    def to_json(self) -> str:
        """Convert dataset to JSON string."""
        import json
        return json.dumps({
            "episodes": [e.to_dict() for e in self._episodes],
            "statistics": self.get_statistics(),
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "EpisodeDataset":
        """Create dataset from JSON string."""
        import json
        data = json.loads(json_str)

        episodes = []
        for ep_data in data.get("episodes", []):
            episode = VLNEpisode(
                episode_id=ep_data["episode_id"],
                scene_id=ep_data["scene_id"],
                start_position=tuple(ep_data["start_position"]),
                start_rotation=ep_data["start_rotation"],
                goal_position=tuple(ep_data["goal_position"]),
                instruction=ep_data["instruction"],
                geodesic_distance=ep_data.get("geodesic_distance", 0),
                difficulty=ep_data.get("difficulty", "medium"),
                task_type=ep_data.get("task_type", "Type-1"),
            )
            episodes.append(episode)

        return cls(episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)