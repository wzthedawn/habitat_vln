"""Evaluation metrics for VLN navigation.

Reference: Microsoft VLN Papers (Anderson et al., 2018)
Standard VLN metrics:
- SR (Success Rate): Fraction of episodes ending within success_distance of goal
- SPL (Success weighted by Path Length): SR weighted by path efficiency
- OSR (Oracle Success Rate): Fraction where any point on trajectory reaches goal
- NE (Navigation Error): Average distance to goal at episode end
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int
    success: bool = False
    spl: float = 0.0
    oracle_success: bool = False
    navigation_error: float = 0.0  # NE: distance to goal at episode end
    trajectory_length: float = 0.0
    shortest_path_length: float = 0.0
    steps: int = 0
    distance_to_goal: float = 0.0
    nDTW: float = 0.0
    SDTW: float = 0.0


class VLNMetrics:
    """
    VLN evaluation metrics calculator.

    Computes standard VLN metrics (following Anderson et al., 2018):
    - SR (Success Rate): Fraction of successful episodes
    - SPL (Success weighted by Path Length): Path efficiency weighted success
    - OSR (Oracle Success Rate): Success if any trajectory point reaches goal
    - NE (Navigation Error): Average final distance to goal

    Reference: "Vision-and-Language Navigation: Interpreting Visually-Grounded
    Navigation Instructions in Real Environments" (CVPR 2018)
    """

    def __init__(self, success_distance: float = 3.0):
        """
        Initialize metrics calculator.

        Args:
            success_distance: Distance threshold for success (default 3.0m)
        """
        self.success_distance = success_distance
        self._episodes: List[EpisodeMetrics] = []

    def add_episode(
        self,
        episode_id: int,
        trajectory: List[Tuple[float, float, float]],
        goal_position: Tuple[float, float, float],
        shortest_path: Optional[List[Tuple[float, float, float]]] = None,
    ) -> EpisodeMetrics:
        """
        Add episode and compute all metrics.

        Args:
            episode_id: Episode identifier
            trajectory: Agent's trajectory [(x,y,z), ...]
            goal_position: Goal position (x, y, z)
            shortest_path: Optional ground truth path

        Returns:
            EpisodeMetrics object
        """
        # Calculate trajectory length
        trajectory_length = self._calculate_path_length(trajectory)

        # Calculate Navigation Error (NE) - distance to goal at episode end
        if trajectory:
            final_position = trajectory[-1]
            navigation_error = self._euclidean_distance(final_position, goal_position)
        else:
            navigation_error = float('inf')

        # Success Rate (SR) - final position within threshold
        success = navigation_error <= self.success_distance

        # Calculate shortest path length
        if shortest_path:
            shortest_path_length = self._calculate_path_length(shortest_path)
        else:
            if trajectory:
                start = trajectory[0]
                shortest_path_length = self._euclidean_distance(start, goal_position)
            else:
                shortest_path_length = trajectory_length

        # SPL (Success weighted by Path Length)
        # SPL = success * (shortest_path / max(actual_path, shortest_path))
        if success and trajectory_length > 0:
            spl = shortest_path_length / max(trajectory_length, shortest_path_length)
            spl = min(1.0, spl)  # Cap at 1.0
        else:
            spl = 0.0

        # Oracle Success Rate (OSR) - any point on trajectory within threshold
        oracle_success = False
        if trajectory:
            oracle_success = any(
                self._euclidean_distance(pos, goal_position) <= self.success_distance
                for pos in trajectory
            )

        metrics = EpisodeMetrics(
            episode_id=episode_id,
            success=success,
            spl=spl,
            oracle_success=oracle_success,
            navigation_error=navigation_error,
            trajectory_length=trajectory_length,
            shortest_path_length=shortest_path_length,
            steps=len(trajectory) - 1 if trajectory else 0,
            distance_to_goal=navigation_error,
        )

        self._episodes.append(metrics)
        return metrics

    def add_episode_with_ndtw(
        self,
        episode_id: int,
        trajectory: List[Tuple[float, float, float]],
        goal_position: Tuple[float, float, float],
        reference_path: List[Tuple[float, float, float]],
    ) -> EpisodeMetrics:
        """
        Add episode with nDTW calculation.

        Args:
            episode_id: Episode identifier
            trajectory: Agent's trajectory
            goal_position: Goal position
            reference_path: Ground truth reference path

        Returns:
            EpisodeMetrics with nDTW
        """
        metrics = self.add_episode(episode_id, trajectory, goal_position, reference_path)

        # Calculate nDTW
        if trajectory and reference_path:
            ndtw = self._compute_ndtw(trajectory, reference_path)
            metrics.nDTW = ndtw

            # SDTW = nDTW if success else 0
            metrics.SDTW = ndtw if metrics.success else 0.0

        return metrics

    def _compute_ndtw(
        self,
        trajectory: List[Tuple[float, float, float]],
        reference: List[Tuple[float, float, float]],
    ) -> float:
        """
        Compute normalized Dynamic Time Warping.

        nDTW = exp(-DTW / (d * max(len(traj), len(ref))))
        where d is the average step distance in reference path.
        """
        if not trajectory or not reference:
            return 0.0

        n, m = len(trajectory), len(reference)

        # Compute DTW distance
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._euclidean_distance(trajectory[i-1], reference[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1],
                )

        dtw_distance = dtw_matrix[n, m]

        # Calculate average step distance in reference
        if len(reference) > 1:
            avg_step = np.mean([
                self._euclidean_distance(reference[i], reference[i-1])
                for i in range(1, len(reference))
            ])
        else:
            avg_step = 1.0

        # Normalize
        max_len = max(n, m)
        ndtw = math.exp(-dtw_distance / (avg_step * max_len))

        return ndtw

    def _calculate_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
        return sum(self._euclidean_distance(path[i-1], path[i]) for i in range(1, len(path)))

    def _euclidean_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregate metrics across all episodes.

        Returns standard VLN metrics:
        - SR: Success Rate
        - SPL: Success weighted by Path Length
        - OSR: Oracle Success Rate
        - NE: Navigation Error (average distance to goal)
        """
        if not self._episodes:
            return {}

        n = len(self._episodes)

        # Core VLN metrics (Microsoft standard)
        sr = sum(1 for e in self._episodes if e.success) / n  # Success Rate
        spl = sum(e.spl for e in self._episodes) / n  # Average SPL
        osr = sum(1 for e in self._episodes if e.oracle_success) / n  # Oracle Success Rate
        ne = sum(e.navigation_error for e in self._episodes) / n  # Navigation Error

        return {
            "num_episodes": n,
            "SR": sr,  # Success Rate
            "SPL": spl,  # Success weighted by Path Length
            "OSR": osr,  # Oracle Success Rate
            "NE": ne,  # Navigation Error (meters)
            "avg_trajectory_length": sum(e.trajectory_length for e in self._episodes) / n,
            "avg_steps": sum(e.steps for e in self._episodes) / n,
        }

    def get_summary_string(self) -> str:
        """Get human-readable summary following Microsoft VLN format."""
        metrics = self.get_aggregate_metrics()

        if not metrics:
            return "No episodes recorded"

        lines = [
            "=" * 50,
            "VLN Evaluation Results (Microsoft Standard Metrics)",
            "=" * 50,
            f"Episodes: {metrics['num_episodes']}",
            "-" * 50,
            f"SR  (Success Rate):              {metrics['SR']*100:.1f}%",
            f"SPL (Success weighted by Path):  {metrics['SPL']:.3f}",
            f"OSR (Oracle Success Rate):       {metrics['OSR']*100:.1f}%",
            f"NE  (Navigation Error):          {metrics['NE']:.2f}m",
            "-" * 50,
            f"Avg Trajectory Length: {metrics['avg_trajectory_length']:.2f}m",
            f"Avg Steps: {metrics['avg_steps']:.1f}",
            "=" * 50,
        ]

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all episode data."""
        self._episodes.clear()

    def get_episodes(self) -> List[EpisodeMetrics]:
        """Get all episode metrics."""
        return self._episodes.copy()


class TokenMetrics:
    """Metrics for token usage tracking."""

    def __init__(self):
        """Initialize token metrics."""
        self._token_counts: Dict[str, List[int]] = {}

    def record(self, category: str, tokens: int) -> None:
        """Record token usage."""
        if category not in self._token_counts:
            self._token_counts[category] = []
        self._token_counts[category].append(tokens)

    def get_total(self, category: str = None) -> int:
        """Get total tokens used."""
        if category:
            return sum(self._token_counts.get(category, []))
        else:
            return sum(sum(v) for v in self._token_counts.values())

    def get_average(self, category: str) -> float:
        """Get average tokens per request."""
        counts = self._token_counts.get(category, [])
        return sum(counts) / len(counts) if counts else 0.0

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get token usage summary."""
        summary = {}

        for category, counts in self._token_counts.items():
            if counts:
                summary[category] = {
                    "total": sum(counts),
                    "average": sum(counts) / len(counts),
                    "count": len(counts),
                }

        return summary