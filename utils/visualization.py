"""Visualization utilities for VLN navigation."""

from typing import Dict, Any, List, Optional, Tuple
import logging

from core.action import ActionType


class TrajectoryVisualizer:
    """
    Visualizer for navigation trajectories.

    Provides visualization for:
    - Agent trajectory
    - Action history
    - Goal position
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("TrajectoryVisualizer")

        # Settings
        self.figsize = self.config.get("figsize", (10, 10))
        self.marker_size = self.config.get("marker_size", 50)

    def visualize_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        goal_position: Optional[Tuple[float, float, float]] = None,
        actions: Optional[List[Any]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Visualize navigation trajectory.

        Args:
            trajectory: List of positions
            goal_position: Optional goal position
            actions: Optional list of actions
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=self.figsize)

            if not trajectory:
                ax.set_title("No trajectory data")
                return fig

            # Extract x and z coordinates (ignoring y/height)
            xs = [p[0] for p in trajectory]
            zs = [p[2] for p in trajectory]

            # Plot trajectory
            ax.plot(xs, zs, 'b-', linewidth=2, label='Trajectory', alpha=0.7)

            # Plot start position
            ax.scatter([xs[0]], [zs[0]], c='green', s=self.marker_size*2,
                       marker='o', label='Start', zorder=5)

            # Plot end position
            ax.scatter([xs[-1]], [zs[-1]], c='red', s=self.marker_size*2,
                       marker='x', label='End', zorder=5)

            # Plot goal if available
            if goal_position:
                ax.scatter([goal_position[0]], [goal_position[2]],
                           c='gold', s=self.marker_size*2, marker='*',
                           label='Goal', zorder=5)

            # Plot action markers
            if actions and len(actions) == len(trajectory):
                self._plot_action_markers(ax, xs, zs, actions)

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title('Navigation Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except ImportError:
            self.logger.warning("matplotlib not available")
            return None

    def _plot_action_markers(
        self,
        ax,
        xs: List[float],
        zs: List[float],
        actions: List[Any],
    ) -> None:
        """Plot markers indicating actions."""
        # Plot turn actions
        for i, action in enumerate(actions):
            if hasattr(action, 'action_type'):
                action_type = action.action_type
            else:
                continue

            if action_type == ActionType.TURN_LEFT:
                ax.annotate('←', (xs[i], zs[i]), fontsize=8, color='orange')
            elif action_type == ActionType.TURN_RIGHT:
                ax.annotate('→', (xs[i], zs[i]), fontsize=8, color='orange')
            elif action_type == ActionType.STOP:
                ax.scatter([xs[i]], [zs[i]], c='purple', s=self.marker_size,
                           marker='s', zorder=4)

    def visualize_trajectory_3d(
        self,
        trajectory: List[Tuple[float, float, float]],
        goal_position: Optional[Tuple[float, float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Visualize trajectory in 3D.

        Args:
            trajectory: List of 3D positions
            goal_position: Optional goal position
            save_path: Optional save path

        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')

            if not trajectory:
                return fig

            xs = [p[0] for p in trajectory]
            ys = [p[1] for p in trajectory]
            zs = [p[2] for p in trajectory]

            # Plot trajectory
            ax.plot(xs, ys, zs, 'b-', linewidth=2, label='Trajectory')

            # Start and end
            ax.scatter([xs[0]], [ys[0]], [zs[0]], c='green', s=100,
                       marker='o', label='Start')
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='red', s=100,
                       marker='x', label='End')

            # Goal
            if goal_position:
                ax.scatter([goal_position[0]], [goal_position[1]], [goal_position[2]],
                           c='gold', s=100, marker='*', label='Goal')

            ax.set_xlabel('X')
            ax.set_ylabel('Y (Height)')
            ax.set_zlabel('Z')
            ax.set_title('3D Navigation Trajectory')
            ax.legend()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except ImportError:
            self.logger.warning("matplotlib 3D not available")
            return None

    def create_trajectory_gif(
        self,
        trajectory: List[Tuple[float, float, float]],
        actions: List[Any] = None,
        output_path: str = "trajectory.gif",
        fps: int = 10,
    ) -> None:
        """
        Create animated GIF of trajectory.

        Args:
            trajectory: List of positions
            actions: Optional list of actions
            output_path: Output file path
            fps: Frames per second
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=self.figsize)

            xs = [p[0] for p in trajectory]
            zs = [p[2] for p in trajectory]

            line, = ax.plot([], [], 'b-', linewidth=2)
            point, = ax.plot([], [], 'ro', markersize=10)

            ax.set_xlim(min(xs) - 1, max(xs) + 1)
            ax.set_ylim(min(zs) - 1, max(zs) + 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title('Navigation Animation')
            ax.grid(True, alpha=0.3)

            def init():
                line.set_data([], [])
                point.set_data([], [])
                return line, point

            def animate(i):
                line.set_data(xs[:i+1], zs[:i+1])
                point.set_data([xs[i]], [zs[i]])
                return line, point

            anim = animation.FuncAnimation(
                fig, animate, init_func=init,
                frames=len(trajectory), interval=1000//fps, blit=True
            )

            anim.save(output_path, writer='pillow', fps=fps)

        except ImportError as e:
            self.logger.warning(f"Animation not available: {e}")


class MetricsVisualizer:
    """Visualizer for evaluation metrics."""

    def __init__(self):
        """Initialize metrics visualizer."""
        self.logger = logging.getLogger("MetricsVisualizer")

    def plot_success_rate_by_task_type(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot success rate by task type.

        Args:
            results: Dictionary with task types as keys
            save_path: Optional save path

        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt

            task_types = list(results.keys())
            success_rates = [
                results[t].get('success_rate', 0) * 100
                for t in task_types
            ]

            fig, ax = plt.subplots(figsize=(10, 6))

            bars = ax.bar(task_types, success_rates, color='steelblue')

            ax.set_ylabel('Success Rate (%)')
            ax.set_xlabel('Task Type')
            ax.set_title('Success Rate by Task Type')
            ax.set_ylim(0, 100)

            # Add value labels
            for bar, rate in zip(bars, success_rates):
                ax.annotate(f'{rate:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           ha='center', va='bottom')

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except ImportError:
            self.logger.warning("matplotlib not available")
            return None

    def plot_spl_distribution(
        self,
        spl_values: List[float],
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot SPL distribution histogram.

        Args:
            spl_values: List of SPL values
            save_path: Optional save path

        Returns:
            Figure object
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(spl_values, bins=20, color='steelblue', edgecolor='white')

            ax.set_xlabel('SPL')
            ax.set_ylabel('Frequency')
            ax.set_title('SPL Distribution')
            ax.axvline(x=sum(spl_values)/len(spl_values), color='red',
                       linestyle='--', label=f'Mean: {sum(spl_values)/len(spl_values):.3f}')
            ax.legend()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')

            return fig

        except ImportError:
            self.logger.warning("matplotlib not available")
            return None


def print_trajectory_summary(
    trajectory: List[Tuple[float, float, float]],
    actions: List[Any] = None,
) -> str:
    """
    Print a text summary of trajectory.

    Args:
        trajectory: List of positions
        actions: Optional list of actions

    Returns:
        Summary string
    """
    if not trajectory:
        return "No trajectory data"

    # Calculate statistics
    total_distance = 0.0
    for i in range(1, len(trajectory)):
        p1, p2 = trajectory[i-1], trajectory[i]
        dist = ((p2[0]-p1[0])**2 + (p2[2]-p1[2])**2)**0.5
        total_distance += dist

    start = trajectory[0]
    end = trajectory[-1]
    straight_line = ((end[0]-start[0])**2 + (end[2]-start[2])**2)**0.5

    # Action distribution
    action_dist = {}
    if actions:
        for action in actions:
            if hasattr(action, 'action_type'):
                name = action.action_type.name
                action_dist[name] = action_dist.get(name, 0) + 1

    lines = [
        "Trajectory Summary",
        "=" * 40,
        f"Total steps: {len(trajectory)}",
        f"Total distance: {total_distance:.2f}m",
        f"Straight-line distance: {straight_line:.2f}m",
        f"Efficiency ratio: {straight_line/total_distance:.2f}" if total_distance > 0 else "N/A",
        f"Start: ({start[0]:.2f}, {start[2]:.2f})",
        f"End: ({end[0]:.2f}, {end[2]:.2f})",
    ]

    if action_dist:
        lines.append("\nAction Distribution:")
        for action_name, count in sorted(action_dist.items()):
            lines.append(f"  {action_name}: {count}")

    return "\n".join(lines)