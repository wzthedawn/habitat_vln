"""Episode output management for VLN system.

Saves visual images, trajectory plots, and agent outputs for each episode.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


@dataclass
class StepOutput:
    """Output data for a single step."""
    step: int
    action: str
    position: List[float]
    rotation: float
    distance_to_goal: float

    # Agent outputs
    perception_output: Optional[Dict[str, Any]] = None
    trajectory_output: Optional[Dict[str, Any]] = None
    decision_output: Optional[Dict[str, Any]] = None
    evaluation_output: Optional[Dict[str, Any]] = None
    strategy_output: Optional[Dict[str, Any]] = None

    # Subtask info
    current_subtask: Optional[Dict[str, Any]] = None


@dataclass
class EpisodeOutput:
    """Output manager for a single episode."""

    episode_id: int
    scene_id: str
    instruction: str
    goal_position: List[float]
    start_position: List[float]
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())

    # Results
    success: bool = False
    final_distance: float = 0.0
    min_distance: float = float('inf')
    steps: int = 0
    trajectory: List[List[float]] = field(default_factory=list)

    # Task info
    task_level: str = "中等"
    subtasks: List[Dict[str, Any]] = field(default_factory=list)

    # Step-by-step outputs
    step_outputs: List[StepOutput] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step_output(self, step_output: StepOutput) -> None:
        """Add a step output."""
        self.step_outputs.append(step_output)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "instruction": self.instruction,
            "goal_position": self.goal_position,
            "start_position": self.start_position,
            "start_time": self.start_time,
            "success": self.success,
            "final_distance": self.final_distance,
            "min_distance": self.min_distance,
            "steps": self.steps,
            "trajectory": self.trajectory,
            "task_level": self.task_level,
            "subtasks": self.subtasks,
            "step_outputs": [asdict(s) for s in self.step_outputs],
            "metadata": self.metadata,
        }


class EpisodeOutputManager:
    """Manages episode outputs including images, trajectory plots, and agent outputs."""

    def __init__(
        self,
        output_dir: str = "results",
        enable_video: bool = True,
        enable_trajectory: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.enable_video = enable_video
        self.enable_trajectory = enable_trajectory
        self.logger = logging.getLogger("EpisodeOutputManager")
        self.current_episode: Optional[EpisodeOutput] = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        if not self.enable_video:
            self.logger.info("Video generation disabled")
        if not self.enable_trajectory:
            self.logger.info("Trajectory plot generation disabled")

    def start_episode(
        self,
        episode_id: int,
        scene_id: str,
        instruction: str,
        goal_position: List[float],
        start_position: List[float],
    ) -> EpisodeOutput:
        """Start a new episode output."""
        self.current_episode = EpisodeOutput(
            episode_id=episode_id,
            scene_id=scene_id,
            instruction=instruction,
            goal_position=goal_position,
            start_position=start_position,
        )

        # Create episode directory (session dir already has timestamp from run_vln_experiment.py)
        episode_dir = self.output_dir / f"episode{episode_id}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (episode_dir / "images").mkdir(exist_ok=True)
        (episode_dir / "trajectory").mkdir(exist_ok=True)

        self.logger.info(f"Started episode {episode_id}, output dir: {episode_dir}")
        return self.current_episode

    def _get_episode_dir(self) -> Path:
        """Get current episode directory."""
        if self.current_episode is None:
            return self.output_dir
        return self.output_dir / f"episode{self.current_episode.episode_id}"

    def save_rgb_image(
        self,
        rgb_image: np.ndarray,
        step: int,
    ) -> str:
        """Save RGB image for a step."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return ""

        episode_dir = self._get_episode_dir()
        image_path = episode_dir / "images" / f"step_{step:04d}_rgb.png"

        try:
            import cv2
            # RGB to BGR for OpenCV
            if rgb_image is not None and rgb_image.size > 0:
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), bgr_image)
                self.logger.debug(f"Saved RGB image: {image_path}")
                return str(image_path)
        except ImportError:
            # Fallback to PIL if cv2 not available
            try:
                from PIL import Image
                if rgb_image is not None and rgb_image.size > 0:
                    img = Image.fromarray(rgb_image)
                    img.save(str(image_path))
                    return str(image_path)
            except ImportError:
                self.logger.warning("Neither cv2 nor PIL available for image saving")

        return ""

    def save_depth_image(
        self,
        depth_image: np.ndarray,
        step: int,
    ) -> str:
        """Save depth image for a step."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return ""

        episode_dir = self._get_episode_dir()
        image_path = episode_dir / "images" / f"step_{step:04d}_depth.png"

        try:
            import cv2
            if depth_image is not None and depth_image.size > 0:
                # Normalize depth for visualization
                # IMPORTANT: JET colormap 0=blue(far), 255=red(near)
                # depth值=距离，需要反转：小距离(近处)→大像素值→红色
                depth_clipped = np.clip(depth_image, 0, 10)  # 10 meters max
                depth_normalized = (255 - depth_clipped / 10.0 * 255).astype(np.uint8)
                # Apply colormap for better visualization
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                cv2.imwrite(str(image_path), depth_colored)
                self.logger.debug(f"Saved depth image: {image_path}")
                return str(image_path)
        except ImportError:
            try:
                from PIL import Image
                if depth_image is not None and depth_image.size > 0:
                    # Normalize depth
                    depth_clipped = np.clip(depth_image, 0, 10)
                    depth_normalized = (depth_clipped / 10.0 * 255).astype(np.uint8)
                    img = Image.fromarray(depth_normalized)
                    img.save(str(image_path))
                    return str(image_path)
            except ImportError:
                self.logger.warning("Neither cv2 nor PIL available for image saving")

        return ""

    def add_step_output(
        self,
        step: int,
        action: str,
        position: Tuple[float, float, float],
        rotation: float,
        distance_to_goal: float,
        perception_output: Optional[Dict] = None,
        trajectory_output: Optional[Dict] = None,
        decision_output: Optional[Dict] = None,
        evaluation_output: Optional[Dict] = None,
        strategy_output: Optional[Dict] = None,
        current_subtask: Optional[Dict] = None,
    ) -> None:
        """Add step output data."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return

        step_output = StepOutput(
            step=step,
            action=action,
            position=list(position),
            rotation=rotation,
            distance_to_goal=distance_to_goal,
            perception_output=perception_output,
            trajectory_output=trajectory_output,
            decision_output=decision_output,
            evaluation_output=evaluation_output,
            strategy_output=strategy_output,
            current_subtask=current_subtask,
        )
        self.current_episode.add_step_output(step_output)

    def save_trajectory_plot(
        self,
        trajectory: List[List[float]],
        goal_position: List[float],
        reference_path: Optional[List[List[float]]] = None,
    ) -> str:
        """Save trajectory plot."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return ""

        episode_dir = self._get_episode_dir()
        plot_path = episode_dir / "trajectory" / "trajectory_plot.png"

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 10))

            # Plot trajectory
            if trajectory:
                traj_x = [p[0] for p in trajectory]
                traj_z = [p[2] for p in trajectory]
                ax.plot(traj_x, traj_z, 'b-', linewidth=2, label='Agent Trajectory', marker='o', markersize=3)

                # Mark start
                ax.plot(traj_x[0], traj_z[0], 'go', markersize=12, label='Start')

                # Mark end
                ax.plot(traj_x[-1], traj_z[-1], 'bo', markersize=12, label='End')

            # Plot reference path if available
            if reference_path:
                ref_x = [p[0] for p in reference_path]
                ref_z = [p[2] for p in reference_path]
                ax.plot(ref_x, ref_z, 'g--', linewidth=2, label='Reference Path', alpha=0.7)

            # Plot goal
            ax.plot(goal_position[0], goal_position[2], 'r*', markersize=20, label='Goal')

            ax.set_xlabel('X (meters)', fontsize=12)
            ax.set_ylabel('Z (meters)', fontsize=12)
            ax.set_title(f'Trajectory Plot - Episode {self.current_episode.episode_id}\n'
                        f'Success: {self.current_episode.success}, Steps: {self.current_episode.steps}',
                        fontsize=14)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
            plt.close(fig)

            self.logger.info(f"Saved trajectory plot: {plot_path}")
            return str(plot_path)

        except ImportError:
            self.logger.warning("matplotlib not available for trajectory plotting")
            return ""
        except Exception as e:
            self.logger.error(f"Error creating trajectory plot: {e}")
            return ""

    def save_agent_outputs(self) -> str:
        """Save all agent outputs to JSON file."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return ""

        episode_dir = self._get_episode_dir()
        output_path = episode_dir / "agent_outputs.json"

        output_data = self.current_episode.to_dict()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved agent outputs: {output_path}")
        return str(output_path)

    def finish_episode(
        self,
        success: bool,
        final_distance: float,
        min_distance: float,
        steps: int,
        trajectory: List[List[float]],
        task_level: str = "中等",
        subtasks: List[Dict] = None,
        goal_position: List[float] = None,
        reference_path: List[List[float]] = None,
    ) -> str:
        """Finish episode and save all outputs."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return ""

        # Update episode data
        self.current_episode.success = success
        self.current_episode.final_distance = final_distance
        self.current_episode.min_distance = min_distance
        self.current_episode.steps = steps
        self.current_episode.trajectory = trajectory
        self.current_episode.task_level = task_level
        if subtasks:
            self.current_episode.subtasks = subtasks

        # Save trajectory plot (if enabled)
        if goal_position and self.enable_trajectory:
            self.save_trajectory_plot(trajectory, goal_position, reference_path)

        # Save agent outputs
        output_path = self.save_agent_outputs()

        self.logger.info(f"Finished episode {self.current_episode.episode_id}: "
                        f"success={success}, steps={steps}, distance={final_distance:.2f}m")

        return output_path

    def create_summary_video(
        self,
        fps: int = 5,
    ) -> Optional[str]:
        """Create a video from the episode's RGB images."""
        if self.current_episode is None:
            self.logger.warning("No active episode")
            return None

        episode_dir = self._get_episode_dir()
        images_dir = episode_dir / "images"
        video_path = episode_dir / "episode_video.mp4"

        try:
            import cv2

            # Get all RGB images
            rgb_files = sorted(images_dir.glob("step_*_rgb*.png"))
            if not rgb_files:
                self.logger.warning("No RGB images found for video")
                return None

            # Read first image to get dimensions
            first_img = cv2.imread(str(rgb_files[0]))
            height, width = first_img.shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for rgb_file in rgb_files:
                img = cv2.imread(str(rgb_file))
                out.write(img)

            out.release()
            self.logger.info(f"Created summary video: {video_path}")
            return str(video_path)

        except ImportError:
            self.logger.warning("cv2 not available for video creation")
            return None
        except Exception as e:
            self.logger.error(f"Error creating video: {e}")
            return None

    def get_episode_dir(self) -> Path:
        """Get the current episode directory."""
        return self._get_episode_dir()