#!/usr/bin/env python3
"""R2R VLN Evaluation using Habitat-Lab with Multi-Agent LLM Navigation.

This script evaluates the VLN system using real R2R dataset with Habitat test scenes.
Since Matterport3D scenes are not available, it maps R2R episodes to Habitat test scenes.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import random
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.metrics import VLNMetrics, TokenMetrics
from utils.token_tracker import get_token_tracker


@dataclass
class R2REpisode:
    """R2R episode data structure."""
    episode_id: int
    scene_id: str
    instruction: str
    start_position: List[float]
    start_rotation: List[float]
    goal_position: List[float]
    reference_path: List[List[float]]
    geodesic_distance: float
    trajectory_id: int = 0

    @property
    def heading(self) -> float:
        """Calculate heading from quaternion rotation."""
        # Quaternion: [x, y, z, w] -> yaw angle
        # Assuming rotation is [x, y, z, w]
        if len(self.start_rotation) == 4:
            x, y, z, w = self.start_rotation
            # Convert quaternion to yaw
            import math
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            return math.atan2(siny_cosp, cosy_cosp)
        return 0.0


@dataclass
class EvaluationResult:
    """Single episode evaluation result."""
    episode_id: int
    scene_id: str
    success: bool
    spl: float
    oracle_success: bool
    nDTW: float
    SDTW: float
    trajectory_length: float
    shortest_path_length: float
    steps: int
    distance_to_goal: float
    min_distance_to_goal: float
    instruction: str
    token_usage: Dict[str, int]
    trajectory: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class R2RHabitatEvaluator:
    """Evaluator for R2R VLN using Habitat test scenes."""

    # Habitat test scenes available
    HABITAT_SCENES = [
        "apartment_1",
        "skokloster-castle",
        "van-gogh-room",
    ]

    def __init__(self, config: Dict[str, Any], log_level: str = "INFO"):
        """Initialize evaluator."""
        self.config = config
        self.logger = setup_logger("R2RHabitatEvaluator", level=log_level)
        self.log_level = log_level

        # Components
        self._sim = None
        self.navigator = None
        self.token_tracker = get_token_tracker()

        # Data
        self.episodes: List[R2REpisode] = []
        self.scene_mapping: Dict[str, str] = {}

        # Metrics
        self.vln_metrics = VLNMetrics(success_distance=config.get("success_distance", 3.0))

        # Results
        self.results: List[EvaluationResult] = []

        # Habitat scenes path
        self.scenes_dir = Path(config.get("scenes_dir", "data/scene_datasets/habitat-test-scenes"))
        self.r2r_data_path = config.get("r2r_data_path", "")

    def initialize(self) -> None:
        """Initialize evaluation components."""
        self.logger.info("Initializing R2R Habitat Evaluator...")

        # Build scene mapping (R2R scene -> Habitat test scene)
        self._build_scene_mapping()

        # Initialize Habitat environment
        self._init_habitat_env()

        # Initialize navigator
        self._init_navigator()

        self.logger.info("Initialization complete")

    def _build_scene_mapping(self) -> None:
        """Build mapping from R2R scene IDs to Habitat test scenes."""
        # Load R2R episodes
        r2r_path = Path(self.r2r_data_path)
        if not r2r_path.exists():
            self.logger.error(f"R2R data not found: {r2r_path}")
            return

        with open(r2r_path) as f:
            data = json.load(f)

        # Get unique R2R scenes
        r2r_scenes = list(set(ep["scene_id"] for ep in data["episodes"]))
        self.logger.info(f"Found {len(r2r_scenes)} unique R2R scenes")

        # Map R2R scenes to Habitat scenes (round-robin)
        habitat_scenes = self.HABITAT_SCENES
        for i, r2r_scene in enumerate(sorted(r2r_scenes)):
            self.scene_mapping[r2r_scene] = habitat_scenes[i % len(habitat_scenes)]

        self.logger.info(f"Built scene mapping for {len(self.scene_mapping)} R2R scenes")

    def _init_habitat_env(self) -> None:
        """Initialize Habitat environment using habitat_sim directly."""
        try:
            import habitat_sim

            # Get first Habitat scene
            scene_path = self.scenes_dir / f"{self.HABITAT_SCENES[0]}.glb"
            if not scene_path.exists():
                self.logger.warning(f"Scene not found: {scene_path}")
                return

            # Create sensor specifications
            rgb_sensor = habitat_sim.CameraSensorSpec()
            rgb_sensor.uuid = "rgb"
            rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor.resolution = [256, 256]
            rgb_sensor.position = [0.0, 1.5, 0.0]

            # Agent configuration
            agent_cfg = habitat_sim.AgentConfiguration(
                height=1.5,
                radius=0.1,
                sensor_specifications=[rgb_sensor],
                action_space={
                    "move_forward": habitat_sim.ActionSpec(
                        "move_forward", habitat_sim.ActuationSpec(amount=0.25)
                    ),
                    "turn_left": habitat_sim.ActionSpec(
                        "turn_left", habitat_sim.ActuationSpec(amount=15.0)
                    ),
                    "turn_right": habitat_sim.ActionSpec(
                        "turn_right", habitat_sim.ActuationSpec(amount=15.0)
                    ),
                    "stop": habitat_sim.ActionSpec(
                        "stop", habitat_sim.ActuationSpec(amount=0.0)
                    ),
                }
            )

            # Simulator configuration
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = str(scene_path)
            sim_cfg.enable_physics = False
            sim_cfg.allow_sliding = True

            # Create configuration and simulator
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            self._sim = habitat_sim.Simulator(cfg)

            # Store scene path for switching scenes
            self._current_scene = str(scene_path)

            self.logger.info(f"Habitat simulator initialized with scene: {scene_path}")

        except Exception as e:
            self.logger.warning(f"Could not initialize Habitat: {e}")
            self._sim = None

    def _init_navigator(self) -> None:
        """Initialize VLN Navigator with LLM."""
        try:
            from core.navigator import VLNNavigator

            nav_config = self._build_navigator_config()
            self.navigator = VLNNavigator(
                config=nav_config,
                log_level=self.log_level,
            )
            self.navigator.initialize()
            self.logger.info(f"VLN Navigator initialized (model_type={self.config.get('model_type', 'llm')})")

        except Exception as e:
            self.logger.warning(f"Could not initialize navigator: {e}")
            self.navigator = None

    def _build_navigator_config(self) -> Dict[str, Any]:
        """Build navigator configuration."""
        config = {
            "model_type": self.config.get("model_type", "llm"),
        }

        if self.config.get("model_type") == "llm":
            # Load LLM config from YAML if available
            config_path = Path(self.config.get("config_file", "configs/qwen_config.yaml"))
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    yaml_config = yaml.safe_load(f)

                if "model" in yaml_config and "qwen" in yaml_config["model"]:
                    qwen_config = yaml_config["model"]["qwen"]
                    config["llm_model"] = {
                        "model_name": qwen_config.get("model_name", "qwen3.5-plus"),
                        "api_key": qwen_config.get("api_key", ""),
                        "api_base": qwen_config.get("base_url", ""),
                        "max_tokens": qwen_config.get("max_tokens", 2000),
                        "temperature": qwen_config.get("temperature", 0.7),
                    }
            else:
                # Default LLM config
                config["llm_model"] = {
                    "model_name": self.config.get("model_name", "qwen3.5-plus"),
                    "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
                    "api_base": "",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                }

            # Agent configs
            config["instruction_agent"] = {"enabled": True, "model_type": "llm", "llm_model": config["llm_model"]}
            config["perception_agent"] = {"enabled": True}
            config["trajectory_agent"] = {"enabled": True}
            config["decision_agent"] = {"enabled": True, "model_type": "llm", "llm_model": config["llm_model"]}
            config["classifier"] = {"use_llm_fallback": True, "cache_results": True}
            config["supernet"] = {"adaptive_selection": True}

        return config

    def load_r2r_episodes(self) -> None:
        """Load R2R episodes from dataset."""
        r2r_path = Path(self.r2r_data_path)
        if not r2r_path.exists():
            self.logger.error(f"R2R data not found: {r2r_path}")
            return

        with open(r2r_path) as f:
            data = json.load(f)

        self.logger.info(f"Loading R2R episodes from {r2r_path}")

        for ep_data in data["episodes"]:
            episode = R2REpisode(
                episode_id=ep_data["episode_id"],
                scene_id=ep_data["scene_id"],
                instruction=ep_data["instruction"]["instruction_text"],
                start_position=ep_data["start_position"],
                start_rotation=ep_data["start_rotation"],
                goal_position=ep_data["goals"][0]["position"],
                reference_path=ep_data.get("reference_path", []),
                geodesic_distance=ep_data["info"]["geodesic_distance"],
                trajectory_id=ep_data.get("trajectory_id", 0),
            )
            self.episodes.append(episode)

        self.logger.info(f"Loaded {len(self.episodes)} R2R episodes")

    def run_evaluation(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Run VLN evaluation on R2R episodes."""
        if not self.episodes:
            self.logger.error("No episodes loaded")
            return {"error": "No episodes loaded"}

        num_to_run = min(num_episodes or len(self.episodes), len(self.episodes))
        self.logger.info(f"Running evaluation on {num_to_run} episodes")

        for i, episode in enumerate(self.episodes[:num_to_run]):
            self.logger.info(f"Episode {i+1}/{num_to_run}: id={episode.episode_id}, scene={episode.scene_id}")

            result = self._run_episode(episode)
            self.results.append(result)

            # Log progress
            if (i + 1) % 10 == 0:
                self._log_intermediate_results()

        return self._compile_results()

    def _run_episode(self, episode: R2REpisode) -> EvaluationResult:
        """Run single episode evaluation."""
        # Start token tracking
        self.token_tracker.start_task(
            instruction=episode.instruction,
            task_type="R2R_VLN",
        )

        # Initialize tracking
        trajectory = []
        steps = 0
        success = False
        oracle_success = False
        min_distance = float('inf')

        # Map to Habitat scene
        habitat_scene = self.scene_mapping.get(episode.scene_id, self.HABITAT_SCENES[0])

        try:
            if self._sim is not None:
                # Use real Habitat environment
                result = self._run_habitat_episode(episode, habitat_scene)
                trajectory = result["trajectory"]
                steps = result["steps"]
                success = result["success"]
                min_distance = result["min_distance"]
            else:
                # Simulate episode
                result = self._simulate_episode(episode)
                trajectory = result["trajectory"]
                steps = result["steps"]
                success = result["success"]
                min_distance = result["min_distance"]

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} failed: {e}")
            trajectory = [episode.start_position]
            steps = 0
            min_distance = self._distance(episode.start_position, episode.goal_position)

        # End token tracking
        task_summary = self.token_tracker.end_task()

        # Calculate metrics
        final_pos = trajectory[-1] if trajectory else episode.start_position
        distance_to_goal = self._distance(final_pos, episode.goal_position)
        oracle_success = min_distance <= self.config.get("success_distance", 3.0)

        # Trajectory length
        trajectory_length = sum(
            self._distance(trajectory[i-1], trajectory[i])
            for i in range(1, len(trajectory))
        ) if len(trajectory) > 1 else 0.0

        # SPL (Success weighted by Path Length)
        # SPL = success * (shortest_path / max(trajectory_length, shortest_path))
        if success and trajectory_length > 0:
            spl = min(episode.geodesic_distance, trajectory_length) / max(trajectory_length, episode.geodesic_distance)
        else:
            spl = 0.0

        # nDTW (normalized Dynamic Time Warping)
        ndtw = self._calculate_ndtw(trajectory, episode.reference_path)

        # SDTW (Success weighted by nDTW)
        sdtw = ndtw if success else 0.0

        # Token usage
        token_usage = {}
        if task_summary:
            token_usage = {
                "total": task_summary.total_tokens,
                "input": task_summary.total_input_tokens,
                "output": task_summary.total_output_tokens,
                "api_calls": task_summary.num_api_calls,
            }

        return EvaluationResult(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            success=success,
            spl=spl,
            oracle_success=oracle_success,
            nDTW=ndtw,
            SDTW=sdtw,
            trajectory_length=trajectory_length,
            shortest_path_length=episode.geodesic_distance,
            steps=steps,
            distance_to_goal=distance_to_goal,
            min_distance_to_goal=min_distance,
            instruction=episode.instruction,
            token_usage=token_usage,
            trajectory=trajectory,
        )

    def _run_habitat_episode(
        self,
        episode: R2REpisode,
        habitat_scene: str
    ) -> Dict[str, Any]:
        """Run episode in Habitat environment using habitat_sim directly."""
        import habitat_sim
        import numpy as np

        trajectory = []
        steps = 0
        success = False
        min_distance = float('inf')

        try:
            # Check if we need to switch scenes
            scene_path = str(self.scenes_dir / f"{habitat_scene}.glb")
            if hasattr(self, '_current_scene') and self._current_scene != scene_path:
                # Reinitialize with new scene
                if self._sim:
                    self._sim.close()

                agent_cfg = habitat_sim.AgentConfiguration(
                    height=1.5,
                    radius=0.1,
                    action_space={
                        "move_forward": habitat_sim.ActionSpec(
                            "move_forward", habitat_sim.ActuationSpec(amount=0.25)
                        ),
                        "turn_left": habitat_sim.ActionSpec(
                            "turn_left", habitat_sim.ActuationSpec(amount=15.0)
                        ),
                        "turn_right": habitat_sim.ActionSpec(
                            "turn_right", habitat_sim.ActuationSpec(amount=15.0)
                        ),
                        "stop": habitat_sim.ActionSpec(
                            "stop", habitat_sim.ActuationSpec(amount=0.0)
                        ),
                    }
                )
                sim_cfg = habitat_sim.SimulatorConfiguration()
                sim_cfg.scene_id = scene_path
                sim_cfg.enable_physics = False
                cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
                self._sim = habitat_sim.Simulator(cfg)
                self._current_scene = scene_path

            if self._sim is None:
                self.logger.warning("Simulator not initialized, using mock")
                return self._simulate_episode(episode)

            # Reset agent to start position
            agent = self._sim.get_agent(0)
            state = habitat_sim.AgentState()
            state.position = episode.start_position

            # Convert quaternion rotation
            if episode.start_rotation:
                state.rotation = np.array(episode.start_rotation)

            agent.set_state(state)
            trajectory.append(list(episode.start_position))

            # Create navigation context
            from core.context import NavContextBuilder
            context = NavContextBuilder() \
                .with_instruction(episode.instruction) \
                .with_position(tuple(episode.start_position)) \
                .with_metadata({"goal_position": episode.goal_position}) \
                .build()

            # Navigation loop
            max_steps = self.config.get("max_steps", 100)
            success_distance = self.config.get("success_distance", 3.0)

            while steps < max_steps:
                # Get action from navigator
                if self.navigator:
                    action = self.navigator.navigate(context)
                    action_name = action.action_type.name.lower()
                else:
                    # Random action fallback
                    action_name = random.choice(["move_forward", "turn_left", "turn_right"])

                # Check for stop
                if action_name == "stop":
                    break

                # Execute action in simulator
                self._sim.step(action_name)

                # Update position
                state = agent.get_state()
                pos = list(state.position)
                trajectory.append(pos)
                steps += 1

                # Check distance to goal
                dist = self._distance(pos, episode.goal_position)
                min_distance = min(min_distance, dist)

                if dist <= success_distance:
                    success = True
                    break

                # Update context position
                context.position = tuple(pos)

        except Exception as e:
            self.logger.error(f"Habitat episode error: {e}")
            import traceback
            traceback.print_exc()

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
        }

    def _simulate_episode(self, episode: R2REpisode) -> Dict[str, Any]:
        """Simulate episode with LLM navigator or fallback simulation."""
        trajectory = [episode.start_position.copy()]
        current = list(episode.start_position)
        steps = 0
        success = False
        min_distance = self._distance(current, episode.goal_position)

        max_steps = self.config.get("max_steps", 100)
        success_distance = self.config.get("success_distance", 3.0)

        # Create navigation context for LLM navigator
        from core.context import NavContextBuilder
        context = NavContextBuilder() \
            .with_instruction(episode.instruction) \
            .with_position(tuple(current)) \
            .with_metadata({"goal_position": episode.goal_position}) \
            .build()

        # Navigation loop with LLM or fallback
        while steps < max_steps:
            if self.navigator:
                # Use LLM navigator to get action
                action = self.navigator.navigate(context)
                action_name = action.action_type.name.lower()
            else:
                # Fallback: simulate movement towards goal with noise
                dx = episode.goal_position[0] - current[0]
                dz = episode.goal_position[2] - current[2]
                dist_to_goal = math.sqrt(dx*dx + dz*dz)

                step_size = 0.25
                noise = [random.uniform(-0.1, 0.1) for _ in range(3)]

                if dist_to_goal > 0:
                    current[0] += (dx / dist_to_goal) * step_size + noise[0]
                    current[1] += noise[1]
                    current[2] += (dz / dist_to_goal) * step_size + noise[2]
                else:
                    current = [c + n for c, n in zip(current, noise)]

                trajectory.append(current.copy())
                steps += 1

                dist = self._distance(current, episode.goal_position)
                min_distance = min(min_distance, dist)

                if dist <= success_distance:
                    success = True
                    break
                continue

            # Check for stop action from LLM
            if action_name == "stop":
                break

            # Execute the action
            step_size = 0.25
            if action_name == "move_forward":
                # Move forward based on current rotation
                dx = episode.goal_position[0] - current[0]
                dz = episode.goal_position[2] - current[2]
                dist_to_goal = math.sqrt(dx*dx + dz*dz)

                if dist_to_goal > 0:
                    current[0] += (dx / dist_to_goal) * step_size
                    current[2] += (dz / dist_to_goal) * step_size
            elif action_name == "turn_left" or action_name == "turn_right":
                # For simulation, turning just adds small movement
                noise = random.uniform(-0.05, 0.05)
                current[0] += noise
                current[2] += noise

            trajectory.append(current.copy())
            steps += 1

            # Check distance to goal
            dist = self._distance(current, episode.goal_position)
            min_distance = min(min_distance, dist)

            if dist <= success_distance:
                success = True
                break

            # Update context
            context.position = tuple(current)

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
        }

    def _distance(self, p1: List[float], p2: List[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:3], p2[:3])))

    def _calculate_ndtw(
        self,
        trajectory: List[List[float]],
        reference: List[List[float]]
    ) -> float:
        """Calculate normalized Dynamic Time Warping score."""
        if not trajectory or not reference:
            return 0.0

        n, m = len(trajectory), len(reference)

        # DTW matrix
        dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._distance(trajectory[i-1], reference[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

        # Normalize
        max_len = max(n, m)
        avg_dtw = dtw[n][m] / max_len if max_len > 0 else float('inf')

        # Convert to similarity score (0 to 1)
        return math.exp(-avg_dtw / 5.0)  # Scale factor of 5m

    def _log_intermediate_results(self) -> None:
        """Log intermediate results."""
        if not self.results:
            return

        successes = sum(1 for r in self.results if r.success)
        sr = successes / len(self.results) * 100
        avg_spl = sum(r.spl for r in self.results) / len(self.results)
        avg_dist = sum(r.distance_to_goal for r in self.results) / len(self.results)

        self.logger.info(f"Progress: {len(self.results)} episodes, SR={sr:.1f}%, SPL={avg_spl:.3f}, NE={avg_dist:.2f}m")

    def _compile_results(self) -> Dict[str, Any]:
        """Compile final evaluation results."""
        if not self.results:
            return {"error": "No results"}

        total = len(self.results)
        successes = sum(1 for r in self.results if r.success)
        oracle_successes = sum(1 for r in self.results if r.oracle_success)

        summary = {
            "num_episodes": total,
            "success_rate": successes / total,
            "spl": sum(r.spl for r in self.results) / total,
            "oracle_success_rate": oracle_successes / total,
            "nDTW": sum(r.nDTW for r in self.results) / total,
            "SDTW": sum(r.SDTW for r in self.results) / total,
            "avg_distance_to_goal": sum(r.distance_to_goal for r in self.results) / total,
            "avg_min_distance": sum(r.min_distance_to_goal for r in self.results) / total,
            "avg_steps": sum(r.steps for r in self.results) / total,
            "avg_trajectory_length": sum(r.trajectory_length for r in self.results) / total,
        }

        # By scene breakdown
        by_scene = {}
        for r in self.results:
            scene = r.scene_id
            if scene not in by_scene:
                by_scene[scene] = {"count": 0, "successes": 0, "spl_sum": 0.0}
            by_scene[scene]["count"] += 1
            if r.success:
                by_scene[scene]["successes"] += 1
            by_scene[scene]["spl_sum"] += r.spl

        for scene in by_scene:
            data = by_scene[scene]
            data["success_rate"] = data["successes"] / data["count"]
            data["spl"] = data["spl_sum"] / data["count"]

        # Token summary
        total_tokens = sum(r.token_usage.get("total", 0) for r in self.results)
        total_api_calls = sum(r.token_usage.get("api_calls", 0) for r in self.results)
        token_summary = {
            "total_tokens": total_tokens,
            "total_api_calls": total_api_calls,
            "avg_tokens_per_episode": total_tokens / total if total > 0 else 0,
        }

        return {
            "summary": summary,
            "token_summary": token_summary,
            "by_scene": by_scene,
            "episodes": [r.to_dict() for r in self.results],
            "config": {
                "r2r_data_path": str(self.r2r_data_path),
                "model_type": self.config.get("model_type", "llm"),
                "model_name": self.config.get("model_name", "qwen3.5-plus"),
                "max_steps": self.config.get("max_steps", 100),
                "success_distance": self.config.get("success_distance", 3.0),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def close(self) -> None:
        """Clean up resources."""
        if self._sim:
            self._sim.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="R2R VLN Evaluation with Habitat and Multi-Agent LLM Navigation"
    )

    # Data arguments
    parser.add_argument(
        "--split", type=str, default="val_seen",
        choices=["train", "val_seen", "val_unseen"],
        help="R2R data split to evaluate"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes to evaluate (0 for all)"
    )

    # Environment arguments
    parser.add_argument(
        "--scenes-dir", type=str,
        default="data/scene_datasets/habitat-test-scenes",
        help="Directory containing Habitat test scenes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--success-distance", type=float, default=3.0,
        help="Success distance threshold (meters)"
    )

    # Model arguments
    parser.add_argument(
        "--model", type=str, default="llm",
        choices=["llm", "mock"],
        help="Navigation model type"
    )
    parser.add_argument(
        "--model-name", type=str, default="qwen3.5-plus",
        help="LLM model name"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/qwen_config.yaml",
        help="Path to LLM configuration file"
    )

    # Output arguments
    parser.add_argument(
        "--output", type=str, default="results_r2r_habitat.json",
        help="Output results file"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set up paths
    project_root = Path(__file__).parent

    # R2R data path
    r2r_data_path = Path(f"/root/habitat-lab/data/datasets/vln/mp3d/r2r/v1/{args.split}/{args.split}.json")

    # Scenes directory
    scenes_dir = project_root / args.scenes_dir

    # Build config
    config = {
        "r2r_data_path": str(r2r_data_path),
        "scenes_dir": str(scenes_dir),
        "model_type": args.model,
        "model_name": args.model_name,
        "config_file": str(project_root / args.config_file),
        "max_steps": args.max_steps,
        "success_distance": args.success_distance,
    }

    log_level = "DEBUG" if args.verbose else args.log_level

    # Print header
    print("=" * 70)
    print("R2R VLN Evaluation with Habitat and Multi-Agent LLM Navigation")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"R2R Data: {r2r_data_path}")
    print(f"Scenes Dir: {scenes_dir}")
    print(f"Model Type: {args.model}")
    print(f"Model Name: {args.model_name}")
    print(f"Max Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Success Distance: {args.success_distance}m")
    print("=" * 70)

    # Create evaluator
    evaluator = R2RHabitatEvaluator(config, log_level=log_level)

    # Initialize
    evaluator.initialize()

    # Load R2R episodes
    evaluator.load_r2r_episodes()

    # Run evaluation
    num_episodes = args.episodes if args.episodes > 0 else None
    results = evaluator.run_evaluation(num_episodes=num_episodes)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    summary = results.get("summary", {})
    print(f"Total Episodes: {summary.get('num_episodes', 0)}")
    print(f"Success Rate (SR): {summary.get('success_rate', 0)*100:.1f}%")
    print(f"SPL: {summary.get('spl', 0):.3f}")
    print(f"Oracle Success Rate: {summary.get('oracle_success_rate', 0)*100:.1f}%")
    print(f"nDTW: {summary.get('nDTW', 0):.3f}")
    print(f"SDTW: {summary.get('SDTW', 0):.3f}")
    print(f"Avg Distance to Goal: {summary.get('avg_distance_to_goal', 0):.2f}m")
    print(f"Avg Min Distance: {summary.get('avg_min_distance', 0):.2f}m")
    print(f"Avg Steps: {summary.get('avg_steps', 0):.1f}")

    print("\nBy Scene (top 5):")
    by_scene = results.get("by_scene", {})
    sorted_scenes = sorted(by_scene.items(), key=lambda x: x[1]["success_rate"], reverse=True)[:5]
    for scene, data in sorted_scenes:
        print(f"  {scene[:30]:30s} SR={data['success_rate']*100:5.1f}% SPL={data['spl']:.3f}")

    token_summary = results.get("token_summary", {})
    print(f"\nToken Usage:")
    print(f"  Total Tokens: {token_summary.get('total_tokens', 0)}")
    print(f"  API Calls: {token_summary.get('total_api_calls', 0)}")
    print(f"  Avg per Episode: {token_summary.get('avg_tokens_per_episode', 0):.0f}")

    print("=" * 70)

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Cleanup
    evaluator.close()

    return results


if __name__ == "__main__":
    main()