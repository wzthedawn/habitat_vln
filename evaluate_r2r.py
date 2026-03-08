#!/usr/bin/env python3
"""R2R evaluation script for VLN system in Habitat environment.

This script evaluates the multi-agent VLN navigation system using:
- Real LLM-based navigation (qwen3.5-plus by default)
- Multi-agent collaboration (Instruction, Perception, Trajectory, Decision agents)
- Strategy execution (ReAct, CoT, Debate, Reflection)
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
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.metrics import VLNMetrics, TokenMetrics
from utils.token_tracker import get_token_tracker
from data.r2r_loader import R2RDataset, R2REpisode
from data.connectivity_graph import ConnectivityGraph
from data.mock_r2r_generator import MockR2RGenerator


@dataclass
class R2REvaluationConfig:
    """Configuration for R2R evaluation."""
    # Mode
    mode: str = "mock"  # mock or real

    # Scene settings
    scene_path: str = ""
    scenes_dir: str = "data/scene_datasets/habitat-test-scenes"

    # R2R data settings
    data_path: str = ""
    connectivity_path: str = ""

    # Episode settings
    num_episodes: int = 20
    max_steps: int = 100
    success_distance: float = 3.0

    # Navigation settings
    forward_amount: float = 0.25
    turn_angle: float = 15.0

    # Model settings
    model_type: str = "llm"  # "llm" for real LLM, "mock" for testing
    model_name: str = "qwen3.5-plus"
    api_key: str = ""
    api_base: str = ""

    # Config file
    config_file: str = "configs/qwen_config.yaml"

    # Output
    output: str = "results_r2r.json"
    verbose: bool = False

    # LLM config loaded from YAML
    llm_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class R2REpisodeResult:
    """Result of a single R2R episode evaluation."""
    episode_id: int
    scene_id: str
    success: bool
    spl: float
    oracle_success: bool
    nDTW: float
    trajectory_length: float
    shortest_path_length: float
    steps: int
    distance_to_goal: float
    instruction: str
    token_usage: Dict[str, int]
    trajectory: List[Tuple[float, float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class R2REvaluator:
    """Evaluator for VLN system using R2R dataset."""

    def __init__(self, config: R2REvaluationConfig, log_level: str = "INFO"):
        """Initialize R2R evaluator."""
        self.config = config
        self.logger = setup_logger("R2REvaluator", level=log_level)
        self.log_level = log_level

        # Components
        self.env = None
        self.navigator = None
        self.token_tracker = get_token_tracker()

        # Data
        self.episodes: List[R2REpisode] = []
        self.connectivity: Dict[str, ConnectivityGraph] = {}

        # Metrics
        self.vln_metrics = VLNMetrics(success_distance=config.success_distance)
        self.token_metrics = TokenMetrics()

        # Results
        self.results: List[R2REpisodeResult] = []

    def initialize(self) -> None:
        """Initialize evaluation components."""
        self.logger.info("Initializing R2R Evaluator...")

        # Load LLM configuration from YAML
        self._load_llm_config()

        # Initialize environment with first scene
        scenes = self._get_scene_paths()
        if scenes:
            first_scene = scenes[0]
        else:
            first_scene = self.config.scene_path or "mock_scene.glb"

        try:
            from environment.habitat_env import HabitatEnv
            self.env = HabitatEnv({
                "scene_id": first_scene,
                "max_episode_steps": self.config.max_steps,
                "sensor_height": 1.5,
                "forward_amount": self.config.forward_amount,
                "turn_angle": self.config.turn_angle,
            })
            self.env.initialize()
            self.logger.info("Habitat environment initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize Habitat: {e}")
            self.env = None

        # Initialize navigator with LLM configuration
        try:
            from core.navigator import VLNNavigator

            # Build navigator config with LLM settings
            nav_config = self._build_navigator_config()

            self.navigator = VLNNavigator(
                config=nav_config,
                log_level=self.log_level,
            )
            self.navigator.initialize()
            self.logger.info(f"VLN Navigator initialized with model_type={self.config.model_type}")
        except Exception as e:
            self.logger.warning(f"Could not initialize navigator: {e}")
            self.navigator = None

        self.logger.info("R2R Evaluator initialization complete")

    def _load_llm_config(self) -> None:
        """Load LLM configuration from YAML file."""
        config_path = Path(self.config.config_file)
        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Extract LLM configuration
            if 'model' in config_data:
                model_config = config_data['model']

                # Get default model settings
                if 'qwen' in model_config:
                    qwen_config = model_config['qwen']
                    self.config.model_name = qwen_config.get('model_name', self.config.model_name)
                    self.config.api_key = qwen_config.get('api_key', '')
                    self.config.api_base = qwen_config.get('base_url', '')
                    self.config.llm_config = qwen_config

            self.logger.info(f"Loaded LLM config: model={self.config.model_name}")

        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")

    def _build_navigator_config(self) -> Dict[str, Any]:
        """Build navigator configuration with LLM settings."""
        config = {
            "model_type": self.config.model_type,
        }

        if self.config.model_type == "llm":
            # Add LLM model configuration
            config["llm_model"] = {
                "model_name": self.config.model_name,
                "api_key": self.config.api_key,
                "api_base": self.config.api_base,
                "max_tokens": self.config.llm_config.get("max_tokens", 2000),
                "temperature": self.config.llm_config.get("temperature", 0.7),
            }

            # Add classifier config
            config["classifier"] = {
                "use_llm_fallback": True,
                "cache_results": True,
            }

            # Add supernet config
            config["supernet"] = {
                "adaptive_selection": True,
            }

            # Add agent configs
            config["instruction_agent"] = {
                "enabled": True,
                "model_type": "llm",
                "llm_model": config["llm_model"],
            }

            config["perception_agent"] = {
                "enabled": True,
            }

            config["trajectory_agent"] = {
                "enabled": True,
            }

            config["decision_agent"] = {
                "enabled": True,
                "model_type": "llm",
                "llm_model": config["llm_model"],
            }

        return config

    def _get_scene_paths(self) -> List[str]:
        """Get list of scene paths."""
        if self.config.scene_path:
            return [self.config.scene_path]

        scenes_dir = Path(self.config.scenes_dir)
        if not scenes_dir.exists():
            self.logger.warning(f"Scenes directory not found: {scenes_dir}")
            return []

        return [str(f) for f in scenes_dir.glob("*.glb")]

    def load_data(self) -> None:
        """Load or generate R2R data."""
        if self.config.mode == "mock":
            self._generate_mock_data()
        else:
            self._load_real_data()

    def _generate_mock_data(self) -> None:
        """Generate mock R2R data from scenes."""
        self.logger.info("Generating mock R2R data...")

        generator = MockR2RGenerator(
            min_distance=5.0,
            max_distance=20.0,
        )

        scenes = self._get_scene_paths()
        if not scenes:
            # Generate purely mock data with synthetic points
            self.logger.info("No scenes found, generating synthetic mock data...")
            points = [(float(x), 0.0, float(z))
                     for x in range(-10, 11, 2)
                     for z in range(-10, 11, 2)]

            episodes, connectivity = generator.generate_from_navigable_points(
                scene_id="mock_scene",
                navigable_points=points,
                num_episodes=self.config.num_episodes,
            )
            self.episodes = episodes
            self.connectivity["mock_scene"] = connectivity
            self.logger.info(f"Generated {len(self.episodes)} synthetic mock episodes")
            return

        # Generate from Habitat scenes
        all_episodes = []
        episodes_per_scene = max(1, self.config.num_episodes // len(scenes))

        for scene_path in scenes:
            self.logger.info(f"Processing scene: {scene_path}")

            # Build connectivity for this scene
            conn = ConnectivityGraph()

            # Try to build from pathfinder, fall back to synthetic points
            if self.env and self.env._sim:
                try:
                    conn.build_from_habitat_pathfinder(self.env._sim, num_samples=50)
                except Exception as e:
                    self.logger.warning(f"Could not build connectivity from pathfinder: {e}")
                    conn = None
            else:
                conn = None

            # If connectivity failed, use synthetic points
            if not conn or len(conn) < 2:
                self.logger.info("Using synthetic points for connectivity")
                conn = ConnectivityGraph()
                points = [(float(x), 0.0, float(z))
                         for x in range(-8, 9, 2)
                         for z in range(-8, 9, 2)]
                conn.build_from_navigable_points(points, distance_threshold=2.5)

            vp_ids = list(conn.nodes.keys())
            if len(vp_ids) < 2:
                self.logger.warning(f"Not enough nodes in connectivity for {scene_path}")
                continue

            # Generate episodes for this scene
            for _ in range(episodes_per_scene):
                episode = self._create_mock_episode(scene_path, conn, vp_ids)
                if episode:
                    all_episodes.append(episode)

            self.connectivity[scene_path] = conn

        self.episodes = all_episodes
        self.logger.info(f"Generated {len(self.episodes)} mock episodes from {len(scenes)} scenes")

    def _create_mock_episode(
        self,
        scene_id: str,
        connectivity: ConnectivityGraph,
        viewpoint_ids: List[str],
    ) -> Optional[R2REpisode]:
        """Create a single mock episode."""
        if len(viewpoint_ids) < 2:
            return None

        # Select random start and goal
        start_vp = random.choice(viewpoint_ids)
        goal_vp = random.choice([vp for vp in viewpoint_ids if vp != start_vp])

        # Get path
        path = connectivity.shortest_path(start_vp, goal_vp) or [start_vp, goal_vp]

        # Build trajectory
        trajectory = [[vp, 0.0, 0.0] for vp in path]

        # Get positions
        start_pos = connectivity.get_position(start_vp) or (0, 0, 0)
        goal_pos = connectivity.get_position(goal_vp) or (0, 0, 0)

        # Calculate geodesic distance
        geo_dist = connectivity.geodesic_distance(start_vp, goal_vp)

        # Generate instruction
        instructions = [
            f"Navigate from the starting point to the destination.",
            f"Walk towards the goal location.",
            f"Find your way to the target.",
        ]

        # Build reference path
        ref_path = []
        for vp in path:
            pos = connectivity.get_position(vp)
            if pos:
                ref_path.append(pos)

        episode = R2REpisode(
            episode_id=len(self.episodes),
            scene_id=scene_id,
            trajectory=trajectory,
            instructions=instructions,
            start_position=start_pos,
            goal_position=goal_pos,
            geodesic_distance=geo_dist,
            reference_path=ref_path,
        )

        return episode

    def _load_real_data(self) -> None:
        """Load real R2R data."""
        self.logger.info("Loading real R2R data...")

        if not self.config.data_path:
            self.logger.error("No data path specified for real mode")
            return

        dataset = R2RDataset(
            data_path=self.config.data_path,
            connectivity_path=self.config.connectivity_path,
        )

        self.episodes = dataset.load_episodes()

        # Load connectivity for each scene
        for ep in self.episodes:
            if ep.scene_id not in self.connectivity:
                conn = ConnectivityGraph()
                conn.load_from_json(
                    str(Path(self.config.connectivity_path) / f"{ep.scene_id}_connectivity.json")
                )
                self.connectivity[ep.scene_id] = conn

                # Compute positions using ConnectivityGraph methods
                if ep.trajectory:
                    start_vp = ep.trajectory[0][0]
                    goal_vp = ep.trajectory[-1][0]

                    start_pos = conn.get_position(start_vp)
                    goal_pos = conn.get_position(goal_vp)

                    if start_pos:
                        ep.start_position = start_pos
                    if goal_pos:
                        ep.goal_position = goal_pos

                    # Calculate geodesic distance
                    ep.geodesic_distance = conn.geodesic_distance(start_vp, goal_vp)

        self.logger.info(f"Loaded {len(self.episodes)} episodes from R2R data")

    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete R2R evaluation."""
        self.logger.info(f"Starting R2R evaluation with {len(self.episodes)} episodes")

        for i, episode in enumerate(self.episodes[:self.config.num_episodes]):
            self.logger.info(f"Episode {i+1}/{min(len(self.episodes), self.config.num_episodes)}")
            result = self._run_episode(episode)
            self.results.append(result)

        return self._compile_results()

    def _run_episode(self, episode: R2REpisode) -> R2REpisodeResult:
        """Run a single R2R episode."""
        # Start token tracking
        self.token_tracker.start_task(
            instruction=episode.get_instruction(),
            task_type="R2R",
        )

        # Initialize tracking
        trajectory = []
        steps = 0
        success = False
        oracle_success = False
        min_distance = float('inf')

        try:
            if self.env and self.env._sim:
                # Set agent state
                self.env.set_agent_state(
                    episode.start_position,
                    episode.heading * 180 / 3.14159,  # Convert to degrees
                )
                trajectory.append(episode.start_position)

                # Get context and set instruction
                context = self.env.reset()
                context.instruction = episode.get_instruction()

                # Navigation loop
                done = False
                while not done and steps < self.config.max_steps:
                    # Get action
                    if self.navigator:
                        action = self.navigator.navigate(context)
                    else:
                        # Random action
                        from core.action import Action, ActionType
                        action = Action(random.choice([
                            ActionType.MOVE_FORWARD,
                            ActionType.TURN_LEFT,
                            ActionType.TURN_RIGHT,
                        ]))

                    # Check stop
                    if action.action_type.name == "STOP":
                        break

                    # Execute
                    context, reward, done, info = self.env.step(action)

                    # Track trajectory
                    pos = self.env.get_agent_position()
                    trajectory.append(pos)
                    steps += 1

                    # Check distance to goal
                    dist = self._calculate_distance(pos, episode.goal_position)
                    min_distance = min(min_distance, dist)

                    if dist <= self.config.success_distance:
                        success = True
                        oracle_success = True
                        break

            else:
                # Mock mode simulation
                trajectory = self._simulate_trajectory(
                    episode.start_position,
                    episode.goal_position,
                )
                steps = len(trajectory)
                min_distance = self._calculate_distance(trajectory[-1], episode.goal_position)
                success = min_distance <= self.config.success_distance
                oracle_success = success

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} failed: {e}")
            trajectory = [episode.start_position]
            steps = 0

        # End token tracking
        task_summary = self.token_tracker.end_task()

        # Calculate metrics
        final_pos = trajectory[-1] if trajectory else episode.start_position
        distance_to_goal = self._calculate_distance(final_pos, episode.goal_position)

        # Trajectory length
        trajectory_length = sum(
            self._calculate_distance(trajectory[i-1], trajectory[i])
            for i in range(1, len(trajectory))
        ) if len(trajectory) > 1 else 0

        # SPL
        spl = episode.geodesic_distance / trajectory_length if success and trajectory_length > 0 else 0

        # nDTW
        ndtw = self._calculate_ndtw(trajectory, episode.reference_path)

        # Token usage
        token_usage = {}
        if task_summary:
            # task_summary is a TaskTokenSummary dataclass
            token_usage = {
                "total": task_summary.total_tokens,
                "input": task_summary.total_input_tokens,
                "output": task_summary.total_output_tokens,
                "api_calls": task_summary.num_api_calls,
            }

        # Record metrics
        self.vln_metrics.add_episode(
            episode_id=episode.episode_id,
            trajectory=trajectory,
            goal_position=episode.goal_position,
        )

        return R2REpisodeResult(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            success=success,
            spl=spl,
            oracle_success=oracle_success,
            nDTW=ndtw,
            trajectory_length=trajectory_length,
            shortest_path_length=episode.geodesic_distance,
            steps=steps,
            distance_to_goal=distance_to_goal,
            instruction=episode.get_instruction(),
            token_usage=token_usage,
            trajectory=trajectory,
        )

    def _simulate_trajectory(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> List[Tuple[float, float, float]]:
        """Simulate a trajectory from start to goal."""
        trajectory = [start]
        current = list(start)

        for _ in range(random.randint(10, 30)):
            alpha = 0.1
            noise = [random.uniform(-0.3, 0.3) for _ in range(3)]

            current = [
                current[0] + (goal[0] - current[0]) * alpha + noise[0],
                current[1] + (goal[1] - current[1]) * alpha + noise[1],
                current[2] + (goal[2] - current[2]) * alpha + noise[2],
            ]
            trajectory.append(tuple(current))

        return trajectory

    def _calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance."""
        return ((pos1[0] - pos2[0])**2 +
                (pos1[1] - pos2[1])**2 +
                (pos1[2] - pos2[2])**2)**0.5

    def _calculate_ndtw(
        self,
        trajectory: List[Tuple[float, float, float]],
        reference: List[Tuple[float, float, float]],
    ) -> float:
        """Calculate normalized DTW."""
        if not trajectory or not reference:
            return 0.0

        import math

        n, m = len(trajectory), len(reference)
        dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._calculate_distance(trajectory[i-1], reference[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

        max_len = max(n, m)
        return math.exp(-dtw[n][m] / max_len) if max_len > 0 else 0.0

    def _compile_results(self) -> Dict[str, Any]:
        """Compile all results."""
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
            "ndtw": sum(r.nDTW for r in self.results) / total,
            "avg_distance_to_goal": sum(r.distance_to_goal for r in self.results) / total,
            "avg_steps": sum(r.steps for r in self.results) / total,
        }

        # By scene
        by_scene = {}
        for r in self.results:
            if r.scene_id not in by_scene:
                by_scene[r.scene_id] = {"count": 0, "successes": 0, "spl_sum": 0}
            by_scene[r.scene_id]["count"] += 1
            if r.success:
                by_scene[r.scene_id]["successes"] += 1
            by_scene[r.scene_id]["spl_sum"] += r.spl

        for scene in by_scene:
            data = by_scene[scene]
            data["success_rate"] = data["successes"] / data["count"]
            data["spl"] = data["spl_sum"] / data["count"]

        # Token summary
        total_tokens = sum(r.token_usage.get("total", 0) for r in self.results)
        token_summary = {
            "total_tokens": total_tokens,
            "avg_per_episode": total_tokens / total if total > 0 else 0,
        }

        return {
            "summary": summary,
            "token_summary": token_summary,
            "by_scene": by_scene,
            "episodes": [r.to_dict() for r in self.results],
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
        }

    def close(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="R2R VLN Evaluation with Multi-Agent LLM Navigation")

    # Mode arguments
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "real"],
                        help="Evaluation mode: mock (synthetic data) or real (R2R dataset)")
    parser.add_argument("--scene", type=str, default="", help="Single scene path (.glb)")
    parser.add_argument("--scenes", type=str, default="data/scene_datasets/habitat-test-scenes",
                        help="Directory containing scene files")

    # R2R data arguments
    parser.add_argument("--data-path", type=str, default="", help="R2R JSON data path")
    parser.add_argument("--connectivity", type=str, default="", help="Connectivity directory")

    # Episode arguments
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to evaluate")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--success-distance", type=float, default=3.0, help="Success distance threshold (meters)")

    # Model arguments
    parser.add_argument("--model", type=str, default="llm", choices=["llm", "mock"],
                        help="Model type: llm (real LLM) or mock (random testing)")
    parser.add_argument("--model-name", type=str, default="qwen3.5-plus",
                        help="LLM model name (e.g., qwen3.5-plus, gpt-4)")
    parser.add_argument("--config-file", type=str, default="configs/qwen_config.yaml",
                        help="Path to LLM configuration YAML file")

    # Output arguments
    parser.add_argument("--output", type=str, default="results_r2r.json", help="Output results file")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = R2REvaluationConfig(
        mode=args.mode,
        scene_path=args.scene,
        scenes_dir=args.scenes,
        data_path=args.data_path,
        connectivity_path=args.connectivity,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        success_distance=args.success_distance,
        model_type=args.model,
        model_name=args.model_name,
        config_file=args.config_file,
        output=args.output,
        verbose=args.verbose,
    )

    log_level = "DEBUG" if args.verbose else args.log_level

    print("=" * 60)
    print("R2R VLN Evaluation with Multi-Agent LLM Navigation")
    print("=" * 60)
    print(f"Mode: {config.mode}")
    print(f"Model Type: {config.model_type}")
    print(f"Model Name: {config.model_name}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Success Distance: {config.success_distance}m")
    print("=" * 60)

    evaluator = R2REvaluator(config, log_level=log_level)
    evaluator.initialize()
    evaluator.load_data()
    results = evaluator.run_evaluation()

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = results.get("summary", {})
    print(f"Episodes: {summary.get('num_episodes', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
    print(f"SPL: {summary.get('spl', 0):.3f}")
    print(f"Oracle Success: {summary.get('oracle_success_rate', 0)*100:.1f}%")
    print(f"nDTW: {summary.get('ndtw', 0):.3f}")
    print(f"Avg Distance to Goal: {summary.get('avg_distance_to_goal', 0):.2f}m")

    print("\nBy Scene:")
    for scene, data in results.get("by_scene", {}).items():
        scene_name = Path(scene).stem
        print(f"  {scene_name}: SR={data['success_rate']*100:.1f}%, SPL={data['spl']:.3f}")

    print("=" * 60)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    evaluator.close()
    return results


if __name__ == "__main__":
    main()