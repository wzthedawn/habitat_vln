#!/usr/bin/env python3
"""Main evaluation script for VLN system in Habitat environment."""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.metrics import VLNMetrics, TokenMetrics, EpisodeMetrics
from utils.token_tracker import TokenTracker, get_token_tracker
from utils.episode_generator import EpisodeGenerator, EpisodeDataset, VLNEpisode


@dataclass
class EvaluationConfig:
    """Configuration for VLN evaluation."""
    # Scene settings
    scenes_dir: str = "data/scene_datasets/habitat-test-scenes"
    scene_filter: str = "*.glb"

    # Episode settings
    num_episodes: int = 20
    min_distance: float = 5.0
    max_distance: float = 20.0
    success_distance: float = 3.0

    # Navigation settings
    max_steps: int = 100
    forward_amount: float = 0.25
    turn_angle: float = 15.0

    # Model settings
    model_type: str = "mock"  # mock, openai, anthropic
    model_name: str = "gpt-4"

    # Task type distribution
    task_type_distribution: Dict[str, float] = None

    def __post_init__(self):
        if self.task_type_distribution is None:
            self.task_type_distribution = {
                "Type-0": 0.1,
                "Type-1": 0.3,
                "Type-2": 0.25,
                "Type-3": 0.2,
                "Type-4": 0.15,
            }


@dataclass
class EpisodeResult:
    """Result of a single episode evaluation."""
    episode_id: int
    success: bool
    spl: float
    oracle_success: bool
    trajectory_length: float
    shortest_path_length: float
    steps: int
    distance_to_goal: float
    task_type: str
    instruction: str
    token_usage: Dict[str, int]
    trajectory: List[Tuple[float, float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VLNEvaluator:
    """Evaluator for VLN navigation system."""

    def __init__(
        self,
        config: EvaluationConfig,
        log_level: str = "INFO",
    ):
        """Initialize evaluator."""
        self.config = config
        self.logger = setup_logger("VLNEvaluator", level=log_level)
        self.log_level = log_level

        # Components
        self.env = None
        self.navigator = None
        self.token_tracker = get_token_tracker()

        # Metrics
        self.vln_metrics = VLNMetrics(success_distance=config.success_distance)
        self.token_metrics = TokenMetrics()

        # Results
        self.results: List[EpisodeResult] = []

    def initialize(self) -> None:
        """Initialize evaluation components."""
        self.logger.info("Initializing VLN Evaluator...")

        # Initialize environment
        try:
            from environment.habitat_env import HabitatEnv

            # Get first scene for initialization
            scenes = self._get_scene_paths()
            if scenes:
                self.env = HabitatEnv({
                    "scene_id": scenes[0],
                    "max_episode_steps": self.config.max_steps,
                    "sensor_height": 1.5,
                    "forward_amount": self.config.forward_amount,
                    "turn_angle": self.config.turn_angle,
                })
                self.env.initialize()
                self.logger.info("Habitat environment initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize Habitat environment: {e}")
            self.logger.info("Running in mock mode")

        # Initialize navigator
        try:
            from core.navigator import VLNNavigator
            self.navigator = VLNNavigator(
                config={"model_type": self.config.model_type},
                log_level=self.log_level,
            )
            self.navigator.initialize()
            self.logger.info("VLN Navigator initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize navigator: {e}")
            self.navigator = None

        self.logger.info("VLN Evaluator initialization complete")

    def _get_scene_paths(self) -> List[str]:
        """Get list of scene paths."""
        scenes_dir = Path(self.config.scenes_dir)
        if not scenes_dir.exists():
            self.logger.warning(f"Scenes directory not found: {scenes_dir}")
            return []

        scene_files = list(scenes_dir.glob(self.config.scene_filter))
        return [str(f) for f in scene_files]

    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation."""
        self.logger.info(f"Starting evaluation with {self.config.num_episodes} episodes")

        # Get scenes
        scenes = self._get_scene_paths()
        if not scenes:
            self.logger.warning("No scenes found, using mock evaluation")
            return self._run_mock_evaluation()

        # Generate episodes
        episode_generator = EpisodeGenerator(
            min_distance=self.config.min_distance,
            max_distance=self.config.max_distance,
            success_distance=self.config.success_distance,
        )

        # Distribute episodes across scenes
        episodes_per_scene = max(1, self.config.num_episodes // len(scenes))
        all_episodes = []

        for scene_path in scenes:
            sim = self._get_simulator_for_scene(scene_path)
            episodes = episode_generator.generate_episodes(
                scene_path=scene_path,
                num_episodes=episodes_per_scene,
                task_type_distribution=self.config.task_type_distribution,
                sim=sim,
            )
            all_episodes.extend(episodes)

        self.logger.info(f"Generated {len(all_episodes)} episodes")

        # Run episodes
        for i, episode in enumerate(all_episodes[:self.config.num_episodes]):
            self.logger.info(f"Running episode {i+1}/{min(len(all_episodes), self.config.num_episodes)}")
            result = self._run_episode(episode)
            self.results.append(result)

        # Compile results
        return self._compile_results()

    def _get_simulator_for_scene(self, scene_path: str):
        """Get simulator instance for a scene."""
        if self.env is not None and hasattr(self.env, '_sim'):
            # Check if we need to load new scene
            if hasattr(self.env._sim, 'config') and hasattr(self.env._sim.config, 'sim_cfg'):
                current_scene = self.env._sim.config.sim_cfg.scene_id
                if current_scene != scene_path:
                    # Reinitialize with new scene
                    self.env.close()
                    from environment.habitat_env import HabitatEnv
                    self.env = HabitatEnv({
                        "scene_id": scene_path,
                        "max_episode_steps": self.config.max_steps,
                    })
                    self.env.initialize()
            return self.env._sim
        return None

    def _run_episode(self, episode: VLNEpisode) -> EpisodeResult:
        """Run a single navigation episode."""
        # Start token tracking
        task_id = self.token_tracker.start_task(
            instruction=episode.instruction,
            task_type=episode.task_type,
        )

        # Reset environment to episode start
        trajectory = []
        steps = 0
        success = False
        done = False

        try:
            if self.env is not None and self.env._sim is not None:
                # Set agent to start position
                self.env.set_agent_state(
                    episode.start_position,
                    episode.start_rotation,
                )
                trajectory.append(episode.start_position)

                # Navigation loop
                context = self.env.reset()
                context.instruction = episode.instruction

                while not done and steps < self.config.max_steps:
                    # Get action from navigator
                    if self.navigator is not None:
                        action = self.navigator.navigate(context)
                    else:
                        # Mock action: random movement
                        from core.action import Action, ActionType
                        import random
                        action_type = random.choice([
                            ActionType.MOVE_FORWARD,
                            ActionType.TURN_LEFT,
                            ActionType.TURN_RIGHT,
                        ])
                        action = Action(action_type=action_type)

                    # Check for stop
                    if action.action_type.name == "STOP":
                        break

                    # Execute action
                    context, reward, done, info = self.env.step(action)

                    # Record trajectory
                    position = self.env.get_agent_position()
                    trajectory.append(position)
                    steps += 1

                    # Check success
                    dist_to_goal = self.env.get_geodesic_distance(
                        position, episode.goal_position
                    )
                    if dist_to_goal <= self.config.success_distance:
                        success = True
                        done = True

                final_position = trajectory[-1] if trajectory else episode.start_position

            else:
                # Mock mode - simulate navigation
                trajectory = self._simulate_mock_trajectory(
                    episode.start_position,
                    episode.goal_position,
                )
                steps = len(trajectory)
                final_position = trajectory[-1] if trajectory else episode.start_position

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} failed: {e}")
            trajectory = [episode.start_position]
            steps = 0
            final_position = episode.start_position

        # End token tracking
        task_summary = self.token_tracker.end_task()

        # Calculate metrics
        distance_to_goal = self._calculate_distance(final_position, episode.goal_position)
        success = distance_to_goal <= self.config.success_distance

        trajectory_length = self._calculate_path_length(trajectory)
        shortest_path = episode.geodesic_distance

        spl = shortest_path / trajectory_length if success and trajectory_length > 0 else 0.0

        # Oracle success - check if any point was close enough
        oracle_success = any(
            self._calculate_distance(pos, episode.goal_position) <= self.config.success_distance
            for pos in trajectory
        )

        # Record metrics
        self.vln_metrics.add_episode(
            episode_id=episode.episode_id,
            trajectory=trajectory,
            goal_position=episode.goal_position,
        )

        # Get token usage
        token_usage = {}
        if task_summary:
            token_usage = {
                "total": task_summary.get("summary", {}).get("total_tokens", 0),
                "input": task_summary.get("summary", {}).get("total_input_tokens", 0),
                "output": task_summary.get("summary", {}).get("total_output_tokens", 0),
            }
            by_agent = task_summary.get("by_agent", {})
            for agent, stats in by_agent.items():
                token_usage[f"{agent}_total"] = stats.get("total_tokens", 0)

        return EpisodeResult(
            episode_id=episode.episode_id,
            success=success,
            spl=spl,
            oracle_success=oracle_success,
            trajectory_length=trajectory_length,
            shortest_path_length=shortest_path,
            steps=steps,
            distance_to_goal=distance_to_goal,
            task_type=episode.task_type,
            instruction=episode.instruction,
            token_usage=token_usage,
            trajectory=trajectory,
        )

    def _simulate_mock_trajectory(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> List[Tuple[float, float, float]]:
        """Simulate a mock trajectory from start to goal."""
        import random

        trajectory = [start]
        current = list(start)
        goal_list = list(goal)

        # Simple interpolation with some randomness
        num_steps = random.randint(10, 50)

        for i in range(num_steps):
            # Move towards goal with noise
            alpha = (i + 1) / num_steps
            noise = [random.uniform(-0.5, 0.5) for _ in range(3)]

            new_pos = [
                current[0] + (goal_list[0] - current[0]) * 0.1 + noise[0],
                current[1] + (goal_list[1] - current[1]) * 0.1 + noise[1],
                current[2] + (goal_list[2] - current[2]) * 0.1 + noise[2],
            ]

            trajectory.append(tuple(new_pos))
            current = new_pos

        return trajectory

    def _calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance between positions."""
        import numpy as np
        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def _calculate_path_length(
        self,
        trajectory: List[Tuple[float, float, float]],
    ) -> float:
        """Calculate total path length."""
        if len(trajectory) < 2:
            return 0.0

        length = 0.0
        for i in range(1, len(trajectory)):
            length += self._calculate_distance(trajectory[i-1], trajectory[i])
        return length

    def _run_mock_evaluation(self) -> Dict[str, Any]:
        """Run evaluation in mock mode without Habitat."""
        self.logger.info("Running mock evaluation")

        # Generate mock episodes
        episode_generator = EpisodeGenerator(
            min_distance=self.config.min_distance,
            max_distance=self.config.max_distance,
        )

        mock_scene = "mock_scene.glb"
        episodes = episode_generator.generate_episodes(
            scene_path=mock_scene,
            num_episodes=self.config.num_episodes,
        )

        # Run each episode
        for episode in episodes:
            result = self._run_episode(episode)
            self.results.append(result)

        return self._compile_results()

    def _compile_results(self) -> Dict[str, Any]:
        """Compile all results into summary."""
        if not self.results:
            return {"error": "No results to compile"}

        # Calculate aggregate metrics
        total = len(self.results)
        successes = sum(1 for r in self.results if r.success)
        oracle_successes = sum(1 for r in self.results if r.oracle_success)

        summary = {
            "num_episodes": total,
            "success_rate": successes / total,
            "spl": sum(r.spl for r in self.results) / total,
            "oracle_success_rate": oracle_successes / total,
            "avg_distance_to_goal": sum(r.distance_to_goal for r in self.results) / total,
            "avg_steps": sum(r.steps for r in self.results) / total,
            "avg_trajectory_length": sum(r.trajectory_length for r in self.results) / total,
        }

        # Token summary
        total_tokens = sum(r.token_usage.get("total", 0) for r in self.results)
        token_summary = {
            "total_tokens": total_tokens,
            "avg_per_episode": total_tokens / total if total > 0 else 0,
        }

        # Token by agent
        agent_tokens = {}
        for r in self.results:
            for key, value in r.token_usage.items():
                if key.endswith("_total"):
                    agent_name = key.replace("_total", "")
                    if agent_name not in agent_tokens:
                        agent_tokens[agent_name] = {"total": 0, "count": 0}
                    agent_tokens[agent_name]["total"] += value
                    agent_tokens[agent_name]["count"] += 1

        token_summary["by_agent"] = {
            agent: {
                "total": stats["total"],
                "avg": stats["total"] / stats["count"] if stats["count"] > 0 else 0,
            }
            for agent, stats in agent_tokens.items()
        }

        # Results by task type
        by_task_type = {}
        for r in self.results:
            if r.task_type not in by_task_type:
                by_task_type[r.task_type] = {
                    "count": 0,
                    "successes": 0,
                    "spl_sum": 0,
                }
            by_task_type[r.task_type]["count"] += 1
            if r.success:
                by_task_type[r.task_type]["successes"] += 1
            by_task_type[r.task_type]["spl_sum"] += r.spl

        for task_type in by_task_type:
            data = by_task_type[task_type]
            data["success_rate"] = data["successes"] / data["count"] if data["count"] > 0 else 0
            data["spl"] = data["spl_sum"] / data["count"] if data["count"] > 0 else 0
            del data["successes"]
            del data["spl_sum"]

        return {
            "summary": summary,
            "token_summary": token_summary,
            "by_task_type": by_task_type,
            "episodes": [r.to_dict() for r in self.results],
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
        }

    def close(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLN Navigator in Habitat environment"
    )

    parser.add_argument(
        "--scenes", type=str,
        default="data/scene_datasets/habitat-test-scenes",
        help="Directory containing scene files",
    )
    parser.add_argument(
        "--episodes", type=int,
        default=20,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max-steps", type=int,
        default=100,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--success-distance", type=float,
        default=3.0,
        help="Distance threshold for success",
    )
    parser.add_argument(
        "--min-distance", type=float,
        default=5.0,
        help="Minimum goal distance",
    )
    parser.add_argument(
        "--max-distance", type=float,
        default=20.0,
        help="Maximum goal distance",
    )
    parser.add_argument(
        "--output", type=str,
        default="results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--model", type=str,
        default="mock",
        choices=["mock", "openai", "anthropic"],
        help="Model type to use",
    )
    parser.add_argument(
        "--log-level", type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create config
    config = EvaluationConfig(
        scenes_dir=args.scenes,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        success_distance=args.success_distance,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        model_type=args.model,
    )

    # Create evaluator
    evaluator = VLNEvaluator(
        config=config,
        log_level="DEBUG" if args.verbose else args.log_level,
    )

    # Run evaluation
    print("=" * 60)
    print("VLN Navigation Evaluation")
    print("=" * 60)
    print(f"Scenes: {config.scenes_dir}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Max Steps: {config.max_steps}")
    print(f"Success Distance: {config.success_distance}m")
    print("=" * 60)

    evaluator.initialize()
    results = evaluator.run_evaluation()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    summary = results.get("summary", {})
    print(f"Total Episodes: {summary.get('num_episodes', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
    print(f"SPL: {summary.get('spl', 0):.3f}")
    print(f"Oracle Success: {summary.get('oracle_success_rate', 0)*100:.1f}%")
    print(f"Avg Distance to Goal: {summary.get('avg_distance_to_goal', 0):.2f}m")
    print(f"Avg Steps: {summary.get('avg_steps', 0):.1f}")

    print("\nBy Task Type:")
    for task_type, data in results.get("by_task_type", {}).items():
        print(f"  {task_type}: SR={data.get('success_rate', 0)*100:.1f}%, "
              f"SPL={data.get('spl', 0):.3f}, n={data.get('count', 0)}")

    print("\nToken Usage:")
    token_summary = results.get("token_summary", {})
    print(f"  Total: {token_summary.get('total_tokens', 0):,}")
    print(f"  Avg per Episode: {token_summary.get('avg_per_episode', 0):.1f}")

    by_agent = token_summary.get("by_agent", {})
    for agent, stats in by_agent.items():
        print(f"    {agent}: {stats.get('total', 0):,} (avg: {stats.get('avg', 0):.1f})")

    print("=" * 60)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # Cleanup
    evaluator.close()

    return results


if __name__ == "__main__":
    main()