#!/usr/bin/env python3
"""Evaluation script for VLN navigation system."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.default_config import load_config
from core.navigator import VLNNavigator
from core.action import Action, ActionType
from environment.habitat_env import HabitatEnv
from utils.logger import setup_logger
from utils.metrics import VLNMetrics, TokenMetrics
from utils.token_tracker import get_token_tracker
from utils.episode_generator import EpisodeGenerator, VLNEpisode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate VLN Navigator")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="data/scene_datasets/habitat-test-scenes",
        help="Directory containing scene files",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Single scene ID for evaluation (optional)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--success-distance",
        type=float,
        default=3.0,
        help="Distance threshold for success",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=5.0,
        help="Minimum goal distance for episodes",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=20.0,
        help="Maximum goal distance for episodes",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def get_scene_paths(scenes_dir: str, scene_filter: str = "*.glb") -> List[str]:
    """Get list of scene file paths."""
    scenes_path = Path(scenes_dir)
    if not scenes_path.exists():
        logging.warning(f"Scenes directory not found: {scenes_dir}")
        return []

    scene_files = list(scenes_path.glob(scene_filter))
    return [str(f) for f in scene_files]


def generate_episodes_for_scenes(
    scenes: List[str],
    num_episodes_per_scene: int,
    min_distance: float,
    max_distance: float,
    sim=None,
) -> List[VLNEpisode]:
    """Generate episodes for multiple scenes."""
    generator = EpisodeGenerator(
        min_distance=min_distance,
        max_distance=max_distance,
    )

    all_episodes = []
    for scene_path in scenes:
        episodes = generator.generate_episodes(
            scene_path=scene_path,
            num_episodes=num_episodes_per_scene,
            sim=sim,
        )
        all_episodes.extend(episodes)

    return all_episodes


def run_episode(
    navigator: VLNNavigator,
    env: HabitatEnv,
    episode: VLNEpisode,
    max_steps: int,
    success_distance: float,
    token_tracker,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode.

    Args:
        navigator: VLNNavigator instance
        env: HabitatEnv instance
        episode: VLNEpisode to run
        max_steps: Maximum number of steps
        success_distance: Distance threshold for success
        token_tracker: TokenTracker instance

    Returns:
        Episode results dictionary
    """
    # Start token tracking
    task_id = token_tracker.start_task(
        instruction=episode.instruction,
        task_type=episode.task_type,
    )

    # Set agent to start position
    env.set_agent_state(episode.start_position, episode.start_rotation)

    # Reset environment
    context = env.reset()
    context.instruction = episode.instruction

    # Initialize tracking
    trajectory = [episode.start_position]
    steps = 0
    done = False
    success = False
    oracle_success = False
    min_distance_to_goal = float('inf')

    # Navigation loop
    while not done and steps < max_steps:
        # Update navigator context
        navigator.set_instruction(context.instruction)
        navigator.set_position(context.position, context.rotation)

        # Get action from navigator
        action = navigator.navigate(context)

        # Check for stop action
        if action.action_type == ActionType.STOP:
            done = True
            break

        # Execute action in environment
        context, reward, step_done, info = env.step(action)

        # Record trajectory
        position = env.get_agent_position()
        trajectory.append(position)
        steps += 1

        # Calculate distance to goal
        distance_to_goal = env.get_geodesic_distance(position, episode.goal_position)
        min_distance_to_goal = min(min_distance_to_goal, distance_to_goal)

        # Check for success
        if distance_to_goal <= success_distance:
            success = True
            oracle_success = True
            done = True

        # Check for oracle success (any point on trajectory)
        if distance_to_goal <= success_distance:
            oracle_success = True

        if step_done:
            done = True

    # End token tracking
    task_summary = token_tracker.end_task()

    # Calculate final distance to goal
    final_position = trajectory[-1] if trajectory else episode.start_position
    distance_to_goal = env.get_geodesic_distance(final_position, episode.goal_position)

    # Calculate trajectory length
    trajectory_length = sum(
        ((trajectory[i][0] - trajectory[i-1][0])**2 +
         (trajectory[i][1] - trajectory[i-1][1])**2 +
         (trajectory[i][2] - trajectory[i-1][2])**2)**0.5
        for i in range(1, len(trajectory))
    ) if len(trajectory) > 1 else 0

    # Calculate SPL
    if success and trajectory_length > 0:
        spl = episode.geodesic_distance / trajectory_length
    else:
        spl = 0.0

    # Compile token usage
    token_usage = {}
    if task_summary:
        token_usage = {
            "total": task_summary.get("summary", {}).get("total_tokens", 0),
            "by_agent": task_summary.get("by_agent", {}),
        }

    return {
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,
        "success": success,
        "oracle_success": oracle_success,
        "spl": spl,
        "steps": steps,
        "trajectory_length": trajectory_length,
        "shortest_path_length": episode.geodesic_distance,
        "distance_to_goal": distance_to_goal,
        "min_distance_to_goal": min_distance_to_goal,
        "task_type": episode.task_type,
        "instruction": episode.instruction,
        "trajectory": trajectory,
        "goal_position": list(episode.goal_position),
        "start_position": list(episode.start_position),
        "token_usage": token_usage,
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main evaluation function.

    Args:
        args: Command line arguments

    Returns:
        Evaluation results
    """
    log_level = "DEBUG" if args.verbose else args.log_level
    logger = setup_logger("Eval", level=log_level)
    logger.info("Starting VLN Navigator evaluation...")

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    config["max_steps"] = args.max_steps
    config["success_distance"] = args.success_distance
    if args.scene:
        config["scene_id"] = args.scene

    # Initialize navigator
    navigator = VLNNavigator(config=config, log_level=log_level)
    navigator.initialize()

    # Initialize token tracker
    token_tracker = get_token_tracker()

    # Initialize metrics
    vln_metrics = VLNMetrics(success_distance=args.success_distance)
    token_metrics = TokenMetrics()

    # Get scene paths
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = get_scene_paths(args.scenes)

    if not scenes:
        logger.warning("No scenes found, running mock evaluation")
        scenes = ["mock_scene.glb"]

    logger.info(f"Found {len(scenes)} scenes: {scenes}")

    # Initialize environment with first scene
    env = HabitatEnv({
        "scene_id": scenes[0],
        "max_episode_steps": args.max_steps,
        "sensor_height": 1.5,
    })
    env.initialize()

    # Generate episodes
    episodes_per_scene = max(1, args.episodes // len(scenes))
    all_episodes = generate_episodes_for_scenes(
        scenes=scenes,
        num_episodes_per_scene=episodes_per_scene,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        sim=env._sim if hasattr(env, '_sim') else None,
    )

    logger.info(f"Generated {len(all_episodes)} episodes")

    # Run evaluation episodes
    episode_results: List[Dict[str, Any]] = []

    for i, episode in enumerate(all_episodes[:args.episodes]):
        logger.info(f"Evaluating episode {i + 1}/{min(len(all_episodes), args.episodes)}")

        # Update scene if needed
        if env.scene_id != episode.scene_id:
            env.close()
            env = HabitatEnv({
                "scene_id": episode.scene_id,
                "max_episode_steps": args.max_steps,
            })
            env.initialize()

        # Run episode
        result = run_episode(
            navigator=navigator,
            env=env,
            episode=episode,
            max_steps=args.max_steps,
            success_distance=args.success_distance,
            token_tracker=token_tracker,
        )

        # Record metrics
        vln_metrics.add_episode(
            episode_id=episode.episode_id,
            trajectory=result["trajectory"],
            goal_position=tuple(result["goal_position"]),
        )

        # Record token usage
        if "token_usage" in result:
            total_tokens = result["token_usage"].get("total", 0)
            token_metrics.record("total", total_tokens)

            by_agent = result["token_usage"].get("by_agent", {})
            for agent, stats in by_agent.items():
                token_metrics.record(f"agent_{agent}", stats.get("total", 0))

        episode_results.append(result)

        # Log progress
        if args.verbose or (i + 1) % 5 == 0:
            logger.info(
                f"Episode {i+1}: success={result['success']}, "
                f"SPL={result['spl']:.3f}, steps={result['steps']}"
            )

    # Get aggregate metrics
    aggregate_metrics = vln_metrics.get_aggregate_metrics()
    token_summary = token_metrics.get_summary()

    # Calculate summary by task type
    by_task_type = {}
    for result in episode_results:
        task_type = result.get("task_type", "unknown")
        if task_type not in by_task_type:
            by_task_type[task_type] = {
                "count": 0,
                "successes": 0,
                "spl_sum": 0,
                "trajectory_lengths": [],
            }
        by_task_type[task_type]["count"] += 1
        if result["success"]:
            by_task_type[task_type]["successes"] += 1
        by_task_type[task_type]["spl_sum"] += result["spl"]
        by_task_type[task_type]["trajectory_lengths"].append(result["trajectory_length"])

    # Calculate rates
    for task_type in by_task_type:
        data = by_task_type[task_type]
        data["success_rate"] = data["successes"] / data["count"] if data["count"] > 0 else 0
        data["spl"] = data["spl_sum"] / data["count"] if data["count"] > 0 else 0
        data["avg_trajectory_length"] = sum(data["trajectory_lengths"]) / len(data["trajectory_lengths"]) if data["trajectory_lengths"] else 0

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VLN Evaluation Summary")
    logger.info("=" * 60)
    logger.info(vln_metrics.get_summary_string())

    # Token usage
    logger.info("\nToken Usage:")
    for category, stats in token_summary.items():
        logger.info(f"  {category}: total={stats['total']}, avg={stats['average']:.1f}")

    # By task type
    logger.info("\nResults by Task Type:")
    for task_type, data in by_task_type.items():
        logger.info(
            f"  {task_type}: SR={data['success_rate']*100:.1f}%, "
            f"SPL={data['spl']:.3f}, n={data['count']}"
        )

    logger.info("=" * 60)

    # Visualizations
    if args.visualize:
        generate_visualizations(episode_results, args.output)

    # Save results
    results = {
        "summary": aggregate_metrics,
        "token_summary": token_summary,
        "by_task_type": by_task_type,
        "episodes": episode_results,
        "config": {
            "scenes": scenes,
            "num_episodes": args.episodes,
            "max_steps": args.max_steps,
            "success_distance": args.success_distance,
        },
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")

    # Cleanup
    env.close()

    return results


def run_episode(
    navigator: VLNNavigator,
    config: Dict[str, Any],
    episode_id: int,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode.

    Args:
        navigator: VLNNavigator instance
        config: Configuration
        episode_id: Episode identifier

    Returns:
        Episode results
    """
    from environment.habitat_env import HabitatEnv
    from core.context import NavContextBuilder

    # Initialize environment
    env = HabitatEnv(config)
    env.initialize()

    # Reset environment
    context = env.reset()

    # Episode loop
    max_steps = config.get("navigation", {}).get("max_steps", 500)
    done = False
    steps = 0
    trajectory = [context.position]

    while not done and steps < max_steps:
        # Set context
        navigator.set_instruction(context.instruction)
        navigator.set_position(context.position, context.rotation)

        # Get action
        action = navigator.navigate()

        # Execute action
        context, reward, done, info = env.step(action)

        trajectory.append(context.position)
        steps += 1

    # Get results
    success = context.confidence > 0.8

    env.close()

    return {
        "episode_id": episode_id,
        "success": success,
        "steps": steps,
        "trajectory": trajectory,
        "instruction": context.instruction,
        "goal_position": None,  # Would be set from environment
        "distance_to_goal": context.confidence,  # Simplified
    }


def generate_visualizations(
    results: List[Dict[str, Any]],
    output_prefix: str,
) -> None:
    """
    Generate visualization plots.

    Args:
        results: List of episode results
        output_prefix: Output file prefix
    """
    from utils.visualization import TrajectoryVisualizer, MetricsVisualizer

    # Create visualizers
    traj_viz = TrajectoryVisualizer()
    metrics_viz = MetricsVisualizer()

    # Plot trajectories for first few episodes
    for i, result in enumerate(results[:5]):
        if result["trajectory"]:
            traj_viz.visualize_trajectory(
                trajectory=result["trajectory"],
                save_path=f"{output_prefix}_traj_{i}.png",
            )

    # Plot success rate by task type
    task_results = {}
    for result in results:
        # Simplified task type classification
        task_type = "Type-1"  # Default
        if task_type not in task_results:
            task_results[task_type] = {"successes": 0, "total": 0}
        task_results[task_type]["total"] += 1
        if result["success"]:
            task_results[task_type]["successes"] += 1

    for task_type in task_results:
        total = task_results[task_type]["total"]
        successes = task_results[task_type]["successes"]
        task_results[task_type]["success_rate"] = successes / total if total > 0 else 0

    metrics_viz.plot_success_rate_by_task_type(
        task_results,
        save_path=f"{output_prefix}_success_rate.png",
    )


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)