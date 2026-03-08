#!/usr/bin/env python3
"""Training script for VLN navigation system."""

import argparse
import logging
from typing import Dict, Any

from configs.default_config import load_config
from core.navigator import VLNNavigator
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VLN Navigator")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Scene ID for training",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
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
        default="output/",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint to resume from",
    )

    return parser.parse_args()


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        args: Command line arguments

    Returns:
        Training results
    """
    logger = setup_logger("Train", level=args.log_level)
    logger.info("Starting VLN Navigator training...")

    # Load config
    config = load_config(args.config)

    if args.scene:
        config["scene_id"] = args.scene

    # Initialize navigator
    navigator = VLNNavigator(config=config, log_level=args.log_level)
    navigator.initialize()

    # Training loop
    results = {
        "episodes": 0,
        "successes": 0,
        "total_steps": 0,
        "total_distance": 0.0,
    }

    for episode in range(args.episodes):
        logger.info(f"Episode {episode + 1}/{args.episodes}")

        # Reset navigator
        navigator.reset()

        # Run episode
        episode_result = run_episode(navigator, config)

        # Update results
        results["episodes"] += 1
        if episode_result["success"]:
            results["successes"] += 1
        results["total_steps"] += episode_result["steps"]
        results["total_distance"] += episode_result["distance"]

        # Log progress
        if (episode + 1) % 100 == 0:
            success_rate = results["successes"] / results["episodes"]
            logger.info(f"Success rate: {success_rate:.2%}")

    # Final results
    success_rate = results["successes"] / results["episodes"] if results["episodes"] > 0 else 0

    logger.info(f"Training complete!")
    logger.info(f"Final success rate: {success_rate:.2%}")
    logger.info(f"Average steps: {results['total_steps'] / results['episodes']:.1f}")

    return results


def run_episode(navigator: VLNNavigator, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single training episode.

    Args:
        navigator: VLNNavigator instance
        config: Configuration

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
    distance = 0.0

    while not done and steps < max_steps:
        # Set context
        navigator.set_instruction(context.instruction)
        navigator.set_position(context.position, context.rotation)

        # Get action
        action = navigator.navigate()

        # Execute action
        context, reward, done, info = env.step(action)

        steps += 1

    # Calculate results
    success = context.confidence > 0.8  # Simplified success criterion

    env.close()

    return {
        "success": success,
        "steps": steps,
        "distance": distance,
    }


if __name__ == "__main__":
    args = parse_args()
    train(args)