#!/usr/bin/env python3
"""Inference script for VLN navigation system."""

import argparse
import json
import logging
from typing import Dict, Any, List

from configs.default_config import load_config
from core.navigator import VLNNavigator
from core.context import NavContextBuilder, VisualFeatures
from core.action import Action
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VLN Navigator inference")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Navigation instruction",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum navigation steps",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main inference function.

    Args:
        args: Command line arguments

    Returns:
        Inference results
    """
    logger = setup_logger("Inference", level=args.log_level)
    logger.info("Running VLN Navigator inference...")

    # Load config
    config = load_config(args.config)

    # Initialize navigator
    navigator = VLNNavigator(config=config, log_level=args.log_level)
    navigator.initialize()

    if args.interactive:
        return run_interactive(navigator, config, args)
    else:
        return run_single_instruction(navigator, config, args)


def run_single_instruction(
    navigator: VLNNavigator,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run inference for a single instruction.

    Args:
        navigator: VLNNavigator instance
        config: Configuration
        args: Command line arguments

    Returns:
        Inference results
    """
    logger = logging.getLogger("Inference")

    # Set instruction
    navigator.set_instruction(args.instruction)

    # Initialize context
    context = navigator.get_context_for_step()

    # Run navigation loop
    actions: List[Dict[str, Any]] = []
    done = False
    step = 0

    logger.info(f"Instruction: {args.instruction}")

    while not done and step < args.max_steps:
        # Get action
        action = navigator.navigate()

        # Record action
        actions.append({
            "step": step,
            "action": action.to_habitat_action(),
            "confidence": action.confidence,
            "reasoning": action.reasoning,
        })

        logger.info(f"Step {step}: {action.action_type.name} (conf: {action.confidence:.2f})")

        # Check for stop
        if action.action_type.name == "STOP":
            done = True

        step += 1

    # Results
    results = {
        "instruction": args.instruction,
        "task_type": context.task_type.value if hasattr(context, 'task_type') else "Type-0",
        "total_steps": step,
        "actions": actions,
        "final_confidence": context.confidence if hasattr(context, 'confidence') else 0.0,
    }

    return results


def run_interactive(
    navigator: VLNNavigator,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run interactive inference session.

    Args:
        navigator: VLNNavigator instance
        config: Configuration
        args: Command line arguments

    Returns:
        Session results
    """
    logger = logging.getLogger("Inference")

    print("\n" + "=" * 50)
    print("VLN Navigator - Interactive Mode")
    print("=" * 50)
    print("Enter navigation instructions (or 'quit' to exit)")
    print()

    sessions = []

    while True:
        # Get instruction from user
        instruction = input("Instruction> ").strip()

        if instruction.lower() in ["quit", "exit", "q"]:
            break

        if not instruction:
            continue

        # Run navigation
        navigator.set_instruction(instruction)

        print(f"\nNavigating: {instruction}")
        print("-" * 40)

        actions = []
        step = 0

        while step < args.max_steps:
            action = navigator.navigate()

            print(f"Step {step}: {action.action_type.name} (conf: {action.confidence:.2f})")

            actions.append({
                "step": step,
                "action": action.to_habitat_action(),
                "confidence": action.confidence,
            })

            if action.action_type.name == "STOP":
                print("\nNavigation complete!")
                break

            step += 1

        sessions.append({
            "instruction": instruction,
            "actions": actions,
            "steps": step,
        })

        print()

    print("Interactive session ended.")
    print(f"Total sessions: {len(sessions)}")

    return {"sessions": sessions}


def run_from_file(
    navigator: VLNNavigator,
    config: Dict[str, Any],
    input_file: str,
    output_file: str,
) -> None:
    """
    Run inference from file of instructions.

    Args:
        navigator: VLNNavigator instance
        config: Configuration
        input_file: Path to input file with instructions
        output_file: Path to output file
    """
    with open(input_file, "r") as f:
        instructions = [line.strip() for line in f if line.strip()]

    results = []

    for i, instruction in enumerate(instructions):
        navigator.reset()
        navigator.set_instruction(instruction)

        actions = []
        step = 0
        max_steps = config.get("navigation", {}).get("max_steps", 100)

        while step < max_steps:
            action = navigator.navigate()
            actions.append(action.to_habitat_action())

            if action.action_type.name == "STOP":
                break

            step += 1

        results.append({
            "id": i,
            "instruction": instruction,
            "actions": actions,
            "steps": step,
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    results = run_inference(args)

    # Print summary
    print("\n" + "=" * 50)
    print("Inference Results")
    print("=" * 50)
    print(f"Instruction: {results.get('instruction', 'N/A')}")
    print(f"Total steps: {results.get('total_steps', 0)}")

    if "actions" in results:
        print("\nAction sequence:")
        for action in results["actions"][-5:]:
            print(f"  Step {action['step']}: {action['action']} (conf: {action['confidence']:.2f})")