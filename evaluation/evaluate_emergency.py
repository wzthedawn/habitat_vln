"""Emergency Navigation Evaluation Script.

This script evaluates emergency navigation performance with:
- Dynamic obstacle scenarios
- Path replanning success rate
- Response time measurement

Comparison experiments:
- Baseline: Phase 1 fixes, no PathReplanner
- Exp-A: With PathReplanner, no fine-tuning
- Exp-B: With PathReplanner and fine-tuning
"""

import json
import os
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmergencyEval")


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    name: str
    description: str
    use_path_replanner: bool = True
    use_finetuned_model: bool = False
    finetune_path: str = ""
    num_episodes: int = 50
    max_steps: int = 100
    obstacle_trigger_range: tuple = (5, 20)


@dataclass
class EvalResult:
    """Single episode evaluation result."""
    episode_id: int
    success: bool
    spl: float
    steps: int
    distance_to_goal: float
    emergency_triggered: bool = False
    emergency_response_time_ms: float = 0.0
    replan_count: int = 0
    obstacle_avoided: bool = True


class EmergencyEvaluator:
    """Evaluates emergency navigation scenarios."""

    # Experiment configurations
    EXPERIMENTS = {
        "baseline": EvalConfig(
            name="baseline",
            description="Phase 1 fixes, no PathReplanner",
            use_path_replanner=False,
            use_finetuned_model=False,
        ),
        "exp_a": EvalConfig(
            name="exp_a",
            description="With PathReplanner, no fine-tuning",
            use_path_replanner=True,
            use_finetuned_model=False,
        ),
        "exp_b": EvalConfig(
            name="exp_b",
            description="With PathReplanner and fine-tuning",
            use_path_replanner=True,
            use_finetuned_model=True,
        ),
    }

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.results: List[EvalResult] = []

    def run_experiment(
        self,
        experiment_name: str,
        dataset_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """Run evaluation experiment.

        Args:
            experiment_name: Name of experiment (baseline/exp_a/exp_b)
            dataset_path: Path to emergency dataset
            output_dir: Output directory for results

        Returns:
            Experiment results summary
        """
        if experiment_name not in self.EXPERIMENTS:
            logger.error(f"Unknown experiment: {experiment_name}")
            return {}

        exp_config = self.EXPERIMENTS[experiment_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_config.name}")
        logger.info(f"Description: {exp_config.description}")
        logger.info(f"{'='*60}\n")

        # Load test dataset
        test_data = self._load_dataset(dataset_path, "test")
        if not test_data:
            logger.error("No test data found")
            return {}

        logger.info(f"Loaded {len(test_data)} test episodes")

        # Initialize components based on config
        components = self._init_components(exp_config)

        # Run evaluation
        results = []

        for i, episode in enumerate(test_data[:exp_config.num_episodes]):
            logger.info(f"Evaluating episode {i+1}/{min(len(test_data), exp_config.num_episodes)}")

            result = self._evaluate_episode(episode, exp_config, components)
            results.append(result)

            # Log progress
            if (i + 1) % 10 == 0:
                self._log_progress(results)

        # Calculate metrics
        summary = self._calculate_metrics(results, exp_config)

        # Save results
        self._save_results(summary, results, exp_config, output_dir)

        return summary

    def _load_dataset(self, dataset_path: str, split: str = "test") -> List[Dict]:
        """Load dataset split."""
        file_path = os.path.join(dataset_path, f"{split}.json")

        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get("episodes", [])

    def _init_components(self, config: EvalConfig) -> Dict:
        """Initialize evaluation components."""
        import sys
        import os
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        components = {}

        # Initialize PathReplanner if needed
        if config.use_path_replanner:
            from emergency import PathReplanner
            components["path_replanner"] = PathReplanner()
            logger.info("PathReplanner initialized")

        # Initialize EmergencyDetector
        from emergency import EmergencyDetector
        components["emergency_detector"] = EmergencyDetector()
        logger.info("EmergencyDetector initialized")

        # Load fine-tuned model if needed
        if config.use_finetuned_model and config.finetune_path:
            # Would load fine-tuned model here
            logger.info(f"Would load fine-tuned model from: {config.finetune_path}")

        return components

    def _evaluate_episode(
        self,
        episode: Dict,
        config: EvalConfig,
        components: Dict
    ) -> EvalResult:
        """Evaluate a single episode.

        This is a simulated evaluation. In real deployment,
        this would integrate with Habitat environment.
        """
        episode_id = episode.get("episode_id", 0)
        obstacle_config = episode.get("obstacle_config", {})

        # Simulate navigation
        success = False
        steps = 0
        distance = episode.get("geodesic_distance", 10)
        spl = 0.0
        emergency_triggered = False
        response_time = 0.0
        replan_count = 0
        obstacle_avoided = True

        # Simulate step-by-step navigation
        trigger_step = obstacle_config.get("trigger_step", 10)

        for step in range(config.max_steps):
            steps = step + 1

            # Check for obstacle trigger
            if step == trigger_step:
                emergency_triggered = True

                if config.use_path_replanner and "path_replanner" in components:
                    # Measure response time
                    start_time = time.time()

                    # Simulate path replanning
                    result = components["path_replanner"].replan(
                        current_pos=(0, 0, 0),
                        goal_pos=episode.get("goal_position", [10, 0, 10]),
                        blocked_positions=[obstacle_config.get("position", [5, 0, 5])]
                    )

                    response_time = (time.time() - start_time) * 1000
                    replan_count = 1

                    if not result.success:
                        obstacle_avoided = False
                else:
                    # Without path replanner, rely on fallback
                    response_time = 0
                    obstacle_avoided = False

            # Simulate progress (simplified)
            distance = max(0, distance - 0.3)

            if distance < 3.0:  # Success threshold
                success = True
                break

        # Calculate SPL
        if success and steps > 0:
            shortest = episode.get("geodesic_distance", 10)
            actual = steps * 0.3  # Approximate
            spl = shortest / max(actual, shortest)

        return EvalResult(
            episode_id=episode_id,
            success=success,
            spl=spl,
            steps=steps,
            distance_to_goal=distance,
            emergency_triggered=emergency_triggered,
            emergency_response_time_ms=response_time,
            replan_count=replan_count,
            obstacle_avoided=obstacle_avoided,
        )

    def _log_progress(self, results: List[EvalResult]) -> None:
        """Log evaluation progress."""
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_spl = sum(r.spl for r in results) / len(results)
        emergency_rate = sum(1 for r in results if r.emergency_triggered) / len(results)
        obstacle_avoid_rate = sum(1 for r in results if r.obstacle_avoided) / len(results)

        logger.info(f"  Progress: {len(results)} episodes")
        logger.info(f"  SR: {success_rate:.1%}, SPL: {avg_spl:.3f}")
        logger.info(f"  Emergency triggered: {emergency_rate:.1%}")
        logger.info(f"  Obstacle avoided: {obstacle_avoid_rate:.1%}")

    def _calculate_metrics(
        self,
        results: List[EvalResult],
        config: EvalConfig
    ) -> Dict[str, Any]:
        """Calculate summary metrics."""
        if not results:
            return {}

        total = len(results)
        successes = sum(1 for r in results if r.success)

        # Core metrics
        success_rate = successes / total
        avg_spl = sum(r.spl for r in results) / total
        avg_steps = sum(r.steps for r in results) / total
        avg_distance = sum(r.distance_to_goal for r in results) / total

        # Emergency-specific metrics
        emergency_count = sum(1 for r in results if r.emergency_triggered)
        emergency_rate = emergency_count / total

        # Response time (only for triggered episodes)
        triggered_results = [r for r in results if r.emergency_triggered]
        avg_response_time = 0.0
        if triggered_results:
            avg_response_time = sum(r.emergency_response_time_ms for r in triggered_results) / len(triggered_results)

        # Obstacle avoidance
        avoid_count = sum(1 for r in results if r.obstacle_avoided)
        avoid_rate = avoid_count / total

        # By difficulty
        by_difficulty = defaultdict(lambda: {"count": 0, "success": 0})
        # Would need difficulty info in results

        return {
            "experiment": config.name,
            "description": config.description,
            "total_episodes": total,
            "success_rate": success_rate,
            "spl": avg_spl,
            "avg_steps": avg_steps,
            "avg_distance_to_goal": avg_distance,
            "emergency_triggered_rate": emergency_rate,
            "avg_emergency_response_time_ms": avg_response_time,
            "obstacle_avoidance_rate": avoid_rate,
            "use_path_replanner": config.use_path_replanner,
            "use_finetuned_model": config.use_finetuned_model,
        }

    def _save_results(
        self,
        summary: Dict,
        results: List[EvalResult],
        config: EvalConfig,
        output_dir: str
    ) -> None:
        """Save evaluation results."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"eval_{config.name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output = {
            "summary": summary,
            "results": [
                {
                    "episode_id": r.episode_id,
                    "success": r.success,
                    "spl": r.spl,
                    "steps": r.steps,
                    "distance_to_goal": r.distance_to_goal,
                    "emergency_triggered": r.emergency_triggered,
                    "response_time_ms": r.emergency_response_time_ms,
                    "obstacle_avoided": r.obstacle_avoided,
                }
                for r in results
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {filepath}")

    def run_all_experiments(
        self,
        dataset_path: str,
        output_dir: str,
        experiments: List[str] = None
    ) -> Dict[str, Dict]:
        """Run multiple experiments and compare results."""
        experiments = experiments or list(self.EXPERIMENTS.keys())

        all_summaries = {}

        for exp_name in experiments:
            summary = self.run_experiment(exp_name, dataset_path, output_dir)
            all_summaries[exp_name] = summary

        # Compare results
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPARISON")
        print("=" * 60)

        for exp_name, summary in all_summaries.items():
            if summary:
                print(f"\n{exp_name.upper()}:")
                print(f"  Success Rate: {summary['success_rate']:.1%}")
                print(f"  SPL: {summary['spl']:.3f}")
                print(f"  Obstacle Avoidance: {summary['obstacle_avoidance_rate']:.1%}")
                if summary['use_path_replanner']:
                    print(f"  Response Time: {summary['avg_emergency_response_time_ms']:.1f}ms")

        # Save comparison
        comparison_file = os.path.join(output_dir, "experiment_comparison.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        return all_summaries


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate emergency navigation")
    parser.add_argument("--dataset", type=str, default="/data/WZ/Dataset/emergency_dataset",
                        help="Path to emergency dataset")
    parser.add_argument("--output", type=str, default="results/emergency_eval",
                        help="Output directory")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["baseline", "exp_a"],
                        help="Experiments to run")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Number of episodes per experiment")

    args = parser.parse_args()

    # Update experiment configs
    for exp_name in args.experiments:
        if exp_name in EmergencyEvaluator.EXPERIMENTS:
            EmergencyEvaluator.EXPERIMENTS[exp_name].num_episodes = args.num_episodes

    # Run evaluation
    evaluator = EmergencyEvaluator()
    evaluator.run_all_experiments(
        dataset_path=args.dataset,
        output_dir=args.output,
        experiments=args.experiments,
    )


if __name__ == "__main__":
    main()