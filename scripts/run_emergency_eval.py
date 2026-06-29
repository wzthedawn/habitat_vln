#!/usr/bin/env python3
"""
VLN Emergency Navigation Evaluation Script

Runs comparison experiments with real Habitat environment:
- Baseline: Phase 1 fixes, pure LLM, no PathReplanner
- Exp-A: PathReplanner enabled, base model
- Exp-B: PathReplanner enabled, LoRA fine-tuned model
- Exp-C: No PathReplanner, LoRA fine-tuned model

Usage:
    python scripts/run_emergency_eval.py --exp baseline --episodes 10
    python scripts/run_emergency_eval.py --exp all --episodes 20
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import torch.multiprocessing as mp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EmergencyEpisode:
    """Emergency episode data structure."""
    episode_id: int
    scene_id: str
    original_instruction: str
    emergency_instruction: str
    start_position: List[float]
    goal_position: List[float]
    obstacle_config: Dict[str, Any]
    difficulty: str = "medium"
    geodesic_distance: float = 10.0
    scenario_type: str = "emergency"  # "emergency" or "normal"


@dataclass
class EmergencyEvalResult:
    """Single episode evaluation result."""
    episode_id: int
    scene_id: str
    success: bool
    spl: float
    steps: int
    distance_to_goal: float
    emergency_triggered: bool = False
    emergency_response_time_ms: float = 0.0
    obstacle_avoided: bool = False
    obstacle_type: str = ""
    use_path_replanner: bool = False
    use_lora: bool = False


def worker_process(
    worker_id: int,
    gpu_id: int,
    episodes: List[dict],
    exp_config: Dict[str, Any],
    config: Dict[str, Any],
    result_queue,
    llm_server: str,
):
    """Worker process for parallel evaluation.

    Args:
        worker_id: Worker process ID
        gpu_id: GPU to use for this worker (physical GPU ID)
        episodes: List of episode dicts to process
        exp_config: Experiment configuration
        config: Base configuration
        result_queue: Queue to put results
        llm_server: LLM server URL
    """
    import os
    # Suppress habitat-sim logs
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["MAGNUM_LOG"] = "quiet"

    import torch
    torch.cuda.set_device(gpu_id)

    import logging
    logger = logging.getLogger(f"Worker-{worker_id}")
    logger.info(f"Worker {worker_id} started on GPU {gpu_id}, processing {len(episodes)} episodes")

    # Create evaluator for this worker
    worker_config = config.copy()
    worker_config["device"] = "cuda"
    worker_config["llm_server"] = llm_server
    worker_config["gpu_id"] = gpu_id  # Physical GPU ID for Habitat

    evaluator = EmergencyVLNEvaluator(worker_config)

    # Initialize components
    if not evaluator.initialize_components(exp_config):
        result_queue.put({"worker_id": worker_id, "error": "Failed to initialize"})
        return

    results = []
    for i, ep_dict in enumerate(episodes):
        # Convert dict to EmergencyEpisode
        episode = EmergencyEpisode(
            episode_id=ep_dict["episode_id"],
            scene_id=ep_dict["scene_id"],
            original_instruction=ep_dict["original_instruction"],
            emergency_instruction=ep_dict["emergency_instruction"],
            start_position=ep_dict["start_position"],
            goal_position=ep_dict["goal_position"],
            obstacle_config=ep_dict["obstacle_config"],
            difficulty=ep_dict.get("difficulty", "medium"),
            geodesic_distance=ep_dict.get("geodesic_distance", 10.0),
            scenario_type=ep_dict.get("scenario_type", "emergency"),
        )

        logger.info(f"Worker {worker_id}: Episode {i+1}/{len(episodes)}")
        result = evaluator.run_episode(episode, exp_config)
        results.append(result)

    # Put results in queue
    result_queue.put({
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "results": results,
        "success_count": sum(1 for r in results if r.success),
    })

    logger.info(f"Worker {worker_id} completed: {sum(1 for r in results if r.success)}/{len(results)} success")


class EmergencyVLNEvaluator:
    """Emergency Navigation Evaluator with real Habitat integration."""

    # Experiment configurations
    EXPERIMENTS = {
        "baseline": {
            "use_path_replanner": False,
            "use_lora_model": False,
            "description": "Phase 1 fixes, no PathReplanner",
        },
        "exp-a": {
            "use_path_replanner": True,
            "use_lora_model": False,
            "description": "PathReplanner enabled, base model",
        },
        "exp-b": {
            "use_path_replanner": True,
            "use_lora_model": True,
            "description": "PathReplanner + LoRA fine-tuned",
        },
        "exp-c": {
            "use_path_replanner": False,
            "use_lora_model": True,
            "description": "LoRA fine-tuned, no PathReplanner",
        },
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("EmergencyEval")
        logging.basicConfig(level=logging.INFO)

        # Data
        self.episodes: List[EmergencyEpisode] = []

        # Components
        self.base_evaluator = None
        self.obstacle_manager = None
        self.emergency_detector = None

        # Results
        self.results: List[EmergencyEvalResult] = []

        # Paths
        self.mp3d_path = Path(config.get("mp3d_path", "/data/WZ/Dataset/mp3d_dataset/mp3d"))
        self.emergency_dataset_path = Path(config.get("emergency_dataset_path", "/data/WZ/Dataset/emergency_dataset"))
        self.balanced_dataset_path = Path(config.get("balanced_dataset_path", "/data/WZ/Dataset/qlora_balanced"))

    def load_balanced_episodes(self, split: str = "test") -> int:
        """Load episodes from original data sources (emergency + normal) with proper scene_id.

        Uses the split ratios from qlora_balanced/stats.json to match training data.

        Args:
            split: Dataset split (train, val, test)

        Returns:
            Number of episodes loaded
        """
        self.episodes = []

        # Load stats to get split counts
        stats_file = self.balanced_dataset_path / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            split_stats = stats.get(split, {})
            num_emergency = split_stats.get("emergency", 41)
            num_normal = split_stats.get("normal", 40)
        else:
            num_emergency = 41
            num_normal = 40

        # Load emergency episodes from emergency_instructions.json
        emergency_file = self.emergency_dataset_path / "emergency_instructions.json"
        if emergency_file.exists():
            with open(emergency_file, 'r') as f:
                emergency_data = json.load(f)

            all_emergency_episodes = emergency_data.get("episodes", [])

            # Use test split (last 60 episodes)
            splits = emergency_data.get("splits", {"train": 280, "val": 60, "test": 60})
            train_end = splits.get("train", 280)
            val_end = train_end + splits.get("val", 60)

            if split == "test":
                split_episodes = all_emergency_episodes[val_end:]
            elif split == "val":
                split_episodes = all_emergency_episodes[train_end:val_end]
            else:
                split_episodes = all_emergency_episodes[:train_end]

            # Take required number
            for ep in split_episodes[:num_emergency]:
                episode = EmergencyEpisode(
                    episode_id=ep.get("episode_id", 0),
                    scene_id=ep.get("scene_id", ""),
                    original_instruction=ep.get("original_instruction", ""),
                    emergency_instruction=ep.get("emergency_instruction", ""),
                    start_position=ep.get("start_position", [0, 0, 0]),
                    goal_position=ep.get("goal_position", [0, 0, 0]),
                    obstacle_config=ep.get("obstacle_config", {}),
                    difficulty=ep.get("difficulty", "medium"),
                    geodesic_distance=ep.get("geodesic_distance", 10.0),
                    scenario_type="emergency",
                )
                self.episodes.append(episode)

        # Load normal episodes from R2R
        r2r_file = Path("/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json")
        if r2r_file.exists():
            with open(r2r_file, 'r') as f:
                r2r_data = json.load(f)

            r2r_episodes = r2r_data.get("episodes", [])

            # Filter out episodes whose goals overlap with emergency goals
            emergency_goals = set()
            for ep in self.episodes:
                goal = tuple(round(x, 2) for x in ep.goal_position)
                emergency_goals.add(goal)

            available_normal = []
            for ep in r2r_episodes:
                goals = ep.get("goals", [])
                if goals:
                    goal_pos = tuple(round(x, 2) for x in goals[0].get("position", [0, 0, 0]))
                    if goal_pos not in emergency_goals:
                        available_normal.append(ep)

            # Take required number
            for ep in available_normal[:num_normal]:
                inst = ep.get("instruction", {})
                inst_text = inst.get("instruction_text", "") if isinstance(inst, dict) else str(inst)

                episode = EmergencyEpisode(
                    episode_id=len(self.episodes),
                    scene_id=ep.get("scene_id", ""),
                    original_instruction=inst_text,
                    emergency_instruction="",
                    start_position=ep.get("start_position", [0, 0, 0]),
                    goal_position=ep.get("goals", [{}])[0].get("position", [0, 0, 0]) if ep.get("goals") else [0, 0, 0],
                    obstacle_config={},
                    difficulty="medium",
                    geodesic_distance=ep.get("info", {}).get("geodesic_distance", 10.0),
                    scenario_type="normal",
                )
                self.episodes.append(episode)

        emergency_count = sum(1 for e in self.episodes if e.scenario_type == "emergency")
        normal_count = len(self.episodes) - emergency_count

        self.logger.info(f"Loaded {len(self.episodes)} episodes from {split} split")
        self.logger.info(f"  Emergency: {emergency_count}, Normal: {normal_count}")
        return len(self.episodes)

    def load_emergency_episodes(self, split: str = "test") -> int:
        """Load emergency episodes from balanced dataset (new method).

        Args:
            split: Dataset split (train, val, test)

        Returns:
            Number of episodes loaded
        """
        return self.load_balanced_episodes(split)

    def initialize_components(self, exp_config: Dict[str, Any]) -> bool:
        """Initialize evaluation components.

        Args:
            exp_config: Experiment configuration

        Returns:
            True if successful
        """
        try:
            # Import and initialize base evaluator
            from run_vln_experiment import MultiAgentVLNEvaluator

            base_config = self.config.copy()
            base_config["use_path_replanner"] = exp_config.get("use_path_replanner", False)
            base_config["use_lora_model"] = exp_config.get("use_lora_model", False)

            self.base_evaluator = MultiAgentVLNEvaluator(base_config)
            self.base_evaluator.initialize()

            # Note: obstacle_manager and emergency_detector are created inside
            # run_vln_experiment.py based on episode.obstacle_config

            # Configure PathReplanner if enabled
            if exp_config.get("use_path_replanner") and self.base_evaluator.decision_agent:
                from emergency import PathReplanner
                path_replanner = PathReplanner({})
                self.base_evaluator.decision_agent.set_path_replanner(path_replanner)
                self.logger.info("PathReplanner enabled for DecisionAgent")

            # Configure LoRA if enabled
            if exp_config.get("use_lora_model"):
                if self.base_evaluator.decision_agent:
                    self.base_evaluator.decision_agent.set_use_lora(True)
                    self.logger.info("LoRA enabled for DecisionAgent: decision-lora")
            else:
                # Explicitly disable LoRA for baseline/exp-a
                if self.base_evaluator.decision_agent:
                    self.base_evaluator.decision_agent.set_use_lora(False)
                    self.logger.info("LoRA disabled for DecisionAgent (using base model)")

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_episode(self, episode: EmergencyEpisode, exp_config: Dict[str, Any]) -> EmergencyEvalResult:
        """Run a single episode (emergency or normal).

        Args:
            episode: Episode to run
            exp_config: Experiment configuration

        Returns:
            Evaluation result
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Episode {episode.episode_id}: {episode.scenario_type}")
        if episode.scenario_type == "emergency":
            self.logger.info(f"Obstacle type: {episode.obstacle_config.get('type', 'unknown')}")
            self.logger.info(f"Trigger step: {episode.obstacle_config.get('trigger_step', 10)}")
        else:
            self.logger.info(f"Normal navigation task")
        self.logger.info(f"{'='*60}")

        try:

            # Run episode using base evaluator with obstacle integration
            result = self._run_episode_with_obstacles(episode, exp_config)

            return result

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} failed: {e}")
            import traceback
            traceback.print_exc()

            return EmergencyEvalResult(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                success=False,
                spl=0.0,
                steps=0,
                distance_to_goal=episode.geodesic_distance,
                obstacle_type=episode.obstacle_config.get("type", "") if episode.scenario_type == "emergency" else "normal",
                use_path_replanner=exp_config.get("use_path_replanner", False),
                use_lora=exp_config.get("use_lora_model", False),
            )

    def _run_episode_with_obstacles(
        self,
        episode: EmergencyEpisode,
        exp_config: Dict[str, Any]
    ) -> EmergencyEvalResult:
        """Run episode with obstacle integration in navigation loop.

        This is a simplified version that wraps the base evaluator.
        Full implementation would modify the navigation loop directly.
        """
        # For now, use the base evaluator's episode runner
        # The obstacle integration happens through context metadata

        from run_vln_experiment import R2REpisode

        # Create R2REpisode with emergency instruction
        base_episode = R2REpisode(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            instruction=episode.emergency_instruction,  # Use emergency instruction
            start_position=episode.start_position,
            start_rotation=[0, 0, 0, 1],  # Default rotation
            goal_position=episode.goal_position,
            reference_path=[],  # Not needed for emergency
            geodesic_distance=episode.geodesic_distance,
        )

        # Add obstacle config to episode for navigation loop
        base_episode.obstacle_config = episode.obstacle_config

        # Run the episode (run_vln_experiment.py will create its own obstacle_manager
        # based on episode.obstacle_config)
        result = self.base_evaluator._run_habitat_episode(base_episode)

        # Extract emergency-specific metrics from context metadata
        context_metadata = getattr(self.base_evaluator, '_last_context_metadata', {})
        emergency_signal = context_metadata.get("emergency_signal", {})
        emergency_triggered = emergency_signal.get("trigger", False)
        emergency_response_time = 0.0

        return EmergencyEvalResult(
            episode_id=episode.episode_id,
            scene_id=episode.scene_id,
            success=result.get("success", False),
            spl=result.get("spl", 0.0),
            steps=result.get("steps", 0),
            distance_to_goal=result.get("distance_to_goal", episode.geodesic_distance),
            emergency_triggered=emergency_triggered,
            emergency_response_time_ms=emergency_response_time,
            obstacle_avoided=result.get("success", False),  # If succeeded, obstacle was avoided
            obstacle_type=episode.obstacle_config.get("type", "unknown"),
            use_path_replanner=exp_config.get("use_path_replanner", False),
            use_lora=exp_config.get("use_lora_model", False),
        )

    def run_experiment(
        self,
        exp_name: str,
        num_episodes: int = 10,
        split: str = "test"
    ) -> Dict[str, Any]:
        """Run a single experiment.

        Args:
            exp_name: Experiment name (baseline, exp-a, exp-b, exp-c)
            num_episodes: Number of episodes to run
            split: Dataset split

        Returns:
            Summary dictionary
        """
        if exp_name not in self.EXPERIMENTS:
            self.logger.error(f"Unknown experiment: {exp_name}")
            return {}

        exp_config = self.EXPERIMENTS[exp_name]

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Experiment: {exp_name}")
        self.logger.info(f"Description: {exp_config['description']}")
        self.logger.info(f"PathReplanner: {exp_config['use_path_replanner']}")
        self.logger.info(f"LoRA Model: {exp_config['use_lora_model']}")
        self.logger.info(f"{'='*60}\n")

        # Load episodes
        if not self.episodes:
            self.load_emergency_episodes(split)

        episodes_to_run = self.episodes[:num_episodes]

        # Initialize components
        if not self.initialize_components(exp_config):
            return {}

        # Run episodes
        results = []
        for i, episode in enumerate(episodes_to_run):
            self.logger.info(f"\nRunning episode {i+1}/{len(episodes_to_run)}")
            result = self.run_episode(episode, exp_config)
            results.append(result)

            # Progress update
            success_count = sum(1 for r in results if r.success)
            self.logger.info(f"Progress: {success_count}/{len(results)} success ({success_count/len(results)*100:.1f}%)")

        # Calculate summary
        summary = self._calculate_summary(results, exp_name)

        return summary

    def _calculate_summary(
        self,
        results: List[EmergencyEvalResult],
        exp_name: str
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
        emergency_triggered_count = sum(1 for r in results if r.emergency_triggered)
        emergency_rate = emergency_triggered_count / total

        # Obstacle avoidance rate
        obstacle_avoided_count = sum(1 for r in results if r.obstacle_avoided)
        obstacle_avoidance_rate = obstacle_avoided_count / total

        # By obstacle type
        by_type = {}
        for r in results:
            otype = r.obstacle_type or "normal"
            if otype not in by_type:
                by_type[otype] = {"total": 0, "success": 0}
            by_type[otype]["total"] += 1
            if r.success:
                by_type[otype]["success"] += 1

        # Calculate success rate by type
        for otype in by_type:
            by_type[otype]["success_rate"] = by_type[otype]["success"] / by_type[otype]["total"]

        # Separate emergency vs normal metrics
        emergency_results = [r for r in results if r.obstacle_type and r.obstacle_type != "normal"]
        normal_results = [r for r in results if not r.obstacle_type or r.obstacle_type == "normal"]

        emergency_success_rate = 0.0
        if emergency_results:
            emergency_success_rate = sum(1 for r in emergency_results if r.success) / len(emergency_results)

        normal_success_rate = 0.0
        if normal_results:
            normal_success_rate = sum(1 for r in normal_results if r.success) / len(normal_results)

        return {
            "experiment": exp_name,
            "description": self.EXPERIMENTS[exp_name]["description"],
            "total_episodes": total,
            "success_rate": success_rate,
            "avg_spl": avg_spl,
            "avg_steps": avg_steps,
            "avg_distance_to_goal": avg_distance,
            "emergency_triggered_rate": emergency_rate,
            "obstacle_avoidance_rate": obstacle_avoidance_rate,
            "emergency_success_rate": emergency_success_rate,
            "normal_success_rate": normal_success_rate,
            "by_obstacle_type": by_type,
            "use_path_replanner": self.EXPERIMENTS[exp_name]["use_path_replanner"],
            "use_lora_model": self.EXPERIMENTS[exp_name]["use_lora_model"],
        }

    def run_parallel_experiment(
        self,
        exp_name: str,
        num_episodes: int = 10,
        split: str = "test",
        num_workers: int = 3,
        gpu_ids: List[int] = [1, 2, 3],
        llm_server: str = "http://localhost:8000",
    ) -> Dict[str, Any]:
        """Run experiment with parallel workers.

        Args:
            exp_name: Experiment name
            num_episodes: Total episodes to run
            split: Dataset split
            num_workers: Number of parallel workers
            gpu_ids: GPU IDs for each worker
            llm_server: LLM server URL

        Returns:
            Summary dictionary
        """
        if exp_name not in self.EXPERIMENTS:
            self.logger.error(f"Unknown experiment: {exp_name}")
            return {}

        exp_config = self.EXPERIMENTS[exp_name]

        # Load episodes
        if not self.episodes:
            self.load_emergency_episodes(split)

        episodes_to_run = self.episodes[:num_episodes]

        # Convert episodes to dicts for pickling
        episode_dicts = [
            {
                "episode_id": ep.episode_id,
                "scene_id": ep.scene_id,
                "original_instruction": ep.original_instruction,
                "emergency_instruction": ep.emergency_instruction,
                "start_position": ep.start_position,
                "goal_position": ep.goal_position,
                "obstacle_config": ep.obstacle_config,
                "difficulty": ep.difficulty,
                "geodesic_distance": ep.geodesic_distance,
                "scenario_type": ep.scenario_type,
            }
            for ep in episodes_to_run
        ]

        # Split episodes across workers
        chunk_size = len(episode_dicts) // num_workers
        episode_chunks = []
        for i in range(num_workers):
            start_idx = i * chunk_size
            if i == num_workers - 1:
                # Last worker takes remaining episodes
                end_idx = len(episode_dicts)
            else:
                end_idx = start_idx + chunk_size
            episode_chunks.append(episode_dicts[start_idx:end_idx])

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Parallel Experiment: {exp_name}")
        self.logger.info(f"Workers: {num_workers}, GPUs: {gpu_ids}")
        self.logger.info(f"Total episodes: {num_episodes}")
        for i, chunk in enumerate(episode_chunks):
            self.logger.info(f"  Worker {i} (GPU {gpu_ids[i]}): {len(chunk)} episodes")
        self.logger.info(f"{'='*60}\n")

        # Create result queue
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()

        # Start workers
        processes = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    gpu_ids[worker_id],
                    episode_chunks[worker_id],
                    exp_config,
                    self.config,
                    result_queue,
                    llm_server,
                ),
            )
            p.start()
            processes.append(p)

        # Collect results
        all_results = []
        worker_stats = []
        for _ in range(num_workers):
            worker_result = result_queue.get()
            if "error" in worker_result:
                self.logger.error(f"Worker {worker_result['worker_id']} failed: {worker_result['error']}")
            else:
                all_results.extend(worker_result["results"])
                worker_stats.append(worker_result)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Calculate summary
        summary = self._calculate_summary(all_results, exp_name)

        # Add parallel stats
        summary["parallel_workers"] = num_workers
        summary["gpu_ids"] = gpu_ids
        summary["worker_stats"] = worker_stats

        return summary

    def run_all_experiments(
        self,
        num_episodes: int = 10,
        split: str = "test",
        output_dir: str = "results/emergency_eval"
    ) -> Dict[str, Dict]:
        """Run all experiments and compare results.

        Args:
            num_episodes: Episodes per experiment
            split: Dataset split
            output_dir: Output directory for results

        Returns:
            Dictionary of all experiment summaries
        """
        os.makedirs(output_dir, exist_ok=True)

        all_summaries = {}

        for exp_name in self.EXPERIMENTS.keys():
            # Reset episodes for each experiment
            self.episodes = []
            self.results = []

            summary = self.run_experiment(exp_name, num_episodes, split)
            all_summaries[exp_name] = summary

        # Print comparison
        self._print_comparison(all_summaries)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(output_dir, f"comparison_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        self.logger.info(f"\nResults saved to: {output_file}")

        return all_summaries

    def _print_comparison(self, summaries: Dict[str, Dict]) -> None:
        """Print experiment comparison table."""
        print("\n" + "=" * 100)
        print("EMERGENCY NAVIGATION EXPERIMENT COMPARISON")
        print("=" * 100)
        print(f"{'Experiment':<12} {'Overall SR':>12} {'Emergency SR':>14} {'Normal SR':>12} {'SPL':>8} {'PathRep':>10} {'LoRA':>8}")
        print("-" * 100)

        for exp_name, summary in summaries.items():
            if summary:
                sr = summary.get("success_rate", 0) * 100
                em_sr = summary.get("emergency_success_rate", 0) * 100
                norm_sr = summary.get("normal_success_rate", 0) * 100
                spl = summary.get("avg_spl", 0)
                path_rep = "Yes" if summary.get("use_path_replanner") else "No"
                lora = "Yes" if summary.get("use_lora_model") else "No"
                print(f"{exp_name:<12} {sr:>11.1f}% {em_sr:>13.1f}% {norm_sr:>11.1f}% {spl:>8.3f} {path_rep:>10} {lora:>8}")

        print("=" * 100)

        # Print by obstacle type
        print("\nBy Obstacle Type:")
        print("-" * 100)

        for exp_name, summary in summaries.items():
            if summary and "by_obstacle_type" in summary:
                print(f"\n{exp_name}:")
                for otype, stats in summary["by_obstacle_type"].items():
                    sr = stats["success_rate"] * 100
                    print(f"  {otype}: {sr:.1f}% ({stats['success']}/{stats['total']})")


def main():
    parser = argparse.ArgumentParser(description="VLN Emergency Navigation Evaluation")
    parser.add_argument("--exp", type=str, default="baseline",
                       choices=["baseline", "exp-a", "exp-b", "exp-c", "all"],
                       help="Experiment to run")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes per experiment")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to use")
    parser.add_argument("--mp3d-path", type=str, default="/data/WZ/Dataset/mp3d_dataset/mp3d")
    parser.add_argument("--emergency-dataset", type=str, default="/data/WZ/Dataset/emergency_dataset")
    parser.add_argument("--output", type=str, default="results/emergency_eval")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--use-remote-llm", action="store_true", default=False)
    parser.add_argument("--llm-server", type=str, default="http://localhost:8000")
    parser.add_argument("--no-video", action="store_true", default=False,
                       help="Disable video generation for faster evaluation")
    parser.add_argument("--no-trajectory", action="store_true", default=False,
                       help="Disable trajectory plot generation")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel processes (default: 1, serial)")
    parser.add_argument("--gpus", type=str, default="1,2,3",
                       help="GPU IDs to use for parallel processes (comma-separated, e.g., '1,2,3')")

    args = parser.parse_args()

    # Configuration
    config = {
        "mp3d_path": args.mp3d_path,
        "emergency_dataset_path": args.emergency_dataset,
        "max_steps": args.max_steps,
        "success_distance": 3.0,
        "device": "cuda",
        "use_int8": True,
        "use_remote_llm": args.use_remote_llm,
        "llm_server": args.llm_server,
        "sequence_length": 5,
        "adaptive_sequence": False,
        "enable_video": not args.no_video,
        "enable_trajectory": not args.no_trajectory,
    }

    # Create evaluator
    evaluator = EmergencyVLNEvaluator(config)

    # Parse GPU IDs
    gpu_ids = [int(g) for g in args.gpus.split(",")]

    # Run experiments
    if args.parallel > 1:
        # Parallel mode
        if args.exp == "all":
            # Run all experiments sequentially, but each with parallel workers
            all_summaries = {}
            for exp_name in evaluator.EXPERIMENTS.keys():
                evaluator.episodes = []
                evaluator.results = []
                summary = evaluator.run_parallel_experiment(
                    exp_name=exp_name,
                    num_episodes=args.episodes,
                    split=args.split,
                    num_workers=args.parallel,
                    gpu_ids=gpu_ids,
                    llm_server=args.llm_server,
                )
                all_summaries[exp_name] = summary
            evaluator._print_comparison(all_summaries)
        else:
            summary = evaluator.run_parallel_experiment(
                exp_name=args.exp,
                num_episodes=args.episodes,
                split=args.split,
                num_workers=args.parallel,
                gpu_ids=gpu_ids,
                llm_server=args.llm_server,
            )

            # Print summary
            if summary:
                print(f"\n{args.exp.upper()} Summary:")
                print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
                print(f"  SPL: {summary['avg_spl']:.3f}")
                print(f"  Obstacle Avoidance: {summary['obstacle_avoidance_rate']*100:.1f}%")
                print(f"  Parallel Workers: {summary.get('parallel_workers', 1)}")
    else:
        # Serial mode (existing logic)
        if args.exp == "all":
            evaluator.run_all_experiments(
                num_episodes=args.episodes,
                split=args.split,
                output_dir=args.output
            )
        else:
            summary = evaluator.run_experiment(
                exp_name=args.exp,
                num_episodes=args.episodes,
                split=args.split
            )

            # Print summary
            if summary:
                print(f"\n{args.exp.upper()} Summary:")
                print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
                print(f"  SPL: {summary['avg_spl']:.3f}")
                print(f"  Obstacle Avoidance: {summary['obstacle_avoidance_rate']*100:.1f}%")


if __name__ == "__main__":
    main()