#!/usr/bin/env python3
"""
R2R VLN Evaluation Experiment Script
Uses real Matterport3D scenes and R2R dataset
Integrates multi-agent LLM navigation system

Multi-Agent Architecture:
- InstructionAgent: Rule matching, subtask decomposition
- PerceptionAgent: Visual perception
- TrajectoryAgent: Mapping + trajectory summary
- DecisionAgent: Decision making
- EvaluationAgent: Evaluation (optional)

Dual-environment IPC Architecture:
- Python 3.9 (Habitat): VLN main process, habitat-sim, YOLO
- Python 3.10 (LLM Server): Qwen3.5 model inference service

Usage:
    # Option 1: Start LLM service (Python 3.10)
    conda activate habitat_py310
    python llm_server.py --port 8000

    # Run VLN evaluation (Python 3.9)
    conda activate Habitat
    python run_vln_experiment.py --use-remote-llm --llm-server http://localhost:8000 ...

    # Option 2: Use SiliconFlow API (no local LLM server needed)
    conda activate Habitat
    python run_vln_experiment.py --use-siliconflow ...
"""

import argparse
import json
import logging
import os
import sys
import gc
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import random
import math
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.token_tracker import get_token_tracker
from utils.timeout_fallback import TimeoutError, timeout, StepTimeout, DEFAULT_TIMEOUTS
from utils.episode_output import EpisodeOutputManager

# Strategy imports
from strategies.cot import CoTStrategy
from strategies.reflection import ReflectionStrategy
from strategies.debate import DebateStrategy
from strategies.base_strategy import StrategyResult


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
    time_elapsed: float = 0.0
    task_level: str = "medium"
    subtask_count: int = 0
    evaluation_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiAgentVLNEvaluator:
    """Multi-agent VLN Evaluator"""

    def __init__(self, config: Dict[str, Any], log_level: str = "INFO"):
        self.config = config
        self.logger = setup_logger("MultiAgentVLNEvaluator", level=log_level)
        self.log_level = log_level

        # Video/Trajectory control
        self.enable_video = config.get("enable_video", True)
        self.enable_trajectory = config.get("enable_trajectory", True)

        # Agents
        self.instruction_agent = None
        self.perception_agent = None
        self.trajectory_agent = None
        self.decision_agent = None
        self.evaluation_agent = None

        # Model manager
        self.model_manager = None

        # Data
        self.episodes: List[R2REpisode] = []
        self.scene_paths: Dict[str, str] = {}

        # Results
        self.results: List[EvaluationResult] = []

        # Paths
        self.mp3d_path = Path(config.get("mp3d_path", "data/mp3d_dataset/mp3d"))
        self.r2r_path = Path(config.get("r2r_path", "/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json"))

        # Timeout configuration
        self.scene_load_timeout = config.get("scene_load_timeout", DEFAULT_TIMEOUTS["scene_load"])
        self.step_timeout = config.get("step_timeout", DEFAULT_TIMEOUTS["step"])
        self.episode_timeout = config.get("episode_timeout", DEFAULT_TIMEOUTS["episode"])

        # Simulator cache
        self._sim_cache: Dict[str, Any] = {}
        self._current_scene: str = ""

        # Image settings
        self.image_width = config.get("image_width", 640)
        self.image_height = config.get("image_height", 480)

        # Episode output manager
        self.output_manager = EpisodeOutputManager(
            config.get("output_dir", "results"),
            enable_video=self.enable_video,
            enable_trajectory=self.enable_trajectory,
        )

        self.logger.info(f"Multi-agent VLN Evaluator initialized")

    def initialize(self) -> None:
        """Initialize evaluation system"""
        self.logger.info("=" * 60)
        self.logger.info("Initializing multi-agent VLN evaluation system")
        self.logger.info("=" * 60)

        # 1. Build scene path mapping
        self._build_scene_paths()

        # 2. Load R2R data
        self._load_r2r_data()

        # 3. Initialize model manager
        self._init_model_manager()

        # 4. Initialize Agents
        self._init_agents()

        self.logger.info("Initialization complete!")

    def _build_scene_paths(self) -> None:
        """Build MP3D scene path mapping"""
        self.logger.info(f"Scanning MP3D scenes: {self.mp3d_path}")

        # 1. Scan main MP3D dataset (90 scenes)
        if self.mp3d_path.exists():
            for scene_dir in self.mp3d_path.iterdir():
                if scene_dir.is_dir():
                    scene_id = scene_dir.name
                    glb_file = scene_dir / f"{scene_id}.glb"
                    if glb_file.exists():
                        # Support R2R_VLNCE format: mp3d/scene_id/scene_id.glb
                        self.scene_paths[scene_id] = str(glb_file)
                        self.scene_paths[f"mp3d/{scene_id}/{scene_id}.glb"] = str(glb_file)

        self.logger.info(f"Found {len(self.scene_paths)} MP3D scenes")

    def _load_r2r_data(self) -> None:
        """Load R2R dataset"""
        self.logger.info(f"Loading R2R data: {self.r2r_path}")

        if not self.r2r_path.exists():
            self.logger.warning(f"R2R data path does not exist: {self.r2r_path}")
            return

        with open(self.r2r_path) as f:
            data = json.load(f)

        # Support two data formats: standard R2R format and simplified format
        if isinstance(data, dict) and 'episodes' in data:
            # Standard R2R format
            for ep_data in data['episodes']:
                episode = R2REpisode(
                    episode_id=ep_data['episode_id'],
                    scene_id=ep_data['scene_id'].split('/')[-1].replace('.glb', ''),
                    instruction=ep_data['instruction']['instruction_text'],
                    start_position=ep_data['start_position'],
                    start_rotation=ep_data['start_rotation'],
                    goal_position=ep_data['goals'][0]['position'],
                    reference_path=ep_data.get('reference_path', []),
                    geodesic_distance=ep_data['info']['geodesic_distance'],
                    trajectory_id=ep_data.get('trajectory_id', 0),
                )
                self.episodes.append(episode)
        elif isinstance(data, list):
            # Simplified format: [{"episode_id": 0, "instructions": [...], "trajectory": [...], "scan": "..."}, ...]
            for ep_data in data:
                # Extract path points from trajectory
                trajectory = ep_data.get('trajectory', [])
                reference_path = [[t[1], t[2], 0.0] if len(t) >= 3 else [0, 0, 0] for t in trajectory]

                # Take first instruction
                instructions = ep_data.get('instructions', [])
                instruction = instructions[0] if instructions else "Navigate to the target."

                episode = R2REpisode(
                    episode_id=ep_data.get('episode_id', 0),
                    scene_id=ep_data.get('scan', 'sample_scene'),
                    instruction=instruction,
                    start_position=[0.0, 0.0, 0.0],  # Simplified format uses default start position
                    start_rotation=[0.0, 0.0, 0.0],
                    goal_position=reference_path[-1] if reference_path else [0.0, 0.0, 0.0],
                    reference_path=reference_path,
                    geodesic_distance=float(len(trajectory)) if trajectory else 0.0,
                    trajectory_id=ep_data.get('path_id', 0),
                )
                self.episodes.append(episode)

        self.logger.info(f"Loaded {len(self.episodes)} R2R episodes")

    def _check_depth_clear_direction(self, depth_image) -> Optional[str]:
        """Check which direction is clear based on depth image.

        Args:
            depth_image: Depth image (numpy array or similar)

        Returns:
            "left", "right", or None if can't determine
        """
        import numpy as np

        try:
            if depth_image is None:
                return None

            # Convert to numpy if needed
            if hasattr(depth_image, 'cpu'):
                depth = depth_image.cpu().numpy()
            else:
                depth = np.array(depth_image)

            # Handle different depth formats
            if depth.ndim == 3:
                depth = depth[:, :, 0]  # Take first channel if RGB

            height, width = depth.shape[:2]
            mid_y = height // 2

            # Sample depth at center-left and center-right
            left_start = int(width * 0.1)
            left_end = int(width * 0.4)
            right_start = int(width * 0.6)
            right_end = int(width * 0.9)

            # Average depth in left and right regions (lower rows = closer to agent)
            left_region = depth[mid_y-10:mid_y+10, left_start:left_end]
            right_region = depth[mid_y-10:mid_y+10, right_start:right_end]

            left_avg = np.mean(left_region) if left_region.size > 0 else 0
            right_avg = np.mean(right_region) if right_region.size > 0 else 0

            self.logger.debug(f"[Depth] Left avg: {left_avg:.2f}, Right avg: {right_avg:.2f}")

            # Higher depth = farther = clearer path
            # Threshold: prefer side with > 2m average depth
            min_clear_depth = 2.0

            if left_avg > min_clear_depth and left_avg > right_avg:
                return "left"
            elif right_avg > min_clear_depth and right_avg > left_avg:
                return "right"
            elif left_avg > min_clear_depth:
                return "left"
            elif right_avg > min_clear_depth:
                return "right"
            else:
                # Both blocked, default to right
                return "right" if right_avg >= left_avg else "left"

        except Exception as e:
            self.logger.warning(f"Depth check failed: {e}")
            return None

    def _init_model_manager(self) -> None:
        """Initialize model manager"""
        self.logger.info("Initializing model manager...")

        try:
            from models.model_manager import get_model_manager

            # Build config with remote LLM settings if enabled
            model_config = {
                "device": self.config.get("device", "cuda"),
                "use_int8": self.config.get("use_int8", True),
                "load_all_qwen": False,  # Lazy loading
            }

            # Add remote LLM configuration
            if self.config.get("use_remote_llm", False):
                model_config["use_remote"] = True
                model_config["remote_server_url"] = self.config.get(
                    "llm_server", "http://localhost:8000"
                )
                model_config["remote_timeout"] = self.config.get("remote_timeout", 120.0)  # Increased for instruction decomposition
                self.logger.info(f"Using remote LLM service: {model_config['remote_server_url']}")

            # Add SiliconFlow API configuration
            if self.config.get("use_siliconflow", False):
                model_config["use_siliconflow"] = True
                model_config["siliconflow_api_key"] = self.config.get("siliconflow_api_key")
                model_config["siliconflow_model"] = self.config.get("siliconflow_model", "Qwen/Qwen3.5-397B-A17B")
                # Also set use_remote=True so agents know to use remote mode
                model_config["use_remote"] = True
                self.logger.info(f"Using SiliconFlow API with model: {model_config['siliconflow_model']}")

            self.model_manager = get_model_manager(model_config)
            self.logger.info("Model manager initialized successfully")
        except Exception as e:
            self.logger.warning(f"Model manager initialization failed: {e}")

    def _init_agents(self) -> None:
        """Initialize all agents"""
        self.logger.info("Initializing agents...")

        agent_config = {
            "confidence_threshold": 0.6,
            "max_history_steps": 5,
            # Sequence length configuration
            "sequence_length": self.config.get("sequence_length", 5),
            "adaptive_sequence": self.config.get("adaptive_sequence", False),
            "min_sequence_length": self.config.get("min_sequence_length", 2),
            "max_sequence_length": self.config.get("max_sequence_length", 20),
            # Pass remote LLM config to agents
            "use_remote": self.config.get("use_remote_llm", False) or self.config.get("use_siliconflow", False),
            "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
            # SiliconFlow config
            "use_siliconflow": self.config.get("use_siliconflow", False),
            "siliconflow_model": self.config.get("siliconflow_model"),
            # Pass model_manager for LLM-based decomposition
            "model_manager": self.model_manager if hasattr(self, 'model_manager') else None,
            "use_llm_decompose": self.config.get("use_llm_decompose", True),
        }

        try:
            from agents.instruction_agent import InstructionAgent
            from agents.perception_agent import PerceptionAgent
            from agents.trajectory_agent import TrajectoryAgent
            from agents.decision_agent import DecisionAgent
            from agents.evaluation_agent import EvaluationAgent

            self.instruction_agent = InstructionAgent(agent_config)
            self.perception_agent = PerceptionAgent(agent_config)
            self.trajectory_agent = TrajectoryAgent(agent_config)
            self.decision_agent = DecisionAgent(agent_config)
            self.evaluation_agent = EvaluationAgent(agent_config)

            # Initialize agents
            self.instruction_agent.initialize()
            self.perception_agent.initialize()
            self.trajectory_agent.initialize()
            self.decision_agent.initialize()
            self.evaluation_agent.initialize()

            self.logger.info("All agents initialized successfully (including EvaluationAgent)")

        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")

    def run_evaluation(self, num_episodes: int = None, start_episode_id: int = None) -> Dict[str, Any]:
        """Run VLN evaluation

        Args:
            num_episodes: Number of episodes to run
            start_episode_id: Starting episode ID (None to start from first)
        """
        # Find starting position
        if start_episode_id is not None:
            start_idx = None
            for i, ep in enumerate(self.episodes):
                if ep.episode_id == start_episode_id:
                    start_idx = i
                    break
            if start_idx is None:
                self.logger.warning(f"Episode {start_episode_id} not found, starting from first")
                start_idx = 0
        else:
            start_idx = 0

        episodes_to_run = self.episodes[start_idx:]
        num_to_run = min(num_episodes or len(episodes_to_run), len(episodes_to_run))

        self.logger.info("=" * 60)
        self.logger.info(f"Starting evaluation - {num_to_run} episodes (from Episode {episodes_to_run[0].episode_id if episodes_to_run else 'N/A'})")
        self.logger.info("=" * 60)

        start_time = time.time()

        for i, episode in enumerate(episodes_to_run[:num_to_run]):
            ep_start = time.time()

            self.logger.info(f"\n[{i+1}/{num_to_run}] Episode {episode.episode_id}")
            self.logger.info(f"  Scene: {episode.scene_id}")
            self.logger.info(f"  Instruction: {episode.instruction[:80]}...")

            result = self._run_episode(episode)
            result.time_elapsed = time.time() - ep_start
            self.results.append(result)

            # Output progress
            self._log_progress(i + 1, num_to_run)

            # Save intermediate results every 10 episodes
            if (i + 1) % 10 == 0:
                self._save_intermediate_results()

        total_time = time.time() - start_time

        # Clean up all Simulator resources
        self._cleanup_sims()

        return self._compile_results(total_time)

    def _run_episode(self, episode: R2REpisode) -> EvaluationResult:
        """Run single episode"""
        trajectory = []
        steps = 0
        success = False
        min_distance = float('inf')
        evaluation_scores = []

        start_time = time.time()

        try:
            # Check if scene is available
            if episode.scene_id in self.scene_paths:
                result = self._run_habitat_episode(episode)
            else:
                self.logger.warning(f"Scene not available: {episode.scene_id}, using simulation mode")
                result = self._run_simulated_episode(episode)

            trajectory = result["trajectory"]
            steps = result["steps"]
            success = result["success"]
            min_distance = result["min_distance"]
            evaluation_scores = result.get("evaluation_scores", [])
            task_level = result.get("task_level", "medium")
            subtask_count = result.get("subtask_count", 0)

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} failed: {e}")
            trajectory = [episode.start_position]
            steps = 0
            min_distance = self._distance(episode.start_position, episode.goal_position)
            task_level = "medium"
            subtask_count = 0

        # Calculate metrics
        final_pos = trajectory[-1] if trajectory else episode.start_position
        distance_to_goal = self._distance(final_pos, episode.goal_position)
        oracle_success = min_distance <= self.config.get("success_distance", 3.0)

        # Trajectory length
        trajectory_length = sum(
            self._distance(trajectory[i-1], trajectory[i])
            for i in range(1, len(trajectory))
        ) if len(trajectory) > 1 else 0.0

        # SPL
        if success and trajectory_length > 0:
            spl = min(episode.geodesic_distance, trajectory_length) / max(trajectory_length, episode.geodesic_distance)
        else:
            spl = 0.0

        # nDTW
        ndtw = self._calculate_ndtw(trajectory, episode.reference_path)
        sdtw = ndtw if success else 0.0

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
            token_usage={"total": 0, "input": 0, "output": 0},  # Local model has no token count
            trajectory=trajectory,
            task_level=task_level,
            subtask_count=subtask_count,
            evaluation_scores=evaluation_scores,
        )

    def _run_habitat_episode(self, episode: R2REpisode) -> Dict[str, Any]:
        """Run episode in Habitat environment"""
        import habitat_sim

        scene_path = self.scene_paths[episode.scene_id]
        sim = None
        trajectory = []
        steps = 0
        success = False
        min_distance = float('inf')
        evaluation_scores = []
        task_level = "medium"
        subtask_count = 0

        try:
            # 1. Get Simulator
            sim = self._get_simulator(episode.scene_id, scene_path)
            if sim is None:
                return self._run_simulated_episode(episode)

            # 2. Set start position
            agent = sim.get_agent(0)
            state = habitat_sim.AgentState()

            start_pos = self._validate_position(sim, episode.start_position)
            if start_pos is None:
                return self._run_simulated_episode(episode)

            state.position = start_pos
            if episode.start_rotation:
                # R2R dataset uses [x, y, z, w] quaternion format
                # habitat_sim uses [x, y, z, w] format (Eigen convention)
                # Direct assignment - habitat_sim handles the coordinate system
                r = episode.start_rotation
                state.rotation = np.array(r, dtype=np.float32)
            agent.set_state(state)

            trajectory = [[float(x) for x in start_pos]]
            min_distance = self._distance(start_pos, episode.goal_position)

            # Calculate initial rotation (yaw angle) from start_rotation quaternion
            initial_yaw = 0.0
            if episode.start_rotation:
                q = episode.start_rotation
                if len(q) == 4:
                    # R2R dataset uses [x, y, z, w] quaternion format
                    # habitat_sim uses [x, y, z, w] format
                    x, y, z, w = q[0], q[1], q[2], q[3]
                    siny_cosp = 2 * (w * y + x * z)
                    cosy_cosp = 1 - 2 * (y * y + z * z)
                    initial_yaw = math.atan2(siny_cosp, cosy_cosp)

            # 3. Create navigation context
            from core.context import NavContextBuilder, VisualFeatures
            from utils.status_reporter import init_reporter
            visual_features = VisualFeatures()

            # Initialize real-time status reporter
            reporter = init_reporter(self.config.get("output_dir", "results"))
            reporter.start_episode(
                episode_id=episode.episode_id,
                instruction=episode.instruction,
                max_steps=self.config.get("max_steps", 100),
                start_y=start_pos[1]
            )
            reporter.update_position(start_pos[0], start_pos[1], start_pos[2])

            context = NavContextBuilder() \
                .with_instruction(episode.instruction) \
                .with_position(tuple(start_pos)) \
                .with_rotation(initial_yaw) \
                .with_visual_features(visual_features) \
                .with_metadata({
                    "episode_id": episode.episode_id,
                    "goal_position": tuple(episode.goal_position),
                    "success_distance": self.config.get("success_distance", 3.0),
                }) \
                .build()

            # 3.5 Initialize episode output
            self.output_manager.start_episode(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                instruction=episode.instruction,
                goal_position=episode.goal_position,
                start_position=list(start_pos),
            )

            # 4. Process instruction - InstructionAgent
            if self.instruction_agent:
                reporter.update_phase("InstructionAgent decomposing instruction...")
                reporter.update_agent("instruction", "thinking")
                instruction_output = self.instruction_agent.process(context)
                context.metadata["instruction_output"] = instruction_output.data
                task_level = instruction_output.data.get("task_level", "medium")
                subtask_count = len(instruction_output.data.get("subtasks", []))
                subtasks = instruction_output.data.get("subtasks", [])
                reporter.update_agent("instruction", "done", subtasks=subtask_count)
                reporter.log(f"Task level: {task_level}, Subtasks: {subtask_count}")
                self.logger.info(f"  Task level: {task_level}, Subtasks: {subtask_count}")
                # Log each subtask with its individual level
                for st in subtasks:
                    self.logger.info(f"    Subtask{st['id']}: [{st['level']}] {st['description'][:50]}...")

                # Start the first subtask (records initial state)
                if context.subtasks:
                    context.start_subtask()
                    self.logger.info(f"  Starting subtask 0: {context.subtasks[0].description[:50]}...")

            # 5. Reset TrajectoryAgent map
            if self.trajectory_agent:
                self.trajectory_agent.reset_map()

            # 6. Reset EvaluationAgent history
            if self.evaluation_agent:
                self.evaluation_agent.reset_history()

            # 7. Reset DecisionAgent stuck detection counter and spatial memory
            if self.decision_agent:
                self.decision_agent.reset_stuck_counter()
                self.decision_agent.reset_spatial_memory()

            # 8. Reset DebateStrategy performance tracker (episode level)
            # Note: This resets episode stats, not cross-episode history
            from strategies.debate import DebateStrategy
            temp_strategy = DebateStrategy(self.config)
            temp_strategy.reset_episode()

            # 9. Clear context history data
            context.rgb_history.clear()
            context.depth_history.clear()
            context.stuck_regions.clear()
            context.is_stuck = False
            context.stuck_counter = 0

            max_steps = self.config.get("max_steps", 100)
            success_distance = self.config.get("success_distance", 3.0)
            task_level = "medium"  # Default

            # Sequence mode related variables
            current_sequence = None
            last_sequence_subtask_id = None
            sequence_step_count = 0  # Current sequence executed steps

            self.logger.info("[SEQUENCE MODE] Action sequence mode enabled")

            # === EMERGENCY MODE SETUP ===
            # Initialize obstacle manager and emergency detector for emergency episodes
            obstacle_manager = None
            emergency_detector = None
            if hasattr(episode, 'obstacle_config') and episode.obstacle_config:
                try:
                    from emergency.dynamic_obstacle_manager import DynamicObstacleManager
                    from emergency.emergency_detector import EmergencyDetector

                    obstacle_manager = DynamicObstacleManager({})
                    obstacle_manager.start_episode(episode.episode_id, [episode.obstacle_config])

                    emergency_detector = EmergencyDetector({})

                    self.logger.info(f"[EMERGENCY] Obstacle configured: type={episode.obstacle_config.get('type')}, "
                                     f"trigger_step={episode.obstacle_config.get('trigger_step')}")
                except Exception as e:
                    self.logger.warning(f"[EMERGENCY] Failed to initialize obstacle manager: {e}")
                    obstacle_manager = None
                    emergency_detector = None

            # ThreadPoolExecutor for parallel execution
            executor = ThreadPoolExecutor(max_workers=2)

            # Import strategies
            from strategies.cot import CoTStrategy
            from strategies.reflection import ReflectionStrategy
            from strategies.debate import DebateStrategy

            # 7. Navigation main loop
            while steps < max_steps:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Step {steps + 1}/{max_steps}")

                # === EMERGENCY: Check and trigger obstacles ===
                if obstacle_manager:
                    triggered = obstacle_manager.check_and_trigger(steps)
                    if triggered:
                        self.logger.warning(f"[EMERGENCY] Obstacle triggered at step {steps}!")
                        obstacle_state = obstacle_manager.get_obstacle_state()
                        context.metadata["obstacle_state"] = obstacle_state

                        # Update emergency detector
                        if emergency_detector:
                            event = emergency_detector.update(
                                perception_output=context.metadata.get("perception_output", {}),
                                obstacle_state=obstacle_state,
                                current_position=tuple(context.position) if context.position else (0, 0, 0),
                                stuck_counter=getattr(self.decision_agent, '_stuck_counter', 0) if self.decision_agent else 0
                            )
                            if event:
                                # Convert EmergencyEvent to dict for context
                                from dataclasses import asdict
                                context.metadata["emergency_signal"] = {
                                    "trigger": True,
                                    "event_type": event.event_type,
                                    "position": event.position,
                                    "severity": event.severity,
                                    "description": event.description,
                                }
                                self.logger.warning(f"[EMERGENCY] Signal sent to DecisionAgent: {event.event_type}")

                # Log current subtask info
                current_subtask = context.get_current_subtask()
                if current_subtask:
                    reporter.update_subtask(
                        current_subtask.id,
                        current_subtask.description,
                        current_subtask.completion_condition
                    )
                    self.logger.info(f"Current subtask [{current_subtask.level}]: {current_subtask.description[:50]}...")
                self.logger.info(f"{'='*60}")

                # Get RGB and Depth images
                rgb_image, depth_image = self._get_observations(sim)

                # Update context images
                context.rgb_image = rgb_image
                context.depth_image = depth_image
                context.metadata['rgb_image'] = rgb_image
                context.metadata['depth_image'] = depth_image

                # Save observation history for stuck analysis
                context.add_observation(rgb_image, depth_image)

                # Save RGB and Depth images to output directory
                self.output_manager.save_rgb_image(rgb_image, steps)
                self.output_manager.save_depth_image(depth_image, steps)

                # Get task level from current subtask
                current_subtask = context.get_current_subtask()
                task_level = current_subtask.level if current_subtask else "medium"

                # Save last strategy result (for next call with history info)
                if not hasattr(self, '_last_strategy_result'):
                    self._last_strategy_result = None

                # ============================================================
                # Sequence mode: Call strategy at subtask start, generate action sequence then execute
                # ============================================================
                from core.action import ActionSequence

                # Check if need to generate new sequence
                need_new_sequence = False

                if current_sequence is None:
                    need_new_sequence = True
                    reason = "no sequence"
                elif current_sequence.is_complete():
                    need_new_sequence = True
                    reason = "sequence complete"
                elif current_subtask and hasattr(current_subtask, 'id'):
                    if current_subtask.id != last_sequence_subtask_id:
                        need_new_sequence = True
                        reason = "subtask switch"

                # Only call strategy and Agents when new sequence is needed
                if need_new_sequence:
                    self.logger.info(f"[SEQUENCE] Generating new sequence: {reason}, difficulty: {task_level}")

                    # Call PerceptionAgent
                    perception_output = None
                    if self.perception_agent:
                        try:
                            reporter.update_phase("PerceptionAgent perceiving environment...")
                            reporter.update_agent("perception", "thinking")
                            perception_result = self.perception_agent.process(context)
                            perception_output = perception_result.data
                            context.metadata["perception_output"] = perception_output
                            reporter.update_agent("perception", "done", output=f"Room:{perception_output.get('room_type','?')}")
                            reporter.log(f"Perception: Room={perception_output.get('room_type','?')}, Objects={len(perception_output.get('objects',[]))}")
                            self.logger.info(f"[PerceptionAgent] Room: {perception_output.get('room_type', 'unknown')}")
                            self.logger.info(f"[PerceptionAgent] Objects: {[o.get('object', o.get('name', 'unknown')) for o in perception_output.get('objects', [])[:5]]}")
                            self.logger.info(f"[PerceptionAgent] nav_hint: {perception_output.get('nav_hint', '')}")
                        except Exception as e:
                            reporter.update_agent("perception", "error", output=str(e))
                            self.logger.warning(f"[PerceptionAgent] Execution failed: {e}")
                            perception_output = {}

                    # Call TrajectoryAgent
                    trajectory_output = None
                    if self.trajectory_agent:
                        try:
                            reporter.update_phase("TrajectoryAgent analyzing trajectory...")
                            reporter.update_agent("trajectory", "thinking")
                            trajectory_result = self.trajectory_agent.process(context)
                            trajectory_output = trajectory_result.data
                            context.metadata["trajectory_output"] = trajectory_output
                            reporter.update_agent("trajectory", "done")
                            reporter.log(f"Trajectory: Traveled {trajectory_output.get('distance_traveled', 0):.1f}m")
                            self.logger.info(f"[TrajectoryAgent] Distance traveled: {trajectory_output.get('distance_traveled', 0):.1f}m")
                        except Exception as e:
                            reporter.update_agent("trajectory", "error", output=str(e))
                            self.logger.warning(f"[TrajectoryAgent] Execution failed: {e}")
                            trajectory_output = {}

                    # Call InstructionAgent (get subtask semantic analysis)
                    instruction_output = None
                    if self.instruction_agent:
                        try:
                            instruction_result = self.instruction_agent.process(context)
                            instruction_output = instruction_result.data
                            context.metadata["instruction_output"] = instruction_output
                        except Exception as e:
                            self.logger.warning(f"[InstructionAgent] Execution failed: {e}")
                            instruction_output = {}

                    # Strategy config (pass remote LLM settings)
                    strategy_config = {
                        "use_remote": self.config.get("use_remote_llm", False),
                        "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
                    }

                    # Select strategy based on subtask difficulty
                    if task_level == "easy":
                        # Easy task: No strategy execution, DecisionAgent directly synthesizes info
                        strategy_result = None
                        self.logger.info("[SEQUENCE] Easy task: Skip strategy, DecisionAgent decides directly")
                    elif task_level == "medium":
                        strategy = CoTStrategy(config=strategy_config)
                        self.logger.info(f"[SEQUENCE] Using strategy: {strategy.name}")
                        reporter.update_phase(f"{strategy.name} strategy executing...")
                        reporter.log(f"Strategy: {strategy.name}")
                        agents_list = [self.perception_agent, self.trajectory_agent,
                                       self.instruction_agent, self.evaluation_agent]
                        # Pass last strategy result
                        strategy_result = strategy.execute(context, agents_list, self._last_strategy_result)
                    else:  # hard
                        strategy = DebateStrategy(config=strategy_config)
                        self.logger.info(f"[SEQUENCE] Using strategy: {strategy.name}")
                        reporter.update_phase(f"{strategy.name} strategy executing...")
                        reporter.log(f"Strategy: {strategy.name}")
                        agents_list = [self.perception_agent, self.trajectory_agent,
                                       self.instruction_agent, self.evaluation_agent]
                        strategy_result = strategy.execute(context, agents_list)

                    # Easy task has no strategy reasoning log
                    if strategy_result:
                        reporter.log(f"Strategy reasoning: {strategy_result.reasoning[:80] if strategy_result.reasoning else 'none'}")
                        self.logger.info(f"[{strategy.name}] Reasoning: {strategy_result.reasoning[:100] if strategy_result.reasoning else 'none'}")
                        # Print to console
                        print(f"\n[{strategy.name} Strategy] Reasoning: {strategy_result.reasoning[:300] if strategy_result.reasoning else 'none'}")

                    # Generate action sequence
                    reporter.update_phase("DecisionAgent generating action sequence...")
                    reporter.update_agent("decision", "thinking")
                    try:
                        # Pass trajectory_agent for history query
                        if strategy_result and self.trajectory_agent:
                            strategy_result.metadata["trajectory_agent"] = self.trajectory_agent

                        current_sequence = self.decision_agent.generate_action_sequence(
                            context, strategy_result, current_subtask
                        )
                    except RuntimeError as e:
                        # LLM service unavailable, stop experiment
                        self.logger.error(f"[SEQUENCE] LLM service error, stopping experiment: {e}")
                        reporter.update_agent("decision", "error", output=str(e))
                        raise  # Re-raise exception, stop experiment
                    reporter.update_agent("decision", "done", sequence_progress="0%")

                    # Save decision and strategy output to context metadata for logging
                    context.metadata["decision_output"] = {
                        "sequence": [a[0].name if isinstance(a, tuple) else str(a) for a in current_sequence.actions],
                        "reasoning": current_sequence.reasoning,
                        "subtask_completed": current_sequence.subtask_completed,
                        "confidence": current_sequence.confidence,
                    }
                    if strategy_result:
                        from strategies.base_strategy import StrategyResult
                        if isinstance(strategy_result, StrategyResult):
                            context.metadata["strategy_output"] = {
                                "strategy": strategy.name,
                                "success": strategy_result.success,
                                "reasoning": strategy_result.reasoning,
                                "confidence": strategy_result.confidence,
                            }
                            # Save strategy result for next call
                            self._last_strategy_result = strategy_result
                    reporter.log(f"Generated sequence: {len(current_sequence.actions)} steps, completed={current_sequence.subtask_completed}")
                    last_sequence_subtask_id = current_subtask.id if current_subtask else None
                    sequence_step_count = 0

                # Check if sequence needs to be aborted
                # Initialize history record variable (used when first generating sequence)
                if 'last_sequence_start_pos' not in locals():
                    last_sequence_start_pos = None
                    last_sequence_actions = None

                if current_sequence:
                    # Get current action name for stuck detection
                    current_action_name = None
                    if current_sequence.actions and current_sequence.current_index < len(current_sequence.actions):
                        action_tuple = current_sequence.actions[current_sequence.current_index]
                        if action_tuple:
                            from core.action import ActionType
                            action_type = action_tuple[0]
                            action_map_inv = {
                                ActionType.MOVE_FORWARD: "move_forward",
                                ActionType.TURN_LEFT: "turn_left",
                                ActionType.TURN_RIGHT: "turn_right",
                                ActionType.STOP: "stop",
                            }
                            current_action_name = action_map_inv.get(action_type, "forward")

                    should_abort, abort_reason = self.decision_agent.check_sequence_abort(
                        context, current_sequence, depth_image, current_action_name
                    )

                    if should_abort:
                        self.logger.info(f"[SEQUENCE] Aborting sequence: {abort_reason}")
                        current_sequence = None

                        # Reset stuck counter
                        self.decision_agent._stuck_counter = 0

                        # Check if we have depth info for obstacle avoidance
                        depth_clear = self._check_depth_clear_direction(depth_image) if depth_image is not None else None

                        if depth_clear:
                            # Use simple turn sequence based on depth (fast, no LLM call)
                            self.logger.info(f"[SEQUENCE] Quick turn to {depth_clear} (depth-based obstacle avoidance)")
                            turn_action = "turn_left" if depth_clear == "left" else "turn_right"
                            # Generate simple 5-step sequence: turn + explore
                            from core.action import ActionSequence
                            if depth_clear == "left":
                                actions = [(ActionType.TURN_LEFT, 1)] * 3 + [(ActionType.MOVE_FORWARD, 1)] * 2
                            else:
                                actions = [(ActionType.TURN_RIGHT, 1)] * 3 + [(ActionType.MOVE_FORWARD, 1)] * 2
                            current_sequence = ActionSequence(
                                subtask_id=current_subtask.id if current_subtask else 0,
                                subtask_description=current_subtask.description if current_subtask else "stuck recovery",
                                actions=actions,
                                estimated_steps=len(actions),
                                abort_conditions={"stuck_for_steps": 5},
                                reasoning=f"Stuck recovery: turn {depth_clear} based on depth analysis",
                                confidence=0.9,
                                subtask_completed=False
                            )
                        else:
                            # Use Debate strategy only when depth is not available
                            self.logger.info("[SEQUENCE] Stuck triggers Debate re-planning...")
                            strategy_config = {
                                "use_remote": self.config.get("use_remote_llm", False),
                                "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
                            }
                            strategy = DebateStrategy(config=strategy_config)
                            agents_list = [self.perception_agent, self.trajectory_agent,
                                           self.instruction_agent, self.evaluation_agent]
                            strategy_result = strategy.execute(context, agents_list)

                            # Generate new sequence
                            try:
                                current_sequence = self.decision_agent.generate_action_sequence(
                                    context, strategy_result, current_subtask
                                )
                            except RuntimeError as e:
                                self.logger.error(f"[SEQUENCE] LLM service error, stopping experiment: {e}")
                                raise
                            self.logger.info(f"[SEQUENCE] New sequence after Debate: {current_sequence.reasoning[:50] if current_sequence else 'N/A'}")
                            # Save decision output to context metadata for logging
                            context.metadata["decision_output"] = {
                            "sequence": [a[0].name if isinstance(a, tuple) else str(a) for a in current_sequence.actions],
                                "reasoning": current_sequence.reasoning,
                                "subtask_completed": current_sequence.subtask_completed,
                                "confidence": current_sequence.confidence,
                            }
                            # Save strategy output for stuck recovery
                            context.metadata["strategy_output"] = {
                                "strategy": strategy.name,
                                "success": strategy_result.success if strategy_result else False,
                                "reasoning": strategy_result.reasoning if strategy_result else "Stuck recovery",
                                "confidence": strategy_result.confidence if strategy_result else 0.5,
                            }

                # Get next action from sequence
                action_name = "move_forward"  # Default action
                if current_sequence:
                    from core.action import ActionType
                    next_action_type = current_sequence.get_next_action()

                    # Sequence complete, check if subtask is done
                    if next_action_type is None:
                        if current_sequence.subtask_completed:
                            reporter.update_phase("Subtask completed!")
                            reporter.log(f"Subtask completed: {current_sequence.subtask_description[:40]}")
                            self.logger.info(f"[SEQUENCE] LLM judged subtask completed: {current_sequence.subtask_description[:40]}")
                            if context.advance_subtask():
                                current_subtask = context.get_current_subtask()
                                if current_subtask:
                                    reporter.update_subtask(
                                        current_subtask.id,
                                        current_subtask.description,
                                        current_subtask.completion_condition
                                    )
                                self.logger.info(f"[SEQUENCE] Moving to next subtask: {current_subtask.description[:40] if current_subtask else 'N/A'}")
                            else:
                                reporter.log("All subtasks completed!")
                                self.logger.info("[SEQUENCE] All subtasks completed, preparing to stop")
                                action_name = "stop"
                        else:
                            reporter.log("Sequence complete but subtask not done, re-planning")
                            self.logger.info(f"[SEQUENCE] Sequence complete but subtask not done, re-planning")

                        # Record action sequence history
                        if self.trajectory_agent and last_sequence_start_pos and last_sequence_actions:
                            self.logger.info(f"[ACTION HISTORY] Recording sequence: {len(last_sequence_actions)} actions, start={last_sequence_start_pos}")
                            current_pos = context.position
                            current_dist = context.get_distance_to_goal() if hasattr(context, 'get_distance_to_goal') else 5.0
                            stuck_triggered = hasattr(context, 'stuck_counter') and context.stuck_counter > 0

                            self.trajectory_agent.record_action_sequence(
                                step_id=last_sequence_subtask_id if last_sequence_subtask_id else 0,
                                actions=last_sequence_actions,
                                start_pos=last_sequence_start_pos,
                                end_pos=current_pos,
                                goal_distance_before=last_sequence_start_dist,
                                goal_distance_after=current_dist,
                                stuck_triggered=stuck_triggered,
                                perception_feedback=context.metadata.get("perception_output", {}).get("scene_description", "")[:50]
                            )

                        current_sequence = None
                        last_sequence_subtask_id = None
                        last_sequence_start_pos = None
                        last_sequence_actions = None
                        continue  # Skip to next loop iteration to regenerate sequence

                    if next_action_type:
                        action_map = {
                            ActionType.MOVE_FORWARD: "move_forward",
                            ActionType.TURN_LEFT: "turn_left",
                            ActionType.TURN_RIGHT: "turn_right",
                            ActionType.STOP: "stop",
                            ActionType.LOOK_UP: "look_up",
                            ActionType.LOOK_DOWN: "look_down",
                        }
                        action_name = action_map.get(next_action_type, "move_forward")
                        sequence_step_count += 1

                        reporter.update_step(steps + 1)
                        reporter.update_phase(f"Executing: {action_name}")
                        reporter.log_action(action_name)
                        reporter.update_agent("decision", "done", sequence_progress=f"{current_sequence.get_progress():.0%}")

                        self.logger.info(f"[SEQUENCE] Executing action: {action_name} "
                                       f"(progress: {current_sequence.get_progress():.0%}, "
                                       f"remaining: {current_sequence.get_remaining_steps()} steps)")

                        context.metadata["sequence_output"] = {
                            "subtask": current_sequence.subtask_description[:50],
                            "progress": current_sequence.get_progress(),
                            "action": action_name,
                        }

                # Execute action
                if action_name == "stop":
                    self.logger.info(f"  Agent stopped voluntarily at step {steps}")
                    break

                try:
                    sim.step(action_name)
                except Exception as e:
                    self.logger.warning(f"  Action execution failed: {e}")

                # Update state
                state = agent.get_state()
                pos = [float(x) for x in state.position]
                trajectory.append(pos)
                steps += 1

                reporter.update_position(pos[0], pos[1], pos[2])
                reporter.update_step(steps)

                context.position = tuple(pos)
                context.add_trajectory_point(tuple(pos))

                # Update rotation from simulator (convert quaternion to yaw angle)
                # Habitat uses quaternions for rotation, we need yaw (rotation around Y axis)
                if hasattr(state, 'rotation') and state.rotation is not None:
                    # Quaternion to euler yaw angle
                    # Habitat uses numpy-quaternion which has components: w, x, y, z
                    try:
                        import quaternion
                        q = state.rotation
                        # Get quaternion components as numpy array [w, x, y, z]
                        q_arr = quaternion.as_float_array(q)  # Returns [w, x, y, z]
                        w, x, y, z = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
                        # Calculate yaw (rotation around Y axis)
                        siny_cosp = 2 * (w * y + x * z)
                        cosy_cosp = 1 - 2 * (y * y + z * z)
                        yaw = math.atan2(siny_cosp, cosy_cosp)
                        context.rotation = yaw
                    except Exception as e:
                        self.logger.warning(f"Failed to extract rotation: {e}")

                dist = self._distance(pos, episode.goal_position)
                min_distance = min(min_distance, dist)

                # Record step output
                self.output_manager.add_step_output(
                    step=steps,
                    action=action_name,
                    position=tuple(pos),
                    rotation=context.rotation,
                    distance_to_goal=dist,
                    perception_output=context.metadata.get("perception_output"),
                    trajectory_output=context.metadata.get("trajectory_output"),
                    decision_output=context.metadata.get("decision_output"),
                    evaluation_output=context.metadata.get("evaluation_output"),
                    strategy_output=context.metadata.get("strategy_output"),
                    current_subtask={"id": current_subtask.id, "description": current_subtask.description, "level": current_subtask.level} if current_subtask else None,
                )

                if dist <= success_distance:
                    success = True
                    self.logger.info(f"  Successfully reached goal at step {steps}")
                    break

        except Exception as e:
            self.logger.error(f"  Habitat error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up ThreadPoolExecutor
            try:
                executor.shutdown(wait=False)
            except NameError:
                pass  # executor wasn't created yet

            # Save episode output
            try:
                # Get final position
                final_pos = pos if 'pos' in locals() else (trajectory[-1] if trajectory else start_pos)
                self.output_manager.finish_episode(
                    success=success,
                    final_distance=self._distance(final_pos, episode.goal_position),
                    min_distance=min_distance,
                    steps=steps,
                    trajectory=trajectory,
                    task_level=task_level,
                    subtasks=context.metadata.get("instruction_output", {}).get("subtasks", []),
                    goal_position=episode.goal_position,
                    reference_path=episode.reference_path,
                )
                # Try to create summary video (if enabled)
                if self.enable_video:
                    self.output_manager.create_summary_video(fps=5)
            except Exception as e:
                self.logger.error(f"Failed to save episode output: {e}")

            # Save context metadata for emergency evaluation metrics
            try:
                self._last_context_metadata = dict(context.metadata)
            except Exception as e:
                self.logger.debug(f"Failed to save context metadata: {e}")
                self._last_context_metadata = {}

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
            "evaluation_scores": evaluation_scores,
            "task_level": task_level,
            "subtask_count": subtask_count,
        }

    def _get_observations(self, sim: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Get RGB and Depth images from Habitat"""
        try:
            observations = sim.get_sensor_observations(0)

            rgb = observations.get("rgb", None)
            depth = observations.get("depth", None)

            if rgb is not None:
                rgb = np.array(rgb)
                # Handle different image formats
                if rgb.dtype == np.uint8:
                    pass  # Already correct
                elif rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)

                # Handle RGBA -> RGB conversion (Habitat outputs RGBA by default)
                if rgb.ndim == 3 and rgb.shape[-1] == 4:
                    rgb = rgb[:, :, :3]  # Take only RGB channels, drop Alpha
                    self.logger.debug(f"Converted RGBA to RGB: {rgb.shape}")
            else:
                rgb = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

            if depth is not None:
                depth = np.array(depth).squeeze()
            else:
                depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)

            return rgb, depth

        except Exception as e:
            self.logger.warning(f"Failed to get images: {e}")
            return (
                np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8),
                np.zeros((self.image_height, self.image_width), dtype=np.float32)
            )

    def _get_simulator(self, scene_id: str, scene_path: str) -> Any:
        """Get or create Simulator"""
        import habitat_sim

        if scene_id in self._sim_cache:
            self._current_scene = scene_id
            return self._sim_cache[scene_id]

        try:
            self.logger.info(f"  Creating Simulator: {scene_id}")

            # Configure sensors
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "rgb"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [self.image_height, self.image_width]
            color_sensor_spec.position = np.array([0.0, 1.5, 0.0])

            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.image_height, self.image_width]
            depth_sensor_spec.position = np.array([0.0, 1.5, 0.0])

            # Configure Agent
            agent_cfg = habitat_sim.AgentConfiguration(
                height=1.5,
                radius=0.1,
                sensor_specifications=[color_sensor_spec, depth_sensor_spec],
                action_space={
                    "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.25)),
                    "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
                    "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
                    "stop": habitat_sim.ActionSpec("stop", habitat_sim.ActuationSpec(amount=0.0)),
                }
            )

            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = scene_path
            sim_cfg.enable_physics = False
            sim_cfg.gpu_device_id = 0

            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            sim = habitat_sim.Simulator(cfg)

            self._sim_cache[scene_id] = sim
            self._current_scene = scene_id

            return sim

        except Exception as e:
            self.logger.error(f"  Simulator creation failed: {e}")
            return None

    def _validate_position(self, sim: Any, position: List[float]) -> Optional[List[float]]:
        """Validate if position is navigable"""
        try:
            if not hasattr(sim, 'pathfinder') or sim.pathfinder is None:
                return [float(x) for x in position]

            if sim.pathfinder.is_navigable(position):
                return [float(x) for x in position]

            snap_pos = sim.pathfinder.snap_point(position)
            if sim.pathfinder.is_navigable(snap_pos):
                return [float(x) for x in snap_pos]

            random_pos = sim.pathfinder.get_random_navigable_point()
            return [float(x) for x in random_pos]

        except Exception as e:
            self.logger.warning(f"Position validation failed: {e}")
            return [float(x) for x in position]

    def _should_call_evaluation(self, task_level: str, step_count: int) -> bool:
        """
        Decide whether to call EvaluationAgent based on task level.

        Args:
            task_level: Task level (easy/medium/hard)
            step_count: Current step count

        Returns:
            True if evaluation should be called
        """
        if task_level == "easy":
            return False  # Easy task does not call evaluation
        elif task_level == "medium":
            return step_count % 5 == 0  # Evaluate every 5 steps
        elif task_level == "hard":
            return True  # Evaluate every step
        return False

    def _select_strategy(self, task_level: str):
        """
        Select reasoning strategy based on task level.

        Args:
            task_level: Task level (easy/medium/hard)

        Returns:
            Corresponding strategy instance, returns None for easy tasks
        """
        strategy_config = {
            "use_remote": self.config.get("use_remote_llm", False),
            "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
        }
        if task_level == "easy":
            # Easy task: No strategy execution, DecisionAgent directly synthesizes info
            return None
        elif task_level == "medium":
            return CoTStrategy(config=strategy_config)
        else:  # hard
            return DebateStrategy(config=strategy_config)

    def _run_simulated_episode(self, episode: R2REpisode) -> Dict[str, Any]:
        """Run episode in simulation mode"""
        trajectory = [episode.start_position.copy()]
        current = list(episode.start_position)
        steps = 0
        success = False
        min_distance = self._distance(current, episode.goal_position)
        evaluation_scores = []
        task_level = "medium"
        subtask_count = 0

        from core.context import NavContextBuilder
        context = NavContextBuilder() \
            .with_instruction(episode.instruction) \
            .with_position(tuple(current)) \
            .with_metadata({
                "goal_position": tuple(episode.goal_position),
                "success_distance": self.config.get("success_distance", 3.0),
            }) \
            .build()

        # Process instruction
        if self.instruction_agent:
            instruction_output = self.instruction_agent.process(context)
            context.metadata["instruction_output"] = instruction_output.data
            task_level = instruction_output.data.get("task_level", "medium")
            subtask_count = len(instruction_output.data.get("subtasks", []))

        max_steps = self.config.get("max_steps", 100)
        success_distance = self.config.get("success_distance", 3.0)

        while steps < max_steps:
            # Simple simulation decision
            if self.decision_agent:
                decision_output = self.decision_agent.process(context)
                action_name = decision_output.data.get("action", "forward")
            else:
                # Move towards goal direction
                dx = episode.goal_position[0] - current[0]
                dz = episode.goal_position[2] - current[2]
                dist = math.sqrt(dx*dx + dz*dz)
                step_size = 0.25

                if dist > 0:
                    current[0] += (dx / dist) * step_size
                    current[2] += (dz / dist) * step_size

                trajectory.append(current.copy())
                steps += 1

                dist = self._distance(current, episode.goal_position)
                min_distance = min(min_distance, dist)

                if dist <= success_distance:
                    success = True
                    break

                context.position = tuple(current)
                continue

            if action_name == "stop":
                break

            step_size = 0.25
            if action_name == "move_forward":
                dx = episode.goal_position[0] - current[0]
                dz = episode.goal_position[2] - current[2]
                dist = math.sqrt(dx*dx + dz*dz)
                if dist > 0:
                    current[0] += (dx / dist) * step_size
                    current[2] += (dz / dist) * step_size

            trajectory.append(current.copy())
            steps += 1

            dist = self._distance(current, episode.goal_position)
            min_distance = min(min_distance, dist)

            if dist <= success_distance:
                success = True
                break

            context.position = tuple(current)

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
            "evaluation_scores": evaluation_scores,
            "task_level": task_level,
            "subtask_count": subtask_count,
        }

    def _distance(self, p1: List[float], p2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:3], p2[:3])))

    def _calculate_ndtw(self, trajectory: List[List[float]], reference: List[List[float]]) -> float:
        if not trajectory or not reference:
            return 0.0

        n, m = len(trajectory), len(reference)
        dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._distance(trajectory[i-1], reference[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

        max_len = max(n, m)
        return math.exp(-dtw[n][m] / max_len / 5.0) if max_len > 0 else 0.0

    def _cleanup_sims(self) -> None:
        """Clean up all Simulators"""
        self.logger.info(f"Cleaning up {len(self._sim_cache)} Simulators")
        for scene_id, sim in self._sim_cache.items():
            try:
                if sim is not None:
                    sim.close()
            except Exception as e:
                self.logger.warning(f"Failed to close Simulator: {e}")

        self._sim_cache.clear()
        self._current_scene = ""
        gc.collect()

    def _log_progress(self, current: int, total: int) -> None:
        """Output progress"""
        if not self.results:
            return

        successes = sum(1 for r in self.results if r.success)
        sr = successes / len(self.results) * 100
        avg_spl = sum(r.spl for r in self.results) / len(self.results)
        avg_ne = sum(r.distance_to_goal for r in self.results) / len(self.results)
        avg_steps = sum(r.steps for r in self.results) / len(self.results)

        self.logger.info(f"  Progress: {current}/{total} | SR: {sr:.1f}% | SPL: {avg_spl:.3f} | NE: {avg_ne:.2f}m | Steps: {avg_steps:.1f}")

    def _save_intermediate_results(self) -> None:
        """Save intermediate results"""
        output_path = Path(self.config.get("output", "results_vln.json"))
        with open(output_path, "w") as f:
            json.dump({
                "status": "in_progress",
                "episodes_completed": len(self.results),
                "results": [r.to_dict() for r in self.results],
            }, f, indent=2)

    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """Compile final results"""
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
            "total_time_seconds": total_time,
            "avg_time_per_episode": total_time / total,
        }

        # Statistics by task level
        by_task_level = {}
        for r in self.results:
            level = r.task_level
            if level not in by_task_level:
                by_task_level[level] = {"count": 0, "successes": 0, "spl_sum": 0.0}
            by_task_level[level]["count"] += 1
            if r.success:
                by_task_level[level]["successes"] += 1
            by_task_level[level]["spl_sum"] += r.spl

        for level in by_task_level:
            data = by_task_level[level]
            data["success_rate"] = data["successes"] / data["count"]
            data["spl"] = data["spl_sum"] / data["count"]

        return {
            "summary": summary,
            "by_task_level": by_task_level,
            "episodes": [r.to_dict() for r in self.results],
            "config": {
                "mp3d_path": str(self.mp3d_path),
                "r2r_path": str(self.r2r_path),
                "max_steps": self.config.get("max_steps", 100),
                "success_distance": self.config.get("success_distance", 3.0),
                "use_remote_llm": self.config.get("use_remote_llm", False),
                "llm_server": self.config.get("llm_server", "http://localhost:8000"),
            },
            "timestamp": datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(description="Multi-agent R2R VLN Evaluation")
    parser.add_argument("--mp3d-path", type=str, default="/data/WZ/Dataset/mp3d_dataset/mp3d")
    parser.add_argument("--r2r-path", type=str, default="/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate")
    parser.add_argument("--start-episode", type=int, default=None, help="Starting episode ID (default: start from first)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--success-distance", type=float, default=3.0, help="Success distance threshold (meters)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename (default: results_timestamp.json)")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-int8", action="store_true", default=True, help="Use INT8 quantization")

    # Remote LLM arguments for dual-environment IPC
    parser.add_argument("--use-remote-llm", action="store_true", default=False,
                        help="Use remote LLM service (Python 3.10 environment)")
    parser.add_argument("--llm-server", type=str, default="http://localhost:8000",
                        help="Remote LLM server address")
    parser.add_argument("--remote-timeout", type=float, default=120.0,
                        help="Remote LLM request timeout (seconds)")

    # SiliconFlow API arguments
    parser.add_argument("--use-siliconflow", action="store_true", default=False,
                        help="Use SiliconFlow API for text LLM, local VLM server for vision")
    parser.add_argument("--siliconflow-api-key", type=str, default=None,
                        help="SiliconFlow API key (default: built-in key)")
    parser.add_argument("--siliconflow-model", type=str, default="Qwen/Qwen3.5-397B-A17B",
                        help="SiliconFlow model name (default: Qwen/Qwen3.5-397B-A17B)")
    parser.add_argument("--vlm-server", type=str, default="http://localhost:8000",
                        help="Local VLM server URL for vision tasks (used with --use-siliconflow)")

    # Strategy mode arguments
    parser.add_argument("--use-strategy-mode", action="store_true", default=False,
                        help="Use strategy mode for decision making (CoT/Reflection/Debate)")

    # Sequence mode arguments (subtask-level planning)
    parser.add_argument("--use-sequence-mode", action="store_true", default=False,
                        help="Use action sequence mode (subtask-level planning, reduces LLM calls)")


    # Sequence length configuration
    parser.add_argument("--sequence-length", type=int, default=5,
                        help="Action sequence length (default 5 steps, used when adaptive-sequence=False)")
    parser.add_argument("--adaptive-sequence", action="store_true", default=False,
                        help="Enable adaptive sequence length (LLM decides step count)")
    parser.add_argument("--min-sequence-length", type=int, default=3,
                        help="Minimum sequence length (default 2 steps, used when adaptive-sequence=True)")
    parser.add_argument("--max-sequence-length", type=int, default=10,
                        help="Maximum sequence length (default 20 steps, used when adaptive-sequence=True)")
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for episode visual images, trajectory maps and agent outputs")

    args = parser.parse_args()

    # Create timestamped session directory under results/
    session_timestamp = datetime.now().strftime("%Y-%m%d-%H%M")
    # Always create timestamped directory under results/
    args.output_dir = f"results/episode-{session_timestamp}"

    # Ensure the session directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename in session directory
    if args.output is None:
        args.output = os.path.join(args.output_dir, "results.json")
    else:
        # If output path is relative, put it in session directory
        if not os.path.isabs(args.output):
            args.output = os.path.join(args.output_dir, args.output)

    config = {
        "mp3d_path": args.mp3d_path,
        "r2r_path": args.r2r_path,
        "max_steps": args.max_steps,
        "success_distance": args.success_distance,
        "output": args.output,
        "device": args.device,
        "use_int8": args.use_int8,
        "use_remote_llm": args.use_remote_llm,
        "llm_server": args.llm_server,
        "remote_timeout": args.remote_timeout,
        "use_siliconflow": args.use_siliconflow,
        "siliconflow_api_key": args.siliconflow_api_key,
        "siliconflow_model": args.siliconflow_model,
        "vlm_server_url": args.vlm_server,
        "use_strategy_mode": args.use_strategy_mode,
        "use_sequence_mode": args.use_sequence_mode,
        "sequence_length": args.sequence_length,
        "adaptive_sequence": args.adaptive_sequence,
        "min_sequence_length": args.min_sequence_length,
        "max_sequence_length": args.max_sequence_length,
        "output_dir": args.output_dir,
    }

    print("=" * 70)
    print("Multi-agent R2R VLN Evaluation Experiment")
    print("=" * 70)
    print(f"MP3D Path: {args.mp3d_path}")
    print(f"R2R Data: {args.r2r_path}")
    print(f"Start Episode: {args.start_episode if args.start_episode else 'from beginning'}")
    print(f"Episode Count: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Success Distance: {args.success_distance}m")
    print(f"Output File: {args.output}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"INT8 Quantization: {args.use_int8}")
    print(f"Remote LLM: {'enabled' if args.use_remote_llm else 'disabled'}")
    if args.use_remote_llm:
        print(f"LLM Server: {args.llm_server}")
    print(f"SiliconFlow API: {'enabled' if args.use_siliconflow else 'disabled'}")
    if args.use_siliconflow:
        print(f"SiliconFlow Model: {args.siliconflow_model}")
        print(f"Local VLM Server: {args.vlm_server}")
    print(f"Strategy Mode: {'enabled' if args.use_strategy_mode else 'disabled'}")
    print("=" * 70)

    evaluator = MultiAgentVLNEvaluator(config, log_level=args.log_level)
    evaluator.initialize()

    results = evaluator.run_evaluation(num_episodes=args.episodes, start_episode_id=args.start_episode)

    # Output results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    summary = results.get("summary", {})
    print(f"Episodes: {summary.get('num_episodes', 0)}")
    print(f"Success Rate (SR): {summary.get('success_rate', 0)*100:.1f}%")
    print(f"Oracle Success Rate: {summary.get('oracle_success_rate', 0)*100:.1f}%")
    print(f"SPL: {summary.get('spl', 0):.3f}")
    print(f"nDTW: {summary.get('nDTW', 0):.3f}")
    print(f"SDTW: {summary.get('SDTW', 0):.3f}")
    print(f"Navigation Error (NE): {summary.get('avg_distance_to_goal', 0):.2f}m")
    print(f"Avg Steps: {summary.get('avg_steps', 0):.1f}")
    print(f"Total Time: {summary.get('total_time_seconds', 0):.1f}s")

    # Output by task level
    by_level = results.get("by_task_level", {})
    if by_level:
        print("\nStatistics by task level:")
        for level, data in by_level.items():
            print(f"  {level}: {data['count']} episodes, SR: {data['success_rate']*100:.1f}%, SPL: {data['spl']:.3f}")

    print("=" * 70)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()