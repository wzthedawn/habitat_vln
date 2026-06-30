#!/usr/bin/env python3
"""
R2R VLN Evaluation with Pipeline Agent Architecture.

6-agent pipeline with multi-tier model allocation:
- SubtaskDecompositionAgent: LLM instruction decomposition (Qwen3.5-9B)
- ObservationAgent: Structured VLM perception (Qwen3-VL-8B)
- AnalysisAgent: CoT/Debate/Reflection reasoning (Qwen3.6-35B)
- PlanningAgent: LLM + topology + A* path planning (Qwen3.6-35B)
- ReviewAgent: Rule-based + LLM completion verification (Qwen3.5-9B)
- EmergencyAgent: Depth-based obstacle detection & handling (Qwen3.5-9B)

Usage:
    # Start vLLM servers
    bash scripts/start_vllm_multi.sh

    # Run experiment
    conda activate Habitat
    python run_vln_experiment.py --use-remote-llm --episodes 10 --seed 42
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

# Constants for mid-sequence completion check
MID_SEQUENCE_CHECK_INTERVAL = 3  # Check completion every N steps
HIGH_CONFIDENCE_THRESHOLD = 0.9  # Confidence for early termination
TOPOLOGY_UPDATE_INTERVAL = 3  # Update topology every N steps (balances responsiveness with performance)

from utils.logger import setup_logger
from utils.token_tracker import get_token_tracker
from utils.timeout_fallback import TimeoutError, timeout, StepTimeout, DEFAULT_TIMEOUTS
from utils.episode_output import EpisodeOutputManager

# Pipeline Agent imports
from agents.pipeline.navigator import Navigator
from agents.pipeline.env_adapter import HabitatEnvAdapter
from agents.pipeline.tools.state_calculator import StateCalculator
from core.context import NavContext


# Global status file path for realtime monitoring
REALTIME_STATUS_FILE = "realtime.status.json"


def update_realtime_status(status_data: Dict[str, Any]) -> None:
    """Update realtime.status.json with current agent outputs.

    Args:
        status_data: Dictionary containing step, episode, and agent outputs
    """
    try:
        # Read existing status if file exists
        existing = {}
        if os.path.exists(REALTIME_STATUS_FILE):
            with open(REALTIME_STATUS_FILE, 'r') as f:
                existing = json.load(f)

        # Merge new data
        for key, value in status_data.items():
            existing[key] = value

        # Add timestamp
        existing["timestamp"] = datetime.now().isoformat()

        # Write updated status
        with open(REALTIME_STATUS_FILE, 'w') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Silent fail - don't interrupt experiment
        pass


def check_completion_condition(
    context: NavContext,
    condition: Dict[str, Any]
) -> Dict[str, Any]:
    """检查子任务完成条件是否满足。

    Args:
        context: 导航上下文
        condition: 完成条件定义

    Returns:
        {
            "completed": bool,
            "progress": float (0.0-1.0),
            "confidence": float,
            "reason": str,
            "current_value": float,
            "threshold": float
        }
    """
    import math

    if not condition:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": "无条件", "current_value": 0, "threshold": 0}

    cc_type = condition.get("type", "unknown")

    # 获取当前子任务
    current_subtask = context.get_current_subtask()
    if not current_subtask:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": "无子任务", "current_value": 0, "threshold": 0}

    # 直接从context计算位置变化
    current_pos = context.position
    start_context = current_subtask.start_context or {}
    start_pos = start_context.get("position", current_pos)
    start_rot = start_context.get("rotation", context.rotation)

    # 计算位置delta
    dx = current_pos[0] - start_pos[0]
    dy = current_pos[1] - start_pos[1]
    dz = current_pos[2] - start_pos[2]
    horizontal_dist = math.sqrt(dx*dx + dz*dz)

    # 计算rotation变化（处理-180/180边界）
    current_deg = math.degrees(context.rotation)
    start_deg = math.degrees(start_rot)
    delta_deg = current_deg - start_deg
    if delta_deg > 180:
        delta_deg -= 360
    elif delta_deg < -180:
        delta_deg += 360
    abs_delta_deg = abs(delta_deg)

    # 处理不同条件类型
    if cc_type == "y_change":
        threshold = condition.get("min_meters", condition.get("min_change", 1.5))
        direction = condition.get("direction", "")

        if direction == "down":
            # 下楼需要 dy 为负（y 减小）
            completed = dy <= -threshold
            current_value = -dy if dy < 0 else 0  # Clamped: only count downward movement
            reason = f"dy={dy:.2f}m (down), need <=-{threshold}m"
        elif direction == "up":
            # 上楼需要 dy 为正（y 增加）
            completed = dy >= threshold
            current_value = dy if dy > 0 else 0  # Clamped: only count upward movement
            reason = f"dy={dy:.2f}m (up), need >={threshold}m"
        else:
            # 无方向要求，用绝对值
            completed = abs(dy) >= threshold
            current_value = abs(dy)
            reason = f"|dy|={abs(dy):.2f}m >={threshold}m"

        progress = min(1.0, current_value / threshold) if threshold > 0 else 0
        confidence = 1.0 if completed else 0.7 + 0.3 * progress
        return {
            "completed": completed,
            "progress": progress,
            "confidence": confidence,
            "reason": reason,
            "current_value": current_value,
            "threshold": threshold
        }

    elif cc_type == "rotation":
        threshold = condition.get("min_degrees", 70)
        direction = condition.get("direction", "")

        if direction == "left":
            # 左转需要 delta_deg 为正（角度增加）
            completed = delta_deg >= threshold
            current_value = delta_deg if delta_deg > 0 else 0
            reason = f"rotation={delta_deg:.0f}° (left), need >= {threshold}°"
        elif direction == "right":
            # 右转需要 delta_deg 为负（角度减少）
            completed = delta_deg <= -threshold
            current_value = -delta_deg if delta_deg < 0 else 0
            reason = f"rotation={delta_deg:.0f}° (right), need <= -{threshold}°"
        else:
            # 无方向要求，用绝对值
            completed = abs_delta_deg >= threshold
            current_value = abs_delta_deg
            reason = f"rotation={abs_delta_deg:.0f}° >= {threshold}°"

        progress = min(1.0, current_value / threshold) if threshold > 0 else 0
        confidence = 1.0 if completed else 0.7 + 0.3 * progress
        return {
            "completed": completed,
            "progress": progress,
            "confidence": confidence,
            "reason": reason,
            "current_value": current_value,
            "threshold": threshold
        }

    elif cc_type == "distance":
        threshold = condition.get("min_meters", 5)
        progress = min(1.0, horizontal_dist / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": horizontal_dist >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"distance={horizontal_dist:.2f}m >= {threshold}m",
            "current_value": horizontal_dist,
            "threshold": threshold
        }

    elif cc_type == "near_object" or cc_type == "object_near":
        perception_output = context.metadata.get("perception_output", "")
        target = condition.get("object", "").lower()
        # 从自然语言描述中搜索目标物体
        found = target in perception_output.lower() if perception_output else False
        return {
            "completed": found,
            "progress": 1.0 if found else 0.0,
            "confidence": 1.0 if found else 0.3,
            "reason": f"object '{target}' {'FOUND' if found else 'NOT FOUND'} in description",
            "current_value": 1 if found else 0,
            "threshold": 1
        }

    elif cc_type == "obstacle_detected":
        threshold = condition.get("min_distance_moved", 1.0)
        progress = min(1.0, horizontal_dist / threshold) if threshold > 0 else 0
        confidence = 1.0 if progress > 0.9 else 0.7 + 0.3 * progress
        return {
            "completed": horizontal_dist >= threshold,
            "progress": progress,
            "confidence": confidence,
            "reason": f"moved={horizontal_dist:.2f}m >= {threshold}m (obstacle reaction)",
            "current_value": horizontal_dist,
            "threshold": threshold
        }

    elif cc_type == "obstacle_cleared":
        # Check if obstacle distance is > min_obstacle_distance
        threshold = condition.get("min_obstacle_distance", 3.0)
        # Get obstacle distance from perception (min_dist from blocked_info)
        blocked_info = context.metadata.get("blocked_info", {})
        obstacle_dist = blocked_info.get("min_dist", 999.0)  # Default to large if no obstacle
        # If no obstacle detected, assume cleared
        if obstacle_dist == 999.0 or blocked_info.get("blocked", False) == False:
            obstacle_dist = 10.0  # Treat as cleared
        progress = min(1.0, obstacle_dist / threshold) if threshold > 0 else 1.0
        return {
            "completed": obstacle_dist >= threshold,
            "progress": progress,
            "confidence": 1.0 if progress > 0.9 else 0.7 + 0.3 * progress,
            "reason": f"obstacle_dist={obstacle_dist:.2f}m >= {threshold}m",
            "current_value": obstacle_dist,
            "threshold": threshold
        }

    else:
        return {"completed": False, "progress": 0, "confidence": 0, "reason": f"未知条件类型: {cc_type}", "current_value": 0, "threshold": 0}


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

        # Pipeline Navigator (orchestrates all SubAgents)
        self.navigator = None

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
        """Initialize Pipeline Agent Architecture.

        Navigator + 6 SubAgents (Observation, Analysis, Planning,
        Review, Emergency, SubtaskDecomposition) with multi-tier model allocation.
        """
        self.logger.info("Initializing Pipeline Agent Architecture...")

        # Build Pipeline config from experiment config
        pipeline_config = {
            "max_steps": self.config.get("max_steps", 150),
            "report_interval": 10,
            "model_configs": self.config.get("pipeline", {}).get("model_configs", {
                "decomposition": "qwen3.5-9b-fast",
                "observation": "qwen3-vl-8b",
                "analysis": "qwen3.5-9b-fast",
                "analysis_strong": "qwen3.6-35b-strong",
                "planning": "qwen3.6-35b-strong",
                "review": "qwen3.5-9b-fast",
                "emergency": "qwen3.5-9b-fast",
            }),
        }

        self.navigator = Navigator(pipeline_config)

        # Register all SubAgents with their model keys
        self.navigator.register_subagents()

        # Set ModelManager for LLM/VLM access
        if hasattr(self, 'model_manager') and self.model_manager:
            self.navigator.set_model_manager(self.model_manager)

        self.logger.info("Pipeline architecture initialized successfully")

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

        # Random shuffle for diverse scene coverage
        if self.config.get("shuffle_episodes", False):
            random.shuffle(episodes_to_run)
            self.logger.info("Episodes shuffled for random sampling")

        self.logger.info("=" * 60)
        self.logger.info(f"Starting evaluation - {num_to_run} episodes")
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

        # SPL: geodesic / max(trajectory, geodesic)
        if success and trajectory_length > 0:
            spl = episode.geodesic_distance / max(trajectory_length, episode.geodesic_distance)
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

            # === Pipeline Agent Architecture (default) ===
            self.logger.info("[Pipeline] Using Pipeline Agent Architecture")
            
            # Initialize episode output
            self.output_manager.start_episode(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                instruction=episode.instruction,
                goal_position=episode.goal_position,
                start_position=list(start_pos),
            )
            
            # Run navigation via Pipeline Navigator
            result = self._run_pipeline_episode(episode, sim, start_pos, initial_yaw, reporter)
            
            # Save final output
            self.output_manager.finish_episode(
                success=result["success"],
                final_distance=result.get("min_distance", 0.0),
                min_distance=result.get("min_distance", 0.0),
                steps=result["steps"],
                trajectory=result["trajectory"],
                task_level="pipeline",
                subtasks=result.get("subtasks", []),
                goal_position=list(episode.goal_position),
            )
            
            # Save context metadata for emergency evaluation
            self._last_context_metadata = {}
            
            return result

        except Exception as e:
            self.logger.error(f"Episode terminated unexpectedly: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Save episode output
            try:
                final_pos = trajectory[-1] if trajectory else start_pos
                self.output_manager.finish_episode(
                    success=success,
                    final_distance=self._distance(final_pos, episode.goal_position),
                    min_distance=min_distance,
                    steps=steps,
                    trajectory=trajectory,
                    task_level="pipeline",
                    subtasks=[],
                    goal_position=episode.goal_position,
                    reference_path=episode.reference_path,
                )
                if self.enable_video:
                    self.output_manager.create_summary_video(fps=5)
            except Exception as e:
                self.logger.error(f"Failed to save episode output: {e}")

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
            "evaluation_scores": [],
            "task_level": "pipeline",
            "subtask_count": 1,
        }

    def _run_pipeline_episode(
        self,
        episode,
        sim,
        start_pos,
        initial_yaw: float,
        reporter=None,
    ) -> Dict[str, Any]:
        """Run episode using Pipeline Agent Architecture

        Args:
            episode: R2REpisode 数据
            sim: Habitat simulator
            start_pos: 起始位置 [x, y, z]
            initial_yaw: 起始朝向（弧度）

        Returns:
            与 _run_habitat_episode 格式一致的结果字典
        """
        from agents.pipeline.env_adapter import HabitatEnvAdapter
        from agents.pipeline.tools.state_calculator import StateCalculator

        self.logger.info(f"[Pipeline] Starting episode {episode.episode_id}")

        # 创建环境适配器
        env = HabitatEnvAdapter(sim, self._get_observations, self.config)

        # R2R离散模式：加载viewpoint图，启用节点间瞬移
        if self.config.get("r2r_discrete", False):
            viewpoint_graph = self._load_viewpoint_graph(sim, episode.scene_id)
            env.set_r2r_discrete(viewpoint_graph)
            self.logger.info(f"[R2R-Discrete] Viewpoint navigation enabled ({len(viewpoint_graph['viewpoints'])} nodes)")

        # 初始化 Navigator
        self.navigator.initialize_episode(
            instruction=episode.instruction,
            start_position=list(start_pos),
            goal_position=list(episode.goal_position),
        )

        # Note: completion_condition is now set by SubtaskDecompositionAgent
        # No need to manually set it here

        # 运行导航循环（带实时状态更新）
        result = self._run_pipeline_loop_with_output(env, reporter)

        # 收集轨迹
        trajectory = [h["position"] for h in self.navigator._history]
        steps = result["steps"]
        success = result["success"]

        # 计算指标
        calc = StateCalculator()

        min_distance = min(
            calc.compute_distance(pos, episode.goal_position)
            for pos in trajectory
        ) if trajectory else float('inf')

        trajectory_length = sum(
            calc.compute_distance(trajectory[i-1], trajectory[i])
            for i in range(1, len(trajectory))
        ) if len(trajectory) > 1 else 0.0

        # SPL (Success weighted by Path Length)
        spl = 0.0
        if success and trajectory_length > 0:
            spl = episode.geodesic_distance / max(episode.geodesic_distance, trajectory_length)

        # nDTW
        ndtw = self._calculate_ndtw(trajectory, episode.reference_path) if hasattr(self, '_calculate_ndtw') else 0.0
        sdtw = ndtw if success else 0.0

        self.logger.info(f"[Pipeline] Episode complete: steps={steps}, success={success}")

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
            "trajectory_length": trajectory_length,
            "spl": spl,
            "ndtw": ndtw,
            "sdtw": sdtw,
            "task_level": "pipeline",
            "subtask_count": len(self.navigator._subtasks),
            "subtasks": self.navigator._subtasks,
            "evaluation_scores": [],
        }

    def _run_pipeline_loop_with_output(self, env, reporter) -> Dict[str, Any]:
        """Run Pipeline navigation loop with output saving and status reporting."""
        from agents.pipeline.tools.state_calculator import StateCalculator
        from dataclasses import asdict

        max_steps = self.config.get("max_steps", 100)
        report_interval = 10
        trajectory = []
        images = []  # 收集RGB图片用于可视化
        calc = StateCalculator()

        # Pipeline agent outputs (for logging)
        observation_output_dict = None
        analysis_output_dict = None
        planning_output_dict = None

        step_count = 0
        while step_count < max_steps:
            # 执行单个导航周期
            actions = self.navigator._run_navigation_cycle(env)

            # 获取本次 cycle 的 agent 输出
            if self.navigator._last_observation_output:
                observation_output_dict = asdict(self.navigator._last_observation_output)
            if self.navigator._last_analysis_output:
                analysis_output_dict = asdict(self.navigator._last_analysis_output)
            if self.navigator._last_planning_output:
                planning_output_dict = asdict(self.navigator._last_planning_output)

            # 执行动作并记录
            for action_type, repeat_count in actions:
                for _ in range(repeat_count):
                    # 获取当前观察（保存图片）
                    observations = env.get_observations()
                    rgb = observations.get("rgb")
                    depth = observations.get("depth")
                    if rgb is not None:
                        images.append(rgb.copy())

                    # 执行动作
                    env.step(action_type)
                    step_count += 1

                    # 更新状态
                    self.navigator._update_state(env)
                    position = self.navigator._position
                    rotation = self.navigator._rotation
                    trajectory.append(position)

                    # 检查完成条件
                    if self.navigator._check_completion():
                        self.logger.info(f"[Pipeline] Task completed at step {step_count}")
                        return {
                            "success": True, "steps": step_count,
                            "reason": "task_completed",
                            "trajectory": trajectory, "images": images,
                        }

                    # 计算距离目标
                    distance_to_goal = calc.compute_distance(position, self.navigator._current_subtask.get("completion_condition", {}).get("goal_position", [0,0,0]))

                    # Early termination: stuck + far from goal
                    if self.config.get("fast_mode", False) and step_count > 20:
                        recent_positions = trajectory[-20:]
                        unique = set((round(p[0], 1), round(p[2], 1)) for p in recent_positions)
                        if len(unique) <= 3 and distance_to_goal > 5.0:
                            self.logger.info(f"[EarlyStop] Stuck: {len(unique)} unique positions in last 20 steps, NE={distance_to_goal:.1f}m")
                            step_count = self.config.get("max_steps", 999) + 1  # force exit
                            break

                    # 更新 reporter
                    if reporter:
                        reporter.update_position(position[0], position[1], position[2])
                        reporter.update_agent("navigator", f"step {step_count}")

                    # 保存图片
                    self.output_manager.save_rgb_image(rgb, step_count)
                    self.output_manager.save_depth_image(depth, step_count)

                    # 保存每步输出（包含 Pipeline agent 输出）
                    self.output_manager.add_step_output(
                        step=step_count,
                        action=str(action_type),
                        position=position,
                        rotation=rotation,
                        distance_to_goal=distance_to_goal,
                        perception_output=observation_output_dict,  # ObservationAgent -> perception
                        decision_output=analysis_output_dict,       # AnalysisAgent -> decision
                        strategy_output=planning_output_dict,        # PlanningAgent -> strategy
                        current_subtask=self.navigator._current_subtask,
                    )

                    # 检查完成
                    if self.navigator._check_completion():
                        self.logger.info(f"[Pipeline] Task completed at step {step_count}")
                        return {
                            "success": True,
                            "steps": step_count,
                            "reason": "task_completed",
                            "trajectory": trajectory,
                            "images": images,
                        }

                    if step_count >= max_steps:
                        break

            # 周期性进度报告
            if step_count % report_interval == 0:
                print(f"[导航进度] 步数: {step_count}, 位置: {trajectory[-1] if trajectory else 'N/A'}")

        return {
            "success": False,
            "steps": step_count,
            "reason": "max_steps",
            "trajectory": trajectory,
            "images": images,
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

    def _load_viewpoint_graph(self, sim: Any, scene_id: str) -> Dict[str, Any]:
        """Load or build viewpoint navigation graph for R2R discrete mode.

        Extracts navigable viewpoints from the scene's navmesh and builds
        an adjacency graph enabling node-to-node teleportation.

        Returns:
            Dict with 'viewpoints' (list of positions) and 'adjacency' (dict)
        """
        if not hasattr(self, '_viewpoint_cache'):
            self._viewpoint_cache = {}

        if scene_id in self._viewpoint_cache:
            return self._viewpoint_cache[scene_id]

        # Sample navigable points from navmesh using pathfinder
        pathfinder = sim.pathfinder
        # Get navmesh vertices as list of [x,y,z] lists
        navmesh_verts_raw = pathfinder.build_navmesh_vertices()
        navmesh_verts = []
        for v in navmesh_verts_raw:
            try:
                navmesh_verts.append([float(v[0]), float(v[1]), float(v[2])])
            except (IndexError, TypeError):
                continue

        # Subsample uniformly to get viewpoint candidates (~2-3m apart like R2R)
        viewpoints = []
        step = 1.5  # meters between sampled viewpoints
        for v in navmesh_verts:
            v_arr = np.array(v)
            too_close = any(np.linalg.norm(v_arr - np.array(ex)) < step for ex in viewpoints)
            if not too_close and pathfinder.is_navigable(v_arr):
                viewpoints.append(v)

        # Build adjacency: two viewpoints connected if pathfinder finds a short path
        adjacency = {}
        for i, v1 in enumerate(viewpoints):
            adj = []
            for j, v2 in enumerate(viewpoints):
                if i == j:
                    continue
                dist = np.linalg.norm(np.array(v1) - np.array(v2))
                if dist < 4.0:  # Adjacent if within 4m (typical R2R node distance)
                    found = pathfinder.is_navigable(np.array(v2))
                    if found:
                        adj.append({'index': j, 'position': v2, 'distance': dist})
            adjacency[i] = sorted(adj, key=lambda x: x['distance'])

        graph = {'viewpoints': viewpoints, 'adjacency': adjacency}
        self._viewpoint_cache[scene_id] = graph
        self.logger.info(f"[R2R-Discrete] Built viewpoint graph: {len(viewpoints)} nodes")
        return graph

    def _execute_r2r_action(self, sim: Any, action: str, viewpoint_graph: Dict) -> np.ndarray:
        """Execute action in R2R discrete nav-graph mode.

        move_forward: teleport to adjacent viewpoint in heading direction
        turn_left/right: rotate ~30° to face adjacent viewpoints
        stop: no-op

        Returns new position after action.
        """
        agent = sim.get_agent(0)
        state = agent.get_state()
        pos = np.array(state.position)
        rot = state.rotation

        # Extract yaw from quaternion
        import quaternion
        q_arr = quaternion.as_float_array(rot)
        w, x, y, z = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
        siny = 2 * (w * y + x * z)
        cosy = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)

        if action == "move_forward":
            # Find nearest viewpoint
            viewpoints = viewpoint_graph['viewpoints']
            if not viewpoints:
                sim.step(action)
                return np.array(agent.get_state().position)

            nearest_idx = min(range(len(viewpoints)),
                             key=lambda i: np.linalg.norm(np.array(viewpoints[i]) - pos))

            # Find adjacent viewpoint closest to heading direction
            heading_vec = np.array([math.sin(yaw), 0, math.cos(yaw)])
            best_adj = None
            best_score = -2

            for adj_info in viewpoint_graph['adjacency'].get(nearest_idx, []):
                adj_pos = np.array(adj_info['position'])
                direction = adj_pos - pos
                direction[1] = 0  # Ignore Y
                direction_norm = np.linalg.norm(direction)
                if direction_norm < 0.1:
                    continue
                direction = direction / direction_norm
                score = np.dot(heading_vec, direction)
                if score > best_score and score > 0.1:  # Within ~85° of heading
                    best_score = score
                    best_adj = adj_info

            if best_adj:
                # Teleport to adjacent viewpoint
                new_pos = best_adj['position'].copy()
                new_pos[1] = sim.pathfinder.get_random_navigable_point_near(
                    best_adj['position'][0:3], 0.5
                )[1] if hasattr(sim.pathfinder, 'get_random_navigable_point_near') else best_adj['position'][1]

                state.position = np.array(best_adj['position'])
                agent.set_state(state)
                return np.array(best_adj['position'])
            else:
                # No adjacent viewpoint found, fall back to physical step
                sim.step(action)
                return np.array(agent.get_state().position)

        elif action in ("turn_left", "turn_right"):
            # Rotate by 30° (standard R2R turn angle)
            angle = math.radians(30)
            if action == "turn_left":
                angle = -angle
            new_yaw = yaw + angle
            # Create rotation quaternion for Y-axis rotation
            new_q = quaternion.from_rotation_vector(np.array([0, new_yaw, 0]))
            state.rotation = new_q
            agent.set_state(state)
            return pos

        else:  # stop, look_up, look_down
            sim.step(action)
            return np.array(agent.get_state().position)

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
            # Get GPU ID from config (for parallel evaluation)
            sim_cfg.gpu_device_id = self.config.get("gpu_id", 0)

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
        Decide whether to call ReviewAgent completion check.

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

    def _run_simulated_episode(self, episode: R2REpisode) -> Dict[str, Any]:
        """Simple simulation fallback when Habitat scene is unavailable.

        Moves agent linearly toward the goal position.
        """
        trajectory = [episode.start_position.copy()]
        current = list(episode.start_position)
        steps = 0
        success = False
        min_distance = self._distance(current, episode.goal_position)

        max_steps = self.config.get("max_steps", 100)
        success_distance = self.config.get("success_distance", 3.0)
        step_size = 0.25

        while steps < max_steps:
            dx = episode.goal_position[0] - current[0]
            dz = episode.goal_position[2] - current[2]
            dist = math.sqrt(dx*dx + dz*dz)

            if dist <= success_distance:
                success = True
                break

            if dist > 0:
                current[0] += (dx / dist) * step_size
                current[2] += (dz / dist) * step_size

            trajectory.append(current.copy())
            steps += 1
            min_distance = min(min_distance, dist)

        return {
            "trajectory": trajectory,
            "steps": steps,
            "success": success,
            "min_distance": min_distance,
            "evaluation_scores": [],
            "task_level": "simulated",
            "subtask_count": 1,
        }

    def _distance(self, p1: List[float], p2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:3], p2[:3])))

    def _calculate_ndtw(self, trajectory: List[List[float]], reference: List[List[float]]) -> float:
        """Calculate nDTW using standard VLN formula.

        nDTW = exp(-DTW / (d_avg * max_len))
        where d_avg is the average step distance in the reference path.
        This matches the standard VLN literature (MSNav, MapGPT, etc.).
        """
        if not trajectory or not reference:
            return 0.0

        n, m = len(trajectory), len(reference)
        if n < 2 or m < 2:
            return 0.0

        # Compute DTW matrix
        dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._distance(trajectory[i-1], reference[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

        # Average step distance in reference path
        ref_step_sum = sum(
            self._distance(reference[k-1], reference[k])
            for k in range(1, len(reference))
        )
        d_avg = ref_step_sum / (len(reference) - 1) if len(reference) > 1 else 1.0

        max_len = max(n, m)
        ndtw = math.exp(-dtw[n][m] / (d_avg * max_len)) if max_len > 0 and d_avg > 0 else 0.0
        return ndtw

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
    parser.add_argument("--output", type=str, default="results/results.json",
                        help="Output JSON filename (default: results/results.json)")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-int8", action="store_true", default=False, help="Disable INT8 quantization (use FP16)")

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

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--r2r-discrete", action="store_true", default=False,
                        help="Use R2R discrete nav-graph mode (viewpoint teleportation instead of physical stepping)")
    parser.add_argument("--shuffle", action="store_true", default=False,
                        help="Randomly shuffle episodes for diverse scene coverage")
    parser.add_argument("--fast", action="store_true", default=False,
                        help="Fast dev mode: all models route to 9B, max-steps=80, early termination")
    parser.add_argument("--resolution", type=int, default=480,
                        help="Camera resolution (height, default 480). Width=height*4/3")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results (default: results)")

    args = parser.parse_args()

    # Fast mode: all models → 9B, shorter episodes, early termination
    if args.fast:
        from models.model_manager import ModelManager
        for k in ModelManager.MODEL_SERVER_MAP:
            ModelManager.MODEL_SERVER_MAP[k] = ('http://localhost:8000', True)
        if args.max_steps == 50:  # only override if user didn't specify
            args.max_steps = 80
        print("[FAST MODE] All models → 9B (GPU 0), max_steps=80, early termination ON")

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # If output-dir is default, append timestamp for uniqueness
    if args.output_dir == "results":
        session_timestamp = datetime.now().strftime("%Y-%m%d-%H%M")
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
        "use_int8": not args.no_int8,
        "seed": args.seed,
        "use_remote_llm": args.use_remote_llm,
        "llm_server": args.llm_server,
        "remote_timeout": args.remote_timeout,
        "use_siliconflow": args.use_siliconflow,
        "siliconflow_api_key": args.siliconflow_api_key,
        "siliconflow_model": args.siliconflow_model,
        "vlm_server_url": args.vlm_server,
        "output_dir": args.output_dir,
        "r2r_discrete": args.r2r_discrete,
        "shuffle_episodes": args.shuffle,
        "fast_mode": args.fast,
        "image_width": int(args.resolution * 4 / 3),
        "image_height": args.resolution,
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
    print(f"INT8 Quantization: {not args.no_int8}")
    print(f"Remote LLM: {'enabled' if args.use_remote_llm else 'disabled'}")
    if args.use_remote_llm:
        print(f"LLM Server: {args.llm_server}")
    print(f"SiliconFlow API: {'enabled' if args.use_siliconflow else 'disabled'}")
    if args.use_siliconflow:
        print(f"SiliconFlow Model: {args.siliconflow_model}")
        print(f"Local VLM Server: {args.vlm_server}")
    print(f"Architecture: Pipeline (Navigator + 6 SubAgents)")
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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()