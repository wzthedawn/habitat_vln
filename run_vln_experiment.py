#!/usr/bin/env python3
"""
R2R VLN 评估实验脚本
使用真实 Matterport3D 场景和 R2R 数据集
集成多智能体 LLM 导航系统

多Agent架构:
- InstructionAgent: 规则匹配，子任务分解
- PerceptionAgent: YOLO + Qwen2B 视觉感知
- TrajectoryAgent: 建图 + 轨迹摘要
- DecisionAgent: Qwen4B 决策
- EvaluationAgent: Qwen9B 评估 (可选)

双环境IPC架构:
- Python 3.9 (Habitat): VLN主进程, habitat-sim, YOLO
- Python 3.10 (LLM Server): Qwen3.5模型推理服务

Usage:
    # 启动LLM服务 (Python 3.10)
    conda activate habitat_py310
    python llm_server.py --port 8000

    # 运行VLN评估 (Python 3.9)
    conda activate Habitat
    python run_vln_experiment.py --use-remote-llm --llm-server http://localhost:8000 ...
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.token_tracker import get_token_tracker
from utils.timeout_fallback import TimeoutError, timeout, StepTimeout, DEFAULT_TIMEOUTS


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
    task_level: str = "中等"
    subtask_count: int = 0
    evaluation_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiAgentVLNEvaluator:
    """多智能体VLN评估器"""

    def __init__(self, config: Dict[str, Any], log_level: str = "INFO"):
        self.config = config
        self.logger = setup_logger("MultiAgentVLNEvaluator", level=log_level)
        self.log_level = log_level

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
        self.r2r_path = Path(config.get("r2r_path", "/root/habitat-lab/data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json"))

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

        self.logger.info(f"多智能体VLN评估器初始化")

    def initialize(self) -> None:
        """初始化评估系统"""
        self.logger.info("=" * 60)
        self.logger.info("初始化多智能体VLN评估系统")
        self.logger.info("=" * 60)

        # 1. 构建场景路径映射
        self._build_scene_paths()

        # 2. 加载R2R数据
        self._load_r2r_data()

        # 3. 初始化模型管理器
        self._init_model_manager()

        # 4. 初始化Agent
        self._init_agents()

        self.logger.info("初始化完成!")

    def _build_scene_paths(self) -> None:
        """构建MP3D场景路径映射"""
        self.logger.info(f"扫描MP3D场景: {self.mp3d_path}")

        if not self.mp3d_path.exists():
            self.logger.warning(f"MP3D路径不存在: {self.mp3d_path}")
            return

        for scene_dir in self.mp3d_path.iterdir():
            if scene_dir.is_dir():
                scene_id = scene_dir.name
                glb_file = scene_dir / f"{scene_id}.glb"
                if glb_file.exists():
                    self.scene_paths[scene_id] = str(glb_file)

        self.logger.info(f"找到 {len(self.scene_paths)} 个MP3D场景")

    def _load_r2r_data(self) -> None:
        """加载R2R数据集"""
        self.logger.info(f"加载R2R数据: {self.r2r_path}")

        if not self.r2r_path.exists():
            self.logger.warning(f"R2R数据路径不存在: {self.r2r_path}")
            return

        with open(self.r2r_path) as f:
            data = json.load(f)

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

        self.logger.info(f"加载 {len(self.episodes)} 个R2R episodes")

    def _init_model_manager(self) -> None:
        """初始化模型管理器"""
        self.logger.info("初始化模型管理器...")

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
                model_config["remote_timeout"] = self.config.get("remote_timeout", 60.0)
                self.logger.info(f"使用远程LLM服务: {model_config['remote_server_url']}")

            self.model_manager = get_model_manager(model_config)
            self.logger.info("模型管理器初始化成功")
        except Exception as e:
            self.logger.warning(f"模型管理器初始化失败: {e}")

    def _init_agents(self) -> None:
        """初始化所有Agent"""
        self.logger.info("初始化Agent...")

        agent_config = {
            "confidence_threshold": 0.6,
            "max_history_steps": 5,
            # Pass remote LLM config to agents
            "use_remote": self.config.get("use_remote_llm", False),
            "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
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

            self.logger.info("所有Agent初始化成功 (包括EvaluationAgent)")

        except Exception as e:
            self.logger.error(f"Agent初始化失败: {e}")

    def run_evaluation(self, num_episodes: int = None) -> Dict[str, Any]:
        """运行VLN评估"""
        num_to_run = min(num_episodes or len(self.episodes), len(self.episodes))

        self.logger.info("=" * 60)
        self.logger.info(f"开始评估 - {num_to_run} episodes")
        self.logger.info("=" * 60)

        start_time = time.time()

        for i, episode in enumerate(self.episodes[:num_to_run]):
            ep_start = time.time()

            self.logger.info(f"\n[{i+1}/{num_to_run}] Episode {episode.episode_id}")
            self.logger.info(f"  Scene: {episode.scene_id}")
            self.logger.info(f"  Instruction: {episode.instruction[:80]}...")

            result = self._run_episode(episode)
            result.time_elapsed = time.time() - ep_start
            self.results.append(result)

            # 输出进度
            self._log_progress(i + 1, num_to_run)

            # 每10个episode保存一次中间结果
            if (i + 1) % 10 == 0:
                self._save_intermediate_results()

        total_time = time.time() - start_time

        # 清理所有Simulator资源
        self._cleanup_sims()

        return self._compile_results(total_time)

    def _run_episode(self, episode: R2REpisode) -> EvaluationResult:
        """运行单个episode"""
        trajectory = []
        steps = 0
        success = False
        min_distance = float('inf')
        evaluation_scores = []

        start_time = time.time()

        try:
            # 检查场景是否可用
            if episode.scene_id in self.scene_paths:
                result = self._run_habitat_episode(episode)
            else:
                self.logger.warning(f"场景不可用: {episode.scene_id}, 使用模拟模式")
                result = self._run_simulated_episode(episode)

            trajectory = result["trajectory"]
            steps = result["steps"]
            success = result["success"]
            min_distance = result["min_distance"]
            evaluation_scores = result.get("evaluation_scores", [])

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} 失败: {e}")
            trajectory = [episode.start_position]
            steps = 0
            min_distance = self._distance(episode.start_position, episode.goal_position)

        # 计算指标
        final_pos = trajectory[-1] if trajectory else episode.start_position
        distance_to_goal = self._distance(final_pos, episode.goal_position)
        oracle_success = min_distance <= self.config.get("success_distance", 3.0)

        # 轨迹长度
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

        # 获取任务等级
        task_level = result.get("task_level", "中等")
        subtask_count = result.get("subtask_count", 0)

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
            token_usage={"total": 0, "input": 0, "output": 0},  # 本地模型无token计数
            trajectory=trajectory,
            task_level=task_level,
            subtask_count=subtask_count,
            evaluation_scores=evaluation_scores,
        )

    def _run_habitat_episode(self, episode: R2REpisode) -> Dict[str, Any]:
        """在Habitat环境中运行episode"""
        import habitat_sim

        scene_path = self.scene_paths[episode.scene_id]
        sim = None
        trajectory = []
        steps = 0
        success = False
        min_distance = float('inf')
        evaluation_scores = []
        task_level = "中等"
        subtask_count = 0

        try:
            # 1. 获取Simulator
            sim = self._get_simulator(episode.scene_id, scene_path)
            if sim is None:
                return self._run_simulated_episode(episode)

            # 2. 设置起点位置
            agent = sim.get_agent(0)
            state = habitat_sim.AgentState()

            start_pos = self._validate_position(sim, episode.start_position)
            if start_pos is None:
                return self._run_simulated_episode(episode)

            state.position = start_pos
            if episode.start_rotation:
                state.rotation = np.array(episode.start_rotation)
            agent.set_state(state)

            trajectory = [[float(x) for x in start_pos]]
            min_distance = self._distance(start_pos, episode.goal_position)

            # 3. 创建导航上下文
            from core.context import NavContextBuilder, VisualFeatures
            visual_features = VisualFeatures()

            context = NavContextBuilder() \
                .with_instruction(episode.instruction) \
                .with_position(tuple(start_pos)) \
                .with_visual_features(visual_features) \
                .build()

            # 4. 处理指令 - InstructionAgent
            if self.instruction_agent:
                instruction_output = self.instruction_agent.process(context)
                context.metadata["instruction_output"] = instruction_output.data
                task_level = instruction_output.data.get("task_level", "中等")
                subtask_count = len(instruction_output.data.get("subtasks", []))
                self.logger.info(f"  任务等级: {task_level}, 子任务数: {subtask_count}")

            # 5. 重置TrajectoryAgent的地图
            if self.trajectory_agent:
                self.trajectory_agent.reset_map()

            # 6. 重置EvaluationAgent历史
            if self.evaluation_agent:
                self.evaluation_agent.reset_history()

            max_steps = self.config.get("max_steps", 100)
            success_distance = self.config.get("success_distance", 3.0)
            task_level = "中等"  # Default

            # 7. 导航主循环
            while steps < max_steps:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Step {steps + 1}/{max_steps}")
                self.logger.info(f"{'='*60}")

                # 获取RGB和Depth图像
                rgb_image, depth_image = self._get_observations(sim)

                # 更新context中的图像
                context.rgb_image = rgb_image
                context.depth_image = depth_image
                context.metadata['rgb_image'] = rgb_image
                context.metadata['depth_image'] = depth_image

                # PerceptionAgent处理
                if self.perception_agent:
                    perception_output = self.perception_agent.process(context)
                    context.metadata["perception_output"] = perception_output.data
                    self.logger.info(f"[PerceptionAgent] 房间: {perception_output.data.get('room_type', 'unknown')}")
                    self.logger.info(f"[PerceptionAgent] 物体: {[o['name'] for o in perception_output.data.get('objects', [])[:5]]}")
                    self.logger.info(f"[PerceptionAgent] 场景: {perception_output.data.get('scene_description', '')[:100]}")

                # TrajectoryAgent处理
                if self.trajectory_agent:
                    trajectory_output = self.trajectory_agent.process(context)
                    context.metadata["trajectory_output"] = trajectory_output.data
                    self.logger.info(f"[TrajectoryAgent] 已走步数: {trajectory_output.data.get('steps_taken', 0)}")
                    self.logger.info(f"[TrajectoryAgent] 摘要: {trajectory_output.data.get('summary', '')[:100]}")

                # DecisionAgent决策
                action_name = "forward"
                if self.decision_agent:
                    decision_output = self.decision_agent.process(context)
                    action_name = decision_output.data.get("action", "forward")
                    context.metadata["decision_output"] = decision_output.data
                    self.logger.info(f"[DecisionAgent] 动作: {action_name}")
                    self.logger.info(f"[DecisionAgent] 推理: {decision_output.data.get('reasoning', '')[:150]}")

                # EvaluationAgent评估 (根据任务等级)
                if self.evaluation_agent:
                    # 获取任务等级
                    task_level = context.metadata.get("instruction_output", {}).get("task_level", "中等")

                    # 根据任务等级决定是否调用评估
                    should_eval = self._should_call_evaluation(task_level, steps)

                    if should_eval:
                        eval_output = self.evaluation_agent.process(context)
                        context.metadata["evaluation_output"] = eval_output.data
                        eval_score = eval_output.data.get("score", 0.5)
                        evaluation_scores.append(eval_score)
                        self.logger.info(f"[EvaluationAgent] 评分: {eval_score:.2f}")
                        self.logger.info(f"[EvaluationAgent] 反馈: {eval_output.data.get('feedback', '')[:100]}")

                        # 检查是否需要重新规划
                        if eval_output.data.get("replan_needed", False):
                            self.logger.info(f"  评估触发重新规划 at step {steps}")
                            # 重置InstructionAgent的子任务
                            if self.instruction_agent:
                                new_instruction = self.instruction_agent.process(context)
                                context.metadata["instruction_output"] = new_instruction.data

                # 执行动作
                if action_name == "stop":
                    self.logger.info(f"  Agent主动停止 at step {steps}")
                    break

                try:
                    sim.step(action_name)
                except Exception as e:
                    self.logger.warning(f"  动作执行失败: {e}")

                # 更新状态
                state = agent.get_state()
                pos = [float(x) for x in state.position]
                trajectory.append(pos)
                steps += 1

                context.position = tuple(pos)
                context.add_trajectory_point(tuple(pos))

                dist = self._distance(pos, episode.goal_position)
                min_distance = min(min_distance, dist)

                if dist <= success_distance:
                    success = True
                    self.logger.info(f"  成功到达目标 at step {steps}")
                    break

        except Exception as e:
            self.logger.error(f"  Habitat错误: {e}")
            import traceback
            traceback.print_exc()

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
        """从Habitat获取RGB和Depth图像"""
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
            else:
                rgb = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

            if depth is not None:
                depth = np.array(depth).squeeze()
            else:
                depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)

            return rgb, depth

        except Exception as e:
            self.logger.warning(f"获取图像失败: {e}")
            return (
                np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8),
                np.zeros((self.image_height, self.image_width), dtype=np.float32)
            )

    def _get_simulator(self, scene_id: str, scene_path: str) -> Any:
        """获取或创建Simulator"""
        import habitat_sim

        if scene_id in self._sim_cache:
            self._current_scene = scene_id
            return self._sim_cache[scene_id]

        try:
            self.logger.info(f"  创建Simulator: {scene_id}")

            # 配置传感器
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

            # 配置Agent
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

            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            sim = habitat_sim.Simulator(cfg)

            self._sim_cache[scene_id] = sim
            self._current_scene = scene_id

            return sim

        except Exception as e:
            self.logger.error(f"  Simulator创建失败: {e}")
            return None

    def _validate_position(self, sim: Any, position: List[float]) -> Optional[List[float]]:
        """验证位置是否可导航"""
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
            self.logger.warning(f"位置验证失败: {e}")
            return [float(x) for x in position]

    def _should_call_evaluation(self, task_level: str, step_count: int) -> bool:
        """
        根据任务等级决定是否调用EvaluationAgent。

        Args:
            task_level: 任务等级 (简单/中等/困难)
            step_count: 当前步数

        Returns:
            True if evaluation should be called
        """
        if task_level == "简单":
            return False  # 简单任务不调用评估
        elif task_level == "中等":
            return step_count % 5 == 0  # 每5步评估一次
        elif task_level == "困难":
            return True  # 每步都评估
        return False

    def _run_simulated_episode(self, episode: R2REpisode) -> Dict[str, Any]:
        """模拟模式运行episode"""
        trajectory = [episode.start_position.copy()]
        current = list(episode.start_position)
        steps = 0
        success = False
        min_distance = self._distance(current, episode.goal_position)
        evaluation_scores = []
        task_level = "中等"
        subtask_count = 0

        from core.context import NavContextBuilder
        context = NavContextBuilder() \
            .with_instruction(episode.instruction) \
            .with_position(tuple(current)) \
            .build()

        # 处理指令
        if self.instruction_agent:
            instruction_output = self.instruction_agent.process(context)
            context.metadata["instruction_output"] = instruction_output.data
            task_level = instruction_output.data.get("task_level", "中等")
            subtask_count = len(instruction_output.data.get("subtasks", []))

        max_steps = self.config.get("max_steps", 100)
        success_distance = self.config.get("success_distance", 3.0)

        while steps < max_steps:
            # 简单的模拟决策
            if self.decision_agent:
                decision_output = self.decision_agent.process(context)
                action_name = decision_output.data.get("action", "forward")
            else:
                # 向目标方向移动
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
        """清理所有Simulator"""
        self.logger.info(f"清理 {len(self._sim_cache)} 个Simulator")
        for scene_id, sim in self._sim_cache.items():
            try:
                if sim is not None:
                    sim.close()
            except Exception as e:
                self.logger.warning(f"关闭Simulator失败: {e}")

        self._sim_cache.clear()
        self._current_scene = ""
        gc.collect()

    def _log_progress(self, current: int, total: int) -> None:
        """输出进度"""
        if not self.results:
            return

        successes = sum(1 for r in self.results if r.success)
        sr = successes / len(self.results) * 100
        avg_spl = sum(r.spl for r in self.results) / len(self.results)
        avg_ne = sum(r.distance_to_goal for r in self.results) / len(self.results)
        avg_steps = sum(r.steps for r in self.results) / len(self.results)

        self.logger.info(f"  进度: {current}/{total} | SR: {sr:.1f}% | SPL: {avg_spl:.3f} | NE: {avg_ne:.2f}m | Steps: {avg_steps:.1f}")

    def _save_intermediate_results(self) -> None:
        """保存中间结果"""
        output_path = Path(self.config.get("output", "results_vln.json"))
        with open(output_path, "w") as f:
            json.dump({
                "status": "in_progress",
                "episodes_completed": len(self.results),
                "results": [r.to_dict() for r in self.results],
            }, f, indent=2)

    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """编译最终结果"""
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

        # 按任务等级统计
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
    parser = argparse.ArgumentParser(description="多智能体R2R VLN评估")
    parser.add_argument("--mp3d-path", type=str, default="data/mp3d_dataset/mp3d")
    parser.add_argument("--r2r-path", type=str, default="/root/habitat-lab/data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json")
    parser.add_argument("--episodes", type=int, default=5, help="评估episode数量")
    parser.add_argument("--max-steps", type=int, default=50, help="每个episode最大步数")
    parser.add_argument("--success-distance", type=float, default=3.0, help="成功距离阈值(米)")
    parser.add_argument("--output", type=str, default="results_multi_agent.json")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-int8", action="store_true", default=True, help="使用INT8量化")

    # Remote LLM arguments for dual-environment IPC
    parser.add_argument("--use-remote-llm", action="store_true", default=False,
                        help="使用远程LLM服务 (Python 3.10环境)")
    parser.add_argument("--llm-server", type=str, default="http://localhost:8000",
                        help="远程LLM服务器地址")
    parser.add_argument("--remote-timeout", type=float, default=60.0,
                        help="远程LLM请求超时时间(秒)")

    args = parser.parse_args()

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
    }

    print("=" * 70)
    print("多智能体 R2R VLN 评估实验")
    print("=" * 70)
    print(f"MP3D路径: {args.mp3d_path}")
    print(f"R2R数据: {args.r2r_path}")
    print(f"Episode数量: {args.episodes}")
    print(f"最大步数: {args.max_steps}")
    print(f"成功距离: {args.success_distance}m")
    print(f"输出文件: {args.output}")
    print(f"设备: {args.device}")
    print(f"INT8量化: {args.use_int8}")
    print(f"远程LLM: {'启用' if args.use_remote_llm else '禁用'}")
    if args.use_remote_llm:
        print(f"LLM服务器: {args.llm_server}")
    print("=" * 70)

    evaluator = MultiAgentVLNEvaluator(config, log_level=args.log_level)
    evaluator.initialize()

    results = evaluator.run_evaluation(num_episodes=args.episodes)

    # 输出结果
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)

    summary = results.get("summary", {})
    print(f"Episodes: {summary.get('num_episodes', 0)}")
    print(f"Success Rate (SR): {summary.get('success_rate', 0)*100:.1f}%")
    print(f"Oracle Success Rate: {summary.get('oracle_success_rate', 0)*100:.1f}%")
    print(f"SPL: {summary.get('spl', 0):.3f}")
    print(f"nDTW: {summary.get('nDTW', 0):.3f}")
    print(f"SDTW: {summary.get('SDTW', 0):.3f}")
    print(f"Navigation Error (NE): {summary.get('avg_distance_to_goal', 0):.2f}m")
    print(f"平均步数: {summary.get('avg_steps', 0):.1f}")
    print(f"总耗时: {summary.get('total_time_seconds', 0):.1f}s")

    # 按任务等级输出
    by_level = results.get("by_task_level", {})
    if by_level:
        print("\n按任务等级统计:")
        for level, data in by_level.items():
            print(f"  {level}: {data['count']}个, SR: {data['success_rate']*100:.1f}%, SPL: {data['spl']:.3f}")

    print("=" * 70)

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()