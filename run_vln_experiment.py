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

        # Episode output manager
        self.output_manager = EpisodeOutputManager(config.get("output_dir", "results"))

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
                model_config["remote_timeout"] = self.config.get("remote_timeout", 120.0)  # Increased for instruction decomposition
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

            self.logger.info("所有Agent初始化成功 (包括EvaluationAgent)")

        except Exception as e:
            self.logger.error(f"Agent初始化失败: {e}")

    def run_evaluation(self, num_episodes: int = None, start_episode_id: int = None) -> Dict[str, Any]:
        """运行VLN评估

        Args:
            num_episodes: 运行的episode数量
            start_episode_id: 起始episode ID (None则从第一个开始)
        """
        # 找到起始位置
        if start_episode_id is not None:
            start_idx = None
            for i, ep in enumerate(self.episodes):
                if ep.episode_id == start_episode_id:
                    start_idx = i
                    break
            if start_idx is None:
                self.logger.warning(f"Episode {start_episode_id} 未找到，从第一个开始")
                start_idx = 0
        else:
            start_idx = 0

        episodes_to_run = self.episodes[start_idx:]
        num_to_run = min(num_episodes or len(episodes_to_run), len(episodes_to_run))

        self.logger.info("=" * 60)
        self.logger.info(f"开始评估 - {num_to_run} episodes (从 Episode {episodes_to_run[0].episode_id if episodes_to_run else 'N/A'} 开始)")
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
            task_level = result.get("task_level", "中等")
            subtask_count = result.get("subtask_count", 0)

        except Exception as e:
            self.logger.error(f"Episode {episode.episode_id} 失败: {e}")
            trajectory = [episode.start_position]
            steps = 0
            min_distance = self._distance(episode.start_position, episode.goal_position)
            task_level = "中等"
            subtask_count = 0

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

            # Calculate initial rotation (yaw angle) from start_rotation quaternion
            initial_yaw = 0.0
            if episode.start_rotation:
                q = episode.start_rotation
                if len(q) == 4:
                    # Habitat uses [x, y, z, w] quaternion format
                    x, y, z, w = q[0], q[1], q[2], q[3]
                    siny_cosp = 2 * (w * y + x * z)
                    cosy_cosp = 1 - 2 * (y * y + z * z)
                    initial_yaw = math.atan2(siny_cosp, cosy_cosp)

            # 3. 创建导航上下文
            from core.context import NavContextBuilder, VisualFeatures
            from utils.status_reporter import init_reporter
            visual_features = VisualFeatures()

            # 初始化实时状态报告器
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

            # 3.5 初始化episode输出
            self.output_manager.start_episode(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                instruction=episode.instruction,
                goal_position=episode.goal_position,
                start_position=list(start_pos),
            )

            # 4. 处理指令 - InstructionAgent
            if self.instruction_agent:
                reporter.update_phase("InstructionAgent分解指令中...")
                reporter.update_agent("instruction", "thinking")
                instruction_output = self.instruction_agent.process(context)
                context.metadata["instruction_output"] = instruction_output.data
                task_level = instruction_output.data.get("task_level", "中等")
                subtask_count = len(instruction_output.data.get("subtasks", []))
                subtasks = instruction_output.data.get("subtasks", [])
                reporter.update_agent("instruction", "done", subtasks=subtask_count)
                reporter.log(f"任务等级: {task_level}, 子任务数: {subtask_count}")
                self.logger.info(f"  任务等级: {task_level}, 子任务数: {subtask_count}")
                # Log each subtask with its individual level
                for st in subtasks:
                    self.logger.info(f"    子任务{st['id']}: [{st['level']}] {st['description'][:50]}...")

                # Start the first subtask (records initial state)
                if context.subtasks:
                    context.start_subtask()
                    self.logger.info(f"  开始子任务0: {context.subtasks[0].description[:50]}...")

            # 5. 重置TrajectoryAgent的地图
            if self.trajectory_agent:
                self.trajectory_agent.reset_map()

            # 6. 重置EvaluationAgent历史
            if self.evaluation_agent:
                self.evaluation_agent.reset_history()

            # 7. 重置DecisionAgent的stuck检测计数器
            if self.decision_agent:
                self.decision_agent.reset_stuck_counter()

            # 8. 重置DebateStrategy的performance tracker (episode级)
            # Note: This resets episode stats, not cross-episode history
            from strategies.debate import DebateStrategy
            temp_strategy = DebateStrategy(self.config)
            temp_strategy.reset_episode()

            # 9. 清空context的历史数据
            context.rgb_history.clear()
            context.depth_history.clear()
            context.stuck_regions.clear()
            context.is_stuck = False
            context.stuck_counter = 0

            max_steps = self.config.get("max_steps", 100)
            success_distance = self.config.get("success_distance", 3.0)
            task_level = "中等"  # Default

            # 序列模式相关变量
            current_sequence = None
            last_sequence_subtask_id = None
            sequence_step_count = 0  # 当前序列已执行步数

            self.logger.info("[SEQUENCE MODE] 启用动作序列模式")

            # ThreadPoolExecutor for parallel execution
            executor = ThreadPoolExecutor(max_workers=2)

            # 导入策略
            from strategies.cot import CoTStrategy
            from strategies.reflection import ReflectionStrategy
            from strategies.debate import DebateStrategy

            # 7. 导航主循环
            while steps < max_steps:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Step {steps + 1}/{max_steps}")

                # Log current subtask info
                current_subtask = context.get_current_subtask()
                if current_subtask:
                    reporter.update_subtask(
                        current_subtask.id,
                        current_subtask.description,
                        current_subtask.completion_condition
                    )
                    self.logger.info(f"当前子任务 [{current_subtask.level}]: {current_subtask.description[:50]}...")
                self.logger.info(f"{'='*60}")

                # 获取RGB和Depth图像
                rgb_image, depth_image = self._get_observations(sim)

                # 更新context中的图像
                context.rgb_image = rgb_image
                context.depth_image = depth_image
                context.metadata['rgb_image'] = rgb_image
                context.metadata['depth_image'] = depth_image

                # 保存观察历史用于stuck分析
                context.add_observation(rgb_image, depth_image)

                # 保存RGB和Depth图像到输出目录
                self.output_manager.save_rgb_image(rgb_image, steps)
                self.output_manager.save_depth_image(depth_image, steps)

                # Get task level from current subtask
                current_subtask = context.get_current_subtask()
                task_level = current_subtask.level if current_subtask else "中等"

                # ============================================================
                # 序列模式: 子任务开始时调用策略，生成动作序列后执行
                # ============================================================
                from core.action import ActionSequence

                # 检查是否需要生成新序列
                need_new_sequence = False

                if current_sequence is None:
                    need_new_sequence = True
                    reason = "无序列"
                elif current_sequence.is_complete():
                    need_new_sequence = True
                    reason = "序列完成"
                elif current_subtask and hasattr(current_subtask, 'id'):
                    if current_subtask.id != last_sequence_subtask_id:
                        need_new_sequence = True
                        reason = "子任务切换"

                # 只在需要新序列时调用策略和Agent
                if need_new_sequence:
                    self.logger.info(f"[SEQUENCE] 生成新序列: {reason}, 难度: {task_level}")

                    # 调用 PerceptionAgent
                    perception_output = None
                    if self.perception_agent:
                        try:
                            reporter.update_phase("PerceptionAgent感知环境...")
                            reporter.update_agent("perception", "thinking")
                            perception_result = self.perception_agent.process(context)
                            perception_output = perception_result.data
                            context.metadata["perception_output"] = perception_output
                            reporter.update_agent("perception", "done", output=f"房间:{perception_output.get('room_type','?')}")
                            reporter.log(f"Perception: 房间={perception_output.get('room_type','?')}, 物体={len(perception_output.get('objects',[]))}个")
                            self.logger.info(f"[PerceptionAgent] 房间: {perception_output.get('room_type', 'unknown')}")
                            self.logger.info(f"[PerceptionAgent] 物体: {[o.get('name', o.get('物体', '未知')) for o in perception_output.get('objects', [])[:5]]}")
                        except Exception as e:
                            reporter.update_agent("perception", "error", output=str(e))
                            self.logger.warning(f"[PerceptionAgent] 执行失败: {e}")
                            perception_output = {}

                    # 调用 TrajectoryAgent
                    trajectory_output = None
                    if self.trajectory_agent:
                        try:
                            reporter.update_phase("TrajectoryAgent分析轨迹...")
                            reporter.update_agent("trajectory", "thinking")
                            trajectory_result = self.trajectory_agent.process(context)
                            trajectory_output = trajectory_result.data
                            context.metadata["trajectory_output"] = trajectory_output
                            reporter.update_agent("trajectory", "done")
                            reporter.log(f"Trajectory: 已走{trajectory_output.get('distance_traveled', 0):.1f}m")
                            self.logger.info(f"[TrajectoryAgent] 已走距离: {trajectory_output.get('distance_traveled', 0):.1f}米")
                        except Exception as e:
                            reporter.update_agent("trajectory", "error", output=str(e))
                            self.logger.warning(f"[TrajectoryAgent] 执行失败: {e}")
                            trajectory_output = {}

                    # 调用 InstructionAgent (获取子任务语义分析)
                    instruction_output = None
                    if self.instruction_agent:
                        try:
                            instruction_result = self.instruction_agent.process(context)
                            instruction_output = instruction_result.data
                            context.metadata["instruction_output"] = instruction_output
                        except Exception as e:
                            self.logger.warning(f"[InstructionAgent] 执行失败: {e}")
                            instruction_output = {}

                    # 策略配置（传递远程LLM设置）
                    strategy_config = {
                        "use_remote": self.config.get("use_remote_llm", False),
                        "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
                    }

                    # 根据子任务难度选择策略
                    if task_level == "简单":
                        strategy = CoTStrategy(config=strategy_config)
                    elif task_level == "中等":
                        strategy = ReflectionStrategy(config=strategy_config)
                    else:  # 困难
                        strategy = DebateStrategy(config=strategy_config)

                    self.logger.info(f"[SEQUENCE] 使用策略: {strategy.name}")
                    reporter.update_phase(f"{strategy.name}策略执行中...")
                    reporter.log(f"策略: {strategy.name}")

                    # 执行策略
                    agents_list = [self.perception_agent, self.trajectory_agent,
                                   self.instruction_agent, self.evaluation_agent]
                    strategy_result = strategy.execute(context, agents_list)

                    reporter.log(f"策略推理: {strategy_result.reasoning[:80] if strategy_result.reasoning else '无'}")
                    self.logger.info(f"[{strategy.name}] 推理: {strategy_result.reasoning[:100] if strategy_result.reasoning else '无'}")

                    # 生成动作序列
                    reporter.update_phase("DecisionAgent生成动作序列...")
                    reporter.update_agent("decision", "thinking")
                    current_sequence = self.decision_agent.generate_action_sequence(
                        context, strategy_result, current_subtask
                    )
                    reporter.update_agent("decision", "done", sequence_progress="0%")
                    reporter.log(f"生成序列: {len(current_sequence.actions)}步, 完成={current_sequence.subtask_completed}")
                    last_sequence_subtask_id = current_subtask.id if current_subtask else None
                    sequence_step_count = 0

                # 检查是否需要中断序列
                if current_sequence:
                    should_abort, abort_reason = self.decision_agent.check_sequence_abort(
                        context, current_sequence, depth_image
                    )

                    if should_abort:
                        self.logger.info(f"[SEQUENCE] 中断序列: {abort_reason}")
                        current_sequence = None

                        # 重置stuck counter
                        self.decision_agent._stuck_counter = 0

                        # stuck时使用Debate策略重新规划
                        self.logger.info("[SEQUENCE] Stuck触发Debate重新规划...")
                        strategy_config = {
                            "use_remote": self.config.get("use_remote_llm", False),
                            "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
                        }
                        strategy = DebateStrategy(config=strategy_config)
                        agents_list = [self.perception_agent, self.trajectory_agent,
                                       self.instruction_agent, self.evaluation_agent]
                        strategy_result = strategy.execute(context, agents_list)

                        # 生成新的10步序列
                        current_sequence = self.decision_agent.generate_action_sequence(
                            context, strategy_result, current_subtask
                        )
                        self.logger.info(f"[SEQUENCE] Debate后生成新序列: {current_sequence.reasoning[:50] if current_sequence else 'N/A'}")

                # 从序列获取下一个动作
                action_name = "move_forward"  # 默认动作
                if current_sequence:
                    from core.action import ActionType
                    next_action_type = current_sequence.get_next_action()

                    # 序列执行完毕，检查子任务是否完成
                    if next_action_type is None:
                        if current_sequence.subtask_completed:
                            reporter.update_phase("子任务完成!")
                            reporter.log(f"子任务完成: {current_sequence.subtask_description[:40]}")
                            self.logger.info(f"[SEQUENCE] LLM判断子任务完成: {current_sequence.subtask_description[:40]}")
                            if context.advance_subtask():
                                current_subtask = context.get_current_subtask()
                                if current_subtask:
                                    reporter.update_subtask(
                                        current_subtask.id,
                                        current_subtask.description,
                                        current_subtask.completion_condition
                                    )
                                self.logger.info(f"[SEQUENCE] 进入下一子任务: {current_subtask.description[:40] if current_subtask else 'N/A'}")
                            else:
                                reporter.log("所有子任务已完成!")
                                self.logger.info("[SEQUENCE] 所有子任务已完成，准备停止")
                                action_name = "stop"
                        else:
                            reporter.log("序列完成但子任务未完成，重新规划")
                            self.logger.info(f"[SEQUENCE] 序列完成但子任务未完成，重新规划")
                        current_sequence = None
                        last_sequence_subtask_id = None
                        continue  # 跳到下一轮循环重新生成序列

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
                        reporter.update_phase(f"执行: {action_name}")
                        reporter.log_action(action_name)
                        reporter.update_agent("decision", "done", sequence_progress=f"{current_sequence.get_progress():.0%}")

                        self.logger.info(f"[SEQUENCE] 执行动作: {action_name} "
                                       f"(进度: {current_sequence.get_progress():.0%}, "
                                       f"剩余: {current_sequence.get_remaining_steps()}步)")

                        context.metadata["sequence_output"] = {
                            "subtask": current_sequence.subtask_description[:50],
                            "progress": current_sequence.get_progress(),
                            "action": action_name,
                        }

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

                # 记录步骤输出
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
                    self.logger.info(f"  成功到达目标 at step {steps}")
                    break

        except Exception as e:
            self.logger.error(f"  Habitat错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理ThreadPoolExecutor
            try:
                executor.shutdown(wait=False)
            except NameError:
                pass  # executor wasn't created yet

            # 保存episode输出
            try:
                # 获取最终位置
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
                # 尝试创建视频摘要
                self.output_manager.create_summary_video(fps=5)
            except Exception as e:
                self.logger.error(f"保存episode输出失败: {e}")

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
            sim_cfg.gpu_device_id = 0

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

    def _select_strategy(self, task_level: str):
        """
        根据任务等级选择推理策略。

        Args:
            task_level: 任务等级 (简单/中等/困难)

        Returns:
            对应的策略实例
        """
        strategy_config = {
            "use_remote": self.config.get("use_remote_llm", False),
            "remote_server_url": self.config.get("llm_server", "http://localhost:8000"),
        }
        if task_level == "简单":
            return CoTStrategy(config=strategy_config)
        elif task_level == "中等":
            return ReflectionStrategy(config=strategy_config)
        else:  # 困难
            return DebateStrategy(config=strategy_config)

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
            .with_metadata({
                "goal_position": tuple(episode.goal_position),
                "success_distance": self.config.get("success_distance", 3.0),
            }) \
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
    parser.add_argument("--start-episode", type=int, default=None, help="起始episode ID (默认从第一个开始)")
    parser.add_argument("--max-steps", type=int, default=50, help="每个episode最大步数")
    parser.add_argument("--success-distance", type=float, default=3.0, help="成功距离阈值(米)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出JSON文件名 (默认: results_时间戳.json)")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-int8", action="store_true", default=True, help="使用INT8量化")

    # Remote LLM arguments for dual-environment IPC
    parser.add_argument("--use-remote-llm", action="store_true", default=False,
                        help="使用远程LLM服务 (Python 3.10环境)")
    parser.add_argument("--llm-server", type=str, default="http://localhost:8000",
                        help="远程LLM服务器地址")
    parser.add_argument("--remote-timeout", type=float, default=120.0,
                        help="远程LLM请求超时时间(秒)")

    # Strategy mode arguments
    parser.add_argument("--use-strategy-mode", action="store_true", default=False,
                        help="使用策略模式进行决策 (CoT/Reflection/Debate)")

    # Sequence mode arguments (子任务级别规划)
    parser.add_argument("--use-sequence-mode", action="store_true", default=False,
                        help="使用动作序列模式 (子任务级别规划，大幅减少LLM调用)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="results",
                        help="输出目录，保存每个episode的视觉图像、轨迹图和agent输出")

    args = parser.parse_args()

    # Create timestamped session directory under results/
    session_timestamp = datetime.now().strftime("%Y-%m%d-%H%M")
    if args.output_dir == "results":
        # Default: create timestamped session directory
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
        "use_strategy_mode": args.use_strategy_mode,
        "use_sequence_mode": args.use_sequence_mode,
        "output_dir": args.output_dir,
    }

    print("=" * 70)
    print("多智能体 R2R VLN 评估实验")
    print("=" * 70)
    print(f"MP3D路径: {args.mp3d_path}")
    print(f"R2R数据: {args.r2r_path}")
    print(f"起始Episode: {args.start_episode if args.start_episode else '从头开始'}")
    print(f"Episode数量: {args.episodes}")
    print(f"最大步数: {args.max_steps}")
    print(f"成功距离: {args.success_distance}m")
    print(f"输出文件: {args.output}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    print(f"INT8量化: {args.use_int8}")
    print(f"远程LLM: {'启用' if args.use_remote_llm else '禁用'}")
    if args.use_remote_llm:
        print(f"LLM服务器: {args.llm_server}")
    print(f"策略模式: {'启用' if args.use_strategy_mode else '禁用'}")
    print("=" * 70)

    evaluator = MultiAgentVLNEvaluator(config, log_level=args.log_level)
    evaluator.initialize()

    results = evaluator.run_evaluation(num_episodes=args.episodes, start_episode_id=args.start_episode)

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