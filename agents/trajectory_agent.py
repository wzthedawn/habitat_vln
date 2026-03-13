"""Trajectory Agent for planning navigation paths.

This version implements:
1. Simple mapping (position recording)
2. LLM-enhanced trajectory summary using Qwen3.5-2B
3. Visited location detection
4. Path quality assessment
"""

from typing import Dict, Any, Optional, List, Tuple
import math
import logging
from collections import defaultdict

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext


class TrajectoryAgent(BaseAgent):
    """
    Agent responsible for trajectory planning and progress tracking.

    Uses Qwen3.5-2B (independent instance) for trajectory summarization.

    Key responsibilities:
    1. Track navigation progress
    2. Build simple map of visited locations
    3. Generate LLM-enhanced trajectory summaries
    4. Detect if current location was visited before
    """

    # Direction names for cardinal directions
    DIRECTION_NAMES = {
        0: "北",
        45: "东北",
        90: "东",
        135: "东南",
        180: "南",
        225: "西南",
        270: "西",
        315: "西北",
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("TrajectoryAgent")

        # Navigation parameters
        self.step_size = self.config.get("step_size", 0.25)
        self.turn_angle = self.config.get("turn_angle", 15.0)
        self.max_waypoints = self.config.get("max_waypoints", 10)
        self.visited_threshold = self.config.get("visited_threshold", 0.5)
        self.use_llm = self.config.get("use_llm", True)

        # Map storage (simple grid-based)
        self._visited_cells: Dict[Tuple[int, int], int] = defaultdict(int)
        self._cell_size = 0.5

        # State tracking
        self._goal_position = None
        self._waypoints = []

        # Model reference
        self._model_manager = None
        self._initialized = False

        # LLM conversation history (independent instance)
        self._conversation_history: List[Dict[str, str]] = []

    @property
    def name(self) -> str:
        return "trajectory_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.TRAJECTORY

    def get_required_inputs(self) -> List[str]:
        return ["position", "rotation"]

    def get_output_keys(self) -> List[str]:
        return ["waypoints", "progress", "path_confidence", "trajectory_summary", "visited"]

    def initialize(self) -> None:
        """Initialize model manager and load LLM."""
        if self._initialized:
            return

        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)

            # Check if using remote LLM
            use_remote = self.config.get("use_remote", False)

            if use_remote:
                self.logger.info("Using remote LLM service for trajectory")
                self._model_manager.load_all_models()  # Only loads YOLO locally
            else:
                self._model_manager.load_all_models()

                # Load Qwen3.5-2B for trajectory if LLM is enabled
                if self.use_llm:
                    self.logger.info("Loading Qwen3.5-2B for trajectory...")
                    if self._model_manager.load_llm("qwen-2b-trajectory"):
                        self.logger.info("Qwen3.5-2B (trajectory) loaded successfully")
                    else:
                        self.logger.warning("Failed to load Qwen3.5-2B, using template-based summaries")
                        self.use_llm = False

            self._initialized = True
            self.logger.info("TrajectoryAgent initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model manager: {e}")
            self._initialized = True

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Process trajectory planning."""
        self.initialize()

        try:
            position = context.position
            rotation = context.rotation
            trajectory = context.trajectory

            # Update map with current position
            self._update_map(position)

            # Check if visited before
            visited = self._check_visited(position)

            # Calculate progress
            progress = self._calculate_progress(context)

            # Evaluate current path
            path_confidence = self._evaluate_path(context)

            # Generate waypoints
            waypoints = self._generate_waypoints(context)

            # Check for course corrections
            corrections = self._check_corrections(context, waypoints)

            # Generate trajectory summary (template-based)
            trajectory_summary = self._generate_trajectory_summary(
                position, rotation, trajectory, context
            )

            # Get current heading
            heading = self._get_heading_name(rotation)

            # Calculate distance traveled
            distance_traveled = self._calculate_distance(trajectory)

            # Update context
            context.metadata["trajectory"] = {
                "progress": progress,
                "path_confidence": path_confidence,
                "waypoints": waypoints,
                "visited": visited,
                "heading": heading,
                "distance_traveled": distance_traveled,
            }

            return AgentOutput.success_output(
                data={
                    "waypoints": waypoints,
                    "progress": progress,
                    "progress_percentage": progress * 100,
                    "path_confidence": path_confidence,
                    "corrections": corrections,
                    "distance_traveled": distance_traveled,
                    "heading": heading,
                    "visited": visited,
                    "trajectory_summary": trajectory_summary,
                    "num_visited_cells": len(self._visited_cells),
                },
                confidence=path_confidence,
                reasoning=f"Progress: {progress*100:.1f}%, visited: {visited}, heading: {heading}",
            )

        except Exception as e:
            self.logger.error(f"Trajectory planning error: {e}")
            return AgentOutput.failure_output([str(e)], "Trajectory planning failed")

    def _update_map(self, position: Tuple[float, float, float]) -> None:
        """Update map with current position."""
        grid_x = int(position[0] / self._cell_size)
        grid_z = int(position[2] / self._cell_size)
        self._visited_cells[(grid_x, grid_z)] += 1

    def _check_visited(self, position: Tuple[float, float, float]) -> bool:
        """Check if current position was visited before."""
        grid_x = int(position[0] / self._cell_size)
        grid_z = int(position[2] / self._cell_size)

        if self._visited_cells.get((grid_x, grid_z), 0) > 1:
            return True

        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue
                if self._visited_cells.get((grid_x + dx, grid_z + dz), 0) > 0:
                    return True

        return False

    def _calculate_progress(self, context: NavContext) -> float:
        """Calculate navigation progress."""
        if not context.trajectory:
            return 0.0

        max_steps = self.config.get("max_steps", 500)
        progress = min(context.step_count / max_steps, 1.0)

        if context.subtasks:
            completed = sum(1 for s in context.subtasks if s.status == "completed")
            total = len(context.subtasks)
            if total > 0:
                progress = max(progress, completed / total)

        return progress

    def _evaluate_path(self, context: NavContext) -> float:
        """Evaluate the quality of the current path."""
        if len(context.trajectory) < 2:
            return 1.0

        backtrack_score = self._check_backtracking(context.trajectory)
        efficiency = self._calculate_efficiency(context.trajectory)
        stuck_penalty = self._check_stuck(context.trajectory)

        confidence = 0.4 * backtrack_score + 0.4 * efficiency + 0.2 * (1 - stuck_penalty)
        return confidence

    def _check_backtracking(self, trajectory: List[Tuple]) -> float:
        """Check for backtracking behavior."""
        if len(trajectory) < 3:
            return 1.0

        changes = 0
        for i in range(1, len(trajectory) - 1):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            next_pos = trajectory[i + 1]

            v1 = (curr[0] - prev[0], curr[2] - prev[2])
            v2 = (next_pos[0] - curr[0], next_pos[2] - curr[2])

            if v1 != (0, 0) and v2 != (0, 0):
                angle = self._angle_between_vectors(v1, v2)
                if angle > 90:
                    changes += 1

        return max(0.0, 1.0 - changes * 0.1)

    def _calculate_efficiency(self, trajectory: List[Tuple]) -> float:
        """Calculate path efficiency."""
        if len(trajectory) < 2:
            return 1.0

        start = trajectory[0]
        end = trajectory[-1]
        direct_distance = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[2] - start[2]) ** 2
        )

        actual_length = self._calculate_distance(trajectory)

        if actual_length == 0:
            return 1.0

        return min(direct_distance / actual_length, 1.0)

    def _check_stuck(self, trajectory: List[Tuple]) -> float:
        """Check if agent is stuck."""
        if len(trajectory) < 5:
            return 0.0

        recent = trajectory[-5:]
        xs = [p[0] for p in recent]
        zs = [p[2] for p in recent]

        variance = (max(xs) - min(xs)) ** 2 + (max(zs) - min(zs)) ** 2

        if variance < 0.01:
            return 1.0
        elif variance < 0.1:
            return 0.5

        return 0.0

    def _calculate_distance(self, trajectory: List[Tuple]) -> float:
        """Calculate total path distance."""
        distance = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]
            distance += math.sqrt(
                (curr[0] - prev[0]) ** 2 + (curr[2] - prev[2]) ** 2
            )
        return distance

    def _angle_between_vectors(self, v1: Tuple, v2: Tuple) -> float:
        """Calculate angle between two 2D vectors in degrees."""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if mag1 == 0 or mag2 == 0:
            return 0

        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))

        return math.degrees(math.acos(cos_angle))

    def _generate_waypoints(self, context: NavContext) -> List[Dict[str, Any]]:
        """Generate navigation waypoints."""
        waypoints = []

        if context.visual_features.object_detections:
            for obj in context.visual_features.object_detections[:self.max_waypoints]:
                if obj.get("is_landmark") or obj.get("is_navigation_object"):
                    waypoints.append({
                        "type": "landmark",
                        "name": obj.get("name", "unknown"),
                        "distance": obj.get("distance", 0),
                        "angle": obj.get("angle", 0),
                        "priority": 1 if obj.get("is_landmark") else 2,
                    })

        if context.subtasks:
            for subtask in context.subtasks:
                if subtask.status == "pending":
                    waypoints.append({
                        "type": "subtask",
                        "subtask_id": subtask.id,
                        "description": subtask.description,
                        "priority": 3,
                    })

        waypoints.sort(key=lambda w: w.get("priority", 3))
        return waypoints[:self.max_waypoints]

    def _check_corrections(
        self, context: NavContext, waypoints: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Check if course corrections are needed."""
        corrections = []

        if len(context.trajectory) >= 5:
            recent = context.trajectory[-5:]
            stuck_score = self._check_stuck(recent)
            if stuck_score > 0.5:
                corrections.append({
                    "type": "stuck",
                    "message": "导航似乎卡住了，考虑更换路线",
                    "severity": "high",
                })

        path_confidence = self._evaluate_path(context)
        if path_confidence < 0.5:
            corrections.append({
                "type": "low_confidence",
                "message": "路径质量较低，考虑重新规划",
                "severity": "medium",
            })

        backtrack_score = self._check_backtracking(context.trajectory) if context.trajectory else 1.0
        if backtrack_score < 0.5:
            corrections.append({
                "type": "backtracking",
                "message": "检测到频繁往返，可能偏离目标",
                "severity": "medium",
            })

        return corrections

    def _get_heading_name(self, rotation: float) -> str:
        """Get heading name from rotation angle."""
        rotation = rotation % 360
        if rotation < 0:
            rotation += 360

        closest_angle = min(self.DIRECTION_NAMES.keys(), key=lambda x: abs(x - rotation))

        if abs(rotation - closest_angle) > 22.5:
            return f"朝向{rotation:.0f}度"

        return self.DIRECTION_NAMES[closest_angle]

    def _generate_trajectory_summary(
        self,
        position: Tuple[float, float, float],
        rotation: float,
        trajectory: List[Tuple],
        context: NavContext = None,
    ) -> str:
        """Generate trajectory summary using LLM or template-based fallback."""
        # Try LLM-enhanced summary first
        if self.use_llm and self._model_manager:
            llm_summary = self._generate_llm_trajectory_summary(position, rotation, trajectory, context)
            if llm_summary:
                return llm_summary

        # Fallback to template-based summary
        return self._generate_template_summary(position, rotation, trajectory)

    def _generate_llm_trajectory_summary(
        self,
        position: Tuple[float, float, float],
        rotation: float,
        trajectory: List[Tuple],
        context: NavContext = None,
    ) -> str:
        """Generate LLM-enhanced trajectory summary."""
        if not self._model_manager:
            return ""

        try:
            distance = self._calculate_distance(trajectory)
            heading = self._get_heading_name(rotation)
            visited_cells = len(self._visited_cells)
            is_stuck = self._check_stuck(trajectory) if len(trajectory) >= 5 else False
            backtrack_score = self._check_backtracking(trajectory) if len(trajectory) >= 3 else 1.0

            # Build prompt
            prompt = f"""导航状态:
- 已走: {distance:.1f}米
- 朝向: {heading}
- 探索: {visited_cells}个区域
- 状态: {"卡住" if is_stuck else "正常"}

要求: 用1句话(不超过30字)总结进度。
直接输出，不要解释。"""

            # Get episode_id for conversation context isolation
            episode_id = context.metadata.get("episode_id", 0) if context else 0
            conversation_id = f"trajectory_ep{episode_id}"

            # Generate summary with conversation context
            response = self._model_manager.generate(
                "qwen-2b-trajectory",
                prompt,
                max_new_tokens=30,  # Reduced from 100 for faster inference
                temperature=0.2,
                conversation_id=conversation_id,
                keep_context=True,
            )

            if response:
                # Update conversation history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                # Keep only recent history
                if len(self._conversation_history) > 10:
                    self._conversation_history = self._conversation_history[-10:]

                return response

        except Exception as e:
            self.logger.warning(f"LLM trajectory summary failed: {e}")

        return ""

    def _generate_template_summary(
        self,
        position: Tuple[float, float, float],
        rotation: float,
        trajectory: List[Tuple],
    ) -> str:
        """Generate template-based trajectory summary (fallback)."""
        distance = self._calculate_distance(trajectory)
        heading = self._get_heading_name(rotation)
        visited_cells = len(self._visited_cells)

        if len(trajectory) <= 1:
            return "刚开始导航"

        # Check if stuck
        if self._check_stuck(trajectory):
            return f"已在原地停留，已探索{visited_cells}个区域"

        # Check if backtracking
        backtrack_score = self._check_backtracking(trajectory)
        if backtrack_score < 0.5:
            return f"已行走{distance:.1f}米，可能偏离目标，朝向{heading}"

        return f"已行走{distance:.1f}米，当前朝向{heading}，已探索{visited_cells}个区域"

    def reset_map(self) -> None:
        """Reset the visited cells map."""
        self._visited_cells.clear()
        self._waypoints.clear()
        self._goal_position = None
        self._conversation_history.clear()
        self._conversation_history.clear()