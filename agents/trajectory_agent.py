"""Trajectory Agent for planning navigation paths.

This version implements:
1. Simple mapping (position recording)
2. LLM-enhanced trajectory summary using Qwen3.5-4B
3. Visited location detection
4. Path quality assessment
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import math
import logging
import time
from collections import defaultdict

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext
from .topology_graph import TopologyGraph, KeyPositionDetector


@dataclass
class ActionRecord:
    """记录一次动作序列及其结果"""
    step_id: int                              # 步骤 ID
    actions: List[str]                        # 动作序列
    start_pos: Tuple[float, float, float]     # 执行前位置
    end_pos: Tuple[float, float, float]       # 执行后位置
    goal_distance_before: float               # 执行前距离目标
    goal_distance_after: float                # 执行后距离目标
    stuck_triggered: bool                     # 是否触发 stuck
    perception_feedback: str = ""             # 感知反馈
    timestamp: float = field(default_factory=time.time)  # 时间戳


class TrajectoryAgent(BaseAgent):
    """
    Agent responsible for trajectory planning and progress tracking.

    Uses Qwen3.5-4B (independent instance) for trajectory summarization.

    Key responsibilities:
    1. Track navigation progress
    2. Build simple map of visited locations
    3. Generate LLM-enhanced trajectory summaries
    4. Detect if current location was visited before
    5. Track stuck regions for escape planning
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

        # NEW: Stuck region tracking with escape history
        # NOTE: Two different structures are used by record_stuck_region() and mark_stuck_region():
        #
        # record_stuck_region() creates:
        # {
        #     "position": Tuple[float, float, float],
        #     "radius": float (default 1.0),
        #     "entry_step": int,
        #     "exit_step": Optional[int],
        #     "escape_actions": List[str],
        #     "failed_attempts": List[str],
        # }
        #
        # mark_stuck_region() creates/updates:
        # {
        #     "position": Tuple[float, float, float],
        #     "radius": float (default 1.0),
        #     "escape_attempts": int,
        #     "escape_success": bool,
        #     "successful_direction": Optional[str] ("left"/"right"/None),
        #     "escape_actions": List[str],
        #     "timestamp": float,
        # }
        self._stuck_regions: List[Dict] = []
        self._stuck_paths: List[List[Tuple]] = []

        # NEW: Action history tracking
        self._action_history: List[ActionRecord] = []

        # NEW: Topology graph for trajectory compression
        self.topology_graph = TopologyGraph()
        self.key_detector = KeyPositionDetector()
        self.last_room = ""
        self._last_topology_node_id: Optional[str] = None

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
        return ["waypoints", "progress", "path_confidence", "trajectory_summary", "visited", "topology_summary"]

    def initialize(self) -> None:
        """Initialize model manager and load LLM."""
        if self._initialized:
            return

        try:
            # First, try to use model_manager from config (passed by experiment)
            if self.config.get("model_manager"):
                self._model_manager = self.config["model_manager"]
                self._initialized = True
                self.logger.info("TrajectoryAgent: using provided model_manager (remote mode)")
                return

            # Fallback: create new model manager (for standalone usage)
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)

            # Check if using remote LLM
            use_remote = self.config.get("use_remote", False)

            if use_remote:
                self.logger.info("Using remote LLM service for trajectory")
                self._model_manager.load_all_models()  # Only loads YOLO locally
            else:
                self._model_manager.load_all_models()

                # Load Qwen3.5-4B for trajectory if LLM is enabled
                if self.use_llm:
                    self.logger.info("Loading Qwen3.5-4B for trajectory...")
                    if self._model_manager.load_llm("qwen-9b-trajectory"):
                        self.logger.info("Qwen3.5-4B (trajectory) loaded successfully")
                    else:
                        self.logger.warning("Failed to load Qwen3.5-4B, using template-based summaries")
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

            # === NEW: Calculate height change ===
            y_change = 0.0
            y_direction = "稳定"
            if len(trajectory) >= 5:
                y_values = [p[1] for p in trajectory[-10:]] if len(trajectory) >= 10 else [p[1] for p in trajectory]
                y_change = y_values[-1] - y_values[0]
                if abs(y_change) > 0.3:
                    y_direction = "上升" if y_change > 0 else "下降"

            # === NEW: Calculate subtask state change ===
            subtask_delta = self._calculate_subtask_delta(context)

            # === NEW: Calculate distance to goal ===
            # DEBUG: Verify goal_position is available
            goal_pos_debug = context.metadata.get("goal_position")
            self.logger.info(f"[DEBUG-Trajectory] goal_position from metadata: {goal_pos_debug}")
            self.logger.info(f"[DEBUG-Trajectory] context.position: {context.position}")
            distance_to_goal = self._calculate_distance_to_goal(context)
            self.logger.info(f"[DEBUG-Trajectory] calculated distance_to_goal: {distance_to_goal}")

            # === NEW: Topology graph detection ===
            room_type = ""
            if hasattr(context, 'visual_features') and context.visual_features:
                room_type = getattr(context.visual_features, 'room_type', '') or ""
            elif context.metadata.get("perception"):
                room_type = context.metadata.get("perception", {}).get("room_type", "")

            # Build action history for key detector (list of action names)
            action_names = []
            if self._action_history:
                # Get recent action names from the last few records
                for record in self._action_history[-3:]:
                    action_names.extend(record.actions)

            # Detect key position
            if len(trajectory) >= 2:
                current_pos = trajectory[-1]
                prev_pos = trajectory[-2] if len(trajectory) > 1 else trajectory[-1]

                node_type = self.key_detector.detect(
                    current_pos=current_pos,
                    prev_pos=prev_pos,
                    action_history=action_names,
                    semantic_info={
                        "room_changed": room_type != self.last_room,
                        "room_type": room_type
                    }
                )

                if node_type:
                    # Add new node to topology graph
                    semantic_info = None
                    if node_type == "room_entrance" and room_type:
                        semantic_info = f"{room_type}入口"
                    elif node_type == "stairs":
                        y_diff = current_pos[1] - prev_pos[1]
                        semantic_info = "楼梯上升" if y_diff > 0 else "楼梯下降"

                    new_node_id = self.topology_graph.add_node(
                        position=current_pos,
                        node_type=node_type,
                        semantic_info=semantic_info,
                        timestamp=context.step_count
                    )

                    # Add edge from last node to new node if exists
                    if self._last_topology_node_id:
                        self.topology_graph.add_edge(
                            source_id=self._last_topology_node_id,
                            target_id=new_node_id,
                            action_sequence=action_names[-5:] if action_names else []
                        )

                    self._last_topology_node_id = new_node_id
                    self.logger.debug(f"[Topology] Added node: {node_type} at {current_pos}")

                # Update current node position
                self.topology_graph.update_current_node(position)

                # Update last room
                if room_type:
                    self.last_room = room_type

            # Periodic pruning check
            if context.step_count % 50 == 0 and len(self.topology_graph.nodes) > 20:
                self.topology_graph.prune_nodes()
                self.logger.debug(f"[Topology] Pruned to {len(self.topology_graph.nodes)} nodes")

            # Get topology summary
            topology_summary = self.topology_graph.get_summary()

            # Update context
            context.metadata["trajectory"] = {
                "progress": progress,
                "path_confidence": path_confidence,
                "waypoints": waypoints,
                "visited": visited,
                "heading": heading,
                "distance_traveled": distance_traveled,
                "y_change": y_change,
                "y_direction": y_direction,
                # NEW: Structured state data
                "subtask_delta": subtask_delta,
                "distance_to_goal": distance_to_goal,
            }

            return AgentOutput.success_output(
                data={
                    # ===== NEW: Structured state data =====
                    "state": {
                        "position": [round(position[0], 2), round(position[1], 2), round(position[2], 2)],
                        "rotation_deg": round(math.degrees(rotation), 1),
                        "step": context.step_count,
                    },

                    # ===== NEW: Subtask state change (core) =====
                    "subtask_delta": subtask_delta,

                    # ===== NEW: Global navigation state =====
                    "navigation": {
                        "total_distance": round(distance_traveled, 2),
                        "distance_to_goal": round(distance_to_goal, 2),
                        "heading": heading,
                        "visited_cells": len(self._visited_cells),
                        "efficiency": round(path_confidence, 2),
                    },

                    # ===== Keep existing fields for compatibility =====
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
                    # ===== NEW: Topology summary for DecisionAgent =====
                    "topology_summary": topology_summary,
                },
                confidence=path_confidence,
                reasoning=f"Progress: {progress*100:.1f}%, visited: {visited}, heading: {heading}",
            )

        except Exception as e:
            self.logger.error(f"[Trajectory] 错误: {e}")
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

    def _calculate_subtask_delta(self, context: NavContext) -> Dict[str, Any]:
        """Calculate state change within current subtask (generic, no judgment).

        Returns position_delta and rotation_delta for DecisionAgent to interpret.
        """
        current_subtask = context.get_current_subtask()

        # Basic info
        result = {
            "subtask_id": current_subtask.id if current_subtask else -1,
            "subtask_description": current_subtask.description if current_subtask else "",
        }

        # Current state
        current_pos = context.position
        current_rot = context.rotation

        # Get subtask start state
        start_context = current_subtask.start_context if current_subtask and current_subtask.start_context else {}
        start_pos = start_context.get("position", current_pos)
        start_rot = start_context.get("rotation", current_rot)

        # Position change (XYZ all included)
        dx = current_pos[0] - start_pos[0]
        dy = current_pos[1] - start_pos[1]  # Y change (vertical)
        dz = current_pos[2] - start_pos[2]
        horizontal_dist = math.sqrt(dx*dx + dz*dz)
        total_dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        result["position_delta"] = {
            "start": [round(start_pos[0], 2), round(start_pos[1], 2), round(start_pos[2], 2)],
            "current": [round(current_pos[0], 2), round(current_pos[1], 2), round(current_pos[2], 2)],
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "dz": round(dz, 2),
            "horizontal_distance": round(horizontal_dist, 2),
            "total_distance": round(total_dist, 2),
        }

        # Rotation change (handle boundary crossing)
        start_deg = math.degrees(start_rot)
        current_deg = math.degrees(current_rot)
        delta_deg = current_deg - start_deg

        # Handle -180/180 boundary
        if delta_deg > 180:
            delta_deg -= 360
        elif delta_deg < -180:
            delta_deg += 360

        # Determine direction
        if abs(delta_deg) < 5:
            rot_direction = "none"
        elif delta_deg > 0:
            rot_direction = "left"   # Positive angle = turn left
        else:
            rot_direction = "right"  # Negative angle = turn right

        result["rotation_delta"] = {
            "start_deg": round(start_deg, 1),
            "current_deg": round(current_deg, 1),
            "delta_deg": round(abs(delta_deg), 1),
            "direction": rot_direction,
        }

        # Steps in subtask
        start_step = start_context.get("step", context.step_count)
        result["steps_in_subtask"] = context.step_count - start_step

        return result

    def _calculate_distance_to_goal(self, context: NavContext) -> float:
        """Calculate distance to goal position."""
        goal_pos = context.metadata.get("goal_position")
        if not goal_pos or not context.position:
            return 0.0

        dx = goal_pos[0] - context.position[0]
        dy = goal_pos[1] - context.position[1]
        dz = goal_pos[2] - context.position[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

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
                if subtask.status in ["pending", "in_progress"]:
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
        return self._generate_template_summary(position, rotation, trajectory, context)

    def _generate_llm_trajectory_summary(
        self,
        position: Tuple[float, float, float],
        rotation: float,
        trajectory: List[Tuple],
        context: NavContext = None,
    ) -> str:
        """Generate LLM-enhanced trajectory summary with height awareness."""
        if not self._model_manager:
            return ""

        try:
            distance = self._calculate_distance(trajectory)
            heading = self._get_heading_name(rotation)
            visited_cells = len(self._visited_cells)
            is_stuck = self._check_stuck(trajectory) if len(trajectory) >= 5 else False
            backtrack_score = self._check_backtracking(trajectory) if len(trajectory) >= 3 else 1.0

            # === NEW: Height change analysis ===
            y_trajectory = ""
            y_direction = "稳定"
            y_change_value = 0.0

            if len(trajectory) >= 5:
                y_values = [p[1] for p in trajectory[-10:]] if len(trajectory) >= 10 else [p[1] for p in trajectory]
                y_start = y_values[0]
                y_end = y_values[-1]
                y_change_value = y_end - y_start

                if abs(y_change_value) > 0.3:
                    y_direction = "上升" if y_change_value > 0 else "下降"
                    y_trajectory = f"高度{y_direction}{abs(y_change_value):.1f}米"

            # === NEW: Enhanced prompt template ===
            # Calculate goal distance if available
            goal_distance = 0.0
            if context and hasattr(context, 'metadata'):
                goal_pos = context.metadata.get("goal_position")
                if goal_pos:
                    goal_distance = math.sqrt(
                        (goal_pos[0] - position[0])**2 +
                        (goal_pos[2] - position[2])**2
                    )

            # Convert rotation to heading degrees (0-360)
            heading_degrees = int(math.degrees(rotation)) % 360

            prompt = f"""/no_think
直接输出JSON:

{{
  "当前位置": [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}],
  "当前朝向": {heading_degrees},
  "已走距离": {distance:.1f},
  "距离目标": {goal_distance:.1f}
}}"""

            # Get episode_id for conversation context isolation
            episode_id = context.metadata.get("episode_id", 0) if context else 0
            conversation_id = f"trajectory_ep{episode_id}"

            # Generate summary with conversation context
            response = self._model_manager.generate(
                "qwen-9b-trajectory",
                prompt,
                max_new_tokens=50,  # Optimized: actual usage 21 tokens (was 256)
                temperature=0.1,  # Lower temperature for more consistent JSON
                conversation_id=conversation_id,
                keep_context=True,
            )

            if response:
                # Clean the response to remove thinking artifacts
                cleaned_response = self._clean_llm_output(response)

                # Parse JSON response
                parsed = self._parse_trajectory_json(cleaned_response)
                if parsed:
                    # Log in the specified format
                    self.logger.info(
                        f"[Trajectory] 位置: {parsed['当前位置']}, "
                        f"朝向: {parsed['当前朝向']}°, "
                        f"已走: {parsed['已走距离']:.1f}m"
                    )
                    # Return formatted string for display
                    return self._format_trajectory_output(parsed)

                # Fallback to cleaned response if JSON parsing failed
                # Return empty string to trigger template-based summary
                self.logger.debug(f"[Trajectory] JSON解析失败，使用模板")
                return ""

        except Exception as e:
            self.logger.warning(f"LLM trajectory summary failed: {e}")

        return ""

    def _parse_trajectory_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse trajectory JSON from LLM response.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed dictionary or None if parsing failed
        """
        import json
        import re

        # Try to find JSON in response - handle various formats
        # Pattern 1: Full JSON with Chinese keys
        json_patterns = [
            r'\{[^{}]*"当前位置"[^{}]*\}',  # Single level JSON
            r'\{(?:[^{}]|\{[^{}]*\})*\}',   # Nested JSON up to 2 levels
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for json_str in matches:
                # Check if this match contains our expected keys
                if '"当前位置"' in json_str:
                    try:
                        data = json.loads(json_str)
                        return {
                            "当前位置": data.get("当前位置", [0.0, 0.0, 0.0]),
                            "当前朝向": int(data.get("当前朝向", 0)) % 360,
                            "已走距离": float(data.get("已走距离", 0.0)),
                            "距离目标": float(data.get("距离目标", 0.0)),
                        }
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        self.logger.debug(f"Failed to parse trajectory JSON: {e}")
                        continue

        return None

    def _format_trajectory_output(self, data: Dict[str, Any]) -> str:
        """Format trajectory data into readable string.

        Args:
            data: Parsed trajectory data dictionary

        Returns:
            Formatted string for display
        """
        pos = data["当前位置"]
        return (
            f"当前位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
            f"当前朝向: {data['当前朝向']}度, "
            f"已走距离: {data['已走距离']:.1f}米, "
            f"距离目标: {data['距离目标']:.1f}米"
        )

    def _clean_llm_output(self, response: str) -> str:
        """Clean LLM output by removing thinking process and artifacts."""
        import re

        cleaned = response.strip()

        # Remove thinking process markers (Qwen3.5 style)
        thinking_patterns = [
            r'\d+\.\s*\*\*[^*]+\*\*:',  # "1. **Analyze...:"
            r'\d+\.\s*\*[^*]+\*:',       # "1. *Analyze...:"
            r'<think>.*?</think>',       # <think>...</think>
            r'```.*?```',                # code blocks
        ]

        for pattern in thinking_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)

        # Remove leading numbers like "2. " at the start
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned.strip())

        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _generate_template_summary(
        self,
        position: Tuple[float, float, float],
        rotation: float,
        trajectory: List[Tuple],
        context: NavContext = None,
    ) -> str:
        """Generate template-based trajectory summary (fallback) in JSON format."""
        import json

        distance = self._calculate_distance(trajectory)
        heading_degrees = int(math.degrees(rotation)) % 360

        # Calculate goal distance if available
        goal_distance = 0.0
        if context and hasattr(context, 'metadata'):
            goal_pos = context.metadata.get("goal_position")
            if goal_pos:
                goal_distance = math.sqrt(
                    (goal_pos[0] - position[0])**2 +
                    (goal_pos[2] - position[2])**2
                )

        # Build structured output
        output_data = {
            "当前位置": [round(position[0], 2), round(position[1], 2), round(position[2], 2)],
            "当前朝向": heading_degrees,
            "已走距离": round(distance, 1),
            "距离目标": round(goal_distance, 1)
        }

        # Log in the specified format
        self.logger.info(
            f"[Trajectory] 位置: {output_data['当前位置']}, "
            f"朝向: {output_data['当前朝向']}°, "
            f"已走: {output_data['已走距离']:.1f}m"
        )

        # Print to console
        print(f"[TrajectoryAgent] 位置: {output_data['当前位置']}, 朝向: {output_data['当前朝向']}°, 已走: {output_data['已走距离']:.1f}m")

        return json.dumps(output_data, ensure_ascii=False)

    def reset_map(self) -> None:
        """Reset the visited cells map and topology graph."""
        self._visited_cells.clear()
        self._waypoints.clear()
        self._goal_position = None
        self._conversation_history.clear()
        self._stuck_regions.clear()
        self._stuck_paths.clear()
        self._action_history.clear()
        # Reset topology graph
        self.topology_graph = TopologyGraph()
        self.last_room = ""
        self._last_topology_node_id = None

    def update_topology_only(
        self,
        current_pos: Tuple[float, float, float],
        prev_pos: Tuple[float, float, float],
        current_rot: float,
        prev_rot: float,
        step_count: int,
        action_sequence: List[str] = []
    ) -> Optional[str]:
        """轻量级拓扑更新（序列执行期间调用，仅检测stairs/junction）。

        Args:
            current_pos: 当前位置 (x, y, z)
            prev_pos: 前一步位置
            current_rot: 当前旋转角度（弧度）
            prev_rot: 前一步旋转角度
            step_count: 当前步数
            action_sequence: 当前执行的动作序列（用于边的 action_sequence 属性）

        Returns:
            新节点ID（如果添加了节点），否则None
        """
        # 检测stairs（高度变化 > 0.5m）
        y_diff = current_pos[1] - prev_pos[1]
        if abs(y_diff) > 0.5:
            semantic_info = "楼梯上升" if y_diff > 0 else "楼梯下降"
            new_node_id = self.topology_graph.add_node(
                position=current_pos,
                node_type="stairs",
                semantic_info=semantic_info,
                timestamp=step_count
            )
            if self._last_topology_node_id:
                self.topology_graph.add_edge(
                    source_id=self._last_topology_node_id,
                    target_id=new_node_id,
                    action_sequence=action_sequence
                )
            self._last_topology_node_id = new_node_id
            self.logger.debug(f"[Topology-lite] Added stairs node: {semantic_info}")
            return new_node_id

        # 检测junction（转向 > 28°，约0.5弧度）
        rot_diff = current_rot - prev_rot
        # 处理-π/π边界
        if rot_diff > math.pi:
            rot_diff -= 2 * math.pi
        elif rot_diff < -math.pi:
            rot_diff += 2 * math.pi

        if abs(rot_diff) > 0.5:
            new_node_id = self.topology_graph.add_node(
                position=current_pos,
                node_type="junction",
                semantic_info="转向点",
                timestamp=step_count
            )
            if self._last_topology_node_id:
                self.topology_graph.add_edge(
                    source_id=self._last_topology_node_id,
                    target_id=new_node_id,
                    action_sequence=action_sequence
                )
            self._last_topology_node_id = new_node_id
            self.logger.debug(f"[Topology-lite] Added junction node")
            return new_node_id

        # 无关键位置变化，更新当前节点
        self.topology_graph.update_current_node(current_pos)
        return None

    def record_action_sequence(
        self,
        step_id: int,
        actions: List[str],
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        goal_distance_before: float,
        goal_distance_after: float,
        stuck_triggered: bool,
        perception_feedback: str = ""
    ) -> None:
        """记录一次动作序列及其结果。

        Args:
            step_id: 步骤 ID
            actions: 动作序列 (如 ['turn_right', 'turn_right', 'forward'])
            start_pos: 执行前位置 (x, y, z)
            end_pos: 执行后位置 (x, y, z)
            goal_distance_before: 执行前距离目标
            goal_distance_after: 执行后距离目标
            stuck_triggered: 是否触发 stuck
            perception_feedback: 感知反馈
        """
        record = ActionRecord(
            step_id=step_id,
            actions=actions,
            start_pos=start_pos,
            end_pos=end_pos,
            goal_distance_before=goal_distance_before,
            goal_distance_after=goal_distance_after,
            stuck_triggered=stuck_triggered,
            perception_feedback=perception_feedback,
        )
        self._action_history.append(record)
        self.logger.info(f"[Trajectory] 记录动作序列：{len(actions)}步，stuck={stuck_triggered}")

    def record_stuck_region(
        self,
        position: Tuple[float, float, float],
        radius: float = 1.0,
        escape_actions: List[str] = None,
        failed_attempts: List[str] = None
    ) -> None:
        """Record a stuck region for future reference.

        Args:
            position: Position where stuck occurred
            radius: Radius of stuck region
            escape_actions: Actions that successfully escaped
            failed_attempts: Actions that failed to escape
        """
        stuck_record = {
            "position": position,
            "radius": radius,
            "entry_step": len(self._visited_cells),
            "exit_step": None,
            "escape_actions": escape_actions or [],
            "failed_attempts": failed_attempts or [],
        }
        self._stuck_regions.append(stuck_record)
        self.logger.info(f"[Trajectory] 记录卡住区域: {position}")

    def is_in_stuck_region(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is within any known stuck region.

        Args:
            position: Position to check

        Returns:
            True if position is within a stuck region
        """
        for region in self._stuck_regions:
            dx = position[0] - region["position"][0]
            dz = position[2] - region["position"][2]
            dist = math.sqrt(dx*dx + dz*dz)
            if dist < region["radius"]:
                return True
        return False

    def get_stuck_region_info(self, position: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        """Get information about a stuck region at a position.

        Args:
            position: Position to check

        Returns:
            Stuck region info if found, None otherwise
        """
        for region in self._stuck_regions:
            dx = position[0] - region["position"][0]
            dz = position[2] - region["position"][2]
            dist = math.sqrt(dx * dx + dz * dz)

            if dist < region["radius"]:
                return {
                    "is_stuck_region": True,
                    "escape_attempts": region.get("escape_attempts", 0),
                    "escape_success": region.get("escape_success", False),
                    "successful_direction": region.get("successful_direction"),
                }
        return None

    def get_stuck_recovery_suggestion(
        self,
        context: NavContext,
        depth_clear_direction: Optional[str] = None
    ) -> Dict[str, Any]:
        """根据拓扑历史 + 深度图分析返回恢复方向建议。

        Args:
            context: 当前导航上下文
            depth_clear_direction: 来自run_vln_experiment的深度图分析结果

        Returns:
            Dict包含：
            - preferred_direction: 推荐方向 ("left"/"right"/None)
            - avoid_directions: 需避免的方向列表
            - reason: 推荐理由
            - confidence: 置信度 (0.5-0.8)
            - use_depth_analysis: 是否需要深度图辅助
        """
        current_pos = context.position if hasattr(context, 'position') else (0, 0, 0)
        current_step = context.step_count if hasattr(context, 'step_count') else 0

        # 查找匹配的stuck_region
        matched_region = self._find_matching_stuck_region(current_pos)

        if matched_region:
            # 检查是否有历史成功方向
            if matched_region.get("successful_direction"):
                return {
                    "preferred_direction": matched_region["successful_direction"],
                    "avoid_directions": matched_region.get("failed_directions", []),
                    "reason": "历史成功方向",
                    "confidence": 0.8,
                    "use_depth_analysis": False
                }

            # 有历史失败方向（无成功）
            return {
                "preferred_direction": depth_clear_direction,
                "avoid_directions": matched_region.get("failed_directions", []),
                "reason": "避开历史失败方向",
                "confidence": 0.6,
                "use_depth_analysis": True
            }

        # 无历史记录 → 创建新stuck_region
        self._create_stuck_region(current_pos, current_step)
        return {
            "preferred_direction": depth_clear_direction,
            "avoid_directions": [],
            "reason": "首次卡住，使用深度图分析",
            "confidence": 0.5,
            "use_depth_analysis": True
        }

    def _find_matching_stuck_region(
        self,
        position: Tuple[float, float, float]
    ) -> Optional[Dict]:
        """通过距离阈值匹配已知stuck_region。

        Note: Returns full region dict unlike get_stuck_region_info which returns
        a subset. Kept separate for backward compatibility with recovery logic.

        Args:
            position: 当前位置

        Returns:
            匹配的stuck_region或None
        """
        if not self._stuck_regions:
            return None

        for region in self._stuck_regions:
            dx = position[0] - region["position"][0]
            dz = position[2] - region["position"][2]
            distance = math.sqrt(dx * dx + dz * dz)

            if distance < region.get("radius", 1.0):
                return region

        return None

    def _create_stuck_region(
        self,
        position: Tuple[float, float, float],
        step: int
    ) -> None:
        """创建新的stuck_region记录（扩展结构）。

        Schema is aligned with mark_stuck_region for compatibility:
        - radius: 1.0 (matching mark_stuck_region)
        - escape_success: False (will be updated on successful escape)
        - escape_actions: [] (will be populated on successful escapes)
        - timestamp: 0.0 (placeholder, will be updated on actual escape)

        Extended fields for recovery feature:
        - failed_directions: list of directions that didn't work
        - last_attempt_step: step of last escape attempt
        - created_at: step when region was created

        Args:
            position: 卡住位置
            step: 当前步数
        """
        import time
        new_region = {
            # Core fields (matching mark_stuck_region schema)
            "position": position,
            "radius": 1.0,  # 匹配 mark_stuck_region
            "escape_attempts": 0,
            "escape_success": False,  # 兼容字段
            "successful_direction": None,
            "escape_actions": [],  # 兼容字段
            "timestamp": time.time(),  # 兼容字段
            # Extended fields for recovery feature
            "failed_directions": [],
            "last_attempt_step": step,
            "created_at": step,
        }

        self._stuck_regions.append(new_region)
        self.logger.info(f"[Trajectory] Created stuck_region at {position}")

    def mark_escape_result(
        self,
        position: Tuple[float, float, float],
        direction: str,
        success: bool,
        step: int
    ) -> None:
        """延迟更新escape结果。

        Args:
            position: 卡住位置
            direction: 尝试方向 ("left"/"right")
            success: 是否成功逃离
            step: 当前步数
        """
        region = self._find_matching_stuck_region(position)
        if not region:
            self.logger.warning(f"[Trajectory] No stuck_region found at {position}")
            return

        region["last_attempt_step"] = step
        region["escape_attempts"] = region.get("escape_attempts", 0) + 1

        if success:
            region["successful_direction"] = direction
            region["escape_success"] = True  # 兼容字段
            # 从失败列表移除（如果之前标记过）
            failed_dirs = region.get("failed_directions", [])
            if direction in failed_dirs:
                failed_dirs.remove(direction)
                region["failed_directions"] = failed_dirs
            self.logger.info(f"[Trajectory] Escape success with {direction} at {region['position']}")
        else:
            failed_dirs = region.get("failed_directions", [])
            if direction not in failed_dirs:
                failed_dirs.append(direction)
                region["failed_directions"] = failed_dirs
            self.logger.warning(f"[Trajectory] Escape failed with {direction} at {region['position']}")

    # ========== Action History Methods ==========

    def get_similar_location_history(
        self,
        current_pos: Tuple[float, float, float],
        radius: float = 2.0,
        max_records: int = 5
    ) -> List[ActionRecord]:
        """获取相似位置的历史动作记录。

        Args:
            current_pos: 当前位置 (x, y, z)
            radius: 搜索半径（米）
            max_records: 返回最大记录数

        Returns:
            历史记录列表，按距离排序
        """
        similar_records = []

        for record in self._action_history:
            # 计算与起始位置的距离
            start_pos = record.start_pos
            dx = current_pos[0] - start_pos[0]
            dz = current_pos[2] - start_pos[2]
            dist = math.sqrt(dx * dx + dz * dz)

            if dist <= radius:
                similar_records.append((dist, record))

        # 按距离排序
        similar_records.sort(key=lambda x: x[0])

        # 返回最近的 max_records 条记录
        return [record for _, record in similar_records[:max_records]]

    def get_history_summary(self, current_pos: Tuple[float, float, float], radius: float = 2.0) -> str:
        """获取位置历史记录的摘要文本。

        Args:
            current_pos: 当前位置
            radius: 搜索半径

        Returns:
            历史记录摘要文本
        """
        history = self.get_similar_location_history(current_pos, radius)

        if not history:
            return "附近无历史记录"

        summaries = []
        for record in history[:3]:  # 最多显示 3 条
            action_str = " → ".join(record.actions[:5])
            if len(record.actions) > 5:
                action_str += "..."

            # 判断动作效果
            dist_change = record.goal_distance_after - record.goal_distance_before
            if dist_change < -0.5:
                effect = "靠近目标"
            elif dist_change > 0.5:
                effect = "远离目标"
            elif record.stuck_triggered:
                effect = "触发 stuck"
            else:
                effect = "无明显效果"

            summaries.append(f"动作 [{action_str}] {effect}")

        return "; ".join(summaries)

    def build_action_history_opinion(
        self,
        current_pos: Tuple[float, float, float],
        radius: float = 2.0
    ) -> Dict[str, Any]:
        """基于动作历史生成决策意见。

        分析相似位置的历史记录，提供：
        1. 应避免的动作（导致 stuck 或远离目标）
        2. 推荐的动作（成功靠近目标）

        Args:
            current_pos: 当前位置
            radius: 搜索半径

        Returns:
            包含建议和约束的意见字典
        """
        opinion = {
            "recommended_actions": [],
            "avoid_actions": [],
            "reasoning": "基于历史动作分析",
            "confidence": 0.5,
        }

        history = self.get_similar_location_history(current_pos, radius)

        if not history:
            opinion["reasoning"] = "附近无历史记录，采用默认策略"
            return opinion

        # 分析历史记录
        successful_actions = []
        failed_actions = []

        for record in history:
            dist_change = record.goal_distance_after - record.goal_distance_before

            if record.stuck_triggered:
                # 触发 stuck 的动作序列
                failed_actions.extend(record.actions[:3])  # 记录前 3 个动作
            elif dist_change > 0.5:
                # 远离目标的动作
                failed_actions.extend(record.actions[:3])
            elif dist_change < -0.3:
                # 靠近目标的动作
                successful_actions.extend(record.actions[:3])

        # 统计动作频率
        from collections import Counter
        if failed_actions:
            failed_counts = Counter(failed_actions)
            opinion["avoid_actions"] = [action for action, _ in failed_counts.most_common(3)]
            opinion["reasoning"] += f"，避免{opinion['avoid_actions'][0] if opinion['avoid_actions'] else '无效动作'}"

        if successful_actions:
            success_counts = Counter(successful_actions)
            opinion["recommended_actions"] = [action for action, _ in success_counts.most_common(3)]
            opinion["reasoning"] += f"，推荐{opinion['recommended_actions'][0] if opinion['recommended_actions'] else '成功动作'}"
            opinion["confidence"] = 0.7

        return opinion