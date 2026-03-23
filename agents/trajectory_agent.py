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

        # NEW: Stuck region tracking
        self._stuck_regions: List[Dict] = []
        self._stuck_paths: List[List[Tuple]] = []

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

            # === NEW: Calculate height change ===
            y_change = 0.0
            y_direction = "稳定"
            if len(trajectory) >= 5:
                y_values = [p[1] for p in trajectory[-10:]] if len(trajectory) >= 10 else [p[1] for p in trajectory]
                y_change = y_values[-1] - y_values[0]
                if abs(y_change) > 0.3:
                    y_direction = "上升" if y_change > 0 else "下降"

            # Update context
            context.metadata["trajectory"] = {
                "progress": progress,
                "path_confidence": path_confidence,
                "waypoints": waypoints,
                "visited": visited,
                "heading": heading,
                "distance_traveled": distance_traveled,
                # NEW: Height change info
                "y_change": y_change,
                "y_direction": y_direction,
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

            prompt = f"""直接输出JSON，不要思考或解释:

{{
  "当前位置": [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}],
  "当前朝向": {heading_degrees},
  "已走距离": {distance:.1f},
  "距离目标": {goal_distance:.1f}
}}

只输出上面JSON，无其他内容。"""

            # Get episode_id for conversation context isolation
            episode_id = context.metadata.get("episode_id", 0) if context else 0
            conversation_id = f"trajectory_ep{episode_id}"

            # Generate summary with conversation context
            response = self._model_manager.generate(
                "qwen-2b-trajectory",
                prompt,
                max_new_tokens=80,  # Enough for JSON output
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

        return json.dumps(output_data, ensure_ascii=False)

    def reset_map(self) -> None:
        """Reset the visited cells map."""
        self._visited_cells.clear()
        self._waypoints.clear()
        self._goal_position = None
        self._conversation_history.clear()
        self._stuck_regions.clear()
        self._stuck_paths.clear()

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

    def get_stuck_escape_opinion(
        self,
        trajectory: List[Tuple],
        stuck_regions: List[Dict],
        current_rotation: float = 0.0
    ) -> Dict[str, Any]:
        """Provide stuck escape opinion based on trajectory history.

        Analyzes the trajectory to find unexplored directions and
        escape routes from stuck regions.

        Args:
            trajectory: List of positions in the trajectory
            stuck_regions: List of known stuck regions
            current_rotation: Current rotation angle in radians

        Returns:
            Dict with escape direction, confidence, and reasoning
        """
        opinion = {
            "direction": "right",
            "confidence": 0.5,
            "reason": "默认建议",
            "stop_condition": "",
            "agent_source": "trajectory",
        }

        if len(trajectory) < 3:
            opinion["reason"] = "轨迹数据不足"
            return opinion

        # Analyze recent trajectory for movement patterns
        recent = trajectory[-10:] if len(trajectory) >= 10 else trajectory

        # Calculate movement directions
        directions = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]
            curr = recent[i]
            dx = curr[0] - prev[0]
            dz = curr[2] - prev[2]
            dist = math.sqrt(dx*dx + dz*dz)
            if dist > 0.05:
                angle = math.atan2(-dx, dz)  # Habitat coordinate system
                directions.append(angle)

        if not directions:
            opinion["reason"] = "无有效移动记录"
            return opinion

        # Find explored directions
        explored_angles = set()
        for angle in directions:
            # Quantize to 45-degree sectors
            sector = int(math.degrees(angle) // 45) * 45
            explored_angles.add(sector)

        # All possible directions (8 sectors)
        all_sectors = {-180, -135, -90, -45, 0, 45, 90, 135, 180}
        unexplored = all_sectors - explored_angles

        # Current facing direction
        current_sector = int(math.degrees(current_rotation) // 45) * 45

        # Find best unexplored direction relative to current facing
        if unexplored:
            # Find closest unexplored sector to current direction
            min_diff = 360
            best_sector = current_sector
            for sector in unexplored:
                diff = abs(sector - current_sector)
                if diff > 180:
                    diff = 360 - diff
                if diff < min_diff:
                    min_diff = diff
                    best_sector = sector

            # Determine turn direction
            angle_diff = best_sector - current_sector
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            if angle_diff > 22:
                opinion["direction"] = "left"
                opinion["confidence"] = 0.7
                opinion["reason"] = f"未探索方向在左侧({angle_diff}°)"
            elif angle_diff < -22:
                opinion["direction"] = "right"
                opinion["confidence"] = 0.7
                opinion["reason"] = f"未探索方向在右侧({-angle_diff}°)"
            else:
                opinion["direction"] = "forward"
                opinion["confidence"] = 0.75
                opinion["reason"] = "前方为未探索方向"

        else:
            # All directions explored - check for blocked paths
            opinion["reason"] = "所有方向已探索"

            # Check if in known stuck region
            if stuck_regions:
                current_pos = trajectory[-1] if trajectory else (0, 0, 0)
                for region in stuck_regions:
                    dx = current_pos[0] - region["position"][0]
                    dz = current_pos[2] - region["position"][2]
                    dist = math.sqrt(dx*dx + dz*dz)

                    if dist < region["radius"] * 1.5:
                        # In or near a stuck region - use escape actions if available
                        if region.get("escape_actions"):
                            last_escape = region["escape_actions"][-1]
                            opinion["direction"] = last_escape
                            opinion["confidence"] = 0.8
                            opinion["reason"] = "使用已知逃离路径"
                        break

        opinion["stop_condition"] = "移动1米或进入新区域"

        return opinion

    def get_stuck_regions_summary(self) -> List[Dict]:
        """Get summary of all stuck regions."""
        return self._stuck_regions.copy()

    def mark_stuck_region(
        self,
        position: Tuple[float, float, float],
        escape_success: bool,
        escape_direction: Optional[str] = None
    ) -> None:
        """Mark a stuck region with escape result for future reference.

        Phase 5: Part of escape verification closed loop.

        Args:
            position: Position where stuck occurred
            escape_success: Whether escape was successful
            escape_direction: Direction that successfully escaped (if any)
        """
        # Check if this region already exists
        for region in self._stuck_regions:
            dx = position[0] - region["position"][0]
            dz = position[2] - region["position"][2]
            dist = math.sqrt(dx * dx + dz * dz)

            if dist < region["radius"]:
                # Update existing record
                region["escape_attempts"] = region.get("escape_attempts", 0) + 1
                if escape_success:
                    region["escape_success"] = True
                    region["successful_direction"] = escape_direction
                    if escape_direction and escape_direction not in region.get("escape_actions", []):
                        region.setdefault("escape_actions", []).append(escape_direction)
                self.logger.info(f"[Trajectory] 更新卡住区域: {position}, 成功={escape_success}")
                return

        # Create new stuck region record
        import time
        new_region = {
            "position": position,
            "radius": 1.0,
            "escape_attempts": 1,
            "escape_success": escape_success,
            "successful_direction": escape_direction if escape_success else None,
            "escape_actions": [escape_direction] if escape_success and escape_direction else [],
            "timestamp": time.time(),
        }
        self._stuck_regions.append(new_region)
        self.logger.info(f"[Trajectory] 新卡住区域: {position}")

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

    def build_debate_opinion(
        self,
        context: "NavContext",
        stuck_regions: List[Dict] = None
    ) -> "DebateOpinion":
        """Build a DebateOpinion for the debate strategy using LLM.

        Args:
            context: Navigation context
            stuck_regions: Known stuck regions

        Returns:
            DebateOpinion with trajectory-based constraints
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        trajectory = context.trajectory if context.trajectory else []
        stuck_regions = stuck_regions or []

        # Analyze recent action history
        turn_left_count = 0
        turn_right_count = 0
        forward_failures = 0

        if context.action_history:
            recent = context.action_history[-10:]
            turn_left_count = sum(1 for a in recent if a.action_type.name == "TURN_LEFT")
            turn_right_count = sum(1 for a in recent if a.action_type.name == "TURN_RIGHT")

        # Use LLM for trajectory-based opinion
        if self._model_manager:
            return self._build_trajectory_opinion_with_llm(
                context, turn_left_count, turn_right_count, stuck_regions
            )
        else:
            return self._build_trajectory_opinion_fallback(
                context, turn_left_count, turn_right_count, stuck_regions
            )

    def _build_trajectory_opinion_with_llm(
        self,
        context: "NavContext",
        turn_left_count: int,
        turn_right_count: int,
        stuck_regions: List[Dict]
    ) -> "DebateOpinion":
        """Build trajectory opinion using LLM with enhanced spatial awareness."""
        from core.debate_types import DebateOpinion, ActionConstraint
        import math

        # Analyze trajectory data
        trajectory = context.trajectory if context.trajectory else []
        recent_actions = [a.action_type.name for a in context.action_history[-10:]] if context.action_history else []

        # Calculate path metrics
        total_distance = 0.0
        if len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                dx = trajectory[i][0] - trajectory[i-1][0]
                dz = trajectory[i][2] - trajectory[i-1][2]
                total_distance += math.sqrt(dx*dx + dz*dz)

        # Calculate efficiency
        efficiency = 1.0
        if len(trajectory) >= 2:
            start = trajectory[0]
            end = trajectory[-1]
            direct = math.sqrt((end[0]-start[0])**2 + (end[2]-start[2])**2)
            if total_distance > 0:
                efficiency = min(direct / total_distance, 1.0)

        # Height analysis
        y_trend = "稳定"
        y_change = 0.0
        if len(trajectory) >= 3:
            recent_y = [p[1] for p in trajectory[-5:]]
            y_change = recent_y[-1] - recent_y[0]
            if abs(y_change) > 0.2:
                y_trend = f"{'上升' if y_change > 0 else '下降'}{abs(y_change):.1f}米"

        # Goal direction analysis
        goal_pos = context.metadata.get("goal_position")
        current_pos = context.position if context.position else (0, 0, 0)

        goal_direction = "未知"
        goal_distance = 0.0
        goal_angle = 0.0
        floor_relation = "同层"

        if goal_pos:
            dx = goal_pos[0] - current_pos[0]
            dz = goal_pos[2] - current_pos[2]
            goal_distance = math.sqrt(dx*dx + dz*dz)

            # Angle to goal
            goal_angle = math.degrees(math.atan2(dx, -dz))
            if goal_angle < 0:
                goal_angle += 360

            # Direction name
            if goal_angle < 45 or goal_angle >= 315:
                goal_direction = "正前方"
            elif 45 <= goal_angle < 135:
                goal_direction = "右侧"
            elif 135 <= goal_angle < 225:
                goal_direction = "后方"
            else:
                goal_direction = "左侧"

            # Floor relation
            vert_dist = goal_pos[1] - current_pos[1]
            if vert_dist < -0.5:
                floor_relation = "目标在下层"
            elif vert_dist > 0.5:
                floor_relation = "目标在上层"

        # Detect looping pattern
        is_looping = False
        loop_pattern = "无"
        if len(recent_actions) >= 6:
            # Check for alternating left-right pattern
            pattern_str = "".join(["L" if "LEFT" in a else "R" if "RIGHT" in a else "F" for a in recent_actions[-6:]])
            if "LRLR" in pattern_str or "RLRL" in pattern_str:
                is_looping = True
                loop_pattern = "左右摇摆"
            elif recent_actions.count("TURN_LEFT") >= 4:
                is_looping = True
                loop_pattern = "连续左转"
            elif recent_actions.count("TURN_RIGHT") >= 4:
                is_looping = True
                loop_pattern = "连续右转"

        # Build enhanced prompt
        prompt = f"""你是轨迹规划专家。基于导航轨迹分析最佳动作。

## 轨迹状态
- 总步数: {context.step_count}
- 已走距离: {total_distance:.1f}米
- 路径效率: {efficiency:.2f}

## 空间关系
- 目标方向: {goal_direction}
- 目标角度: {goal_angle:.0f}度
- 目标距离: {goal_distance:.1f}米
- 楼层关系: {floor_relation}

## 高度分析
- 当前高度: {current_pos[1]:.2f}米
- 高度变化: {y_trend}
- 总高度差: {y_change:+.2f}米

## 动作历史分析
- 最近10动作: {recent_actions if recent_actions else "无"}
- 左转次数: {turn_left_count}
- 右转次数: {turn_right_count}
- 循环模式: {loop_pattern}
- 是否原地打转: {"是" if is_looping else "否"}

## 卡住检测
- 卡住步数: {context.stuck_counter if hasattr(context, 'stuck_counter') else 0}
- 已知卡住区域: {len(stuck_regions)}个

## 决策推理要求
1. 如果原地打转(左右摇摆)，必须选择新方向
2. 如果目标在不同楼层，应寻找楼梯
3. 如果效率低于0.3，说明路径曲折，应重新规划

## 输出格式(JSON)
{{
  "primary_action": "forward/turn_left/turn_right/stop",
  "confidence": 0.0-1.0,
  "reasoning": "推荐理由",
  "path_quality": {{
    "efficiency": {efficiency:.2f},
    "stuck_risk": {min(context.stuck_counter if hasattr(context, 'stuck_counter') else 0, 10)}/10,
    "recommendation": "继续当前方向/转向探索/寻找楼梯"
  }},
  "constraints": {{
    "hard": [{{"action": "应避免的动作", "blocked": true, "reason": "原因"}}],
    "soft": [{{"action": "优先动作", "weight": 0.5, "reason": "原因"}}]
  }}
}}

只输出JSON。"""
        try:
            response = self._model_manager.generate(
                "qwen-2b-trajectory",
                prompt,
                max_new_tokens=200,
                temperature=0.1,
            )
            return self._parse_trajectory_opinion_response(response, turn_left_count, turn_right_count, efficiency, is_looping)
        except Exception as e:
            self.logger.error(f"LLM trajectory opinion failed: {e}")
            return self._build_trajectory_opinion_fallback(
                context, turn_left_count, turn_right_count, stuck_regions
            )

    def _parse_trajectory_opinion_response(
        self,
        response: str,
        turn_left_count: int,
        turn_right_count: int,
        efficiency: float = 1.0,
        is_looping: bool = False
    ) -> "DebateOpinion":
        """Parse LLM response into DebateOpinion."""
        import json
        import re
        from core.debate_types import DebateOpinion, ActionConstraint

        primary_action = "turn_right"
        confidence = 0.6
        reasoning = ""
        constraints = {"hard": [], "soft": []}
        path_quality = {}

        try:
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                primary_action = data.get("primary_action", "turn_right")
                confidence = float(data.get("confidence", 0.6))
                reasoning = data.get("reasoning", "")
                path_quality = data.get("path_quality", {})

                # Parse hard constraints
                hard = data.get("constraints", {}).get("hard", [])
                for c in hard:
                    constraints["hard"].append(ActionConstraint(
                        action=c.get("action", "turn_right"),
                        blocked=c.get("blocked", True),
                        reason=c.get("reason", ""),
                    ))

                # Parse soft constraints
                soft = data.get("constraints", {}).get("soft", [])
                for c in soft:
                    constraints["soft"].append(ActionConstraint(
                        action=c.get("action", "turn_right"),
                        weight_multiplier=c.get("weight", 1.0),
                        reason=c.get("reason", ""),
                    ))

        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback logic based on trajectory analysis
        if not reasoning:
            if is_looping:
                reasoning = "检测到原地打转，建议选择新方向"
            elif efficiency < 0.3:
                reasoning = "路径效率低，建议重新规划"
            else:
                reasoning = "轨迹分析建议"

        return DebateOpinion(
            agent="trajectory",
            primary_action=primary_action,
            confidence=confidence,
            evidence={
                "recent_turns": {"left": turn_left_count, "right": turn_right_count},
                "efficiency": efficiency,
                "is_looping": is_looping,
                "path_quality": path_quality,
            },
            reasoning=reasoning,
            constraints=constraints,
        )

    def _build_trajectory_opinion_fallback(
        self,
        context: "NavContext",
        turn_left_count: int,
        turn_right_count: int,
        stuck_regions: List[Dict]
    ) -> "DebateOpinion":
        """Fallback rule-based trajectory opinion."""
        from core.debate_types import DebateOpinion, ActionConstraint

        trajectory = context.trajectory if context.trajectory else []
        forward_failures = 0
        if hasattr(context, 'stuck_counter'):
            forward_failures = context.stuck_counter

        # Determine primary action
        primary_action = "turn_right"
        confidence = 0.6

        if turn_right_count > turn_left_count + 2:
            primary_action = "turn_left"
            reasoning = f"已多次右转({turn_right_count})，建议左转"
        elif turn_left_count > turn_right_count + 2:
            primary_action = "turn_right"
            reasoning = f"已多次左转({turn_left_count})，建议右转"
        else:
            reasoning = "探索新方向"

        # Build constraints
        constraints = {"hard": [], "soft": []}

        # Penalize overused directions
        if turn_right_count > 3:
            constraints["soft"].append(ActionConstraint(
                action="turn_right",
                weight_multiplier=0.5,
                reason=f"already_tried_{turn_right_count}_times",
            ))
        if turn_left_count > 3:
            constraints["soft"].append(ActionConstraint(
                action="turn_left",
                weight_multiplier=0.5,
                reason=f"already_tried_{turn_left_count}_times",
            ))

        # Check for known stuck regions
        if trajectory:
            current_pos = trajectory[-1]
            stuck_info = self.get_stuck_region_info(current_pos)
            if stuck_info and stuck_info.get("successful_direction"):
                success_dir = stuck_info["successful_direction"]
                constraints["soft"].append(ActionConstraint(
                    action=success_dir,
                    weight_multiplier=1.3,
                    reason="known_escape_direction",
                ))

        return DebateOpinion(
            agent="trajectory",
            primary_action=primary_action,
            confidence=confidence,
            evidence={
                "recent_turns": {"left": turn_left_count, "right": turn_right_count},
                "consecutive_forward_failures": forward_failures,
                "visited_cells": len(self._visited_cells) if hasattr(self, '_visited_cells') else len(trajectory),
                "stuck_regions": len(self._stuck_regions) if hasattr(self, '_stuck_regions') else len(stuck_regions),
            },
            reasoning=reasoning,
            constraints=constraints,
        )