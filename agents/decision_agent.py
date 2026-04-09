"""Decision Agent for generating navigation action sequences.

This version uses Qwen3.5-9B-AWQ (via remote LLM server) for:
1. adopt-step action sequence generation
2. Subtask completion judgment
3. Reasoning generation
4. Emergency response for dynamic obstacles
"""

from typing import Dict, Any, Optional, List
import logging
import json
import re
import math

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext
from core.action import Action, ActionType


class DecisionAgent(BaseAgent):
    """
    Agent responsible for generating navigation action sequences.

    Uses Qwen3.5-9B-AWQ (remote) for sequence generation.

    Key responsibilities:
    1. Generate adopt-step action sequences based on strategy results
    2. Judge subtask completion via LLM reasoning
    3. Handle stuck detection for sequence abortion
    4. Emergency response for dynamic obstacles (Phase 2a)
    """

    # Action mapping
    ACTION_MAP = {
        "forward": ActionType.MOVE_FORWARD,
        "move_forward": ActionType.MOVE_FORWARD,
        "turn_left": ActionType.TURN_LEFT,
        "left": ActionType.TURN_LEFT,
        "turn_right": ActionType.TURN_RIGHT,
        "right": ActionType.TURN_RIGHT,
        "stop": ActionType.STOP,
        "look_up": ActionType.LOOK_UP,
        "look_down": ActionType.LOOK_DOWN,
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("DecisionAgent")

        # Stuck detection parameters
        self._last_position = None
        self._stuck_counter = 0
        self._stuck_threshold = 5

        # Sequence length configuration
        self.sequence_length = self.config.get("sequence_length", 5)
        self.adaptive_sequence = self.config.get("adaptive_sequence", False)
        self.min_sequence_length = self.config.get("min_sequence_length", 2)
        self.max_sequence_length = self.config.get("max_sequence_length", 20)

        # Model reference
        self._model_manager = None
        self._initialized = False

        # === SPATIAL MEMORY ===
        # Track visited positions to avoid circling
        self._visited_positions: List[tuple] = []  # List of (x, z) tuples
        self._stair_detections: List[Dict] = []  # Remember where stairs were found
        self._last_stair_direction: Optional[str] = None  # Track consistent stair direction
        self._exploration_directions: List[str] = []  # Track which directions explored

        # === EMERGENCY RESPONSE (Phase 2a) ===
        self._path_replanner = None  # PathReplanner instance
        self._emergency_mode = False  # Current emergency state

        # === LoRA CONFIGURATION ===
        self._use_lora = True  # Default: use LoRA for decision tasks
        self._lora_name = "decision-lora"  # LoRA adapter name

    @property
    def name(self) -> str:
        return "decision_agent"

    def set_use_lora(self, use_lora: bool):
        """Enable or disable LoRA for decision tasks."""
        self._use_lora = use_lora
        self.logger.info(f"LoRA {'enabled' if use_lora else 'disabled'} for DecisionAgent")

    @property
    def role(self) -> AgentRole:
        return AgentRole.DECISION

    def get_required_inputs(self) -> List[str]:
        return ["context"]

    def get_output_keys(self) -> List[str]:
        return ["action", "confidence", "reasoning", "subtask_completed"]

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Process navigation decision (delegates to generate_action_sequence)."""
        self.initialize()

        # This method is kept for BaseAgent compatibility
        # Actual navigation uses generate_action_sequence
        current_subtask = context.get_current_subtask()
        if current_subtask:
            sequence = self.generate_action_sequence(context, strategy_result, current_subtask)
            return AgentOutput.success_output(
                data={
                    "sequence": sequence,
                    "reasoning": sequence.reasoning,
                    "subtask_completed": sequence.subtask_completed,
                },
                confidence=sequence.confidence,
                reasoning=sequence.reasoning,
            )

        return AgentOutput.success_output(
            data={"action": "forward"},
            confidence=0.5,
            reasoning="Default forward",
        )

    def initialize(self) -> None:
        """Initialize model manager."""
        if self._initialized:
            return

        try:
            # First, try to use model_manager from config (passed by experiment)
            if self.config.get("model_manager"):
                self._model_manager = self.config["model_manager"]
                self._initialized = True
                self.logger.info("DecisionAgent: using provided model_manager (remote mode)")
                return

            # Fallback: create new model manager (for standalone usage)
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)
            self._model_manager.load_all_models()
            self._initialized = True
            self.logger.info("DecisionAgent: created new model_manager (local mode)")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model manager: {e}")
            self._initialized = True

    def generate_action_sequence(
        self,
        context: NavContext,
        strategy_result,
        subtask: 'SubTask'
    ):
        """Generate 10-step action sequence based on strategy result.

        Args:
            context: Navigation context
            strategy_result: Strategy execution result (StrategyResult)
            subtask: Current subtask

        Returns:
            ActionSequence with actions, reasoning, and subtask_completed
        """
        from core.action import ActionSequence, ActionType

        self.initialize()

        # === EMERGENCY RESPONSE (Phase 2a) ===
        # Check for emergency signal from EmergencyDetector
        emergency_signal = context.metadata.get("emergency_signal", {})
        if emergency_signal.get("trigger", False):
            self.logger.warning(f"[Decision] Emergency triggered: {emergency_signal.get('event_type')}")
            return self._emergency_response(context, emergency_signal, subtask)

        # Get strategy data
        strategy_data = strategy_result.metadata if strategy_result else {}

        # ===== NEW: Update spatial memory =====
        perception_output = context.metadata.get("perception_output", {}) if context else {}
        current_position = context.position if context else None
        self.update_spatial_memory(current_position, perception_output)

        # Get subtask level
        level = subtask.level if subtask and hasattr(subtask, 'level') else "medium"

        # ===== NEW: Auto-check completion condition before LLM call =====
        auto_completed = False
        completion_condition = subtask.completion_condition if subtask else None
        if completion_condition:
            trajectory = strategy_data.get("trajectory", {})
            subtask_delta = trajectory.get("subtask_delta", {})
            pos_delta = subtask_delta.get("position_delta", {})
            rot_delta = subtask_delta.get("rotation_delta", {})
            perception = strategy_data.get("perception", {})
            visible_objects = perception.get("objects", [])

            _, auto_completed = self._format_completion_check(
                completion_condition, pos_delta, rot_delta, visible_objects
            )

            if auto_completed:
                self.logger.info(f"[Decision] AUTO-COMPLETED: Condition '{completion_condition.get('type')}' satisfied")

        # Build prompt based on difficulty
        prompt = self._build_sequence_prompt_v2(context, subtask, strategy_data, level)

        actions = []
        reasoning = ""
        subtask_completed = False

        try:
            # Call LLM to generate sequence
            # Use LoRA adapter for decision tasks (fine-tuned for navigation decisions)
            if self._model_manager:
                # Determine LoRA usage based on configuration
                lora_name = self._lora_name if self._use_lora else None

                response = self._model_manager.generate(
                    "qwen-9b-decision",
                    prompt,
                    max_new_tokens=2000,
                    temperature=0.01,
                    lora_name=lora_name,
                )
                if response:
                    self.logger.info(f"[Decision] LLM response : {response[:500]}...")
                    self.logger.info(f"[Decision] LLM response length: {len(response)} chars")
                    actions, reasoning, subtask_completed = self._parse_sequence_response(response)
                    self.logger.info(f"[Decision] Parsed: {len(actions)} steps, completed:{subtask_completed}")
                else:
                    self.logger.error("[Decision] LLM returned empty response")
            else:
                raise RuntimeError("Model manager not initialized")

        except Exception as e:
            raise RuntimeError(f"[SEQUENCE] LLM generation failed: {e}")

        # ===== NEW: Override with auto-detected completion =====
        if auto_completed and not subtask_completed:
            self.logger.info(f"[Decision] Overriding LLM completion: False -> True (auto-detected)")
            subtask_completed = True

        # Determine minimum actions based on adaptive mode
        if self.adaptive_sequence:
            min_actions = self.min_sequence_length
        else:
            min_actions = max(3, self.sequence_length - 2)

        if len(actions) < min_actions:
            raise RuntimeError(f"[SEQUENCE] Parse failed, got {len(actions)} actions, need at least {min_actions}")

        self.logger.info(f"[Decision] Level:{level}, {len(actions)} steps, completed:{subtask_completed}")

        # Print to console
        action_names = [a[0].name if isinstance(a, tuple) else str(a) for a in actions]
        print(f"\n[DecisionAgent] Generated {len(actions)} actions: {action_names[:5]}{'...' if len(actions) > 5 else ''}")
        print(f"[DecisionAgent] Reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")

        subtask_id = subtask.id if subtask and hasattr(subtask, 'id') else 0

        return ActionSequence(
            subtask_id=subtask_id,
            subtask_description=subtask.description if subtask else "navigation",
            actions=actions,
            estimated_steps=len(actions),
            abort_conditions={"stuck_for_steps": 5},
            reasoning=reasoning,
            confidence=0.8,
            subtask_completed=subtask_completed
        )

    def check_sequence_abort(
        self,
        context: NavContext,
        sequence: 'ActionSequence',
        depth_image,
        current_action: str = None
    ) -> tuple:
        """Check if current sequence should be aborted (stuck detection only).

        Only triggers stuck detection when executing forward actions.
        Turning in place (turn_left/turn_right) should not trigger stuck.
        """
        from core.action import ActionType

        # Only check stuck when moving forward, not when turning
        is_forward_action = False
        if current_action:
            is_forward_action = current_action in ["forward", "move_forward"]

        if is_forward_action:
            self._check_position_stuck(context.position)
        else:
            # Reset stuck counter when turning (position unchanged is expected)
            self._stuck_counter = 0

        if self._stuck_counter >= self._stuck_threshold:
            return True, f"Stuck for {self._stuck_counter} steps"

        return False, ""

    # ========== Emergency Response Methods (Phase 2a) ==========

    def set_path_replanner(self, path_replanner) -> None:
        """Set PathReplanner instance for emergency replanning.

        Args:
            path_replanner: PathReplanner instance from emergency module
        """
        self._path_replanner = path_replanner
        self.logger.info("[Decision] PathReplanner set for emergency response")

    def _emergency_response(
        self,
        context: NavContext,
        emergency_signal: Dict[str, Any],
        subtask: 'SubTask'
    ):
        """Handle emergency situation with fast path replanning.

        Args:
            context: Navigation context
            emergency_signal: Emergency signal from EmergencyDetector
            subtask: Current subtask

        Returns:
            ActionSequence with emergency response actions
        """
        from core.action import ActionSequence

        self._emergency_mode = True
        event_type = emergency_signal.get("event_type", "unknown")
        event_position = emergency_signal.get("position")

        self.logger.warning(f"[Decision] Emergency response for: {event_type}")
        print(f"\n[DecisionAgent] EMERGENCY MODE: {event_type}")

        actions = []
        reasoning = f"Emergency response: {event_type}"

        # Check if PathReplanner is available
        if self._path_replanner and event_position:
            try:
                # Get goal position
                goal_position = context.metadata.get("goal_position")
                current_position = tuple(context.position) if context.position else (0, 0, 0)

                # Get obstacle state
                obstacle_state = context.metadata.get("obstacle_state", {})
                blocked_positions = obstacle_state.get("blocked_positions", [])
                blocked_areas = obstacle_state.get("blocked_areas", [])

                # Run path replanning
                result = self._path_replanner.replan(
                    current_pos=current_position,
                    goal_pos=goal_position,
                    blocked_positions=blocked_positions,
                    blocked_areas=blocked_areas
                )

                if result.success and result.primary_path:
                    # Convert path to actions
                    path_actions = self._path_replanner.path_to_actions(
                        result.primary_path,
                        current_rotation=context.rotation if context.rotation else 0
                    )

                    # Convert to action tuples
                    for action_name in path_actions[:10]:  # Limit to 10 actions
                        if action_name.lower() in self.ACTION_MAP:
                            actions.append((self.ACTION_MAP[action_name.lower()], 1))

                    reasoning = f"Emergency replan: found {len(result.primary_path)}-point path"
                    self.logger.info(f"[Decision] Emergency replan success: {len(actions)} actions")

            except Exception as e:
                self.logger.error(f"[Decision] Emergency replan failed: {e}")

        # Fallback: Simple escape actions if replanning failed
        if not actions:
            actions = self._generate_escape_actions(context, event_type)
            reasoning = f"Emergency fallback: escape actions for {event_type}"

        # Ensure minimum actions
        if len(actions) < 3:
            # Add default forward movement
            while len(actions) < 3:
                actions.append((ActionType.MOVE_FORWARD, 1))

        self._emergency_mode = False

        return ActionSequence(
            actions=actions,
            reasoning=reasoning,
            subtask_completed=False,  # Don't mark as completed during emergency
            confidence=0.7
        )

    def _generate_escape_actions(
        self,
        context: NavContext,
        event_type: str
    ) -> List[tuple]:
        """Generate simple escape actions for emergency.

        Args:
            context: Navigation context
            event_type: Type of emergency

        Returns:
            List of action tuples
        """
        actions = []

        if event_type == "obstacle_blocked":
            # Turn and explore
            actions = [
                (ActionType.TURN_LEFT, 1),
                (ActionType.TURN_LEFT, 1),
                (ActionType.MOVE_FORWARD, 1),
                (ActionType.MOVE_FORWARD, 1),
                (ActionType.MOVE_FORWARD, 1),
            ]

        elif event_type == "stuck":
            # Try to escape by turning
            actions = [
                (ActionType.TURN_RIGHT, 1),
                (ActionType.TURN_RIGHT, 1),
                (ActionType.MOVE_FORWARD, 1),
                (ActionType.MOVE_FORWARD, 1),
            ]

        else:
            # Default: forward movement
            actions = [
                (ActionType.MOVE_FORWARD, 1),
                (ActionType.MOVE_FORWARD, 1),
                (ActionType.MOVE_FORWARD, 1),
            ]

        return actions

    def reset_stuck_counter(self) -> None:
        """Reset stuck detection counter for new episode."""
        self._last_position = None
        self._stuck_counter = 0

    def reset_spatial_memory(self) -> None:
        """Reset spatial memory for new episode."""
        self._visited_positions = []
        self._stair_detections = []
        self._last_stair_direction = None
        self._exploration_directions = []
        self.logger.info("[Decision] Spatial memory reset for new episode")

    def update_spatial_memory(self, position: List[float], perception_output: Dict = None) -> None:
        """Update spatial memory with current position and perception.

        Args:
            position: Current [x, y, z] position
            perception_output: Latest perception output with stair info
        """
        if position and len(position) >= 3:
            # Round to 0.5m grid for visited positions
            x, z = round(position[0] * 2) / 2, round(position[2] * 2) / 2
            pos_key = (x, z)

            if pos_key not in self._visited_positions:
                self._visited_positions.append(pos_key)
                # Keep last 50 positions
                if len(self._visited_positions) > 50:
                    self._visited_positions.pop(0)

        # Track stair detections
        if perception_output:
            stair_entrance = perception_output.get("stair_entrance", {})
            stairs = perception_output.get("stairs", {})

            if stair_entrance.get("found"):
                detection = {
                    "direction": stair_entrance.get("direction", "unknown"),
                    "stair_direction": stair_entrance.get("stair_direction", "unknown"),
                    "distance": stair_entrance.get("distance", 0),
                    "position": position[:3] if position else None,
                }
                self._stair_detections.append(detection)
                # Keep last 10 detections
                if len(self._stair_detections) > 10:
                    self._stair_detections.pop(0)

                # Track most recent stair direction
                self._last_stair_direction = detection["stair_direction"]
                self.logger.info(f"[Decision] Remembered stair: {detection}")

    def get_spatial_memory_guidance(self) -> str:
        """Get guidance from spatial memory.

        Returns:
            String with spatial memory guidance for the prompt
        """
        guidance_parts = []

        # Report visited area count
        if self._visited_positions:
            guidance_parts.append(f"Visited {len(self._visited_positions)} unique positions.")

        # Report remembered stairs
        if self._stair_detections:
            recent = self._stair_detections[-1]
            stair_info = f"Last detected: {recent['stair_direction']} stairs {recent['direction']}"
            if recent['distance'] > 0:
                stair_info += f" ({recent['distance']:.1f}m away)"
            guidance_parts.append(stair_info)

        # Consistency warning
        if len(self._stair_detections) >= 3:
            directions = [d["direction"] for d in self._stair_detections[-3:]]
            if len(set(directions)) >= 3:
                guidance_parts.append("WARNING: Stair direction keeps changing - you may be circling!")

        return " ".join(guidance_parts) if guidance_parts else ""

    # ========== Prompt Building ==========

    def _build_sequence_prompt_v2(
        self,
        context: NavContext,
        subtask: 'SubTask',
        strategy_data: Dict[str, Any],
        level: str
    ) -> str:
        """Build prompt based on subtask difficulty."""
        import math

        # Basic info
        distance_to_goal = context.get_distance_to_goal() if hasattr(context, 'get_distance_to_goal') else 5.0

        # Get sequence length from config
        seq_len = getattr(self, 'sequence_length', 5)

        # Extract from strategy data
        perception = strategy_data.get("perception", {})
        trajectory = strategy_data.get("trajectory", {})
        instruction = strategy_data.get("instruction", {})
        trajectory_agent = strategy_data.get("trajectory_agent", None)

        # Perception info
        room_type = perception.get("room_type", "unknown")
        objects_raw = perception.get("objects", [])
        objects = [o.get("object", o.get("name", str(o))) for o in objects_raw[:3]]
        scene_desc = perception.get("scene_description", "")[:80]
        walkable = perception.get("walkable_analysis", {})
        obstacle = perception.get("obstacle_ahead", {})
        nav_hint = perception.get("nav_hint", "")

        # Open directions
        open_dirs = []
        if isinstance(walkable, dict):
            if walkable.get("left", {}).get("clear", True):
                open_dirs.append("left")
            if walkable.get("center", {}).get("clear", True):
                open_dirs.append("front")
            if walkable.get("right", {}).get("clear", True):
                open_dirs.append("right")
        open_dirs_str = "/".join(open_dirs) if open_dirs else "unknown"

        # Obstacle
        if isinstance(obstacle, dict):
            blocked = obstacle.get("blocked", False)
            min_dist = obstacle.get("min_distance", 5.0)
        elif isinstance(obstacle, bool):
            blocked = obstacle
            min_dist = 5.0
        else:
            blocked = False
            min_dist = 5.0

        # Trajectory info
        dist_traveled = trajectory.get("distance_traveled", 0)
        heading = trajectory.get("heading", "unknown")

        # ===== NEW: Extract structured state data =====
        subtask_delta = trajectory.get("subtask_delta", {})
        navigation = trajectory.get("navigation", {})
        current_state = trajectory.get("state", {})

        # Position delta for prompt
        pos_delta = subtask_delta.get("position_delta", {})
        rot_delta = subtask_delta.get("rotation_delta", {})

        # Instruction semantics
        directions = instruction.get("directions", [])
        instruction_analysis = instruction.get("instruction_analysis", {})
        landmarks = instruction_analysis.get("landmarks", []) if instruction_analysis else instruction.get("landmarks", [])
        goals = instruction_analysis.get("goals", []) if instruction_analysis else instruction.get("goals", [])

        # Completion condition
        completion_condition = subtask.completion_condition if subtask else None

        # Get action history summary from TrajectoryAgent
        action_history_summary = ""
        if trajectory_agent and hasattr(trajectory_agent, 'get_history_summary'):
            current_pos = context.position
            if current_pos:
                action_history_summary = trajectory_agent.get_history_summary(current_pos)

        # ===== NEW: Get spatial memory guidance =====
        spatial_memory_guidance = self.get_spatial_memory_guidance()

        # Build prompt based on level
        if level == "easy":
            return self._build_simple_prompt_direct_v2(seq_len,
                context, subtask, blocked, min_dist, dist_traveled,
                heading, distance_to_goal, directions, nav_hint,
                landmarks, goals, completion_condition, action_history_summary,
                pos_delta, rot_delta, navigation,  # state change data
                spatial_memory_guidance  # NEW: spatial memory
            )
        elif level == "medium":
            return self._build_medium_prompt_with_analysis_v2(
                seq_len, subtask, room_type, objects, scene_desc, open_dirs_str,
                blocked, min_dist, dist_traveled, heading, distance_to_goal,
                strategy_data.get("analysis", ""),
                directions, nav_hint, landmarks, goals, completion_condition,
                action_history_summary,
                pos_delta, rot_delta, navigation,  # state change data
                context,  # pass context for stair info
                spatial_memory_guidance  # NEW: spatial memory
            )
        else:  # hard
            return self._build_hard_prompt(
                subtask, room_type, objects, scene_desc, open_dirs_str,
                blocked, min_dist, dist_traveled, heading, distance_to_goal,
                strategy_data.get("opinions", {}),
                strategy_data.get("consensus", {}),
                directions, nav_hint, landmarks, goals, completion_condition,
                action_history_summary,
                pos_delta, rot_delta, navigation,  # state change data
                context,  # pass context for stair info
                spatial_memory_guidance  # NEW: spatial memory
            )

    def _build_simple_prompt_direct_v2(
        self, seq_len: int, context, subtask, blocked, min_dist, dist_traveled,
        heading, distance_to_goal, directions, nav_hint, landmarks,
        goals, completion_condition=None, action_history_summary="",
        pos_delta=None, rot_delta=None, navigation=None, spatial_memory_guidance=""
    ) -> str:
        """Easy task prompt: directly synthesize agent info without strategy analysis."""
        # Get perception info directly from context
        perception_output = context.metadata.get("perception_output", {})
        room_type = perception_output.get("room_type", "unknown")
        objects_raw = perception_output.get("objects", [])
        objects = [o.get("object", o.get("name", str(o))) for o in objects_raw[:3]]
        scene_desc = perception_output.get("scene_description", "")[:80]

        # Get goal direction info
        angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
        direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

        # Get obstacle info
        obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}

        # Build obstacle section string
        obstacle_section = ""
        if obstacle_info:
            obstacle_section = f"""
- 障碍物: {obstacle_info.get('direction', '')} {obstacle_info.get('distance', 0):.1f}m, 半径 {obstacle_info.get('radius', 1.0):.1f}m
- 绕行建议: 向{obstacle_info.get('bypass_direction', '右')}绕行"""

        # NEW: Get stair entrance info
        stair_entrance = perception_output.get("stair_entrance", {})
        stairs = perception_output.get("stairs", {})

        directions_str = "/".join(directions) if directions else "unknown"
        landmarks_str = ", ".join(landmarks[:3]) if landmarks else "none"
        goals_str = ", ".join(goals[:2]) if goals else "none"

        # Get visible objects for near_object check
        visible_objects = context.metadata.get("perception_output", {}).get("objects", [])

        # Format completion condition with auto-check
        completion_check_str, auto_completed = self._format_completion_check(
            completion_condition, pos_delta, rot_delta, visible_objects
        )

        # NEW: Build stair entrance guidance with direction info
        stair_guidance = ""
        required_steps = seq_len  # Default to configured sequence length

        if stair_entrance.get("found"):
            entrance_dir = stair_entrance.get("direction", "unknown")
            entrance_dist = stair_entrance.get("distance", 0)
            entrance_angle = stair_entrance.get("angle", 0)
            stair_dir = stair_entrance.get("stair_direction", "unknown")
            action_hint = stair_entrance.get("action_hint", "")

            # Calculate required steps based on distance
            # Each forward step ~0.25m, plus turning steps
            if entrance_dist > 0 and stair_dir == "down":  # Only for correct direction
                # Steps needed: distance * 4 (for forward) + turns
                turn_steps = abs(entrance_angle) // 15 + 2  # Rough turn estimate
                forward_steps = int(entrance_dist * 4)
                required_steps = min(15, max(self.min_sequence_length, turn_steps + forward_steps))
                self.logger.info(f"[Decision] Stairs at {entrance_dist:.1f}m, generating {required_steps} steps")

            # Add direction-specific guidance
            direction_note = ""
            if stair_dir == "down":
                direction_note = "\n**DIRECTION MATCH**: These stairs go DOWN - correct for this subtask!"
                direction_note += f"\n**CRITICAL**: Generate {required_steps} steps to reach stairs {entrance_dist:.1f}m away!"
            elif stair_dir == "up":
                direction_note = "\n**WARNING**: These stairs go UP - need DOWN stairs for this subtask!"
            elif stair_dir == "unknown":
                direction_note = "\n**NOTE**: Stair direction unclear - verify before proceeding."

            stair_guidance = f"""
## CRITICAL: Stair Entrance Located!
- Direction: {entrance_dir}
- Distance: {entrance_dist:.1f}m
- Angle from center: {entrance_angle}°
- Stair Direction: {stair_dir}
- Action hint: {action_hint}{direction_note}
**PRIORITY**: Navigate toward this stair entrance if direction matches subtask!
"""
        elif stairs.get("detected"):
            stair_dir = stairs.get('direction', 'unknown')
            direction_note = ""
            if stair_dir == "down":
                direction_note = " (DOWNWARD - correct direction)"
            elif stair_dir == "up":
                direction_note = " (UPWARD - wrong direction for 'down' subtask)"
            stair_guidance = f"""
## Stairs Detected (but entrance not precisely located)
- Direction: {stair_dir}{direction_note}
- Relative position: {stairs.get('relative_pos', 'unknown')}
- Distance: {stairs.get('distance', 'unknown')}m
**PRIORITY**: Find the stair entrance by exploring in the indicated direction.
"""

        # Generate example actions based on adaptive mode
        if self.adaptive_sequence:
            example_actions = '{"action":"forward"}, {"action":"turn_right"}, {"action":"forward"}'
        else:
            example_actions = ', '.join(['{"action":"forward"}' if i % 2 == 0 else '{"action":"turn_right"}' for i in range(seq_len)])

        # Build action history section
        history_section = ""
        if action_history_summary:
            history_section = f"\n## Action History Reference (TrajectoryAgent)\n{action_history_summary}\n"

        # ===== NEW: Build subtask state change section =====
        state_change_section = ""
        if pos_delta:
            start_pos = pos_delta.get("start", [0, 0, 0])
            curr_pos = pos_delta.get("current", [0, 0, 0])
            dx = pos_delta.get("dx", 0)
            dy = pos_delta.get("dy", 0)
            dz = pos_delta.get("dz", 0)
            h_dist = pos_delta.get("horizontal_distance", 0)

            rot_start = rot_delta.get("start_deg", 0) if rot_delta else 0
            rot_curr = rot_delta.get("current_deg", 0) if rot_delta else 0
            rot_change = rot_delta.get("delta_deg", 0) if rot_delta else 0
            rot_dir = rot_delta.get("direction", "none") if rot_delta else "none"

            dist_to_goal = navigation.get("distance_to_goal", 0) if navigation else 0

            state_change_section = f"""
## Subtask State Change (from subtask start)
- Position: {start_pos} → {curr_pos}
- Delta: dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m
- Horizontal Distance Moved: {h_dist:.2f}m
- Rotation: {rot_start:.0f}° → {rot_curr:.0f}° (change: {rot_change:.0f}°, {rot_dir})
- Distance to Goal: {dist_to_goal:.2f}m

{completion_check_str}
"""

        # Determine step constraint based on adaptive mode and stair distance
        if required_steps > seq_len:
            # Override when stairs detected far away
            seq_constraint = f"Generate {required_steps} steps to reach detected stairs."
            rule_1 = f"Must output exactly {required_steps} steps to reach stairs"
        elif self.adaptive_sequence:
            seq_constraint = f"""Based on current situation, output {self.min_sequence_length}-{self.max_sequence_length} steps.

**Step Selection Rules**:
- 3-5 steps: Target is ahead and direction is correct, just go straight
- 6-10 steps: Need small turn (45-90 degrees) then move forward
- 11-15 steps: Need large turn (180 degrees) or complex exploration

**Current Scene Analysis**: Determine which case applies based on environment"""
            rule_1 = f"Output {self.min_sequence_length}-{self.max_sequence_length} steps based on actual situation"
        else:
            seq_constraint = f"Generate {seq_len} step action sequence"
            rule_1 = f"Must output exactly {seq_len} steps"

        return f"""/no_think
You are a navigation decision system. Synthesize information from all agents and {seq_constraint}.

## Subtask
{subtask.description}
{state_change_section}
## Instruction Semantics
- Key directions: {directions_str}
- Target locations: {goals_str}
- Reference objects: {landmarks_str}

## Environment Analysis (PerceptionAgent)
- Room: {room_type}
- Visible objects: {objects if objects else "none"}
- Scene: {scene_desc if scene_desc else "no description"}
- Navigation hint: {nav_hint if nav_hint else "none"}
{stair_guidance}
## Spatial Memory
{spatial_memory_guidance if spatial_memory_guidance else "First exploration in this area."}

## Navigation State (TrajectoryAgent)
- Heading: {heading}
- Distance traveled: {dist_traveled:.1f}m
- Distance to goal: {distance_to_goal:.1f}m
- Goal direction: 目标在{direction_hint} (相对角度: {angle_to_goal:.0f}°){obstacle_section}{history_section}
## Action Types
- forward: move forward one step
- turn_left: turn left 15 degrees
- turn_right: turn right 15 degrees
- stop: stop

## Rules
1. {rule_1}
2. Must include turning actions
3. If COMPLETION CHECK shows "COMPLETED", you MUST set subtask_completed=true

## Output Format (JSON)
{{"reasoning":"brief reasoning in 1-2 sentences","subtask_completed":false,"actions":[{example_actions}]}}

IMPORTANT: Keep reasoning brief (1-2 sentences). Output JSON directly:"""

    def _build_medium_prompt_with_analysis_v2(
        self, seq_len: int, subtask, room_type, objects, scene_desc, open_dirs_str,
        blocked, min_dist, dist_traveled, heading, distance_to_goal,
        analysis, directions, nav_hint, landmarks, goals, completion_condition=None,
        action_history_summary="",
        pos_delta=None, rot_delta=None, navigation=None, context=None, spatial_memory_guidance=""
    ) -> str:
        """Medium task prompt: use CoT strategy analysis."""
        directions_str = "/".join(directions) if directions else "unknown"
        landmarks_str = ", ".join(landmarks[:3]) if landmarks else "none"
        goals_str = ", ".join(goals[:2]) if goals else "none"

        # Get goal direction info
        angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
        direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

        # Get obstacle info
        obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}

        # Build obstacle section string
        obstacle_section = ""
        if obstacle_info:
            obstacle_section = f"""
- 障碍物: {obstacle_info.get('direction', '')} {obstacle_info.get('distance', 0):.1f}m, 半径 {obstacle_info.get('radius', 1.0):.1f}m
- 绕行建议: 向{obstacle_info.get('bypass_direction', '右')}绕行"""

        # Format completion condition with auto-check
        visible_objects = [{"name": o} if isinstance(o, str) else o for o in (objects or [])]
        completion_check_str, auto_completed = self._format_completion_check(
            completion_condition, pos_delta, rot_delta, visible_objects
        )

        # NEW: Build stair entrance guidance from context with direction info
        stair_guidance = ""
        if context:
            perception_output = context.metadata.get("perception_output", {})
            stair_entrance = perception_output.get("stair_entrance", {})
            stairs = perception_output.get("stairs", {})
            if stair_entrance.get("found"):
                entrance_dir = stair_entrance.get("direction", "unknown")
                entrance_dist = stair_entrance.get("distance", 0)
                entrance_angle = stair_entrance.get("angle", 0)
                stair_dir = stair_entrance.get("stair_direction", "unknown")
                action_hint = stair_entrance.get("action_hint", "")

                direction_note = ""
                if stair_dir == "down":
                    direction_note = "\n**DIRECTION MATCH**: These stairs go DOWN - correct for this subtask!"
                elif stair_dir == "up":
                    direction_note = "\n**WARNING**: These stairs go UP - need DOWN stairs for this subtask!"

                stair_guidance = f"""
## CRITICAL: Stair Entrance Located!
- Direction: {entrance_dir}
- Distance: {entrance_dist:.1f}m
- Angle from center: {entrance_angle}°
- Stair Direction: {stair_dir}
- Action hint: {action_hint}{direction_note}
**PRIORITY**: Navigate toward this stair entrance if direction matches subtask!
"""
            elif stairs.get("detected"):
                stair_dir = stairs.get('direction', 'unknown')
                direction_note = ""
                if stair_dir == "down":
                    direction_note = " (DOWNWARD - correct direction)"
                elif stair_dir == "up":
                    direction_note = " (UPWARD - wrong direction for 'down' subtask)"
                stair_guidance = f"""
## Stairs Detected (but entrance not precisely located)
- Direction: {stair_dir}{direction_note}
- Relative position: {stairs.get('relative_pos', 'unknown')}
- Distance: {stairs.get('distance', 'unknown')}m
**PRIORITY**: Find the stair entrance by exploring in the indicated direction.
"""

        # Generate example actions based on adaptive mode
        if self.adaptive_sequence:
            example_actions = '{"action":"forward"}, {"action":"turn_right"}, {"action":"forward"}'
        else:
            example_actions = ', '.join(['{"action":"forward"}' if i % 2 == 0 else '{"action":"turn_right"}' for i in range(seq_len)])

        # Build action history section
        history_section = ""
        if action_history_summary:
            history_section = f"\n## Action History Reference\n{action_history_summary}\n"

        # ===== NEW: Build subtask state change section =====
        state_change_section = ""
        if pos_delta:
            start_pos = pos_delta.get("start", [0, 0, 0])
            curr_pos = pos_delta.get("current", [0, 0, 0])
            dx = pos_delta.get("dx", 0)
            dy = pos_delta.get("dy", 0)
            dz = pos_delta.get("dz", 0)
            h_dist = pos_delta.get("horizontal_distance", 0)

            rot_start = rot_delta.get("start_deg", 0) if rot_delta else 0
            rot_curr = rot_delta.get("current_deg", 0) if rot_delta else 0
            rot_change = rot_delta.get("delta_deg", 0) if rot_delta else 0
            rot_dir = rot_delta.get("direction", "none") if rot_delta else "none"

            dist_to_goal = navigation.get("distance_to_goal", 0) if navigation else 0

            state_change_section = f"""
## Subtask State Change (from subtask start)
- Position: {start_pos} → {curr_pos}
- Delta: dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m
- Horizontal Distance Moved: {h_dist:.2f}m
- Rotation: {rot_start:.0f}° → {rot_curr:.0f}° (change: {rot_change:.0f}°, {rot_dir})
- Distance to Goal: {dist_to_goal:.2f}m

{completion_check_str}
"""

        # Determine step constraint based on adaptive mode
        if self.adaptive_sequence:
            seq_constraint = f"""Based on current situation, output {self.min_sequence_length}-{self.max_sequence_length} steps

**Step Selection Rules**:
- 3-5 steps: Target is ahead and direction is correct, just go straight
- 6-10 steps: Need small turn (45-90 degrees) then move forward
"""
            seq_note = f"Determine which case applies based on strategy analysis"
            rule_1 = f"Output {self.min_sequence_length}-{self.max_sequence_length} steps based on actual needs"
        else:
            seq_constraint = f"Based on analysis result, generate {seq_len} step action sequence"
            seq_note = f"Action sequence length must be exactly {seq_len} steps"
            rule_1 = f"Must output exactly {seq_len} steps"

        return f"""/no_think
You are a navigation decision system. {seq_constraint}.

## Subtask
{subtask.description}
{state_change_section}
## Instruction Semantics
- Key directions: {directions_str}
- Target locations: {goals_str}
- Reference objects: {landmarks_str}

## Environment Analysis
- Room: {room_type}
- Visible objects: {objects if objects else "none"}
- Scene: {scene_desc if scene_desc else "no description"}
- Open directions: {open_dirs_str}
- Obstacle ahead: {"yes (" + str(min_dist) + "m)" if blocked else "no"}
- Navigation hint: {nav_hint if nav_hint else "none"}
{stair_guidance}
## Spatial Memory
{spatial_memory_guidance if spatial_memory_guidance else "First exploration in this area."}

## Navigation State
- Heading: {heading}
- Distance traveled: {dist_traveled:.1f}m
- Distance to goal: {distance_to_goal:.1f}m
- Goal direction: 目标在{direction_hint} (相对角度: {angle_to_goal:.0f}°){obstacle_section}

## Strategy Analysis (CoT)
{analysis[:300] if analysis else "none"}{history_section}

## Action Planning Guidelines
Generate action sequence based on strategy analysis:

**Direction-aware action generation (soft guidance)**:
- If analysis mentions "forward", "straight", "no turn needed": mainly forward actions
- If analysis mentions "left", "left side": add appropriate turn_left actions, then forward to explore
- If analysis mentions "right", "right side": add appropriate turn_right actions, then forward to explore
- If analysis mentions "turn around", "backward", "large turn": need multiple turns

**Action combination suggestions** (non-mandatory):
- Small adjustment: 1-3 turns + forward or just turns
- Medium turn (~90 degrees): 4-8 turns + forward or just turns
- Large turn (~180 degrees): 8-12 turns

**Note**:
- Each turn = 15 degrees
- Flexibly adjust turn count based on natural language description in analysis
- {seq_note}

## Action Types
- forward: move forward one step
- turn_left: turn left 15 degrees
- turn_right: turn right 15 degrees
- stop: stop

## Rules
1. {rule_1}
2. Must include turning actions
3. If COMPLETION CHECK shows "COMPLETED", you MUST set subtask_completed=true
4. Do not output stop action until all subtasks are completed

## Output Format (JSON)
{{"reasoning":"brief reasoning","subtask_completed":false,"actions":[{{"action":"turn_right"}},{{"action":"turn_right"}},{{"action":"forward"}}]}}

**Output JSON only, no explanation!**"""


    def _build_hard_prompt(
        self, subtask, room_type, objects, scene_desc, open_dirs_str,
        blocked, min_dist, dist_traveled, heading, distance_to_goal,
        opinions, consensus, directions, nav_hint, landmarks, goals, completion_condition=None,
        action_history_summary="",
        pos_delta=None, rot_delta=None, navigation=None, context=None, spatial_memory_guidance=""
    ) -> str:
        """Hard task prompt with agent opinions."""
        opinions_str = ""
        if opinions:
            for agent_name, opinion in opinions.items():
                if isinstance(opinion, dict):
                    opinions_str += f"\n### {agent_name}\n"
                    opinions_str += f"- Recommendation: {opinion.get('primary_action', 'unknown')}\n"
                    opinions_str += f"- Reason: {opinion.get('reasoning', '')[:50]}\n"

        consensus_str = consensus.get("reasoning", "none") if consensus else "none"
        directions_str = "/".join(directions) if directions else "unknown"
        landmarks_str = ", ".join(landmarks[:3]) if landmarks else "none"
        goals_str = ", ".join(goals[:2]) if goals else "none"

        # Get goal direction info
        angle_to_goal = context.metadata.get("angle_to_goal", 0) if context else 0
        direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"

        # Get obstacle info
        obstacle_info = context.metadata.get("obstacle_info", {}) if context else {}

        # Build obstacle section string
        obstacle_section = ""
        if obstacle_info:
            obstacle_section = f"""
- 障碍物: {obstacle_info.get('direction', '')} {obstacle_info.get('distance', 0):.1f}m, 半径 {obstacle_info.get('radius', 1.0):.1f}m
- 绕行建议: 向{obstacle_info.get('bypass_direction', '右')}绕行"""

        # Format completion condition with auto-check
        visible_objects = [{"name": o} if isinstance(o, str) else o for o in (objects or [])]
        completion_check_str, auto_completed = self._format_completion_check(
            completion_condition, pos_delta, rot_delta, visible_objects
        )

        # NEW: Build stair entrance guidance from context with direction info
        stair_guidance = ""
        if context:
            perception_output = context.metadata.get("perception_output", {})
            stair_entrance = perception_output.get("stair_entrance", {})
            stairs = perception_output.get("stairs", {})
            if stair_entrance.get("found"):
                entrance_dir = stair_entrance.get("direction", "unknown")
                entrance_dist = stair_entrance.get("distance", 0)
                entrance_angle = stair_entrance.get("angle", 0)
                stair_dir = stair_entrance.get("stair_direction", "unknown")
                action_hint = stair_entrance.get("action_hint", "")

                direction_note = ""
                if stair_dir == "down":
                    direction_note = "\n**DIRECTION MATCH**: These stairs go DOWN - correct for this subtask!"
                elif stair_dir == "up":
                    direction_note = "\n**WARNING**: These stairs go UP - need DOWN stairs for this subtask!"

                stair_guidance = f"""
## CRITICAL: Stair Entrance Located!
- Direction: {entrance_dir}
- Distance: {entrance_dist:.1f}m
- Angle from center: {entrance_angle}°
- Stair Direction: {stair_dir}
- Action hint: {action_hint}{direction_note}
**PRIORITY**: Navigate toward this stair entrance if direction matches subtask!
"""
            elif stairs.get("detected"):
                stair_dir = stairs.get('direction', 'unknown')
                direction_note = ""
                if stair_dir == "down":
                    direction_note = " (DOWNWARD - correct direction)"
                elif stair_dir == "up":
                    direction_note = " (UPWARD - wrong direction for 'down' subtask)"
                stair_guidance = f"""
## Stairs Detected (but entrance not precisely located)
- Direction: {stair_dir}{direction_note}
- Relative position: {stairs.get('relative_pos', 'unknown')}
- Distance: {stairs.get('distance', 'unknown')}m
**PRIORITY**: Find the stair entrance by exploring in the indicated direction.
"""

        # Get sequence length from config
        seq_len = getattr(self, 'sequence_length', 5)
        # Generate example actions based on adaptive mode
        if self.adaptive_sequence:
            example_actions = '{"action":"forward"}, {"action":"turn_right"}, {"action":"forward"}'
        else:
            example_actions = ', '.join(['{"action":"forward"}' if i % 2 == 0 else '{"action":"turn_right"}' for i in range(seq_len)])

        # ===== NEW: Build subtask state change section =====
        state_change_section = ""
        if pos_delta:
            start_pos = pos_delta.get("start", [0, 0, 0])
            curr_pos = pos_delta.get("current", [0, 0, 0])
            dx = pos_delta.get("dx", 0)
            dy = pos_delta.get("dy", 0)
            dz = pos_delta.get("dz", 0)
            h_dist = pos_delta.get("horizontal_distance", 0)

            rot_start = rot_delta.get("start_deg", 0) if rot_delta else 0
            rot_curr = rot_delta.get("current_deg", 0) if rot_delta else 0
            rot_change = rot_delta.get("delta_deg", 0) if rot_delta else 0
            rot_dir = rot_delta.get("direction", "none") if rot_delta else "none"

            dist_to_goal = navigation.get("distance_to_goal", 0) if navigation else 0

            state_change_section = f"""
## Subtask State Change (from subtask start)
- Position: {start_pos} → {curr_pos}
- Delta: dx={dx:.2f}m, dy={dy:.2f}m, dz={dz:.2f}m
- Horizontal Distance Moved: {h_dist:.2f}m
- Rotation: {rot_start:.0f}° → {rot_curr:.0f}° (change: {rot_change:.0f}°, {rot_dir})
- Distance to Goal: {dist_to_goal:.2f}m

{completion_check_str}
"""

        # Determine step constraint based on adaptive mode
        if self.adaptive_sequence:
            seq_constraint = f"Based on all agent opinions, output {self.min_sequence_length}-{self.max_sequence_length} step action sequence"
            rule_1 = f"Output {self.min_sequence_length}-{self.max_sequence_length} steps"
        else:
            seq_constraint = f"Based on all agent opinions, generate {seq_len} step action sequence"
            rule_1 = f"Must output exactly {seq_len} steps"

        return f"""/no_think
You are a navigation decision system. {seq_constraint}.

## Subtask
{subtask.description}
{state_change_section}
## Instruction Semantics
- Key directions: {directions_str}
- Target locations: {goals_str}
- Reference objects: {landmarks_str}

## Environment Analysis
- Room: {room_type}
- Visible objects: {objects if objects else "none"}
- Scene: {scene_desc if scene_desc else "no description"}
- Open directions: {open_dirs_str}
- Obstacle ahead: {"yes(" + str(min_dist) + "m)" if blocked else "no"}
- Navigation hint: {nav_hint if nav_hint else "none"}
{stair_guidance}
## Spatial Memory
{spatial_memory_guidance if spatial_memory_guidance else "First exploration in this area."}

## Navigation State
- Heading: {heading}
- Distance traveled: {dist_traveled:.1f}m
- Distance to goal: {distance_to_goal:.1f}m
- Goal direction: 目标在{direction_hint} (相对角度: {angle_to_goal:.0f}°){obstacle_section}

## Agent Opinions
{opinions_str if opinions_str else "none"}

## Consensus
{consensus_str[:200]}

## Action Types
- forward: move forward one step
- turn_left: turn left 15 degrees
- turn_right: turn right 15 degrees
- stop: stop

## Rules
1. {rule_1}
2. Make decision based on all agent opinions
3. Must include turning actions
4. If COMPLETION CHECK shows "COMPLETED", you MUST set subtask_completed=true

## Output Format (JSON)
{{"reasoning":"brief reasoning in 1-2 sentences","subtask_completed":false,"actions":[{example_actions}]}}

IMPORTANT: Keep reasoning brief (1-2 sentences). Output JSON directly:"""

    # ========== Response Parsing ==========

    def _parse_sequence_response(self, response: str) -> tuple:
        """Parse LLM response to extract actions with multi-layer fallback.

        Returns:
            (actions, reasoning, subtask_completed)
        """
        actions = []
        reasoning = ""
        subtask_completed = False
        parse_success = False

        # Log raw response for debugging
        self.logger.info(f"[Decision] Raw response length: {len(response)} chars")

        # Strategy 1: Clean and extract JSON
        json_str = response.strip()

        # Handle markdown code blocks: ```json ... ``` or ``` ... ```
        if '```json' in json_str:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', json_str)
            if json_match:
                json_str = json_match.group(1).strip()
        elif '```' in json_str:
            lines = json_str.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            json_str = "\n".join(lines)

        # Strategy 2: Find outermost JSON object with balanced braces
        brace_count = 0
        start_idx = -1
        for i, c in enumerate(json_str):
            if c == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    json_candidate = json_str[start_idx:i+1]
                    try:
                        data = json.loads(json_candidate)
                        reasoning, subtask_completed = self._extract_actions_from_json(data, actions)
                        if len(actions) > 0:
                            parse_success = True
                            self.logger.info(f"[Decision] JSON parse SUCCESS: {len(actions)} actions")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"[Decision] JSON parse failed: {e}")
                        # Try partial JSON extraction
                        reasoning, subtask_completed, parse_success = self._try_partial_json_extraction(json_candidate, actions)
                    break

        # Strategy 3: Regex fallback if JSON parsing failed or no actions found
        if not parse_success or len(actions) == 0:
            self.logger.info("[Decision] Attempting regex fallback extraction...")
            reasoning, parse_success = self._regex_fallback_extract(response, actions)
            if parse_success:
                self.logger.info(f"[Decision] Regex fallback SUCCESS: {len(actions)} actions")

        # Strategy 4: Generate default actions if all parsing failed
        if len(actions) == 0:
            self.logger.warning("[Decision] All parsing failed, generating default actions")
            actions = self._generate_default_actions()
            reasoning = "Default action: forward movement"

        # Log final result
        self.logger.info(f"[Decision] Final parse result: {len(actions)} actions, reasoning_len={len(reasoning)}")

        return actions, reasoning, subtask_completed

    def _extract_actions_from_json(self, data: dict, actions: list) -> tuple:
        """Extract actions from parsed JSON data.

        Returns:
            (reasoning, subtask_completed) - extracted metadata
        """
        # Extract reasoning and completion status
        reasoning = data.get("reasoning", "")
        subtask_completed = data.get("subtask_completed", False)

        # Action name mapping for compatibility
        action_map = {
            "forward": ActionType.MOVE_FORWARD,
            "move_forward": ActionType.MOVE_FORWARD,
            "turn_left": ActionType.TURN_LEFT,
            "left": ActionType.TURN_LEFT,
            "turn_right": ActionType.TURN_RIGHT,
            "right": ActionType.TURN_RIGHT,
            "stop": ActionType.STOP,
        }

        # Extract actions from various formats
        for item in data.get("actions", []):
            if isinstance(item, str):
                action_name = item.lower()
            elif isinstance(item, dict):
                action_name = item.get("action", "forward").lower()
                # Also check for count/repeat field
                count = item.get("count", item.get("repeat", 1))
                if isinstance(count, int) and count > 0:
                    for _ in range(count):
                        if action_name in action_map:
                            actions.append((action_map[action_name], 1))
                    continue
            else:
                continue

            if action_name in action_map:
                actions.append((action_map[action_name], 1))

        return reasoning, subtask_completed

    def _try_partial_json_extraction(self, json_str: str, actions: list) -> tuple:
        """Try to extract partial information from malformed JSON.

        Returns:
            (reasoning, subtask_completed, success) - extracted metadata and success flag
        """
        reasoning = ""
        subtask_completed = False
        success = False

        # Try to extract reasoning
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', json_str)
        if reasoning_match:
            reasoning = reasoning_match.group(1)
            self.logger.info(f"[Decision] Partial extract: reasoning found")

        # Try to extract subtask_completed
        completed_match = re.search(r'"subtask_completed"\s*:\s*(true|false)', json_str, re.IGNORECASE)
        if completed_match:
            subtask_completed = completed_match.group(1).lower() == "true"

        # Try to extract actions array items individually
        # Pattern: "action": "forward" or "forward" as standalone
        action_patterns = [
            r'"action"\s*:\s*"(forward|move_forward|turn_left|turn_right|left|right|stop)"',
            r'"(forward|move_forward|turn_left|turn_right|left|right|stop)"\s*,?',
        ]

        action_map = {
            "forward": ActionType.MOVE_FORWARD,
            "move_forward": ActionType.MOVE_FORWARD,
            "turn_left": ActionType.TURN_LEFT,
            "left": ActionType.TURN_LEFT,
            "turn_right": ActionType.TURN_RIGHT,
            "right": ActionType.TURN_RIGHT,
            "stop": ActionType.STOP,
        }

        for pattern in action_patterns:
            matches = re.findall(pattern, json_str, re.IGNORECASE)
            for match in matches:
                action_name = match.lower() if isinstance(match, str) else match[0].lower()
                if action_name in action_map:
                    actions.append((action_map[action_name], 1))
                    success = True

        if success:
            self.logger.info(f"[Decision] Partial extract: {len(actions)} actions")

        return reasoning, subtask_completed, success

    def _regex_fallback_extract(self, response: str, actions: list) -> tuple:
        """Regex fallback extraction for actions when JSON parsing completely fails.

        Returns:
            (reasoning, success) - extracted reasoning and success flag
        """
        reasoning = ""
        success = False

        # Comprehensive action patterns
        action_patterns = [
            # JSON-style patterns
            r'"action"\s*:\s*"(forward|move_forward|turn_left|turn_right|left|right|stop)"',
            r'"(forward|move_forward|turn_left|turn_right|left|right|stop)"',
            # Natural language patterns
            r'(move\s+forward|go\s+forward|walk\s+forward)',
            r'(turn\s+left|rotate\s+left)',
            r'(turn\s+right|rotate\s+right)',
            r'(stop|wait)',
        ]

        action_map = {
            "forward": ActionType.MOVE_FORWARD,
            "move_forward": ActionType.MOVE_FORWARD,
            "move forward": ActionType.MOVE_FORWARD,
            "go forward": ActionType.MOVE_FORWARD,
            "walk forward": ActionType.MOVE_FORWARD,
            "turn_left": ActionType.TURN_LEFT,
            "turn left": ActionType.TURN_LEFT,
            "rotate left": ActionType.TURN_LEFT,
            "left": ActionType.TURN_LEFT,
            "turn_right": ActionType.TURN_RIGHT,
            "turn right": ActionType.TURN_RIGHT,
            "rotate right": ActionType.TURN_RIGHT,
            "right": ActionType.TURN_RIGHT,
            "stop": ActionType.STOP,
            "wait": ActionType.STOP,
        }

        for pattern in action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                action_name = match.lower() if isinstance(match, str) else match[0].lower()
                # Handle multi-word matches
                action_name = action_name.replace(" ", "_")
                if action_name in action_map:
                    actions.append((action_map[action_name], 1))
                    success = True

        # Try to extract reasoning from natural text
        reasoning_match = re.search(r'(reasoning|analysis|plan)\s*:?\s*["\']?([^"\']+)["\']?', response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(2)[:100]

        return reasoning, success

    def _generate_default_actions(self) -> list:
        """Generate default actions when all parsing strategies failed.

        Returns:
            List of default actions (typically forward movement)
        """
        # Default: move forward to continue exploration
        return [(ActionType.MOVE_FORWARD, 1)]

    # ========== Completion Check ==========

    def _format_completion_check(
        self,
        completion_condition: dict,
        pos_delta: dict = None,
        rot_delta: dict = None,
        visible_objects: list = None
    ) -> tuple:
        """Format completion condition with current state and auto-check if satisfied.

        Returns:
            (formatted_str, is_completed): Formatted string and boolean indicating completion
        """
        if not completion_condition:
            return "No completion condition specified", False

        cc_type = completion_condition.get("type", "unknown")
        direction = completion_condition.get("direction", "")
        min_change = completion_condition.get("min_change", completion_condition.get("min_degrees", completion_condition.get("min_meters", 0)))
        target_object = completion_condition.get("object", "")
        description = completion_condition.get("description", "")

        # Extract current state values
        dy = pos_delta.get("dy", 0) if pos_delta else 0
        dx = pos_delta.get("dx", 0) if pos_delta else 0
        dz = pos_delta.get("dz", 0) if pos_delta else 0
        horizontal_dist = pos_delta.get("horizontal_distance", 0) if pos_delta else 0
        rot_change = abs(rot_delta.get("delta_deg", 0)) if rot_delta else 0

        is_completed = False
        current_value = 0
        threshold = min_change
        comparison = ""

        if cc_type == "y_change":
            current_value = abs(dy)
            threshold = min_change

            if direction == "down":
                comparison = f"|dy| = {current_value:.2f}m {'>=' if current_value >= threshold else '<'} {threshold}m (min_change)"
                is_completed = current_value >= threshold
            elif direction == "up":
                # For up, dy should be positive (increasing Y)
                current_value = dy
                comparison = f"dy = {current_value:.2f}m {'>=' if current_value >= threshold else '<'} {threshold}m (min_change)"
                is_completed = current_value >= threshold
            else:
                # Any vertical change
                comparison = f"|dy| = {current_value:.2f}m {'>=' if current_value >= threshold else '<'} {threshold}m (min_change)"
                is_completed = current_value >= threshold

        elif cc_type == "rotation":
            current_value = rot_change
            threshold = min_change
            rot_dir = rot_delta.get("direction", "none") if rot_delta else "none"

            # Check direction match
            dir_match = True
            if direction == "right" and rot_dir != "right":
                dir_match = False
            elif direction == "left" and rot_dir != "left":
                dir_match = False

            comparison = f"rotation = {current_value:.0f}° ({rot_dir}) {'>=' if current_value >= threshold else '<'} {threshold}°"
            is_completed = current_value >= threshold and dir_match

        elif cc_type == "distance":
            current_value = horizontal_dist
            threshold = min_change
            comparison = f"distance = {current_value:.2f}m {'>=' if current_value >= threshold else '<'} {threshold}m"
            is_completed = current_value >= threshold

        elif cc_type == "near_object" or cc_type == "object_near":
            target_lower = target_object.lower()
            visible_objs = [o.get("name", str(o)).lower() for o in (visible_objects or [])]
            found = any(target_lower in v or v in target_lower for v in visible_objs)
            current_value = 1 if found else 0
            threshold = 1
            comparison = f"object '{target_object}' {'FOUND' if found else 'NOT FOUND'} in visible objects"
            is_completed = found

        # Build formatted string
        status = "✅ COMPLETED" if is_completed else "⏳ IN PROGRESS"
        formatted = f"""## COMPLETION CHECK [{status}]
- Type: {cc_type}
- Description: {description}
- Required: {threshold} ({"downward" if direction == "down" else "upward" if direction == "up" else direction})
- Current: {comparison}

**VERDICT**: {"Condition SATISFIED - set subtask_completed=true" if is_completed else "Condition NOT YET satisfied - continue navigation"}
"""
        return formatted, is_completed

    # ========== Stuck Detection ==========

    def _check_position_stuck(self, position: tuple) -> bool:
        """Check if agent is stuck (position unchanged)."""
        if position is None:
            return False

        if self._last_position is None:
            self._last_position = position
            self._stuck_counter = 0
            return False

        # Calculate distance moved
        dx = position[0] - self._last_position[0]
        dz = position[2] - self._last_position[2]
        distance_moved = math.sqrt(dx*dx + dz*dz)

        self._last_position = position

        # If moved less than 0.1m, increment counter
        if distance_moved < 0.1:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0

        return self._stuck_counter >= self._stuck_threshold