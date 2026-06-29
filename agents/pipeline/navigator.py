"""Navigator - Main Agent orchestrator and human-machine interface.

This module provides the main navigation orchestrator that:
- Coordinates SubAgent execution sequence
- Handles emergency situations
- Receives user commands
- Reports navigation progress

Navigation flow:
    1. Receive user instruction
    2. Emergency detection (EmergencyAgent)
    3. ObservationAgent.process (VLM)
    4. AnalysisAgent.process (LLM reasoning)
    5. PlanningAgent.process (LLM planning)
    6. Execute actions (Habitat environment)
    7. ReviewAgent.process (rule verification)
    8. ActionConverter.convert + ensure_5_actions
    9. Execute final actions
    10. TopologyGraph.update
    11. Progress report (periodic)
    Loop until: ReviewAgent judges complete or max_steps reached
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentRole, AgentOutput
from agents.pipeline.base_pipeline_agent import (
    ObservationOutput,
    AnalysisOutput,
    PlanningOutput,
    ReviewOutput,
    EmergencyEvent,
)
from agents.pipeline.observation_agent import ObservationAgent
from agents.pipeline.analysis_agent import AnalysisAgent
from agents.pipeline.planning_agent import PlanningAgent
from agents.pipeline.review_agent import ReviewAgent
from agents.pipeline.emergency_agent import EmergencyAgent
from agents.pipeline.subtask_decomposition_agent import SubtaskDecompositionAgent
from agents.pipeline.tools.subagent_registry import SubAgentRegistry
from agents.pipeline.tools.topology_graph import TopologyGraph
from agents.pipeline.tools.state_calculator import StateCalculator
from agents.pipeline.tools.action_converter import ActionConverter
from core.action import ActionType


class Navigator(BaseAgent):
    """Main Agent - Orchestrator and human-machine interface.

    Responsibilities:
    - Coordination: Execute SubAgent flow in sequence, manage navigation loop
    - Emergency response: Detect triggers, call EmergencyAgent
    - Human-machine interaction: Receive user commands, handle emergency evacuation
    - Status reporting: Periodically report position, progress, exceptions

    LLM role:
    - Auxiliary coordination: LLM determines if flow adjustment needed in complex situations
    - Core understanding: Parse user emergency command semantics
    """

    name = "navigator"

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Navigator.

        Args:
            config: Agent configuration dictionary
                - model_configs: dict of agent_name → model_key mappings
        """
        super().__init__(config)
        self.logger = logging.getLogger("Navigator")

        # Core components
        self._registry = SubAgentRegistry()
        self._topology = TopologyGraph()
        self._state_calculator = StateCalculator()
        self._action_converter = ActionConverter()
        self._model_manager = None

        # Navigation state
        self._current_subtask = None
        self._subtasks = []
        self._current_subtask_index: int = 0
        self._history = []
        self._position = None
        self._rotation = None
        self._step_count = 0
        self._instruction = ""

        # === Difficulty grading state ===
        self._static_difficulty: str = "medium"
        self._dynamic_difficulty: str = "easy"
        self._difficulty_factors: Dict[str, Any] = {}
        # Dynamic difficulty tracking
        self._consecutive_low_confidence: int = 0
        self._last_positions: List[Tuple[float, float, float]] = []
        self._prev_analysis_output: Optional[Dict] = None

        # Last agent outputs (for logging/reporting)
        self._last_observation_output = None
        self._last_analysis_output = None
        self._last_planning_output = None
        self._last_emergency_event = None

        # Configuration
        self._max_steps = self.config.get("max_steps", 100)
        self._report_interval = self.config.get("report_interval", 10)

        # Agent model key configuration
        self._agent_model_configs = self.config.get("model_configs", {
            "decomposition": "qwen3.5-9b-fast",
            "observation": "qwen3-vl-8b",
            "analysis": "qwen3.5-9b-fast",      # CoT default
            "analysis_strong": "qwen3.6-35b-strong",  # Debate/Reflection
            "planning": "qwen3.6-35b-strong",
            "review": "qwen3.5-9b-fast",
            "emergency": "qwen3.5-9b-fast",
        })

    def set_model_manager(self, model_manager) -> None:
        """Set ModelManager for LLM/VLM access.

        Args:
            model_manager: ModelManager instance
        """
        self._model_manager = model_manager

        # Set model_manager for all registered SubAgents
        for agent in self._registry.list_all().values():
            if hasattr(agent, "set_model_manager"):
                agent.set_model_manager(model_manager)

    def register_subagents(self) -> None:
        """Register all SubAgents with their model_key configurations.

        Model allocation:
        - observation: Qwen3-VL-8B (dedicated VLM)
        - decomposition: Qwen3.5-9B (fast, one-time)
        - analysis: Qwen3.6-35B (strong, for CoT/Debate/Reflection)
        - planning: Qwen3.6-35B (strong, multi-source fusion)
        - review: Qwen3.5-9B (fast, rule-first)
        - emergency: Qwen3.5-9B (fast, low-latency)
        """
        mc = self._agent_model_configs

        self._registry.register("decomposition", SubtaskDecompositionAgent(
            config={"model_key": mc["decomposition"]}
        ))
        self._registry.register("observation", ObservationAgent(
            config={"model_key": mc["observation"]}
        ))
        self._registry.register("analysis", AnalysisAgent(
            config={
                "model_key": mc["analysis"],
                "strong_model_key": mc["analysis_strong"],
            }
        ))
        self._registry.register("planning", PlanningAgent(
            config={"model_key": mc["planning"]}
        ))
        self._registry.register("review", ReviewAgent(
            config={"model_key": mc["review"]}
        ))
        self._registry.register("emergency", EmergencyAgent(
            config={"model_key": mc["emergency"]}
        ))

        # Set model_manager for newly registered agents
        if self._model_manager is not None:
            for agent in self._registry.list_all().values():
                if hasattr(agent, "set_model_manager"):
                    agent.set_model_manager(self._model_manager)

    def _decompose_instruction(
        self,
        instruction: str,
        goal_position: List[float],
    ) -> List[Dict[str, Any]]:
        """Decompose instruction into subtasks using LLM.

        Args:
            instruction: Navigation instruction text
            goal_position: Goal position [x, y, z]

        Returns:
            List of subtask dicts
        """
        if self._model_manager is None:
            self.logger.warning("[Navigator] No ModelManager, using fallback decomposition")
            return [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {
                        "type": "distance_to_goal",
                        "threshold": 3.0,
                        "goal_position": goal_position,
                    },
                }
            ]

        try:
            decomposition_output = self._registry.call(
                "decomposition",
                instruction=instruction,
                goal_position=goal_position,
                start_position=self._position,
            )

            # Save static difficulty from decomposition
            self._static_difficulty = getattr(decomposition_output, 'static_difficulty', 'medium')
            self._difficulty_factors = getattr(decomposition_output, 'difficulty_factors', {})

            self.logger.info(
                f"[Navigator] Decomposed into {len(decomposition_output.subtasks)} subtasks, "
                f"static_difficulty={self._static_difficulty}, "
                f"factors={self._difficulty_factors}"
            )
            for i, subtask in enumerate(decomposition_output.subtasks):
                self.logger.info(f"  Subtask {i+1}: {subtask['description']}")

            return decomposition_output.subtasks
        except Exception as e:
            self.logger.warning(f"[Navigator] Decomposition failed: {e}")
            return [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {
                        "type": "distance_to_goal",
                        "threshold": 3.0,
                        "goal_position": goal_position,
                    },
                }
            ]

    def initialize_episode(
        self,
        instruction: str,
        start_position: List[float],
        goal_position: List[float] = None,
    ) -> None:
        """Initialize episode with instruction and starting position.

        Args:
            instruction: Navigation instruction text
            start_position: Starting position [x, y, z]
            goal_position: Goal position [x, y, z] (optional, for decomposition)
        """
        self._position = start_position.copy() if isinstance(start_position, list) else list(start_position)
        self._rotation = 0.0
        self._history = [{"position": self._position, "rotation": self._rotation, "step": 0}]
        self._step_count = 0
        self._instruction = instruction  # NEW: save original instruction

        # Decompose instruction into subtasks
        if goal_position is not None:
            self._subtasks = self._decompose_instruction(instruction, goal_position)
        else:
            # Fallback if no goal_position
            self._subtasks = [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": None,
                }
            ]

        self._current_subtask_index = 0
        self._current_subtask = self._subtasks[0]

        # Clear topology for new episode
        self._topology = TopologyGraph()

        # Reset difficulty state
        self._dynamic_difficulty = "easy"
        self._consecutive_low_confidence = 0
        self._last_positions = []
        self._prev_analysis_output = None

        self.logger.info(
            f"[Navigator] Episode initialized: instruction='{instruction[:50]}...', "
            f"static_difficulty={self._static_difficulty}, "
            f"subtasks={len(self._subtasks)}"
        )

    def run_navigation_loop(self, env) -> Dict[str, Any]:
        """Run main navigation loop.

        Args:
            env: Habitat environment with get_observations, step, get_agent_position, get_agent_rotation

        Returns:
            Navigation result dict with success, steps, reason
        """
        self.logger.info(f"[Navigator] Starting navigation loop, max_steps={self._max_steps}")

        while self._step_count < self._max_steps:
            # Execute single navigation cycle
            actions = self._run_navigation_cycle(env)

            # Execute actions
            for action_type, repeat_count in actions:
                for _ in range(repeat_count):
                    env.step(action_type)
                    self._step_count += 1

                    # Update state after each action
                    self._update_state(env)

                    # Check completion
                    if self._check_completion():
                        self.logger.info(f"[Navigator] Task completed at step {self._step_count}")
                        return {
                            "success": True,
                            "steps": self._step_count,
                            "reason": "task_completed",
                        }

                    # Check max steps
                    if self._step_count >= self._max_steps:
                        break

            # Periodic progress report
            if self._step_count % self._report_interval == 0:
                self._report_progress()

        # Reached max steps without completion
        self.logger.info(f"[Navigator] Max steps reached: {self._step_count}")
        return {
            "success": False,
            "steps": self._step_count,
            "reason": "max_steps",
        }

    def _run_navigation_cycle(self, env) -> List[Tuple[ActionType, int]]:
        """Execute single navigation cycle.

        Flow:
        1. Get observations
        2. ObservationAgent.process (VLM)
        3. Emergency detection
        4. AnalysisAgent.process (LLM)
        5. PlanningAgent.process (LLM)
        6. ActionConverter.convert + ensure_5_actions

        Args:
            env: Habitat environment

        Returns:
            List of (ActionType, repeat_count) tuples (exactly 5 actions)
        """
        # 1. Get observations
        observations = env.get_observations()
        rgb = observations.get("rgb")
        depth = observations.get("depth")

        # Create mock subtask object if needed
        if self._current_subtask is None:
            self._current_subtask = {
                "id": 1,
                "description": "navigate",
                "completion_condition": None,
            }

        # Mock subtask object with proper attributes
        class SubtaskObj:
            def __init__(self, data):
                self.id = data.get("id", 1)
                self.description = data.get("description", "navigate")
                self.completion_condition = data.get("completion_condition")
                self.relevant_objects = data.get("relevant_objects", [])

        subtask_obj = SubtaskObj(self._current_subtask)

        # 2. ObservationAgent - VLM core
        try:
            observation_output = self._registry.call(
                "observation",
                subtask=subtask_obj,
                instruction=self._instruction,  # NEW: pass original instruction
                position=self._position,
                rotation=self._rotation,
                rgb_image=rgb,
                depth_image=depth,
            )
            self._last_observation_output = observation_output
        except Exception as e:
            self.logger.warning(f"[Navigator] ObservationAgent failed: {e}")
            # Fallback: default observation
            observation_output = ObservationOutput(
                subtask_relevant=False,  # renamed field
                instruction_relevant=False,  # new field
                fallback_mode=True,  # new field
                objects=[],
                exploration_hint="",  # new field
                target_direction="forward",
                target_distance="unknown",
                path_blocked=False,
                navigation_cues=[],
                scene_description="",
            )
            self._last_observation_output = observation_output

        # 2.5 Dynamic difficulty classification (rule-based, before AnalysisAgent)
        self._dynamic_difficulty = self._classify_dynamic_difficulty(
            observation_output, self._history
        )
        self.logger.info(
            f"[Navigator] Dynamic difficulty: {self._dynamic_difficulty} "
            f"(static={self._static_difficulty})"
        )

        # 2.6 Depth-based obstacle check (rule, no LLM)
        depth_blocked = self._check_depth_obstacle(depth)

        # 3. Emergency detection
        try:
            emergency = self._registry.call(
                "emergency",
                context={
                    "observation": observation_output,
                    "history": self._history[-5:] if len(self._history) > 5 else self._history,
                    "collision_status": False,
                    "position": self._position,
                    "topology": self._topology,
                    "depth_blocked": depth_blocked,
                },
            )
            self._last_emergency_event = emergency

            if emergency is not None:
                self.logger.warning(f"[Navigator] Emergency detected: {emergency.type}")
                return self._handle_emergency(emergency, env)
        except Exception as e:
            self.logger.warning(f"[Navigator] EmergencyAgent detection failed: {e}")
            self._last_emergency_event = None

        # 4. AnalysisAgent - LLM reasoning (with difficulty-graded strategy)
        try:
            analysis_output = self._registry.call(
                "analysis",
                observation=observation_output,
                subtask=subtask_obj,
                history=self._history,
                static_difficulty=self._static_difficulty,
                dynamic_difficulty=self._dynamic_difficulty,
            )
            self._last_analysis_output = analysis_output
            self._prev_analysis_output = {
                "recommended_action": analysis_output.recommended_action,
                "confidence": analysis_output.confidence,
            }
        except Exception as e:
            self.logger.warning(f"[Navigator] AnalysisAgent failed: {e}")
            analysis_output = AnalysisOutput(
                goal_summary="navigate forward",
                current_gap="unknown",
                recommended_action="forward",
                reasoning="fallback",
                confidence=0.5,
                strategy_used="cot",
            )
            self._last_analysis_output = analysis_output

        # 5. PlanningAgent - LLM planning
        try:
            planning_output = self._registry.call(
                "planning",
                analysis=analysis_output,
                topology=self._topology,
                position=self._position,
                observation=observation_output,  # pass observation for fallback context
            )
            self._last_planning_output = planning_output
        except Exception as e:
            self.logger.warning(f"[Navigator] PlanningAgent failed: {e}")
            # Fallback: default planning
            planning_output = PlanningOutput(
                actions=["forward"] * 5,
                expected_result="move forward",
                algorithm_used="llm",
            )
            self._last_planning_output = planning_output

        # 6. Convert actions
        actions = self._action_converter.convert(planning_output.actions)
        actions = self._action_converter.ensure_5_actions(actions)

        return actions

    def _handle_emergency(
        self,
        event: EmergencyEvent,
        env,
    ) -> List[Tuple[ActionType, int]]:
        """Handle emergency event.

        Args:
            event: EmergencyEvent
            env: Habitat environment

        Returns:
            List of (ActionType, repeat_count) tuples for emergency handling
        """
        self.logger.warning(f"[Navigator] Handling emergency: type={event.type}, severity={event.severity}")

        # Call EmergencyAgent.handle
        emergency_agent = self._registry.get("emergency")

        if emergency_agent is None:
            # Fallback: default obstacle bypass
            actions = self._action_converter.convert(["turn_left", "forward", "forward"])
            return actions

        try:
            emergency_actions = emergency_agent.handle(
                event=event,
                context={
                    "topology": self._topology,
                    "position": self._position,
                    "observation": None,  # Can pass observation_output if needed
                },
            )

            # Convert actions
            actions = self._action_converter.convert(emergency_actions)

            # Ensure at least 5 actions for consistency
            if len(actions) < 5:
                actions = self._action_converter.ensure_5_actions(actions)

            return actions
        except Exception as e:
            self.logger.error(f"[Navigator] Emergency handling failed: {e}")
            # Fallback: turn and move
            actions = self._action_converter.convert(["turn_left", "forward", "forward", "forward", "forward"])
            return actions

    def _handle_emergency_command(self, command: Dict[str, Any]) -> None:
        """Handle user emergency command.

        Args:
            command: User command dict with type and content
        """
        self.logger.warning(f"[Navigator] Emergency command received: {command}")

        # Create emergency event
        event = EmergencyEvent(
            type="evacuate",
            severity="high",
            details={"command": command, "source": "user"},
        )

        # Store for next cycle processing
        self._pending_emergency = event

    def _receive_user_command(self) -> Optional[Dict[str, Any]]:
        """Receive user command.

        In real implementation, this would interface with user input.
        For now, returns None (no command).

        Returns:
            User command dict or None
        """
        # TODO: Implement user command interface
        return None

    def _update_state(self, env) -> None:
        """Update navigation state after action execution.

        Args:
            env: Habitat environment
        """
        # Get new position and rotation
        self._position = env.get_agent_position()
        self._rotation = env.get_agent_rotation()

        # Record history
        self._history.append({
            "position": self._position,
            "rotation": self._rotation,
            "step": self._step_count,
        })

        # Update topology
        self._topology.add_visited_position(self._position)
        # Detect key nodes from last observation
        if self._last_observation_output:
            self._detect_key_nodes(self._last_observation_output, prev_pos=None)

    def _check_completion(self) -> bool:
        """Check if current subtask is completed.

        Uses ReviewAgent for verification.

        Returns:
            True if task completed, False otherwise
        """
        # No completion condition set
        if self._current_subtask is None:
            return False

        completion_condition = self._current_subtask.get("completion_condition")
        if completion_condition is None:
            return False

        # Need at least 2 history entries for state change
        if len(self._history) < 2:
            return False

        # Calculate state change
        start_pos = self._history[-2]["position"]
        start_rot = self._history[-2].get("rotation", 0.0)

        state_change = self._state_calculator.compute_position_change(
            start_pos, self._position
        )

        # Add rotation change
        rotation_change = self._state_calculator.compute_rotation_change(
            start_rot, self._rotation
        )
        state_change["rotation_change"] = rotation_change

        # Create subtask object for ReviewAgent
        class SubtaskObj:
            def __init__(self, data):
                self.id = data.get("id", 1)
                self.description = data.get("description", "")
                self.completion_condition = data.get("completion_condition")

        subtask_obj = SubtaskObj(self._current_subtask)

        # Call ReviewAgent
        try:
            review_output = self._registry.call(
                "review",
                subtask=subtask_obj,
                execution_result={},
                state_change=state_change,
            )

            if review_output.completed:
                self.logger.info(f"[Navigator] Subtask {self._current_subtask_index + 1} completed: {review_output.reason}")

                # Check if there's next subtask
                if self._current_subtask_index < len(self._subtasks) - 1:
                    self._current_subtask_index += 1
                    self._current_subtask = self._subtasks[self._current_subtask_index]
                    self.logger.info(f"[Navigator] Switching to subtask {self._current_subtask_index + 1}: {self._current_subtask['description']}")
                    return False  # Not fully complete, just switched
                else:
                    self.logger.info(f"[Navigator] All {len(self._subtasks)} subtasks completed!")
                    return True  # All subtasks complete

            return False
        except Exception as e:
            self.logger.warning(f"[Navigator] ReviewAgent check failed: {e}")
            return False

    def _classify_dynamic_difficulty(
        self,
        observation: ObservationOutput,
        history: List[dict],
    ) -> str:
        """Classify runtime dynamic difficulty based on current navigation state.

        Evaluates 5 triggers, each contributing a score:
        - stuck (score=3): <0.2m movement over last 4+ steps
        - blocked (score=3): path_blocked=True in observation
        - lost (score=2): target_direction unknown AND fallback_mode active
        - circling (score=2): >=3 similar positions in last 5 history entries
        - uncertain (score=1): >=3 consecutive low-confidence analysis outputs

        Returns:
            "easy" (score 0-1), "medium" (score 2-4), or "hard" (score 5+)
        """
        total_score = 0

        # Trigger 1: Stuck detection (position unchanged)
        pos = self._position
        if pos is not None:
            self._last_positions.append((pos[0], pos[1], pos[2]))
            if len(self._last_positions) > 5:
                self._last_positions = self._last_positions[-5:]

            if len(self._last_positions) >= 4:
                import math
                total_movement = 0.0
                for i in range(1, len(self._last_positions)):
                    dx = self._last_positions[i][0] - self._last_positions[i-1][0]
                    dz = self._last_positions[i][2] - self._last_positions[i-1][2]
                    total_movement += math.sqrt(dx*dx + dz*dz)
                if total_movement < 0.2 * (len(self._last_positions) - 1):
                    total_score += 3
                    self.logger.info(f"[DynamicDiff] Stuck detected: movement={total_movement:.2f}m")

        # Trigger 2: Path blocked
        if observation.path_blocked:
            total_score += 3
            self.logger.info("[DynamicDiff] Path blocked")

        # Trigger 3: Lost (target unknown + fallback mode)
        if observation.target_direction == "unknown" and observation.fallback_mode:
            total_score += 2
            self.logger.info("[DynamicDiff] Lost: target unknown in fallback mode")

        # Trigger 4: Circling (repeated positions)
        if len(self._last_positions) >= 5:
            unique = set((round(p[0], 1), round(p[2], 1)) for p in self._last_positions)
            if len(unique) <= 2:
                total_score += 2
                self.logger.info(f"[DynamicDiff] Circling: {len(unique)} unique positions in last 5")

        # Trigger 5: Consecutive low confidence
        if self._prev_analysis_output:
            conf = self._prev_analysis_output.get("confidence", 0.5)
            if conf < 0.5:
                self._consecutive_low_confidence += 1
            else:
                self._consecutive_low_confidence = 0
            if self._consecutive_low_confidence >= 3:
                total_score += 1
                self.logger.info(f"[DynamicDiff] Uncertain: {self._consecutive_low_confidence} consecutive low conf")

        # Classify
        if total_score >= 5:
            return "hard"
        elif total_score >= 2:
            return "medium"
        else:
            return "easy"

    def _check_depth_obstacle(self, depth_image) -> bool:
        """Rule-based depth obstacle detection.

        Checks if the central region of the depth image shows an obstacle
        closer than obstacle_threshold (default 1.5m).

        Args:
            depth_image: Depth image as numpy array (H, W)

        Returns:
            True if obstacle detected directly ahead
        """
        if depth_image is None:
            return False

        try:
            import numpy as np
            depth = np.array(depth_image)
            if depth.ndim == 3:
                depth = depth[:, :, 0]

            h, w = depth.shape[:2]
            # Check central 30% of image (directly ahead)
            center_h_start, center_h_end = int(h * 0.35), int(h * 0.65)
            center_w_start, center_w_end = int(w * 0.35), int(w * 0.65)

            center_region = depth[center_h_start:center_h_end,
                                  center_w_start:center_w_end]

            # Average depth in central region
            avg_depth = np.mean(center_region) if center_region.size > 0 else 999.0

            # Obstacle if average depth < 0.8m (only very close obstacles)
            # Increased from 1.5m to avoid false positives near walls
            obstacle_threshold = 0.8
            blocked = avg_depth < obstacle_threshold

            if blocked:
                self.logger.info(
                    f"[DepthObstacle] Central avg depth={avg_depth:.2f}m "
                    f"< threshold={obstacle_threshold}m"
                )
            return blocked
        except Exception as e:
            self.logger.debug(f"[DepthObstacle] Check failed: {e}")
            return False

    def _detect_key_nodes(
        self,
        observation_output,
        prev_pos: List[float],
    ) -> None:
        """Detect and register key topology nodes from observation.

        Called after each navigation cycle. Identifies:
        - turn_point: when agent changes direction significantly
        - stairs_entry: when stairs are detected in observation
        - room_entry: when room type changes
        - door: when door is detected

        Args:
            observation_output: ObservationOutput from ObservationAgent
            prev_pos: Previous position before this cycle
        """
        if not observation_output or not self._position:
            return

        # Detect stairs from observation
        scene_desc = getattr(observation_output, 'scene_description', '') or ''
        objects = getattr(observation_output, 'objects', []) or []
        cues = getattr(observation_output, 'navigation_cues', []) or []

        # Check for stairs
        stairs_keywords = ['stair', 'step', 'staircase', 'stairway']
        has_stairs = any(kw in scene_desc.lower() for kw in stairs_keywords)
        if not has_stairs and objects:
            has_stairs = any(
                any(kw in str(obj.get('name', '')).lower() for kw in stairs_keywords)
                for obj in objects if isinstance(obj, dict)
            )

        if has_stairs:
            from agents.pipeline.tools.topology_graph import NodeType
            self._topology.add_key_node(
                position=list(self._position),
                node_type=NodeType.STAIRS_ENTRY,
                rotation=self._rotation or 0.0,
                metadata={"source": "observation"},
            )
            self.logger.info(f"[Topology] Added STAIRS_ENTRY node at {self._position}")

        # Check for door
        door_keywords = ['door', 'doorway', 'entrance', 'archway']
        has_door = any(kw in scene_desc.lower() for kw in door_keywords)
        if not has_door and cues:
            has_door = any(
                any(kw in cue.lower() for kw in door_keywords)
                for cue in cues
            )

        if has_door:
            from agents.pipeline.tools.topology_graph import NodeType
            self._topology.add_key_node(
                position=list(self._position),
                node_type=NodeType.DOOR,
                rotation=self._rotation or 0.0,
                metadata={"source": "observation"},
            )
            self.logger.info(f"[Topology] Added DOOR node at {self._position}")

    def _report_progress(self) -> None:
        """Report current navigation progress."""
        self.logger.info(f"[Navigator] Progress: step={self._step_count}, position={self._position}")
        print(f"[导航进度] 步数: {self._step_count}, 位置: {self._position}")

    def get_status(self) -> Dict[str, Any]:
        """Get current navigation status.

        Returns:
            Status dict with position, step_count, history, topology
        """
        return {
            "position": self._position,
            "rotation": self._rotation,
            "step_count": self._step_count,
            "current_subtask": self._current_subtask,
            "history_length": len(self._history),
            "topology_nodes": len(self._topology.nodes),
            "visited_positions": len(self._topology.visited_positions),
        }

    def process(
        self,
        context=None,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Process navigation context (BaseAgent interface).

        This method satisfies the abstract method from BaseAgent.
        Navigator's main functionality is in run_navigation_loop.

        Args:
            context: Navigation context (optional, may contain env and instruction)
            strategy_result: Optional strategy result (not used by Navigator)

        Returns:
            AgentOutput with navigation result
        """
        # Navigator doesn't use the standard agent process pattern
        # Instead, use run_navigation_loop for actual navigation
        return AgentOutput.success_output(
            data={"status": self.get_status()},
            confidence=1.0,
            reasoning="Navigator orchestrates navigation via run_navigation_loop",
        )