"""Decision Agent for generating navigation action sequences.

This version uses Qwen3.5-9B-AWQ (via remote LLM server) for:
1. adopt-step action sequence generation
2. Subtask completion judgment
3. Reasoning generation
4. Emergency response for dynamic obstacles
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import re
import math

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext
from core.action import Action, ActionType, ActionSequence


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

    # Class constants
    MAX_STEPS = 150  # Maximum navigation steps

    # Action mapping
    # Action mapping - comprehensive for all parsing scenarios
    ACTION_MAP = {
        # Standard actions
        "forward": ActionType.MOVE_FORWARD,
        "move_forward": ActionType.MOVE_FORWARD,
        "turn_left": ActionType.TURN_LEFT,
        "left": ActionType.TURN_LEFT,
        "turn_right": ActionType.TURN_RIGHT,
        "right": ActionType.TURN_RIGHT,
        "stop": ActionType.STOP,
        "look_up": ActionType.LOOK_UP,
        "look_down": ActionType.LOOK_DOWN,
        # Natural language variants
        "move forward": ActionType.MOVE_FORWARD,
        "go forward": ActionType.MOVE_FORWARD,
        "walk forward": ActionType.MOVE_FORWARD,
        "turn left": ActionType.TURN_LEFT,
        "rotate left": ActionType.TURN_LEFT,
        "turn right": ActionType.TURN_RIGHT,
        "rotate right": ActionType.TURN_RIGHT,
        "wait": ActionType.STOP,
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

        # === EMERGENCY RESPONSE (Phase 2a) ===

    @property
    def name(self) -> str:
        return "decision_agent"

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
        self.initialize()

        # === EMERGENCY RESPONSE (Phase 2a) ===
        # Check for emergency signal from EmergencyDetector
        emergency_signal = context.metadata.get("emergency_signal", {})
        if emergency_signal.get("trigger", False):
            self.logger.warning(f"[Decision] Emergency triggered: {emergency_signal.get('event_type')}")
            return self._emergency_response(context, emergency_signal, subtask)

        # Get strategy data
        strategy_data = strategy_result.metadata if strategy_result else {}

        # FIX: For easy tasks (strategy skipped), populate strategy_data from context.metadata
        if not strategy_data:
            perception_output = context.metadata.get("perception_output", {})
            trajectory_output = context.metadata.get("trajectory_output", {})
            instruction_output = context.metadata.get("instruction_output", {})
            strategy_data = {
                "perception": perception_output,
                "trajectory": trajectory_output,
                "instruction": instruction_output,
            }

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

        # === NEW: Priority use structured suggestion from CoT ===
        # EXPERIMENTAL: Temporarily disable skip to test Direct VLM
        FORCE_LLM = True  # TODO: Set to False after testing
        if level == "medium" and strategy_data and not FORCE_LLM:
            suggestion = strategy_data.get("suggestion", {})
            if suggestion and suggestion.get("direction"):
                # Direct conversion - skip LLM call
                self.logger.info(f"[Decision] Using structured suggestion: direction={suggestion.get('direction')}")
                actions = self._convert_suggestion_to_actions(suggestion)

                reasoning = strategy_data.get("analysis", "")[:200] if strategy_data.get("analysis") else "CoT suggestion"
                subtask_completed_flag = strategy_data.get("subtask_completed", False)

                return ActionSequence(
                    subtask_id=subtask.id if subtask else 0,
                    subtask_description=subtask.description if subtask else "",
                    actions=actions,
                    estimated_steps=len(actions),
                    reasoning=f"Suggestion converted: {reasoning[:100]}",
                    confidence=0.8,
                    abort_conditions={"stuck_for_steps": 5},
                    subtask_completed=subtask_completed_flag,
                )

        # === Skip LLM for medium tasks when CoT analysis has clear actions ===
        if level == "medium" and strategy_data and strategy_data.get("analysis"):
            direct_actions = self._extract_actions_from_cot_analysis(strategy_data.get("analysis", ""))
            min_actions_for_medium = 3

            if len(direct_actions) >= min_actions_for_medium:
                self.logger.info(f"[Decision] 直接提取CoT动作: {len(direct_actions)}步，跳过LLM调用")
                reasoning_preview = strategy_data.get("analysis", "")[:200]

                return ActionSequence(
                    subtask_id=subtask.id if subtask else 0,
                    subtask_description=subtask.description if subtask else "",
                    actions=direct_actions,
                    estimated_steps=sum(a[1] for a in direct_actions),
                    reasoning=f"CoT直接提取: {reasoning_preview[:100]}",
                    confidence=0.7,
                    abort_conditions={"stuck_for_steps": 5},
                    subtask_completed=False,
                )
            else:
                self.logger.info(f"[Decision] CoT提取动作不足({len(direct_actions)}步)，继续LLM调用")

        # Build prompt based on difficulty
        base_prompt = self._build_prompt(context, subtask, strategy_data, level)

        actions = []
        reasoning = ""
        subtask_completed = False
        last_response = ""

        # 强制要求5个动作
        REQUIRED_ACTIONS = 5
        MAX_RETRIES = 2

        for attempt in range(MAX_RETRIES + 1):
            prompt = base_prompt
            if attempt > 0 and last_response:
                # 重试时添加失败反馈
                prompt += f"\n\n**上次输出解析失败，必须输出{REQUIRED_ACTIONS}个动作！**\n上次输出: {last_response[:300]}\n\n请重新输出完整JSON，actions数组必须包含{REQUIRED_ACTIONS}个动作对象。"

            try:
                if self._model_manager:
                    response = self._model_manager.generate(
                        "qwen-9b-decision",
                        prompt,
                        max_new_tokens=1500,
                        temperature=0.01,
                    )
                    if response:
                        last_response = response
                        self.logger.info(f"[Decision] LLM response (attempt {attempt+1}): {response[:500]}...")
                        self.logger.info(f"[Decision] LLM response length: {len(response)} chars")
                        actions, reasoning, subtask_completed = self._parse_sequence_response(response)
                        self.logger.info(f"[Decision] Parsed: {len(actions)} steps, completed:{subtask_completed}")

                        # 检查是否满足动作数量要求
                        if len(actions) >= REQUIRED_ACTIONS:
                            break  # 成功，跳出重试循环
                        else:
                            self.logger.warning(f"[Decision] Attempt {attempt+1}: got {len(actions)} actions, need {REQUIRED_ACTIONS}")
                    else:
                        self.logger.error("[Decision] LLM returned empty response")
                else:
                    raise RuntimeError("Model manager not initialized")

            except Exception as e:
                self.logger.error(f"[Decision] Attempt {attempt+1} error: {e}")
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"[SEQUENCE] LLM generation failed after {MAX_RETRIES+1} attempts: {e}")

        # 所有重试都失败
        if len(actions) < REQUIRED_ACTIONS:
            raise RuntimeError(
                f"[SEQUENCE] Parse failed after {MAX_RETRIES+1} attempts. "
                f"Got {len(actions)} actions, need {REQUIRED_ACTIONS}. "
                f"Last LLM output: {last_response[:500]}"
            )

        # ===== NEW: Dual verification with conservative strategy =====
        # 任一判断未完成 → 继续导航（保守策略）
        if auto_completed and subtask_completed:
            # 两者都完成才标记完成
            final_completed = True
            self.logger.info(f"[Decision] 双重验证通过: 自动={auto_completed}, LLM={subtask_completed}")
        elif auto_completed and not subtask_completed:
            # 自动检测完成但LLM未完成 → 继续执行（保守）
            final_completed = False
            self.logger.info(f"[Decision] 保守策略: 自动完成但LLM未确认")
        elif not auto_completed and subtask_completed:
            # LLM完成但自动检测未完成 → 继续执行（保守）
            final_completed = False
            self.logger.warning(f"[Decision] 保守策略: LLM声称完成但自动检测未通过")
        else:
            # 两方都未完成
            final_completed = False

        # Override subtask_completed with final verification result
        subtask_completed = final_completed

        # 动作数量已在重试循环中验证（必须>=5）
        self.logger.info(f"[Decision] Level:{level}, {len(actions)} steps, completed:{subtask_completed}")

        # Print to console
        action_names = [a[0].name if isinstance(a, tuple) else str(a) for a in actions]
        print(f"\n[DecisionAgent] Generated {len(actions)} actions: {action_names[:5]}{'...' if len(actions) > 5 else ''}")
        print(f"[DecisionAgent] Reasoning: {reasoning}")

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

    def _convert_suggestion_to_actions(self, suggestion: dict) -> List[tuple]:
        """将CoT建议转换为精确动作序列.

        Args:
            suggestion: CoT建议字典，格式如下:
                {
                    "direction": "left" | "right" | "forward",
                    "turn_count": {"min": 2, "max": 3},
                    "forward_count": {"min": 3, "max": 5}
                }

        Returns:
            List of (ActionType, count) tuples
        """
        direction = suggestion.get("direction", "forward")
        turn_data = suggestion.get("turn_count", {})
        forward_data = suggestion.get("forward_count", {})

        turn_min = turn_data.get("min", 0) if isinstance(turn_data, dict) else 0
        turn_max = turn_data.get("max", 0) if isinstance(turn_data, dict) else 0
        forward_min = forward_data.get("min", 1) if isinstance(forward_data, dict) else 1
        forward_max = forward_data.get("max", 5) if isinstance(forward_data, dict) else 5

        actions = []

        # 1. 转向动作
        if direction == "left":
            turn_count = max(2, (turn_min + turn_max) // 2)  # Middle value with minimum 2
            actions.extend([(ActionType.TURN_LEFT, 1)] * turn_count)
            self.logger.info(f"[Decision] Suggestion: turn_left {turn_count} times")
        elif direction == "right":
            turn_count = max(2, (turn_min + turn_max) // 2)  # Middle value with minimum 2
            actions.extend([(ActionType.TURN_RIGHT, 1)] * turn_count)
            self.logger.info(f"[Decision] Suggestion: turn_right {turn_count} times")

        # 2. 前进动作
        forward_count = max(3, (forward_min + forward_max) // 2)  # Middle value with minimum 3
        actions.extend([(ActionType.MOVE_FORWARD, 1)] * forward_count)
        self.logger.info(f"[Decision] Suggestion: forward {forward_count} times")

        return actions

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

    def update_spatial_memory(self, position: List[float], perception_output: str = None) -> None:
        """Update spatial memory with current position and perception.

        Args:
            position: Current [x, y, z] position
            perception_output: Natural language scene description (now a string)
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

        # 楼梯检测：从自然语言描述中简单判断
        if perception_output and isinstance(perception_output, str):
            desc_lower = perception_output.lower()
            if "stairs" in desc_lower or "stair" in desc_lower or "step" in desc_lower:
                # 检测到楼梯相关词汇
                if "up" in desc_lower or "ascend" in desc_lower:
                    stair_dir = "up"
                elif "down" in desc_lower or "descend" in desc_lower:
                    stair_dir = "down"
                else:
                    stair_dir = "unknown"

                detection = {
                    "direction": "detected",
                    "stair_direction": stair_dir,
                    "position": position[:3] if position else None,
                }
                self._stair_detections.append(detection)
                if len(self._stair_detections) > 10:
                    self._stair_detections.pop(0)
                self._last_stair_direction = stair_dir
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

    # ========== Coordinate Extraction ==========

    def _extract_position_info(self, context: NavContext) -> Dict[str, Any]:
        """Extract position and vertical direction info from context.

        Returns:
            Dict with: goal_x, goal_y, goal_z, curr_x, curr_y, curr_z,
                       vertical_diff, vertical_ok, vertical_direction, vertical_hint
        """
        goal_position = context.metadata.get("goal_position") if context else None
        current_position = context.position if context else None

        goal_x, goal_y, goal_z = 0.0, 0.0, 0.0
        curr_x, curr_y, curr_z = 0.0, 0.0, 0.0
        vertical_diff = 0.0
        vertical_ok = True
        vertical_direction = "位置未知"
        vertical_hint = ""

        if goal_position and len(goal_position) >= 3:
            goal_x, goal_y, goal_z = goal_position[0], goal_position[1], goal_position[2]
        if current_position and len(current_position) >= 3:
            curr_x, curr_y, curr_z = current_position[0], current_position[1], current_position[2]

        if goal_position and current_position:
            vertical_diff = curr_y - goal_y
            vertical_ok = abs(vertical_diff) < 1.0
            if vertical_diff > 0.1:
                vertical_direction = "当前在目标上方"
                vertical_hint = "需要向下走"
            elif vertical_diff < -0.1:
                vertical_direction = "当前在目标下方"
                vertical_hint = "需要向上走"
            else:
                vertical_direction = "与目标同高度"
                vertical_hint = ""

        return {
            "goal_x": goal_x, "goal_y": goal_y, "goal_z": goal_z,
            "curr_x": curr_x, "curr_y": curr_y, "curr_z": curr_z,
            "vertical_diff": vertical_diff,
            "vertical_ok": vertical_ok,
            "vertical_direction": vertical_direction,
            "vertical_hint": vertical_hint,
        }

    # ========== Prompt Building ==========

    def _build_prompt(
        self,
        context: NavContext,
        subtask: 'SubTask',
        strategy_data: Dict[str, Any],
        level: str
    ) -> str:
        """Unified prompt builder: template + level-specific sections."""
        # Extract all common data
        common = self._extract_prompt_data(context, subtask, strategy_data)

        # Level-specific section
        if level in ("easy", "medium"):
            analysis = strategy_data.get("analysis", "根据场景和目标距离规划路径")
            extra_section = f"## CoT分析\n{analysis}"
        else:  # hard
            opinions = strategy_data.get("opinions", {})
            consensus = strategy_data.get("consensus", {})
            if consensus and consensus.get("confidence", 0) > 0.8:
                opinion_str = f"共识: {consensus.get('recommended_focus', 'forward')}"
            else:
                opinion_str = self._summarize_opinions(opinions)
            extra_section = f"## 观点汇总\n{opinion_str}"

            # Hard level also needs completion check
            completion_condition = subtask.completion_condition if subtask else None
            completion_str, _ = self._format_completion_check(
                completion_condition, common["pos_delta"], common["rot_delta"], common["objects_raw"]
            )
            if "COMPLETED" in completion_str:
                extra_section += "\n\n**完成检测**: 已满足条件"

        return f"""导航决策。生成{common['seq_constraint']}动作序列。

## 任务
{common['task_desc']}

## 目标坐标
- 目标: ({common['goal_x']:.2f}, {common['goal_y']:.2f}, {common['goal_z']:.2f})
- 当前: ({common['curr_x']:.2f}, {common['curr_y']:.2f}, {common['curr_z']:.2f})
- 距离: {common['distance']:.1f}m, 方向: {common['direction_hint']} ({common['angle']:.0f}°)
- 垂直: {common['vertical_dir']} ({abs(common['vertical_diff']):.1f}m) - {common['vertical_hint']}
- 成功: 水平<3m 且 垂直<1m

## 场景感知（自然语言描述）
{common['perception_text']}

## 状态
- 已走: {common['dist_traveled']:.1f}m
{common['topology_str']}

{extra_section}

## 规则
1. 输出{common['seq_constraint']}
2. 方向偏差>15°需转向
3. 仅当: "水平<3m" 且 "垂直满足" 且 "所有子任务完成" 才能输出stop！！！

输出JSON（必须严格遵循，不要添加任何额外文字）:
{{"reasoning": "一句话说明", "subtask_completed": false, "actions": [{{"action": "turn_left"}}, {{"action": "turn_left"}}, {{"action": "forward"}}, {{"action": "forward"}}, {{"action": "forward"}}]}}

## 格式要求（违反会导致解析失败）：
1. 必须是纯JSON，不要有任何前缀或后缀文字
2. actions数组必须包含至少5个动作
3. 动作名称只能是: turn_left, turn_right, forward, stop
4. 不要输出代码块标记，直接输出JSON
5. 如果需要转向，先转向动作，然后forward动作
6. 如果直走，直接输出5个forward动作"""

    def _extract_prompt_data(self, context: NavContext, subtask: 'SubTask', strategy_data: dict) -> dict:
        """Extract all common data for prompt building."""
        # perception现在是自然语言字符串，直接使用
        perception_text = strategy_data.get("perception", "No perception data available")
        trajectory = strategy_data.get("trajectory", {})

        # Position info
        pos_info = self._extract_position_info(context)

        # Trajectory info
        subtask_delta = trajectory.get("subtask_delta", {})
        pos_delta = subtask_delta.get("position_delta", {})
        rot_delta = subtask_delta.get("rotation_delta", {})
        dist_traveled = trajectory.get("distance_traveled", 0)

        # Direction info
        angle = context.metadata.get("angle_to_goal", 0) if context else 0
        direction_hint = context.metadata.get("direction_hint", "前方") if context else "前方"
        distance = context.get_distance_to_goal() if hasattr(context, 'get_distance_to_goal') else 5.0

        # Topology
        topology_summary = self._get_topology_summary(context)
        topology_str = ""
        if topology_summary:
            stuck = topology_summary.get("stuck_regions", [])
            if stuck:
                topology_str = f"- 拓扑: 卡住区域{stuck[:2]}"
            else:
                topology_str = f"- 拓扑: {topology_summary.get('total_nodes', 0)}节点"

        # Step constraint
        seq_len = getattr(self, 'sequence_length', 5)
        seq_constraint = f"{self.min_sequence_length}-{self.max_sequence_length}步" if self.adaptive_sequence else f"{seq_len}步"

        return {
            "task_desc": subtask.description if subtask else "导航",
            "goal_x": pos_info["goal_x"],
            "goal_y": pos_info["goal_y"],
            "goal_z": pos_info["goal_z"],
            "curr_x": pos_info["curr_x"],
            "curr_y": pos_info["curr_y"],
            "curr_z": pos_info["curr_z"],
            "vertical_diff": pos_info["vertical_diff"],
            "vertical_dir": pos_info["vertical_direction"],
            "vertical_hint": pos_info["vertical_hint"],
            "perception_text": perception_text,  # 直接使用自然语言描述
            "dist_traveled": dist_traveled,
            "pos_delta": pos_delta,
            "rot_delta": rot_delta,
            "angle": angle,
            "direction_hint": direction_hint,
            "distance": distance,
            "topology_str": topology_str,
            "seq_constraint": seq_constraint,
        }

    def _get_topology_summary(self, context: NavContext) -> dict:
        """Get topology summary from context metadata."""
        if not context:
            return {}
        trajectory_output = context.metadata.get("trajectory_output", {})
        return trajectory_output.get("topology_summary", {}) if isinstance(trajectory_output, dict) else {}

    def _summarize_opinions(self, opinions: Dict) -> str:
        """精简多Agent观点。

        Args:
            opinions: 多Agent观点字典

        Returns:
            精简的观点摘要字符串
        """
        if not opinions:
            return "无观点"

        lines = []
        for agent_name, opinion in opinions.items():
            action = opinion.get("suggested_action", opinion.get("primary_action", "unknown"))
            confidence = opinion.get("confidence", 0.5)
            reason = opinion.get("reasoning", "")[:50]  # 只取前50字符
            lines.append(f"{agent_name}: {action}({confidence:.0%}) - {reason}")

        return "\n".join(lines[:3])  # 只显示3个Agent

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

        # Strategy 4: All parsing failed - do NOT generate fake sequences
        if len(actions) == 0:
            self.logger.warning("[Decision] All parsing strategies failed - returning empty")
            # 不生成假序列，让重试机制处理
            return [], "Parse failed", False

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
                        if action_name in self.ACTION_MAP:
                            actions.append((self.ACTION_MAP[action_name], 1))
                    continue
            else:
                continue

            if action_name in self.ACTION_MAP:
                actions.append((self.ACTION_MAP[action_name], 1))

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

        for pattern in action_patterns:
            matches = re.findall(pattern, json_str, re.IGNORECASE)
            for match in matches:
                action_name = match.lower() if isinstance(match, str) else match[0].lower()
                if action_name in self.ACTION_MAP:
                    actions.append((self.ACTION_MAP[action_name], 1))
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

        for pattern in action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                action_name = match.lower() if isinstance(match, str) else match[0].lower()
                # Try original action_name first, then underscore variant
                if action_name in self.ACTION_MAP:
                    actions.append((self.ACTION_MAP[action_name], 1))
                    success = True
                else:
                    action_name_underscore = action_name.replace(" ", "_")
                    if action_name_underscore in self.ACTION_MAP:
                        actions.append((self.ACTION_MAP[action_name_underscore], 1))
                        success = True

        # Try to extract reasoning from natural text
        reasoning_match = re.search(r'(reasoning|analysis|plan)\s*:?\s*["\']?([^"\']+)["\']?', response, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(2)[:100]

        return reasoning, success

    def _extract_actions_from_cot_analysis(self, analysis: str) -> List[Tuple[ActionType, int]]:
        """从CoT分析文本中提取动作建议。

        CoT分析格式示例：
        Step 4 - Action suggestion: turn left 2-3 times, then move forward 3 times

        Args:
            analysis: CoT策略生成的分析文本

        Returns:
            List of (ActionType, count) tuples
            返回空列表如果无法提取
        """
        if not analysis:
            return []

        # 1. 定位"Action suggestion"行
        suggestion_patterns = [
            r"Step 4.*Action suggestion[:\s]+(.+)",
            r"动作建议[:\s]+(.+)",
            r"Action suggestion[:\s]+(.+)",
        ]

        suggestion_text = None
        for pattern in suggestion_patterns:
            match = re.search(pattern, analysis, re.IGNORECASE)
            if match:
                suggestion_text = match.group(1).strip()
                self.logger.info(f"[Decision] 找到Action suggestion: {suggestion_text[:80]}")
                break

        if not suggestion_text:
            self.logger.info("[Decision] 未找到Action suggestion行")
            return []

        # 2. 提取动作 - 使用finditer保持顺序
        # 转向模式：turn left/right N-M times
        turn_pattern = r"turn\s+(left|right)\s+(\d+)(?:-\d+)?\s*times?"
        # 前进模式：forward/straight N times
        forward_pattern = r"(?:move\s+forward|go\s+(?:straight\s+)?forward|forward)\s+(\d+)(?:-\d+)?\s*times?"

        # 收集所有匹配及其位置，保持顺序
        matches_with_pos = []

        for match in re.finditer(turn_pattern, suggestion_text, re.IGNORECASE):
            direction = match.group(1).lower()
            count = int(match.group(2))
            action = ActionType.TURN_LEFT if direction == "left" else ActionType.TURN_RIGHT
            matches_with_pos.append((match.start(), action, count))

        for match in re.finditer(forward_pattern, suggestion_text, re.IGNORECASE):
            count = int(match.group(1))
            matches_with_pos.append((match.start(), ActionType.MOVE_FORWARD, count))

        # 按位置排序，保持文本中的顺序
        matches_with_pos.sort(key=lambda x: x[0])
        actions = [(m[1], m[2]) for m in matches_with_pos]

        # 3. 如果没有找到模式化的动作，尝试简单关键词匹配
        if not actions:
            simple_patterns = [
                ("left", ActionType.TURN_LEFT, 2),
                ("right", ActionType.TURN_RIGHT, 2),
                ("forward", ActionType.MOVE_FORWARD, 3),
                ("straight", ActionType.MOVE_FORWARD, 3),
            ]
            for keyword, action_type, default_count in simple_patterns:
                if keyword in suggestion_text.lower():
                    actions.append((action_type, default_count))

        self.logger.info(f"[Decision] CoT提取: {len(actions)}个动作组合")
        return actions

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

        is_completed = False
        current_value = 0
        threshold = min_change
        comparison = ""

        if cc_type == "y_change":
            threshold = min_change

            if direction == "down":
                # 下楼需要 dy 为负（y 减小）
                current_value = -dy if dy < 0 else 0  # 只有向下才计入
                is_completed = dy <= -threshold
                comparison = f"dy={dy:.2f}m, need <=-{threshold}m (down)"
            elif direction == "up":
                # 上楼需要 dy 为正（y 增加）
                current_value = dy if dy > 0 else 0
                is_completed = dy >= threshold
                comparison = f"dy={dy:.2f}m, need >={threshold}m (up)"
            else:
                # 无方向要求，用绝对值
                current_value = abs(dy)
                is_completed = abs(dy) >= threshold
                comparison = f"|dy|={abs(dy):.2f}m >={threshold}m"

        elif cc_type == "rotation":
            # Get raw delta_deg (not absolute) for direction-aware checking
            delta_deg = rot_delta.get("delta_deg", 0) if rot_delta else 0
            threshold = min_change

            if direction == "left":
                # 左转需要 delta_deg 为正（角度增加）
                current_value = delta_deg if delta_deg > 0 else 0
                is_completed = delta_deg >= threshold
                comparison = f"rotation={delta_deg:.0f}°, need >= {threshold}° (left)"
            elif direction == "right":
                # 右转需要 delta_deg 为负（角度减少）
                current_value = -delta_deg if delta_deg < 0 else 0
                is_completed = delta_deg <= -threshold
                comparison = f"rotation={delta_deg:.0f}°, need <= -{threshold}° (right)"
            else:
                # 无方向要求，用绝对值
                current_value = abs(delta_deg)
                is_completed = abs(delta_deg) >= threshold
                comparison = f"rotation={abs(delta_deg):.0f}° >= {threshold}°"

        elif cc_type == "distance":
            current_value = horizontal_dist
            threshold = min_change
            comparison = f"distance = {current_value:.2f}m {'>=' if current_value >= threshold else '<'} {threshold}m"
            is_completed = current_value >= threshold

        elif cc_type == "near_object" or cc_type == "object_near":
            target_lower = target_object.lower()
            visible_objs = [o.get("name").lower() for o in (visible_objects or [])]
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