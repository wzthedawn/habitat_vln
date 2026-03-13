"""Decision Agent for making final navigation decisions.

This version uses Qwen3.5-4B (local, INT8) for:
1. Action decision making
2. Subtask completion judgment
3. Reasoning generation
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
    Agent responsible for making final navigation decisions.

    Uses Qwen3.5-4B (local, INT8) for decision making.

    Key responsibilities:
    1. Synthesize information from other agents
    2. Make navigation action decisions
    3. Judge subtask completion
    4. Handle uncertainty and conflicts
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

        # Decision parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        self.stop_distance = self.config.get("stop_distance", 0.2)
        self.max_history_steps = self.config.get("max_history_steps", 5)
        self.use_llm = self.config.get("use_llm", True)

        # Stuck detection parameters
        self._last_position = None
        self._stuck_counter = 0
        self._stuck_threshold = 3  # Turn if stuck for 3 consecutive steps
        self._turn_counter = 0  # Count consecutive turns
        self._max_turns_before_forward = 2  # After 2 turns, try forward
        self._debate_threshold = 6  # Trigger debate if stuck for 6+ steps
        self._last_debate_step = -999  # Track last debate to avoid spam

        # Model reference
        self._model_manager = None
        self._initialized = False

        # LLM conversation history
        self._conversation_history: List[Dict[str, str]] = []

        # Agent references for debate
        self._perception_agent = None
        self._trajectory_agent = None
        self._instruction_agent = None

    @property
    def name(self) -> str:
        return "decision_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.DECISION

    def get_required_inputs(self) -> List[str]:
        return ["context"]

    def get_output_keys(self) -> List[str]:
        return ["action", "confidence", "reasoning", "subtask_completed"]

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
                self.logger.info("Using remote LLM service for decision making")
                self._model_manager.load_all_models()  # Only loads YOLO locally
            else:
                self._model_manager.load_all_models()

                # Load Qwen3.5-4B for decision making if LLM is enabled
                if self.use_llm:
                    self.logger.info("Loading Qwen3.5-4B for decision making...")
                    if self._model_manager.load_llm("qwen-4b"):
                        self.logger.info("Qwen3.5-4B loaded successfully")
                    else:
                        self.logger.warning("Failed to load Qwen3.5-4B, using rule-based decisions")
                        self.use_llm = False

            self._initialized = True
            self.logger.info("DecisionAgent initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model manager: {e}")
            self._initialized = True

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Make navigation decision."""
        self.initialize()

        try:
            # Gather information from other agents
            agent_outputs = self._gather_agent_outputs(context)

            # Check if we should stop first
            should_stop = self._should_stop(context, agent_outputs)
            self.logger.info(f"[DEBUG] process: _should_stop returned {should_stop}")
            if should_stop:
                action = Action.stop(confidence=0.9)
                reasoning = "目标已到达或所有子任务已完成"

                context.current_action = action
                context.confidence = 0.9

                return AgentOutput.success_output(
                    data={
                        "action": "stop",
                        "action_type": "STOP",
                        "confidence": 0.9,
                        "reasoning": reasoning,
                        "subtask_completed": True,
                    },
                    confidence=0.9,
                    reasoning=reasoning,
                )

            # Check for position-based stuck detection (CRITICAL FIX)
            is_position_stuck = self._check_position_stuck(context.position)
            if is_position_stuck:
                # Check if severely stuck - trigger debate
                if self._stuck_counter >= self._debate_threshold and \
                   context.step_count - self._last_debate_step > 5:
                    # Severe stuck - conduct multi-agent debate
                    self.logger.warning(f"[Stuck Detection] Severe stuck ({self._stuck_counter} steps), triggering debate")
                    action, confidence, reasoning, subtask_completed = self._conduct_debate(context, agent_outputs)

                    context.current_action = action
                    context.confidence = confidence
                    context.add_action(action)

                    if subtask_completed and context.subtasks:
                        current_subtask = context.get_current_subtask()
                        if current_subtask:
                            current_subtask.status = "completed"
                            context.advance_subtask()

                    return AgentOutput.success_output(
                        data={
                            "action": action.to_habitat_action(),
                            "action_type": action.action_type.name,
                            "confidence": confidence,
                            "reasoning": f"[DEBATE] {reasoning}",
                            "subtask_completed": subtask_completed,
                        },
                        confidence=confidence,
                        reasoning=reasoning,
                    )

                # Light stuck - simple turn behavior
                import random

                # Alternate between turning and trying forward
                if self._turn_counter < self._max_turns_before_forward:
                    # Turn to find an open path
                    self._turn_counter += 1
                    turn = random.choice(["turn_left", "turn_right"])
                    if turn == "turn_left":
                        action = Action.turn_left()
                    else:
                        action = Action.turn_right()
                    reasoning = f"检测到卡住(位置未变{self._stuck_counter}步)，转向({self._turn_counter}/{self._max_turns_before_forward})"
                    self.logger.warning(f"[Stuck Detection] Turning {turn} ({self._turn_counter}/{self._max_turns_before_forward})")
                else:
                    # After max turns, try moving forward
                    self._turn_counter = 0  # Reset turn counter
                    action = Action.forward()
                    reasoning = f"转向后尝试前进"
                    self.logger.warning(f"[Stuck Detection] Trying forward after turns")

                context.current_action = action
                context.confidence = 0.7

                # Record action - DO NOT check subtask completion for stuck recovery turns
                # Stuck recovery turns are emergency actions, not intentional subtask execution
                context.add_action(action)
                # subtask_completed is always False for stuck recovery actions
                subtask_completed = False

                return AgentOutput.success_output(
                    data={
                        "action": action.to_habitat_action(),
                        "action_type": action.action_type.name,
                        "confidence": 0.7,
                        "reasoning": reasoning,
                        "subtask_completed": subtask_completed,
                    },
                    confidence=0.7,
                    reasoning=reasoning,
                )

            # Make decision (LLM or rule-based)
            action, confidence, reasoning, subtask_completed = self._make_decision(context, agent_outputs)

            # Update context
            context.current_action = action
            context.confidence = confidence

            # Record action in history
            context.add_action(action)

            # If subtask is completed, mark it
            if subtask_completed and context.subtasks:
                current_subtask = context.get_current_subtask()
                if current_subtask:
                    current_subtask.status = "completed"
                    context.advance_subtask()

            # Generate alternatives
            alternatives = self._generate_alternatives(context, action)

            return AgentOutput.success_output(
                data={
                    "action": action.to_habitat_action(),
                    "action_type": action.action_type.name,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "subtask_completed": subtask_completed,
                    "alternative_actions": alternatives,
                },
                confidence=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.error(f"Decision making error: {e}")
            return AgentOutput.success_output(
                data={
                    "action": "forward",
                    "action_type": "MOVE_FORWARD",
                    "confidence": 0.3,
                    "reasoning": f"决策错误，使用默认动作: {str(e)}",
                    "subtask_completed": False,
                },
                confidence=0.3,
                reasoning=f"Error: {str(e)}",
            )

    def _gather_agent_outputs(self, context: NavContext) -> Dict[str, Any]:
        """Gather outputs from other agents."""
        outputs = {}

        if "instruction_output" in context.metadata:
            outputs["instruction"] = context.metadata["instruction_output"]

        if "perception_output" in context.metadata:
            outputs["perception"] = context.metadata["perception_output"]

        if "trajectory_output" in context.metadata:
            outputs["trajectory"] = context.metadata["trajectory_output"]

        return outputs

    def _should_stop(self, context: NavContext, agent_outputs: Dict) -> bool:
        """Determine if navigation should stop."""
        # Check if all subtasks are completed
        if context.subtasks:
            all_completed = all(s.status == "completed" for s in context.subtasks)
            if all_completed:
                return True

        # Check if we've exceeded max steps
        max_steps = self.config.get("max_steps", 500)
        if context.step_count >= max_steps:
            return True

        # NEW: Check goal_position-based stopping
        # This is the primary success condition - if we're close enough to the goal
        goal_pos = context.metadata.get("goal_position")
        if goal_pos:
            success_distance = context.metadata.get("success_distance", 3.0)
            current_pos = context.position
            dist_to_goal = math.sqrt(
                (goal_pos[0] - current_pos[0])**2 +
                (goal_pos[1] - current_pos[1])**2 +
                (goal_pos[2] - current_pos[2])**2
            )
            if dist_to_goal < success_distance:
                self.logger.info(f"[GOAL CHECK] Close to goal! Distance: {dist_to_goal:.2f}m < {success_distance}m")
                return True

        # Check perception for goal detection
        # Only stop if close landmark matches CURRENT subtask goal
        if "perception" in agent_outputs:
            landmarks = agent_outputs["perception"].get("landmarks", [])
            close_landmarks = [lm for lm in landmarks if lm.get("distance", 999) < 2.0]
            if close_landmarks:
                current_subtask = context.get_current_subtask()
                if current_subtask:
                    subtask_lower = current_subtask.description.lower()
                    # Only stop if subtask mentions stopping/waiting near an object
                    if "stop" in subtask_lower or "wait" in subtask_lower or "near" in subtask_lower:
                        for lm in close_landmarks:
                            lm_name = lm.get("name", "")
                            if lm_name in subtask_lower:
                                return True

        return False

    def _make_decision(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> tuple:
        """Make navigation decision (LLM or rule-based fallback)."""
        # SAFETY FIRST: Check for obstacles before LLM decision
        # Obstacle avoidance is safety-critical and should not rely on LLM
        depth_obstacle = self._check_depth_obstacle(context)
        if depth_obstacle and depth_obstacle.get("has_obstacle"):
            # Use directional turn based on which side has more space
            turn_direction = depth_obstacle.get("suggested_turn", "right")
            if turn_direction == "left":
                return Action.turn_left(), 0.7, f"前方障碍物({depth_obstacle.get('min_distance', 0):.1f}m)，左转", False
            else:
                return Action.turn_right(), 0.7, f"前方障碍物({depth_obstacle.get('min_distance', 0):.1f}m)，右转", False

        # Check for stuck situation
        trajectory = agent_outputs.get("trajectory", {})
        corrections = trajectory.get("corrections", [])
        is_stuck = any(c.get("type") == "stuck" for c in corrections)
        if is_stuck:
            # Check action history to alternate turn direction
            recent = [a.action_type for a in context.action_history[-3:]] if context.action_history else []
            if ActionType.TURN_LEFT in recent and ActionType.TURN_RIGHT not in recent:
                return Action.turn_right(), 0.6, "卡住了，尝试右转", False
            elif ActionType.TURN_RIGHT in recent and ActionType.TURN_LEFT not in recent:
                return Action.turn_left(), 0.6, "卡住了，尝试左转", False
            else:
                return Action.turn_right(), 0.6, "卡住了，尝试转向", False

        # Check for landmarks that need alignment
        landmarks = agent_outputs.get("perception", {}).get("landmarks", [])
        if landmarks:
            closest = min(landmarks, key=lambda x: x.get("distance", 999))
            angle = closest.get("angle", 0)
            name = closest.get("name", "目标")
            # Turn towards landmark if not aligned
            if abs(angle) > 30:
                if angle < 0:
                    return Action.turn_left(), 0.75, f"转向目标 {name} (左侧)", False
                else:
                    return Action.turn_right(), 0.75, f"转向目标 {name} (右侧)", False

        # Try LLM-based decision for high-level navigation
        if self.use_llm and self._model_manager:
            llm_result = self._make_llm_decision(context, agent_outputs)
            if llm_result:
                return llm_result

        # Fallback to rule-based decision
        return self._make_rule_decision(context, agent_outputs)

    def _make_llm_decision(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> Optional[tuple]:
        """Make LLM-based navigation decision using Qwen3.5-4B."""
        if not self._model_manager:
            return None

        try:
            # Build prompt
            prompt = self._build_decision_prompt(context, agent_outputs)

            # Get episode_id for conversation context isolation
            episode_id = context.metadata.get("episode_id", 0)
            conversation_id = f"decision_ep{episode_id}"

            # Generate decision with conversation context
            response = self._model_manager.generate(
                "qwen-4b",
                prompt,
                max_new_tokens=30,  # Reduced from 150 for faster inference
                temperature=0.1,
                conversation_id=conversation_id,
                keep_context=True,
            )

            if response:
                # Parse response
                action, confidence, reasoning = self._parse_llm_response(response)
                if action:
                    # Update conversation history
                    self._conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    if len(self._conversation_history) > 10:
                        self._conversation_history = self._conversation_history[-10:]

                    # Check subtask completion
                    subtask_completed = self._check_subtask_completion(context, action, agent_outputs)

                    return action, confidence, reasoning, subtask_completed

        except Exception as e:
            self.logger.warning(f"LLM decision failed: {e}")

        return None

    def _build_decision_prompt(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> str:
        """Build prompt for decision LLM."""
        instruction = context.instruction
        perception = agent_outputs.get("perception", {})
        trajectory = agent_outputs.get("trajectory", {})

        # Get current subtask
        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description if current_subtask else "无"

        # Get perception info
        room_type = perception.get("room_type", "未知")
        objects = perception.get("objects", [])[:5]
        landmarks = perception.get("landmarks", [])
        scene_desc = perception.get("scene_description", "")

        # Get trajectory info
        heading = trajectory.get("heading", "未知")
        distance = trajectory.get("distance_traveled", 0)
        corrections = trajectory.get("corrections", [])

        # Format objects
        obj_str = ", ".join([f"{o.get('name')}({o.get('distance', 0):.1f}m)" for o in objects]) if objects else "无"

        # Format landmarks
        lm_str = ", ".join([f"{lm.get('name')}({lm.get('distance', 0):.1f}m)" for lm in landmarks[:3]]) if landmarks else "无"

        # Format corrections
        corr_str = "; ".join([c.get("message", "") for c in corrections[:2]]) if corrections else "无"

        # Get recent action history
        recent_actions = context.action_history[-5:] if context.action_history else []
        action_counts = {}
        for a in recent_actions:
            action_type = a.action_type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        action_hist = ", ".join([f"{k}:{v}" for k, v in action_counts.items()]) if action_counts else "无"

        # Extract direction keywords from instruction/subtask
        subtask_lower = subtask_desc.lower() if subtask_desc else ""
        instruction_lower = instruction.lower() if instruction else ""
        need_left = "left" in instruction_lower or "左" in instruction_lower or "left" in subtask_lower
        need_right = "right" in instruction_lower or "右" in instruction_lower or "right" in subtask_lower

        direction_hint = ""
        if need_left and not need_right:
            direction_hint = "指令需要左转"
        elif need_right and not need_left:
            direction_hint = "指令需要右转"

        prompt = f"""根据指令选择动作。

指令: {instruction[:80]}
子任务: {subtask_desc[:50]}
方向提示: {direction_hint if direction_hint else "根据指令判断"}
房间: {room_type}
物体: {obj_str[:60]}
最近动作: {action_hist}

可选动作: forward, turn_left, turn_right, stop

输出格式:
动作: [选择的动作]"""

        return prompt

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract action."""
        response_lower = response.lower().strip()

        # Extract action - priority: formatted response > keywords
        action = None
        confidence = 0.7

        # Try to extract from formatted response first
        # Look for "动作:" pattern and extract the action after it
        action_match = None

        # Find "动作:" or "action:" and extract the action
        if "动作:" in response:
            # Extract content after "动作:"
            action_part = response.split("动作:")[-1].strip()
            action_match = action_part.split()[0] if action_part else None
        elif "action:" in response_lower:
            # Extract content after "action:"
            action_part = response_lower.split("action:")[-1].strip()
            action_match = action_part.split()[0] if action_part else None

        # If we found an action match, parse it properly
        if action_match:
            action_match = action_match.lower().strip()

            # Handle case where LLM outputs multiple options like "forward/turn_right/stop"
            # Split by common delimiters and take the first valid action
            for delimiter in ["/", ",", " ", "或", "或者"]:
                if delimiter in action_match:
                    parts = action_match.split(delimiter)
                    for part in parts:
                        part = part.strip()
                        if part in self.ACTION_MAP:
                            action = Action(action_type=self.ACTION_MAP[part], confidence=confidence)
                            self.logger.debug(f"Parsed action from multi-option response: {part}")
                            break
                    if action:
                        break

            # If no delimiter found, check if the entire string is a valid action
            if action is None and action_match in self.ACTION_MAP:
                action = Action(action_type=self.ACTION_MAP[action_match], confidence=confidence)

        # Fallback: try keyword matching in the response
        if action is None:
            # Check for action keywords with priority - use word boundaries
            import re

            # Priority order: stop > turn_left > turn_right > forward
            # Use word boundary matching to avoid partial matches
            if re.search(r'\bstop\b', response_lower) or "停止" in response:
                action = Action.stop()
                confidence = 0.85
            elif re.search(r'\bturn_left\b', response_lower) or "左转" in response:
                action = Action.turn_left()
            elif re.search(r'\bturn_right\b', response_lower) or "右转" in response:
                action = Action.turn_right()
            elif re.search(r'\bforward\b', response_lower) or "前进" in response:
                action = Action.forward()

        # Default to forward if nothing found
        if action is None:
            action = Action.forward()
            confidence = 0.5

        # Extract reasoning - clean up the text
        reasoning = response
        if "理由:" in response:
            reasoning = response.split("理由:")[-1].strip()
        elif "理由" in response:
            reasoning = response.split("理由")[-1].strip()

        # Remove prompt artifacts that might leak into output
        for artifact in ["user", "##", "指令:", "子任务:", "物体:", "动作选项:"]:
            if artifact in reasoning:
                reasoning = reasoning.split(artifact)[0].strip()

        # Limit reasoning length
        if len(reasoning) > 100:
            reasoning = reasoning[:100]

        return action, confidence, reasoning

    def _check_subtask_completion(
        self,
        context: NavContext,
        action: Action,
        agent_outputs: Dict[str, Any]
    ) -> bool:
        """Check if current subtask is completed based on action and perception."""
        current_subtask = context.get_current_subtask()
        if not current_subtask:
            return False

        # If action is stop, subtask is complete
        if action.action_type == ActionType.STOP:
            return True

        # Check perception for goal detection
        perception = agent_outputs.get("perception", {})
        landmarks = perception.get("landmarks", [])

        description = current_subtask.description.lower()

        # Check if mentioned object is found and close - ONLY for goal-related subtasks
        # The subtask must explicitly mention stopping/waiting/near the object
        is_goal_subtask = "stop" in description or "wait" in description or "near" in description or "arrive" in description
        if is_goal_subtask:
            for lm in landmarks:
                lm_name = lm.get("name", "").lower()
                if lm_name in description:
                    if lm.get("distance", 999) < 1.5:  # Must be closer (1.5m)
                        return True

        # Check if subtask has a direction command and we've turned
        if "turn left" in description or "左转" in description:
            if action.action_type == ActionType.TURN_LEFT:
                return True
        if "turn right" in description or "右转" in description:
            if action.action_type == ActionType.TURN_RIGHT:
                return True

        # Check if subtask is about walking and we've moved forward
        walk_keywords = ["walk", "go", "move", "forward", "down", "up", "pass", "through"]
        has_walk_keyword = any(kw in description for kw in walk_keywords)
        if has_walk_keyword:
            # If we've moved forward at least 8 times for this subtask, consider it done
            if action.action_type == ActionType.MOVE_FORWARD:
                forward_count = sum(1 for a in context.action_history[-12:] if a.action_type == ActionType.MOVE_FORWARD)
                if forward_count >= 8:  # Increased from 5
                    return True

            # Only complete if severely stuck (turned many times without moving forward)
            # This is a last resort to prevent infinite loops
            turn_count = sum(1 for a in context.action_history[-12:]
                           if a.action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT])
            forward_count = sum(1 for a in context.action_history[-12:]
                              if a.action_type == ActionType.MOVE_FORWARD)
            # Require many turns AND almost no forward movement
            if turn_count >= 10 and forward_count <= 2:
                return True

        # For "wait" subtasks, complete only after significant exploration
        if "wait" in description:
            turn_count = sum(1 for a in context.action_history[-15:]
                           if a.action_type in [ActionType.TURN_LEFT, ActionType.TURN_RIGHT])
            if turn_count >= 12:  # Increased threshold
                return True

        return False

    def _make_rule_decision(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> tuple:
        """Make rule-based navigation decision."""
        import random

        instruction = context.instruction.lower()
        perception = agent_outputs.get("perception", {})
        trajectory = agent_outputs.get("trajectory", {})

        # Get landmarks and objects
        landmarks = perception.get("landmarks", [])
        objects = perception.get("objects", [])

        # Check for obstacles from trajectory corrections
        corrections = trajectory.get("corrections", [])
        is_stuck = any(c.get("type") == "stuck" for c in corrections)

        # Check depth image for obstacles
        depth_obstacle = self._check_depth_obstacle(context)

        # Get current subtask
        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description.lower() if current_subtask else ""

        # Decision priority:
        # 1. If depth shows obstacle, avoid it
        # 2. If stuck, try to escape
        # 3. If landmark found, navigate towards it
        # 4. Follow subtask instruction
        # 5. Follow main instruction keywords
        # 6. Check action history to avoid repetition
        # 7. Default forward

        # 1. Depth-based obstacle avoidance
        if depth_obstacle and depth_obstacle.get("has_obstacle"):
            turn = random.choice(["turn_left", "turn_right"])
            action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
            return action, 0.7, f"前方障碍物({depth_obstacle.get('min_distance', 0):.1f}m)，转向", False

        # 2. Handle stuck situation
        if is_stuck:
            # Try random turn to escape
            turn = random.choice(["turn_left", "turn_right"])
            action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
            return action, 0.6, "卡住了，尝试转向", False

        # 2.5. Follow subtask direction instructions FIRST (before landmark navigation)
        # This ensures we execute "turn right" before walking towards landmarks
        # But don't mark subtask as complete - need to verify with perception
        if subtask_desc:
            subtask_lower = subtask_desc.lower()
            if "turn left" in subtask_lower or "左转" in subtask_lower:
                return Action.turn_left(), 0.8, "执行子任务: 左转", False
            elif "turn right" in subtask_lower or "右转" in subtask_lower:
                return Action.turn_right(), 0.8, "执行子任务: 右转", False

        # 3. Navigate towards landmarks (prioritize detected objects matching instruction)
        if landmarks:
            # Find closest landmark
            closest = min(landmarks, key=lambda x: x.get("distance", 999))
            distance = closest.get("distance", 999)
            angle = closest.get("angle", 0)
            name = closest.get("name", "目标")

            # Only stop if landmark matches the instruction goal
            instruction_lower = instruction.lower()
            name_lower = name.lower() if name else ""

            # Check if this landmark is the goal
            # Only stop if:
            # 1. Distance < 1.5m
            # 2. Current subtask explicitly mentions stopping/waiting at this object
            is_goal = False
            if distance < 1.5:
                # Only stop if the current subtask mentions stopping at this object
                # This ensures we don't stop early at objects mentioned later in the instruction
                if subtask_desc:
                    subtask_lower = subtask_desc.lower()
                    # Check if subtask mentions stopping/waiting near this object
                    if ("stop" in subtask_lower or "wait" in subtask_lower or "near" in subtask_lower):
                        if name_lower in subtask_lower:
                            is_goal = True

                if is_goal:
                    return Action.stop(), 0.85, f"到达目标: {name}", True

            # Turn towards landmark if not aligned
            if abs(angle) > 30:
                if angle < 0:
                    return Action.turn_left(), 0.75, f"转向目标 {name} (左侧)", False
                else:
                    return Action.turn_right(), 0.75, f"转向目标 {name} (右侧)", False

            # Move forward towards landmark
            return Action.forward(), 0.8, f"走向目标: {name} ({distance:.1f}米)", False

        # 4. If objects detected but no landmarks, navigate towards relevant objects
        if objects:
            # Filter objects that might be relevant to instruction
            relevant_objects = self._find_relevant_objects(objects, instruction)
            if relevant_objects:
                closest_obj = min(relevant_objects, key=lambda x: x.get("distance", 999))
                distance = closest_obj.get("distance", 999)
                angle = closest_obj.get("angle", 0)
                name = closest_obj.get("name", "物体")

                if abs(angle) > 30:
                    if angle < 0:
                        return Action.turn_left(), 0.7, f"转向 {name}", False
                    else:
                        return Action.turn_right(), 0.7, f"转向 {name}", False
                return Action.forward(), 0.75, f"接近 {name} ({distance:.1f}米)", False

        # 4.5. NEW: Navigate towards goal position if no landmarks/objects detected
        # This is crucial for reaching the goal when visual detection fails
        goal_nav_result = self._navigate_towards_goal(context)
        if goal_nav_result:
            action, confidence, reason, complete = goal_nav_result
            self.logger.info(f"[GOAL NAV] {reason}")
            return action, confidence, reason, complete

        # 5. Follow subtask instruction
        if subtask_desc:
            if "turn left" in subtask_desc or "左转" in subtask_desc:
                return Action.turn_left(), 0.75, "执行子任务: 左转", True
            elif "turn right" in subtask_desc or "右转" in subtask_desc:
                return Action.turn_right(), 0.75, "执行子任务: 右转", True
            elif "stop" in subtask_desc or "停止" in subtask_desc:
                return Action.stop(), 0.85, "执行子任务: 停止", True

        # 6. Follow main instruction keywords
        # Check for direction keywords in instruction
        if "turn left" in instruction or "左转" in instruction:
            return Action.turn_left(), 0.7, "指令指示左转", False
        elif "turn right" in instruction or "右转" in instruction:
            return Action.turn_right(), 0.7, "指令指示右转", False

        # Check for object keywords
        target_keywords = ["find", "locate", "go to", "walk to", "寻找", "走向", "找到", "towards", "near"]
        if any(kw in instruction for kw in target_keywords):
            # Check for room keywords
            rooms = ["kitchen", "bedroom", "bathroom", "living room", "厨房", "卧室", "浴室", "客厅"]
            for room in rooms:
                if room in instruction:
                    # Explore by turning occasionally
                    if random.random() < 0.25:  # 25% chance to turn
                        turn = random.choice(["turn_left", "turn_right"])
                        action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
                        return action, 0.6, f"探索寻找 {room}", False
                    return Action.forward(), 0.65, f"寻找 {room}", False

        # 7. Check action history to avoid repetition
        if context.action_history:
            recent_actions = [a.action_type for a in context.action_history[-3:]]

            # If oscillating between left and right, go forward
            if ActionType.TURN_LEFT in recent_actions and ActionType.TURN_RIGHT in recent_actions:
                return Action.forward(), 0.6, "避免左右摇摆，前进", False

            # If many consecutive forwards, try turning
            if len(recent_actions) >= 3 and all(a == ActionType.MOVE_FORWARD for a in recent_actions):
                turn = random.choice(["turn_left", "turn_right"])
                action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
                return action, 0.55, "尝试新方向", False

        # Default: move forward
        return Action.forward(), 0.5, "默认前进", False

    def _navigate_towards_goal(self, context: NavContext) -> Optional[tuple]:
        """
        Navigate towards goal position when no landmarks detected.
        Returns (action, confidence, reason, subtask_complete) or None if not applicable.
        """
        goal_pos = context.metadata.get("goal_position")
        if not goal_pos:
            return None

        current_pos = context.position
        current_rotation = context.rotation  # in radians

        # Calculate direction to goal
        dx = goal_pos[0] - current_pos[0]
        dz = goal_pos[2] - current_pos[2]
        distance = math.sqrt(dx*dx + dz*dz)

        success_distance = context.metadata.get("success_distance", 3.0)

        # If very close to goal, stop
        if distance < success_distance:
            return Action.stop(), 0.9, f"到达目标位置 (距离: {distance:.1f}m)", True

        # Calculate angle to goal
        # Goal is at angle theta from current position
        # We need to turn to face the goal
        goal_angle = math.atan2(-dx, dz)  # Habitat uses Z-forward, X-right

        # Current rotation is the direction we're facing
        # We need to find the difference between our current direction and the goal direction
        angle_diff = goal_angle - current_rotation

        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Convert to degrees for threshold checking
        angle_diff_deg = math.degrees(angle_diff)

        self.logger.debug(f"[GOAL NAV] Distance: {distance:.1f}m, Angle diff: {angle_diff_deg:.1f}°")

        # If not aligned with goal, turn
        if abs(angle_diff_deg) > 30:  # More than 30 degrees off
            if angle_diff_deg > 0:
                return Action.turn_left(), 0.75, f"转向目标 (左转 {angle_diff_deg:.0f}°)", False
            else:
                return Action.turn_right(), 0.75, f"转向目标 (右转 {-angle_diff_deg:.0f}°)", False

        # Aligned with goal, move forward
        return Action.forward(), 0.8, f"向目标前进 (距离: {distance:.1f}m)", False

    def _check_depth_obstacle(self, context: NavContext) -> Dict[str, Any]:
        """Check for obstacles using depth image."""
        result = {"has_obstacle": False, "min_distance": float('inf'), "suggested_turn": "right"}

        # Get depth image from context
        depth_image = getattr(context, 'depth_image', None)
        if depth_image is None:
            depth_image = context.metadata.get('depth_image')

        if depth_image is None:
            return result

        try:
            import numpy as np
            h, w = depth_image.shape[:2]

            # Check center region for obstacles
            center_region = depth_image[h//3:2*h//3, w//3:2*w//3]
            valid_depths = center_region[center_region > 0]

            if len(valid_depths) > 0:
                min_dist = float(np.min(valid_depths))
                mean_dist = float(np.mean(valid_depths))

                # Consider it an obstacle if minimum distance < 0.5m
                if min_dist < 0.5:
                    result["has_obstacle"] = True
                    result["min_distance"] = min_dist
                    result["mean_distance"] = mean_dist

                    # Determine which side has more space
                    left_region = depth_image[h//3:2*h//3, :w//3]
                    right_region = depth_image[h//3:2*h//3, 2*w//3:]

                    left_valid = left_region[left_region > 0]
                    right_valid = right_region[right_region > 0]

                    left_mean = float(np.mean(left_valid)) if len(left_valid) > 0 else 0
                    right_mean = float(np.mean(right_valid)) if len(right_valid) > 0 else 0

                    # Suggest turn towards the side with more space
                    if left_mean > right_mean:
                        result["suggested_turn"] = "left"
                    else:
                        result["suggested_turn"] = "right"
        except Exception as e:
            self.logger.debug(f"Depth check failed: {e}")

        return result

    def _find_relevant_objects(self, objects: List[Dict], instruction: str) -> List[Dict]:
        """Find objects that might be relevant to the instruction."""
        instruction_lower = instruction.lower()
        relevant = []

        # Keywords for furniture and objects commonly found in navigation instructions
        navigation_objects = [
            "chair", "table", "desk", "bed", "sofa", "couch", "bench",
            "door", "stairs", "rug", "carpet", "piano", "tv", "refrigerator",
            "oven", "sink", "toilet", "bath", "shower",
            # Chinese equivalents
            "椅子", "桌子", "床", "沙发", "门", "楼梯", "地毯", "钢琴"
        ]

        for obj in objects:
            name = obj.get("name", "").lower()
            # Check if object name appears in instruction
            if name in instruction_lower:
                relevant.append(obj)
            # Check if object is a navigation-relevant type
            elif any(nav_obj in name for nav_obj in navigation_objects):
                relevant.append(obj)

        return relevant

    def _create_action(self, action_str: str) -> Action:
        """Create Action object from string."""
        action_str = action_str.lower().strip()
        action_type = self.ACTION_MAP.get(action_str, ActionType.MOVE_FORWARD)
        return Action(action_type=action_type, confidence=0.7)

    def _generate_alternatives(
        self, context: NavContext, primary_action: Action
    ) -> List[Dict[str, Any]]:
        """Generate alternative actions."""
        alternatives = []

        all_actions = [
            (ActionType.MOVE_FORWARD, "forward"),
            (ActionType.TURN_LEFT, "turn_left"),
            (ActionType.TURN_RIGHT, "turn_right"),
            (ActionType.STOP, "stop"),
        ]

        for action_type, name in all_actions:
            if action_type != primary_action.action_type:
                alternatives.append({
                    "action": name,
                    "confidence": 0.3,
                    "reasoning": f"备选: {name}",
                })

        return alternatives[:3]

    def evaluate_subtask_completion(
        self,
        context: NavContext,
        perception: Dict[str, Any]
    ) -> bool:
        """Evaluate if current subtask is completed."""
        current_subtask = context.get_current_subtask()
        if not current_subtask:
            return True

        description = current_subtask.description.lower()
        landmarks = perception.get("landmarks", [])

        # Check if mentioned object is found and close
        for lm in landmarks:
            if lm.get("name", "") in description:
                if lm.get("distance", 999) < 2.0:
                    return True

        # Check for direction completion
        if "turn left" in description:
            if context.action_history:
                last_action = context.action_history[-1]
                if last_action.action_type == ActionType.TURN_LEFT:
                    return True
        elif "turn right" in description:
            if context.action_history:
                last_action = context.action_history[-1]
                if last_action.action_type == ActionType.TURN_RIGHT:
                    return True
        elif "stop" in description:
            return True

        return False

    def _check_position_stuck(self, position: tuple) -> bool:
        """Check if agent is stuck (position hasn't changed significantly).

        Args:
            position: Current position tuple (x, y, z)

        Returns:
            True if stuck (position unchanged for too many steps)
        """
        if position is None:
            return False

        if self._last_position is None:
            self._last_position = position
            self._stuck_counter = 0
            return False

        # Calculate distance moved
        import math
        dx = position[0] - self._last_position[0]
        dz = position[2] - self._last_position[2]
        distance_moved = math.sqrt(dx*dx + dz*dz)

        # Update last position
        self._last_position = position

        # If moved less than 0.1m, increment stuck counter
        if distance_moved < 0.1:
            self._stuck_counter += 1
        else:
            # Position changed - reset both counters
            self._stuck_counter = 0
            self._turn_counter = 0

        # Return True if stuck for threshold steps
        return self._stuck_counter >= self._stuck_threshold

    def reset_stuck_counter(self) -> None:
        """Reset stuck detection counter for new episode."""
        self._last_position = None
        self._stuck_counter = 0
        self._turn_counter = 0
        self._last_debate_step = -999

    def _conduct_debate(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> tuple:
        """
        Conduct a multi-agent debate when stuck.

        Collects opinions from all agents and synthesizes them
        to make a better decision when the agent is stuck.

        Args:
            context: Navigation context
            agent_outputs: Outputs from other agents

        Returns:
            tuple: (action, confidence, reasoning, subtask_completed)
        """
        self.logger.info("[DEBATE] Starting multi-agent debate for stuck situation...")

        # Gather opinions from each agent
        opinions = {}

        # 1. Instruction Agent opinion - what should we do according to instruction?
        instruction_opinion = self._get_instruction_opinion(context, agent_outputs)
        opinions["instruction"] = instruction_opinion

        # 2. Perception Agent opinion - what does visual analysis suggest?
        perception_opinion = self._get_perception_opinion(context, agent_outputs)
        opinions["perception"] = perception_opinion

        # 3. Trajectory Agent opinion - what does path analysis suggest?
        trajectory_opinion = self._get_trajectory_opinion(context, agent_outputs)
        opinions["trajectory"] = trajectory_opinion

        # 4. Decision Agent (self) opinion - what's the strategic decision?
        decision_opinion = self._get_decision_opinion(context, agent_outputs)
        opinions["decision"] = decision_opinion

        # 5. Exploration opinion - when severely stuck, suggest random exploration
        if self._stuck_counter >= self._debate_threshold + 4:  # Very severely stuck (10+ steps)
            exploration_opinion = self._get_exploration_opinion(context, agent_outputs)
            opinions["exploration"] = exploration_opinion
            self.logger.info(f"[DEBATE] Severe stuck ({self._stuck_counter} steps), adding exploration opinion")

        # Log all opinions
        self.logger.info(f"[DEBATE] Opinions collected:")
        for agent, opinion in opinions.items():
            self.logger.info(f"  {agent}: {opinion.get('action', 'unknown')} - {opinion.get('reason', '')[:50]}")

        # Synthesize opinions into final decision
        final_action, confidence, reasoning = self._synthesize_debate_opinions(
            context, opinions, agent_outputs
        )

        self.logger.info(f"[DEBATE] Final decision: {final_action.action_type.name} - {reasoning}")

        # Update last debate step
        self._last_debate_step = context.step_count

        subtask_completed = self._check_subtask_completion(context, final_action, agent_outputs)

        return final_action, confidence, reasoning, subtask_completed

    def _get_instruction_opinion(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get opinion from instruction perspective."""
        current_subtask = context.get_current_subtask()
        instruction = context.instruction.lower()

        opinion = {
            "action": "forward",
            "reason": "按照指令前进",
            "priority": 0.5
        }

        if current_subtask:
            desc = current_subtask.description.lower()

            # Check for direction keywords in current subtask
            if "turn left" in desc or "左转" in desc:
                opinion = {
                    "action": "turn_left",
                    "reason": f"子任务要求左转: {desc[:30]}",
                    "priority": 0.9
                }
            elif "turn right" in desc or "右转" in desc:
                opinion = {
                    "action": "turn_right",
                    "reason": f"子任务要求右转: {desc[:30]}",
                    "priority": 0.9
                }
            elif "stop" in desc or "wait" in desc:
                opinion = {
                    "action": "stop",
                    "reason": f"子任务要求停止: {desc[:30]}",
                    "priority": 0.8
                }
            else:
                # Check for object/landmark keywords
                landmarks = agent_outputs.get("perception", {}).get("landmarks", [])
                for lm in landmarks:
                    if lm.get("name", "") in desc and lm.get("distance", 999) < 3.0:
                        angle = lm.get("angle", 0)
                        if abs(angle) > 30:
                            opinion = {
                                "action": "turn_left" if angle < 0 else "turn_right",
                                "reason": f"子任务地标 {lm.get('name')} 在{'左' if angle < 0 else '右'}侧",
                                "priority": 0.8
                            }
                        break

        return opinion

    def _get_perception_opinion(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get opinion from perception perspective."""
        perception = agent_outputs.get("perception", {})

        opinion = {
            "action": "forward",
            "reason": "无视觉障碍",
            "priority": 0.5
        }

        # Check for obstacles using depth
        depth_obstacle = self._check_depth_obstacle(context)
        if depth_obstacle.get("has_obstacle"):
            turn = depth_obstacle.get("suggested_turn", "right")
            opinion = {
                "action": f"turn_{turn}",
                "reason": f"前方障碍物({depth_obstacle.get('min_distance', 0):.1f}m)",
                "priority": 0.9
            }
            return opinion

        # Check for landmarks
        landmarks = perception.get("landmarks", [])
        if landmarks:
            closest = min(landmarks, key=lambda x: x.get("distance", 999))
            distance = closest.get("distance", 999)
            angle = closest.get("angle", 0)
            name = closest.get("name", "目标")

            if distance < 1.5:
                opinion = {
                    "action": "stop",
                    "reason": f"接近地标 {name} ({distance:.1f}m)",
                    "priority": 0.8
                }
            elif abs(angle) > 30:
                opinion = {
                    "action": "turn_left" if angle < 0 else "turn_right",
                    "reason": f"地标 {name} 在{'左' if angle < 0 else '右'}侧 ({distance:.1f}m)",
                    "priority": 0.7
                }
            else:
                opinion = {
                    "action": "forward",
                    "reason": f"地标 {name} 在前方 ({distance:.1f}m)",
                    "priority": 0.7
                }

        # Check objects for navigation hints
        objects = perception.get("objects", [])
        nav_objects = [o for o in objects if o.get("is_navigation_object") or o.get("is_landmark")]
        if nav_objects and not landmarks:
            # Find closest navigation-relevant object
            closest_obj = min(nav_objects, key=lambda x: x.get("distance", 999))
            angle = closest_obj.get("angle", 0)
            if abs(angle) > 45:
                opinion = {
                    "action": "turn_left" if angle < 0 else "turn_right",
                    "reason": f"导航物体 {closest_obj.get('name')} 在{'左' if angle < 0 else '右'}侧",
                    "priority": 0.6
                }

        return opinion

    def _get_trajectory_opinion(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get opinion from trajectory perspective."""
        trajectory = agent_outputs.get("trajectory", {})

        opinion = {
            "action": "forward",
            "reason": "轨迹正常",
            "priority": 0.5
        }

        # Check for stuck corrections
        corrections = trajectory.get("corrections", [])
        stuck_corrections = [c for c in corrections if c.get("type") == "stuck"]

        if stuck_corrections:
            # Analyze action history to find best escape
            if context.action_history:
                recent = context.action_history[-5:]
                turn_left_count = sum(1 for a in recent if a.action_type == ActionType.TURN_LEFT)
                turn_right_count = sum(1 for a in recent if a.action_type == ActionType.TURN_RIGHT)
                forward_count = sum(1 for a in recent if a.action_type == ActionType.MOVE_FORWARD)

                if turn_left_count > turn_right_count:
                    opinion = {
                        "action": "turn_right",
                        "reason": "已多次左转，尝试右转脱困",
                        "priority": 0.7
                    }
                elif turn_right_count > turn_left_count:
                    opinion = {
                        "action": "turn_left",
                        "reason": "已多次右转，尝试左转脱困",
                        "priority": 0.7
                    }
                else:
                    opinion = {
                        "action": "forward",
                        "reason": "转向均衡，尝试前进",
                        "priority": 0.6
                    }

        # Check distance traveled
        distance = trajectory.get("distance_traveled", 0)
        if distance < 1.0 and context.step_count > 10:
            opinion = {
                "action": "turn_right",
                "reason": f"行进距离短({distance:.1f}m)，需要探索",
                "priority": 0.6
            }

        return opinion

    def _get_decision_opinion(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get strategic opinion from decision perspective."""
        opinion = {
            "action": "forward",
            "reason": "默认前进",
            "priority": 0.4
        }

        # Consider overall progress
        subtask_progress = 0
        if context.subtasks:
            completed = sum(1 for s in context.subtasks if s.status == "completed")
            total = len(context.subtasks)
            subtask_progress = completed / total if total > 0 else 0

        # If making good progress, continue current strategy
        if subtask_progress > 0:
            # Check recent actions for pattern
            if context.action_history:
                recent = context.action_history[-3:]
                last_types = [a.action_type for a in recent]

                # If mostly turning, try forward
                forward_ratio = sum(1 for t in last_types if t == ActionType.MOVE_FORWARD) / len(last_types)
                if forward_ratio < 0.3:
                    opinion = {
                        "action": "forward",
                        "reason": "频繁转向后尝试前进",
                        "priority": 0.7
                    }
                else:
                    opinion = {
                        "action": "turn_right",
                        "reason": "探索新方向",
                        "priority": 0.5
                    }

        return opinion

    def _get_exploration_opinion(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get exploration opinion for severe stuck situations.

        When very stuck, suggests trying completely different directions
        that haven't been tried recently.
        """
        import random

        opinion = {
            "action": "turn_right",
            "reason": "严重卡住，尝试探索",
            "priority": 0.85  # High priority for severe stuck
        }

        # Analyze recent turn patterns
        if context.action_history:
            recent = context.action_history[-10:]
            turn_left_count = sum(1 for a in recent if a.action_type == ActionType.TURN_LEFT)
            turn_right_count = sum(1 for a in recent if a.action_type == ActionType.TURN_RIGHT)

            # Calculate total turning angle (assuming 15 degrees per turn)
            total_turn = (turn_right_count - turn_left_count) * 15

            # If we've turned a lot in one direction, try the other
            if turn_left_count > turn_right_count + 2:
                opinion = {
                    "action": "turn_right",
                    "reason": f"已左转{turn_left_count}次，尝试右转探索",
                    "priority": 0.85
                }
            elif turn_right_count > turn_left_count + 2:
                opinion = {
                    "action": "turn_left",
                    "reason": f"已右转{turn_right_count}次，尝试左转探索",
                    "priority": 0.85
                }
            else:
                # Try random direction with slight forward bias
                if random.random() < 0.3:  # 30% chance to try forward
                    opinion = {
                        "action": "forward",
                        "reason": "探索：尝试前进",
                        "priority": 0.7
                    }
                else:
                    # Try the direction with more open space
                    depth_obstacle = self._check_depth_obstacle(context)
                    if depth_obstacle.get("suggested_turn"):
                        opinion = {
                            "action": f"turn_{depth_obstacle.get('suggested_turn')}",
                            "reason": f"探索：向开阔方向({depth_obstacle.get('suggested_turn')})",
                            "priority": 0.8
                        }

        return opinion

    def _synthesize_debate_opinions(
        self,
        context: NavContext,
        opinions: Dict[str, Dict],
        agent_outputs: Dict[str, Any]
    ) -> tuple:
        """
        Synthesize opinions from all agents into final decision.

        Uses weighted voting based on priority and agent relevance.
        """
        # Count votes for each action
        action_scores = {
            "forward": 0.0,
            "turn_left": 0.0,
            "turn_right": 0.0,
            "stop": 0.0
        }

        # Agent weights - perception and instruction are more important
        weights = {
            "instruction": 1.0,
            "perception": 1.2,  # Higher weight for obstacle avoidance
            "trajectory": 0.8,
            "decision": 0.6,
            "exploration": 1.0  # Weight for severe stuck exploration
        }

        # Aggregate scores
        for agent_name, opinion in opinions.items():
            action = opinion.get("action", "forward")
            priority = opinion.get("priority", 0.5)
            weight = weights.get(agent_name, 1.0)

            if action in action_scores:
                action_scores[action] += priority * weight

        # Log scores
        self.logger.info(f"[DEBATE] Action scores: {action_scores}")

        # Find best action
        best_action = max(action_scores, key=action_scores.get)
        confidence = min(action_scores[best_action] / 3.0, 0.9)

        # Get reasoning from the highest priority opinion for this action
        reasoning = "综合各agent意见"
        for agent_name, opinion in opinions.items():
            if opinion.get("action") == best_action:
                reasoning = opinion.get("reason", reasoning)
                break

        # Create action
        action = self._create_action(best_action)

        return action, confidence, reasoning