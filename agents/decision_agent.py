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

        # Model reference
        self._model_manager = None
        self._initialized = False

        # LLM conversation history
        self._conversation_history: List[Dict[str, str]] = []

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
            if self._should_stop(context, agent_outputs):
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

            # Make decision (LLM or rule-based)
            action, confidence, reasoning, subtask_completed = self._make_decision(context, agent_outputs)

            # Update context
            context.current_action = action
            context.confidence = confidence

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

        # Check perception for goal detection
        if "perception" in agent_outputs:
            landmarks = agent_outputs["perception"].get("landmarks", [])
            close_landmarks = [lm for lm in landmarks if lm.get("distance", 999) < 2.0]
            if close_landmarks:
                current_subtask = context.get_current_subtask()
                if current_subtask:
                    subtask_lower = current_subtask.description.lower()
                    for lm in close_landmarks:
                        if lm.get("name", "") in subtask_lower:
                            return True

        return False

    def _make_decision(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> tuple:
        """Make navigation decision (LLM or rule-based fallback)."""
        # Try LLM-based decision first
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

            # Generate decision
            response = self._model_manager.generate(
                "qwen-4b",
                prompt,
                max_new_tokens=150,
                temperature=0.1
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

        prompt = f"""你是一个室内导航决策专家。请根据当前情况选择最佳导航动作。

## 导航指令
{instruction}

## 当前子任务
{subtask_desc}

## 视觉感知
- 房间: {room_type}
- 物体: {obj_str}
- 目标地标: {lm_str}
- 场景: {scene_desc[:100] if scene_desc else "无"}

## 轨迹状态
- 朝向: {heading}
- 已走: {distance:.1f}米
- 路径问题: {corr_str}
- 步数: {context.step_count}

## 可选动作
- forward: 前进0.25米
- turn_left: 左转15度
- turn_right: 右转15度
- stop: 停止导航

请选择一个动作并解释理由。输出格式:
动作: [forward/turn_left/turn_right/stop]
理由: [一句话解释]

只输出动作和理由，不要其他内容。"""

        return prompt

    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract action."""
        response_lower = response.lower()

        # Extract action
        action = None
        confidence = 0.7

        if "动作:" in response or "action:" in response_lower:
            # Try to extract action from formatted response
            for action_str, action_type in self.ACTION_MAP.items():
                if action_str in response_lower:
                    action = Action(action_type=action_type, confidence=confidence)
                    break
        else:
            # Try to find action keywords
            if "forward" in response_lower or "前进" in response:
                action = Action.forward()
            elif "turn_left" in response_lower or "左转" in response:
                action = Action.turn_left()
            elif "turn_right" in response_lower or "右转" in response:
                action = Action.turn_right()
            elif "stop" in response_lower or "停止" in response:
                action = Action.stop()
                confidence = 0.85

        if action is None:
            action = Action.forward()
            confidence = 0.5

        # Extract reasoning
        reasoning = response.split("理由:")[-1].strip() if "理由:" in response else response[:100]

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

        # Check if mentioned object is found and close
        for lm in landmarks:
            if lm.get("name", "") in description:
                if lm.get("distance", 999) < 2.0:
                    return True

        return False

    def _make_rule_decision(
        self,
        context: NavContext,
        agent_outputs: Dict[str, Any]
    ) -> tuple:
        """Make rule-based navigation decision."""
        instruction = context.instruction.lower()
        perception = agent_outputs.get("perception", {})
        trajectory = agent_outputs.get("trajectory", {})

        # Get landmarks and objects
        landmarks = perception.get("landmarks", [])
        objects = perception.get("objects", [])

        # Check for obstacles
        corrections = trajectory.get("corrections", [])
        is_stuck = any(c.get("type") == "stuck" for c in corrections)

        # Get current subtask
        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description.lower() if current_subtask else ""

        # Decision priority:
        # 1. If stuck, try to escape
        # 2. If landmark found, navigate towards it
        # 3. Follow subtask instruction
        # 4. Follow main instruction keywords
        # 5. Default forward

        # 1. Handle stuck situation
        if is_stuck:
            # Try random turn to escape
            import random
            turn = random.choice(["turn_left", "turn_right"])
            action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
            return action, 0.6, "卡住了，尝试转向", False

        # 2. Navigate towards landmarks
        if landmarks:
            # Find closest landmark
            closest = min(landmarks, key=lambda x: x.get("distance", 999))
            distance = closest.get("distance", 999)
            angle = closest.get("angle", 0)
            name = closest.get("name", "目标")

            # If very close, stop or mark subtask complete
            if distance < 1.5:
                return Action.stop(), 0.85, f"到达目标: {name}", True

            # Turn towards landmark if not aligned
            if abs(angle) > 30:
                if angle < 0:
                    return Action.turn_left(), 0.75, f"转向目标 {name} (左侧)", False
                else:
                    return Action.turn_right(), 0.75, f"转向目标 {name} (右侧)", False

            # Move forward towards landmark
            return Action.forward(), 0.8, f"走向目标: {name} ({distance:.1f}米)", False

        # 3. Follow subtask instruction
        if subtask_desc:
            if "turn left" in subtask_desc or "左转" in subtask_desc:
                return Action.turn_left(), 0.75, "执行子任务: 左转", True
            elif "turn right" in subtask_desc or "右转" in subtask_desc:
                return Action.turn_right(), 0.75, "执行子任务: 右转", True
            elif "stop" in subtask_desc or "停止" in subtask_desc:
                return Action.stop(), 0.85, "执行子任务: 停止", True

        # 4. Follow main instruction keywords
        # Check for direction keywords in instruction
        if "turn left" in instruction or "左转" in instruction:
            return Action.turn_left(), 0.7, "指令指示左转", False
        elif "turn right" in instruction or "右转" in instruction:
            return Action.turn_right(), 0.7, "指令指示右转", False

        # Check for object keywords
        target_keywords = ["find", "locate", "go to", "walk to", "寻找", "走向", "找到"]
        if any(kw in instruction for kw in target_keywords):
            # Check for room keywords
            rooms = ["kitchen", "bedroom", "bathroom", "living room", "厨房", "卧室", "浴室", "客厅"]
            for room in rooms:
                if room in instruction:
                    # Explore by turning occasionally
                    import random
                    if random.random() < 0.2:  # 20% chance to turn
                        turn = random.choice(["turn_left", "turn_right"])
                        action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
                        return action, 0.6, f"探索寻找 {room}", False
                    return Action.forward(), 0.65, f"寻找 {room}", False

        # 5. Check action history to avoid repetition
        if context.action_history:
            recent_actions = [a.action_type for a in context.action_history[-3:]]

            # If oscillating between left and right, go forward
            if ActionType.TURN_LEFT in recent_actions and ActionType.TURN_RIGHT in recent_actions:
                return Action.forward(), 0.6, "避免左右摇摆，前进", False

            # If many consecutive forwards, try turning
            if len(recent_actions) >= 3 and all(a == ActionType.MOVE_FORWARD for a in recent_actions):
                import random
                turn = random.choice(["turn_left", "turn_right"])
                action = Action.turn_left() if turn == "turn_left" else Action.turn_right()
                return action, 0.55, "尝试新方向", False

        # Default: move forward
        return Action.forward(), 0.5, "默认前进", False

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