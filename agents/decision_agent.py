"""Decision Agent for generating navigation action sequences.

This version uses Qwen3.5-4B (via remote LLM server) for:
1. 10-step action sequence generation
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
    Agent responsible for generating navigation action sequences.

    Uses Qwen3.5-4B (remote) for sequence generation.

    Key responsibilities:
    1. Generate 10-step action sequences based on strategy results
    2. Judge subtask completion via LLM reasoning
    3. Handle stuck detection for sequence abortion
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

        # Model reference
        self._model_manager = None
        self._initialized = False

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
            reasoning="默认前进",
        )

    def initialize(self) -> None:
        """Initialize model manager."""
        if self._initialized:
            return

        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)
            self._model_manager.load_all_models()
            self._initialized = True
            self.logger.info("DecisionAgent initialized")
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

        # Get strategy data
        strategy_data = strategy_result.metadata if strategy_result else {}

        # Get subtask level
        level = subtask.level if subtask and hasattr(subtask, 'level') else "中等"

        # Build prompt based on difficulty
        prompt = self._build_sequence_prompt_v2(context, subtask, strategy_data, level)

        actions = []
        reasoning = ""
        subtask_completed = False

        try:
            # Call LLM to generate sequence
            if self._model_manager:
                response = self._model_manager.generate(
                    "qwen-4b-decision",
                    prompt,
                    max_new_tokens=300,
                    temperature=0.1,
                )
                if response:
                    self.logger.info(f"[Decision] LLM响应: {response[:200]}...")
                    actions, reasoning, subtask_completed = self._parse_sequence_response(response)
                    self.logger.info(f"[Decision] 解析: {len(actions)}步, 完成:{subtask_completed}")
                else:
                    self.logger.error("[Decision] LLM返回空响应")
            else:
                raise RuntimeError("Model manager not initialized")

        except Exception as e:
            raise RuntimeError(f"[SEQUENCE] LLM生成失败: {e}")

        # If parsing failed, raise error
        if len(actions) < 5:
            raise RuntimeError(f"[SEQUENCE] 解析失败，只得到{len(actions)}个动作，需要至少5个")

        self.logger.info(f"[Decision] 难度:{level}, {len(actions)}步, 完成:{subtask_completed}")

        subtask_id = subtask.id if subtask and hasattr(subtask, 'id') else 0

        return ActionSequence(
            subtask_id=subtask_id,
            subtask_description=subtask.description if subtask else "导航",
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
        depth_image
    ) -> tuple:
        """Check if current sequence should be aborted (stuck detection only)."""
        self._check_position_stuck(context.position)

        if self._stuck_counter >= self._stuck_threshold:
            return True, f"卡住{self._stuck_counter}步"

        return False, ""

    def reset_stuck_counter(self) -> None:
        """Reset stuck detection counter for new episode."""
        self._last_position = None
        self._stuck_counter = 0

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

        # Extract from strategy data
        perception = strategy_data.get("perception", {})
        trajectory = strategy_data.get("trajectory", {})
        instruction = strategy_data.get("instruction", {})

        # Perception info
        room_type = perception.get("room_type", "未知")
        objects_raw = perception.get("objects", [])
        objects = [o.get("物体", o.get("name", str(o))) for o in objects_raw[:3]]
        scene_desc = perception.get("scene_description", "")[:80]
        walkable = perception.get("walkable_analysis", {})
        obstacle = perception.get("obstacle_ahead", {})
        nav_hint = perception.get("nav_hint", "")

        # Open directions
        open_dirs = []
        if isinstance(walkable, dict):
            if walkable.get("left", {}).get("clear", True):
                open_dirs.append("左")
            if walkable.get("center", {}).get("clear", True):
                open_dirs.append("前")
            if walkable.get("right", {}).get("clear", True):
                open_dirs.append("右")
        open_dirs_str = "/".join(open_dirs) if open_dirs else "未知"

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
        heading = trajectory.get("heading", "未知")

        # Instruction semantics
        directions = instruction.get("directions", [])
        instruction_analysis = instruction.get("instruction_analysis", {})
        landmarks = instruction_analysis.get("landmarks", []) if instruction_analysis else instruction.get("landmarks", [])
        goals = instruction_analysis.get("goals", []) if instruction_analysis else instruction.get("goals", [])

        # Completion condition
        completion_condition = subtask.completion_condition if subtask else None

        # Build prompt based on level
        if level == "简单":
            return self._build_simple_prompt(
                subtask, room_type, objects, scene_desc, open_dirs_str,
                blocked, min_dist, dist_traveled, heading, distance_to_goal,
                strategy_data.get("analysis", ""),
                directions, nav_hint, landmarks, goals, completion_condition
            )
        elif level == "中等":
            return self._build_medium_prompt(
                subtask, room_type, objects, scene_desc, open_dirs_str,
                blocked, min_dist, dist_traveled, heading, distance_to_goal,
                strategy_data.get("reflection", ""),
                strategy_data.get("lessons", []),
                directions, nav_hint, landmarks, goals, completion_condition
            )
        else:  # 困难
            return self._build_hard_prompt(
                subtask, room_type, objects, scene_desc, open_dirs_str,
                blocked, min_dist, dist_traveled, heading, distance_to_goal,
                strategy_data.get("opinions", {}),
                strategy_data.get("consensus", {}),
                directions, nav_hint, landmarks, goals, completion_condition
            )

    def _build_simple_prompt(
        self, subtask, room_type, objects, scene_desc, open_dirs_str,
        blocked, min_dist, dist_traveled, heading, distance_to_goal, analysis,
        directions, nav_hint, landmarks, goals, completion_condition=None
    ) -> str:
        """Simple task prompt."""
        directions_str = "/".join(directions) if directions else "未知"
        landmarks_str = ", ".join(landmarks[:3]) if landmarks else "无"
        goals_str = ", ".join(goals[:2]) if goals else "无"
        condition_str = str(completion_condition) if completion_condition else "无"

        return f"""/no_think
你是导航决策系统。根据分析结果，生成10步动作序列。

## 子任务
{subtask.description}

## 子任务完成条件
{condition_str}

## 指令语义
- 关键方向: {directions_str}
- 目标地点: {goals_str}
- 参照物: {landmarks_str}

## 环境分析
- 房间: {room_type}
- 可见物体: {objects if objects else "无"}
- 场景: {scene_desc if scene_desc else "无描述"}
- 开阔方向: {open_dirs_str}
- 前方障碍: {"是(" + str(min_dist) + "米)" if blocked else "否"}
- 导航提示: {nav_hint if nav_hint else "无"}

## 导航状态
- 朝向: {heading}
- 已走: {dist_traveled:.1f}米
- 距目标: {distance_to_goal:.1f}米

## 策略分析
{analysis[:200] if analysis else "无"}

## 动作类型
- forward: 前进一步
- turn_left: 左转15度
- turn_right: 右转15度
- stop: 停止

## 规则
1. 必须输出恰好10步动作
2. 必须包含转向动作
3. 根据完成条件判断子任务是否完成

## 输出格式(JSON)
{{"reasoning":"推理","subtask_completed":false,"actions":[{{"action":"forward"}},{{"action":"turn_right"}},{{"action":"forward"}},{{"action":"turn_left"}},{{"action":"forward"}},{{"action":"turn_right"}},{{"action":"forward"}},{{"action":"turn_left"}},{{"action":"forward"}},{{"action":"forward"}}]}}

直接输出JSON："""

    def _build_medium_prompt(
        self, subtask, room_type, objects, scene_desc, open_dirs_str,
        blocked, min_dist, dist_traveled, heading, distance_to_goal,
        reflection, lessons, directions, nav_hint, landmarks, goals, completion_condition=None
    ) -> str:
        """Medium task prompt with reflection."""
        lessons_str = "无"
        if lessons:
            lessons_str = "\n".join([f"- {l.get('insight', '')[:50]}" for l in lessons[:3]])

        directions_str = "/".join(directions) if directions else "未知"
        landmarks_str = ", ".join(landmarks[:3]) if landmarks else "无"
        goals_str = ", ".join(goals[:2]) if goals else "无"
        condition_str = str(completion_condition) if completion_condition else "无"

        return f"""/no_think
你是导航决策系统。根据反思分析，生成10步动作序列。

## 子任务
{subtask.description}

## 子任务完成条件
{condition_str}

## 指令语义
- 关键方向: {directions_str}
- 目标地点: {goals_str}
- 参照物: {landmarks_str}

## 环境分析
- 房间: {room_type}
- 可见物体: {objects if objects else "无"}
- 场景: {scene_desc if scene_desc else "无描述"}
- 开阔方向: {open_dirs_str}
- 前方障碍: {"是(" + str(min_dist) + "米)" if blocked else "否"}
- 导航提示: {nav_hint if nav_hint else "无"}

## 导航状态
- 朝向: {heading}
- 已走: {dist_traveled:.1f}米
- 距目标: {distance_to_goal:.1f}米

## 反思分析
{reflection[:300] if reflection else "无"}

## 历史教训
{lessons_str}

## 动作类型
- forward: 前进一步
- turn_left: 左转15度
- turn_right: 右转15度
- stop: 停止

## 规则
1. 必须输出恰好10步动作
2. 吸取历史教训，避免重复错误
3. 必须包含转向动作
4. 根据完成条件判断子任务是否完成

## 输出格式(JSON)
{{"reasoning":"推理","subtask_completed":false,"actions":[{{"action":"forward"}},{{"action":"turn_right"}},{{"action":"forward"}},{{"action":"turn_left"}},{{"action":"forward"}},{{"action":"turn_right"}},{{"action":"forward"}},{{"action":"turn_left"}},{{"action":"forward"}},{{"action":"forward"}}]}}

直接输出JSON："""

    def _build_hard_prompt(
        self, subtask, room_type, objects, scene_desc, open_dirs_str,
        blocked, min_dist, dist_traveled, heading, distance_to_goal,
        opinions, consensus, directions, nav_hint, landmarks, goals, completion_condition=None
    ) -> str:
        """Hard task prompt with agent opinions."""
        opinions_str = ""
        if opinions:
            for agent_name, opinion in opinions.items():
                if isinstance(opinion, dict):
                    opinions_str += f"\n### {agent_name}\n"
                    opinions_str += f"- 推荐: {opinion.get('primary_action', '未知')}\n"
                    opinions_str += f"- 理由: {opinion.get('reasoning', '')[:50]}\n"

        consensus_str = consensus.get("reasoning", "无") if consensus else "无"
        directions_str = "/".join(directions) if directions else "未知"
        landmarks_str = ", ".join(landmarks[:3]) if landmarks else "无"
        goals_str = ", ".join(goals[:2]) if goals else "无"
        condition_str = str(completion_condition) if completion_condition else "无"

        return f"""/no_think
你是导航决策系统。根据各Agent意见，生成10步动作序列。

## 子任务
{subtask.description}

## 子任务完成条件
{condition_str}

## 指令语义
- 关键方向: {directions_str}
- 目标地点: {goals_str}
- 参照物: {landmarks_str}

## 环境分析
- 房间: {room_type}
- 可见物体: {objects if objects else "无"}
- 场景: {scene_desc if scene_desc else "无描述"}
- 开阔方向: {open_dirs_str}
- 前方障碍: {"是(" + str(min_dist) + "米)" if blocked else "否"}
- 导航提示: {nav_hint if nav_hint else "无"}

## 导航状态
- 朝向: {heading}
- 已走: {dist_traveled:.1f}米
- 距目标: {distance_to_goal:.1f}米

## Agent意见
{opinions_str if opinions_str else "无"}

## 综合结论
{consensus_str[:200]}

## 动作类型
- forward: 前进一步
- turn_left: 左转15度
- turn_right: 右转15度
- stop: 停止

## 规则
1. 必须输出恰好10步动作
2. 综合各Agent意见做出决策
3. 必须包含转向动作
4. 根据完成条件判断子任务是否完成

## 输出格式(JSON)
{{"reasoning":"推理","subtask_completed":false,"actions":[{{"action":"forward"}},{{"action":"turn_right"}},{{"action":"forward"}},{{"action":"turn_left"}},{{"action":"forward"}},{{"action":"turn_right"}},{{"action":"forward"}},{{"action":"turn_left"}},{{"action":"forward"}},{{"action":"forward"}}]}}

直接输出JSON："""

    # ========== Response Parsing ==========

    def _parse_sequence_response(self, response: str) -> tuple:
        """Parse LLM response to extract actions.

        Returns:
            (actions, reasoning, subtask_completed)
        """
        actions = []
        reasoning = ""
        subtask_completed = False

        # Remove markdown code blocks if present
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            # Remove opening ```json or ```
            lines = cleaned_response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned_response = "\n".join(lines)

        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', cleaned_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                reasoning = data.get("reasoning", "")
                subtask_completed = data.get("subtask_completed", False)

                # Support both "forward" and "move_forward" for compatibility
                action_map = {
                    "forward": ActionType.MOVE_FORWARD,
                    "move_forward": ActionType.MOVE_FORWARD,
                    "turn_left": ActionType.TURN_LEFT,
                    "left": ActionType.TURN_LEFT,
                    "turn_right": ActionType.TURN_RIGHT,
                    "right": ActionType.TURN_RIGHT,
                    "stop": ActionType.STOP,
                }

                for item in data.get("actions", []):
                    # Support both string format ("forward") and object format ({"action": "forward"})
                    if isinstance(item, str):
                        action_name = item.lower()
                    elif isinstance(item, dict):
                        action_name = item.get("action", "forward").lower()
                    else:
                        continue
                    if action_name in action_map:
                        actions.append((action_map[action_name], 1))

            except Exception as e:
                self.logger.warning(f"[Decision] JSON解析失败: {e}")

        return actions, reasoning, subtask_completed

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