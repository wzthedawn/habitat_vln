"""感知智能体 - 使用VLM分析视觉观测

纯VLM方案，自然语言描述输出。
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext


class PerceptionAgent(BaseAgent):
    """感知智能体 - 负责视觉感知分析

    使用VLM分析RGB+Depth图像，返回自然语言场景描述。
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("PerceptionAgent")
        self._model_manager = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "perception_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.PERCEPTION

    def get_required_inputs(self) -> List[str]:
        return ["rgb_image", "depth_image"]

    def get_output_keys(self) -> List[str]:
        return ["scene_description"]

    def initialize(self) -> None:
        """初始化模型管理器"""
        if self._initialized:
            return
        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)
            self._initialized = True
            self.logger.info("感知智能体初始化完成（RGB+Depth模式）")
        except Exception as e:
            self.logger.warning(f"模型管理器初始化失败: {e}")
            self._initialized = True

    # ============================================================
    # 第二部分：核心处理流程
    # ============================================================

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
        subtask: Optional['SubTask'] = None,  # 新增：当前子任务
    ) -> AgentOutput:
        """处理视觉观测

        使用VLM分析RGB+Depth图像，返回感知结果。

        Args:
            context: 导航上下文，包含视觉特征
            strategy_result: 可选的策略输出
            subtask: 当前子任务（用于任务导向感知）

        Returns:
            包含感知结果的AgentOutput
        """
        self.initialize()

        try:
            # 从上下文获取图像
            rgb_image = self._get_rgb_image(context)
            depth_image = self._get_depth_image(context)

            # 使用VLM进行所有感知（RGB+Depth融合）
            vlm_result = None
            if self._model_manager:
                # 获取目标方向信息
                goal_direction = None
                if context and hasattr(context, 'metadata'):
                    angle = context.metadata.get("angle_to_goal", 0)
                    hint = context.metadata.get("direction_hint", "ahead")
                    goal_direction = {"angle": angle, "hint": hint}

                # 获取子任务描述（用于任务导向感知）
                subtask_desc = subtask.description if subtask else ""

                vlm_result = self._analyze_with_vlm(rgb_image, depth_image, goal_direction, subtask_desc)

            # vlm_result现在是自然语言字符串
            scene_description = vlm_result if vlm_result else "Unable to analyze scene"

            # 直接存储自然语言描述到metadata
            context.metadata["perception_output"] = scene_description
            context.visual_features.scene_description = scene_description

            return AgentOutput.success_output(
                data=scene_description,
                confidence=0.7,
                reasoning=f"Scene analyzed: {scene_description[:100]}...",
            )

        except Exception as e:
            self.logger.error(f"[Perception] Error: {e}")
            return AgentOutput.failure_output([str(e)], "Perception processing failed")

    def _get_rgb_image(self, context: NavContext) -> Optional[np.ndarray]:
        """从上下文获取RGB图像"""
        if hasattr(context, 'rgb_image') and context.rgb_image is not None:
            return context.rgb_image
        if context.metadata.get('rgb_image') is not None:
            return context.metadata['rgb_image']
        return None

    def _get_depth_image(self, context: NavContext) -> Optional[np.ndarray]:
        """从上下文获取深度图像"""
        if hasattr(context, 'depth_image') and context.depth_image is not None:
            return context.depth_image
        if context.metadata.get('depth_image') is not None:
            return context.metadata['depth_image']
        return None

    def _analyze_with_vlm(
        self,
        rgb_image: Optional[np.ndarray],
        depth_image: Optional[np.ndarray],
        goal_direction: Optional[Dict[str, Any]] = None,
        subtask_desc: str = "",  # 新增：子任务描述
    ) -> Optional[Dict[str, Any]]:
        """调用VLM分析图像

        使用VLM分析RGB+Depth图像，包含目标方向和任务导向感知。

        Args:
            rgb_image: RGB图像
            depth_image: 深度图像
            goal_direction: 目标方向信息 {"angle": 角度, "hint": 方向提示}
            subtask_desc: 当前子任务描述（用于任务导向感知）

        Returns:
            自然语言场景描述字符串
        """
        if not self._model_manager:
            return None

        try:
            # 构建目标方向提示
            goal_hint = ""
            if goal_direction:
                angle = goal_direction.get("angle", 0)
                hint = goal_direction.get("hint", "ahead")
                goal_hint = f"""
## Goal Direction
- Goal is {hint} (relative angle {angle:.0f}°)"""

            # 构建任务导向提示
            task_hint = ""
            if subtask_desc:
                task_hint = f"""
## Navigation Task
- Current subtask: {subtask_desc}
- Focus on finding objects and paths related to this task"""

            # RGB+Depth：任务导向感知模式
            prompt = f"""
You are a navigation robot analyzing images to guide movement decisions.

## Images
- First: RGB image (visual scene)
- Second: Depth image showing distance:
  * **RED = CLOSE (near you)** - The more red, the closer
  * **BLUE = FAR (far from you)** - The more blue, the farther
  * Gradient: red → yellow → green → cyan → blue (near → far)
{goal_hint}
subtask：{task_hint}

## Your Task
Describe what you see in your view in 80-100 words, Especially relevant to the subtask.
ONLY describe what you can CLEARLY SEE in the image.
The visual description must be output sequentially according to the following four dimensions:

1. **Environment**: What type of space? (corridor, room, stairwell entrance, open area, etc.)
2. **Visible elements**: List objects by selecting "Close-up (<2 meters)", "Medium shot (2–5 meters)", or "Long shot (>5 meters)":
    -Close-up: Provide detailed descriptions of its material, color, and whether it obstructs passage.
    -Medium shot: This is crucial for positioning.Describe the relative positions of objects (sofa, stairs).For example: "On the left side of the mid-plan, there is a gray fabric sofa seating three people, placed against the wall," or "In front of the mid-plan, there is a staircase extending upward."
    -Long shot: Objects visible in the distance used to determine room depth.
3. **Path analysis**: What does the path ahead look like?
   - Identifies all doors, corridor entrances, and archways.Describe their status (open/closed/semi-open) and type (sliding door/hinged door).
   - Based on ground-level analysis, identifies open spaces and crowded zones
   - What can be seen through the gaps between object？
   - Use depth colors to determine:
     * Red at bottom of image + blue at top → path goes UP (ascending stairs)
     * Red at top of image + blue at bottom → path goes DOWN (descending stairs)
     * Similar colors across path → flat surface
4. **Navigation cues**: What features help navigation? (doorways, railings, clear paths, obstacles)

## Important
- Describe PARTIAL visibility accurately: "At the far end, stairs appear to ascend..."
- If stairs are partially visible at the edge/corner, mention it explicitly
- Do NOT ignore subtle features - they may be critical for navigation
- Base your elevation analysis on depth color gradient, not just visual appearance

Output natural language description only, no JSON, no formatting:"""

            # 调用VLM分析RGB+Depth图像（直接返回自然语言，无需JSON解析）
            max_retries = 2
            scene_description = None

            for attempt in range(max_retries + 1):
                self.logger.info(f"[Perception] Calling VLM (attempt {attempt+1}/{max_retries+1})")
                result = self._model_manager.generate_vision_dual(
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    prompt=prompt,
                    model_key="qwen-9b-perception",
                    max_new_tokens=300,
                    temperature=0.3,
                )

                if not result or not result.get("response"):
                    self.logger.warning(f"[Perception] No VLM response on attempt {attempt+1}")
                    continue

                raw_response = result["response"].strip()
                self.logger.info(f"[Perception] VLM response ({len(raw_response)} chars): {raw_response[:100]}...")

                # 自然语言模式：直接检查响应长度
                if len(raw_response) > 50:
                    scene_description = raw_response
                    self.logger.info(f"[Perception] Valid description received")
                    break
                else:
                    self.logger.warning(f"[Perception] Response too short, retrying...")

            # 打印输出
            print(f"\n[PerceptionAgent] Output (natural language):")
            print(f"  {scene_description if scene_description else 'Failed to get description'}")

            # 直接返回文本字符串
            return scene_description if scene_description else "Unable to analyze the scene."

        except Exception as e:
            self.logger.error(f"[Perception] VLM analysis failed: {e}")
            return f"Perception error: {str(e)}"

    # ============================================================
    # Debate接口（多智能体协商）
    # ============================================================

    def build_debate_opinion(
        self,
        context: NavContext,
        depth_info: Dict[str, Any] = None
    ) -> "DebateOpinion":
        """构建多智能体协商观点，使用LLM进行感知分析。

        Args:
            context: 导航上下文
            depth_info: 深度信息（已忽略，VLM完成所有分析）

        Returns:
            包含基于感知的约束的DebateOpinion对象
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        # perception_output is now a natural language string
        scene_description = context.metadata.get("perception_output", "")

        # Use LLM for perception-based opinion
        if self._model_manager:
            return self._build_perception_opinion_with_llm(
                context, scene_description
            )
        else:
            return self._build_perception_opinion_fallback(scene_description)

    def _build_perception_opinion_with_llm(
        self,
        context: NavContext,
        scene_description: str
    ) -> "DebateOpinion":
        """使用LLM构建感知观点。

        Args:
            context: 导航上下文
            scene_description: 场景自然语言描述

        Returns:
            基于LLM分析的DebateOpinion对象
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        prompt = f"""/no_think
Analyze the best navigation action based on perception information.

## Environment Perception
- Scene description: {scene_description[:150] if scene_description else "none"}
- Room type: {context.room_type}

## Recent Actions
- Last 5 actions: {[a.action_type.name for a in context.action_history[-5:]] if context.action_history else "none"}

## Available Actions
- forward: move forward
- turn_left: turn left
- turn_right: turn right
- stop: stop

## Output Format
Strictly output in the following JSON format:
```json
{{
  "primary_action": "forward/turn_left/turn_right/stop",
  "confidence": 0.0-1.0,
  "reasoning": "recommendation reason",
  "constraints": {{
    "hard": [{{"action": "action", "blocked": true, "reason": "reason"}}],
    "soft": [{{"action": "action", "weight": 0.5, "reason": "reason"}}]
  }}
}}
```
"""
        try:
            response = self._model_manager.generate(
                "qwen-9b-perception",
                prompt,
                max_new_tokens=150,
                temperature=0.1,
            )
            return self._parse_perception_opinion_response(response, scene_description)
        except Exception as e:
            self.logger.error(f"[Perception] LLM perception opinion failed: {e}")
            return self._build_perception_opinion_fallback(scene_description)

    def _parse_perception_opinion_response(
        self,
        response: str,
        scene_description: str
    ) -> "DebateOpinion":
        """解析LLM响应为DebateOpinion对象。

        Args:
            response: LLM原始响应文本
            scene_description: 场景描述

        Returns:
            解析后的DebateOpinion对象
        """
        import json
        import re
        from core.debate_types import DebateOpinion, ActionConstraint

        primary_action = "forward"
        confidence = 0.6
        reasoning = ""
        constraints = {"hard": [], "soft": []}

        # Action name normalization
        action_normalize = {
            "move_forward": "forward",
            "left": "turn_left",
            "right": "turn_right",
        }

        try:
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response)
            if json_match:
                data = json.loads(json_match.group())

                primary_action = data.get("primary_action", "forward")
                confidence = float(data.get("confidence", 0.6))
                reasoning = data.get("reasoning", "")

                # Normalize action name
                primary_action = action_normalize.get(primary_action, primary_action)

                hard = data.get("constraints", {}).get("hard", [])
                soft = data.get("constraints", {}).get("soft", [])

                for c in hard:
                    action = c.get("action", "forward")
                    action = action_normalize.get(action, action)
                    constraints["hard"].append(ActionConstraint(
                        action=action,
                        blocked=c.get("blocked", True),
                        reason=c.get("reason", ""),
                    ))

                for c in soft:
                    action = c.get("action", "forward")
                    action = action_normalize.get(action, action)
                    constraints["soft"].append(ActionConstraint(
                        action=action,
                        weight_multiplier=c.get("weight", 1.0),
                        reason=c.get("reason", ""),
                    ))

        except (json.JSONDecodeError, ValueError):
            pass

        return DebateOpinion(
            agent="perception",
            primary_action=primary_action,
            confidence=confidence,
            evidence={"scene_description": scene_description},
            reasoning=reasoning or "Perception analysis",
            constraints=constraints,
        )

    def _build_perception_opinion_fallback(
        self,
        scene_description: str
    ) -> "DebateOpinion":
        """当LLM不可用时的感知观点fallback方法。

        Args:
            scene_description: 场景描述

        Returns:
            基于规则的DebateOpinion对象
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        constraints = {"hard": [], "soft": []}
        primary_action = "forward"
        confidence = 0.5

        # Simple rule: check for obstacle keywords in description
        if scene_description:
            obstacle_keywords = ["blocked", "obstacle", "wall ahead", "barrier"]
            if any(kw in scene_description.lower() for kw in obstacle_keywords):
                constraints["hard"].append(ActionConstraint(
                    action="forward",
                    blocked=True,
                    reason="obstacle_detected",
                ))
                primary_action = "turn_right"
                confidence = 0.6

        return DebateOpinion(
            agent="perception",
            primary_action=primary_action,
            confidence=confidence,
            evidence={"scene_description": scene_description},
            reasoning="Perception analysis (no LLM)",
            constraints=constraints,
        )