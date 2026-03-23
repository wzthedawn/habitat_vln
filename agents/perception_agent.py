"""Perception Agent for analyzing visual observations.

This version uses VLM (Qwen3.5-4B) for all perception tasks:
- RGB + Depth image fusion analysis
- Object detection and scene description
- Stair/obstacle detection via LLM
- Room classification
- No rule-based detection, everything through LLM
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext


class PerceptionAgent(BaseAgent):
    """
    Agent responsible for visual perception using VLM only.

    Uses Qwen3.5-4B VLM for all perception tasks:
    - RGB + Depth fusion analysis
    - Object detection and scene description
    - Stair/obstacle detection
    - Room classification
    - Navigation hints generation

    No YOLO, no rule-based depth detection.
    """

    # Room type keywords
    ROOM_KEYWORDS = {
        "bedroom": ["bed", "nightstand", "dresser", "wardrobe", "pillow", "blanket"],
        "bathroom": ["toilet", "sink", "shower", "bathtub", "towel", "mirror"],
        "kitchen": ["refrigerator", "stove", "oven", "sink", "counter", "cabinet", "microwave"],
        "living_room": ["sofa", "couch", "television", "tv", "coffee table", "fireplace", "armchair"],
        "dining_room": ["dining table", "chair", "sideboard"],
        "hallway": ["door", "corridor", "passage", "wall"],
        "stairs": ["stairs", "staircase", "steps", "railing", "banister", "handrail"],
        "office": ["desk", "computer", "chair", "bookshelf", "monitor"],
        "garage": ["car", "tool", "workbench"],
    }

    # Object categories relevant for navigation
    NAVIGATION_OBJECTS = {
        "door", "stairs", "staircase", "steps", "corridor", "hallway",
        "entrance", "exit", "wall", "floor", "ceiling",
    }

    # Furniture that can serve as landmarks
    LANDMARK_OBJECTS = {
        "chair", "table", "desk", "bed", "sofa", "couch",
        "cabinet", "shelf", "bookshelf", "refrigerator", "tv",
        "piano", "bench", "rug", "carpet",
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("PerceptionAgent")

        # Configuration
        self.max_objects = self.config.get("max_objects", 10)
        self.use_depth = self.config.get("use_depth", True)  # 是否使用深度图

        # Model references
        self._model_manager = None
        self._initialized = False

    # Default values for perception output
    DEFAULT_WALKABLE = {
        "left": {"clear": True, "depth_m": 0},
        "center": {"clear": True, "depth_m": 0},
        "right": {"clear": True, "depth_m": 0},
        "recommended": "center"
    }
    DEFAULT_OBSTACLE = {"blocked": False, "min_distance": 5.0}

    @property
    def name(self) -> str:
        return "perception_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.PERCEPTION

    def get_required_inputs(self) -> List[str]:
        if self.use_depth:
            return ["rgb_image", "depth_image"]
        return ["rgb_image"]

    def get_output_keys(self) -> List[str]:
        return ["room_type", "objects", "landmarks", "scene_description"]

    def initialize(self) -> None:
        """Initialize model manager for VLM."""
        if self._initialized:
            return

        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)
            self._initialized = True
            self.logger.info(f"PerceptionAgent initialized (VLM-only mode, {'RGB+Depth' if self.use_depth else 'RGB-only'})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model manager: {e}")
            self._initialized = True

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Process visual observations using VLM only.

        RGB + Depth images are fused and sent to LLM for analysis.
        No rule-based detection, everything through LLM.

        Args:
            context: Navigation context with visual features
            strategy_result: Optional strategy output

        Returns:
            AgentOutput with perception results
        """
        self.initialize()

        try:
            # Get images from context
            rgb_image = self._get_rgb_image(context)
            # Only get depth image if use_depth is enabled
            depth_image = self._get_depth_image(context) if self.use_depth else None
            has_depth = depth_image is not None

            # Use VLM for all perception (RGB + Depth fusion)
            vlm_result = None
            if self._model_manager:
                vlm_result = self._analyze_with_vlm(rgb_image, depth_image)

            if vlm_result:
                room_type = vlm_result.get("room_type", "unknown")
                room_confidence = vlm_result.get("room_confidence", 0.7)
                objects = vlm_result.get("objects", [])
                scene_brief = vlm_result.get("scene_brief", vlm_result.get("scene_description", ""))
                nav_hint = vlm_result.get("nav_hint", "")
                walkable = vlm_result.get("walkable_analysis", self.DEFAULT_WALKABLE)
                obstacle = vlm_result.get("obstacle_ahead", self.DEFAULT_OBSTACLE)
            else:
                room_type = "unknown"
                room_confidence = 0.3
                objects = []
                scene_brief = "无法获取视觉信息"
                nav_hint = ""
                walkable = self.DEFAULT_WALKABLE
                obstacle = self.DEFAULT_OBSTACLE

            # 限制场景描述在50字以内
            if len(scene_brief) > 50:
                scene_brief = scene_brief[:50]

            # Match landmarks with instruction
            landmarks = self._match_landmarks(context, objects)

            # 格式化物体信息
            formatted_objects = []
            for obj in objects[:8]:  # 最多8个物体
                if has_depth:
                    formatted_objects.append({
                        "物体": obj.get("name", "未知"),
                        "距离": f"{obj.get('distance', 0):.1f}m",
                        "角度": f"{obj.get('angle', 0):.0f}°"
                    })
                else:
                    formatted_objects.append({
                        "物体": obj.get("name", "未知")
                    })

            # 构建场景描述
            scene_description = f"{room_type}，{scene_brief}" if room_type != "unknown" else scene_brief

            # Update context
            context.room_type = room_type
            context.visual_features.object_detections = objects
            context.visual_features.room_classification = room_type
            context.visual_features.scene_description = scene_description

            # Store in metadata for other agents
            context.metadata["perception_output"] = {
                "room_type": room_type,
                "room_confidence": room_confidence,
                "objects": objects,
                "landmarks": landmarks,
                "scene_description": scene_description,
                "nav_hint": nav_hint,
                "walkable_analysis": walkable,
                "obstacle_ahead": obstacle,
            }

            return AgentOutput.success_output(
                data={
                    "room_type": room_type,
                    "room_confidence": room_confidence,
                    "objects": formatted_objects,
                    "landmarks": landmarks,
                    "scene_description": scene_description,
                    "nav_hint": nav_hint,
                    "num_objects": len(objects),
                    "num_landmarks": len(landmarks),
                    "walkable_analysis": walkable,
                    "obstacle_ahead": obstacle,
                },
                confidence=room_confidence,
                reasoning=f"{scene_description}，检测到{len(objects)}个物体",
            )

        except Exception as e:
            self.logger.error(f"[Perception] 错误: {e}")
            return AgentOutput.failure_output([str(e)], "Perception processing failed")

    def _get_rgb_image(self, context: NavContext) -> Optional[np.ndarray]:
        """Get RGB image from context."""
        if hasattr(context, 'rgb_image') and context.rgb_image is not None:
            return context.rgb_image
        if context.metadata.get('rgb_image') is not None:
            return context.metadata['rgb_image']
        return None

    def _get_depth_image(self, context: NavContext) -> Optional[np.ndarray]:
        """Get depth image from context."""
        if hasattr(context, 'depth_image') and context.depth_image is not None:
            return context.depth_image
        if context.metadata.get('depth_image') is not None:
            return context.metadata['depth_image']
        return None

    def _analyze_with_vlm(
        self,
        rgb_image: Optional[np.ndarray],
        depth_image: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze RGB + Depth images using VLM.

        Returns:
            Dictionary with room_type, objects, scene_description, nav_hint, walkable_analysis, obstacle_ahead
        """
        if not self._model_manager:
            return None

        try:
            has_depth = depth_image is not None

            if has_depth:
                # RGB + Depth: Full perception with distance info
                prompt = """你是室内导航机器人的视觉感知系统。

## 图像说明
- 第一张: RGB彩色图
- 第二张: 深度图(红色=近,蓝色=远)

## 任务
分析图像，输出JSON格式的感知结果。

## 输出格式
{"room_type":"房间类型","scene_brief":"场景描述","objects":[{"name":"物体","distance":1.0,"angle":0}],"stairs":{"detected":false,"direction":"none","distance":5},"obstacle_ahead":{"blocked":false,"min_distance":3.0},"nav_hint":"导航提示"}"""
            else:
                # RGB-only: Simplified perception without distance requirements
                prompt = """你是室内导航机器人的视觉感知系统。

## 任务
分析RGB图像，输出JSON格式的感知结果。

## 输出格式
{"room_type":"房间类型","scene_brief":"场景描述","objects":[{"name":"物体"}],"nav_hint":"导航提示"}"""

            # Call VLM with RGB + Depth images
            if depth_image is not None:
                result = self._model_manager.generate_vision_dual(
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    prompt=prompt,
                    model_key="qwen-4b-perception",
                    max_new_tokens=400,
                    temperature=0.2,
                )
            else:
                result = self._model_manager.generate_vision(
                    image=rgb_image,
                    prompt=prompt,
                    model_key="qwen-4b-perception",
                    max_new_tokens=400,
                    temperature=0.2,
                )

            if not result or not result.get("response"):
                self.logger.warning("[Perception] VLM无响应")
                return None

            raw_response = result["response"]
            self.logger.info(f"[Perception] VLM响应: {raw_response[:150]}...")

            # Parse VLM response
            parsed = self._parse_vlm_response(raw_response)

            if parsed:
                self.logger.info(f"[Perception] 房间:{parsed.get('room_type', 'unknown')}, 物体:{len(parsed.get('objects', []))}")

            return parsed

        except Exception as e:
            self.logger.error(f"[Perception] VLM分析失败: {e}")
            return None

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into structured format.

        Args:
            response: Raw VLM response text

        Returns:
            Dictionary with parsed fields
        """
        import json
        import re

        result = {
            "room_type": "unknown",
            "room_confidence": 0.5,
            "objects": [],
            "scene_description": "",
            "nav_hint": "",
            "walkable_analysis": self.DEFAULT_WALKABLE,
            "obstacle_ahead": self.DEFAULT_OBSTACLE,
        }

        object_names = []

        # Try parsing JSON format (support nested objects)
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # Parse room type
                if "room_type" in data:
                    result["room_type"] = self._normalize_room_type(data["room_type"])
                    result["room_confidence"] = 0.7

                # Parse walkable analysis
                if "walkable_analysis" in data:
                    wa = data["walkable_analysis"]
                    result["walkable_analysis"] = {
                        "left": wa.get("left", {"clear": True, "depth_m": 0}),
                        "center": wa.get("center", {"clear": True, "depth_m": 0}),
                        "right": wa.get("right", {"clear": True, "depth_m": 0}),
                        "recommended": wa.get("recommended", "center")
                    }

                # Parse objects
                if "objects" in data and isinstance(data["objects"], list):
                    for obj in data["objects"][:self.max_objects]:
                        if isinstance(obj, dict):
                            name = obj.get("name", "")
                            if name:
                                object_names.append(name)
                                result["objects"].append({
                                    "name": name,
                                    "confidence": 0.6,
                                    "distance": float(obj.get("distance", 0)),
                                    "angle": float(obj.get("angle", 0)),
                                    "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                                    "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                                    "source": "vlm",
                                })

                # Parse obstacle
                if "obstacle_ahead" in data:
                    obs = data["obstacle_ahead"]
                    if isinstance(obs, dict):
                        result["obstacle_ahead"] = obs
                    elif isinstance(obs, bool):
                        result["obstacle_ahead"] = {"blocked": obs, "min_distance": 5.0}

                # Parse description and hint
                result["scene_description"] = data.get("scene_brief", data.get("scene_description", ""))
                result["nav_hint"] = data.get("nav_hint", "")

                # Infer room type if not provided
                if result["room_type"] == "unknown" and object_names:
                    result["room_type"] = self._infer_room_from_objects(object_names)

                # Generate scene description if not provided
                if not result["scene_description"] and object_names:
                    result["scene_description"] = f"可见: {', '.join(object_names[:5])}"

                return result

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self.logger.debug(f"[Perception] JSON解析失败: {e}")

        # Fallback: Generate scene description
        if result["room_type"] != "unknown" and object_names:
            result["scene_description"] = f"当前在{result['room_type']}，可见: {', '.join(object_names[:5])}"
        else:
            result["scene_description"] = "场景分析中..."

        return result

    def _normalize_room_type(self, room_str: str) -> str:
        """Normalize room type string to standard format."""
        room_str = room_str.lower().strip()

        # Map common variations
        room_mapping = {
            "卧室": "bedroom",
            "浴室": "bathroom",
            "厨房": "kitchen",
            "客厅": "living_room",
            "餐厅": "dining_room",
            "走廊": "hallway",
            "楼梯": "stairs",
            "办公室": "office",
            "车库": "garage",
            "bedroom": "bedroom",
            "bathroom": "bathroom",
            "kitchen": "kitchen",
            "living room": "living_room",
            "livingroom": "living_room",
            "dining room": "dining_room",
            "diningroom": "dining_room",
            "hallway": "hallway",
            "stairs": "stairs",
            "staircase": "stairs",
            "office": "office",
            "garage": "garage",
        }

        for key, value in room_mapping.items():
            if key in room_str:
                return value

        return "unknown"

    def _infer_room_from_objects(self, object_names: List[str]) -> str:
        """Infer room type from detected object names.

        Args:
            object_names: List of detected object names

        Returns:
            Inferred room type string
        """
        if not object_names:
            return "unknown"

        # Convert to lowercase for matching
        obj_lower = [name.lower() for name in object_names]

        # Check each room's keywords
        for room_type, keywords in self.ROOM_KEYWORDS.items():
            for keyword in keywords:
                if any(keyword in obj or obj in keyword for obj in obj_lower):
                    return room_type

        # Check for navigation objects that indicate stairs
        stairs_keywords = ["stairs", "staircase", "steps", "railing", "banister", "楼梯"]
        if any(any(kw in obj for kw in stairs_keywords) for obj in obj_lower):
            return "stairs"

        return "unknown"

    def _match_landmarks(
        self, context: NavContext, objects: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Match detected objects with instruction landmarks."""
        landmarks = []

        # Get expected landmarks from instruction/subtasks
        expected_landmarks = set()

        if context.subtasks:
            for subtask in context.subtasks:
                description = subtask.description.lower()
                for landmark in self.LANDMARK_OBJECTS:
                    if landmark in description:
                        expected_landmarks.add(landmark)

        # Also check instruction directly
        instruction_lower = context.instruction.lower()
        for landmark in self.LANDMARK_OBJECTS:
            if landmark in instruction_lower:
                expected_landmarks.add(landmark)

        # Check for room names as landmarks
        for room in self.ROOM_KEYWORDS.keys():
            if room.replace("_", " ") in instruction_lower:
                expected_landmarks.add(room)

        # Match with detected objects
        for obj in objects:
            obj_name = obj.get("name", "").lower()
            if obj_name in expected_landmarks or any(lm in obj_name for lm in expected_landmarks):
                landmarks.append({
                    "name": obj_name,
                    "matched": True,
                    "distance": obj.get("distance", 0.0),
                    "angle": obj.get("angle", 0.0),
                    "confidence": obj.get("confidence", 0.8),
                    "bbox": obj.get("bbox"),
                })

        return landmarks

    def build_debate_opinion(
        self,
        context: NavContext,
        depth_info: Dict[str, Any] = None
    ) -> "DebateOpinion":
        """Build a DebateOpinion using LLM for perception analysis.

        Args:
            context: Navigation context
            depth_info: Ignored (VLM does all analysis)

        Returns:
            DebateOpinion with perception-based constraints
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        # Get perception output from context (already processed by VLM)
        perception_output = context.metadata.get("perception_output", {})
        obstacle_ahead = perception_output.get("obstacle_ahead", {})
        landmarks = perception_output.get("landmarks", [])
        landmark_names = [lm.get("name", "") for lm in landmarks[:3]]

        # Build obstacles list from obstacle_ahead
        obstacles = []
        if obstacle_ahead.get("blocked"):
            obstacles.append({
                "distance": obstacle_ahead.get("min_distance", 1.0),
                "blocked": True
            })

        # Use LLM for perception-based opinion
        if self._model_manager:
            return self._build_perception_opinion_with_llm(
                context, obstacles, landmark_names
            )
        else:
            return self._build_perception_opinion_fallback(
                obstacles, landmark_names
            )

    def _build_perception_opinion_with_llm(
        self,
        context: NavContext,
        obstacles: List[Dict],
        landmarks: List[str]
    ) -> "DebateOpinion":
        """Build perception opinion using LLM."""
        from core.debate_types import DebateOpinion, ActionConstraint

        prompt = f"""基于感知信息分析最佳导航动作。

## 环境感知
- 障碍物: {obstacles if obstacles else "无"}
- 可见地标: {landmarks if landmarks else "无"}
- 场景描述: {context.visual_features.scene_description[:100] if context.visual_features.scene_description else "无"}
- 房间类型: {context.room_type}

## 最近动作
- 最近5个动作: {[a.action_type.name for a in context.action_history[-5:]] if context.action_history else "无"}

## 可选动作
- forward: 向前移动
- turn_left: 向左转
- turn_right: 向右转
- stop: 停止

## 输出格式
严格按照以下JSON格式输出:
```json
{{
  "primary_action": "forward/turn_left/turn_right/stop",
  "confidence": 0.0-1.0,
  "reasoning": "推荐理由",
  "constraints": {{
    "hard": [{{"action": "动作", "blocked": true, "reason": "原因"}}],
    "soft": [{{"action": "动作", "weight": 0.5, "reason": "原因"}}]
  }}
}}
```
"""
        try:
            response = self._model_manager.generate(
                "qwen-4b-perception",
                prompt,
                max_new_tokens=150,
                temperature=0.1,
            )
            return self._parse_perception_opinion_response(response, obstacles, landmarks)
        except Exception as e:
            self.logger.error(f"LLM perception opinion failed: {e}")
            return self._build_perception_opinion_fallback(obstacles, landmarks)

    def _parse_perception_opinion_response(
        self,
        response: str,
        obstacles: List[Dict],
        landmarks: List[str]
    ) -> "DebateOpinion":
        """Parse LLM response into DebateOpinion."""
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
            evidence={
                "obstacles": obstacles,
                "landmarks": landmarks,
                "obstacle_ahead": len(obstacles) > 0,
            },
            reasoning=reasoning or f"检测到{len(obstacles)}个障碍物",
            constraints=constraints,
        )

    def _build_perception_opinion_fallback(
        self,
        obstacles: List[Dict],
        landmarks: List[str]
    ) -> "DebateOpinion":
        """Fallback perception opinion when LLM unavailable."""
        from core.debate_types import DebateOpinion, ActionConstraint

        constraints = {"hard": [], "soft": []}
        primary_action = "forward"
        confidence = 0.5

        if obstacles:
            min_dist = min(o.get("distance", 999) for o in obstacles)
            if min_dist < 0.5:
                constraints["hard"].append(ActionConstraint(
                    action="forward",
                    blocked=True,
                    reason=f"obstacle_{min_dist:.1f}m",
                ))
                primary_action = "turn_right"
                confidence = 0.7

        return DebateOpinion(
            agent="perception",
            primary_action=primary_action,
            confidence=confidence,
            evidence={
                "obstacles": obstacles,
                "landmarks": landmarks,
                "obstacle_ahead": len(obstacles) > 0,
            },
            reasoning="感知分析(无LLM)",
            constraints=constraints,
        )