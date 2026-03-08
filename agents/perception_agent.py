"""Perception Agent for analyzing visual observations.

This version uses:
- YOLOv5s: Object detection (local, fast, compatible with numpy 2.x)
- Qwen3.5-2B: Structured visual descriptions (local, INT8)
"""

from typing import Dict, Any, Optional, List
import logging
import numpy as np

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext


class PerceptionAgent(BaseAgent):
    """
    Agent responsible for visual perception and environment understanding.

    Uses:
    - YOLOv5s: Object detection (local, fast, compatible with numpy 2.x)
    - Qwen3.5-2B: Structured visual descriptions (local, INT8)

    Key responsibilities:
    1. Detect objects and landmarks using YOLO
    2. Estimate distances using depth
    3. Generate LLM-enhanced visual descriptions
    4. Match objects with instruction landmarks
    """

    # Room type keywords
    ROOM_KEYWORDS = {
        "bedroom": ["bed", "nightstand", "dresser", "wardrobe", "pillow", "blanket"],
        "bathroom": ["toilet", "sink", "shower", "bathtub", "towel", "mirror"],
        "kitchen": ["refrigerator", "stove", "oven", "sink", "counter", "cabinet", "microwave"],
        "living_room": ["sofa", "couch", "television", "tv", "coffee table", "fireplace", "armchair"],
        "dining_room": ["dining table", "chair", "sideboard"],
        "hallway": ["door", "corridor", "passage"],
        "office": ["desk", "computer", "chair", "bookshelf", "monitor"],
        "garage": ["car", "tool", "workbench"],
    }

    # Object categories relevant for navigation
    NAVIGATION_OBJECTS = {
        "door", "stairs", "staircase", "corridor", "hallway",
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
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.max_objects = self.config.get("max_objects", 10)
        self.use_llm = self.config.get("use_llm", True)

        # Model references
        self._model_manager = None
        self._initialized = False

        # LLM conversation history (independent instance)
        self._conversation_history: List[Dict[str, str]] = []

    @property
    def name(self) -> str:
        return "perception_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.PERCEPTION

    def get_required_inputs(self) -> List[str]:
        return ["rgb_image", "depth_image"]

    def get_output_keys(self) -> List[str]:
        return ["room_type", "objects", "landmarks", "scene_description"]

    def initialize(self) -> None:
        """Initialize visual encoder and model manager."""
        if self._initialized:
            return

        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)

            # Check if using remote LLM
            use_remote = self.config.get("use_remote", False)

            if use_remote:
                self.logger.info("Using remote LLM service for perception")
                self._model_manager.load_all_models()  # Only loads YOLO locally
            else:
                self._model_manager.load_all_models()

                # Load Qwen3.5-2B for perception if LLM is enabled
                if self.use_llm:
                    self.logger.info("Loading Qwen3.5-2B for perception...")
                    if self._model_manager.load_llm("qwen-2b-perception"):
                        self.logger.info("Qwen3.5-2B (perception) loaded successfully")
                else:
                    self.logger.warning("Failed to load Qwen3.5-2B, using template-based descriptions")
                    self.use_llm = False

            self._initialized = True
            self.logger.info("PerceptionAgent initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model manager: {e}")
            self._initialized = True

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Process visual observations.

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
            depth_image = self._get_depth_image(context)

            # Detect objects using YOLO
            objects = self._detect_objects(rgb_image, depth_image)

            # Classify room
            room_type, room_confidence = self._classify_room(objects)

            # Match landmarks with instruction
            landmarks = self._match_landmarks(context, objects)

            # Generate template-based description
            scene_description = self._generate_description(objects, room_type, landmarks)

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
            }

            return AgentOutput.success_output(
                data={
                    "room_type": room_type,
                    "room_confidence": room_confidence,
                    "objects": objects,
                    "landmarks": landmarks,
                    "scene_description": scene_description,
                    "num_objects": len(objects),
                    "num_landmarks": len(landmarks),
                },
                confidence=room_confidence,
                reasoning=f"Detected {len(objects)} objects in {room_type}, {len(landmarks)} landmarks matched",
            )

        except Exception as e:
            self.logger.error(f"Perception error: {e}")
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

    def _detect_objects(
        self,
        rgb_image: Optional[np.ndarray],
        depth_image: Optional[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Detect objects using YOLO and estimate distances."""
        objects = []

        if rgb_image is None:
            return objects

        # Use model manager for detection
        if self._model_manager:
            detections = self._model_manager.detect_objects(
                rgb_image,
                confidence_threshold=self.confidence_threshold
            )

            for det in detections[:self.max_objects]:
                obj = {
                    "name": det["name"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                }

                # Estimate distance from depth
                if depth_image is not None:
                    distance = self._model_manager.estimate_distance(
                        depth_image, det["bbox"]
                    )
                    obj["distance"] = distance
                else:
                    obj["distance"] = 0.0

                # Estimate angle
                if rgb_image is not None and len(rgb_image.shape) == 3:
                    h, w = rgb_image.shape[:2]
                    angle = self._model_manager.estimate_angle(det["bbox"], w)
                    obj["angle"] = angle
                else:
                    obj["angle"] = 0.0

                # Mark if it's a navigation-relevant object
                obj["is_navigation_object"] = det["name"].lower() in self.NAVIGATION_OBJECTS
                obj["is_landmark"] = det["name"].lower() in self.LANDMARK_OBJECTS

                objects.append(obj)

        return objects

    def _classify_room(self, objects: List[Dict]) -> tuple:
        """Classify room type based on detected objects."""
        if not objects:
            return "unknown", 0.3

        object_names = [obj.get("name", "").lower() for obj in objects]

        best_room = "unknown"
        best_score = 0

        for room_type, keywords in self.ROOM_KEYWORDS.items():
            score = sum(1 for kw in keywords if any(kw in obj for obj in object_names))
            if score > best_score:
                best_score = score
                best_room = room_type

        confidence = min(best_score / 3.0, 1.0) if best_score > 0 else 0.3
        return best_room, confidence

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

    def _generate_description(
        self,
        objects: List[Dict],
        room_type: str,
        landmarks: List[Dict],
    ) -> str:
        """Generate visual description using LLM or template-based fallback."""
        # Try LLM-enhanced description first
        if self.use_llm and self._model_manager:
            llm_description = self._generate_llm_description(objects, room_type, landmarks)
            if llm_description:
                return llm_description

        # Fallback to template-based description
        return self._generate_template_description(objects, room_type, landmarks)

    def _generate_llm_description(
        self,
        objects: List[Dict],
        room_type: str,
        landmarks: List[Dict],
    ) -> str:
        """Generate LLM-enhanced visual description."""
        if not self._model_manager:
            return ""

        try:
            # Build prompt for LLM
            prompt = self._build_perception_prompt(objects, room_type, landmarks)

            # Generate description
            response = self._model_manager.generate(
                "qwen-2b-perception",
                prompt,
                max_new_tokens=200,
                temperature=0.3
            )

            if response:
                # Update conversation history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                # Keep only recent history
                if len(self._conversation_history) > 10:
                    self._conversation_history = self._conversation_history[-10:]

                return response

        except Exception as e:
            self.logger.warning(f"LLM description generation failed: {e}")

        return ""

    def _build_perception_prompt(
        self,
        objects: List[Dict],
        room_type: str,
        landmarks: List[Dict],
    ) -> str:
        """Build prompt for perception LLM."""
        # Format objects by direction
        front_objs = []
        left_objs = []
        right_objs = []

        for obj in objects:
            angle = obj.get("angle", 0)
            name = obj.get("name", "物体")
            dist = obj.get("distance", 0)

            if abs(angle) < 30:
                front_objs.append(f"{name}({dist:.1f}米)")
            elif angle < 0:
                left_objs.append(f"{name}({dist:.1f}米)")
            else:
                right_objs.append(f"{name}({dist:.1f}米)")

        # Format landmarks
        landmark_str = ""
        if landmarks:
            lm_descs = []
            for lm in landmarks[:3]:
                direction = "前方"
                angle = lm.get("angle", 0)
                if angle < -30:
                    direction = "左侧"
                elif angle > 30:
                    direction = "右侧"
                lm_descs.append(f"{direction}{lm.get('distance', 0):.1f}米处有{lm.get('name')}")
            landmark_str = "目标地标: " + ", ".join(lm_descs)

        prompt = f"""你是一个室内导航助手。请用简洁的中文描述当前视觉场景，帮助导航决策。

场景信息:
- 房间类型: {room_type if room_type != "unknown" else "未知"}
- 前方物体: {', '.join(front_objs[:3]) if front_objs else '无'}
- 左侧物体: {', '.join(left_objs[:3]) if left_objs else '无'}
- 右侧物体: {', '.join(right_objs[:3]) if right_objs else '无'}
- {landmark_str}

请用1-2句话描述场景，重点关注:
1. 当前位置特征
2. 可用于导航的地标
3. 前进方向的情况

直接输出描述，不要其他内容。"""

        return prompt

    def _generate_template_description(
        self,
        objects: List[Dict],
        room_type: str,
        landmarks: List[Dict],
    ) -> str:
        """Generate template-based visual description (fallback)."""
        parts = []

        if room_type != "unknown":
            room_name = room_type.replace("_", " ")
            parts.append(f"当前在{room_name}")

        # Describe objects by direction
        if objects:
            front_objs = []
            left_objs = []
            right_objs = []

            for obj in objects:
                angle = obj.get("angle", 0)
                name = obj.get("name", "物体")
                dist = obj.get("distance", 0)

                if abs(angle) < 30:
                    front_objs.append(f"{name}({dist:.1f}米)")
                elif angle < 0:
                    left_objs.append(f"{name}({dist:.1f}米)")
                else:
                    right_objs.append(f"{name}({dist:.1f}米)")

            if front_objs:
                parts.append(f"前方: {', '.join(front_objs[:3])}")
            if left_objs:
                parts.append(f"左侧: {', '.join(left_objs[:3])}")
            if right_objs:
                parts.append(f"右侧: {', '.join(right_objs[:3])}")

        # Describe matched landmarks
        if landmarks:
            lm_descs = []
            for lm in landmarks[:3]:
                direction = "前方"
                angle = lm.get("angle", 0)
                if angle < -30:
                    direction = "左侧"
                elif angle > 30:
                    direction = "右侧"
                lm_descs.append(f"{direction}{lm.get('distance', 0):.1f}米处有{lm.get('name')}")
            parts.append(f"目标地标: {', '.join(lm_descs)}")

        return "。".join(parts) + "。" if parts else "场景分析中..."

    def check_for_obstacles(self, depth_image: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Check for obstacles in the path using depth image."""
        if depth_image is None:
            return {"has_obstacle": False}

        try:
            h, w = depth_image.shape[:2]
            center_region = depth_image[h//3:2*h//3, w//3:2*w//3]

            close_points = center_region[center_region > 0]
            if len(close_points) > 0:
                min_dist = float(np.min(close_points))
                mean_dist = float(np.mean(close_points))

                has_obstacle = min_dist < threshold

                return {
                    "has_obstacle": has_obstacle,
                    "min_distance": min_dist,
                    "mean_distance": mean_dist,
                    "center_clear": not has_obstacle,
                }
        except Exception as e:
            self.logger.warning(f"Obstacle check failed: {e}")

        return {"has_obstacle": False}