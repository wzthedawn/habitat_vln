"""Perception Agent for analyzing visual observations.

This version uses:
- VLM (Qwen3.5-4B): Multimodal object detection and scene description via remote server
- Qwen3.5-4B: Text-based scene enhancement (remote, INT8)
- YOLOv5s: Fallback object detection (local, if available)
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
    - VLM (Qwen3.5-4B): Multimodal object detection and scene description (remote)
    - Qwen3.5-4B: Text-based scene enhancement (remote)
    - YOLOv5s: Fallback object detection (local, if available)

    Key responsibilities:
    1. Detect objects and landmarks using VLM (or YOLO fallback)
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
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.max_objects = self.config.get("max_objects", 10)
        self.use_llm = self.config.get("use_llm", True)
        self.use_vlm = self.config.get("use_vlm", True)  # Enable VLM by default

        # Model references
        self._model_manager = None
        self._initialized = False

        # LLM conversation history (independent instance)
        self._conversation_history: List[Dict[str, str]] = []

        # VLM prompt template for object detection - 规范化JSON格式
        self._vlm_prompt = """分析这张室内场景图像，识别可见物体。

输出JSON格式:
{
  "物体": ["物体1", "物体2"],
  "距离": [距离1, 距离2],
  "角度": [角度1, 角度2]
}

距离单位为米，角度为相对于视野中心的角度(-180到180度)。
只输出JSON，不要输出其他内容。"""

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

                # Load Qwen3.5-4B for perception if LLM is enabled (方案二)
                if self.use_llm:
                    self.logger.info("Loading Qwen3.5-4B for perception...")
                    if self._model_manager.load_llm("qwen-4b-perception"):
                        self.logger.info("Qwen3.5-4B (perception) loaded successfully")
                else:
                    self.logger.warning("Failed to load Qwen3.5-4B, using template-based descriptions")
                    self.use_llm = False

            self._initialized = True
            self.logger.info("PerceptionAgent initialized (VLM enabled, YOLO fallback)")
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

            # Try VLM-based detection first
            vlm_result = None
            if self.use_vlm and rgb_image is not None and self._model_manager:
                vlm_result = self._detect_with_vlm(rgb_image, context)

            # Detect objects using YOLO (fallback or supplementary)
            objects = self._detect_objects(rgb_image, depth_image)

            # Merge VLM results if available
            if vlm_result:
                objects = self._merge_vlm_objects(objects, vlm_result.get("objects", []))
                room_type = vlm_result.get("room_type", "unknown")
                room_confidence = vlm_result.get("room_confidence", 0.7)
                scene_description = vlm_result.get("scene_description", "")
                nav_hint = vlm_result.get("nav_hint", "")
            else:
                # Classify room from YOLO objects
                room_type, room_confidence = self._classify_room(objects)
                scene_description = ""
                nav_hint = ""

            # Match landmarks with instruction
            landmarks = self._match_landmarks(context, objects)

            # Analyze depth for stair detection
            depth_info = None
            if depth_image is not None:
                depth_info = self._analyze_depth_for_stairs(depth_image)

            # Generate scene description if not from VLM
            if not scene_description:
                scene_description = self._generate_description(objects, room_type, landmarks, context, depth_info)

            # Store navigation hint
            if nav_hint and context:
                context.metadata["nav_hint"] = nav_hint

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
            }

            return AgentOutput.success_output(
                data={
                    "room_type": room_type,
                    "room_confidence": room_confidence,
                    "objects": objects,
                    "landmarks": landmarks,
                    "scene_description": scene_description,
                    "nav_hint": nav_hint,
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

    def _detect_with_vlm(
        self,
        rgb_image: np.ndarray,
        context: NavContext = None,
    ) -> Optional[Dict[str, Any]]:
        """Detect objects and generate scene description using VLM.

        Args:
            rgb_image: RGB image as numpy array
            context: Navigation context (optional, for additional context)

        Returns:
            Dictionary with room_type, objects, scene_description, nav_hint
        """
        if not self._model_manager:
            return None

        try:
            # Get current subtask for context-aware prompt
            current_subtask = ""
            if context and context.current_subtask_idx < len(context.subtasks):
                current_subtask = context.subtasks[context.current_subtask_idx].description

            # Build VLM prompt
            prompt = self._vlm_prompt
            if current_subtask:
                prompt += f"\n当前导航目标: {current_subtask}"

            # Call VLM through model manager
            # Use qwen-4b-perception (Qwen3.5-4B multimodal) for vision understanding
            result = self._model_manager.generate_vision(
                image=rgb_image,
                prompt=prompt,
                model_key="qwen-4b-perception",
                max_new_tokens=150,
                temperature=0.3,
            )

            if not result or not result.get("response"):
                self.logger.warning("[VLM] No response from VLM")
                return None

            # Debug: log raw VLM response
            raw_response = result["response"]
            self.logger.info(f"[VLM] Raw response: {raw_response[:200]}...")

            # Parse VLM response
            parsed = self._parse_vlm_response(raw_response)
            self.logger.info(f"[VLM] Detected room: {parsed['room_type']}, objects: {len(parsed['objects'])}")

            return parsed

        except Exception as e:
            self.logger.warning(f"[VLM] Detection failed: {e}")
            return None

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into structured format.

        Supports multiple response formats:
        1. 规范化JSON format:
           {"物体": [...], "距离": [...], "角度": [...]}
        2. Markdown列表格式:
           1. **楼梯:** 描述
           2. **门 (Door):** 描述
        3. Legacy JSON format:
           {"scene": "room_type", "objects": [...]}
        4. Keyword-based format:
           场景: xxx
           物体: xxx

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
        }

        object_names = []

        # 1. Try parsing new 规范化JSON format first
        # Match JSON with Chinese keys
        json_match = re.search(r'\{[^{}]*"物体"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                objects_list = data.get("物体", [])
                distances = data.get("距离", [])
                angles = data.get("角度", [])

                if isinstance(objects_list, list):
                    object_names = [str(obj).strip() for obj in objects_list if obj]
                    for i, name in enumerate(object_names[:self.max_objects]):
                        obj = {
                            "name": name,
                            "confidence": 0.6,
                            "distance": float(distances[i]) if i < len(distances) else 0.0,
                            "angle": float(angles[i]) if i < len(angles) else 0.0,
                            "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                            "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                            "source": "vlm",
                        }
                        result["objects"].append(obj)

                    # Infer room type from detected objects
                    result["room_type"] = self._infer_room_from_objects(object_names)
                    result["room_confidence"] = 0.7

                    # Generate scene description
                    if object_names:
                        result["scene_description"] = f"可见: {', '.join(object_names[:5])}"
                        # Add distance/angle info for navigation objects
                        nav_objs = [obj for obj in result["objects"]
                                   if obj["is_navigation_object"] and obj["distance"] > 0]
                        if nav_objs:
                            nav_info = ", ".join([f"{o['name']}({o['distance']:.1f}米)"
                                                 for o in nav_objs[:2]])
                            result["scene_description"] += f"。{nav_info}"

                self.logger.info(f"[PerceptionAgent] 物体: {object_names}, 距离: {distances}, 角度: {angles}")
                self.logger.debug(f"[VLM] New format JSON parsed successfully: {len(result['objects'])} objects")
                return result
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self.logger.debug(f"[VLM] New format JSON parse failed: {e}")

        # 2. Fallback: Parse Markdown list format
        # Matches patterns like: 1. **楼梯** or 1. **楼梯:**
        # or 1. **楼梯:** 描述
        if not result["objects"]:
            md_pattern = r'\d+\.\s*\*\*\s*([^(*\n]+?)(?:\s*\([^)]*\))?\s*\*\*'
            md_matches = re.findall(md_pattern, response)
            if md_matches:
                # Strip trailing colons and whitespace from names
                object_names = [name.strip().rstrip(':：').strip() for name in md_matches if name.strip()]
                for name in object_names[:self.max_objects]:
                    obj = {
                        "name": name,
                        "confidence": 0.6,
                        "distance": 0.0,
                        "angle": 0.0,
                        "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                        "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                        "source": "vlm",
                    }
                    result["objects"].append(obj)

                result["room_type"] = self._infer_room_from_objects(object_names)
                result["room_confidence"] = 0.7
                result["scene_description"] = f"可见: {', '.join(object_names[:5])}"

                self.logger.info(f"[VLM] Markdown格式解析成功: {object_names}")
                return result

        # 3. Fallback: Try old JSON format with "scene" and "objects" keys
        json_match = re.search(r'\{[^{}]*"scene"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                result["room_type"] = self._normalize_room_type(data.get("scene", "unknown"))
                result["room_confidence"] = 0.7

                # Parse objects from JSON
                objects_list = data.get("objects", [])
                if isinstance(objects_list, list):
                    object_names = [str(obj).strip() for obj in objects_list if obj]
                    for name in object_names[:self.max_objects]:
                        obj = {
                            "name": name,
                            "confidence": 0.6,
                            "distance": 0.0,
                            "angle": 0.0,
                            "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                            "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                            "source": "vlm",
                        }
                        result["objects"].append(obj)

                # Parse navigation hint
                result["nav_hint"] = data.get("nav_hint", "")

                self.logger.debug(f"[VLM] Legacy JSON parsed successfully: scene={result['room_type']}, objects={len(result['objects'])}")
            except json.JSONDecodeError as e:
                self.logger.debug(f"[VLM] JSON parse failed: {e}, falling back to keyword parsing")

        # 4. Fallback to keyword-based parsing if JSON/Markdown failed or no objects found
        if not result["objects"]:
            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Parse room type
                if line.startswith("场景:") or line.startswith("场景："):
                    room_str = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                    result["room_type"] = self._normalize_room_type(room_str)
                    result["room_confidence"] = 0.7

                # Parse objects
                elif line.startswith("物体:") or line.startswith("物体："):
                    obj_str = line.split(":", 1)[-1].split("：", 1)[-1].strip()
                    object_names = [obj.strip() for obj in obj_str.split(",") if obj.strip()]

                    for name in object_names[:self.max_objects]:
                        obj = {
                            "name": name,
                            "confidence": 0.6,  # VLM doesn't provide confidence
                            "distance": 0.0,  # Will be estimated from depth
                            "angle": 0.0,
                            "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                            "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                            "source": "vlm",
                        }
                        result["objects"].append(obj)

                # Parse navigation hint
                elif line.startswith("导航提示:") or line.startswith("导航提示："):
                    result["nav_hint"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()

        # 3. Generate scene description
        if result["room_type"] != "unknown" and object_names:
            result["scene_description"] = f"当前在{result['room_type']}，可见: {', '.join(object_names[:5])}。"
            if result["nav_hint"]:
                result["scene_description"] += f" {result['nav_hint']}"
        elif result["room_type"] != "unknown":
            result["scene_description"] = f"当前在{result['room_type']}。"
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

    def _merge_vlm_objects(
        self,
        yolo_objects: List[Dict],
        vlm_objects: List[Dict]
    ) -> List[Dict]:
        """Merge YOLO and VLM object detections.

        Prefers YOLO for bounding boxes and distances,
        adds VLM objects that weren't detected by YOLO.

        Args:
            yolo_objects: Objects detected by YOLO
            vlm_objects: Objects detected by VLM

        Returns:
            Merged list of objects
        """
        # Start with YOLO objects (they have bounding boxes)
        merged = list(yolo_objects)
        yolo_names = {obj["name"].lower() for obj in yolo_objects}

        # Add VLM objects not in YOLO results
        for vlm_obj in vlm_objects:
            name = vlm_obj["name"].lower()
            # Check if this object (or similar) was already detected
            if name not in yolo_names and not any(name in yn for yn in yolo_names):
                merged.append(vlm_obj)

        return merged[:self.max_objects]

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
        context: NavContext = None,
        depth_info: Dict = None,
    ) -> str:
        """Generate visual description using LLM or template-based fallback."""
        # Try LLM-enhanced description first
        if self.use_llm and self._model_manager:
            llm_description = self._generate_llm_description(objects, room_type, landmarks, context, depth_info)
            if llm_description:
                return llm_description

        # Fallback to template-based description
        return self._generate_template_description(objects, room_type, landmarks)

    def _generate_llm_description(
        self,
        objects: List[Dict],
        room_type: str,
        landmarks: List[Dict],
        context: NavContext = None,
        depth_info: Dict = None,
    ) -> str:
        """Generate LLM-enhanced visual description."""
        if not self._model_manager:
            return ""

        try:
            # Build prompt for LLM
            prompt = self._build_perception_prompt(objects, room_type, landmarks, depth_info)

            # Get episode_id for conversation context isolation
            episode_id = context.metadata.get("episode_id", 0) if context else 0
            conversation_id = f"perception_ep{episode_id}"

            # Generate description with conversation context
            response = self._model_manager.generate(
                "qwen-4b-perception",
                prompt,
                max_new_tokens=80,  # Increased from 50 for new format
                temperature=0.3,
                conversation_id=conversation_id,
                keep_context=True,
            )

            if response:
                # Clean up LLM output - remove thinking text and extra content
                response = self._clean_llm_output(response)

                # Parse new output format: "场景描述: ...\n导航提示: ..."
                scene_desc = response
                nav_hint = ""

                if "场景描述:" in response:
                    desc_part = response.split("场景描述:")[-1]
                    if "导航提示:" in desc_part:
                        scene_desc = desc_part.split("导航提示:")[0].strip()
                        nav_hint = desc_part.split("导航提示:")[-1].strip()
                    else:
                        scene_desc = desc_part.strip()

                    # Store navigation hint in context for other agents
                    if context and nav_hint and nav_hint != "无":
                        context.metadata["nav_hint"] = nav_hint
                        self.logger.info(f"[PERCEPTION] Nav hint: {nav_hint}")

                # Remove multiple newlines and keep only first paragraph
                if "\n\n" in scene_desc:
                    scene_desc = scene_desc.split("\n\n")[0].strip()

                # Remove single newlines
                scene_desc = scene_desc.replace("\n", " ")

                # Keep only the first sentence
                if "。" in scene_desc:
                    scene_desc = scene_desc.split("。")[0] + "。"
                elif "." in scene_desc:
                    # For English responses
                    parts = scene_desc.split(".")
                    if len(parts) > 1:
                        scene_desc = parts[0] + "."

                # Limit length to 60 characters
                if len(scene_desc) > 70:
                    scene_desc = scene_desc[:70]

                # Update conversation history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                # Keep only recent history
                if len(self._conversation_history) > 10:
                    self._conversation_history = self._conversation_history[-10:]

                return scene_desc

        except Exception as e:
            self.logger.warning(f"LLM description generation failed: {e}")

        return ""

    def _build_perception_prompt(
        self,
        objects: List[Dict],
        room_type: str,
        landmarks: List[Dict],
        depth_info: Dict = None,
    ) -> str:
        """Build prompt for perception LLM with vertical navigation awareness."""
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

        # Check for stairs-related objects
        stairs_detected = False
        for obj in objects:
            name = obj.get("name", "").lower()
            if name in ["stairs", "staircase", "steps", "stair", "step"]:
                stairs_detected = True
                break

        stairs_hint = ""
        if stairs_detected:
            stairs_hint = "\n注意: 检测到楼梯相关物体！"

        # NEW: Add depth-based stair detection info
        depth_str = ""
        if depth_info:
            if depth_info.get("stairs_detected"):
                direction = depth_info.get("stairs_direction", "unknown")
                depth_str = f"\n深度检测: 可能检测到{'上楼' if direction == 'up' else '下楼'}台阶"
            elif depth_info.get("floor_level_change"):
                depth_str = f"\n地面变化: {depth_info.get('floor_level_change')}"

        # === NEW: Enhanced prompt template ===
        prompt = f"""你是室内导航助手。根据视觉信息描述场景并推理导航要素。

## 视觉信息
- 房间类型: {room_type if room_type != "unknown" else "未知"}
- 前方物体: {', '.join(front_objs[:3]) if front_objs else '无'}
- 左侧物体: {', '.join(left_objs[:3]) if left_objs else '无'}
- 右侧物体: {', '.join(right_objs[:3]) if right_objs else '无'}
{landmark_str}{stairs_hint}{depth_str}

## 场景推理要求
1. 描述当前位置特征（1句话，不超过30字）
2. 如果发现楼梯、台阶、斜坡等垂直导航要素，明确指出
3. 如果房间类型暗示多层空间（楼梯间、走廊），说明可能需要上下楼
4. 检测前方是否有高度变化可能

输出格式:
场景描述: [一句话描述当前位置和前方情况]
导航提示: [如有楼梯/台阶/高度变化，说明方向和建议，否则写"无"]"""

        return prompt

    def _clean_llm_output(self, response: str) -> str:
        """Clean LLM output by removing thinking process and artifacts.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response
        """
        import re

        cleaned = response.strip()

        # Remove thinking process markers (Qwen3.5 style)
        # Pattern: numbered list with bold text like "1. **Analyze**:" or "2. **Input**:"
        thinking_patterns = [
            r'\d+\.\s*\*\*[^*]+\*\*:',  # "1. **Analyze...:"
            r'\d+\.\s*\*[^*]+\*:',       # "1. *Analyze...:"
            r'<think>.*?</think>',       # <think>...</think>
            r'```.*?```',                # code blocks
        ]

        for pattern in thinking_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)

        # Remove "Human:" and "Assistant:" markers
        if "Human:" in cleaned:
            cleaned = cleaned.split("Human:")[-1].strip()
        if "Assistant:" in cleaned:
            cleaned = cleaned.split("Assistant:")[-1].strip()

        # Remove leading numbers like "2. " at the start
        cleaned = re.sub(r'^\d+\.\s*', '', cleaned.strip())

        # Remove multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

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

    def analyze_depth_for_path(self, depth_image: np.ndarray) -> Dict[str, Any]:
        """Analyze depth image to find passable directions.

        Divides the depth image into left, center, and right regions,
        and calculates passability scores for each.

        Args:
            depth_image: Depth image as numpy array

        Returns:
            Dict with passable directions and recommendations
        """
        if depth_image is None:
            return {
                "left": {"passable": True, "avg_depth": 0, "min_depth": 0},
                "center": {"passable": True, "avg_depth": 0, "min_depth": 0},
                "right": {"passable": True, "avg_depth": 0, "min_depth": 0},
                "recommended_direction": "forward",
                "obstacle_ahead": False,
            }

        try:
            h, w = depth_image.shape[:2]

            # Divide into three regions
            left_region = depth_image[h//4:3*h//4, :w//3]
            center_region = depth_image[h//4:3*h//4, w//3:2*w//3]
            right_region = depth_image[h//4:3*h//4, 2*w//3:]

            # Calculate depth statistics for each region
            def analyze_region(region):
                valid = region[region > 0]
                if len(valid) == 0:
                    return {"passable": True, "avg_depth": 10.0, "min_depth": 10.0}

                avg_depth = float(np.mean(valid))
                min_depth = float(np.min(valid))

                # Consider passable if min depth > 0.5m
                passable = min_depth > 0.5

                return {
                    "passable": passable,
                    "avg_depth": avg_depth,
                    "min_depth": min_depth,
                }

            left_stats = analyze_region(left_region)
            center_stats = analyze_region(center_region)
            right_stats = analyze_region(right_region)

            # Determine recommended direction
            # Priority: center if passable, then side with more space
            recommended = "forward"
            obstacle_ahead = not center_stats["passable"]

            if obstacle_ahead:
                # Choose side with better passability
                if left_stats["passable"] and right_stats["passable"]:
                    # Both passable, choose one with more space
                    if left_stats["avg_depth"] > right_stats["avg_depth"]:
                        recommended = "left"
                    else:
                        recommended = "right"
                elif left_stats["passable"]:
                    recommended = "left"
                elif right_stats["passable"]:
                    recommended = "right"
                else:
                    # Both blocked, choose side with more space
                    if left_stats["min_depth"] > right_stats["min_depth"]:
                        recommended = "left"
                    else:
                        recommended = "right"

            return {
                "left": left_stats,
                "center": center_stats,
                "right": right_stats,
                "recommended_direction": recommended,
                "obstacle_ahead": obstacle_ahead,
            }

        except Exception as e:
            self.logger.warning(f"Depth path analysis failed: {e}")
            return {
                "left": {"passable": True, "avg_depth": 0, "min_depth": 0},
                "center": {"passable": True, "avg_depth": 0, "min_depth": 0},
                "right": {"passable": True, "avg_depth": 0, "min_depth": 0},
                "recommended_direction": "forward",
                "obstacle_ahead": False,
            }

    def _analyze_depth_for_stairs(self, depth_image: np.ndarray) -> Dict[str, Any]:
        """Analyze depth image to detect potential stairs.

        Looks for patterns in the depth image that might indicate stairs:
        - Horizontal step-like patterns
        - Ground level changes
        - Depth discontinuities

        Args:
            depth_image: Depth image as numpy array

        Returns:
            Dict with stair detection info
        """
        result = {
            "floor_level_change": None,
            "stairs_detected": False,
            "stairs_direction": None,
            "confidence": 0.0,
        }

        if depth_image is None:
            return result

        try:
            h, w = depth_image.shape[:2]

            # Method 1: Analyze bottom half of the image (where stairs would be visible)
            bottom_half = depth_image[h//2:, :]

            # Divide into horizontal strips
            num_strips = 6  # Increased for better resolution
            strip_height = bottom_half.shape[0] // num_strips
            strip_depths = []

            for i in range(num_strips):
                strip = bottom_half[i*strip_height:(i+1)*strip_height, :]
                valid_depths = strip[strip > 0]
                if len(valid_depths) > 0:
                    strip_depths.append(float(np.median(valid_depths)))  # Use median for robustness
                else:
                    strip_depths.append(0)

            # Check for step-like pattern (alternating depth levels)
            if len(strip_depths) >= 3:
                # Look for consistent depth changes between strips
                depth_changes = []
                for i in range(1, len(strip_depths)):
                    if strip_depths[i-1] > 0 and strip_depths[i] > 0:
                        change = strip_depths[i] - strip_depths[i-1]
                        depth_changes.append(change)

                # If there are consistent depth changes, might be stairs
                if depth_changes:
                    avg_change = np.mean(depth_changes)
                    change_std = np.std(depth_changes)
                    change_consistency = change_std < 0.4  # Low variance = consistent steps

                    # Check for ascending pattern (depth increases as we look up)
                    # or descending pattern (depth decreases as we look up)
                    positive_changes = sum(1 for c in depth_changes if c > 0.08)
                    negative_changes = sum(1 for c in depth_changes if c < -0.08)

                    # Require more consistent pattern for detection
                    total_significant = positive_changes + negative_changes
                    if total_significant >= 2 and change_consistency:
                        result["stairs_detected"] = True
                        if positive_changes > negative_changes:
                            result["stairs_direction"] = "up"  # Stairs going up
                            result["floor_level_change"] = "检测到上升台阶模式"
                        else:
                            result["stairs_direction"] = "down"  # Stairs going down
                            result["floor_level_change"] = "检测到下降台阶模式"

                        # Calculate confidence based on pattern strength
                        result["confidence"] = min(total_significant / 4.0, 1.0)

                        self.logger.info(f"[PERCEPTION] Potential stairs detected: {result['stairs_direction']} (conf: {result['confidence']:.2f})")

            # Method 2: Check for horizontal edges (step boundaries) in the depth gradient
            # This is a secondary check to improve detection
            if not result["stairs_detected"]:
                # Compute horizontal gradient in bottom region
                bottom_center = depth_image[int(h*0.7):int(h*0.95), int(w*0.3):int(w*0.7)]
                if bottom_center.size > 0:
                    # Compute gradient along vertical axis
                    gradient_y = np.diff(bottom_center, axis=0)
                    # Look for significant positive/negative gradients (step edges)
                    edge_threshold = 0.15  # 15cm step
                    positive_edges = np.sum(gradient_y > edge_threshold)
                    negative_edges = np.sum(gradient_y < -edge_threshold)

                    # If we see alternating edges, might be stairs
                    if positive_edges > 50 and negative_edges > 50:
                        result["stairs_detected"] = True
                        result["floor_level_change"] = "检测到台阶边缘"
                        result["confidence"] = 0.5
                        self.logger.info(f"[PERCEPTION] Stairs detected via edge analysis: +{positive_edges}/-{negative_edges} edges")

        except Exception as e:
            self.logger.warning(f"Depth stair analysis failed: {e}")

        return result

    def get_stuck_escape_opinion(
        self,
        rgb_history: List[np.ndarray],
        depth_history: List[np.ndarray],
        context: NavContext = None
    ) -> Dict[str, Any]:
        """Provide stuck escape opinion based on RGB and depth history.

        Analyzes recent visual history to determine the best escape direction.

        Args:
            rgb_history: List of recent RGB images
            depth_history: List of recent depth images
            context: Navigation context (optional, for additional info)

        Returns:
            Dict with escape direction, confidence, and reasoning
        """
        opinion = {
            "direction": "right",
            "confidence": 0.5,
            "reason": "默认建议",
            "stop_condition": "",
            "agent_source": "perception",
        }

        # Analyze the most recent depth image
        if depth_history and len(depth_history) > 0:
            latest_depth = depth_history[-1]
            path_analysis = self.analyze_depth_for_path(latest_depth)

            recommended = path_analysis["recommended_direction"]
            center_blocked = path_analysis["obstacle_ahead"]

            # Map direction to turn action
            direction_map = {
                "forward": "forward",
                "left": "left",
                "right": "right",
            }

            opinion["direction"] = direction_map.get(recommended, "right")

            if center_blocked:
                opinion["confidence"] = 0.8
                opinion["reason"] = f"前方有障碍物，建议{opinion['direction']}转"
            else:
                opinion["confidence"] = 0.6
                opinion["reason"] = "前方通畅，可以前进"

            # Check if we have multiple frames for trend analysis
            if len(depth_history) >= 3:
                # Analyze if obstacle has been persistent
                blocked_count = 0
                left_clear_count = 0
                right_clear_count = 0

                for depth in depth_history[-5:]:
                    analysis = self.analyze_depth_for_path(depth)
                    if analysis["obstacle_ahead"]:
                        blocked_count += 1
                    if analysis["left"]["passable"]:
                        left_clear_count += 1
                    if analysis["right"]["passable"]:
                        right_clear_count += 1

                # Persistent obstacle - higher confidence
                if blocked_count >= 3:
                    opinion["confidence"] = 0.9
                    opinion["reason"] = f"持续{blocked_count}帧检测到障碍物"

                # Choose consistently clearer side
                if left_clear_count > right_clear_count + 1:
                    opinion["direction"] = "left"
                    opinion["reason"] += f"，左侧更开阔({left_clear_count}/{right_clear_count})"
                elif right_clear_count > left_clear_count + 1:
                    opinion["direction"] = "right"
                    opinion["reason"] += f"，右侧更开阔({right_clear_count}/{left_clear_count})"

        # Add stop condition
        opinion["stop_condition"] = "移动0.5米或遇到新障碍"

        return opinion