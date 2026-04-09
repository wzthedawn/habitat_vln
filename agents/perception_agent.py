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
                # 获取目标方向信息
                goal_direction = None
                if context and hasattr(context, 'metadata'):
                    angle = context.metadata.get("angle_to_goal", 0)
                    hint = context.metadata.get("direction_hint", "ahead")
                    # Always pass goal_direction when metadata exists - knowing goal is ahead is valuable
                    goal_direction = {"angle": angle, "hint": hint}

                vlm_result = self._analyze_with_vlm(rgb_image, depth_image, goal_direction)

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
                scene_brief = "Unable to get visual information"
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
                        "object": obj.get("name", "unknown"),
                        "distance": f"{obj.get('distance', 0):.1f}m",
                        "angle": f"{obj.get('angle', 0):.0f}°"
                    })
                else:
                    formatted_objects.append({
                        "object": obj.get("name", "unknown")
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
                reasoning=f"{scene_description}, detected {len(objects)} objects",
            )

        except Exception as e:
            self.logger.error(f"[Perception] Error: {e}")
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
        goal_direction: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze RGB + Depth images using VLM with goal direction awareness.

        Args:
            rgb_image: RGB图像
            depth_image: 深度图像
            goal_direction: 目标方向信息 {"angle": 角度, "hint": 方向提示}

        Returns:
            Dictionary with room_type, objects, scene_description, nav_hint, walkable_analysis, obstacle_ahead
        """
        if not self._model_manager:
            return None

        try:
            self.logger.warning(f"[DEBUG] _model_manager: {self._model_manager}, rgb_image: {rgb_image is not None}, depth_image: {depth_image is not None}")
            has_depth = depth_image is not None

            # 构建目标方向提示
            goal_hint = ""
            if goal_direction:
                angle = goal_direction.get("angle", 0)
                hint = goal_direction.get("hint", "ahead")
                goal_hint = f"""
## Goal Direction
- Goal is {hint} (relative angle {angle:.0f}°)
- Navigation advice should align with goal direction"""

            if has_depth:
                # RGB + Depth: Simplified prompt with stair direction detection
                prompt = f"""/no_think
You are a navigation robot. Analyze the images and output JSON.

## Images
- First: RGB image
- Second: Depth image (red=near, blue=far)
{goal_hint}

## Required Output (JSON only)
{{"room_type":"hallway/bedroom/living_room/kitchen/bathroom/stairs",
"scene_brief":"short description in 10-15 words",
"objects":[{{"name":"object name","distance":1.0}}],
"stair_direction":"up/down/none",
"nav_hint":"navigation hint in 20-30 words"}}

## Guidelines
1. room_type: Use simple names (hallway, bedroom, living_room, kitchen, bathroom, stairs)
2. scene_brief: Brief description of what you see
3. objects: List 3-5 visible objects with distance in meters
4. stair_direction: ONLY when stairs are visible:
   - "up" if stairs ascend away from you
   - "down" if stairs descend away from you
   - "none" if no stairs visible
5. nav_hint: Key navigation guidance aligned with goal direction{f" (goal is {goal_direction.get('hint', 'ahead')})" if goal_direction else ""}. If stairs visible, mention direction.

Output JSON only, no explanation:"""
            else:
                # RGB-only: Simplified prompt
                prompt = """/no_think
You are a navigation robot. Analyze the image and output JSON.

## Required Output (JSON only)
{"room_type":"hallway/bedroom/living_room/kitchen/bathroom/stairs",
"scene_brief":"short description in 10-15 words",
"objects":[{"name":"object name"}],
"nav_hint":"navigation hint in 20-30 words"}

## Guidelines
1. room_type: Use simple names (hallway, bedroom, living_room, kitchen, bathroom, stairs)
2. scene_brief: Brief description of what you see
3. objects: List 3-5 visible objects
4. nav_hint: Key navigation guidance (directions, open paths, landmarks)

## Example Output
{"room_type":"living_room","scene_brief":"Spacious room with sofa and TV, door on the right","objects":[{"name":"sofa"},{"name":"tv"},{"name":"door"}],"nav_hint":"Open space ahead, door on right, sofa on left"}

Output JSON only, no explanation:"""

            # Call VLM with RGB + Depth images
            max_retries = 2
            parsed = None

            for attempt in range(max_retries + 1):
                if depth_image is not None:
                    self.logger.info(f"[Perception] Calling generate_vision_dual (attempt {attempt+1}/{max_retries+1}), RGB shape: {rgb_image.shape if hasattr(rgb_image, 'shape') else type(rgb_image)}")
                    result = self._model_manager.generate_vision_dual(
                        rgb_image=rgb_image,
                        depth_image=depth_image,
                        prompt=prompt,
                        model_key="qwen-9b-perception",
                        max_new_tokens=500,  # Reduced since prompt is simpler
                        temperature=0.3,  # Slightly higher for retry variety
                    )
                    self.logger.info(f"[Perception] generate_vision_dual returned: keys={result.keys() if isinstance(result, dict) else type(result)}")
                else:
                    self.logger.info(f"[Perception] Calling generate_vision (attempt {attempt+1}/{max_retries+1}), RGB shape: {rgb_image.shape if hasattr(rgb_image, 'shape') else type(rgb_image)}")
                    result = self._model_manager.generate_vision(
                        image=rgb_image,
                        prompt=prompt,
                        model_key="qwen-9b-perception",
                        max_new_tokens=500,
                        temperature=0.3,
                    )

                if not result or not result.get("response"):
                    self.logger.warning(f"[Perception] No VLM response on attempt {attempt+1}")
                    continue

                raw_response = result["response"]
                self.logger.info(f"[Perception] VLM response: {raw_response[:150]}...")

                # Parse VLM response
                parsed = self._parse_vlm_response(raw_response)
                self.logger.info(f"[Perception] Parsed room_type: {parsed.get('room_type', 'None')}, objects count: {len(parsed.get('objects', []))}")

                # Validate perception output
                is_valid, missing = self._validate_perception_output(parsed)
                if is_valid:
                    self.logger.info(f"[Perception] Valid output received on attempt {attempt+1}")
                    break
                else:
                    self.logger.warning(f"[Perception] Invalid output on attempt {attempt+1}, missing: {missing}")
                    if attempt < max_retries:
                        self.logger.info("[Perception] Retrying...")
                    # parsed will be returned anyway with best effort extraction

            if parsed:
                self.logger.info(f"[Perception] Room: {parsed.get('room_type', 'unknown')}, Objects: {len(parsed.get('objects', []))}")

            # Print full output to console
            print(f"\n[PerceptionAgent] Output:")
            print(f"  Room: {parsed.get('room_type', 'unknown')}")
            print(f"  Objects: {parsed.get('objects', [])}")
            print(f"  nav_hint: {parsed.get('nav_hint', '')}")
            if parsed.get('stair_entrance', {}).get('found'):
                print(f"  Stair entrance: {parsed.get('stair_entrance')}")

            return parsed

        except Exception as e:
            self.logger.error(f"[Perception] VLM analysis failed: {e}")
            return None

    def _validate_perception_output(self, parsed: Dict[str, Any]) -> tuple:
        """Validate perception output has essential fields.

        Args:
            parsed: Parsed perception dictionary

        Returns:
            (is_valid, missing_fields): Boolean and list of missing field names
        """
        if not parsed:
            return False, ["no_output"]

        missing = []

        # Check room_type
        room_type = parsed.get("room_type", "unknown")
        if room_type == "unknown" or not room_type:
            missing.append("room_type")

        # Check objects
        objects = parsed.get("objects", [])
        if not objects or len(objects) == 0:
            missing.append("objects")

        # Check nav_hint
        nav_hint = parsed.get("nav_hint", "")
        if not nav_hint or len(nav_hint) < 10:
            missing.append("nav_hint")

        # Consider valid if at least room_type and one other field are present
        is_valid = len(missing) <= 1 or (room_type != "unknown" and len(objects) > 0)

        return is_valid, missing

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response into structured format with robust fallback strategies.

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
            "stairs": {"detected": False, "direction": "none", "distance": 5.0, "relative_pos": "unknown"},
            "stair_entrance": {"found": False},
            "spatial_structure": {},
        }

        object_names = []
        parse_success = False

        # Strategy 1: Try parsing full JSON
        json_str = response

        # Handle markdown code blocks: ```json ... ``` or ``` ... ```
        if '```json' in response:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()
        elif '```' in response:
            json_match = re.search(r'```\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1).strip()

        # Extract JSON object: first '{' to last '}'
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_candidate = json_str[start:end+1]
            try:
                data = json.loads(json_candidate)
                parse_success = self._extract_from_json(data, result, object_names)
                if parse_success:
                    self.logger.info(f"[Perception] JSON parse SUCCESS: room={result['room_type']}, objects={len(result['objects'])}")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self.logger.warning(f"[Perception] JSON parse failed: {e}")
                # Try partial JSON extraction
                parse_success = self._try_partial_json_extraction(json_candidate, result, object_names)

        # Strategy 2: Regex fallback for key fields if JSON parsing failed
        if not parse_success or result["room_type"] == "unknown":
            self.logger.info("[Perception] Attempting regex fallback extraction...")
            self._regex_fallback_extract(response, result, object_names)

        # Strategy 3: Generate scene description from extracted objects
        if not result["scene_description"] and object_names:
            result["scene_description"] = f"Visible: {', '.join(object_names[:5])}"
        elif not result["scene_description"]:
            result["scene_description"] = "Scene analyzing..."

        # Infer room type from objects if still unknown
        if result["room_type"] == "unknown" and object_names:
            result["room_type"] = self._infer_room_from_objects(object_names)
            if result["room_type"] != "unknown":
                self.logger.info(f"[Perception] Inferred room_type from objects: {result['room_type']}")

        # Log final result
        self.logger.info(f"[Perception] Final parse result: room={result['room_type']}, objects={len(result['objects'])}, nav_hint_len={len(result['nav_hint'])}")

        return result

    def _extract_from_json(self, data: dict, result: dict, object_names: list) -> bool:
        """Extract fields from parsed JSON data.

        Returns:
            True if essential fields were extracted
        """
        success = False

        # Parse room type
        if "room_type" in data:
            result["room_type"] = self._normalize_room_type(data["room_type"])
            result["room_confidence"] = 0.7
            success = True

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
                    name = obj.get("name", "") or ""
                    if name:
                        object_names.append(name)
                        result["objects"].append({
                            "name": name,
                            "confidence": 0.6,
                            "distance": float(obj.get("distance", 0) or 0),
                            "angle": float(obj.get("angle", 0) or 0),
                            "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                            "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                            "source": "vlm",
                        })
            success = True

        # Parse obstacle
        if "obstacle_ahead" in data:
            obs = data["obstacle_ahead"]
            if isinstance(obs, dict):
                result["obstacle_ahead"] = obs
            elif isinstance(obs, bool):
                result["obstacle_ahead"] = {"blocked": obs, "min_distance": 5.0}

        # Parse stairs info
        if "stairs" in data:
            stairs = data["stairs"]
            if isinstance(stairs, dict):
                result["stairs"] = {
                    "detected": stairs.get("detected", False),
                    "direction": stairs.get("direction", "none"),
                    "distance": float(stairs.get("distance", 5.0) or 5.0),
                    "relative_pos": stairs.get("relative_pos", "unknown"),
                }

        # Parse simple stair_direction field (new format)
        if "stair_direction" in data:
            stair_dir = data["stair_direction"]
            if stair_dir in ["up", "down"]:
                result["stairs"]["detected"] = True
                result["stairs"]["direction"] = stair_dir
                self.logger.info(f"[Perception] Stair direction detected: {stair_dir}")
            elif result["room_type"] == "stairs" and stair_dir == "none":
                # If room is stairs but direction not specified, mark as detected
                result["stairs"]["detected"] = True
                result["stairs"]["direction"] = "unknown"

        # Parse stair entrance
        if "stair_entrance" in data:
            entrance = data["stair_entrance"]
            if isinstance(entrance, dict) and entrance.get("found"):
                result["stair_entrance"] = {
                    "found": True,
                    "direction": entrance.get("direction", "unknown"),
                    "distance": float(entrance.get("distance", 0) or 0),
                    "angle": float(entrance.get("angle", 0) or 0),
                    "stair_direction": entrance.get("stair_direction", "unknown"),
                    "action_hint": entrance.get("action_hint", ""),
                }
                self.logger.info(f"[Perception] Stair entrance found: {result['stair_entrance']}")
            else:
                result["stair_entrance"] = {"found": False}
        else:
            result["stair_entrance"] = {"found": False}

        # Parse spatial structure
        if "spatial_structure" in data:
            ss = data["spatial_structure"]
            if isinstance(ss, dict):
                result["spatial_structure"] = ss

        # Parse description and hint
        result["scene_description"] = data.get("scene_brief", data.get("scene_description", ""))
        result["nav_hint"] = data.get("nav_hint", "")

        if result["nav_hint"]:
            success = True

        return success

    def _try_partial_json_extraction(self, json_str: str, result: dict, object_names: list) -> bool:
        """Try to extract partial information from malformed JSON."""
        import re

        success = False

        # Try to extract room_type
        room_match = re.search(r'"room_type"\s*:\s*"([^"]+)"', json_str)
        if room_match:
            result["room_type"] = self._normalize_room_type(room_match.group(1))
            result["room_confidence"] = 0.6
            success = True
            self.logger.info(f"[Perception] Partial extract: room_type={result['room_type']}")

        # Try to extract nav_hint
        nav_hint_match = re.search(r'"nav_hint"\s*:\s*"([^"]+)"', json_str)
        if nav_hint_match:
            result["nav_hint"] = nav_hint_match.group(1)
            success = True
            self.logger.info(f"[Perception] Partial extract: nav_hint found")

        # Try to extract scene_brief
        scene_match = re.search(r'"scene_brief"\s*:\s*"([^"]+)"', json_str)
        if scene_match:
            result["scene_description"] = scene_match.group(1)
            success = True

        # Try to extract objects array (simplified)
        objects_pattern = r'"objects"\s*:\s*\[(.*?)\]'
        objects_match = re.search(objects_pattern, json_str, re.DOTALL)
        if objects_match:
            objects_str = objects_match.group(1)
            # Extract object names
            name_matches = re.findall(r'"name"\s*:\s*"([^"]+)"', objects_str)
            for name in name_matches[:self.max_objects]:
                if name:
                    object_names.append(name)
                    result["objects"].append({
                        "name": name,
                        "confidence": 0.5,
                        "distance": 0,
                        "angle": 0,
                        "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                        "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                        "source": "vlm_partial",
                    })
            if object_names:
                success = True
                self.logger.info(f"[Perception] Partial extract: {len(object_names)} objects")

        return success

    def _regex_fallback_extract(self, response: str, result: dict, object_names: list) -> None:
        """Regex-based fallback extraction for key fields."""
        import re

        # Extract room_type with various patterns
        room_patterns = [
            r'"room_type"\s*:\s*"([^"]+)"',
            r'room[_\s]*type[:\s]+"?([^"\n,]+)"?',
            r'in\s+(?:a\s+)?(\w+)\s+(?:room|area|space)',
            r'(?:this\s+is\s+(?:a\s+)?)?(\w+)\s+(?:room|area)',
        ]
        for pattern in room_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["room_type"] = self._normalize_room_type(match.group(1))
                result["room_confidence"] = 0.5
                self.logger.info(f"[Perception] Regex fallback: room_type={result['room_type']}")
                break

        # Extract nav_hint
        nav_hint_patterns = [
            r'"nav_hint"\s*:\s*"([^"]+)"',
            r'nav[_\s]*hint[:\s]+"?([^"\n]+)"?',
            r'(?:navigation|nav)\s+(?:hint|guide|direction)[:\s]+"?([^"\n]+)"?',
        ]
        for pattern in nav_hint_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["nav_hint"] = match.group(1).strip()
                self.logger.info(f"[Perception] Regex fallback: nav_hint found")
                break

        # Extract stair_direction
        stair_dir_patterns = [
            r'"stair_direction"\s*:\s*"([^"]+)"',
            r'stair[_\s]*direction[:\s]+"?([^"\n,]+)"?',
            r'(?:stairs|staircase)\s+(?:go|going|lead|leads)\s+(up|down)',
            r'(?:ascend|descend)\s+(?:the\s+)?stairs',
        ]
        for pattern in stair_dir_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                stair_dir = match.group(1).lower().strip()
                # Normalize stair direction
                if stair_dir in ["up", "ascend", "ascending", "upward"]:
                    result["stairs"]["detected"] = True
                    result["stairs"]["direction"] = "up"
                    self.logger.info(f"[Perception] Regex fallback: stair_direction=up")
                elif stair_dir in ["down", "descend", "descending", "downward"]:
                    result["stairs"]["detected"] = True
                    result["stairs"]["direction"] = "down"
                    self.logger.info(f"[Perception] Regex fallback: stair_direction=down")
                break

        # Extract scene_brief/description
        scene_patterns = [
            r'"scene_brief"\s*:\s*"([^"]+)"',
            r'"scene_description"\s*:\s*"([^"]+)"',
            r'(?:scene|description)[:\s]+"?([^"\n]+)"?',
        ]
        for pattern in scene_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                result["scene_description"] = match.group(1).strip()
                break

        # Extract objects - look for name fields or object lists
        object_patterns = [
            r'"name"\s*:\s*"([^"]+)"',
            r'(?:object|item|visible)[:\s]+"?([^"\n,]+)"?',
        ]
        for pattern in object_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for name in matches[:self.max_objects]:
                name = name.strip()
                if name and len(name) > 1 and name not in object_names:
                    # Filter out non-object words
                    if name.lower() not in ['true', 'false', 'null', 'none', 'unknown']:
                        object_names.append(name)
                        result["objects"].append({
                            "name": name,
                            "confidence": 0.4,
                            "distance": 0,
                            "angle": 0,
                            "is_navigation_object": name.lower() in self.NAVIGATION_OBJECTS,
                            "is_landmark": name.lower() in self.LANDMARK_OBJECTS,
                            "source": "regex_fallback",
                        })

        if object_names:
            self.logger.info(f"[Perception] Regex fallback: {len(object_names)} objects extracted")

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

        prompt = f"""/no_think
Analyze the best navigation action based on perception information.

## Environment Perception
- Obstacles: {obstacles if obstacles else "none"}
- Visible landmarks: {landmarks if landmarks else "none"}
- Scene description: {context.visual_features.scene_description[:100] if context.visual_features.scene_description else "none"}
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
            return self._parse_perception_opinion_response(response, obstacles, landmarks)
        except Exception as e:
            self.logger.error(f"[Perception] LLM perception opinion failed: {e}")
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
            reasoning=reasoning or f"Detected {len(obstacles)} obstacles",
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
            reasoning="Perception analysis (no LLM)",
            constraints=constraints,
        )