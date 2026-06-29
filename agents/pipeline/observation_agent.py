"""ObservationAgent - VLM core, task-oriented perception.

Uses Qwen3-VL-8B-Instruct for structured scene analysis.
Outputs structured JSON with objects, navigation cues, and task-relevant hints.

Key difference from generic VLMs: outputs task-specific observations
rather than generic scene descriptions.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import os
from datetime import datetime

from agents.pipeline.base_pipeline_agent import SubAgent, ObservationOutput
from agents.base_agent import AgentRole

# Debug: 保存图像目录
DEBUG_IMAGE_DIR = "/tmp/vlm_debug_images"
os.makedirs(DEBUG_IMAGE_DIR, exist_ok=True)


class ObservationAgent(SubAgent):
    """观察Agent - VLM核心，任务导向感知。

    使用VLM分析RGB+Depth图像，输出任务相关的结构化观察结果。
    不是通用的场景描述，而是针对当前子任务的定向观察。
    """

    name = "observation_agent"

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.PERCEPTION

    def __init__(self, config: Dict[str, Any] = None):
        """初始化ObservationAgent。

        Args:
            config: Agent配置字典 (model_key defaults to "qwen3-vl-8b")
        """
        super().__init__(config)
        self.logger = logging.getLogger("ObservationAgent")
        # Default to Qwen3-VL-8B for dedicated VLM perception
        if "model_key" not in self.config:
            self.config["model_key"] = "qwen3-vl-8b"

    def process(
        self,
        subtask,
        position: List[float],
        rotation: float,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        instruction: str = "",  # NEW: original instruction for fallback
    ) -> ObservationOutput:
        """任务导向观察处理。

        Args:
            subtask: 当前子任务（包含description和completion_condition）
            position: 当前位置 [x, y, z]
            rotation: 当前朝向（角度）
            rgb_image: RGB图像 (H, W, 3)
            depth_image: 深度图像 (H, W)
            instruction: 原始导航指令（用于fallback）

        Returns:
            ObservationOutput: 结构化的任务相关观察结果

        Raises:
            RuntimeError: 如果ModelManager未设置
        """
        # 检查ModelManager
        if self._model_manager is None:
            raise RuntimeError("ModelManager not set")

        # 构建任务导向prompt
        prompt = self._build_prompt(subtask, position, rotation, instruction)  # NEW: pass instruction

        # Debug: 打印完整prompt
        relevant_objects = getattr(subtask, 'relevant_objects', [])
        print(f"[ObservationAgent] relevant_objects: {relevant_objects}")
        print(f"[ObservationAgent] Prompt length: {len(prompt)} chars")
        print(f"[ObservationAgent] Prompt preview (first 500 chars): {prompt[:500]}")

        # Debug: 保存RGB和Depth图像
        timestamp = datetime.now().strftime("%H%M%S")
        step_num = len([f for f in os.listdir(DEBUG_IMAGE_DIR) if f.startswith("rgb_")])
        rgb_path = os.path.join(DEBUG_IMAGE_DIR, f"rgb_{step_num:03d}_{timestamp}.png")
        depth_path = os.path.join(DEBUG_IMAGE_DIR, f"depth_{step_num:03d}_{timestamp}.png")

        # Debug: Print image statistics
        print(f"[ObservationAgent] RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        print(f"[ObservationAgent] RGB mean: {rgb_image.mean():.1f}, min: {rgb_image.min()}, max: {rgb_image.max()}")

        # Save RGB
        try:
            from PIL import Image
            Image.fromarray(rgb_image.astype(np.uint8)).save(rgb_path)
            # Save depth as color-mapped image (red=close, blue=far)
            depth_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min() + 1e-6)
            depth_colored = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
            # Red for close, Blue for far
            depth_colored[:, :, 0] = (255 * (1 - depth_normalized)).astype(np.uint8)  # Red channel
            depth_colored[:, :, 2] = (255 * depth_normalized).astype(np.uint8)  # Blue channel
            Image.fromarray(depth_colored).save(depth_path)
            print(f"[ObservationAgent] Saved images: {rgb_path}, {depth_path}")
        except Exception as e:
            print(f"[ObservationAgent] Failed to save images: {e}")

        # 调用VLM（RGB+Depth）
        response = self._call_vlm(
            prompt=prompt,
            images=[rgb_image, depth_image],
            max_tokens=400,  # Increase for detailed objects
            temperature=0.2,
        )

        # 解析响应
        response_text = response.get("response", "")
        print(f"[ObservationAgent] VLM response (full): {response_text}")
        output_dict = self._parse_response(response_text)
        print(f"[ObservationAgent] Parsed objects: {output_dict.get('objects')}")

        return ObservationOutput(**output_dict)

    def _build_prompt(
        self,
        subtask,
        position: List[float],
        rotation: float,
        instruction: str = "",  # NEW: original instruction
    ) -> str:
        """构建任务导向prompt with fallback机制.

        Args:
            subtask: 当前子任务
            position: 当前位置
            rotation: 当前朝向
            instruction: 原始导航指令（用于fallback）

        Returns:
            任务导向的prompt字符串
        """
        subtask_desc = getattr(subtask, 'description', str(subtask))
        completion_cond = getattr(subtask, 'completion_condition', {})

        # 解析完成条件
        cond_type = completion_cond.get("type", "unknown")
        cond_direction = completion_cond.get("direction", "")
        cond_target = completion_cond.get("target", "")
        min_change = completion_cond.get("min_change", "")

        # 获取 relevant_objects
        relevant_objects = getattr(subtask, 'relevant_objects', [])
        if not relevant_objects:
            target = completion_cond.get("target", "")
            if target:
                relevant_objects = [target]

        target_list = ", ".join(relevant_objects) if relevant_objects else "none specified"

        prompt = f"""You are a navigation robot analyzing images.

## Images
- First: RGB image (visual scene)
- Second: Depth image (distance visualization):
  * RED = CLOSE (near you)
  * BLUE = FAR (far from you)
  * Gradient: red → yellow → green → cyan → blue (near → far)

## Navigation Context
### Current Subtask
- Description: {subtask_desc}
- Target Objects: {target_list}

### Original Instruction
{instruction}

## Detection Priority
1. Priority: Subtask target objects → if visible, subtask_relevant=true
2. Fallback: Objects from original instruction → if subtask objects NOT visible, check instruction-related objects
3. Last fallback: Report blocking situation + suggest exploration direction

## Output Format (JSON only)
{{
  "subtask_relevant": true/false,
  "instruction_relevant": true/false,
  "fallback_mode": true/false,
  "objects": [
    {{
      "name": "object_name",
      "direction": "forward_left/forward_right/left/right/forward/backward",
      "location": "center/left_side/right_side/foreground/background",
      "distance": "close/medium/far",
      "features": "specific characteristics"
    }}
  ],
  "exploration_hint": "turn_left to find stairs" or "" if target visible,
  "target_direction": "left/right/forward/backward/unknown",
  "target_distance": "close/medium/far/unknown",
  "path_blocked": true/false,
  "navigation_cues": ["cue1", "cue2"],
  "scene_description": "brief description"
}}

## Rules
- If subtask objects visible: subtask_relevant=true, fallback_mode=false, fill objects with subtask targets
- If subtask objects NOT visible but instruction objects visible: subtask_relevant=false, instruction_relevant=true, fallback_mode=true, fill objects with instruction-related items
- If neither visible: subtask_relevant=false, instruction_relevant=false, fallback_mode=true, objects=[], provide exploration_hint

JSON output only:"""

        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析VLM响应 - 多层fallback。

        优先解析JSON，失败时使用默认值填充缺失字段。
        objects字段为List[Dict]，每个Dict包含name, direction, location, distance, features。

        Args:
            response: VLM响应文本

        Returns:
            解析后的字典，包含所有ObservationOutput所需字段
        """
        # 默认值 - 包含新字段
        defaults = {
            "subtask_relevant": False,  # 新字段名（替代原task_relevant）
            "instruction_relevant": False,  # 新增
            "fallback_mode": False,  # 新增
            "objects": [],  # List[Dict[str, Any]]
            "exploration_hint": "",  # 新增
            "target_direction": "unknown",
            "target_distance": "unknown",
            "path_blocked": False,
            "navigation_cues": [],
            "scene_description": "",
        }

        if not response:
            return defaults

        # 尝试从响应中提取JSON
        json_str = response.strip()

        # 1. 尝试提取markdown代码块中的JSON
        if '```json' in json_str:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', json_str)
            if json_match:
                json_str = json_match.group(1).strip()
        elif '```' in json_str:
            json_match = re.search(r'```\s*([\s\S]*?)\s*```', json_str)
            if json_match:
                json_str = json_match.group(1).strip()

        # 2. 尝试找到JSON对象（使用括号匹配）
        start = json_str.find('{')
        if start != -1:
            brace_count = 0
            end = -1
            for i in range(start, len(json_str)):
                if json_str[i] == '{':
                    brace_count += 1
                elif json_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break

            if end != -1:
                json_candidate = json_str[start:end+1]
                try:
                    data = json.loads(json_candidate)
                    # 合并解析结果和默认值（填补缺失字段）
                    result = defaults.copy()
                    for key in defaults:
                        if key in data:
                            # 特殊处理objects字段：确保是List[Dict]
                            if key == "objects":
                                objects_raw = data[key]
                                if isinstance(objects_raw, list):
                                    # 转换每个元素为标准Dict格式
                                    normalized_objects = []
                                    for obj in objects_raw:
                                        if isinstance(obj, dict):
                                            # 确保必需字段存在
                                            normalized_obj = {
                                                "name": obj.get("name", "unknown"),
                                                "direction": obj.get("direction", "unknown"),
                                                "location": obj.get("location", "unknown"),
                                                "distance": obj.get("distance", "unknown"),
                                                "features": obj.get("features", ""),
                                            }
                                            normalized_objects.append(normalized_obj)
                                        elif isinstance(obj, str):
                                            # 旧格式：字符串 → 转换为Dict
                                            normalized_objects.append({
                                                "name": obj,
                                                "direction": "unknown",
                                                "location": "unknown",
                                                "distance": "unknown",
                                                "features": "",
                                            })
                                    result[key] = normalized_objects
                                else:
                                    result[key] = []
                            else:
                                result[key] = data[key]
                    # Compatibility: map task_relevant → subtask_relevant
                    if "task_relevant" in data and "subtask_relevant" not in data:
                        result["subtask_relevant"] = data["task_relevant"]
                    return result
                except json.JSONDecodeError as e:
                    self.logger.debug(f"JSON parse failed: {e}")

        # 3. Fallback: 返回默认值，scene_description设为原始响应
        result = defaults.copy()
        result["scene_description"] = response.strip()[:200]  # 限制长度
        return result