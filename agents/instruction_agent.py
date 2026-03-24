"""Instruction Agent for decomposing and interpreting navigation instructions.

This version uses rule-based parsing without LLM for subtask decomposition
and task level classification (简单/中等/困难).
"""

from typing import Dict, Any, Optional, List
import re
import logging

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext, SubTask


class InstructionAgent(BaseAgent):
    """
    Agent responsible for instruction understanding and decomposition.

    Uses rule-based parsing (no LLM) for:
    1. Parse natural language navigation instructions
    2. Decompose complex instructions into subtasks
    3. Identify landmarks and goals
    4. Classify task difficulty level (简单/中等/困难)
    """

    # Direction keywords
    DIRECTION_KEYWORDS = {
        "left", "right", "straight", "forward", "back", "backward",
        "turn", "walk", "go", "move", "head", "face",
    }

    # Landmark keywords (common objects in indoor environments)
    LANDMARK_KEYWORDS = {
        # Rooms
        "room", "kitchen", "bedroom", "bathroom", "living room",
        "dining room", "hallway", "corridor", "office", "garage",
        "stairs", "staircase", "stairway", "entrance", "door", "exit",
        # Furniture
        "chair", "table", "desk", "bed", "sofa", "couch",
        "cabinet", "shelf", "bookshelf", "wardrobe", "dresser",
        "counter", "sink", "toilet", "bathtub", "shower",
        "piano", "keyboard", "bench",
        # Objects
        "carpet", "rug", "mat", "curtain", "window", "plant", "lamp",
        "tv", "television", "refrigerator", "oven", "stove",
        "picture", "painting", "mirror", "clock",
    }

    # Action keywords
    ACTION_KEYWORDS = {
        "turn", "go", "walk", "move", "stop", "wait",
        "find", "look", "search", "locate", "reach",
        "enter", "exit", "pass", "cross", "climb",
    }

    # Conditional keywords (indicates 困难 level)
    CONDITIONAL_KEYWORDS = {
        "if", "when", "unless", "either", "or", "otherwise",
        "then", "after", "before", "while", "until",
    }

    # Sequence keywords (indicates 中等 or 困难 level)
    SEQUENCE_KEYWORDS = {
        "then", "after that", "next", "and then", "before",
        "first", "second", "finally", "lastly",
    }

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("InstructionAgent")

        # LLM support for semantic decomposition
        self._model_manager = self.config.get("model_manager")
        self.use_llm_decompose = self.config.get("use_llm_decompose", True)

    @property
    def name(self) -> str:
        return "instruction_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.INSTRUCTION

    def get_required_inputs(self) -> List[str]:
        return ["instruction"]

    def get_output_keys(self) -> List[str]:
        return ["subtasks", "landmarks", "goals", "task_level", "complexity"]

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Process instruction and extract navigation components.

        Args:
            context: Navigation context with instruction
            strategy_result: Optional strategy output

        Returns:
            AgentOutput with parsed instruction components
        """
        self.initialize()

        # Validate
        errors = self.validate_context(context)
        if errors:
            return AgentOutput.failure_output(errors, "Invalid context")

        try:
            instruction = context.instruction

            # Parse instruction (rule-based)
            parsed = self._parse_instruction(instruction)

            # Determine task level (rule-based)
            task_level = self._determine_task_level_fallback(parsed)

            # Get goal position from context metadata
            goal_position = context.metadata.get("goal_position")

            # Create subtasks (with LLM semantic decomposition if available)
            subtasks = self._create_subtasks(parsed, task_level, goal_position)

            # Update context subtasks
            context.subtasks = subtasks

            # Store task level in context
            context.metadata["task_level"] = task_level

            return AgentOutput.success_output(
                data={
                    "subtasks": [
                        {
                            "id": s.id,
                            "description": s.description,
                            "status": s.status,
                            "level": s.level,
                            "precondition": s.precondition,
                            "completion_condition": s.completion_condition,
                            "spatial_constraint": s.spatial_constraint,
                        } for s in subtasks
                    ],
                    "current_subtask": {
                        "id": subtasks[0].id if subtasks else 0,
                        "description": subtasks[0].description if subtasks else "",
                        "status": subtasks[0].status if subtasks else "pending",
                    } if subtasks else None,
                    "instruction_analysis": {
                        "landmarks": parsed["landmarks"],
                        "goals": parsed["goals"],
                        "directions": parsed["directions"],
                        "complexity": parsed["complexity"],
                        "task_level": task_level,
                    },
                    "landmarks": parsed["landmarks"],
                    "goals": parsed["goals"],
                    "directions": parsed["directions"],
                    "task_level": task_level,
                    "complexity": parsed["complexity"],
                    "parsed_instruction": parsed,
                    "decomposition_method": "llm" if self.use_llm_decompose and self._model_manager else "rule",
                },
                confidence=parsed["confidence"],
                reasoning=f"Parsed {len(subtasks)} subtasks, task level: {task_level}",
            )

        except Exception as e:
            self.logger.error(f"Instruction parsing error: {e}")
            return AgentOutput.failure_output([str(e)], "Failed to parse instruction")

    def _parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """Parse instruction into components."""
        # Initialize result
        result = {
            "original": instruction,
            "landmarks": [],
            "goals": [],
            "directions": [],
            "actions": [],
            "conditions": [],
            "sequences": [],
            "complexity": 0.0,
            "confidence": 0.8,
        }

        instruction_lower = instruction.lower()

        # Extract directions
        result["directions"] = self._extract_directions(instruction_lower)

        # Extract landmarks
        result["landmarks"] = self._extract_landmarks(instruction_lower)

        # Extract goals
        result["goals"] = self._extract_goals(instruction_lower)

        # Extract conditions
        result["conditions"] = self._extract_conditions(instruction_lower)

        # Extract sequence markers
        result["sequences"] = self._extract_sequences(instruction_lower)

        # Calculate complexity
        result["complexity"] = self._calculate_complexity(result)

        return result

    def _extract_directions(self, text: str) -> List[str]:
        """Extract directional cues from text."""
        directions = []

        # Direction patterns
        patterns = [
            r"\b(turn|go|walk|move|head)\s+(left|right|straight|forward|back)\b",
            r"\b(left|right)\b",
            r"\b(forward|straight|back|backward)\b",
            r"\b(north|south|east|west)\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    # Take the last element (the actual direction)
                    direction = match[-1]
                else:
                    direction = match
                if direction not in directions:
                    directions.append(direction)

        return directions

    def _extract_landmarks(self, text: str) -> List[str]:
        """Extract landmarks/objects from text."""
        landmarks = []

        # Check each landmark keyword
        for keyword in self.LANDMARK_KEYWORDS:
            if keyword in text:
                landmarks.append(keyword)

        # Extract phrases with "the X" pattern
        patterns = [
            r"\bthe\s+(\w+(?:\s+\w+)?)\s+(door|room|stairs|hallway)\b",
            r"\b(beside|near|next\s+to|by|around)\s+(?:the\s+)?(\w+)\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    landmark = " ".join(m for m in match if m)
                else:
                    landmark = match
                if landmark and landmark not in landmarks:
                    landmarks.append(landmark)

        return landmarks

    def _extract_goals(self, text: str) -> List[str]:
        """Extract navigation goals from text."""
        goals = []

        # Goal patterns
        patterns = [
            r"\b(find|locate|reach|go\s+to|stop\s+at|arrive\s+at)\s+(?:the\s+)?(.+?)(?:\s+and|\s+then|,|$)",
            r"\b(stop|wait)\s+(?:at|by|near)\s+(?:the\s+)?(.+?)(?:\s+and|\s+then|,|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    # The goal is usually the second element
                    goal = match[-1].strip()
                else:
                    goal = match.strip()
                if goal and goal not in goals:
                    goals.append(goal)

        return goals

    def _extract_conditions(self, text: str) -> List[Dict[str, str]]:
        """Extract conditional statements from text."""
        conditions = []

        # Conditional patterns
        patterns = [
            r"\bif\s+(.+?),?\s+(?:then\s+)?(.+?)(?:\s+otherwise|\s+else|,|$)",
            r"\bwhen\s+(?:you\s+)?(.+?),?\s+(.+?)(?:,|$)",
            r"\bunless\s+(.+?),?\s+(.+?)(?:,|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    conditions.append({
                        "type": "conditional",
                        "condition": match[0].strip(),
                        "action": match[1].strip(),
                    })

        return conditions

    def _extract_sequences(self, text: str) -> List[str]:
        """Extract sequence markers from text."""
        sequences = []

        for keyword in self.SEQUENCE_KEYWORDS:
            if keyword in text:
                sequences.append(keyword)

        return sequences

    def _determine_task_level_fallback(self, parsed: Dict[str, Any]) -> str:
        """Fallback rule-based task level determination."""
        # Check for conditional keywords (困难)
        if parsed["conditions"]:
            return "困难"

        # Check for multiple sequences or multiple landmarks (中等 or 困难)
        num_landmarks = len(parsed["landmarks"])
        num_directions = len(parsed["directions"])
        num_goals = len(parsed["goals"])
        num_sequences = len(parsed["sequences"])

        # 困难: Multiple sequences + multiple landmarks + conditions
        if num_sequences >= 2 and num_landmarks >= 2:
            return "困难"

        # 中等: Single goal with landmarks, or multiple directions
        if num_landmarks >= 1 or num_goals >= 1 or num_directions >= 2:
            return "中等"

        # 简单: Basic operations
        if num_directions <= 1 and num_landmarks == 0 and num_goals == 0:
            return "简单"

        # Default to 中等
        return "中等"

    def _determine_subtask_level_fallback(self, segment: str) -> str:
        """Fallback rule-based subtask level determination."""
        segment_lower = segment.lower()

        # Extract features from current subtask
        has_direction = any(word in segment_lower for word in ["turn", "left", "right", "forward", "back", "straight"])
        has_landmark = any(word in segment_lower for word in self.LANDMARK_KEYWORDS)
        has_goal = any(word in segment_lower for word in ["find", "reach", "go to", "stop at", "locate", "arrive", "walk to", "go through"])
        has_condition = any(word in segment_lower for word in self.CONDITIONAL_KEYWORDS)
        has_sequence = any(word in segment_lower for word in self.SEQUENCE_KEYWORDS)

        # 简单: Basic operations (pure direction commands, no landmarks)
        if has_direction and not has_landmark and not has_goal:
            return "简单"

        # 困难: Conditional judgment or multi-step sequences
        if has_condition or has_sequence:
            return "困难"

        # 中等: Involves landmarks or goals
        if has_landmark or has_goal:
            return "中等"

        # Default to 中等
        return "中等"

    def _create_subtasks(self, parsed: Dict[str, Any], task_level: str, goal_position: tuple = None) -> List[SubTask]:
        """Create subtasks from parsed instruction with individual difficulty levels.

        Tries LLM semantic decomposition first, falls back to rule-based.
        """
        instruction = parsed["original"]

        # Try LLM semantic decomposition first
        if self.use_llm_decompose and self._model_manager:
            llm_subtasks = self._semantic_decompose_with_llm(instruction, goal_position)
            if llm_subtasks:
                self.logger.info(f"[Instruction] 语义分解: {len(llm_subtasks)}个子任务")
                return llm_subtasks

        # Fallback to rule-based decomposition
        subtasks = []

        # Split instruction by conjunctions and sequences
        segments = self._split_instruction(instruction)

        for i, segment in enumerate(segments):
            # Determine if this segment is completed
            status = "in_progress" if i == 0 else "pending"

            # Calculate individual level for each subtask (rule-based)
            subtask_level = self._determine_subtask_level_fallback(segment)

            # Parse conditions from segment description (rule-based)
            completion_condition = self._parse_condition_string_fallback(segment)

            subtask = SubTask(
                id=i,
                description=segment.strip(),
                status=status,
                level=subtask_level,
                required_agents=self._determine_required_agents(segment),
                completion_condition=completion_condition,
            )
            subtasks.append(subtask)

        # If no subtasks created, create a single one for the whole instruction
        if not subtasks:
            subtasks.append(SubTask(
                id=0,
                description=instruction,
                status="in_progress",
                level=task_level,
                required_agents=["perception", "trajectory", "decision"],
            ))

        return subtasks

    def _split_instruction(self, text: str) -> List[str]:
        """Split instruction into subtask segments."""
        # Common conjunctions for navigation instructions
        conjunctions = [
            ", and ", " and then ", ", then ", " then ", "; ",
            " after that ", " next ", " finally ",
        ]

        segments = [text]
        for conj in conjunctions:
            new_segments = []
            for segment in segments:
                parts = segment.split(conj)
                new_segments.extend(parts)
            segments = new_segments

        # Also split by periods
        final_segments = []
        for segment in segments:
            parts = segment.split(".")
            final_segments.extend(p.strip() for p in parts if p.strip())

        # Further split by comma if it separates direction instructions
        # e.g., "Walk down the stairs, turn right" -> ["Walk down the stairs", "turn right"]
        refined_segments = []
        for segment in final_segments:
            # Check if segment contains direction change after comma
            lower = segment.lower()
            if ", " in segment:
                parts = segment.split(", ")
                for i, part in enumerate(parts):
                    part_lower = part.lower()
                    # If this part starts with a direction keyword, it's a separate subtask
                    if any(part_lower.startswith(kw) for kw in ["turn ", "go ", "walk ", "move "]) and i > 0:
                        refined_segments.append(part.strip())
                    elif i == 0:
                        # First part is always added
                        refined_segments.append(part.strip())
                    else:
                        # Merge with previous if not a separate instruction
                        if refined_segments:
                            refined_segments[-1] += ", " + part.strip()
                        else:
                            refined_segments.append(part.strip())
            else:
                refined_segments.append(segment)

        return refined_segments if refined_segments else [text]

    def _determine_required_agents(self, segment: str) -> List[str]:
        """Determine which agents are needed for a segment."""
        agents = []

        segment_lower = segment.lower()

        # Check for perception needs
        if any(word in segment_lower for word in ["see", "find", "look", "observe", "detect", "locate"]):
            agents.append("perception")

        # Check for trajectory needs
        if any(word in segment_lower for word in ["go", "walk", "move", "navigate", "follow", "reach"]):
            agents.append("trajectory")

        # Perception is always needed for visual analysis
        if "perception" not in agents:
            agents.append("perception")

        # Trajectory is always needed for navigation
        if "trajectory" not in agents:
            agents.append("trajectory")

        # Decision is always needed
        agents.append("decision")

        return agents

    def _calculate_complexity(self, parsed: Dict[str, Any]) -> float:
        """Calculate instruction complexity score."""
        score = 0.0

        # Number of directions
        score += len(parsed["directions"]) * 0.1

        # Number of landmarks
        score += len(parsed["landmarks"]) * 0.15

        # Conditional logic (significant complexity)
        score += len(parsed["conditions"]) * 0.3

        # Goals
        score += len(parsed["goals"]) * 0.2

        # Sequences
        score += len(parsed["sequences"]) * 0.15

        return min(score, 1.0)

    def _generate_semantic_reasoning(
        self,
        instruction: str,
        parsed: Dict[str, Any],
        subtasks: List[SubTask],
    ) -> str:
        """Generate semantic reasoning for navigation instruction using LLM.

        Args:
            instruction: Original navigation instruction
            parsed: Parsed instruction components
            subtasks: Decomposed subtasks

        Returns:
            Semantic reasoning string
        """
        if not self._model_manager:
            return self._generate_semantic_reasoning_fallback(parsed, subtasks)

        # Build subtask summary
        subtask_summary = ""
        if subtasks:
            subtask_summary = " → ".join([s.description[:30] for s in subtasks[:4]])

        prompt = f"""/no_think
分析导航指令的语义含义，生成简洁的导航推理。

## 导航指令
{instruction}

## 解析信息
- 方向词: {parsed.get('directions', [])}
- 地标: {parsed.get('landmarks', [])}
- 目标: {parsed.get('goals', [])}

## 子任务序列
{subtask_summary if subtask_summary else "无"}

## 输出要求
生成一段简洁的语义推理（50字以内），包括：
1. 主要导航意图
2. 关键路径特征
3. 注意事项

直接输出推理结果，不要JSON格式："""

        try:
            response = self._model_manager.generate(
                "qwen-4b-instruction",
                prompt,
                max_new_tokens=80,
                temperature=0.2,
            )

            # Clean response
            reasoning = response.strip() if response else ""
            if len(reasoning) > 100:
                reasoning = reasoning[:100]
            return reasoning

        except Exception as e:
            self.logger.error(f"LLM semantic reasoning failed: {e}")
            return self._generate_semantic_reasoning_fallback(parsed, subtasks)

    def _generate_semantic_reasoning_fallback(
        self,
        parsed: Dict[str, Any],
        subtasks: List[SubTask],
    ) -> str:
        """Fallback rule-based semantic reasoning."""
        parts = []

        # Direction summary
        if parsed.get("directions"):
            dirs = "/".join(parsed["directions"][:3])
            parts.append(f"方向:{dirs}")

        # Landmark summary
        if parsed.get("landmarks"):
            landmarks = parsed["landmarks"][:2]
            parts.append(f"地标:{'/'.join(landmarks)}")

        # Goal summary
        if parsed.get("goals"):
            goals = parsed["goals"][:2]
            parts.append(f"目标:{'/'.join(goals)}")

        # Subtask count
        if subtasks:
            parts.append(f"{len(subtasks)}步")

        return "，".join(parts) if parts else "简单导航任务"

    def get_subtask_summary(self, context: NavContext) -> str:
        """Get a summary of current subtask progress."""
        if not context.subtasks:
            return "No subtasks"

        completed = sum(1 for s in context.subtasks if s.status == "completed")
        total = len(context.subtasks)

        current = context.get_current_subtask()
        current_desc = current.description[:50] if current else "None"

        return f"Subtask {completed + 1}/{total}: {current_desc}..."

    def mark_subtask_completed(self, context: NavContext, subtask_id: int = None) -> bool:
        """
        Mark a subtask as completed.

        Args:
            context: Navigation context
            subtask_id: ID of subtask to complete (None = current)

        Returns:
            True if successful
        """
        if not context.subtasks:
            return False

        if subtask_id is None:
            subtask_id = context.current_subtask_idx

        for subtask in context.subtasks:
            if subtask.id == subtask_id:
                subtask.status = "completed"
                subtask.result = "completed"

                # Advance to next subtask
                if context.current_subtask_idx < len(context.subtasks) - 1:
                    context.current_subtask_idx += 1
                    context.subtasks[context.current_subtask_idx].status = "in_progress"

                return True

        return False

    def should_replan(self, context: NavContext, evaluation_scores: List[float]) -> bool:
        """
        Determine if re-planning is needed based on evaluation scores.

        Args:
            context: Navigation context
            evaluation_scores: Recent evaluation scores

        Returns:
            True if re-planning should be triggered
        """
        if not evaluation_scores:
            return False

        # Check for continuous low scores
        if len(evaluation_scores) >= 3:
            if all(score < 0.4 for score in evaluation_scores[-3:]):
                return True

        # Check for accumulated low scores
        if len(evaluation_scores) >= 5:
            low_count = sum(1 for score in evaluation_scores if score < 0.5)
            if low_count >= 5:
                return True

        return False

    # === LLM Semantic Decomposition Methods ===

    def _semantic_decompose_with_llm(self, instruction: str, goal_position: tuple = None) -> List[SubTask]:
        """Use LLM for semantic-level subtask decomposition.

        Args:
            instruction: Navigation instruction
            goal_position: Optional goal position for spatial context

        Returns:
            List of SubTask with precondition and completion_condition
        """
        if not self._model_manager:
            self.logger.warning("Model manager not available, using rule-based decomposition")
            return None

        prompt = f"""/no_think
你是导航指令分析专家。深度分解导航指令。

## 指令
{instruction}

## 分析要求
1. **子任务分解**: 将指令分解为独立的可执行步骤
2. **空间拓扑**: 分析房间之间的连接关系
3. **条件约束**: 识别每个子任务的前置条件和完成条件

## 输出格式(JSON)
{{
  "subtasks": [
    {{
      "id": 1,
      "action": "动作描述",
      "level": "简单/中等/困难",
      "precondition": "开始此步骤的前置条件",
      "completion_condition": "如何判断此步骤完成",
      "expected_landmarks": ["预期看到的地标"],
      "spatial_constraint": "空间约束，如'必须在上楼前'"
    }}
  ],
  "spatial_topology": {{
    "rooms_sequence": ["房间序列"],
    "vertical_change": "上楼/下楼/同层",
    "total_distance_estimate": "预估距离"
  }},
  "critical_landmarks": [
    {{"name": "地标名", "role": "转折点/确认点/目标", "expected_room": "所在房间"}}
  ],
  "reasoning": "整体任务流程分析"
}}

只输出JSON。"""

        try:
            # Call LLM with qwen-4b-instruction for better instruction understanding
            response = self._model_manager.generate(
                "qwen-4b-instruction",  # Use dedicated instruction model config
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.1,
            )

            self.logger.info(f"[Instruction] LLM响应: {response[:100] if response else 'None'}...")

            if not response:
                self.logger.warning("LLM returned empty response")
                return None

            # Parse semantic response into subtasks
            return self._parse_semantic_response(response, instruction)

        except Exception as e:
            self.logger.error(f"LLM decomposition failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _parse_semantic_response(self, response: str, fallback_instruction: str) -> List[SubTask]:
        """Parse semantic LLM response into SubTask objects.

        Supports multiple formats:
        1. Enhanced JSON format (new): {"subtasks": [...], "spatial_topology": {...}}
        2. Chinese format: {子任务1：xxx，难度：简单；子任务2：xxx，难度：xx}。语义推理：xxx
        3. JSON format: [{"action":"xxx","level":"easy"}]

        Args:
            response: LLM response
            fallback_instruction: Original instruction for fallback

        Returns:
            List of SubTask objects with inferred conditions
        """
        import json

        self.logger.info(f"[Instruction] 解析响应: {response[:150] if response else 'None'}...")

        if not response:
            return None

        text = response.strip()

        # Strip markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        subtasks = []

        # 1. Try enhanced JSON format first (new format with "subtasks" key)
        json_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # Check for new enhanced format
                if "subtasks" in data and isinstance(data["subtasks"], list):
                    subtasks = self._parse_enhanced_json_format(data)
                    if subtasks:
                        self.logger.info(f"[Instruction] JSON格式: {len(subtasks)}个子任务")
                        # Store additional metadata if available
                        if "spatial_topology" in data:
                            self._last_spatial_topology = data["spatial_topology"]
                        if "critical_landmarks" in data:
                            self._last_critical_landmarks = data["critical_landmarks"]
                        return subtasks
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.debug(f"[Parse] Enhanced JSON parse failed: {e}")

        # 2. Try Chinese format: {子任务1：xxx，难度：简单；子任务2：xxx，难度：xx}
        if '子任务' in text or '难度' in text:
            subtasks = self._parse_chinese_format(text)
            if subtasks:
                self.logger.info(f"[Instruction] 中文格式: {len(subtasks)}个子任务")
                return subtasks

        # 3. Try JSON array format: [{"action":"xxx","level":"easy"}]
        if '[' in text:
            subtasks = self._parse_json_format(text)
            if subtasks:
                self.logger.info(f"[Instruction] 数组格式: {len(subtasks)}个子任务")
                return subtasks

        self.logger.warning("No valid subtask decomposition found")
        return None

    def _parse_enhanced_json_format(self, data: dict) -> List[SubTask]:
        """Parse enhanced JSON format with detailed subtask info."""
        subtasks = []

        for item in data.get("subtasks", []):
            if not isinstance(item, dict):
                continue

            action = item.get("action", "")
            if not action:
                continue

            # Map level
            level = item.get("level", "中等")
            if level in ["简单", "easy", "simple"]:
                level = "简单"
            elif level in ["困难", "hard", "difficult"]:
                level = "困难"
            else:
                level = "中等"

            # Extract precondition
            precondition = item.get("precondition", "")
            if isinstance(precondition, dict):
                precondition = str(precondition)

            # Extract completion condition
            completion_condition = None
            cc = item.get("completion_condition", "")
            if cc:
                # Try to parse as structured condition
                completion_condition = self._parse_completion_condition(cc)

            # Create SubTask
            subtask = SubTask(
                id=len(subtasks),
                description=action,
                status="pending",
                level=level,
                precondition=precondition,
                completion_condition=completion_condition,
                expected_landmarks=item.get("expected_landmarks", []),
                spatial_constraint=item.get("spatial_constraint", ""),
            )
            subtasks.append(subtask)

        return subtasks if subtasks else None

    def _parse_completion_condition(self, condition_str: str) -> Optional[Dict]:
        """Parse completion condition string into structured format."""
        condition_lower = condition_str.lower() if condition_str else ""

        # Detect condition types
        if any(kw in condition_lower for kw in ["下楼", "down the stairs", "descend"]):
            return {"type": "y_change", "direction": "down", "min_change": 1.5}
        elif any(kw in condition_lower for kw in ["上楼", "up the stairs", "ascend"]):
            return {"type": "y_change", "direction": "up", "min_change": 1.5}
        elif any(kw in condition_lower for kw in ["左转", "turn left"]):
            return {"type": "rotation", "direction": "left", "min_degrees": 70}
        elif any(kw in condition_lower for kw in ["右转", "turn right"]):
            return {"type": "rotation", "direction": "right", "min_degrees": 70}
        elif any(kw in condition_lower for kw in ["到达", "reach", "at goal"]):
            return {"type": "at_goal"}
        elif any(kw in condition_lower for kw in ["接近", "near", "close to"]):
            return {"type": "object_near", "object": condition_str}

        # Default: store as description
        return {"type": "description", "text": condition_str}

    def _parse_chinese_format(self, text: str) -> List[SubTask]:
        """Parse Chinese format: {子任务1：xxx，难度：简单；子任务2：xxx，难度：xx}"""
        subtasks = []

        # Normalize whitespace and newlines
        text = text.replace('\n', ' ').replace('\r', ' ')

        # Extract subtask section (before 语义推理)
        subtask_section = text
        if '语义推理' in text:
            parts = text.split('语义推理')
            subtask_section = parts[0]

        # Find content within {} - use DOTALL to match across lines
        match = re.search(r'\{([^}]+)\}', subtask_section)
        if not match:
            return None

        content = match.group(1)

        # Split by ； or ; to get each subtask
        items = re.split(r'[；;]', content)

        for i, item in enumerate(items):
            if not item.strip():
                continue

            # Parse: 子任务1：xxx，难度：简单 (allow space after 子任务)
            # Extract action: matches "子任务 1：" or "子任务1：" with various colon types
            action_match = re.search(r'子任务\s*\d+[：:]\s*([^，,；;]+)', item)
            level_match = re.search(r'难度[：:]\s*(简单|中等|困难)', item)

            if action_match:
                description = action_match.group(1).strip()
                level = level_match.group(1) if level_match else "中等"

                # Infer completion condition
                _, condition = self._extract_action_and_condition(description)

                subtask = SubTask(
                    id=i,
                    description=description,
                    status="pending" if i > 0 else "in_progress",
                    level=level,
                    precondition={"type": "sequence", "after": i} if i > 0 else None,
                    completion_condition=condition,
                    required_agents=self._determine_required_agents(description),
                )
                subtasks.append(subtask)

        return subtasks if subtasks else None

    def _parse_json_format(self, text: str) -> List[SubTask]:
        """Parse JSON format: [{"action":"xxx","level":"easy"}]"""
        import json

        # Find JSON array
        start = text.find('[')
        if start < 0:
            return None

        depth = 0
        json_str = None
        for i, c in enumerate(text[start:]):
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    json_str = text[start:start+i+1]
                    break

        if not json_str:
            return None

        try:
            subtasks_data = json.loads(json_str)
            if not isinstance(subtasks_data, list):
                return None

            subtasks = []
            for i, item in enumerate(subtasks_data):
                if not isinstance(item, dict):
                    continue

                description = item.get("action", item.get("description", ""))
                level_raw = item.get("level", "medium")

                # Map level
                level_map = {"easy": "简单", "medium": "中等", "hard": "困难",
                            "简单": "简单", "中等": "中等", "困难": "困难"}
                level = level_map.get(level_raw.lower() if isinstance(level_raw, str) else "medium", "中等")

                if description:
                    _, condition = self._extract_action_and_condition(description)

                    subtask = SubTask(
                        id=i,
                        description=description,
                        status="pending" if i > 0 else "in_progress",
                        level=level,
                        precondition={"type": "sequence", "after": i} if i > 0 else None,
                        completion_condition=condition,
                        required_agents=self._determine_required_agents(description),
                    )
                    subtasks.append(subtask)

            return subtasks if subtasks else None

        except json.JSONDecodeError:
            return None

    def _extract_action_and_condition_fallback(self, segment: str) -> tuple:
        """Fallback rule-based action and condition extraction."""
        segment = segment.strip()

        # Detect key action patterns
        action = None
        condition = None

        # Stairs down
        if any(kw in segment.lower() for kw in ["down the stairs", "下楼", "下楼梯"]):
            action = "Walk down the stairs"
            condition = {"type": "y_change", "direction": "down", "min_change": 1.5}

        # Stairs up
        elif any(kw in segment.lower() for kw in ["up the stairs", "上楼", "上楼梯"]):
            action = "Walk up the stairs"
            condition = {"type": "y_change", "direction": "up", "min_change": 1.5}

        # Turn right
        elif any(kw in segment.lower() for kw in ["turn right", "右转", "向右"]):
            action = "Turn right"
            condition = {"type": "rotation", "direction": "right", "min_degrees": 70}

        # Turn left
        elif any(kw in segment.lower() for kw in ["turn left", "左转", "向左"]):
            action = "Turn left"
            condition = {"type": "rotation", "direction": "left", "min_degrees": 70}

        # Walk towards / approach
        elif any(kw in segment.lower() for kw in ["walk towards", "towards", "走向", "前往"]):
            # Extract target
            for target in ["rug", "carpet", "bench", "piano", "door", "room"]:
                if target in segment.lower():
                    action = f"Walk towards {target}"
                    break
            if not action:
                action = "Walk towards target"
            condition = {"type": "distance", "min_meters": 2.0}

        # Wait
        elif any(kw in segment.lower() for kw in ["wait", "等待", "停下"]):
            action = "Wait"
            condition = {"type": "wait", "steps": 3}

        # Generic movement
        elif any(kw in segment.lower() for kw in ["walk", "走", "move", "前进"]):
            action = segment[:40] if len(segment) > 40 else segment
            condition = {"type": "distance", "min_meters": 1.5}

        # Use segment as action if nothing matched but it's meaningful
        elif len(segment) > 3:
            action = segment[:50]
            condition = None

        return action, condition

    def _parse_llm_subtasks(self, response: str, fallback_instruction: str) -> List[SubTask]:
        """Parse LLM response into SubTask objects.

        Args:
            response: LLM response text
            fallback_instruction: Original instruction for fallback

        Returns:
            List of SubTask objects
        """
        try:
            # Extract JSON array
            import json
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                subtasks_data = json.loads(json_match.group())

                subtasks = []
                for i, data in enumerate(subtasks_data):
                    # Support both simplified format (d, p, c, s) and full format
                    description = data.get("d") or data.get("description", "")
                    precondition_str = data.get("p") or data.get("precondition", "")
                    completion_str = data.get("c") or data.get("completion_condition", "")
                    spatial_str = data.get("s") or data.get("spatial_constraint", "")

                    # Parse condition strings to structured format
                    precondition = self._parse_condition_string(precondition_str) if precondition_str else None
                    completion_condition = self._parse_condition_string(completion_str) if completion_str else None
                    spatial_constraint = {"description": spatial_str} if spatial_str else None

                    subtask = SubTask(
                        id=i,
                        description=description,
                        status="pending" if i > 0 else "in_progress",
                        level=self._determine_subtask_level(description),
                        precondition=precondition,
                        completion_condition=completion_condition,
                        spatial_constraint=spatial_constraint,
                        required_agents=self._determine_required_agents(description),
                    )
                    subtasks.append(subtask)

                if subtasks:
                    self.logger.info(f"[Instruction] 创建{len(subtasks)}个子任务(含语义条件)")
                    return subtasks

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed: {e}")
        except Exception as e:
            self.logger.warning(f"LLM parsing error: {e}")

        # Fallback to rule-based decomposition
        self.logger.info("Using rule-based decomposition fallback")
        return None

    def _parse_condition_string_fallback(self, condition_str: str) -> Optional[Dict[str, Any]]:
        """Fallback rule-based condition parsing."""
        condition_str = condition_str.lower()

        # Height change patterns
        if "高度下降" in condition_str or "下楼" in condition_str or "down the stairs" in condition_str:
            return {"type": "y_change", "direction": "down", "min_change": 1.5, "description": condition_str}
        elif "高度上升" in condition_str or "上楼" in condition_str or "up the stairs" in condition_str:
            return {"type": "y_change", "direction": "up", "min_change": 1.5, "description": condition_str}

        # Rotation patterns
        if "右转" in condition_str or "turn right" in condition_str or "右旋转" in condition_str:
            return {"type": "rotation", "direction": "right", "min_degrees": 60, "description": condition_str}
        elif "左转" in condition_str or "turn left" in condition_str or "左旋转" in condition_str:
            return {"type": "rotation", "direction": "left", "min_degrees": 60, "description": condition_str}

        # Distance/approach patterns
        if "接近" in condition_str or "near" in condition_str or "reach" in condition_str or "arrive" in condition_str:
            # Extract object name
            for keyword in ["地毯", "rug", "bench", "piano", "door", "stairs", "staircase"]:
                if keyword in condition_str:
                    return {"type": "object_near", "object": keyword, "max_distance": 2.0, "description": condition_str}

        # Walk/distance patterns
        if "walk" in condition_str or "move" in condition_str or "走" in condition_str:
            return {"type": "distance", "min_meters": 3.0, "description": condition_str}

        # At goal pattern
        if "at goal" in condition_str or "目标" in condition_str or "目的地" in condition_str:
            return {"type": "at_goal", "max_distance": 2.0, "description": condition_str}

        # Return as generic condition with description
        return {"type": "generic", "description": condition_str}

    def decompose_with_llm(self, instruction: str, goal_position: tuple = None) -> List[SubTask]:
        """Public method for LLM-based decomposition.

        Args:
            instruction: Navigation instruction
            goal_position: Optional goal position

        Returns:
            List of SubTask objects
        """
        if not self.use_llm_decompose or not self._model_manager:
            return None

        return self._semantic_decompose_with_llm(instruction, goal_position)

    def build_debate_opinion(
        self,
        context: "NavContext",
        agent_outputs: Dict[str, Any] = None
    ) -> "DebateOpinion":
        """Build a DebateOpinion using LLM for instruction-based analysis.

        Args:
            context: Navigation context
            agent_outputs: Outputs from other agents (for perception data)

        Returns:
            DebateOpinion with instruction-based constraints
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        current_subtask = context.get_current_subtask()
        if not current_subtask:
            return DebateOpinion(
                agent="instruction",
                primary_action="forward",
                confidence=0.5,
                evidence={},
                reasoning="无当前子任务",
                constraints={},
            )

        # Use LLM to analyze subtask and determine action
        if self._model_manager:
            return self._build_debate_opinion_with_llm(context, current_subtask, agent_outputs)
        else:
            return self._build_debate_opinion_fallback(context, current_subtask, agent_outputs)

    def _build_debate_opinion_with_llm(
        self,
        context: "NavContext",
        current_subtask,
        agent_outputs: Dict[str, Any] = None
    ) -> "DebateOpinion":
        """Build debate opinion using semantic reasoning + rules."""
        from core.debate_types import DebateOpinion, ActionConstraint

        description = current_subtask.description

        # Get parsed info from context
        instruction_output = context.metadata.get("instruction_output", {})
        parsed = instruction_output.get("parsed_instruction", {})

        # Generate semantic reasoning using LLM (single call)
        semantic_reasoning = self._generate_semantic_reasoning(
            context.instruction,
            parsed,
            context.subtasks or []
        )

        # Determine primary action using rules (no LLM)
        primary_action, confidence = self._determine_primary_action(
            description, context, agent_outputs
        )

        # Build constraints
        constraints = {"hard": [], "soft": []}
        description_lower = description.lower()

        if "turn right" in description_lower or "右转" in description_lower:
            constraints["soft"].append(ActionConstraint(
                action="turn_right", weight_multiplier=1.5, reason="子任务要求右转"
            ))
        elif "turn left" in description_lower or "左转" in description_lower:
            constraints["soft"].append(ActionConstraint(
                action="turn_left", weight_multiplier=1.5, reason="子任务要求左转"
            ))

        # Calculate subtask progress
        total_subtasks = len(context.subtasks) if context.subtasks else 1
        completed = sum(1 for s in context.subtasks if s.status == "completed") if context.subtasks else 0
        progress = completed / total_subtasks if total_subtasks > 0 else 0

        return DebateOpinion(
            agent="instruction",
            primary_action=primary_action,
            confidence=confidence,
            evidence={
                "current_subtask": description[:60],
                "subtask_progress": progress,
            },
            reasoning=semantic_reasoning,
            constraints=constraints,
        )

    def _determine_primary_action(
        self,
        description: str,
        context: "NavContext",
        agent_outputs: Dict[str, Any] = None
    ) -> tuple:
        """Determine primary action using rules."""
        description_lower = description.lower()

        # Turn actions
        if "turn left" in description_lower or "左转" in description_lower:
            return "turn_left", 0.9
        if "turn right" in description_lower or "右转" in description_lower:
            return "turn_right", 0.9

        # Stop actions
        if "stop" in description_lower or "wait" in description_lower:
            # Check if near goal
            if agent_outputs:
                perception = agent_outputs.get("perception", {})
                landmarks = perception.get("landmarks", [])
                for lm in landmarks:
                    if lm.get("distance", 999) < 1.5:
                        return "stop", 0.85
            return "forward", 0.6

        # Stairs
        if "stairs" in description_lower or "楼梯" in description_lower:
            return "forward", 0.75

        # Default: move forward
        return "forward", 0.7

    def _build_debate_opinion_fallback(
        self,
        context: "NavContext",
        current_subtask,
        agent_outputs: Dict[str, Any] = None
    ) -> "DebateOpinion":
        """Fallback rule-based debate opinion."""
        from core.debate_types import DebateOpinion, ActionConstraint

        description = current_subtask.description.lower()

        # Determine primary action from subtask
        primary_action = "forward"
        confidence = 0.7

        if "turn left" in description or "左转" in description:
            primary_action = "turn_left"
            confidence = 0.9
        elif "turn right" in description or "右转" in description:
            primary_action = "turn_right"
            confidence = 0.9
        elif "stop" in description or "wait" in description:
            primary_action = "stop"
            confidence = 0.8
        elif "stairs" in description or "下楼" in description or "上楼" in description:
            primary_action = "forward"
            confidence = 0.75

        # Build constraints
        constraints = {"hard": [], "soft": []}

        if primary_action == "forward":
            constraints["soft"].append(ActionConstraint(
                action="forward",
                weight_multiplier=1.1,
                reason="goal_aligned",
            ))

        # Calculate subtask progress
        total_subtasks = len(context.subtasks) if context.subtasks else 1
        completed = sum(1 for s in context.subtasks if s.status == "completed") if context.subtasks else 0
        progress = completed / total_subtasks if total_subtasks > 0 else 0

        return DebateOpinion(
            agent="instruction",
            primary_action=primary_action,
            confidence=confidence,
            evidence={
                "current_subtask": description[:60],
                "subtask_progress": progress,
            },
            reasoning=f"子任务: {description[:40]}",
            constraints=constraints,
        )