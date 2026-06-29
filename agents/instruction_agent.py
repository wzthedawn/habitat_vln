"""Instruction Agent for decomposing and interpreting navigation instructions.

This version uses rule-based parsing without LLM for subtask decomposition
and task level classification (简单/中等/困难).
"""

from typing import Dict, Any, Optional, List, Tuple
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

    # Emergency instruction template regex - flexible for variable spacing and comma handling
    EMERGENCY_TEMPLATE_PATTERN = r"(.+?),\s*the path is blocked,?\s*(.+?)\s+to\s+reach\s+(.+)"

    # Emergency instruction keywords for detection
    EMERGENCY_KEYWORDS = [
        "blocked", "obstacle", "emergency", "urgent",
        "suddenly", "path is blocked", "route blocked",
        "obstacle detected", "avoid the obstacle"
    ]

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

            # Check if subtasks already exist with runtime state (start_context)
            # If so, preserve them instead of recreating
            existing_subtasks = context.subtasks
            preserve_runtime_state = False

            if existing_subtasks and len(existing_subtasks) > 0:
                # Check if any subtask has start_context set (indicating runtime state)
                for st in existing_subtasks:
                    if st.start_context is not None:
                        preserve_runtime_state = True
                        break

            if preserve_runtime_state:
                # Use existing subtasks with their runtime state
                subtasks = existing_subtasks
                self.logger.info(f"[InstructionAgent] Preserving existing subtasks with runtime state ({len(subtasks)} subtasks)")
            else:
                # Create new subtasks (with LLM semantic decomposition if available)
                subtasks = self._create_subtasks(parsed, task_level, goal_position)
                # Update context subtasks only for new creation
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
                            "start_context": s.start_context,
                            "end_context": s.end_context,
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
        # Check for conditional keywords (hard)
        if parsed["conditions"]:
            return "hard"

        # Check for multiple sequences or multiple landmarks (medium or hard)
        num_landmarks = len(parsed["landmarks"])
        num_directions = len(parsed["directions"])
        num_goals = len(parsed["goals"])
        num_sequences = len(parsed["sequences"])

        # hard: Multiple sequences + multiple landmarks + conditions
        if num_sequences >= 2 and num_landmarks >= 2:
            return "hard"

        # medium: Single goal with landmarks, or multiple directions
        if num_landmarks >= 1 or num_goals >= 1 or num_directions >= 2:
            return "medium"

        # easy: Basic operations
        if num_directions <= 1 and num_landmarks == 0 and num_goals == 0:
            return "easy"

        # Default to medium
        return "medium"

    def _determine_subtask_level_fallback(self, segment: str) -> str:
        """Fallback rule-based subtask level determination."""
        segment_lower = segment.lower()

        # Extract features from current subtask
        has_direction = any(word in segment_lower for word in ["turn", "left", "right", "forward", "back", "straight"])
        has_landmark = any(word in segment_lower for word in self.LANDMARK_KEYWORDS)
        has_goal = any(word in segment_lower for word in ["find", "reach", "go to", "stop at", "locate", "arrive", "walk to", "go through"])
        has_condition = any(word in segment_lower for word in self.CONDITIONAL_KEYWORDS)
        has_sequence = any(word in segment_lower for word in self.SEQUENCE_KEYWORDS)

        # easy: Basic operations (pure direction commands, no landmarks)
        if has_direction and not has_landmark and not has_goal:
            return "easy"

        # hard: Conditional judgment or multi-step sequences
        if has_condition or has_sequence:
            return "hard"

        # medium: Involves landmarks or goals
        if has_landmark or has_goal:
            return "medium"

        # Default to medium
        return "medium"

    def _create_subtasks(self, parsed: Dict[str, Any], task_level: str, goal_position: tuple = None) -> List[SubTask]:
        """Create subtasks from parsed instruction with individual difficulty levels.

        Tries LLM semantic decomposition first, falls back to rule-based.
        """
        instruction = parsed["original"]

        # Try LLM semantic decomposition first
        if self.use_llm_decompose and self._model_manager:
            llm_subtasks = self._semantic_decompose_with_llm(instruction, goal_position)
            if llm_subtasks:
                self.logger.info(f"[Instruction] Semantic decomposition: {len(llm_subtasks)} subtasks")
                # Print to console
                print(f"\n[InstructionAgent] Decomposed into {len(llm_subtasks)} subtasks:")
                for i, st in enumerate(llm_subtasks):
                    print(f"  [{i}] {st.description[:50]}... (level: {st.level})")
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

    def _parse_emergency_instruction(self, instruction: str) -> Optional[List[str]]:
        """Parse emergency instruction template format.

        Emergency instruction template: "{normal_action}, the path is blocked, {reroute_action} to reach {goal}"

        Args:
            instruction: Navigation instruction

        Returns:
            List of parsed subtasks, or None if no match
        """
        match = re.match(self.EMERGENCY_TEMPLATE_PATTERN, instruction, re.IGNORECASE)
        if match:
            normal_action = match.group(1).strip()  # "Go straight"
            reroute_action = match.group(2).strip()  # "go back and try different path"
            goal = match.group(3).strip()  # "the table"

            self.logger.info(f"[Instruction] Matched emergency template: action={normal_action}, reroute={reroute_action}, goal={goal}")

            return [
                f"First attempt: {normal_action}",
                f"Path blocked detected, executing: {reroute_action}",
                f"Continue to reach {goal}"
            ]
        return None

    def _is_emergency_instruction(self, instruction: str) -> bool:
        """检测是否为应急指令

        通过关键词检测识别应急类型的导航指令。

        Args:
            instruction: 导航指令文本

        Returns:
            True如果是应急指令，False否则
        """
        instruction_lower = instruction.lower()
        return any(kw in instruction_lower for kw in self.EMERGENCY_KEYWORDS)

    def _emergency_decompose_with_llm(self, instruction: str) -> Optional[List[str]]:
        """使用LLM分解应急指令

        专门用于应急指令的智能分解，输出稳定的JSON格式。

        Args:
            instruction: 应急导航指令

        Returns:
            分解后的子任务列表，失败返回None
        """
        if not self._model_manager:
            return None

        prompt = f"""/no_think
You are a navigation instruction analyzer. Break down this emergency instruction into 2-3 clear subtasks.

## Instruction
{instruction}

## Rules
1. Each subtask must be a complete sentence (verb + direction + object)
2. Typical emergency pattern: detect obstacle → find alternative route → continue to goal
3. Output strict JSON only, no other text

## Output Format
{{"subtasks":["Complete sentence 1","Complete sentence 2","Complete sentence 3"]}}

## Example
Input: "Move forward, suddenly blocked, quickly navigate around the obstacle."
Output: {{"subtasks":["First attempt to move forward along the planned path","Obstacle detected on the path, find an alternative route around it","Continue navigating toward the goal destination"]}}

Output JSON only:"""

        try:
            response = self._model_manager.generate(
                "qwen-9b-instruction",
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.1,
            )

            if response:
                import json
                data = json.loads(response)
                subtasks = data.get("subtasks", [])
                if subtasks and len(subtasks) >= 2:
                    self.logger.info(f"[Instruction] Emergency decomposition: {len(subtasks)} subtasks")
                    return subtasks
        except json.JSONDecodeError:
            self.logger.warning("[Instruction] Failed to parse emergency decomposition JSON")
        except Exception as e:
            self.logger.error(f"[Instruction] Emergency decomposition error: {e}")

        return None

    def _split_instruction(self, text: str) -> List[str]:
        """Split instruction into subtask segments with multi-level decomposition.

        Decomposition flow:
        1. Emergency detection via keyword matching (_is_emergency_instruction)
        2. If emergency: try LLM intelligent decomposition (_emergency_decompose_with_llm)
        3. Fallback: regex template parsing (_parse_emergency_instruction)
        4. Final fallback: general splitting by conjunctions and actions

        Args:
            text: Navigation instruction text

        Returns:
            List of subtask description strings
        """

        # 首先检测是否为应急指令
        if self._is_emergency_instruction(text):
            self.logger.info(f"[Instruction] Detected emergency instruction: {text[:50]}...")
            # 尝试LLM智能分解
            llm_subtasks = self._emergency_decompose_with_llm(text)
            if llm_subtasks:
                self.logger.info(f"[Instruction] LLM decomposition returned {len(llm_subtasks)} subtasks")
                return llm_subtasks
            else:
                self.logger.warning("[Instruction] LLM decomposition failed, falling back to regex")

        # 原有逻辑：检测应急指令模板
        emergency_segments = self._parse_emergency_instruction(text)
        if emergency_segments:
            return emergency_segments

        # Key action keywords that typically start new subtasks
        action_keywords = ["turn", "walk", "go", "move", "stop", "wait", "find", "enter", "exit", "head", "pass", "cross"]

        # Common conjunctions for navigation instructions
        conjunctions = [
            ", and ", " and then ", ", then ", " then ", "; ",
            " after that ", " next ", " finally ",
        ]

        # Step 1: Split by conjunctions
        segments = [text]
        for conj in conjunctions:
            new_segments = []
            for segment in segments:
                parts = segment.split(conj)
                new_segments.extend(parts)
            segments = new_segments

        # Step 2: Split by periods
        final_segments = []
        for segment in segments:
            parts = segment.split(".")
            final_segments.extend(p.strip() for p in parts if p.strip())

        # Step 3: Enhanced comma splitting with action keyword detection
        refined_segments = []
        for segment in final_segments:
            if ", " in segment:
                parts = segment.split(", ")
                for i, part in enumerate(parts):
                    part_lower = part.lower().strip()
                    # Check if this part starts with an action keyword
                    starts_with_action = any(part_lower.startswith(kw) or part_lower.startswith(f"and {kw}")
                                            for kw in action_keywords)

                    if starts_with_action and i > 0:
                        # This is a new subtask
                        refined_segments.append(part.strip())
                    elif i == 0:
                        # First part is always added
                        refined_segments.append(part.strip())
                    else:
                        # Check for implicit action patterns
                        # e.g., "the rug" after "walk towards" → merge with previous
                        if refined_segments and any(kw in refined_segments[-1].lower() for kw in ["towards", "to", "near", "by"]):
                            refined_segments[-1] += ", " + part.strip()
                        else:
                            refined_segments.append(part.strip())
            else:
                refined_segments.append(segment)

        # Step 4: Merge very short segments with previous if they're continuations
        merged_segments = []
        for segment in refined_segments:
            # Skip very short segments that are just conjunctions
            if len(segment.strip()) < 5 and any(segment.lower().strip() == conj.strip() for conj in ["and", "then"]):
                continue

            # Check if segment is just a landmark (no action) - merge with previous
            if len(segment.split()) <= 2 and not any(kw in segment.lower() for kw in action_keywords):
                if merged_segments:
                    merged_segments[-1] += " " + segment.strip()
                else:
                    merged_segments.append(segment.strip())
            else:
                merged_segments.append(segment)

        return merged_segments if merged_segments else [text]

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
Analyze the semantic meaning of the navigation instruction and generate concise navigation reasoning.

## Navigation Instruction
{instruction}

## Parsed Info
- Directions: {parsed.get('directions', [])}
- Landmarks: {parsed.get('landmarks', [])}
- Goals: {parsed.get('goals', [])}

## Subtask Sequence
{subtask_summary if subtask_summary else "none"}

## Output Requirements
Generate a concise semantic reasoning (within 50 words), including:
1. Main navigation intent
2. Key path features
3. Notes/attention points

Output reasoning result directly, no JSON format:"""

        try:
            response = self._model_manager.generate(
                "qwen-9b-instruction",
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
            parts.append(f"directions:{dirs}")

        # Landmark summary
        if parsed.get("landmarks"):
            landmarks = parsed["landmarks"][:2]
            parts.append(f"landmarks:{'/'.join(landmarks)}")

        # Goal summary
        if parsed.get("goals"):
            goals = parsed["goals"][:2]
            parts.append(f"goals:{'/'.join(goals)}")

        # Subtask count
        if subtasks:
            parts.append(f"{len(subtasks)} steps")

        return ", ".join(parts) if parts else "Simple navigation task"

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
你是导航指令分析专家。将指令分解为可执行的子任务。

## 指令
{instruction}

## 分解规则
1. 每个子任务必须是完整的句子（动词+方向+目标）
2. 应急指令（含"blocked"、"obstacle"、"path"）分解为：检测障碍 → 寻找替代路线 → 继续前进
3. 不要按逗号机械分割，要理解语义
4. 子任务数量控制在2-4个

## 输出格式（严格JSON，无其他内容）
{{
  "subtasks": [
    {{
      "id": 0,
      "description": "检测到路径阻塞，准备绕行",
      "level": "easy"
    }},
    {{
      "id": 1,
      "description": "寻找替代路线避开障碍物",
      "level": "medium"
    }},
    {{
      "id": 2,
      "description": "继续向目标前进",
      "level": "easy"
    }}
  ]
}}

只输出JSON，无其他内容。"""

        try:
            # Call LLM with qwen-9b-instruction for better instruction understanding
            response = self._model_manager.generate(
                "qwen-9b-instruction",  # Use dedicated instruction model config
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.1,
            )

            self.logger.info(f"[Instruction] LLM response: {response[:100] if response else 'None'}...")

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

        self.logger.info(f"[Instruction] Parsing response: {response[:150] if response else 'None'}...")

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
                        self.logger.info(f"[Instruction] JSON format: {len(subtasks)} subtasks")
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
                self.logger.info(f"[Instruction] Chinese format: {len(subtasks)} subtasks")
                return subtasks

        # 3. Try JSON array format: [{"action":"xxx","level":"easy"}]
        if '[' in text:
            subtasks = self._parse_json_format(text)
            if subtasks:
                self.logger.info(f"[Instruction] Array format: {len(subtasks)} subtasks")
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
            level = item.get("level", "medium")
            if level in ["简单", "easy", "simple"]:
                level = "easy"
            elif level in ["困难", "hard", "difficult"]:
                level = "hard"
            else:
                level = "medium"

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

            # Parse: subtask 1: xxx, level: easy (allow space after subtask)
            # Extract action: matches "subtask 1:" or "子任务1：" with various colon types
            action_match = re.search(r'(?:subtask|子任务)\s*\d+[：:]\s*([^，,；;]+)', item)
            level_match = re.search(r'(?:level|难度)[：:]\s*(easy|medium|hard|简单|中等|困难)', item)

            if action_match:
                description = action_match.group(1).strip()
                level_raw = level_match.group(1) if level_match else "medium"
                # Normalize level
                level_map = {"easy": "easy", "medium": "medium", "hard": "hard",
                            "简单": "easy", "中等": "medium", "困难": "hard"}
                level = level_map.get(level_raw.lower() if isinstance(level_raw, str) else "medium", "medium")

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

    def _extract_action_and_condition(self, description: str) -> Tuple[str, Dict[str, Any]]:
        """从子任务描述中提取动作和条件。

        Args:
            description: 子任务描述文本

        Returns:
            (action, condition) 元组，condition为字典
        """
        action_verbs = ["go", "move", "turn", "walk", "proceed", "continue", "find", "avoid", "navigate"]

        # 提取动词作为动作
        action = "navigate"
        desc_lower = description.lower()
        for verb in action_verbs:
            if verb in desc_lower:
                action = verb
                break

        # Use _parse_condition_string_fallback for proper condition parsing
        # This includes obstacle_detected and obstacle_cleared patterns
        condition = self._parse_condition_string_fallback(description)
        if not condition:
            condition = {"type": "description", "description": description.strip()}

        return action, condition

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

                # Map level (normalize to English)
                level_map = {"easy": "easy", "medium": "medium", "hard": "hard",
                            "简单": "easy", "中等": "medium", "困难": "hard"}
                level = level_map.get(level_raw.lower() if isinstance(level_raw, str) else "medium", "medium")

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

    def _parse_condition_string_fallback(self, condition_str: str) -> Optional[Dict[str, Any]]:
        """Fallback rule-based condition parsing with enhanced pattern recognition.

        Enhanced rules:
        1. More accurate height change detection (stairs up/down)
        2. Rotation direction extraction with landmark context
        3. Object/landmark approach detection
        4. Spatial relationship parsing
        """
        condition_str_lower = condition_str.lower()

        # === Height change patterns (stairs) ===
        if any(kw in condition_str_lower for kw in ["down the stairs", "下楼", "下楼梯", "descend", "stair going down", "downstairs"]):
            return {"type": "y_change", "direction": "down", "min_change": 1.5, "description": condition_str}
        elif any(kw in condition_str_lower for kw in ["up the stairs", "上楼", "上楼梯", "ascend", "stair going up", "upstairs"]):
            return {"type": "y_change", "direction": "up", "min_change": 1.5, "description": condition_str}

        # === Rotation patterns with direction ===
        # Right turn
        if any(kw in condition_str_lower for kw in ["turn right", "右转", "向右转", "right turn", "turn to the right"]):
            return {"type": "rotation", "direction": "right", "min_degrees": 60, "description": condition_str}
        # Left turn
        elif any(kw in condition_str_lower for kw in ["turn left", "左转", "向左转", "left turn", "turn to the left"]):
            return {"type": "rotation", "direction": "left", "min_degrees": 60, "description": condition_str}

        # === Object/landmark approach patterns ===
        # Common landmark objects
        landmarks = ["rug", "carpet", "bench", "piano", "door", "stairs", "staircase", "kitchen",
                     "bedroom", "bathroom", "hallway", "corridor", "table", "chair", "sofa",
                     "window", "exit", "entrance", "floor", "wall"]

        # "walk towards/to/near/by [landmark]" patterns
        approach_patterns = [
            r"walk\s+(towards|to|near|by)\s+(?:the\s+)?(\w+)",
            r"go\s+(towards|to|near|by)\s+(?:the\s+)?(\w+)",
            r"reach\s+(?:the\s+)?(\w+)",
            r"find\s+(?:the\s+)?(\w+)",
            r"stop\s+(at|by|near)\s+(?:the\s+)?(\w+)",
            r"wait\s+(at|by|near)\s+(?:the\s+)?(\w+)",
        ]

        import re
        for pattern in approach_patterns:
            match = re.search(pattern, condition_str_lower)
            if match:
                target = match.group(match.lastindex).strip()
                # Check if target is a known landmark
                if target in landmarks or any(lk in target for lk in landmarks):
                    return {"type": "object_near", "object": target, "max_distance": 2.0, "description": condition_str}

        # === Spatial relationship patterns ===
        # "along [direction] side of wall/floor"
        wall_patterns = [
            r"(?:along|on)\s+(?:the\s+)?(left|right)\s+side\s+of\s+(?:the\s+)?(\w+)",
            r"(?:beside|near)\s+(?:the\s+)?(\w+)\s+(?:along|on)\s+(?:the\s+)?(left|right)",
        ]

        for pattern in wall_patterns:
            match = re.search(pattern, condition_str_lower)
            if match:
                direction = match.group(1) if "left" in match.group(1) or "right" in match.group(1) else match.group(2)
                object_name = match.group(2) if direction == match.group(1) else match.group(1)
                return {"type": "spatial_position", "direction": direction, "object": object_name, "description": condition_str}

        # === Distance/walking patterns ===
        if any(kw in condition_str_lower for kw in ["walk", "move", "走", "前进", "forward", "straight"]):
            # Check for specific distance mentions
            dist_match = re.search(r"(\d+)\s*(meter|m|step)", condition_str_lower)
            if dist_match:
                distance = float(dist_match.group(1))
                return {"type": "distance", "min_meters": distance, "description": condition_str}
            return {"type": "distance", "min_meters": 3.0, "description": condition_str}

        # === Obstacle avoidance patterns (NEW) ===
        # For emergency navigation subtasks
        obstacle_keywords = ["障碍物", "障碍", "obstacle", "blocked", "阻塞", "路径阻塞", "挡住"]
        avoidance_keywords = ["避开", "绕行", "绕开", "avoid", "绕过", "alternate route", "替代路线"]

        # Obstacle detection subtask: completed when moved at least 1m (reacted to obstacle)
        if any(kw in condition_str_lower for kw in obstacle_keywords) and not any(kw in condition_str_lower for kw in avoidance_keywords):
            return {"type": "obstacle_detected", "min_distance_moved": 1.0, "description": condition_str}

        # Obstacle avoidance subtask: completed when obstacle is cleared (distance > 3m from obstacle position)
        if any(kw in condition_str_lower for kw in avoidance_keywords):
            return {"type": "obstacle_cleared", "min_obstacle_distance": 3.0, "description": condition_str}

        # === Goal/destination patterns ===
        if any(kw in condition_str_lower for kw in ["at goal", "目标", "目的地", "destination", "end point", "final"]):
            return {"type": "at_goal", "max_distance": 2.0, "description": condition_str}

        # === Wait/stop patterns ===
        if any(kw in condition_str_lower for kw in ["wait", "stop", "停下", "等待"]):
            return {"type": "wait", "steps": 3, "description": condition_str}

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
                reasoning="No current subtask",
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
                action="turn_right", weight_multiplier=1.5, reason="Subtask requires right turn"
            ))
        elif "turn left" in description_lower or "左转" in description_lower:
            constraints["soft"].append(ActionConstraint(
                action="turn_left", weight_multiplier=1.5, reason="Subtask requires left turn"
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
            reasoning=f"Subtask: {description[:40]}",
            constraints=constraints,
        )