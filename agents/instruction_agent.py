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

            # Parse instruction
            parsed = self._parse_instruction(instruction)

            # Determine task level
            task_level = self._determine_task_level(parsed)

            # Create subtasks
            subtasks = self._create_subtasks(parsed, task_level)

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
                            "level": s.level,  # Include individual subtask level
                        } for s in subtasks
                    ],
                    "landmarks": parsed["landmarks"],
                    "goals": parsed["goals"],
                    "directions": parsed["directions"],
                    "task_level": task_level,  # Overall task level (for reference)
                    "complexity": parsed["complexity"],
                    "parsed_instruction": parsed,
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

    def _determine_task_level(self, parsed: Dict[str, Any]) -> str:
        """
        Determine task difficulty level for the entire instruction.

        Returns:
            "简单", "中等", or "困难"
        """
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

    def _determine_subtask_level(self, segment: str) -> str:
        """
        Determine difficulty level for a single subtask segment.

        Args:
            segment: Subtask description text

        Returns:
            "简单", "中等", or "困难"
        """
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

    def _create_subtasks(self, parsed: Dict[str, Any], task_level: str) -> List[SubTask]:
        """Create subtasks from parsed instruction with individual difficulty levels."""
        subtasks = []

        # Split instruction by conjunctions and sequences
        instruction = parsed["original"]
        segments = self._split_instruction(instruction)

        for i, segment in enumerate(segments):
            # Determine if this segment is completed
            status = "in_progress" if i == 0 else "pending"

            # Calculate individual level for each subtask
            subtask_level = self._determine_subtask_level(segment)

            subtask = SubTask(
                id=i,
                description=segment.strip(),
                status=status,
                level=subtask_level,  # Use subtask's own level
                required_agents=self._determine_required_agents(segment),
            )
            subtasks.append(subtask)

        # If no subtasks created, create a single one for the whole instruction
        if not subtasks:
            subtasks.append(SubTask(
                id=0,
                description=instruction,
                status="in_progress",
                level=task_level,  # Fall back to overall task level
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