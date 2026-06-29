"""SubtaskDecompositionAgent - LLM-driven instruction decomposition.

Decomposes navigation instruction into multiple subtasks with
verifiable completion conditions and relevant objects for visual focus.

Responsibilities:
- Parse instruction text
- Identify subtask boundaries (stairs, turns, object approach)
- Generate completion_condition for each subtask
- Generate relevant_objects for visual focus
"""

import json
import re
import logging
from typing import Dict, Any, List

from agents.pipeline.base_pipeline_agent import SubAgent, DecompositionOutput
from agents.base_agent import AgentRole


class SubtaskDecompositionAgent(SubAgent):
    """Subtask Decomposition Agent - LLM core.

    Uses LLM to decompose navigation instruction into multiple subtasks.
    Each subtask has:
    - id: Sequential number
    - description: Clear action description with implicit direction
    - completion_condition: Verifiable condition type
    - relevant_objects: Objects the vision model should focus on
    """

    name = "subtask_decomposition_agent"

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SubtaskDecompositionAgent.

        Args:
            config: Agent configuration dictionary
                - model_key: LLM for decomposition (default: "qwen3.5-9b-fast")
        """
        super().__init__(config)
        self.logger = logging.getLogger("SubtaskDecompositionAgent")
        # Default to fast model for one-time decomposition
        if "model_key" not in self.config:
            self.config["model_key"] = "qwen3.5-9b-fast"

    def process(
        self,
        instruction: str,
        goal_position: List[float],
        start_position: List[float],
    ) -> DecompositionOutput:
        """Decompose instruction into subtasks.

        Args:
            instruction: Full navigation instruction text
            goal_position: Final goal position [x, y, z]
            start_position: Starting position [x, y, z]

        Returns:
            DecompositionOutput with subtasks list and reasoning
        """
        # Fallback if no model manager
        if self._model_manager is None:
            self.logger.warning("[SubtaskDecompositionAgent] No ModelManager, using fallback")
            return self._fallback_decomposition(instruction, goal_position)

        # Build prompt
        prompt = self._build_prompt(instruction, goal_position, start_position)

        # Call LLM
        try:
            response = self._call_llm(prompt, max_tokens=500, temperature=0.3)
            self.logger.info(f"[SubtaskDecompositionAgent] LLM response: {response[:200]}")

            # Parse response
            result = self._parse_response(response, fallback_instruction=instruction)

            return DecompositionOutput(
                subtasks=result["subtasks"],
                reasoning=result.get("reasoning", ""),
                static_difficulty=result.get("static_difficulty", "medium"),
                difficulty_factors=result.get("difficulty_factors", {}),
            )
        except Exception as e:
            self.logger.warning(f"[SubtaskDecompositionAgent] LLM call failed: {e}")
            return self._fallback_decomposition(instruction, goal_position)

    def _build_prompt(
        self,
        instruction: str,
        goal_position: List[float],
        start_position: List[float],
    ) -> str:
        """Build decomposition prompt with static difficulty assessment.

        Args:
            instruction: Navigation instruction
            goal_position: Goal position
            start_position: Start position

        Returns:
            Prompt string for LLM
        """
        elevation_diff = goal_position[1] - start_position[1]

        prompt = f"""You are decomposing a navigation instruction into subtasks for a robot.
You also assess the overall difficulty of the instruction.

## Input
- Instruction: {instruction}
- Start position: {start_position} (x, y, z - y is elevation)
- Goal position: {goal_position}
- Elevation difference: {elevation_diff:.2f} (positive means goal is higher)

## Task
1. Break the instruction into 2-6 subtasks. Each subtask should:
   - Have a clear description with implicit action direction
   - Have a verifiable completion_condition
   - Have relevant_objects (2-5 objects the vision model should FOCUS ON)

2. Assess static difficulty of this instruction based on:
   - Number of subtasks (1-2=easy, 3-4=medium, 5-6=hard) → contributes 0-2 points
   - Vertical navigation required (stairs/ramps) → contributes 2 points if yes
   - Complex turns (3+ turns across all subtasks) → contributes 2 points if yes
   - Ambiguous targets ("near the rug", "by the bench") → contributes 2 points if yes
   - Long instruction (>20 words) → contributes 1 point if yes
   - Score 7+ → "hard", 3-6 → "medium", 0-2 → "easy"

## Completion Condition Types + relevant_objects examples
1. y_change: Elevation change (stairs/ramps)
   relevant_objects: ["stairs", "steps", "railings", "staircase", "stairway"]

2. rotation: Turning action
   relevant_objects: ["door", "corridor", "passage", "wall", "corner"]

3. near_object: Approaching an object
   relevant_objects: include synonyms and variations of the target object

4. distance_to_goal: Final distance
   relevant_objects: final target objects mentioned in instruction

5. combined: Multiple conditions
   relevant_objects: all target objects for final position

## Output Format (JSON only)
```json
{{
  "subtasks": [
    {{
      "id": 1,
      "description": "Locate and descend stairs",
      "completion_condition": {{"type": "y_change", "direction": "down", "threshold": 1.0}},
      "relevant_objects": ["stairs", "steps", "railings"]
    }}
  ],
  "static_difficulty": "easy|medium|hard",
  "difficulty_factors": {{
    "num_subtasks": 3,
    "has_vertical_nav": true,
    "max_turns_per_subtask": 2,
    "has_ambiguous_target": false,
    "instruction_length": 15
  }},
  "reasoning": "Brief explanation of decomposition and difficulty"
}}
```

JSON output only:"""

        return prompt

    def _parse_response(
        self,
        response: str,
        fallback_instruction: str = None,
    ) -> Dict[str, Any]:
        """Parse LLM response into structured result.

        Args:
            response: LLM response text
            fallback_instruction: Instruction for fallback

        Returns:
            Dict with subtasks and reasoning
        """
        if not response:
            return self._fallback_dict(fallback_instruction)

        # Extract JSON from response
        json_str = response.strip()

        # Handle markdown code blocks
        if "```json" in json_str:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", json_str)
            if match:
                json_str = match.group(1).strip()
        elif "```" in json_str:
            match = re.search(r"```\s*([\s\S]*?)\s*```", json_str)
            if match:
                json_str = match.group(1).strip()

        # Find JSON object
        start = json_str.find("{")
        if start == -1:
            return self._fallback_dict(fallback_instruction)

        # Match braces
        brace_count = 0
        end = -1
        for i in range(start, len(json_str)):
            if json_str[i] == "{":
                brace_count += 1
            elif json_str[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if end == -1:
            return self._fallback_dict(fallback_instruction)

        try:
            data = json.loads(json_str[start:end+1])

            # Validate subtasks
            if "subtasks" not in data or not isinstance(data["subtasks"], list):
                return self._fallback_dict(fallback_instruction)

            # Ensure each subtask has required fields including relevant_objects
            validated_subtasks = []
            for i, subtask in enumerate(data["subtasks"]):
                validated = {
                    "id": subtask.get("id", i + 1),
                    "description": subtask.get("description", f"Subtask {i+1}"),
                    "completion_condition": subtask.get("completion_condition", {"type": "distance_to_goal", "threshold": 3.0}),
                    "relevant_objects": subtask.get("relevant_objects", []),
                }
                validated_subtasks.append(validated)

            return {
                "subtasks": validated_subtasks,
                "reasoning": data.get("reasoning", ""),
                "static_difficulty": data.get("static_difficulty", "medium"),
                "difficulty_factors": data.get("difficulty_factors", {}),
            }
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parse failed: {e}")
            return self._fallback_dict(fallback_instruction)

    def _fallback_decomposition(
        self,
        instruction: str,
        goal_position: List[float],
    ) -> DecompositionOutput:
        """Create fallback single-subtask decomposition.

        Args:
            instruction: Original instruction
            goal_position: Goal position

        Returns:
            DecompositionOutput with single subtask
        """
        return DecompositionOutput(
            subtasks=[
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {
                        "type": "distance_to_goal",
                        "threshold": 3.0,
                        "goal_position": goal_position,
                    },
                    "relevant_objects": [],
                }
            ],
            reasoning="Fallback: No LLM available or LLM failed",
            static_difficulty="medium",
            difficulty_factors={"fallback": True},
        )

    def _fallback_dict(self, instruction: str) -> Dict[str, Any]:
        """Create fallback dict for parse failure.

        Args:
            instruction: Original instruction

        Returns:
            Dict with single subtask
        """
        return {
            "subtasks": [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {"type": "distance_to_goal", "threshold": 3.0},
                    "relevant_objects": [],
                }
            ],
            "reasoning": "Fallback: JSON parsing failed",
            "static_difficulty": "medium",
            "difficulty_factors": {"fallback": True},
        }