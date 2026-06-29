"""ReviewAgent - Rule-based verification with LLM assistance.

This agent verifies task completion using:
1. Rule-based verification (deterministic, no LLM)
2. LLM secondary confirmation (for complex conditions)

Verification flow:
    Step 1: Rule verification (deterministic, no LLM)
        ↓ pass → return completed=True
        ↓ fail + semantic signals → Step 2

    Step 2: LLM secondary confirmation (auxiliary)
        ↓ complex conditions like "near carpet" → LLM confirms scene description
"""

from typing import Dict, Any, Optional

from agents.pipeline.base_pipeline_agent import SubAgent, ReviewOutput, ObservationOutput
from agents.base_agent import AgentRole


class ReviewAgent(SubAgent):
    """Review Agent - Rule core + LLM assistance for completion verification.

    Responsibilities:
    - Complete verification, prevent erroneous STOP
    - LLM role: Rule core, complex conditions LLM assisted

    Condition verification rules:
    | Condition Type      | Rule                                           |
    |---------------------|------------------------------------------------|
    | y_change            | dy >= threshold (up) or dy <= -threshold (down)|
    | near_object         | object visible + distance < 3m                 |
    | distance_to_goal    | distance < threshold                           |
    | rotation            | delta_deg >= threshold (left) or <= -threshold (right)|
    """

    name = "review_agent"

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def process(
        self,
        subtask,
        execution_result: Dict[str, Any],
        state_change: Dict[str, Any],
        observation: Optional[ObservationOutput] = None,
    ) -> ReviewOutput:
        """Complete verification.

        Args:
            subtask: Current subtask (with completion_condition)
            execution_result: Execution result from PlanningAgent
            state_change: State change dict {dy, dx, dz, rotation_change, distance_to_goal}
            observation: ObservationAgent output (optional, used for LLM assistance)

        Returns:
            ReviewOutput with completed, reason, progress, current_value, threshold
        """
        # Get completion condition
        if subtask is None or not hasattr(subtask, "completion_condition"):
            return ReviewOutput(
                completed=False,
                reason="No subtask or completion_condition",
                progress=0.0,
                current_value=0.0,
                threshold=0.0,
            )

        condition = subtask.completion_condition
        if condition is None:
            return ReviewOutput(
                completed=False,
                reason="No completion_condition specified",
                progress=0.0,
                current_value=0.0,
                threshold=0.0,
            )

        condition_type = condition.get("type")

        # Step 1: Rule verification
        rule_result = self._rule_verification(condition, state_change, observation)

        # Remove non-ReviewOutput fields before creating ReviewOutput
        clean_result = self._clean_result(rule_result)

        if clean_result["completed"]:
            return ReviewOutput(**clean_result)

        # Step 2: LLM secondary confirmation (for complex conditions)
        if self._needs_llm_verification(condition_type, rule_result):
            llm_result = self._llm_verification(subtask, observation, state_change)
            return ReviewOutput(**llm_result)

        return ReviewOutput(**clean_result)

    def _clean_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-ReviewOutput fields from result dict.

        Args:
            result: Raw result dict from verification

        Returns:
            Cleaned dict with only ReviewOutput fields
        """
        return {
            "completed": result.get("completed", False),
            "reason": result.get("reason", ""),
            "progress": result.get("progress", 0.0),
            "current_value": result.get("current_value", 0.0),
            "threshold": result.get("threshold", 0.0),
        }

    def _rule_verification(
        self,
        condition: Dict[str, Any],
        state_change: Dict[str, Any],
        observation: Optional[ObservationOutput],
    ) -> Dict[str, Any]:
        """Rule-based verification (no LLM).

        Args:
            condition: Completion condition dict
            state_change: State change dict
            observation: Observation output (for near_object verification)

        Returns:
            Dict with completed, reason, progress, current_value, threshold
        """
        condition_type = condition.get("type")

        if condition_type == "y_change":
            return self._verify_y_change(condition, state_change)
        elif condition_type == "near_object":
            return self._verify_near_object(condition, observation)
        elif condition_type == "distance_to_goal":
            return self._verify_distance_to_goal(condition, state_change)
        elif condition_type == "rotation":
            return self._verify_rotation(condition, state_change)

        # Unknown condition type - default result
        return self._default_result(condition)

    def _verify_y_change(
        self,
        condition: Dict[str, Any],
        state_change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify Y-axis change (vertical movement).

        Args:
            condition: {"type": "y_change", "direction": "down/up", "min_change": 1.5}
            state_change: {"dy": -2.0}

        Returns:
            Verification result dict
        """
        direction = condition.get("direction", "down")
        # Support both 'threshold' and 'min_change' field names
        threshold = condition.get("threshold", condition.get("min_change", 1.5))
        dy = state_change.get("dy", 0)

        if direction == "down":
            # Going down: dy should be negative (y decreases)
            completed = dy <= -threshold
            current_value = abs(dy) if dy < 0 else 0
        elif direction == "up":
            # Going up: dy should be positive (y increases)
            completed = dy >= threshold
            current_value = dy if dy > 0 else 0
        else:
            # No direction requirement, use absolute value
            completed = abs(dy) >= threshold
            current_value = abs(dy)

        progress = min(1.0, current_value / threshold)

        return {
            "completed": completed,
            "reason": f"dy={dy:.2f}m, need {direction} {threshold}m",
            "progress": progress,
            "current_value": current_value,
            "threshold": threshold,
        }

    def _verify_rotation(
        self,
        condition: Dict[str, Any],
        state_change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify rotation change.

        Args:
            condition: {"type": "rotation", "direction": "left/right", "min_degrees": 70}
            state_change: {"rotation_change": 80}

        Returns:
            Verification result dict
        """
        direction = condition.get("direction", "")
        # Support both 'threshold' and 'min_degrees' field names
        threshold = condition.get("threshold", condition.get("min_degrees", 70))
        rotation_change = state_change.get("rotation_change", 0)

        if direction == "left":
            # Turning left: rotation_change should be positive (angle increases)
            completed = rotation_change >= threshold
            current_value = rotation_change if rotation_change > 0 else 0
        elif direction == "right":
            # Turning right: rotation_change should be negative (angle decreases)
            completed = rotation_change <= -threshold
            current_value = abs(rotation_change) if rotation_change < 0 else 0
        else:
            # No direction requirement, use absolute value
            completed = abs(rotation_change) >= threshold
            current_value = abs(rotation_change)

        progress = min(1.0, current_value / threshold)

        return {
            "completed": completed,
            "reason": f"rotation={rotation_change:.0f}deg, need {direction} {threshold}deg",
            "progress": progress,
            "current_value": current_value,
            "threshold": threshold,
        }

    def _verify_near_object(
        self,
        condition: Dict[str, Any],
        observation: Optional[ObservationOutput],
    ) -> Dict[str, Any]:
        """Verify near_object condition.

        Args:
            condition: {"type": "near_object", "object": "stairs"}
            observation: ObservationOutput with objects (List[Dict]) and target_distance

        Returns:
            Verification result dict
        """
        target_object = condition.get("target", condition.get("object", ""))

        if observation is None:
            return self._default_result(condition, reason="No observation available")

        # Check object visibility - objects is List[Dict], extract names
        objects = observation.objects or []
        object_names = []
        object_details = None  # Store matching object details for distance check
        for obj in objects:
            if isinstance(obj, dict):
                obj_name = obj.get("name", "")
                object_names.append(obj_name)
                # Check if this is the target object
                if target_object.lower() in obj_name.lower():
                    object_details = obj  # Store details for this object
            elif isinstance(obj, str):
                object_names.append(obj)
                if target_object.lower() in obj.lower():
                    # Old format - no details
                    object_details = {"name": obj, "distance": observation.target_distance}

        object_visible = target_object in object_names or any(
            target_object.lower() in name.lower() for name in object_names
        )

        # Use object-specific distance if available, otherwise use observation.target_distance
        # Use subtask_relevant field (backward compatibility via property alias)
        task_relevant = observation.subtask_relevant
        target_distance = observation.target_distance
        if object_details and "distance" in object_details:
            target_distance = object_details.get("distance", observation.target_distance)

        # Completed if: task_relevant AND object visible AND distance is close
        completed = task_relevant and object_visible and target_distance == "close"

        # Calculate progress
        progress = 0.0
        if task_relevant:
            progress = 0.3
        if object_visible:
            progress += 0.4
        if target_distance in ("close", "medium"):
            progress += 0.3 if target_distance == "close" else 0.1

        # Build reason with object details
        reason_parts = [f"object '{target_object}': visible={object_visible}"]
        if object_details:
            reason_parts.append(f"direction={object_details.get('direction', 'unknown')}")
            reason_parts.append(f"features={object_details.get('features', '')}")
        reason_parts.append(f"distance={target_distance}")

        return {
            "completed": completed,
            "reason": ", ".join(reason_parts),
            "progress": progress,
            "current_value": 1.0 if object_visible else 0.0,
            "threshold": 1.0,
            "needs_confirmation": not completed and object_visible,  # May need LLM for ambiguous cases
        }

    def _verify_distance_to_goal(
        self,
        condition: Dict[str, Any],
        state_change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify distance to goal condition.

        Args:
            condition: {"type": "distance_to_goal", "max_distance": 3.0}
            state_change: {"distance_to_goal": 2.5}

        Returns:
            Verification result dict
        """
        threshold = condition.get("threshold", condition.get("max_distance", 3.0))
        current_distance = state_change.get("distance_to_goal", 10.0)

        completed = current_distance <= threshold
        progress = max(0.0, 1.0 - current_distance / threshold)

        return {
            "completed": completed,
            "reason": f"distance={current_distance:.2f}m, threshold={threshold}m",
            "progress": progress,
            "current_value": current_distance,
            "threshold": threshold,
        }

    def _default_result(
        self,
        condition: Dict[str, Any],
        reason: str = "Unknown condition type",
    ) -> Dict[str, Any]:
        """Return default result for unknown conditions.

        Args:
            condition: The condition dict
            reason: Custom reason string

        Returns:
            Default verification result dict
        """
        return {
            "completed": False,
            "reason": reason,
            "progress": 0.0,
            "current_value": 0.0,
            "threshold": condition.get("min_change", condition.get("min_degrees", condition.get("max_distance", 0))),
        }

    def _needs_llm_verification(
        self,
        condition_type: str,
        rule_result: Dict[str, Any],
    ) -> bool:
        """Determine if LLM secondary confirmation is needed.

        Args:
            condition_type: Type of completion condition
            rule_result: Result from rule verification

        Returns:
            True if LLM verification is needed
        """
        # Complex conditions like near_object may need LLM assistance
        if condition_type == "near_object" and rule_result.get("needs_confirmation"):
            return True
        return False

    def _llm_verification(
        self,
        subtask,
        observation: Optional[ObservationOutput],
        state_change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM secondary confirmation for complex conditions.

        Args:
            subtask: Current subtask
            observation: Observation output
            state_change: State change dict

        Returns:
            Verification result dict
        """
        if observation is None:
            return self._default_result(subtask.completion_condition, reason="No observation for LLM verification")

        prompt = self._build_verification_prompt(subtask, observation, state_change)

        try:
            response = self._call_llm(prompt, max_tokens=150, temperature=0.1)
            return self._parse_llm_response(response)
        except RuntimeError:
            # ModelManager not set - return rule result
            return self._default_result(
                subtask.completion_condition,
                reason="LLM unavailable, use rule result",
            )

    def _build_verification_prompt(
        self,
        subtask,
        observation: ObservationOutput,
        state_change: Dict[str, Any],
    ) -> str:
        """Build verification prompt for LLM.

        Args:
            subtask: Current subtask
            observation: Observation output
            state_change: State change dict

        Returns:
            Prompt string for LLM
        """
        condition = subtask.completion_condition
        target_object = condition.get("target", condition.get("object", ""))

        return f"""Verify completion condition for navigation task.

## Task
Subtask: {subtask.description}

## Completion Condition
Type: near_object
Target object: {target_object}

## Current Observation
Scene: {observation.scene_description}
Visible objects: {', '.join(observation.objects)}
Task relevant: {observation.task_relevant}
Target distance: {observation.target_distance}
Target direction: {observation.target_direction}

## Question
Is the target object '{target_object}' close enough (within 3 meters) for the task to be completed?

Answer with JSON:
{
    "completed": true/false,
    "reason": "brief explanation",
    "confidence": 0.0-1.0
}"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM verification response.

        Args:
            response: LLM response string

        Returns:
            Verification result dict
        """
        import json
        import re

        # Try to extract JSON from response
        try:
            # Find JSON block
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                completed = data.get("completed", False)
                reason = data.get("reason", "LLM verification")
                confidence = data.get("confidence", 0.5)

                return {
                    "completed": completed,
                    "reason": f"LLM: {reason}",
                    "progress": confidence,
                    "current_value": confidence,
                    "threshold": 1.0,
                }
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for keywords
        if "completed" in response.lower() or "yes" in response.lower():
            return {
                "completed": True,
                "reason": "LLM: completed (keyword match)",
                "progress": 0.8,
                "current_value": 0.8,
                "threshold": 1.0,
            }

        return {
            "completed": False,
            "reason": "LLM: not completed (default)",
            "progress": 0.3,
            "current_value": 0.3,
            "threshold": 1.0,
        }