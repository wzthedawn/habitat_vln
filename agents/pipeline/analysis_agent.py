"""AnalysisAgent - LLM core with multi-strategy reasoning.

This agent performs analysis of the current navigation state and provides
action recommendations using multiple reasoning strategies:
- CoT (Chain of Thought): Standard step-by-step reasoning
- Debate: Multiple perspective comparison for consensus
- Reflection: Failure analysis and strategy adjustment when stuck
"""

import json
import logging
import math
from typing import Dict, Any, List, Optional

from agents.pipeline.base_pipeline_agent import (
    ObservationOutput,
    AnalysisOutput,
    SubAgent,
)
from agents.base_agent import AgentRole


class AnalysisAgent(SubAgent):
    """Analysis Agent - LLM core with multi-strategy reasoning.

    Responsibilities:
    - Analyze current state vs goal gap
    - Select appropriate reasoning strategy
    - Output action recommendations

    Strategies:
    - CoT: Standard step-by-step reasoning for regular navigation
    - Debate: Multi-perspective comparison for complex/unknown scenarios
    - Reflection: Failure analysis and adjustment when stuck
    """

    name = "analysis_agent"

    STRATEGIES = ["cot", "debate", "reflection"]

    # Position tolerance for stuck detection (in meters)
    STUCK_TOLERANCE = 0.2

    # Minimum consecutive similar positions to be considered stuck
    STUCK_THRESHOLD = 3

    # Debate mode constants
    DEBATE_LIGHT = "light"       # 2 perspectives + rule arbitration, 2 LLM calls
    DEBATE_STANDARD = "standard" # 2 LLM perspectives + LLM arbitration, 3 LLM calls
    DEBATE_DEEP = "deep"         # 3 perspectives + LLM arb + adversarial verify, 4-5 LLM calls

    # Difficulty thresholds
    STATIC_HARD_THRESHOLD = 6    # static_difficulty score >= 6 → hard
    STATIC_MEDIUM_THRESHOLD = 3  # static_difficulty score >= 3 → medium
    DYNAMIC_HARD_THRESHOLD = 5   # dynamic trigger score >= 5 → hard
    DYNAMIC_MEDIUM_THRESHOLD = 2 # dynamic trigger score >= 2 → medium

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize AnalysisAgent.

        Args:
            config: Agent configuration dictionary
                - model_key: default LLM for CoT (default: "qwen3.5-9b-fast")
                - strong_model_key: LLM for debate/reflection (default: "qwen3.6-35b-strong")
                - debate_mode: "light" | "standard" | "deep" (default: "standard")
        """
        super().__init__(config)
        self.logger = logging.getLogger("AnalysisAgent")
        # Default model keys for multi-tier allocation
        if "model_key" not in self.config:
            self.config["model_key"] = "qwen3.5-9b-fast"
        if "strong_model_key" not in self.config:
            self.config["strong_model_key"] = "qwen3.6-35b-strong"
        # Debate history for persistent-hard detection
        self._debate_history: List[Dict] = []
        self._consecutive_debates_without_progress = 0

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def process(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        prev_result: Optional[dict] = None,
        static_difficulty: str = "medium",
        dynamic_difficulty: str = "easy",
        subtask_progress: Optional[dict] = None,
    ) -> AnalysisOutput:
        """Process observation and produce analysis output.

        Uses two-layer difficulty grading (static + dynamic) to select
        the appropriate reasoning strategy and model tier.

        Args:
            observation: Observation output from ObservationAgent
            subtask: Current subtask with description
            history: Navigation history with positions
            prev_result: Optional previous analysis result
            static_difficulty: Pre-computed static difficulty (easy/medium/hard)
            dynamic_difficulty: Runtime dynamic difficulty (easy/medium/hard)
            subtask_progress: Current subtask completion progress dict

        Returns:
            AnalysisOutput with goal analysis and action recommendation
        """
        # Two-layer difficulty: final = max(static, dynamic), only escalate
        final_difficulty = self._resolve_difficulty(static_difficulty, dynamic_difficulty)

        # Select strategy based on final difficulty
        strategy = self._select_strategy(final_difficulty, observation, history)

        self.logger.info(
            f"[AnalysisAgent] static={static_difficulty}, dynamic={dynamic_difficulty}, "
            f"final={final_difficulty} → strategy={strategy}"
        )

        # Execute corresponding strategy (LLM-based)
        if strategy == "cot":
            result = self._cot_analysis(observation, subtask, history, subtask_progress)
        elif strategy == "debate_light":
            result = self._debate_analysis(observation, subtask, history, mode=self.DEBATE_LIGHT)
        elif strategy == "debate_standard":
            result = self._debate_analysis(observation, subtask, history, mode=self.DEBATE_STANDARD)
        elif strategy == "debate_deep":
            result = self._debate_analysis(observation, subtask, history, mode=self.DEBATE_DEEP)
        elif strategy == "reflection":
            result = self._reflection_analysis(observation, subtask, history, prev_result)
        else:
            result = self._cot_analysis(observation, subtask, history, subtask_progress)

        # Track debate history
        if "debate" in strategy:
            self._debate_history.append({
                "strategy": strategy,
                "position": history[-1]["position"] if history else None,
                "step": len(history),
            })
            # Keep last 10 entries
            if len(self._debate_history) > 10:
                self._debate_history = self._debate_history[-10:]
        else:
            # Reset consecutive debate counter when using non-debate strategy
            self._consecutive_debates_without_progress = 0

        result["strategy_used"] = strategy

        # Always print key decision for visibility
        print(f"\n[AnalysisAgent] strategy={strategy} action={result.get('recommended_action')} "
              f"confidence={result.get('confidence', 0):.2f} "
              f"reasoning={result.get('reasoning', '')[:100]}")

        return AnalysisOutput(**result)

    def _resolve_difficulty(self, static: str, dynamic: str) -> str:
        """Resolve final difficulty: max(static, dynamic), only escalate.

        Args:
            static: Static difficulty from instruction decomposition
            dynamic: Dynamic difficulty from runtime triggers

        Returns:
            Final difficulty: "easy", "medium", or "hard"
        """
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        final_level = max(difficulty_order.get(static, 1),
                          difficulty_order.get(dynamic, 0))
        return ["easy", "medium", "hard"][final_level]

    def _select_strategy(self, difficulty: str, observation: ObservationOutput,
                         history: List[dict]) -> str:
        """Select reasoning strategy based on difficulty level.

        Strategy selection matrix:
          easy   → CoT (fast model, low cost)
          medium → CoT (strong model for quality)
          hard   → Debate depth depends on persistence:
                   - First hard trigger → Light debate (preventive)
                   - Second hard trigger → Standard debate
                   - Third+ hard trigger → Deep debate (with adversarial verify)

        Args:
            difficulty: Final difficulty level
            observation: Current observation
            history: Navigation history

        Returns:
            Strategy name: "cot", "debate_light", "debate_standard",
            "debate_deep", or "reflection"
        """
        if difficulty == "easy":
            return "cot"

        if difficulty == "medium":
            return "cot"

        if difficulty == "hard":
            # Determine debate depth based on persistence
            debate_count = len([d for d in self._debate_history
                               if d.get("strategy", "").startswith("debate")])

            if debate_count == 0:
                return "debate_light"
            elif debate_count == 1:
                return "debate_standard"
            else:
                return "debate_deep"

        return "cot"

    def _count_stuck_positions(self, history: List[dict]) -> int:
        """Count consecutive positions within tolerance.

        Args:
            history: History entries with position field

        Returns:
            Count of consecutive similar positions
        """
        if not history:
            return 0

        # Get positions from history
        positions = []
        for entry in history:
            if "position" in entry and isinstance(entry["position"], (list, tuple)):
                positions.append(tuple(entry["position"]))
            elif "state" in entry and "position" in entry["state"]:
                positions.append(tuple(entry["state"]["position"]))

        if len(positions) < 2:
            return 0

        # Count consecutive similar positions
        stuck_count = 0
        reference_pos = positions[0]

        for pos in positions[1:]:
            # Calculate distance from reference
            distance = math.sqrt(
                (pos[0] - reference_pos[0]) ** 2 +
                (pos[1] - reference_pos[1]) ** 2 +
                (pos[2] - reference_pos[2]) ** 2
            )

            if distance <= self.STUCK_TOLERANCE:
                stuck_count += 1
            else:
                # Reset count if position changed significantly
                stuck_count = 0
                reference_pos = pos

        return stuck_count

    def _cot_analysis(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        subtask_progress: Optional[dict] = None,
    ) -> dict:
        """Chain of Thought analysis - LLM core.

        Uses strong_model_key for quality on medium+ tasks.

        Args:
            observation: Current observation
            subtask: Current subtask
            history: Navigation history
            subtask_progress: Optional subtask completion progress

        Returns:
            Analysis result dictionary
        """
        prompt = self._build_cot_prompt(observation, subtask, history, subtask_progress)
        # Use strong model for better reasoning quality
        model = self.config.get("strong_model_key", "qwen3.6-35b-strong")
        response = self._call_llm(prompt, max_tokens=400, temperature=0.3,
                                  model_key=model)

        return self._parse_response(response)

    def _build_cot_prompt(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        subtask_progress: Optional[dict] = None,
    ) -> str:
        """Build CoT analysis prompt with subtask progress awareness.

        Args:
            observation: Current observation
            subtask: Current subtask
            history: Navigation history
            subtask_progress: Optional subtask completion progress

        Returns:
            Prompt string for LLM
        """
        # Extract history info
        history_summary = self._summarize_history(history)

        # Build subtask progress section
        progress_section = ""
        if subtask_progress and subtask_progress.get("available"):
            progress_section = f"""
## SUBTASK PROGRESS (CRITICAL)
{subtask_progress.get('hint', '')}
- Type: {subtask_progress.get('type', 'unknown')}
- Current progress: {subtask_progress.get('progress_pct', 0)}%
- Your job: keep working toward completing this subtask.
  If progress < 50%, you MUST continue the same action direction.
  If target not visible, explore to find it rather than switching goals.
"""

        # Format objects with details
        objects_text = self._format_objects(observation.objects)

        # Add fallback note if in fallback mode
        fallback_note = ""
        if hasattr(observation, 'fallback_mode') and observation.fallback_mode:
            fallback_note = f"""
[FALLBACK MODE ACTIVE]
- exploration_hint: {observation.exploration_hint}
"""

        prompt = f"""You are a navigation analysis expert. Analyze the current state and recommend the best action.

## Navigation Goal
Current subtask: {subtask.description if hasattr(subtask, 'description') else str(subtask)}
{progress_section}
## Current Observation
- Subtask relevant: {observation.subtask_relevant if hasattr(observation, 'subtask_relevant') else observation.task_relevant}
- Instruction relevant: {observation.instruction_relevant if hasattr(observation, 'instruction_relevant') else False}
- Stair position: {observation.stair_position if hasattr(observation, 'stair_position') else 'none'}
- Stair direction: {observation.stair_direction if hasattr(observation, 'stair_direction') else 'none'}
- Fallback mode: {observation.fallback_mode if hasattr(observation, 'fallback_mode') else False}
- Detected objects (with details): {objects_text}
- Exploration hint: {observation.exploration_hint if hasattr(observation, 'exploration_hint') else ''}
{fallback_note}
- Target direction: {observation.target_direction}
- Target distance: {observation.target_distance}
- Path blocked: {observation.path_blocked}
- Navigation cues: {', '.join(observation.navigation_cues) if observation.navigation_cues else 'none'}
- Scene description: {observation.scene_description}

## Recent History
{history_summary}

## Analysis Steps
1. Goal Summary: Identify the navigation goal in one sentence
2. Current Gap: What is missing or blocking progress toward the goal?
3. ALIGNMENT CHECK: Turn to face the target BEFORE going forward.
   - TARGET DIRECTION RULES:
     "forward_left" → turn_left ×1, then forward
     "forward_right" → turn_right ×1, then forward
     "left" → turn_left ×2-3, then forward
     "right" → turn_right ×2-3, then forward
     ONLY "forward" → straight forward
   - STAIR RULES:
     stair_position=top + stair_direction=descend → MUST walk onto stairs (go forward toward them)
     stair_position=bottom + stair_direction=ascend → MUST walk up stairs
     If stairs are visible but you keep not descending → you may need to turn more to face the stairway entrance
4. Reasoning: Explain the alignment and movement plan
5. Action: Recommend one action: forward, turn_left, or turn_right

## Output Format (JSON)
Output only valid JSON:
{{"goal_summary": "one sentence goal description", "current_gap": "what is missing or blocking", "recommended_action": "forward|turn_left|turn_right", "reasoning": "step-by-step reasoning explanation", "confidence": 0.0-1.0}}

## Rules
- recommended_action must be one of: forward, turn_left, turn_right
- confidence must be between 0.0 and 1.0
- Pay attention to object features (direction, distance) when deciding action
- Output ONLY the JSON, no additional text

## CRITICAL: Fallback Mode Decision Rule
If fallback_mode is true and no objects detected:
  - Target objects are NOT visible in current view
  - MUST follow exploration_hint direction if provided
  - Example: hint="turn_left to find stairs" -> recommended_action="turn_left"
  - DO NOT output "forward" when target is not visible and you need to explore
"""

        return prompt

    def _format_objects(self, objects: List) -> str:
        """Format objects list for prompt display.

        Args:
            objects: List of objects (Dict or str format)

        Returns:
            Formatted string for prompt
        """
        if not objects:
            return "none"

        formatted = []
        for obj in objects:
            if isinstance(obj, dict):
                # New format with features
                name = obj.get("name", "unknown")
                direction = obj.get("direction", "unknown")
                location = obj.get("location", "unknown")
                distance = obj.get("distance", "unknown")
                features = obj.get("features", "")
                formatted.append(f"{name} (dir:{direction}, loc:{location}, dist:{distance}, feat:{features})")
            elif isinstance(obj, str):
                # Old format - just name
                formatted.append(obj)

        return "; ".join(formatted)

    def _summarize_history(self, history: List[dict]) -> str:
        """Summarize recent history for prompt.

        Args:
            history: Navigation history entries

        Returns:
            Summary string
        """
        if not history:
            return "No history available"

        # Summarize last 5 entries
        recent = history[-5:] if len(history) > 5 else history

        lines = []
        for i, entry in enumerate(recent):
            pos = entry.get("position", "unknown")
            action = entry.get("action", "unknown")
            lines.append(f"Step {i+1}: position={pos}, action={action}")

        return "\n".join(lines)

    def _debate_analysis(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        mode: str = "standard",
    ) -> dict:
        """Multi-perspective debate with geometry grounding.

        Supports three modes:
        - light: 1 LLM perspective + geometry anchor + rule arbitration (2 LLM calls)
        - standard: 2 LLM perspectives + geometry anchor + LLM arbitration (3 LLM calls)
        - deep: 2 LLM perspectives + geometry anchor + LLM arb + adversarial verify (4-5 LLM calls)

        All LLM perspectives use the strong model (qwen3.6-35b-strong) for quality.
        The geometry anchor is zero-LLM (pure math), providing an unbiased baseline.

        Args:
            observation: Current observation
            subtask: Current subtask
            history: Navigation history
            mode: "light", "standard", or "deep"

        Returns:
            Analysis result dictionary
        """
        strong_model = self.config.get("strong_model_key", "qwen3.6-35b-strong")

        # Always include geometry anchor (zero LLM cost, provides unbiased baseline)
        geo_perspective = self._geometry_anchor_perspective(observation, history)
        perspectives = [("Geometry Anchor", geo_perspective)]

        # Light mode: 1 LLM perspective
        p_direct = self._direct_inference(observation, subtask, model_key=strong_model)
        perspectives.append(("Direct Inference", p_direct))

        if mode in (self.DEBATE_STANDARD, self.DEBATE_DEEP):
            # Add history-based perspective
            p_history = self._history_inference(observation, subtask, history,
                                                model_key=strong_model)
            perspectives.append(("History Inference", p_history))

        # Arbitration
        if mode == self.DEBATE_LIGHT:
            # Rule-based arbitration: weighted vote (fast, no LLM)
            result = self._rule_arbitrate(perspectives, observation)
        else:
            # LLM arbitration for standard and deep modes
            result = self._llm_arbitrate(perspectives, observation, subtask,
                                         model_key=strong_model)

        # Deep mode: adversarial verification
        if mode == self.DEBATE_DEEP:
            result = self._adversarial_verify(result, observation, subtask,
                                              model_key=strong_model)

        self.logger.info(
            f"[Debate] mode={mode}, perspectives={len(perspectives)}, "
            f"final_action={result.get('recommended_action')}, "
            f"confidence={result.get('confidence')}"
        )
        return result

    def _geometry_anchor_perspective(
        self,
        observation: ObservationOutput,
        history: List[dict],
    ) -> dict:
        """Zero-LLM geometric baseline perspective.

        Provides a pure mathematical recommendation based on:
        - Target direction from ObservationAgent
        - Path blocked status
        - Recent movement trend

        This serves as an unbiased anchor for the debate — LLM perspectives
        must reconcile with physical constraints.

        Returns:
            Dict with recommended_action, reasoning, confidence
        """
        target_dir = observation.target_direction
        path_blocked = observation.path_blocked

        # Determine action from pure geometry
        if path_blocked:
            # Path blocked: turn to explore
            if target_dir == "left":
                action = "turn_left"
                reason = "Path blocked, target direction is left"
            elif target_dir == "right":
                action = "turn_right"
                reason = "Path blocked, target direction is right"
            else:
                # Blocked but unknown target — turn right as default explore
                action = "turn_right"
                reason = "Path blocked, target direction unknown, default explore right"
            confidence = 0.6
        elif target_dir == "forward":
            action = "forward"
            reason = "Target ahead, path clear"
            confidence = 0.85
        elif target_dir == "left":
            action = "turn_left"
            reason = "Target is to the left"
            confidence = 0.75
        elif target_dir == "right":
            action = "turn_right"
            reason = "Target is to the right"
            confidence = 0.75
        elif target_dir == "backward":
            # Need to turn around
            action = "turn_left"
            reason = "Target behind, turning to search"
            confidence = 0.5
        else:
            # Unknown target direction: check recent movement
            if history and len(history) >= 3:
                # Check if recent forward movement was effective
                action = "forward"
                reason = "Target direction unknown, continue exploring forward"
                confidence = 0.4
            else:
                action = "forward"
                reason = "Insufficient data, default forward"
                confidence = 0.3

        return {
            "goal_summary": f"Geometric: target={target_dir}, blocked={path_blocked}",
            "current_gap": "Pure geometric analysis, no semantic understanding",
            "recommended_action": action,
            "reasoning": f"[GEOMETRY] {reason}",
            "confidence": confidence,
        }

    def _rule_arbitrate(
        self,
        perspectives: List[tuple],
        observation: ObservationOutput,
    ) -> dict:
        """Rule-based arbitration (no LLM).

        Weighted vote across perspectives. Used in Light debate mode
        for speed — no additional LLM call needed.

        Weights: Geometry=0.3 (baseline), LLM perspectives=0.7 (more informed)
        """
        action_scores = {"forward": 0.0, "turn_left": 0.0, "turn_right": 0.0}

        for i, (name, result) in enumerate(perspectives):
            action = result.get("recommended_action", "forward")
            confidence = result.get("confidence", 0.5)

            # Geometry anchor gets weight 0.3, LLM perspectives get 0.7
            weight = 0.3 if "Geometry" in name else 0.7

            if action in action_scores:
                action_scores[action] += confidence * weight

        # Pick highest scoring action
        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]
        total_score = sum(action_scores.values()) or 1.0

        # Build consensus from perspectives
        perspective_actions = [p[1].get("recommended_action") for p in perspectives]
        all_agree = len(set(perspective_actions)) == 1

        return {
            "goal_summary": f"Rule arbitration: {best_action}",
            "current_gap": f"Action scores: { {k: round(v,2) for k,v in action_scores.items()} }",
            "recommended_action": best_action,
            "reasoning": f"Rule-based vote: {best_action} (score={best_score:.2f}, "
                         f"agreement={'unanimous' if all_agree else 'split'})",
            "confidence": min(best_score / total_score, 0.9),
        }

    def _direct_inference(
        self,
        observation: ObservationOutput,
        subtask,
        model_key: str = None,
    ) -> dict:
        """Direct inference from current observation only (no history).

        Uses strong model for quality reasoning from raw observation.

        Args:
            observation: Current observation
            subtask: Current subtask
            model_key: Model to use (defaults to strong_model_key)

        Returns:
            Inference result dictionary
        """
        objects_text = self._format_objects(observation.objects)

        prompt = f"""Based ONLY on current observation, infer the best action.

## Subtask
{subtask.description if hasattr(subtask, 'description') else str(subtask)}

## Observation
- Objects (with details): {objects_text}
- Target direction: {observation.target_direction}
- Target distance: {observation.target_distance}
- Scene: {observation.scene_description}
- Path blocked: {observation.path_blocked}

## Output (JSON only)
{{"goal_summary": "goal description", "current_gap": "what is missing", "recommended_action": "forward|turn_left|turn_right", "reasoning": "direct inference reasoning", "confidence": 0.0-1.0}}"""

        mk = model_key or self.config.get("strong_model_key", "qwen3.6-35b-strong")
        response = self._call_llm(prompt, max_tokens=200, temperature=0.4,
                                  model_key=mk)
        return self._parse_response(response)

    def _history_inference(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        model_key: str = None,
    ) -> dict:
        """History-based inference considering past navigation patterns.

        Uses strong model to detect patterns (circling, progress, etc.).

        Args:
            observation: Current observation
            subtask: Current subtask
            history: Navigation history
            model_key: Model to use (defaults to strong_model_key)

        Returns:
            Inference result dictionary
        """
        history_summary = self._summarize_history(history)
        objects_text = self._format_objects(observation.objects)

        prompt = f"""Based on observation AND history pattern, infer the best action.

## Subtask
{subtask.description if hasattr(subtask, 'description') else str(subtask)}

## Observation
- Objects (with details): {objects_text}
- Target direction: {observation.target_direction}
- Scene: {observation.scene_description}

## History Pattern
{history_summary}

## Output (JSON Only)
{{"goal_summary": "goal description", "current_gap": "what is missing", "recommended_action": "forward|turn_left|turn_right", "reasoning": "history-based reasoning", "confidence": 0.0-1.0}}"""

        mk = model_key or self.config.get("strong_model_key", "qwen3.6-35b-strong")
        response = self._call_llm(prompt, max_tokens=200, temperature=0.4,
                                  model_key=mk)
        return self._parse_response(response)

    def _llm_arbitrate(
        self,
        perspectives: List[tuple],
        observation: ObservationOutput,
        subtask,
        model_key: str = None,
    ) -> dict:
        """LLM arbitration to synthesize multiple perspectives.

        The LLM judge compares perspectives (including the geometry anchor)
        and produces a reasoned consensus.

        Args:
            perspectives: List of (name, result) tuples
            observation: Current observation
            subtask: Current subtask
            model_key: Model to use (defaults to strong_model_key)

        Returns:
            Synthesized result dictionary
        """
        perspectives_text = ""
        for name, result in perspectives:
            perspectives_text += f"""
### {name}
- Action: {result.get('recommended_action', 'unknown')}
- Reasoning: {result.get('reasoning', 'no reasoning')}
- Confidence: {result.get('confidence', 0.5)}
"""

        objects_text = self._format_objects(observation.objects)

        prompt = f"""Synthesize multiple perspectives to reach a consensus action recommendation.

## Subtask
{subtask.description if hasattr(subtask, 'description') else str(subtask)}

## Current Observation
- Scene: {observation.scene_description}
- Target: {observation.target_direction} at {observation.target_distance}
- Path blocked: {observation.path_blocked}
- Objects (with details): {objects_text}

## Perspectives
{perspectives_text}

## Task
Compare the perspectives. The Geometry Anchor is a pure mathematical baseline —
it may be wrong when semantic context matters (e.g., the target is a specific object,
not just a direction). Synthesize the best recommendation considering all viewpoints.

## Output (JSON Only)
{{"goal_summary": "synthesized goal", "current_gap": "synthesized gap analysis", "recommended_action": "forward|turn_left|turn_right", "reasoning": "synthesized reasoning explaining why this perspective was chosen", "confidence": 0.0-1.0}}"""

        mk = model_key or self.config.get("strong_model_key", "qwen3.6-35b-strong")
        response = self._call_llm(prompt, max_tokens=300, temperature=0.3,
                                  model_key=mk)
        return self._parse_response(response)

    def _adversarial_verify(
        self,
        result: dict,
        observation: ObservationOutput,
        subtask,
        model_key: str = "qwen3.6-35b-strong",
    ) -> dict:
        """Adversarial verification: ask the LLM to find flaws in its own decision.

        Used only in Deep debate mode. If the LLM finds serious flaws,
        confidence is downgraded and reasoning is annotated.

        Args:
            result: The arbitrated result to verify
            observation: Current observation
            subtask: Current subtask
            model_key: Model to use for verification

        Returns:
            Updated result dict (confidence may be downgraded)
        """
        action = result.get("recommended_action", "forward")
        reasoning = result.get("reasoning", "")

        prompt = f"""You are a skeptical reviewer. Find potential flaws in this navigation decision.

## Decision Under Review
- Recommended action: {action}
- Reasoning: {reasoning}

## Current Observation
- Path blocked: {observation.path_blocked}
- Target direction: {observation.target_direction}
- Scene: {observation.scene_description}
- Objects visible: {self._format_objects(observation.objects)}

## Task
Play devil's advocate: what could go wrong with the decision to {action}?
Consider:
1. Could there be obstacles not visible in the observation?
2. Is there a better alternative direction?
3. Could following this action lead to circling or getting stuck?

Output JSON only:
{{"flaws_found": true/false, "severity": "none/minor/major", "critique": "brief critique", "alternative_action": "better action or same", "confidence_adjustment": 0.0 (downgrade amount if flaws found)}}"""

        try:
            response = self._call_llm(prompt, max_tokens=200, temperature=0.3,
                                      model_key=model_key)
            import json, re
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                flaws_found = data.get("flaws_found", False)
                severity = data.get("severity", "none")
                adjustment = float(data.get("confidence_adjustment", 0))
                critique = data.get("critique", "")

                if flaws_found and severity in ("major",):
                    # Downgrade confidence
                    new_confidence = max(0.3, result.get("confidence", 0.7) - adjustment)
                    result["confidence"] = new_confidence
                    result["reasoning"] = (result.get("reasoning", "") +
                                          f" [ADVERSARIAL: {critique}]")

                    # If alternative is different and severity is major, consider switching
                    alt = data.get("alternative_action", action)
                    if alt != action and severity == "major":
                        result["current_gap"] = (result.get("current_gap", "") +
                                                f" Adversarial suggests: {alt}")

                self.logger.info(
                    f"[Adversarial] flaws={flaws_found}, severity={severity}, "
                    f"adjustment={adjustment}"
                )
        except Exception as e:
            self.logger.warning(f"[Adversarial] verification failed: {e}")

        return result

    def _reflection_analysis(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        prev_result: Optional[dict],
    ) -> dict:
        """Reflection analysis - analyze stuck situation and adjust strategy.

        Args:
            observation: Current observation
            subtask: Current subtask
            history: Navigation history (stuck positions)
            prev_result: Previous analysis result

        Returns:
            Analysis result dictionary with adjusted strategy
        """
        # Analyze stuck reason
        stuck_reason = self._analyze_stuck_reason(observation, history, prev_result)

        prompt = self._build_reflection_prompt(observation, subtask, history, prev_result, stuck_reason)
        response = self._call_llm(prompt, max_tokens=400, temperature=0.4)

        return self._parse_response(response)

    def _analyze_stuck_reason(
        self,
        observation: ObservationOutput,
        history: List[dict],
        prev_result: Optional[dict],
    ) -> str:
        """Analyze why the agent is stuck.

        Args:
            observation: Current observation
            history: Navigation history
            prev_result: Previous analysis result

        Returns:
            Stuck reason description
        """
        reasons = []

        # Check path blocking
        if observation.path_blocked:
            reasons.append("path is blocked")

        # Check if target direction is unknown
        if observation.target_direction == "unknown":
            reasons.append("target direction unknown")

        # Check if previous action was ineffective
        if prev_result:
            prev_action = prev_result.get("recommended_action", "unknown")
            if prev_action == "forward" and observation.path_blocked:
                reasons.append(f"previous {prev_action} action was ineffective")

        if not reasons:
            reasons.append("repeated similar positions without progress")

        return ", ".join(reasons)

    def _build_reflection_prompt(
        self,
        observation: ObservationOutput,
        subtask,
        history: List[dict],
        prev_result: Optional[dict],
        stuck_reason: str,
    ) -> str:
        """Build reflection analysis prompt.

        Args:
            observation: Current observation
            subtask: Current subtask
            history: Navigation history
            prev_result: Previous analysis result
            stuck_reason: Analyzed stuck reason

        Returns:
            Prompt string for LLM
        """
        prev_action = prev_result.get("recommended_action", "none") if prev_result else "none"
        prev_reasoning = prev_result.get("reasoning", "none") if prev_result else "none"
        objects_text = self._format_objects(observation.objects)

        prompt = f"""Reflect on the stuck situation and propose a different strategy.

## Navigation Goal
{subtask.description if hasattr(subtask, 'description') else str(subtask)}

## Stuck Situation
- Analysis: Agent is stuck in same position
- Reason: {stuck_reason}

## Previous Attempt (Failed)
- Action: {prev_action}
- Reasoning: {prev_reasoning}

## Current Observation
- Objects (with details): {objects_text}
- Target direction: {observation.target_direction}
- Target distance: {observation.target_distance}
- Path blocked: {observation.path_blocked}
- Scene: {observation.scene_description}

## Reflection Task
1. Identify why previous strategy failed
2. Propose a different approach
3. Recommend a NEW action (different from previous)

## Output (JSON Only)
{{"goal_summary": "goal description", "current_gap": "stuck situation analysis", "recommended_action": "forward|turn_left|turn_right", "reasoning": "reflection reasoning explaining strategy change", "confidence": 0.0-1.0}}

## Rules
- recommended_action should be DIFFERENT from previous: {prev_action}
- If previous was forward, consider turn_left or turn_right
- If previous was turn, consider different turn direction or forward
- Output ONLY valid JSON"""

        return prompt

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response to structured dictionary.

        Multi-layer fallback for robust parsing.

        Args:
            response: LLM response text

        Returns:
            Parsed dictionary with required fields
        """
        import re

        # Default result
        default = {
            "goal_summary": "unknown goal",
            "current_gap": "unknown",
            "recommended_action": "forward",
            "reasoning": "unable to parse LLM response",
            "confidence": 0.5,
        }

        if not response:
            return default

        # Layer 1: Try direct JSON parsing
        try:
            result = json.loads(response.strip())
            return self._validate_and_fill(result)
        except (json.JSONDecodeError, ValueError):
            pass

        # Layer 2: Extract JSON from markdown code block
        if "```json" in response:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    return self._validate_and_fill(result)
                except (json.JSONDecodeError, ValueError):
                    pass
        elif "```" in response:
            match = re.search(r"```\s*([\s\S]*?)\s*```", response)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    return self._validate_and_fill(result)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Layer 3: Regex extract key fields
        action_match = re.search(r'"recommended_action":\s*"(\w+)"', response)
        if action_match:
            action = action_match.group(1)
            if action in ["forward", "turn_left", "turn_right"]:
                result = default.copy()
                result["recommended_action"] = action

                # Try to extract other fields
                goal_match = re.search(r'"goal_summary":\s*"([^"]+)"', response)
                if goal_match:
                    result["goal_summary"] = goal_match.group(1)

                gap_match = re.search(r'"current_gap":\s*"([^"]+)"', response)
                if gap_match:
                    result["current_gap"] = gap_match.group(1)

                conf_match = re.search(r'"confidence":\s*([\d.]+)', response)
                if conf_match:
                    try:
                        result["confidence"] = float(conf_match.group(1))
                    except ValueError:
                        pass

                return result

        # Layer 4: Keyword inference
        response_lower = response.lower()
        if "turn_left" in response_lower or "左转" in response:
            default["recommended_action"] = "turn_left"
        elif "turn_right" in response_lower or "右转" in response:
            default["recommended_action"] = "turn_right"
        elif "forward" in response_lower or "前进" in response or "straight" in response_lower:
            default["recommended_action"] = "forward"

        # Extract reasoning from response
        reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', response)
        if reasoning_match:
            default["reasoning"] = reasoning_match.group(1)
        else:
            default["reasoning"] = response[:100] if len(response) > 100 else response

        return default

    def _validate_and_fill(self, result: dict) -> dict:
        """Validate parsed result and fill missing fields.

        Args:
            result: Parsed result dictionary

        Returns:
            Validated and filled dictionary
        """
        # Ensure all required fields exist
        required_fields = {
            "goal_summary": "unknown goal",
            "current_gap": "unknown",
            "recommended_action": "forward",
            "reasoning": "no reasoning provided",
            "confidence": 0.5,
        }

        for field, default_value in required_fields.items():
            if field not in result:
                result[field] = default_value

        # Validate action
        valid_actions = ["forward", "turn_left", "turn_right"]
        if result["recommended_action"] not in valid_actions:
            result["recommended_action"] = "forward"

        # Validate confidence
        try:
            result["confidence"] = float(result["confidence"])
            if not (0.0 <= result["confidence"] <= 1.0):
                result["confidence"] = 0.5
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        return result