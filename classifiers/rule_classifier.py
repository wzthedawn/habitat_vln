"""Rule-based task classifier."""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import re

from core.context import NavContext, TaskType


@dataclass
class ClassificationResult:
    """Result of classification."""
    task_type: TaskType
    confidence: float
    reasoning: str


class RuleClassifier:
    """
    Rule-based task classifier.

    Uses heuristics and pattern matching to quickly classify task types.
    """

    # Keywords for each task type
    TASK_KEYWORDS: Dict[TaskType, Dict[str, Any]] = {
        TaskType.TYPE_0: {
            "keywords": ["turn", "go straight", "move forward", "stop"],
            "patterns": [r"^(turn|go|move|stop)\s*(left|right|forward|straight)?$"],
            "max_length": 20,
            "single_step": True,
        },

        TaskType.TYPE_1: {
            "keywords": ["walk", "follow", "corridor", "hallway", "passage"],
            "patterns": [
                r"walk\s+(down|along|through)\s+(the\s+)?(hallway|corridor)",
                r"go\s+(straight|ahead)\s+(and\s+)?(turn|stop)",
                r"follow\s+(the\s+)?(path|corridor)",
            ],
            "max_rooms": 1,
        },

        TaskType.TYPE_2: {
            "keywords": ["find", "look for", "search", "locate", "where is"],
            "patterns": [
                r"find\s+(the\s+)?\w+",
                r"look\s+for\s+(the\s+)?\w+",
                r"where\s+(is|are)\s+(the\s+)?\w+",
                r"go\s+to\s+(the\s+)?\w+",
            ],
            "object_focused": True,
        },

        TaskType.TYPE_3: {
            "keywords": ["room", "bedroom", "kitchen", "bathroom", "living", "enter", "exit"],
            "patterns": [
                r"(enter|go\s+into)\s+(the\s+)?(bedroom|kitchen|bathroom|living\s+room)",
                r"from\s+\w+\s+to\s+\w+",
                r"through\s+(the\s+)?\w+\s+(room|door)",
            ],
            "multi_room": True,
        },

        TaskType.TYPE_4: {
            "keywords": ["if", "unless", "depending", "choose", "either", "or", "maybe"],
            "patterns": [
                r"if\s+you\s+(see|find|reach)",
                r"unless\s+",
                r"choose\s+(the\s+)?(left|right)",
                r"either\s+.*\s+or",
                r"when\s+you\s+see",
            ],
            "conditional": True,
            "ambiguous": True,
        },
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize classifier with optional config."""
        self.config = config or {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self.compiled_patterns: Dict[TaskType, List[re.Pattern]] = {}
        for task_type, config in self.TASK_KEYWORDS.items():
            patterns = config.get("patterns", [])
            self.compiled_patterns[task_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify(self, context: NavContext) -> ClassificationResult:
        """
        Classify task type based on rules.

        Args:
            context: Navigation context

        Returns:
            ClassificationResult with task type and confidence
        """
        instruction = context.instruction.lower().strip()
        scores: List[Tuple[TaskType, float, str]] = []

        # Check each task type
        for task_type, config in self.TASK_KEYWORDS.items():
            score, reasoning = self._score_task_type(instruction, task_type, config)
            scores.append((task_type, score, reasoning))

        # Sort by score and get best match
        scores.sort(key=lambda x: x[1], reverse=True)
        best_type, best_score, best_reasoning = scores[0]

        # Normalize score to confidence
        confidence = min(best_score / 10.0, 1.0)

        return ClassificationResult(
            task_type=best_type,
            confidence=confidence,
            reasoning=best_reasoning,
        )

    def _score_task_type(
        self, instruction: str, task_type: TaskType, config: Dict[str, Any]
    ) -> Tuple[float, str]:
        """Score instruction against a task type."""
        score = 0.0
        reasons = []

        # Check keyword matches
        keywords = config.get("keywords", [])
        keyword_matches = sum(1 for kw in keywords if kw in instruction)
        if keyword_matches > 0:
            score += keyword_matches * 2
            reasons.append(f"matched {keyword_matches} keywords")

        # Check pattern matches
        patterns = self.compiled_patterns.get(task_type, [])
        pattern_matches = sum(1 for p in patterns if p.search(instruction))
        if pattern_matches > 0:
            score += pattern_matches * 3
            reasons.append(f"matched {pattern_matches} patterns")

        # Check instruction length
        max_length = config.get("max_length", 999)
        if len(instruction.split()) <= max_length:
            score += 1

        # Check for specific attributes
        if config.get("single_step") and len(instruction.split()) <= 5:
            score += 2

        if config.get("object_focused") and any(
            word in instruction for word in ["the ", "a ", "an "]
        ):
            score += 1

        if config.get("multi_room"):
            room_count = sum(
                1 for room in ["room", "bedroom", "kitchen", "bathroom", "living"]
                if room in instruction
            )
            if room_count >= 2:
                score += 2

        if config.get("conditional"):
            condition_words = ["if", "unless", "when", "depending"]
            if any(word in instruction for word in condition_words):
                score += 3

        reasoning = "; ".join(reasons) if reasons else "no specific matches"
        return score, reasoning

    def get_task_complexity_score(self, context: NavContext) -> float:
        """
        Get a continuous complexity score (0-1).

        Args:
            context: Navigation context

        Returns:
            Complexity score between 0 and 1
        """
        result = self.classify(context)
        type_to_complexity = {
            TaskType.TYPE_0: 0.1,
            TaskType.TYPE_1: 0.3,
            TaskType.TYPE_2: 0.5,
            TaskType.TYPE_3: 0.7,
            TaskType.TYPE_4: 0.9,
        }
        base = type_to_complexity.get(result.task_type, 0.5)
        return base * result.confidence + (1 - result.confidence) * 0.5