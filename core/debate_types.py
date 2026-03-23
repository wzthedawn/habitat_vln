"""Debate types for VLN multi-agent system.

This module defines data structures for JSON-based opinion exchange
between agents during debate-based decision making.

Phase 1 of Debate Strategy Redesign.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class ConstraintType(Enum):
    """Type of action constraint."""
    HARD = "hard"   # Hard constraint: must be obeyed (e.g., obstacle avoidance)
    SOFT = "soft"   # Soft constraint: influences weight (e.g., exploration preference)


@dataclass
class ActionConstraint:
    """Constraint on an action.

    Hard constraints block actions entirely.
    Soft constraints modify the weight multiplier.
    """
    action: str
    blocked: bool = False
    weight_multiplier: float = 1.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "blocked": self.blocked,
            "weight_multiplier": self.weight_multiplier,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionConstraint":
        """Create from dictionary."""
        return cls(
            action=data.get("action", ""),
            blocked=data.get("blocked", False),
            weight_multiplier=data.get("weight_multiplier", 1.0),
            reason=data.get("reason", ""),
        )


@dataclass
class DebateOpinion:
    """Agent's opinion in a debate.

    Each agent submits a structured opinion containing:
    - Recommended action with confidence
    - Evidence supporting the recommendation
    - Reasoning explanation
    - Optional constraints on actions
    """
    agent: str                          # Agent name (perception, instruction, trajectory, decision)
    primary_action: str                 # Recommended action: move_forward, turn_left, turn_right, stop
    confidence: float                   # Confidence level 0.0-1.0
    evidence: Dict[str, Any]            # Supporting evidence data
    reasoning: str                      # Human-readable reasoning
    constraints: Dict[str, List[ActionConstraint]] = field(default_factory=dict)
    # constraints = {"hard": [...], "soft": [...]}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent": self.agent,
            "opinion": {
                "primary_action": self.primary_action,
                "confidence": self.confidence,
                "evidence": self.evidence,
                "reasoning": self.reasoning,
            },
            "constraints": {
                k: [c.to_dict() for c in v] for k, v in self.constraints.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebateOpinion":
        """Create from dictionary."""
        opinion_data = data.get("opinion", {})
        constraints_data = data.get("constraints", {})

        constraints = {}
        for constraint_type, constraint_list in constraints_data.items():
            constraints[constraint_type] = [
                ActionConstraint.from_dict(c) for c in constraint_list
            ]

        return cls(
            agent=data.get("agent", ""),
            primary_action=opinion_data.get("primary_action", "move_forward"),
            confidence=opinion_data.get("confidence", 0.5),
            evidence=opinion_data.get("evidence", {}),
            reasoning=opinion_data.get("reasoning", ""),
            constraints=constraints,
        )

    def add_hard_constraint(self, action: str, reason: str) -> None:
        """Add a hard constraint blocking an action."""
        if "hard" not in self.constraints:
            self.constraints["hard"] = []
        self.constraints["hard"].append(ActionConstraint(
            action=action,
            blocked=True,
            reason=reason,
        ))

    def add_soft_constraint(self, action: str, weight_multiplier: float, reason: str) -> None:
        """Add a soft constraint modifying action weight."""
        if "soft" not in self.constraints:
            self.constraints["soft"] = []
        self.constraints["soft"].append(ActionConstraint(
            action=action,
            weight_multiplier=weight_multiplier,
            reason=reason,
        ))


@dataclass
class DebateResult:
    """Result of a debate round.

    Contains the final decision and all supporting information.
    """
    best_action: str
    action_scores: Dict[str, float]
    weights_used: Dict[str, float]
    confidence: float
    reasoning: str
    opinions: List[DebateOpinion] = field(default_factory=list)
    hard_constraints_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_action": self.best_action,
            "action_scores": self.action_scores,
            "weights_used": self.weights_used,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "opinions": [o.to_dict() for o in self.opinions],
            "hard_constraints_applied": self.hard_constraints_applied,
        }


@dataclass
class EscapeVerification:
    """Result of escape sequence verification.

    Used to determine if the escape was successful and what to do next.
    """
    success: bool
    position_change: float
    still_stuck: bool
    visited_new_area: bool
    action: str  # "continue_subtask", "redebate", "continue_sequence"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "position_change": self.position_change,
            "still_stuck": self.still_stuck,
            "visited_new_area": self.visited_new_area,
            "action": self.action,
        }


# Action name standardization
ACTION_NAMES = {
    "forward": "move_forward",
    "move_forward": "move_forward",
    "left": "turn_left",
    "turn_left": "turn_left",
    "right": "turn_right",
    "turn_right": "turn_right",
    "stop": "stop",
}

# Valid action names for scoring
VALID_ACTIONS = ["move_forward", "turn_left", "turn_right", "stop"]


def normalize_action(action: str) -> str:
    """Normalize action name to standard format."""
    return ACTION_NAMES.get(action.lower(), action.lower())