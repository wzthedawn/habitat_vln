"""Context compressor for token optimization."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from core.context import NavContext


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    minimal_tokens: int = 50
    standard_tokens: int = 150
    detailed_tokens: int = 400
    full_tokens: int = 800


class ContextCompressor:
    """
    Context Compressor for token optimization.

    Compresses navigation context to different detail levels to minimize
    token usage while preserving essential information.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize context compressor.

        Args:
            config: Compression configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("ContextCompressor")

        # Compression settings
        self.compression_config = CompressionConfig(
            minimal_tokens=self.config.get("minimal_tokens", 50),
            standard_tokens=self.config.get("standard_tokens", 150),
            detailed_tokens=self.config.get("detailed_tokens", 400),
            full_tokens=self.config.get("full_tokens", 800),
        )

        # Compression templates
        self.templates = {
            "minimal": "Nav: {instruction[:30]} | Step {step} | {action}",
            "standard": "Task: {instruction[:50]}\nStep: {step} | Room: {room}\nLast: {last_action}",
            "detailed": "Instruction: {instruction}\nStep: {step}/{max_steps}\nRoom: {room}\nScene: {scene}\nHistory: {history}",
        }

    def compress(
        self, context: NavContext, level: str = "standard"
    ) -> str:
        """
        Compress context to specified level.

        Args:
            context: Navigation context
            level: Compression level (minimal, standard, detailed, full)

        Returns:
            Compressed context string
        """
        if level == "minimal":
            return self._minimal_compress(context)
        elif level == "standard":
            return self._standard_compress(context)
        elif level == "detailed":
            return self._detailed_compress(context)
        else:
            return self._full_context(context)

    def _minimal_compress(self, context: NavContext) -> str:
        """Minimal compression (~50 tokens)."""
        # Essential information only
        instruction = context.instruction[:30] + "..." if len(context.instruction) > 30 else context.instruction

        last_action = "none"
        if context.action_history:
            last_action = context.action_history[-1].action_type.name

        return f"Nav: {instruction} | Step {context.step_count} | {last_action}"

    def _standard_compress(self, context: NavContext) -> str:
        """Standard compression (~150 tokens)."""
        parts = []

        # Instruction (truncated)
        instruction = context.instruction[:50] + "..." if len(context.instruction) > 50 else context.instruction
        parts.append(f"Task: {instruction}")

        # Current state
        parts.append(f"Step: {context.step_count} | Room: {context.room_type}")

        # Last action
        if context.action_history:
            last = context.action_history[-1].action_type.name
            parts.append(f"Last: {last}")

        # Current subtask if any
        current_subtask = context.get_current_subtask()
        if current_subtask:
            parts.append(f"Current: {current_subtask.description[:40]}")

        return "\n".join(parts)

    def _detailed_compress(self, context: NavContext) -> str:
        """Detailed compression (~400 tokens)."""
        parts = []

        # Full instruction
        parts.append(f"Instruction: {context.instruction}")

        # Navigation state
        parts.append(f"Step: {context.step_count}")
        parts.append(f"Position: {self._format_position(context.position)}")
        parts.append(f"Room: {context.room_type}")

        # Scene description
        if context.visual_features.scene_description:
            parts.append(f"Scene: {context.visual_features.scene_description[:100]}")

        # Object detections (limited)
        if context.visual_features.object_detections:
            objects = [o.get("name", "") for o in context.visual_features.object_detections[:5]]
            parts.append(f"Objects: {', '.join(objects)}")

        # Action history (recent)
        if context.action_history:
            history = context.get_action_summary(5)
            parts.append(f"History:\n{history}")

        # Subtasks
        if context.subtasks:
            parts.append("Subtasks:")
            for i, subtask in enumerate(context.subtasks[:3]):
                status = subtask.status[:1].upper()  # First letter
                parts.append(f"  {i+1}. [{status}] {subtask.description[:40]}")

        return "\n".join(parts)

    def _full_context(self, context: NavContext) -> str:
        """Full context without compression."""
        parts = []

        # Header
        parts.append("=" * 50)
        parts.append("NAVIGATION CONTEXT")
        parts.append("=" * 50)

        # Instruction
        parts.append(f"\nInstruction: {context.instruction}")
        parts.append(f"Task Type: {context.task_type.value}")

        # State
        parts.append(f"\nState:")
        parts.append(f"  Step: {context.step_count}")
        parts.append(f"  Position: {context.position}")
        parts.append(f"  Rotation: {context.rotation}")
        parts.append(f"  Room: {context.room_type}")

        # Visual Features
        parts.append(f"\nVisual Features:")
        if context.visual_features.scene_description:
            parts.append(f"  Scene: {context.visual_features.scene_description}")
        if context.visual_features.object_detections:
            parts.append(f"  Objects: {len(context.visual_features.object_detections)} detected")
        if context.visual_features.room_classification:
            parts.append(f"  Room Class: {context.visual_features.room_classification}")

        # Trajectory
        if context.trajectory:
            parts.append(f"\nTrajectory: {len(context.trajectory)} points")
            parts.append(f"  Start: {context.trajectory[0]}")
            parts.append(f"  Current: {context.trajectory[-1]}")

        # Action History
        if context.action_history:
            parts.append(f"\nAction History ({len(context.action_history)} actions):")
            for i, action in enumerate(context.action_history[-10:]):
                parts.append(f"  {i+1}. {action.action_type.name} (conf: {action.confidence:.2f})")

        # Subtasks
        if context.subtasks:
            parts.append(f"\nSubtasks ({len(context.subtasks)} total):")
            for subtask in context.subtasks:
                parts.append(f"  [{subtask.status}] {subtask.description}")

        # Decision History
        if context.decision_history:
            parts.append(f"\nDecision History: {len(context.decision_history)} decisions")

        # Metadata
        if context.metadata:
            parts.append(f"\nMetadata:")
            for key, value in list(context.metadata.items())[:5]:
                parts.append(f"  {key}: {str(value)[:50]}")

        return "\n".join(parts)

    def _format_position(self, position: tuple) -> str:
        """Format position tuple for display."""
        if position:
            return f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
        return "(0.0, 0.0, 0.0)"

    def compress_history(
        self, action_history: List, max_items: int = 10
    ) -> str:
        """
        Compress action history.

        Args:
            action_history: List of actions
            max_items: Maximum items to include

        Returns:
            Compressed history string
        """
        if not action_history:
            return "No actions"

        # Group consecutive same actions
        grouped = []
        current_action = None
        count = 0

        for action in action_history[-max_items:]:
            if action.action_type == current_action:
                count += 1
            else:
                if current_action is not None:
                    grouped.append(f"{current_action.name}x{count}")
                current_action = action.action_type
                count = 1

        if current_action is not None:
            grouped.append(f"{current_action.name}x{count}")

        return " → ".join(grouped)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def get_compression_ratio(
        self, context: NavContext, level: str
    ) -> float:
        """
        Get compression ratio for a context.

        Args:
            context: Navigation context
            level: Compression level

        Returns:
            Compression ratio (compressed/original)
        """
        original = len(self._full_context(context))
        compressed = len(self.compress(context, level))

        return compressed / original if original > 0 else 1.0