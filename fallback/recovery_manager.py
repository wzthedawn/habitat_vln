"""Recovery manager for navigation recovery points."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import time

from core.context import NavContext
from core.action import Action


@dataclass
class RecoveryPoint:
    """A recovery point in navigation."""
    id: int
    step: int
    position: Tuple[float, float, float]
    rotation: float
    room_type: str
    subtask_idx: int
    action_history_snapshot: List[str]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """
    Recovery Manager for navigation recovery points.

    Manages:
    - Checkpoint creation
    - State rollback
    - Recovery strategies
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize recovery manager.

        Args:
            config: Manager configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("RecoveryManager")

        # Settings
        self.checkpoint_interval = self.config.get("checkpoint_interval", 10)
        self.max_checkpoints = self.config.get("max_checkpoints", 50)
        self.auto_checkpoint = self.config.get("auto_checkpoint", True)

        # Storage
        self._checkpoints: List[RecoveryPoint] = []
        self._checkpoint_counter = 0

        # Recovery tracking
        self._recovery_count = 0
        self._last_recovery_step = -1

    def create_checkpoint(
        self,
        context: NavContext,
        metadata: Dict[str, Any] = None,
    ) -> RecoveryPoint:
        """
        Create a recovery checkpoint.

        Args:
            context: Current navigation context
            metadata: Optional metadata

        Returns:
            Created recovery point
        """
        self._checkpoint_counter += 1

        checkpoint = RecoveryPoint(
            id=self._checkpoint_counter,
            step=context.step_count,
            position=context.position,
            rotation=context.rotation,
            room_type=context.room_type,
            subtask_idx=context.current_subtask_idx,
            action_history_snapshot=[
                a.action_type.name for a in context.action_history[-10:]
            ],
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._checkpoints.append(checkpoint)

        # Enforce max checkpoints
        while len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)

        self.logger.debug(f"Created checkpoint {checkpoint.id} at step {checkpoint.step}")

        return checkpoint

    def restore_checkpoint(
        self,
        checkpoint_id: int,
        context: NavContext,
    ) -> bool:
        """
        Restore state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore
            context: Navigation context to restore to

        Returns:
            True if restoration successful
        """
        checkpoint = self._find_checkpoint(checkpoint_id)

        if checkpoint is None:
            self.logger.warning(f"Checkpoint {checkpoint_id} not found")
            return False

        # Restore state
        context.position = checkpoint.position
        context.rotation = checkpoint.rotation
        context.room_type = checkpoint.room_type
        context.current_subtask_idx = checkpoint.subtask_idx

        # Note: Cannot restore action_history as it contains Action objects
        # Store snapshot for reference instead
        context.metadata["restored_from"] = checkpoint_id
        context.metadata["restored_actions"] = checkpoint.action_history_snapshot

        self._recovery_count += 1
        self._last_recovery_step = context.step_count

        self.logger.info(f"Restored from checkpoint {checkpoint_id}")

        return True

    def get_latest_checkpoint(self) -> Optional[RecoveryPoint]:
        """
        Get the most recent checkpoint.

        Returns:
            Latest recovery point or None
        """
        if self._checkpoints:
            return self._checkpoints[-1]
        return None

    def get_checkpoint_before_step(self, step: int) -> Optional[RecoveryPoint]:
        """
        Get checkpoint before a specific step.

        Args:
            step: Step number

        Returns:
            Recovery point before step or None
        """
        for checkpoint in reversed(self._checkpoints):
            if checkpoint.step < step:
                return checkpoint
        return None

    def auto_checkpoint_if_needed(self, context: NavContext) -> Optional[RecoveryPoint]:
        """
        Create automatic checkpoint if interval reached.

        Args:
            context: Navigation context

        Returns:
            Created checkpoint or None
        """
        if not self.auto_checkpoint:
            return None

        # Check if interval reached
        last_step = self._checkpoints[-1].step if self._checkpoints else 0

        if context.step_count - last_step >= self.checkpoint_interval:
            return self.create_checkpoint(context)

        return None

    def should_recover(self, context: NavContext) -> bool:
        """
        Determine if recovery should be attempted.

        Args:
            context: Navigation context

        Returns:
            True if recovery recommended
        """
        # Don't recover too frequently
        if context.step_count - self._last_recovery_step < 5:
            return False

        # Check for signs of being stuck
        if self._is_stuck(context):
            return True

        # Check for excessive backtracking
        if self._excessive_backtracking(context):
            return True

        return False

    def _is_stuck(self, context: NavContext) -> bool:
        """Check if navigation is stuck."""
        if len(context.trajectory) < 5:
            return False

        # Check recent positions
        recent = context.trajectory[-5:]
        total_movement = sum(
            ((recent[i+1][0] - recent[i][0])**2 + (recent[i+1][2] - recent[i][2])**2)**0.5
            for i in range(len(recent)-1)
        )

        # If very little movement, might be stuck
        return total_movement < 0.1

    def _excessive_backtracking(self, context: NavContext) -> bool:
        """Check for excessive backtracking."""
        if len(context.action_history) < 10:
            return False

        # Count direction changes
        recent = context.action_history[-10:]
        turns = sum(1 for a in recent if "TURN" in a.action_type.name)

        # If many turns, might be backtracking
        return turns > 5

    def _find_checkpoint(self, checkpoint_id: int) -> Optional[RecoveryPoint]:
        """Find checkpoint by ID."""
        for checkpoint in self._checkpoints:
            if checkpoint.id == checkpoint_id:
                return checkpoint
        return None

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        return {
            "total_checkpoints": len(self._checkpoints),
            "recovery_count": self._recovery_count,
            "last_recovery_step": self._last_recovery_step,
            "checkpoint_ids": [c.id for c in self._checkpoints],
        }

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        self._checkpoints.clear()

    def get_checkpoints_in_range(
        self,
        start_step: int,
        end_step: int,
    ) -> List[RecoveryPoint]:
        """
        Get checkpoints within a step range.

        Args:
            start_step: Start step
            end_step: End step

        Returns:
            List of checkpoints in range
        """
        return [
            c for c in self._checkpoints
            if start_step <= c.step <= end_step
        ]


class RecoveryStrategy:
    """Strategy for navigation recovery."""

    @staticmethod
    def simple_recovery(
        context: NavContext,
        checkpoint: RecoveryPoint,
    ) -> Action:
        """
        Simple recovery - try different action.

        Args:
            context: Navigation context
            checkpoint: Recovery checkpoint

        Returns:
            Recovery action
        """
        # Get last action and try alternative
        if context.action_history:
            last = context.action_history[-1].action_type.name

            if last == "MOVE_FORWARD":
                return Action.turn_left(
                    confidence=0.5,
                    reasoning="Recovery: trying turn after failed forward"
                )
            elif "TURN" in last:
                return Action.forward(
                    confidence=0.5,
                    reasoning="Recovery: trying forward after turn"
                )

        return Action.forward(confidence=0.5, reasoning="Recovery: default")

    @staticmethod
    def backtracking_recovery(
        context: NavContext,
        checkpoint: RecoveryPoint,
    ) -> Action:
        """
        Backtracking recovery - reverse last actions.

        Args:
            context: Navigation context
            checkpoint: Recovery checkpoint

        Returns:
            Backtracking action
        """
        if context.action_history:
            last = context.action_history[-1].action_type.name

            if last == "MOVE_FORWARD":
                # Turn around
                return Action.turn_left(
                    confidence=0.6,
                    reasoning="Recovery: turning to backtrack"
                )
            elif last == "TURN_LEFT":
                return Action.turn_right(
                    confidence=0.6,
                    reasoning="Recovery: reversing turn"
                )
            elif last == "TURN_RIGHT":
                return Action.turn_left(
                    confidence=0.6,
                    reasoning="Recovery: reversing turn"
                )

        return Action.stop(confidence=0.3, reasoning="Recovery: stopping")

    @staticmethod
    def exploration_recovery(
        context: NavContext,
        checkpoint: RecoveryPoint,
    ) -> Action:
        """
        Exploration recovery - try random direction.

        Args:
            context: Navigation context
            checkpoint: Recovery checkpoint

        Returns:
            Exploration action
        """
        import random

        actions = [
            Action.turn_left(confidence=0.4),
            Action.turn_right(confidence=0.4),
            Action.forward(confidence=0.4),
        ]

        return random.choice(actions)