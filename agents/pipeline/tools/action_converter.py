"""ActionConverter tool for mapping action names to Habitat actions.

This module provides utilities for converting between action names (strings)
and Habitat ActionType enums, with support for sequence padding and truncation.
"""

from typing import List, Tuple
from core.action import ActionType


class ActionConverter:
    """Converts action names to Habitat ActionType enums.

    Provides bidirectional conversion between string action names and
    ActionType enums, with support for:
    - Multiple action name aliases (e.g., "forward", "move_forward")
    - Unknown action fallback to MOVE_FORWARD
    - Sequence padding/truncation to ensure fixed-length output
    """

    ACTION_MAP = {
        "forward": ActionType.MOVE_FORWARD,
        "move_forward": ActionType.MOVE_FORWARD,
        "turn_left": ActionType.TURN_LEFT,
        "left": ActionType.TURN_LEFT,
        "turn_right": ActionType.TURN_RIGHT,
        "right": ActionType.TURN_RIGHT,
        "stop": ActionType.STOP,
        "look_up": ActionType.LOOK_UP,
        "look_down": ActionType.LOOK_DOWN,
    }

    # Reverse mapping: ActionType -> canonical name
    NAME_MAP = {
        ActionType.MOVE_FORWARD: "forward",
        ActionType.TURN_LEFT: "turn_left",
        ActionType.TURN_RIGHT: "turn_right",
        ActionType.STOP: "stop",
        ActionType.LOOK_UP: "look_up",
        ActionType.LOOK_DOWN: "look_down",
    }

    def convert(self, action_names: List[str]) -> List[Tuple[ActionType, int]]:
        """Convert action names to Habitat actions.

        Args:
            action_names: List of action name strings (e.g., ["forward", "turn_left"])

        Returns:
            List of (ActionType, repeat_count) tuples, where repeat_count is always 1
            for single action names. Unknown actions fall back to MOVE_FORWARD.
        """
        actions = []
        for name in action_names:
            # Normalize: lowercase and strip whitespace
            normalized = name.lower().strip()

            # Look up in map, fallback to MOVE_FORWARD for unknown actions
            action_type = self.ACTION_MAP.get(normalized, ActionType.MOVE_FORWARD)
            actions.append((action_type, 1))

        return actions

    def ensure_5_actions(
        self, actions: List[Tuple[ActionType, int]]
    ) -> List[Tuple[ActionType, int]]:
        """Ensure output has exactly 5 actions.

        If fewer than 5 actions: pad with (MOVE_FORWARD, 1) at the end.
        If more than 5 actions: truncate to first 5.

        Args:
            actions: List of (ActionType, repeat_count) tuples

        Returns:
            List of exactly 5 (ActionType, repeat_count) tuples
        """
        if len(actions) < 5:
            # Pad with forward actions
            padding_count = 5 - len(actions)
            padded = actions + [(ActionType.MOVE_FORWARD, 1)] * padding_count
            return padded
        elif len(actions) > 5:
            # Truncate to first 5
            return actions[:5]
        else:
            return actions

    def generate_stop(self) -> List[Tuple[ActionType, int]]:
        """Generate a STOP action sequence.

        Returns:
            List containing a single STOP action tuple
        """
        return [(ActionType.STOP, 1)]

    def actions_to_names(self, actions: List[Tuple[ActionType, int]]) -> List[str]:
        """Convert ActionType enums back to action names.

        Args:
            actions: List of (ActionType, repeat_count) tuples

        Returns:
            List of canonical action name strings
        """
        names = []
        for action_type, _ in actions:
            name = self.NAME_MAP.get(action_type, "forward")  # fallback to "forward"
            names.append(name)

        return names