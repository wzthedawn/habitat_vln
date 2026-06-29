"""Unit tests for check_completion_condition function."""

import sys
import os
# Add project root to path for importing run_vln_experiment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import math
from run_vln_experiment import check_completion_condition
from core.context import NavContext, SubTask


class TestCheckCompletionCondition:
    """Tests for check_completion_condition function."""

    def _create_mock_context(self, position, rotation, subtask_start_pos=None, subtask_start_rot=None):
        """Helper to create mock NavContext."""
        context = NavContext(instruction="test instruction")
        context.position = position
        context.rotation = rotation

        # Create mock subtask with start_context
        subtask = SubTask(
            id=0,
            description="test subtask",
            completion_condition={"type": "distance", "min_meters": 5}
        )
        if subtask_start_pos:
            subtask.start_context = {
                "position": subtask_start_pos,
                "rotation": subtask_start_rot if subtask_start_rot is not None else rotation,
            }
        context.subtasks = [subtask]
        context.current_subtask_idx = 0

        return context

    # ==================== y_change condition tests ====================

    def test_y_change_completed(self):
        """Test y_change condition when threshold met."""
        context = self._create_mock_context(
            position=(0.0, 2.0, 0.0),  # dy = 2.0
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "y_change", "min_change": 1.5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 1.5
        assert "dy" in result["reason"]

    def test_y_change_not_completed(self):
        """Test y_change condition when threshold not met."""
        context = self._create_mock_context(
            position=(0.0, 0.5, 0.0),  # dy = 0.5
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "y_change", "min_change": 1.5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["progress"] < 1.0

    # ==================== distance condition tests ====================

    def test_distance_completed(self):
        """Test distance condition when threshold met."""
        context = self._create_mock_context(
            position=(5.0, 0.0, 0.0),  # horizontal_dist = 5.0
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 5.0

    def test_distance_not_completed(self):
        """Test distance condition when threshold not met."""
        context = self._create_mock_context(
            position=(2.0, 0.0, 0.0),  # horizontal_dist = 2.0
            rotation=0.0,
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["progress"] == 0.4  # 2.0/5.0

    # ==================== rotation condition tests ====================

    def test_rotation_completed(self):
        """Test rotation condition when threshold met."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(90),  # 90 degrees
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 70

    def test_rotation_boundary_handling(self):
        """Test rotation handles -180/180 boundary."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(-170),  # -170 degrees
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=math.radians(170)  # 170 degrees
        )
        # delta should be -170 - 170 = -340 -> +20 after boundary fix
        condition = {"type": "rotation", "min_degrees": 70}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["current_value"] == 20  # boundary handled

    # ==================== rotation direction tests ====================

    def test_rotation_left_completed(self):
        """Test rotation left direction when threshold met."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(90),  # 90 degrees (positive = left turn)
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70, "direction": "left"}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 70
        assert "left" in result["reason"]

    def test_rotation_left_not_completed(self):
        """Test rotation left direction when wrong direction."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(-90),  # -90 degrees (negative = right turn)
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70, "direction": "left"}
        result = check_completion_condition(context, condition)

        # Should not complete because we turned right instead of left
        assert result["completed"] == False
        assert result["current_value"] == 0  # Clamped to 0 for wrong direction

    def test_rotation_right_completed(self):
        """Test rotation right direction when threshold met."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(-90),  # -90 degrees (negative = right turn)
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70, "direction": "right"}
        result = check_completion_condition(context, condition)

        assert result["completed"] == True
        assert result["current_value"] >= 70
        assert "right" in result["reason"]

    def test_rotation_right_not_completed(self):
        """Test rotation right direction when wrong direction."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(90),  # 90 degrees (positive = left turn)
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70, "direction": "right"}
        result = check_completion_condition(context, condition)

        # Should not complete because we turned left instead of right
        assert result["completed"] == False
        assert result["current_value"] == 0  # Clamped to 0 for wrong direction

    def test_rotation_partial_progress(self):
        """Test rotation progress calculation with partial rotation."""
        context = self._create_mock_context(
            position=(0.0, 0.0, 0.0),
            rotation=math.radians(35),  # 35 degrees
            subtask_start_pos=(0.0, 0.0, 0.0),
            subtask_start_rot=0.0
        )
        condition = {"type": "rotation", "min_degrees": 70, "direction": "left"}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert result["progress"] == pytest.approx(35/70, rel=0.01)
        assert result["current_value"] == 35

    # ==================== edge cases tests ====================

    def test_no_subtask(self):
        """Test when no current subtask."""
        context = NavContext(instruction="test instruction")
        context.position = (1.0, 0.0, 0.0)
        context.subtasks = []

        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        assert result["completed"] == False
        assert "无子任务" in result["reason"]

    def test_no_start_context(self):
        """Test when subtask has no start_context."""
        context = self._create_mock_context(
            position=(5.0, 0.0, 0.0),
            rotation=0.0,
            subtask_start_pos=None  # No start_context
        )
        condition = {"type": "distance", "min_meters": 5}
        result = check_completion_condition(context, condition)

        # Should use current position as start -> delta = 0
        assert result["completed"] == False
        assert result["current_value"] == 0

    def test_no_condition(self):
        """Test when condition is empty."""
        context = self._create_mock_context(
            position=(1.0, 0.0, 0.0),
            rotation=0.0
        )
        result = check_completion_condition(context, None)

        assert result["completed"] == False
        assert "无条件" in result["reason"]