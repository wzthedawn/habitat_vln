"""Tests for stuck recovery enhancement."""

import pytest
import math
import time
from agents.trajectory_agent import TrajectoryAgent
from core.context import NavContext


class TestStuckRecovery:
    """Test stuck recovery functionality."""

    def test_find_matching_stuck_region(self):
        """Test distance threshold matching."""
        agent = TrajectoryAgent()

        # 创建stuck_region
        agent._create_stuck_region((1.0, 0.0, 1.0), 0)

        # 在范围内应匹配 (radius=1.0, so positions within 1.0m)
        matched = agent._find_matching_stuck_region((1.2, 0.0, 1.2))
        # distance = sqrt(0.2^2 + 0.2^2) = 0.28 < 1.0, should match
        assert matched is not None

        # 超出范围不匹配
        not_matched = agent._find_matching_stuck_region((5.0, 0.0, 5.0))
        assert not_matched is None

    def test_find_matching_stuck_region_boundary(self):
        """Test matching at boundary distance."""
        agent = TrajectoryAgent()

        # 创建stuck_region at (0, 0, 0)
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)

        # Just inside boundary (distance < 1.0)
        matched_inside = agent._find_matching_stuck_region((0.7, 0.0, 0.7))
        distance = math.sqrt(0.7**2 + 0.7**2)  # ~0.99 < 1.0
        assert matched_inside is not None

        # Just outside boundary (distance > 1.0)
        matched_outside = agent._find_matching_stuck_region((0.8, 0.0, 0.8))
        distance = math.sqrt(0.8**2 + 0.8**2)  # ~1.13 > 1.0
        assert matched_outside is None

    def test_create_stuck_region_structure(self):
        """Test stuck_region has extended structure matching implementation."""
        agent = TrajectoryAgent()

        agent._create_stuck_region((0.0, 0.0, 0.0), 10)

        assert len(agent._stuck_regions) == 1
        region = agent._stuck_regions[0]

        # Core fields
        assert region["position"] == (0.0, 0.0, 0.0)
        assert region["radius"] == 1.0  # Actual implementation uses 1.0
        assert region["escape_attempts"] == 0
        assert region["successful_direction"] is None

        # Compatibility fields (matching mark_stuck_region schema)
        assert region["escape_success"] == False
        assert region["escape_actions"] == []
        assert "timestamp" in region  # Should have timestamp

        # Extended fields for recovery feature
        assert region["failed_directions"] == []
        assert region["last_attempt_step"] == 10
        assert region["created_at"] == 10

    def test_get_recovery_suggestion_no_history(self):
        """Test recovery suggestion without history."""
        agent = TrajectoryAgent()
        context = NavContext(instruction="test")
        context.position = (0.0, 0.0, 0.0)
        context.step_count = 0

        # 无历史，应创建新region并返回深度图方向
        suggestion = agent.get_stuck_recovery_suggestion(context, "left")

        assert suggestion["preferred_direction"] == "left"
        assert suggestion["avoid_directions"] == []
        assert suggestion["reason"] == "首次卡住，使用深度图分析"
        assert suggestion["confidence"] == 0.5
        assert suggestion["use_depth_analysis"] == True
        assert len(agent._stuck_regions) == 1

    def test_get_recovery_suggestion_with_success_history(self):
        """Test recovery suggestion with successful history."""
        agent = TrajectoryAgent()

        # 创建有成功历史的region
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)
        agent._stuck_regions[0]["successful_direction"] = "left"

        context = NavContext(instruction="test")
        context.position = (0.5, 0.0, 0.5)  # 在范围内 (distance ~0.7 < 1.0)
        context.step_count = 10

        suggestion = agent.get_stuck_recovery_suggestion(context, "right")

        # 应返回历史成功方向，忽略深度图建议
        assert suggestion["preferred_direction"] == "left"
        assert suggestion["reason"] == "历史成功方向"
        assert suggestion["confidence"] == 0.8
        assert suggestion["use_depth_analysis"] == False

    def test_get_recovery_suggestion_with_failed_history(self):
        """Test recovery suggestion with failed directions history."""
        agent = TrajectoryAgent()

        # 创建有失败历史的region (no success)
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)
        agent._stuck_regions[0]["failed_directions"] = ["right", "forward"]

        context = NavContext(instruction="test")
        context.position = (0.3, 0.0, 0.3)  # 在范围内
        context.step_count = 10

        suggestion = agent.get_stuck_recovery_suggestion(context, "left")

        # 应返回深度图方向并避开失败方向
        assert suggestion["preferred_direction"] == "left"
        assert suggestion["avoid_directions"] == ["right", "forward"]
        assert suggestion["reason"] == "避开历史失败方向"
        assert suggestion["confidence"] == 0.6
        assert suggestion["use_depth_analysis"] == True

    def test_mark_escape_result_success(self):
        """Test marking escape success."""
        agent = TrajectoryAgent()
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)

        agent.mark_escape_result((0.0, 0.0, 0.0), "left", success=True, step=10)

        region = agent._stuck_regions[0]
        assert region["successful_direction"] == "left"
        assert region["escape_success"] == True  # Compatibility field
        assert region["escape_attempts"] == 1
        assert region["last_attempt_step"] == 10

    def test_mark_escape_result_failure(self):
        """Test marking escape failure."""
        agent = TrajectoryAgent()
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)

        agent.mark_escape_result((0.0, 0.0, 0.0), "right", success=False, step=10)

        region = agent._stuck_regions[0]
        assert "right" in region["failed_directions"]
        assert region["escape_attempts"] == 1
        assert region["last_attempt_step"] == 10
        assert region["successful_direction"] is None

    def test_mark_escape_result_removes_from_failed_on_success(self):
        """Test that success removes direction from failed_directions."""
        agent = TrajectoryAgent()
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)

        # First mark as failed
        agent.mark_escape_result((0.0, 0.0, 0.0), "left", success=False, step=5)
        assert "left" in agent._stuck_regions[0]["failed_directions"]

        # Then mark as success - should remove from failed
        agent.mark_escape_result((0.0, 0.0, 0.0), "left", success=True, step=10)

        region = agent._stuck_regions[0]
        assert region["successful_direction"] == "left"
        assert "left" not in region["failed_directions"]
        assert region["escape_attempts"] == 2

    def test_mark_escape_result_no_region_found(self):
        """Test mark_escape_result when no matching region exists."""
        agent = TrajectoryAgent()
        # No region created

        # Should not crash, just log warning
        agent.mark_escape_result((0.0, 0.0, 0.0), "left", success=True, step=10)

        # No stuck_regions should be created
        assert len(agent._stuck_regions) == 0

    def test_multiple_stuck_regions(self):
        """Test handling multiple stuck regions."""
        agent = TrajectoryAgent()

        # Create two regions at different positions
        agent._create_stuck_region((0.0, 0.0, 0.0), 0)
        agent._create_stuck_region((10.0, 0.0, 10.0), 50)

        assert len(agent._stuck_regions) == 2

        # Each region should be independent
        region1 = agent._find_matching_stuck_region((0.5, 0.0, 0.5))
        assert region1 is not None
        assert region1["created_at"] == 0

        region2 = agent._find_matching_stuck_region((10.5, 0.0, 10.5))
        assert region2 is not None
        assert region2["created_at"] == 50

        # Position far from both
        no_match = agent._find_matching_stuck_region((5.0, 0.0, 5.0))
        assert no_match is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])