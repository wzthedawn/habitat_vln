"""Unit tests for SubtaskDecompositionAgent."""

import pytest
from unittest.mock import Mock, patch
from agents.pipeline.subtask_decomposition_agent import SubtaskDecompositionAgent
from agents.pipeline.base_pipeline_agent import DecompositionOutput


class TestSubtaskDecompositionAgent:
    """Test SubtaskDecompositionAgent functionality."""

    def test_agent_creation(self):
        """Test agent can be created."""
        agent = SubtaskDecompositionAgent()
        assert agent.name == "subtask_decomposition_agent"

    def test_process_returns_decomposition_output(self):
        """Test process returns DecompositionOutput."""
        agent = SubtaskDecompositionAgent()
        # Mock model manager
        mock_mm = Mock()
        mock_mm.generate_sync.return_value = """
        {
            "subtasks": [
                {"id": 1, "description": "Find stairs and go down", "completion_condition": {"type": "y_change", "direction": "down", "threshold": 1.0}},
                {"id": 2, "description": "Turn right", "completion_condition": {"type": "rotation", "direction": "right", "threshold": 45}}
            ],
            "reasoning": "Instruction has 2 stages"
        }
        """
        agent.set_model_manager(mock_mm)

        result = agent.process(
            instruction="Walk down the stairs and turn right",
            goal_position=[10.0, -3.0, 2.0],
            start_position=[5.0, -1.0, 3.0],
        )

        assert isinstance(result, DecompositionOutput)
        assert len(result.subtasks) == 2
        assert result.subtasks[0]["id"] == 1
        assert result.subtasks[0]["completion_condition"]["type"] == "y_change"

    def test_fallback_on_llm_failure(self):
        """Test fallback when LLM fails."""
        agent = SubtaskDecompositionAgent()
        agent.set_model_manager(None)  # No model manager

        result = agent.process(
            instruction="Walk down the stairs",
            goal_position=[10.0, -3.0, 2.0],
            start_position=[5.0, -1.0, 3.0],
        )

        assert isinstance(result, DecompositionOutput)
        assert len(result.subtasks) == 1
        assert result.subtasks[0]["completion_condition"]["type"] == "distance_to_goal"

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        agent = SubtaskDecompositionAgent()

        response = """
        ```json
        {
            "subtasks": [
                {"id": 1, "description": "desc1", "completion_condition": {"type": "y_change"}},
                {"id": 2, "description": "desc2", "completion_condition": {"type": "rotation"}}
            ],
            "reasoning": "test"
        }
        ```
        """

        result = agent._parse_response(response)
        assert len(result["subtasks"]) == 2
        assert result["reasoning"] == "test"

    def test_parse_malformed_response_fallback(self):
        """Test fallback for malformed response."""
        agent = SubtaskDecompositionAgent()

        response = "This is not JSON at all"

        result = agent._parse_response(response, fallback_instruction="Walk forward")
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["description"] == "Walk forward"