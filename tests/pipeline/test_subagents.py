"""Tests for ObservationAgent and AnalysisAgent.

Following TDD: Write test first, watch it fail, then implement.
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from agents.pipeline.base_pipeline_agent import (
    ObservationOutput,
    AnalysisOutput,
    SubAgent,
)
from agents.base_agent import AgentRole


class TestObservationAgent:
    """Test suite for ObservationAgent."""

    def test_observation_agent_process(self):
        """Test ObservationAgent processing with valid VLM response."""
        # Import here to allow test to fail if module doesn't exist
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()

        # Mock model_manager
        mock_model_manager = Mock()
        mock_model_manager.generate_vision_dual.return_value = {
            "response": """
            {
                "task_relevant": true,
                "objects": ["stairs", "railing"],
                "target_direction": "forward",
                "target_distance": "medium",
                "path_blocked": false,
                "navigation_cues": ["stairs entrance visible"],
                "scene_description": "stairwell entrance"
            }
            """
        }
        agent.set_model_manager(mock_model_manager)

        # Mock subtask
        mock_subtask = Mock()
        mock_subtask.description = "walk down the stairs"
        mock_subtask.completion_condition = {
            "type": "y_change",
            "direction": "down",
            "min_change": 1.5
        }

        # Mock images
        rgb_image = np.zeros((224, 224, 3), dtype=np.uint8)
        depth_image = np.zeros((224, 224), dtype=np.float32)

        # Process
        output = agent.process(
            subtask=mock_subtask,
            position=[0.0, 0.0, 0.0],
            rotation=0.0,
            rgb_image=rgb_image,
            depth_image=depth_image,
        )

        # Assertions
        assert isinstance(output, ObservationOutput)
        assert output.task_relevant == True
        assert output.target_direction == "forward"
        assert "stairs" in output.objects
        assert output.path_blocked == False

    def test_observation_agent_parse_fallback(self):
        """Test parsing incomplete JSON response with fallback defaults."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()

        # Simulate incomplete JSON response
        output_dict = agent._parse_response('{"target_direction": "left"}')

        # Should use defaults for missing fields
        assert output_dict["target_direction"] == "left"
        assert output_dict["task_relevant"] == False  # default
        assert output_dict["objects"] == []  # default
        assert output_dict["target_distance"] == "unknown"  # default
        assert output_dict["path_blocked"] == False  # default
        assert output_dict["navigation_cues"] == []  # default
        assert output_dict["scene_description"] == ""  # default

    def test_observation_agent_parse_malformed_json(self):
        """Test parsing completely malformed response."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()

        # Simulate completely invalid response
        output_dict = agent._parse_response("This is not JSON at all!")

        # Should return all defaults
        assert output_dict["task_relevant"] == False
        assert output_dict["target_direction"] == "unknown"
        assert output_dict["objects"] == []
        assert output_dict["path_blocked"] == False
        assert output_dict["navigation_cues"] == []
        assert output_dict["scene_description"] == "This is not JSON at all!"

    def test_observation_agent_build_prompt(self):
        """Test prompt building is task-oriented."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()

        # Mock subtask with completion condition
        mock_subtask = Mock()
        mock_subtask.description = "walk down the stairs"
        mock_subtask.completion_condition = {
            "type": "y_change",
            "direction": "down",
            "min_change": 1.5
        }

        prompt = agent._build_prompt(
            subtask=mock_subtask,
            position=[1.0, 2.0, 3.0],
            rotation=90.0
        )

        # Prompt should be task-oriented, not generic
        assert "walk down the stairs" in prompt.lower()
        assert "stairs" in prompt.lower() or "down" in prompt.lower()

        # Should NOT ask for generic scene description
        assert "describe the scene" not in prompt.lower() or "task" in prompt.lower()

    def test_observation_agent_no_model_manager(self):
        """Test that process raises error without model_manager."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()
        # Don't set model_manager

        mock_subtask = Mock()
        mock_subtask.description = "test"
        mock_subtask.completion_condition = {}

        rgb_image = np.zeros((224, 224, 3), dtype=np.uint8)
        depth_image = np.zeros((224, 224), dtype=np.float32)

        with pytest.raises(RuntimeError, match="ModelManager not set"):
            agent.process(
                subtask=mock_subtask,
                position=[0.0, 0.0, 0.0],
                rotation=0.0,
                rgb_image=rgb_image,
                depth_image=depth_image,
            )

    def test_observation_agent_extract_json_from_response(self):
        """Test extracting JSON from response with markdown formatting."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()

        # Response with markdown code block
        response = """
        Here's my analysis:
        ```json
        {
            "task_relevant": true,
            "objects": ["door"],
            "target_direction": "right",
            "target_distance": "close",
            "path_blocked": true,
            "navigation_cues": ["door on right"],
            "scene_description": "corridor with door"
        }
        ```
        """

        output_dict = agent._parse_response(response)

        assert output_dict["task_relevant"] == True
        assert output_dict["target_direction"] == "right"
        assert "door" in output_dict["objects"]
        assert output_dict["path_blocked"] == True

    def test_observation_agent_name_property(self):
        """Test agent name property."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()
        assert agent.name == "observation_agent"

    def test_observation_agent_vlm_call_parameters(self):
        """Test VLM is called with correct parameters."""
        from agents.pipeline.observation_agent import ObservationAgent

        agent = ObservationAgent()

        mock_model_manager = Mock()
        mock_model_manager.generate_vision_dual.return_value = {
            "response": '{"task_relevant": false, "objects": [], "target_direction": "unknown", "target_distance": "unknown", "path_blocked": false, "navigation_cues": [], "scene_description": ""}'
        }
        agent.set_model_manager(mock_model_manager)

        mock_subtask = Mock()
        mock_subtask.description = "find the kitchen"
        mock_subtask.completion_condition = {"type": "object_found", "target": "kitchen"}

        rgb_image = np.zeros((224, 224, 3), dtype=np.uint8)
        depth_image = np.zeros((224, 224), dtype=np.float32)

        agent.process(
            subtask=mock_subtask,
            position=[0.0, 0.0, 0.0],
            rotation=0.0,
            rgb_image=rgb_image,
            depth_image=depth_image,
        )

        # Verify VLM was called
        mock_model_manager.generate_vision_dual.assert_called_once()

        # Check call parameters
        call_kwargs = mock_model_manager.generate_vision_dual.call_args[1]
        assert "prompt" in call_kwargs
        assert "rgb_image" in call_kwargs
        assert "depth_image" in call_kwargs
        assert call_kwargs.get("max_new_tokens", 300) <= 300
        assert call_kwargs.get("temperature", 0.2) <= 0.3


class TestAnalysisAgentStrategySelection:
    """Test AnalysisAgent strategy selection logic."""

    def test_analysis_agent_select_strategy_cot_default(self):
        """Test strategy selection: default case returns cot."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        # Normal observation with known target distance
        observation = ObservationOutput(
            task_relevant=True,
            objects=["stairs", "railing"],
            target_direction="forward",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=["stairs entrance visible"],
            scene_description="stairwell entrance",
        )

        # Non-stuck history (different positions)
        history = [{"position": [float(i), 0.0, float(i)]} for i in range(5)]

        strategy = agent._select_strategy(observation, history)
        assert strategy == "cot"

    def test_analysis_agent_select_strategy_reflection_when_stuck(self):
        """Test strategy selection: stuck situation returns reflection."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        observation = ObservationOutput(
            task_relevant=True,
            objects=["wall", "door"],
            target_direction="unknown",
            target_distance="medium",
            path_blocked=True,
            navigation_cues=["blocked path"],
            scene_description="hallway with blocked path",
        )

        # Stuck history (same positions repeated >= 3 times)
        stuck_history = [{"position": [5.0, 0.0, 5.0]} for _ in range(6)]

        strategy = agent._select_strategy(observation, stuck_history)
        assert strategy == "reflection"

    def test_analysis_agent_select_strategy_debate_when_unknown(self):
        """Test strategy selection: unknown target distance returns debate."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        # Observation with unknown target distance
        observation = ObservationOutput(
            task_relevant=True,
            objects=["door"],
            target_direction="unknown",
            target_distance="unknown",
            path_blocked=False,
            navigation_cues=[],
            scene_description="room entrance",
        )

        # Non-stuck history
        history = [{"position": [float(i), 0.0, float(i)]} for i in range(5)]

        strategy = agent._select_strategy(observation, history)
        assert strategy == "debate"

    def test_analysis_agent_select_strategy_debate_when_not_relevant(self):
        """Test strategy selection: task not relevant returns debate."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        observation = ObservationOutput(
            task_relevant=False,
            objects=["chair", "table"],
            target_direction="forward",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=[],
            scene_description="living room",
        )

        history = [{"position": [float(i), 0.0, float(i)]} for i in range(5)]

        strategy = agent._select_strategy(observation, history)
        assert strategy == "debate"


class TestAnalysisAgentCoTAnalysis:
    """Test AnalysisAgent CoT analysis."""

    def test_analysis_agent_cot_analysis(self):
        """Test CoT analysis produces correct output."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''
        {
            "goal_summary": "go downstairs",
            "current_gap": "stairs not found yet",
            "recommended_action": "turn_right",
            "reasoning": "Based on scene description, stairs might be on the right side of the corridor",
            "confidence": 0.8
        }
        '''
        agent.set_model_manager(mock_model_manager)

        observation = ObservationOutput(
            task_relevant=True,
            objects=["door", "corridor"],
            target_direction="unknown",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=["door on the right"],
            scene_description="corridor with door on the right side",
        )

        mock_subtask = Mock()
        mock_subtask.description = "walk down stairs"

        result = agent._cot_analysis(observation, mock_subtask, [])

        assert result["goal_summary"] == "go downstairs"
        assert result["recommended_action"] == "turn_right"
        assert result["confidence"] == 0.8

    def test_analysis_agent_cot_with_history(self):
        """Test CoT analysis includes history information."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''
        {
            "goal_summary": "find kitchen",
            "current_gap": "kitchen not visible",
            "recommended_action": "forward",
            "reasoning": "Continue forward based on previous exploration",
            "confidence": 0.7
        }
        '''
        agent.set_model_manager(mock_model_manager)

        observation = ObservationOutput(
            task_relevant=True,
            objects=["hallway"],
            target_direction="forward",
            target_distance="far",
            path_blocked=False,
            navigation_cues=["open corridor ahead"],
            scene_description="long hallway",
        )

        mock_subtask = Mock()
        mock_subtask.description = "go to kitchen"

        history = [
            {"position": [0.0, 0.0, 0.0], "action": "forward"},
            {"position": [1.0, 0.0, 1.0], "action": "forward"},
        ]

        result = agent._cot_analysis(observation, mock_subtask, history)

        assert result["recommended_action"] == "forward"
        assert mock_model_manager.generate.called


class TestAnalysisAgentDebateAnalysis:
    """Test AnalysisAgent Debate analysis."""

    def test_analysis_agent_debate_analysis(self):
        """Test Debate analysis produces correct output."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        # Mock model manager for multiple perspectives and arbitration
        mock_model_manager = Mock()
        mock_model_manager.generate.side_effect = [
            # Direct inference perspective
            '{"goal_summary": "explore area", "current_gap": "unknown area", "recommended_action": "forward", "reasoning": "direct inference", "confidence": 0.6}',
            # History inference perspective
            '{"goal_summary": "continue path", "current_gap": "on track", "recommended_action": "turn_left", "reasoning": "history suggests left turn", "confidence": 0.7}',
            # Arbitration result
            '{"goal_summary": "explore left side", "current_gap": "area unexplored", "recommended_action": "turn_left", "reasoning": "Synthesizing direct and history: history suggests left has higher confidence", "confidence": 0.75}',
        ]
        agent.set_model_manager(mock_model_manager)

        observation = ObservationOutput(
            task_relevant=False,
            objects=["room"],
            target_direction="unknown",
            target_distance="unknown",
            path_blocked=False,
            navigation_cues=[],
            scene_description="unfamiliar room",
        )

        mock_subtask = Mock()
        mock_subtask.description = "explore the area"

        result = agent._debate_analysis(observation, mock_subtask, [])

        assert result["recommended_action"] in ["forward", "turn_left", "turn_right"]
        assert result["confidence"] >= 0.5


class TestAnalysisAgentReflectionAnalysis:
    """Test AnalysisAgent Reflection analysis."""

    def test_analysis_agent_reflection_analysis(self):
        """Test Reflection analysis when stuck."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''
        {
            "goal_summary": "find exit",
            "current_gap": "stuck in same position",
            "recommended_action": "turn_right",
            "reasoning": "Previous strategy failed, need to try different direction. Reflecting on stuck positions, turning right might reveal new path",
            "confidence": 0.6
        }
        '''
        agent.set_model_manager(mock_model_manager)

        observation = ObservationOutput(
            task_relevant=True,
            objects=["wall"],
            target_direction="forward",
            target_distance="unknown",
            path_blocked=True,
            navigation_cues=["blocked"],
            scene_description="corner with wall blocking",
        )

        mock_subtask = Mock()
        mock_subtask.description = "find exit"

        stuck_history = [{"position": [5.0, 0.0, 5.0]} for _ in range(4)]

        prev_result = {
            "recommended_action": "forward",
            "reasoning": "previous forward attempt",
            "confidence": 0.5,
        }

        result = agent._reflection_analysis(observation, mock_subtask, stuck_history, prev_result)

        assert result["recommended_action"] == "turn_right"
        assert "stuck" in result["reasoning"].lower() or "failed" in result["reasoning"].lower()


class TestAnalysisAgentProcess:
    """Test AnalysisAgent process method."""

    def test_analysis_agent_process_returns_analysis_output(self):
        """Test process returns AnalysisOutput."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''
        {
            "goal_summary": "navigate to target",
            "current_gap": "target not visible",
            "recommended_action": "forward",
            "reasoning": "clear path ahead",
            "confidence": 0.85
        }
        '''
        agent.set_model_manager(mock_model_manager)

        observation = ObservationOutput(
            task_relevant=True,
            objects=["door"],
            target_direction="forward",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=["door ahead"],
            scene_description="corridor with door",
        )

        mock_subtask = Mock()
        mock_subtask.description = "go through door"

        history = [{"position": [float(i), 0.0, float(i)]} for i in range(3)]

        output = agent.process(observation, mock_subtask, history)

        assert isinstance(output, AnalysisOutput)
        assert output.strategy_used == "cot"
        assert output.recommended_action == "forward"

    def test_analysis_agent_process_includes_strategy_used(self):
        """Test process output includes strategy_used field."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''
        {
            "goal_summary": "explore",
            "current_gap": "unknown",
            "recommended_action": "turn_left",
            "reasoning": "exploring left",
            "confidence": 0.7
        }
        '''
        agent.set_model_manager(mock_model_manager)

        observation = ObservationOutput(
            task_relevant=False,
            objects=[],
            target_direction="unknown",
            target_distance="unknown",
            path_blocked=False,
            navigation_cues=[],
            scene_description="unknown area",
        )

        mock_subtask = Mock()
        mock_subtask.description = "explore"

        output = agent.process(observation, mock_subtask, [])

        assert output.strategy_used == "debate"


class TestAnalysisAgentCountStuck:
    """Test stuck position counting."""

    def test_count_stuck_positions_no_stuck(self):
        """Test counting when positions are different."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        history = [
            {"position": [0.0, 0.0, 0.0]},
            {"position": [1.0, 0.0, 1.0]},
            {"position": [2.0, 0.0, 2.0]},
        ]

        count = agent._count_stuck_positions(history)
        assert count == 0

    def test_count_stuck_positions_with_stuck(self):
        """Test counting when positions are repeated."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        history = [
            {"position": [5.0, 0.0, 5.0]},
            {"position": [5.0, 0.0, 5.0]},
            {"position": [5.0, 0.0, 5.0]},
            {"position": [5.0, 0.0, 5.0]},
        ]

        count = agent._count_stuck_positions(history)
        assert count >= 3

    def test_count_stuck_positions_with_tolerance(self):
        """Test counting with position tolerance."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()

        # Positions within tolerance should count as stuck
        history = [
            {"position": [5.0, 0.0, 5.0]},
            {"position": [5.1, 0.0, 5.1]},  # Within 0.2m tolerance
            {"position": [4.9, 0.0, 4.9]},  # Within tolerance
            {"position": [5.05, 0.0, 5.05]},
        ]

        count = agent._count_stuck_positions(history)
        assert count >= 3


class TestAnalysisAgentProperties:
    """Test AnalysisAgent basic properties."""

    def test_analysis_agent_name(self):
        """Test agent name is correct."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        assert agent.name == "analysis_agent"

    def test_analysis_agent_role(self):
        """Test agent role is correct."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        assert agent.role == AgentRole.DECISION

    def test_analysis_agent_strategies(self):
        """Test agent has correct strategies."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        assert agent.STRATEGIES == ["cot", "debate", "reflection"]

    def test_analysis_agent_inherits_from_subagent(self):
        """Test AnalysisAgent inherits from SubAgent."""
        from agents.pipeline.analysis_agent import AnalysisAgent

        agent = AnalysisAgent()
        assert isinstance(agent, SubAgent)


class TestReviewAgentRuleVerification:
    """Test ReviewAgent rule-based verification."""

    def test_verify_y_change_down_complete(self):
        """Test y_change verification - going down complete."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "y_change", "direction": "down", "min_change": 1.5}
        state_change = {"dy": -2.0}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == True
        assert result["progress"] >= 1.0
        assert "dy=-2.00m" in result["reason"]

    def test_verify_y_change_down_not_complete(self):
        """Test y_change verification - going down not complete."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "y_change", "direction": "down", "min_change": 1.5}
        state_change = {"dy": -0.5}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == False
        assert result["progress"] < 1.0
        assert result["current_value"] == 0.5

    def test_verify_y_change_up_complete(self):
        """Test y_change verification - going up complete."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "y_change", "direction": "up", "min_change": 1.5}
        state_change = {"dy": 2.0}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == True
        assert result["current_value"] == 2.0

    def test_verify_rotation_left_complete(self):
        """Test rotation verification - turning left complete."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "rotation", "direction": "left", "min_degrees": 70}
        state_change = {"rotation_change": 80}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == True
        assert result["progress"] >= 1.0

    def test_verify_rotation_right_complete(self):
        """Test rotation verification - turning right complete."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "rotation", "direction": "right", "min_degrees": 70}
        state_change = {"rotation_change": -80}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == True

    def test_verify_rotation_not_complete(self):
        """Test rotation verification - not complete."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "rotation", "direction": "left", "min_degrees": 70}
        state_change = {"rotation_change": 30}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == False
        assert result["progress"] < 1.0

    def test_verify_near_object_visible_and_close(self):
        """Test near_object verification - object visible and close."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ObservationOutput

        agent = ReviewAgent()
        condition = {"type": "near_object", "object": "stairs"}

        observation = ObservationOutput(
            task_relevant=True,
            objects=["stairs", "railing"],
            target_direction="forward",
            target_distance="close",
            path_blocked=False,
            navigation_cues=["stairs visible"],
            scene_description="stairwell",
        )

        result = agent._rule_verification(condition, {}, observation)

        assert result["completed"] == True

    def test_verify_near_object_not_visible(self):
        """Test near_object verification - object not visible."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ObservationOutput

        agent = ReviewAgent()
        condition = {"type": "near_object", "object": "stairs"}

        observation = ObservationOutput(
            task_relevant=False,
            objects=["door", "wall"],
            target_direction="forward",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=["door visible"],
            scene_description="hallway",
        )

        result = agent._rule_verification(condition, {}, observation)

        assert result["completed"] == False

    def test_verify_distance_to_goal_complete(self):
        """Test distance_to_goal verification - close enough."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "distance_to_goal", "max_distance": 3.0}
        state_change = {"distance_to_goal": 2.5}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == True

    def test_verify_unknown_condition_type(self):
        """Test unknown condition type returns default."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        condition = {"type": "unknown_type"}
        state_change = {}

        result = agent._rule_verification(condition, state_change, None)

        assert result["completed"] == False
        assert result["progress"] == 0.0


class TestReviewAgentProcess:
    """Test ReviewAgent process method."""

    def test_process_y_change_complete(self):
        """Test complete process flow with y_change."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ReviewOutput

        agent = ReviewAgent()

        mock_subtask = Mock()
        mock_subtask.completion_condition = {
            "type": "y_change",
            "direction": "down",
            "min_change": 1.5,
        }

        output = agent.process(mock_subtask, {}, {"dy": -2.0})

        assert isinstance(output, ReviewOutput)
        assert output.completed == True
        assert output.reason != ""

    def test_process_y_change_not_complete(self):
        """Test process with incomplete y_change."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ReviewOutput

        agent = ReviewAgent()

        mock_subtask = Mock()
        mock_subtask.completion_condition = {
            "type": "y_change",
            "direction": "down",
            "min_change": 1.5,
        }

        output = agent.process(mock_subtask, {}, {"dy": -0.5})

        assert isinstance(output, ReviewOutput)
        assert output.completed == False
        assert output.progress < 1.0

    def test_process_rotation_complete(self):
        """Test process with rotation complete."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ReviewOutput

        agent = ReviewAgent()

        mock_subtask = Mock()
        mock_subtask.completion_condition = {
            "type": "rotation",
            "direction": "left",
            "min_degrees": 70,
        }

        output = agent.process(mock_subtask, {}, {"rotation_change": 80})

        assert isinstance(output, ReviewOutput)
        assert output.completed == True

    def test_process_with_observation(self):
        """Test process with observation for near_object."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ReviewOutput, ObservationOutput

        agent = ReviewAgent()

        mock_subtask = Mock()
        mock_subtask.completion_condition = {"type": "near_object", "object": "stairs"}

        observation = ObservationOutput(
            task_relevant=True,
            objects=["stairs"],
            target_direction="forward",
            target_distance="close",
            path_blocked=False,
            navigation_cues=["stairs close"],
            scene_description="stairwell",
        )

        output = agent.process(mock_subtask, {}, {}, observation)

        assert isinstance(output, ReviewOutput)

    def test_process_without_completion_condition(self):
        """Test process handles missing completion_condition."""
        from agents.pipeline.review_agent import ReviewAgent
        from agents.pipeline.base_pipeline_agent import ReviewOutput

        agent = ReviewAgent()

        mock_subtask = Mock()
        mock_subtask.completion_condition = None

        output = agent.process(mock_subtask, {}, {"dy": -1.0})

        assert isinstance(output, ReviewOutput)
        assert output.completed == False


class TestReviewAgentProperties:
    """Test ReviewAgent basic properties."""

    def test_review_agent_name(self):
        """Test agent name is correct."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        assert agent.name == "review_agent"

    def test_review_agent_inherits_from_subagent(self):
        """Test ReviewAgent inherits from SubAgent."""
        from agents.pipeline.review_agent import ReviewAgent

        agent = ReviewAgent()
        assert isinstance(agent, SubAgent)


class TestEmergencyAgentDetection:
    """Test suite for EmergencyAgent detection."""

    def test_detect_collision(self):
        """Test collision detection."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()
        context = {"collision_status": True}

        event = agent.process(context)
        assert event is not None
        assert event.type == "obstacle"

    def test_detect_path_blocked(self):
        """Test path blocked detection."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import ObservationOutput

        agent = EmergencyAgent()
        observation = ObservationOutput(
            task_relevant=True,
            objects=[],
            target_direction="forward",
            target_distance="medium",
            path_blocked=True,
            navigation_cues=[],
        )
        context = {"observation": observation}

        event = agent.process(context)
        assert event is not None
        assert event.type == "obstacle"

    def test_detect_user_evacuate(self):
        """Test user evacuate command detection."""
        from agents.pipeline.emergency_agent import EmergencyAgent

        agent = EmergencyAgent()
        context = {"user_command": {"type": "emergency"}}

        event = agent.process(context)
        assert event is not None
        assert event.type == "evacuate"
        assert event.severity == "high"

    def test_detect_no_emergency(self):
        """Test no emergency detected in normal situation."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import ObservationOutput

        agent = EmergencyAgent()
        observation = ObservationOutput(
            task_relevant=True,
            objects=["door"],
            target_direction="forward",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=["clear path"],
        )
        context = {
            "observation": observation,
            "collision_status": False,
        }

        event = agent.process(context)
        assert event is None

    def test_detect_priority_user_evacuate_over_collision(self):
        """Test user evacuate has priority over collision."""
        from agents.pipeline.emergency_agent import EmergencyAgent

        agent = EmergencyAgent()
        context = {
            "collision_status": True,
            "user_command": {"type": "emergency"},
        }

        event = agent.process(context)
        assert event.type == "evacuate"  # Evacuate takes priority


class TestEmergencyAgentHandle:
    """Test suite for EmergencyAgent handling."""

    def test_handle_obstacle_with_llm(self):
        """Test obstacle handling with LLM decision."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()

        # Mock model manager for LLM call
        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '{"actions": ["turn_right", "forward", "forward"]}'
        agent.set_model_manager(mock_model_manager)

        event = EmergencyEvent(type="obstacle", severity="medium", details={"collision": True})
        actions = agent.handle(event, {})

        assert len(actions) >= 2
        assert "turn_right" in actions or "turn_left" in actions
        assert mock_model_manager.generate.called

    def test_handle_obstacle_without_model_manager(self):
        """Test obstacle handling fallback without model manager."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()
        # Don't set model_manager - should use default fallback

        event = EmergencyEvent(type="obstacle", severity="medium", details={"collision": True})
        actions = agent.handle(event, {})

        # Should return default fallback actions
        assert len(actions) >= 1
        assert actions[0] in ["turn_left", "turn_right"]

    def test_handle_evacuate_default_fallback(self):
        """Test evacuate handling default fallback."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()
        # No topology info - should use default fallback

        event = EmergencyEvent(type="evacuate", severity="high", details={})
        actions = agent.handle(event, {})

        # Default fallback: turn around and move forward
        assert len(actions) >= 3
        assert "turn_left" in actions or "turn_right" in actions
        assert "forward" in actions

    def test_handle_unknown_event_type(self):
        """Test handling unknown event type returns empty list."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()
        event = EmergencyEvent(type="unknown", severity="low", details={})
        actions = agent.handle(event, {})

        assert actions == []


class TestEmergencyAgentSeverityAssessment:
    """Test suite for EmergencyAgent severity assessment."""

    def test_assess_severity_with_llm(self):
        """Test severity assessment with LLM."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()
        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '{"severity": "high"}'
        agent.set_model_manager(mock_model_manager)

        event = EmergencyEvent(type="obstacle", severity="medium", details={"collision": True})
        context = {"collision_status": True}

        severity = agent._assess_severity(event, context)
        assert severity in ["high", "medium", "low"]

    def test_assess_severity_fallback(self):
        """Test severity assessment fallback without model manager."""
        from agents.pipeline.emergency_agent import EmergencyAgent
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        agent = EmergencyAgent()
        # No model_manager - should use rule-based fallback

        event = EmergencyEvent(type="obstacle", severity="medium", details={"collision": True})
        context = {"collision_status": True}

        severity = agent._assess_severity(event, context)
        assert severity == "high"  # Collision should be high severity


class TestEmergencyAgentProperties:
    """Test EmergencyAgent basic properties."""

    def test_emergency_agent_name(self):
        """Test agent name is correct."""
        from agents.pipeline.emergency_agent import EmergencyAgent

        agent = EmergencyAgent()
        assert agent.name == "emergency_agent"

    def test_emergency_agent_inherits_from_subagent(self):
        """Test EmergencyAgent inherits from SubAgent."""
        from agents.pipeline.emergency_agent import EmergencyAgent

        agent = EmergencyAgent()
        assert isinstance(agent, SubAgent)


class TestPlanningAgentAlgorithmSelection:
    """Test PlanningAgent algorithm selection logic."""

    def test_select_algorithm_llm_default(self):
        """Test default uses LLM when no topology and no goal."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.tools.topology_graph import TopologyGraph

        agent = PlanningAgent()
        topology = TopologyGraph()  # No key nodes

        assert agent._select_algorithm(topology, None) == "llm"

    def test_select_algorithm_topology_priority(self):
        """Test topology algorithm has priority when key nodes exist."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.tools.topology_graph import TopologyGraph, NodeType

        agent = PlanningAgent()
        topology = TopologyGraph()
        topology.add_key_node([0, 0, 0], NodeType.TURN_POINT, 0)

        # Topology should be selected even when goal exists
        assert agent._select_algorithm(topology, [1, 0, 1]) == "topology"

    def test_select_algorithm_astar_when_goal(self):
        """Test A* algorithm when goal exists but no key nodes."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.tools.topology_graph import TopologyGraph

        agent = PlanningAgent()
        topology = TopologyGraph()  # No key nodes

        assert agent._select_algorithm(topology, [5, 0, 5]) == "astar"


class TestPlanningAgentProcess:
    """Test PlanningAgent process method."""

    def test_llm_planning(self):
        """Test LLM planning produces correct output."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.base_pipeline_agent import AnalysisOutput
        from agents.pipeline.tools.topology_graph import TopologyGraph

        agent = PlanningAgent()

        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''{
            "actions": ["forward", "forward", "forward", "forward", "forward"],
            "expected_result": "reach stairs"
        }'''
        agent.set_model_manager(mock_model_manager)

        analysis = AnalysisOutput(
            goal_summary="go downstairs",
            current_gap="stairs not found",
            recommended_action="forward",
            reasoning="clear path ahead",
            confidence=0.8,
            strategy_used="cot",
        )
        topology = TopologyGraph()

        output = agent.process(analysis, topology, [0, 0, 0])

        assert len(output.actions) == 5
        assert output.algorithm_used == "llm"
        assert output.expected_result == "reach stairs"

    def test_topology_planning(self):
        """Test topology-based planning."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.base_pipeline_agent import AnalysisOutput
        from agents.pipeline.tools.topology_graph import TopologyGraph, NodeType

        agent = PlanningAgent()

        # Mock model manager
        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '''{
            "actions": ["turn_left", "forward", "forward", "forward", "forward"],
            "expected_result": "reach turn point"
        }'''
        agent.set_model_manager(mock_model_manager)

        analysis = AnalysisOutput(
            goal_summary="find exit",
            current_gap="exit not visible",
            recommended_action="turn_left",
            reasoning="turn point detected nearby",
            confidence=0.75,
            strategy_used="cot",
        )
        topology = TopologyGraph()
        topology.add_key_node([2, 0, 2], NodeType.TURN_POINT, 90)

        output = agent.process(analysis, topology, [0, 0, 0])

        assert output.algorithm_used == "topology"
        assert output.path is not None or len(output.actions) > 0

    def test_astar_planning(self):
        """Test A* planning with goal position."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.base_pipeline_agent import AnalysisOutput
        from agents.pipeline.tools.topology_graph import TopologyGraph

        agent = PlanningAgent()

        analysis = AnalysisOutput(
            goal_summary="reach goal",
            current_gap="goal not reached",
            recommended_action="forward",
            reasoning="goal visible",
            confidence=0.9,
            strategy_used="cot",
        )
        topology = TopologyGraph()

        output = agent.process(analysis, topology, [0, 0, 0], goal=[5, 0, 5])

        assert output.algorithm_used == "astar"
        assert len(output.actions) == 5

    def test_process_without_model_manager_raises(self):
        """Test process raises error without model_manager for LLM planning."""
        from agents.pipeline.planning_agent import PlanningAgent
        from agents.pipeline.base_pipeline_agent import AnalysisOutput
        from agents.pipeline.tools.topology_graph import TopologyGraph

        agent = PlanningAgent()
        # Don't set model_manager

        analysis = AnalysisOutput(
            goal_summary="test",
            current_gap="test",
            recommended_action="forward",
            reasoning="test",
            confidence=0.5,
            strategy_used="cot",
        )
        topology = TopologyGraph()  # No key nodes, will use LLM

        with pytest.raises(RuntimeError, match="ModelManager not set"):
            agent.process(analysis, topology, [0, 0, 0])


class TestPlanningAgentProperties:
    """Test PlanningAgent basic properties."""

    def test_planning_agent_name(self):
        """Test agent name is correct."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()
        assert agent.name == "planning_agent"

    def test_planning_agent_algorithms(self):
        """Test agent has correct algorithms."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()
        assert agent.ALGORITHMS == ["llm", "topology", "astar"]

    def test_planning_agent_inherits_from_subagent(self):
        """Test PlanningAgent inherits from SubAgent."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()
        assert isinstance(agent, SubAgent)


class TestPlanningAgentActionGeneration:
    """Test action generation methods."""

    def test_generate_direction_actions_forward(self):
        """Test generating forward actions when goal is ahead."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        # Goal is directly ahead (positive z)
        actions = agent._generate_direction_actions(0, 5)

        assert actions.count("forward") >= 3
        assert "turn_left" not in actions
        assert "turn_right" not in actions

    def test_generate_direction_actions_turn_left(self):
        """Test generating turn_left when goal is to the left."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        # Goal is to the left (negative x)
        actions = agent._generate_direction_actions(-5, 0)

        assert "turn_left" in actions

    def test_generate_direction_actions_turn_right(self):
        """Test generating turn_right when goal is to the right."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        # Goal is to the right (positive x)
        actions = agent._generate_direction_actions(5, 0)

        assert "turn_right" in actions


class TestPlanningAgentResponseParsing:
    """Test LLM response parsing."""

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        response = '''{
            "actions": ["forward", "turn_left", "forward", "forward", "forward"],
            "expected_result": "reach the door"
        }'''

        result = agent._parse_response(response)

        assert result["actions"] == ["forward", "turn_left", "forward", "forward", "forward"]
        assert result["expected_result"] == "reach the door"

    def test_parse_response_markdown_json(self):
        """Test parsing JSON from markdown code block."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        response = '''
        Here's my plan:
        ```json
        {
            "actions": ["turn_right", "forward", "forward", "forward", "forward"],
            "expected_result": "navigate to corridor"
        }
        ```
        '''

        result = agent._parse_response(response)

        assert result["actions"] == ["turn_right", "forward", "forward", "forward", "forward"]

    def test_parse_response_incomplete_json(self):
        """Test parsing incomplete JSON with defaults."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        response = '{"actions": ["forward"]}'

        result = agent._parse_response(response)

        # Should fill in defaults
        assert result["expected_result"] != ""  # Has default

    def test_parse_response_malformed(self):
        """Test parsing malformed response with fallback."""
        from agents.pipeline.planning_agent import PlanningAgent

        agent = PlanningAgent()

        response = "This is not JSON at all!"

        result = agent._parse_response(response)

        # Should return valid defaults
        assert len(result["actions"]) == 5
        assert "expected_result" in result