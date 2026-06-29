"""Tests for Navigator - main agent orchestrator.

Following TDD: Write test first, watch it fail, then implement.
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from agents.pipeline.base_pipeline_agent import ObservationOutput, AnalysisOutput, PlanningOutput, ReviewOutput
from core.action import ActionType


class TestNavigatorInit:
    """Test suite for Navigator initialization."""

    def test_navigator_init(self):
        """Test Navigator initialization with default config."""
        # Import here to allow test to fail if module doesn't exist
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        assert nav._registry is not None
        assert nav._topology is not None
        assert nav._action_converter is not None
        assert nav._state_calculator is not None
        assert nav._step_count == 0
        assert nav._history == []

    def test_navigator_init_with_config(self):
        """Test Navigator initialization with custom config."""
        from agents.pipeline.navigator import Navigator

        config = {"max_steps": 50, "report_interval": 5}
        nav = Navigator(config)
        assert nav._max_steps == 50
        assert nav._report_interval == 5

    def test_navigator_register_subagents(self):
        """Test registering all SubAgents."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.register_subagents()

        # Check all agents are registered
        assert nav._registry.get("observation") is not None
        assert nav._registry.get("analysis") is not None
        assert nav._registry.get("planning") is not None
        assert nav._registry.get("review") is not None
        assert nav._registry.get("emergency") is not None

    def test_navigator_set_model_manager(self):
        """Test setting model_manager propagates to all SubAgents."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.register_subagents()

        mock_model_manager = Mock()
        nav.set_model_manager(mock_model_manager)

        # Check model_manager is set for all agents
        for agent in nav._registry.list_all().values():
            assert agent._model_manager == mock_model_manager


class TestNavigatorEpisode:
    """Test suite for episode initialization."""

    def test_initialize_episode(self):
        """Test initializing episode with instruction and position."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.initialize_episode("go downstairs", [0.0, 0.0, 0.0])

        assert nav._position == [0.0, 0.0, 0.0]
        assert len(nav._subtasks) > 0
        assert nav._current_subtask is not None
        assert nav._step_count == 0
        assert len(nav._history) == 1

    def test_initialize_episode_creates_default_subtask(self):
        """Test that episode initialization creates a default subtask."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.initialize_episode("turn left and go to the kitchen", [1.0, 2.0, 3.0])

        # Should create a single subtask with the full instruction
        assert len(nav._subtasks) == 1
        assert nav._subtasks[0]["description"] == "turn left and go to the kitchen"


class TestNavigatorCycle:
    """Test suite for navigation cycle."""

    def test_run_navigation_cycle(self):
        """Test single navigation cycle execution."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.register_subagents()
        nav.initialize_episode("test instruction", [0.0, 0.0, 0.0])

        # Mock model_manager with VLM and LLM responses
        mock_model_manager = Mock()

        # VLM response for ObservationAgent
        vlm_response = {
            "response": """
            {
                "task_relevant": true,
                "objects": ["stairs"],
                "target_direction": "forward",
                "target_distance": "medium",
                "path_blocked": false,
                "navigation_cues": ["stairs ahead"],
                "scene_description": "corridor with stairs"
            }
            """
        }
        mock_model_manager.generate_vision.return_value = vlm_response
        mock_model_manager.generate_vision_dual.return_value = vlm_response

        # LLM response for AnalysisAgent
        llm_analysis_response = """
        {
            "goal_summary": "navigate to stairs",
            "current_gap": "need to move forward",
            "recommended_action": "forward",
            "reasoning": "stairs visible ahead",
            "confidence": 0.8
        }
        """
        mock_model_manager.generate.return_value = llm_analysis_response

        nav.set_model_manager(mock_model_manager)

        # Mock env
        mock_env = Mock()
        mock_env.get_observations.return_value = {
            "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "depth": np.zeros((224, 224), dtype=np.float32)
        }

        # Run cycle
        actions = nav._run_navigation_cycle(mock_env)

        # Should return 5 actions
        assert len(actions) == 5
        assert all(isinstance(a, tuple) for a in actions)
        assert all(isinstance(a[0], ActionType) for a in actions)

    def test_run_navigation_cycle_with_path_blocked(self):
        """Test navigation cycle when path is blocked (emergency handling)."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.register_subagents()
        nav.initialize_episode("test instruction", [0.0, 0.0, 0.0])

        # Mock model_manager
        mock_model_manager = Mock()

        # VLM response with path blocked
        vlm_response = {
            "response": """
            {
                "task_relevant": true,
                "objects": ["wall"],
                "target_direction": "unknown",
                "target_distance": "unknown",
                "path_blocked": true,
                "navigation_cues": ["blocked"],
                "scene_description": "blocked path"
            }
            """
        }
        mock_model_manager.generate_vision.return_value = vlm_response
        mock_model_manager.generate_vision_dual.return_value = vlm_response

        # LLM responses
        mock_model_manager.generate.return_value = '{"severity": "medium"}'

        nav.set_model_manager(mock_model_manager)

        mock_env = Mock()
        mock_env.get_observations.return_value = {
            "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "depth": np.zeros((224, 224), dtype=np.float32)
        }

        # Run cycle - should handle emergency
        actions = nav._run_navigation_cycle(mock_env)

        # Should return some actions (emergency handling)
        assert len(actions) >= 1


class TestNavigatorEmergency:
    """Test suite for emergency handling."""

    def test_handle_emergency_obstacle(self):
        """Test handling obstacle emergency."""
        from agents.pipeline.navigator import Navigator
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        nav = Navigator()
        nav.register_subagents()

        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '{"actions": ["turn_left", "forward", "forward"]}'
        nav.set_model_manager(mock_model_manager)

        event = EmergencyEvent(
            type="obstacle",
            severity="medium",
            details={"collision": True}
        )

        mock_env = Mock()
        actions = nav._handle_emergency(event, mock_env)

        # Should return bypass actions
        assert len(actions) >= 1
        assert all(isinstance(a, tuple) for a in actions)

    def test_handle_emergency_evacuate(self):
        """Test handling evacuate emergency."""
        from agents.pipeline.navigator import Navigator
        from agents.pipeline.base_pipeline_agent import EmergencyEvent

        nav = Navigator()
        nav.register_subagents()

        mock_model_manager = Mock()
        mock_model_manager.generate.return_value = '{"actions": ["turn_left", "turn_left", "forward", "forward", "forward"]}'
        nav.set_model_manager(mock_model_manager)

        nav.initialize_episode("test", [0.0, 0.0, 0.0])

        event = EmergencyEvent(
            type="evacuate",
            severity="high",
            details={"command": {"type": "emergency"}}
        )

        mock_env = Mock()
        actions = nav._handle_emergency(event, mock_env)

        # Should return evacuation actions
        assert len(actions) >= 1


class TestNavigatorStateUpdate:
    """Test suite for state update."""

    def test_update_state(self):
        """Test state update after action execution."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.initialize_episode("test", [0.0, 0.0, 0.0])

        mock_env = Mock()
        mock_env.get_agent_position.return_value = [0.5, 0.0, 0.5]
        mock_env.get_agent_rotation.return_value = 0.0

        nav._update_state(mock_env)

        assert nav._position == [0.5, 0.0, 0.5]
        assert nav._rotation == 0.0
        assert len(nav._history) == 2

    def test_topology_update_on_state_change(self):
        """Test topology graph is updated when state changes."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.initialize_episode("test", [0.0, 0.0, 0.0])

        mock_env = Mock()
        mock_env.get_agent_position.return_value = [1.0, 0.0, 1.0]
        mock_env.get_agent_rotation.return_value = 90.0

        nav._update_state(mock_env)

        # Check topology has visited positions
        assert len(nav._topology.visited_positions) >= 1


class TestNavigatorCompletionCheck:
    """Test suite for completion checking."""

    def test_check_completion_not_complete(self):
        """Test completion check returns False when not complete."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.register_subagents()
        nav.initialize_episode("test", [0.0, 0.0, 0.0])

        # Mock subtask without completion condition
        nav._current_subtask = {"id": 1, "description": "test", "completion_condition": None}

        completed = nav._check_completion()
        assert completed == False


class TestNavigatorReportProgress:
    """Test suite for progress reporting."""

    def test_report_progress(self):
        """Test progress reporting."""
        from agents.pipeline.navigator import Navigator

        nav = Navigator()
        nav.initialize_episode("test", [0.0, 0.0, 0.0])
        nav._step_count = 10

        # Should not raise any error
        nav._report_progress()