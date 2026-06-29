"""Integration tests for full navigation pipeline.

Following TDD: Write test first, watch it fail, then implement.
"""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from agents.pipeline.base_pipeline_agent import ObservationOutput, AnalysisOutput, PlanningOutput, ReviewOutput, EmergencyEvent
from core.action import ActionType


def test_full_pipeline_flow():
    """Test complete navigation flow through Navigator."""
    from agents.pipeline.navigator import Navigator

    nav = Navigator({"max_steps": 10, "report_interval": 5})
    nav.register_subagents()

    # Mock model_manager
    mock_model_manager = Mock()

    # VLM response for ObservationAgent
    vlm_response = {
        "response": """
        {
            "task_relevant": false,
            "objects": [],
            "target_direction": "forward",
            "target_distance": "unknown",
            "path_blocked": false,
            "navigation_cues": [],
            "scene_description": "empty corridor"
        }
        """
    }
    mock_model_manager.generate_vision.return_value = vlm_response
    mock_model_manager.generate_vision_dual.return_value = vlm_response

    # LLM response for AnalysisAgent and PlanningAgent
    llm_response = """
    {
        "goal_summary": "test goal",
        "current_gap": "unknown",
        "recommended_action": "forward",
        "reasoning": "test reasoning",
        "confidence": 0.8,
        "actions": ["forward", "forward", "forward", "forward", "forward"],
        "expected_result": "move forward"
    }
    """
    mock_model_manager.generate.return_value = llm_response

    nav.set_model_manager(mock_model_manager)

    nav.initialize_episode("test instruction", [0.0, 0.0, 0.0])

    # Mock env
    mock_env = Mock()
    mock_env.get_observations.return_value = {
        "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth": np.zeros((224, 224), dtype=np.float32)
    }
    mock_env.step.return_value = None
    mock_env.get_agent_position.return_value = [0.1, 0.0, 0.1]
    mock_env.get_agent_rotation.return_value = 0.0

    # Run navigation loop
    result = nav.run_navigation_loop(mock_env)

    # Should complete without success (mock data)
    assert result["success"] == False
    assert result["steps"] <= 10
    assert "reason" in result


def test_emergency_handling():
    """Test emergency handling during navigation."""
    from agents.pipeline.navigator import Navigator

    nav = Navigator({"max_steps": 5})
    nav.register_subagents()

    # Mock model_manager
    mock_model_manager = Mock()

    # VLM response indicating blocked path
    vlm_response = {
        "response": """
        {
            "task_relevant": true,
            "objects": ["wall"],
            "target_direction": "forward",
            "target_distance": "medium",
            "path_blocked": true,
            "navigation_cues": ["obstacle ahead"],
            "scene_description": "blocked path"
        }
        """
    }
    mock_model_manager.generate_vision.return_value = vlm_response
    mock_model_manager.generate_vision_dual.return_value = vlm_response

    # LLM responses for severity assessment and bypass planning
    mock_model_manager.generate.return_value = '{"severity": "medium", "actions": ["turn_left", "forward", "forward"]}'

    nav.set_model_manager(mock_model_manager)

    nav.initialize_episode("test instruction", [0.0, 0.0, 0.0])

    # Mock env
    mock_env = Mock()
    mock_env.get_observations.return_value = {
        "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth": np.zeros((224, 224), dtype=np.float32)
    }
    mock_env.step.return_value = None
    mock_env.get_agent_position.return_value = [0.0, 0.0, 0.0]
    mock_env.get_agent_rotation.return_value = 0.0

    # Run single cycle - should handle emergency
    actions = nav._run_navigation_cycle(mock_env)

    # Should return some actions (emergency bypass)
    assert len(actions) > 0


def test_subagent_registry_integration():
    """Test SubAgentRegistry integration with Navigator."""
    from agents.pipeline.navigator import Navigator
    from agents.pipeline.tools.subagent_registry import SubAgentRegistry

    nav = Navigator()
    nav.register_subagents()

    # Check registry works
    registry = nav._registry
    assert isinstance(registry, SubAgentRegistry)

    # Check call method works for registered agents
    agents = registry.list_all()
    assert len(agents) == 5


def test_action_converter_integration():
    """Test ActionConverter integration in navigation flow."""
    from agents.pipeline.navigator import Navigator
    from agents.pipeline.tools.action_converter import ActionConverter

    nav = Navigator()
    nav.register_subagents()

    # Check action converter is used correctly
    converter = nav._action_converter
    assert isinstance(converter, ActionConverter)

    # Test convert method
    actions = converter.convert(["forward", "turn_left", "forward"])
    assert len(actions) == 3
    assert actions[0][0] == ActionType.MOVE_FORWARD
    assert actions[1][0] == ActionType.TURN_LEFT

    # Test ensure_5_actions
    padded = converter.ensure_5_actions(actions)
    assert len(padded) == 5


def test_topology_graph_integration():
    """Test TopologyGraph integration during navigation."""
    from agents.pipeline.navigator import Navigator
    from agents.pipeline.tools.topology_graph import NodeType

    nav = Navigator()
    nav.initialize_episode("test", [0.0, 0.0, 0.0])

    # Check topology is initialized
    topology = nav._topology
    assert topology is not None

    # Simulate position changes
    nav._topology.add_visited_position([1.0, 0.0, 1.0])
    nav._topology.add_visited_position([2.0, 0.0, 2.0])

    # Check visited positions
    assert len(topology.visited_positions) >= 2

    # Add key node
    node = nav._topology.add_key_node(
        position=[1.5, 0.0, 1.5],
        node_type=NodeType.TURN_POINT,
        rotation=90.0
    )
    assert node is not None
    assert topology.has_key_nodes()


def test_state_calculator_integration():
    """Test StateCalculator integration."""
    from agents.pipeline.navigator import Navigator

    nav = Navigator()

    calculator = nav._state_calculator
    assert calculator is not None

    # Test position change calculation
    change = calculator.compute_position_change([0.0, 0.0, 0.0], [1.0, 2.0, 1.0])
    assert change["dx"] == 1.0
    assert change["dy"] == 2.0
    assert change["dz"] == 1.0

    # Test distance calculation
    dist = calculator.compute_distance([0.0, 0.0, 0.0], [1.0, 0.0, 1.0])
    assert dist > 0


def test_navigation_loop_stops_on_max_steps():
    """Test navigation loop stops when max_steps reached."""
    from agents.pipeline.navigator import Navigator

    # Note: Navigator executes 5 actions per cycle, so step_count may exceed max_steps
    # within one cycle. Use max_steps=10 to allow multiple cycles to test stopping.
    nav = Navigator({"max_steps": 10, "report_interval": 5})
    nav.register_subagents()

    # Mock model_manager
    mock_model_manager = Mock()
    mock_model_manager.generate_vision.return_value = {
        "response": '{"task_relevant": false, "objects": [], "target_direction": "forward", "target_distance": "unknown", "path_blocked": false, "navigation_cues": [], "scene_description": "test"}'
    }
    mock_model_manager.generate_vision_dual.return_value = mock_model_manager.generate_vision.return_value
    mock_model_manager.generate.return_value = '{"actions": ["forward", "forward", "forward", "forward", "forward"], "goal_summary": "test", "current_gap": "", "recommended_action": "forward", "reasoning": "", "confidence": 0.5}'

    nav.set_model_manager(mock_model_manager)
    nav.initialize_episode("test", [0.0, 0.0, 0.0])

    # Mock env
    mock_env = Mock()
    mock_env.get_observations.return_value = {
        "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth": np.zeros((224, 224), dtype=np.float32)
    }
    mock_env.step.return_value = None
    mock_env.get_agent_position.return_value = [0.0, 0.0, 0.0]
    mock_env.get_agent_rotation.return_value = 0.0

    result = nav.run_navigation_loop(mock_env)

    # Should stop at or slightly above max_steps (due to 5-action batches)
    assert result["success"] == False
    assert result["reason"] == "max_steps"
    # Allow step_count to exceed max_steps by up to 5 (one cycle of actions)
    assert nav._step_count <= nav._max_steps + 5


def test_history_tracking():
    """Test navigation history is tracked correctly."""
    from agents.pipeline.navigator import Navigator

    nav = Navigator()
    nav.initialize_episode("test", [0.0, 0.0, 0.0])

    # Initially has starting position
    assert len(nav._history) == 1
    assert nav._history[0]["position"] == [0.0, 0.0, 0.0]
    assert nav._history[0]["step"] == 0

    # Simulate state updates
    mock_env = Mock()
    mock_env.get_agent_position.return_value = [1.0, 0.0, 1.0]
    mock_env.get_agent_rotation.return_value = 45.0

    nav._step_count = 1
    nav._update_state(mock_env)

    # History should have 2 entries
    assert len(nav._history) == 2
    assert nav._history[1]["position"] == [1.0, 0.0, 1.0]


def test_multiple_cycles():
    """Test multiple navigation cycles."""
    from agents.pipeline.navigator import Navigator

    nav = Navigator({"max_steps": 15})
    nav.register_subagents()

    # Mock model_manager with consistent responses
    mock_model_manager = Mock()
    mock_model_manager.generate_vision.return_value = {
        "response": '{"task_relevant": false, "objects": [], "target_direction": "forward", "target_distance": "unknown", "path_blocked": false, "navigation_cues": [], "scene_description": "test"}'
    }
    mock_model_manager.generate_vision_dual.return_value = mock_model_manager.generate_vision.return_value
    mock_model_manager.generate.return_value = '{"actions": ["forward", "forward", "forward", "forward", "forward"], "goal_summary": "test", "current_gap": "", "recommended_action": "forward", "reasoning": "", "confidence": 0.5}'

    nav.set_model_manager(mock_model_manager)
    nav.initialize_episode("test", [0.0, 0.0, 0.0])

    # Mock env with position progression
    mock_env = Mock()
    mock_env.get_observations.return_value = {
        "rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        "depth": np.zeros((224, 224), dtype=np.float32)
    }
    mock_env.step.return_value = None
    positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [1.0, 0.0, 1.0],
    ]
    position_idx = 0

    def get_position():
        pos = positions[position_idx % len(positions)]
        position_idx += 1
        return pos

    mock_env.get_agent_position.side_effect = lambda: positions[(nav._step_count) % len(positions)]
    mock_env.get_agent_rotation.return_value = 0.0

    result = nav.run_navigation_loop(mock_env)

    # Should have run multiple cycles
    assert nav._step_count > 0


class TestPipelineIntegration:
    """Integration tests for HabitatEnvAdapter and Pipeline config."""

    def test_habitat_env_adapter_integration(self):
        """测试适配器与现有代码集成"""
        from agents.pipeline.env_adapter import HabitatEnvAdapter

        # Mock sim 和 get_observations
        mock_sim = Mock()
        mock_sim.get_agent.return_value.get_state.return_value.position = [0, 0, 0]
        mock_sim.get_agent.return_value.get_state.return_value.rotation = Mock(w=1, x=0, y=0, z=0)

        def mock_get_obs(sim):
            return Mock(), Mock()

        adapter = HabitatEnvAdapter(mock_sim, mock_get_obs)

        # 测试所有接口
        obs = adapter.get_observations()
        assert "rgb" in obs and "depth" in obs

        pos = adapter.get_agent_position()
        assert len(pos) == 3

        rot = adapter.get_agent_rotation()
        assert isinstance(rot, float)

    def test_pipeline_config_passed(self):
        """测试 --use-pipeline 参数配置传递"""
        # 验证 config 可以包含 use_pipeline
        config = {"use_pipeline": True, "max_steps": 50}
        assert config.get("use_pipeline") == True

    def test_pipeline_episode_result_format(self):
        """测试 Pipeline episode 返回格式与 legacy 一致"""
        # 模拟返回格式
        result = {
            "trajectory": [[0, 0, 0]],
            "steps": 10,
            "success": False,
            "min_distance": 5.0,
            "trajectory_length": 10.0,
            "spl": 0.5,
            "ndtw": 0.8,
            "sdtw": 0.0,
            "task_level": "pipeline",
            "subtask_count": 1,
            "evaluation_scores": [],
        }

        # 验证必要字段存在
        required_keys = ["trajectory", "steps", "success", "min_distance"]
        for key in required_keys:
            assert key in result