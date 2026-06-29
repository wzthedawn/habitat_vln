# tests/pipeline/test_env_adapter.py

import pytest
from unittest.mock import Mock, MagicMock
import math


class TestHabitatEnvAdapter:
    def test_adapter_init(self):
        """测试适配器初始化"""
        # Import here to avoid import errors before implementation
        from agents.pipeline.env_adapter import HabitatEnvAdapter

        mock_sim = Mock()
        mock_get_obs = Mock(return_value=(Mock(), Mock()))

        adapter = HabitatEnvAdapter(mock_sim, mock_get_obs)

        assert adapter._sim == mock_sim
        assert adapter._get_observations == mock_get_obs

    def test_get_observations(self):
        """测试获取观察"""
        from agents.pipeline.env_adapter import HabitatEnvAdapter

        mock_rgb = Mock()
        mock_depth = Mock()
        mock_sim = Mock()
        mock_get_obs = Mock(return_value=(mock_rgb, mock_depth))

        adapter = HabitatEnvAdapter(mock_sim, mock_get_obs)
        obs = adapter.get_observations()

        assert "rgb" in obs
        assert "depth" in obs
        assert obs["rgb"] == mock_rgb
        assert obs["depth"] == mock_depth
        mock_get_obs.assert_called_once_with(mock_sim)

    def test_step(self):
        """测试执行动作"""
        from agents.pipeline.env_adapter import HabitatEnvAdapter

        mock_agent = Mock()
        mock_sim = Mock()
        mock_sim.get_agent.return_value = mock_agent

        adapter = HabitatEnvAdapter(mock_sim, Mock())

        from core.action import ActionType
        adapter.step(ActionType.MOVE_FORWARD)

        mock_sim.get_agent.assert_called_once_with(0)
        mock_agent.act.assert_called_once_with(ActionType.MOVE_FORWARD)

    def test_get_agent_position(self):
        """测试获取位置"""
        from agents.pipeline.env_adapter import HabitatEnvAdapter

        mock_state = Mock()
        mock_state.position = [1.0, 2.0, 3.0]

        mock_agent = Mock()
        mock_agent.get_state.return_value = mock_state

        mock_sim = Mock()
        mock_sim.get_agent.return_value = mock_agent

        adapter = HabitatEnvAdapter(mock_sim, Mock())
        pos = adapter.get_agent_position()

        assert pos == [1.0, 2.0, 3.0]

    def test_get_agent_rotation(self):
        """测试获取朝向"""
        from agents.pipeline.env_adapter import HabitatEnvAdapter

        # Mock quaternion (yaw = 45 degrees = pi/4)
        mock_quat = Mock()
        mock_quat.w = 0.924  # cos(22.5 degrees)
        mock_quat.x = 0.0
        mock_quat.y = 0.383  # sin(22.5 degrees)
        mock_quat.z = 0.0

        mock_state = Mock()
        mock_state.rotation = mock_quat

        mock_agent = Mock()
        mock_agent.get_state.return_value = mock_state

        mock_sim = Mock()
        mock_sim.get_agent.return_value = mock_agent

        adapter = HabitatEnvAdapter(mock_sim, Mock())
        yaw = adapter.get_agent_rotation()

        # Should be approximately pi/4 (45 degrees)
        assert abs(yaw - math.pi/4) < 0.1