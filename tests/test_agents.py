"""Tests for agents module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from core.context import NavContext, NavContextBuilder, SubTask, VisualFeatures
from core.action import Action, ActionType
from agents.base_agent import BaseAgent, AgentOutput, AgentRole, AgentRegistry
from agents.instruction_agent import InstructionAgent
from agents.perception_agent import PerceptionAgent
from agents.trajectory_agent import TrajectoryAgent
from agents.decision_agent import DecisionAgent


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_agent_output_success(self):
        """Test creating successful agent output."""
        output = AgentOutput.success_output(
            data={"test": "value"},
            confidence=0.9,
            reasoning="Test reasoning",
        )

        assert output.success is True
        assert output.data["test"] == "value"
        assert output.confidence == 0.9

    def test_agent_output_failure(self):
        """Test creating failure agent output."""
        output = AgentOutput.failure_output(
            errors=["Error 1", "Error 2"],
            reasoning="Failed",
        )

        assert output.success is False
        assert len(output.errors) == 2

    def test_agent_output_to_dict(self):
        """Test converting output to dictionary."""
        output = AgentOutput.success_output(data={"key": "value"})

        result = output.to_dict()

        assert result["success"] is True
        assert result["data"]["key"] == "value"


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        AgentRegistry.clear()

    def test_register_agent(self):
        """Test registering an agent."""
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "test_agent"

        AgentRegistry.register(mock_agent)

        assert AgentRegistry.get("test_agent") == mock_agent

    def test_get_nonexistent_agent(self):
        """Test getting non-existent agent."""
        result = AgentRegistry.get("nonexistent")

        assert result is None

    def test_get_all_agents(self):
        """Test getting all registered agents."""
        mock1 = Mock(spec=BaseAgent)
        mock1.name = "agent1"
        mock2 = Mock(spec=BaseAgent)
        mock2.name = "agent2"

        AgentRegistry.register(mock1)
        AgentRegistry.register(mock2)

        all_agents = AgentRegistry.get_all()

        assert len(all_agents) == 2


class TestInstructionAgent:
    """Tests for InstructionAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = InstructionAgent()

    def test_agent_properties(self):
        """Test agent properties."""
        assert self.agent.name == "instruction_agent"
        assert self.agent.role == AgentRole.INSTRUCTION

    def test_process_simple_instruction(self):
        """Test processing simple instruction."""
        context = NavContextBuilder().with_instruction(
            "turn left and go forward"
        ).build()

        output = self.agent.process(context)

        assert output.success is True
        assert "subtasks" in output.data

    def test_process_instruction_with_landmarks(self):
        """Test processing instruction with landmarks."""
        context = NavContextBuilder().with_instruction(
            "go to the kitchen and find the table"
        ).build()

        output = self.agent.process(context)

        assert output.success is True
        assert "landmarks" in output.data

    def test_parse_instruction(self):
        """Test instruction parsing."""
        parsed = self.agent._parse_instruction("turn left and go forward")

        assert "directions" in parsed
        assert len(parsed["directions"]) > 0

    def test_create_subtasks(self):
        """Test subtask creation."""
        parsed = {
            "original": "turn left and go forward",
            "landmarks": [],
            "goals": [],
            "directions": ["left", "forward"],
            "actions": [],
            "conditions": [],
            "complexity": 0.5,
            "confidence": 0.8,
        }

        subtasks = self.agent._create_subtasks(parsed)

        assert len(subtasks) >= 1


class TestPerceptionAgent:
    """Tests for PerceptionAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PerceptionAgent()

    def test_agent_properties(self):
        """Test agent properties."""
        assert self.agent.name == "perception_agent"
        assert self.agent.role == AgentRole.PERCEPTION

    def test_process_visual_features(self):
        """Test processing visual features."""
        visual_features = VisualFeatures(
            scene_description="A living room with a sofa",
            object_detections=[{"name": "sofa", "confidence": 0.9}],
        )

        context = NavContextBuilder().with_instruction("test").build()
        context.visual_features = visual_features

        output = self.agent.process(context)

        assert output.success is True
        assert "scene_description" in output.data

    def test_classify_room(self):
        """Test room classification."""
        objects = [
            {"name": "bed", "confidence": 0.9},
            {"name": "nightstand", "confidence": 0.8},
        ]

        room_type, confidence = self.agent._classify_room(None, objects)

        assert room_type in ["bedroom", "unknown"]


class TestTrajectoryAgent:
    """Tests for TrajectoryAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = TrajectoryAgent()

    def test_agent_properties(self):
        """Test agent properties."""
        assert self.agent.name == "trajectory_agent"
        assert self.agent.role == AgentRole.TRAJECTORY

    def test_process_trajectory(self):
        """Test trajectory processing."""
        context = NavContextBuilder().with_instruction("test").build()
        context.trajectory = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        context.step_count = 3

        output = self.agent.process(context)

        assert output.success is True
        assert "progress" in output.data

    def test_calculate_progress(self):
        """Test progress calculation."""
        context = NavContextBuilder().with_instruction("test").build()
        context.step_count = 50
        context.trajectory = [(0, 0, 0)] * 10

        progress = self.agent._calculate_progress(context)

        assert 0 <= progress <= 1

    def test_check_backtracking(self):
        """Test backtracking detection."""
        # Straight trajectory
        straight = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        score = self.agent._check_backtracking(straight)

        assert score > 0.5  # Good score for straight path


class TestDecisionAgent:
    """Tests for DecisionAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.agent = DecisionAgent()

    def test_agent_properties(self):
        """Test agent properties."""
        assert self.agent.name == "decision_agent"
        assert self.agent.role == AgentRole.DECISION

    def test_process_decision(self):
        """Test decision making."""
        context = NavContextBuilder().with_instruction("turn left").build()

        output = self.agent.process(context)

        assert output.success is True
        assert "action" in output.data

    def test_decide_from_instruction(self):
        """Test instruction-based decision."""
        context = NavContextBuilder().with_instruction("turn right").build()

        action, confidence, reasoning = self.agent._decide_from_instruction(context, {})

        assert action.action_type == ActionType.TURN_RIGHT

    def test_generate_alternatives(self):
        """Test alternative action generation."""
        context = NavContextBuilder().with_instruction("test").build()
        action = Action.forward()

        alternatives = self.agent._generate_alternatives(context, action)

        assert len(alternatives) > 0
        assert all(a["action"] != "forward" for a in alternatives)


class TestAgentIntegration:
    """Integration tests for agents working together."""

    def test_agents_pipeline(self):
        """Test agents working in pipeline."""
        # Create agents
        instruction_agent = InstructionAgent()
        decision_agent = DecisionAgent()

        # Create context
        context = NavContextBuilder().with_instruction(
            "turn left and go forward"
        ).build()

        # Run instruction agent
        inst_output = instruction_agent.process(context)
        assert inst_output.success

        # Run decision agent
        dec_output = decision_agent.process(context)
        assert dec_output.success

        # Check that decision was made
        assert dec_output.data["action"] in ["forward", "turn_left", "turn_right", "stop"]