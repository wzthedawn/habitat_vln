"""Tests for SubAgent base class and output data structures."""

import pytest
from dataclasses import fields
from typing import List, Dict, Any, Optional

from agents.pipeline.base_pipeline_agent import (
    ObservationOutput,
    AnalysisOutput,
    PlanningOutput,
    ReviewOutput,
    EmergencyEvent,
    SubAgent,
)
from agents.base_agent import BaseAgent, AgentRole


class TestObservationOutput:
    """Test ObservationOutput dataclass."""

    def test_observation_output_creation(self):
        """Test creating ObservationOutput instance."""
        output = ObservationOutput(
            task_relevant=True,
            objects=["stairs", "railing"],
            target_direction="forward",
            target_distance="medium",
            path_blocked=False,
            navigation_cues=["stairs entrance visible"],
            scene_description="stairwell entrance",
        )
        assert output.task_relevant == True
        assert output.objects == ["stairs", "railing"]
        assert output.target_direction == "forward"
        assert output.target_distance == "medium"
        assert output.path_blocked == False
        assert output.navigation_cues == ["stairs entrance visible"]
        assert output.scene_description == "stairwell entrance"

    def test_observation_output_default_scene_description(self):
        """Test ObservationOutput with default scene_description."""
        output = ObservationOutput(
            task_relevant=False,
            objects=[],
            target_direction="unknown",
            target_distance="unknown",
            path_blocked=False,
            navigation_cues=[],
        )
        assert output.scene_description == ""

    def test_observation_output_fields(self):
        """Test ObservationOutput has required fields."""
        field_names = {f.name for f in fields(ObservationOutput)}
        required_fields = {
            "task_relevant",
            "objects",
            "target_direction",
            "target_distance",
            "path_blocked",
            "navigation_cues",
            "scene_description",
        }
        assert required_fields.issubset(field_names)


class TestAnalysisOutput:
    """Test AnalysisOutput dataclass."""

    def test_analysis_output_creation(self):
        """Test creating AnalysisOutput instance."""
        output = AnalysisOutput(
            goal_summary="go downstairs",
            current_gap="not found stairs",
            recommended_action="turn_right",
            reasoning="stairs might be on right",
            confidence=0.8,
            strategy_used="cot",
        )
        assert output.goal_summary == "go downstairs"
        assert output.current_gap == "not found stairs"
        assert output.recommended_action == "turn_right"
        assert output.reasoning == "stairs might be on right"
        assert output.confidence == 0.8
        assert output.strategy_used == "cot"

    def test_analysis_output_fields(self):
        """Test AnalysisOutput has required fields."""
        field_names = {f.name for f in fields(AnalysisOutput)}
        required_fields = {
            "goal_summary",
            "current_gap",
            "recommended_action",
            "reasoning",
            "confidence",
            "strategy_used",
        }
        assert required_fields.issubset(field_names)


class TestPlanningOutput:
    """Test PlanningOutput dataclass."""

    def test_planning_output_creation(self):
        """Test creating PlanningOutput instance."""
        output = PlanningOutput(
            actions=["forward", "turn_right", "forward", "forward", "forward"],
            expected_result="reach stairs",
            algorithm_used="llm",
        )
        assert len(output.actions) == 5
        assert output.expected_result == "reach stairs"
        assert output.algorithm_used == "llm"
        assert output.path is None

    def test_planning_output_with_path(self):
        """Test PlanningOutput with path."""
        path = [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
        output = PlanningOutput(
            actions=["forward", "forward"],
            expected_result="reach target",
            algorithm_used="topology",
            path=path,
        )
        assert output.path == path
        assert output.algorithm_used == "topology"

    def test_planning_output_fields(self):
        """Test PlanningOutput has required fields."""
        field_names = {f.name for f in fields(PlanningOutput)}
        required_fields = {
            "actions",
            "expected_result",
            "algorithm_used",
            "path",
        }
        assert required_fields.issubset(field_names)


class TestReviewOutput:
    """Test ReviewOutput dataclass."""

    def test_review_output_creation(self):
        """Test creating ReviewOutput instance."""
        output = ReviewOutput(
            completed=False,
            reason="dy=-0.5m, need -1.5m",
            progress=0.33,
            current_value=0.5,
            threshold=1.5,
        )
        assert output.completed == False
        assert output.reason == "dy=-0.5m, need -1.5m"
        assert output.progress == 0.33
        assert output.current_value == 0.5
        assert output.threshold == 1.5

    def test_review_output_completed(self):
        """Test ReviewOutput when completed."""
        output = ReviewOutput(
            completed=True,
            reason="target reached",
            progress=1.0,
            current_value=1.5,
            threshold=1.5,
        )
        assert output.completed == True
        assert output.progress == 1.0

    def test_review_output_fields(self):
        """Test ReviewOutput has required fields."""
        field_names = {f.name for f in fields(ReviewOutput)}
        required_fields = {
            "completed",
            "reason",
            "progress",
            "current_value",
            "threshold",
        }
        assert required_fields.issubset(field_names)


class TestEmergencyEvent:
    """Test EmergencyEvent dataclass."""

    def test_emergency_event_creation(self):
        """Test creating EmergencyEvent instance."""
        event = EmergencyEvent(
            type="obstacle",
            severity="medium",
            details={"collision": False, "path_blocked": True},
        )
        assert event.type == "obstacle"
        assert event.severity == "medium"
        assert event.details == {"collision": False, "path_blocked": True}

    def test_emergency_event_evacuate(self):
        """Test EmergencyEvent for evacuation."""
        event = EmergencyEvent(
            type="evacuate",
            severity="high",
            details={"reason": "stuck detected", "steps": 50},
        )
        assert event.type == "evacuate"
        assert event.severity == "high"
        assert event.details["reason"] == "stuck detected"

    def test_emergency_event_fields(self):
        """Test EmergencyEvent has required fields."""
        field_names = {f.name for f in fields(EmergencyEvent)}
        required_fields = {"type", "severity", "details"}
        assert required_fields.issubset(field_names)


class TestSubAgent:
    """Test SubAgent base class."""

    def test_sub_agent_is_abstract(self):
        """Test that SubAgent is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SubAgent()

    def test_sub_agent_inherits_from_base_agent(self):
        """Test that SubAgent inherits from BaseAgent."""

        class ConcreteSubAgent(SubAgent):
            @property
            def name(self) -> str:
                return "test_agent"

            @property
            def role(self) -> AgentRole:
                return AgentRole.DECISION

            def process(self, *args, **kwargs):
                return {"result": "ok"}

        agent = ConcreteSubAgent()
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, SubAgent)

    def test_sub_agent_has_model_manager(self):
        """Test that SubAgent has model_manager setter."""

        class ConcreteSubAgent(SubAgent):
            @property
            def name(self) -> str:
                return "test_agent"

            @property
            def role(self) -> AgentRole:
                return AgentRole.DECISION

            def process(self, *args, **kwargs):
                return {"result": "ok"}

        agent = ConcreteSubAgent()
        assert hasattr(agent, "_model_manager")
        assert agent._model_manager is None

        # Test set_model_manager
        mock_manager = object()
        agent.set_model_manager(mock_manager)
        assert agent._model_manager is mock_manager

    def test_sub_agent_call_llm_without_manager(self):
        """Test _call_llm raises error without model manager."""

        class ConcreteSubAgent(SubAgent):
            @property
            def name(self) -> str:
                return "test_agent"

            @property
            def role(self) -> AgentRole:
                return AgentRole.DECISION

            def process(self, *args, **kwargs):
                return {"result": "ok"}

        agent = ConcreteSubAgent()
        with pytest.raises(RuntimeError, match="ModelManager not set"):
            agent._call_llm("test prompt")

    def test_sub_agent_call_vlm_without_manager(self):
        """Test _call_vlm raises error without model manager."""

        class ConcreteSubAgent(SubAgent):
            @property
            def name(self) -> str:
                return "test_agent"

            @property
            def role(self) -> AgentRole:
                return AgentRole.DECISION

            def process(self, *args, **kwargs):
                return {"result": "ok"}

        agent = ConcreteSubAgent()
        with pytest.raises(RuntimeError, match="ModelManager not set"):
            agent._call_vlm("test prompt", [])

    def test_sub_agent_process_is_abstract(self):
        """Test that process method must be implemented."""

        class IncompleteSubAgent(SubAgent):
            @property
            def name(self) -> str:
                return "incomplete_agent"

            @property
            def role(self) -> AgentRole:
                return AgentRole.DECISION

        with pytest.raises(TypeError):
            IncompleteSubAgent()

    def test_sub_agent_with_mock_model_manager(self):
        """Test SubAgent with mock model manager."""

        class MockModelManager:
            def generate(self, prompt, max_tokens=300, temperature=0.3, **kwargs):
                return "LLM response"

            def generate_vision(self, prompt, images, max_tokens=300, temperature=0.2, **kwargs):
                return {"response": "VLM response", "objects": []}

        class ConcreteSubAgent(SubAgent):
            @property
            def name(self) -> str:
                return "test_agent"

            @property
            def role(self) -> AgentRole:
                return AgentRole.DECISION

            def process(self, *args, **kwargs):
                llm_result = self._call_llm("test prompt")
                return {"result": llm_result}

        agent = ConcreteSubAgent()
        agent.set_model_manager(MockModelManager())
        result = agent.process()
        assert result == {"result": "LLM response"}