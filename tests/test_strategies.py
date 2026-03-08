"""Tests for strategies module."""

import pytest
from unittest.mock import Mock, patch

from core.context import NavContext, NavContextBuilder, VisualFeatures
from core.action import Action, ActionType
from strategies.base_strategy import BaseStrategy, StrategyResult, StrategyType, StrategyChain
from strategies.react import ReActStrategy
from strategies.cot import CoTStrategy
from strategies.debate import DebateStrategy
from strategies.reflection import ReflectionStrategy
from agents.decision_agent import DecisionAgent


class TestStrategyResult:
    """Tests for StrategyResult."""

    def test_create_result(self):
        """Test creating strategy result."""
        result = StrategyResult(
            success=True,
            action=Action.forward(),
            reasoning="Test reasoning",
            confidence=0.8,
        )

        assert result.success is True
        assert result.action.action_type == ActionType.MOVE_FORWARD
        assert result.confidence == 0.8

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = StrategyResult(
            success=True,
            action=Action.stop(),
            reasoning="Stop",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["action"] == "stop"


class TestStrategyChain:
    """Tests for StrategyChain."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_strategy1 = Mock(spec=BaseStrategy)
        self.mock_strategy1.execute.return_value = StrategyResult(
            success=True,
            action=Action.forward(),
        )

        self.mock_strategy2 = Mock(spec=BaseStrategy)
        self.mock_strategy2.execute.return_value = StrategyResult(
            success=True,
            action=Action.stop(),
        )

    def test_chain_execution(self):
        """Test executing strategy chain."""
        chain = StrategyChain([self.mock_strategy1, self.mock_strategy2])

        context = NavContextBuilder().with_instruction("test").build()
        result = chain.execute(context, [])

        assert result.success is True
        assert self.mock_strategy1.execute.called
        assert self.mock_strategy2.execute.called

    def test_chain_stops_on_failure(self):
        """Test that chain stops on failure."""
        self.mock_strategy1.execute.return_value = StrategyResult(
            success=False,
        )

        chain = StrategyChain([self.mock_strategy1, self.mock_strategy2])

        context = NavContextBuilder().with_instruction("test").build()
        result = chain.execute(context, [])

        assert result.success is False


class TestReActStrategy:
    """Tests for ReActStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ReActStrategy()

    def test_strategy_properties(self):
        """Test strategy properties."""
        assert self.strategy.name == "ReAct"
        assert self.strategy.strategy_type == StrategyType.REACT

    def test_execute_simple(self):
        """Test simple execution."""
        context = NavContextBuilder().with_instruction("turn left").build()
        agents = [DecisionAgent()]

        result = self.strategy.execute(context, agents)

        assert result.success is True
        assert result.action is not None

    def test_generate_thought(self):
        """Test thought generation."""
        context = NavContextBuilder().with_instruction(
            "go forward"
        ).build()
        context.action_history = [Action.forward()]

        thought = self.strategy._generate_thought(context, [], [])

        assert len(thought) > 0

    def test_keyword_decision(self):
        """Test keyword-based decision."""
        context = NavContextBuilder().with_instruction("turn right").build()

        action, confidence = self.strategy._keyword_decision(context, "")

        assert action.action_type == ActionType.TURN_RIGHT


class TestCoTStrategy:
    """Tests for CoTStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = CoTStrategy()

    def test_strategy_properties(self):
        """Test strategy properties."""
        assert self.strategy.name == "CoT"
        assert self.strategy.strategy_type == StrategyType.COT

    def test_execute(self):
        """Test CoT execution."""
        context = NavContextBuilder().with_instruction("find the chair").build()
        agents = [DecisionAgent()]

        result = self.strategy.execute(context, agents)

        assert result.success is True
        assert len(result.steps) > 0

    def test_define_question(self):
        """Test question definition."""
        context = NavContextBuilder().with_instruction("turn left").build()

        question = self.strategy._define_question(context)

        assert "turn left" in question.lower()

    def test_reasoning_steps(self):
        """Test reasoning step generation."""
        context = NavContextBuilder().with_instruction("test").build()

        steps = self.strategy._generate_reasoning_steps(context, "", [])

        assert len(steps) > 0


class TestDebateStrategy:
    """Tests for DebateStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = DebateStrategy()

    def test_strategy_properties(self):
        """Test strategy properties."""
        assert self.strategy.name == "Debate"
        assert self.strategy.strategy_type == StrategyType.DEBATE

    def test_execute_debate(self):
        """Test debate execution."""
        context = NavContextBuilder().with_instruction("test").build()
        agents = [DecisionAgent()]

        result = self.strategy.execute(context, agents)

        assert result.success is True

    def test_gather_proposals(self):
        """Test proposal gathering."""
        context = NavContextBuilder().with_instruction("turn left").build()
        agents = [DecisionAgent()]

        proposals = self.strategy._gather_proposals(context, agents)

        assert len(proposals) > 0

    def test_check_consensus(self):
        """Test consensus checking."""
        proposals = [
            {"action": "forward", "confidence": 0.8, "agent": "agent1"},
            {"action": "forward", "confidence": 0.7, "agent": "agent2"},
        ]

        consensus = self.strategy._check_consensus(proposals)

        assert consensus is True  # Both agree on forward


class TestReflectionStrategy:
    """Tests for ReflectionStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ReflectionStrategy()

    def test_strategy_properties(self):
        """Test strategy properties."""
        assert self.strategy.name == "Reflection"
        assert self.strategy.strategy_type == StrategyType.REFLECTION

    def test_execute_reflection(self):
        """Test reflection execution."""
        context = NavContextBuilder().with_instruction("test").build()
        context.action_history = [Action.forward(), Action.turn_left()]

        agents = [DecisionAgent()]

        result = self.strategy.execute(context, agents)

        assert result.success is True

    def test_review_recent_actions(self):
        """Test action review."""
        context = NavContextBuilder().with_instruction("test").build()
        context.action_history = [
            Action.forward(),
            Action.turn_left(),
            Action.forward(),
        ]

        review = self.strategy._review_recent_actions(context)

        assert "forward" in review.lower() or "turn" in review.lower()

    def test_generate_reflection(self):
        """Test reflection generation."""
        context = NavContextBuilder().with_instruction("turn left").build()
        context.action_history = [Action.forward(), Action.forward()]

        reflection = self.strategy._generate_reflection(context, "", "")

        assert len(reflection) > 0

    def test_store_lesson(self):
        """Test lesson storage."""
        context = NavContextBuilder().with_instruction("test").build()

        self.strategy._store_lesson(context, Action.forward(), "Test reflection")

        lessons = self.strategy.get_lessons()
        assert len(lessons) == 1


class TestStrategyIntegration:
    """Integration tests for strategies."""

    def test_strategies_in_sequence(self):
        """Test running strategies in sequence."""
        # Create context
        context = NavContextBuilder().with_instruction(
            "find the kitchen and turn left"
        ).build()
        context.visual_features = VisualFeatures(
            scene_description="A hallway",
        )

        # Create agents
        agents = [DecisionAgent()]

        # Run strategies
        react = ReActStrategy()
        cot = CoTStrategy()

        react_result = react.execute(context, agents)
        cot_result = cot.execute(context, agents, react_result)

        assert react_result.success
        assert cot_result.success

    def test_debate_resolution(self):
        """Test debate resolution with different proposals."""
        strategy = DebateStrategy()

        proposals = [
            {"agent": "agent1", "action": "forward", "confidence": 0.6},
            {"agent": "agent2", "action": "turn_left", "confidence": 0.7},
            {"agent": "agent3", "action": "forward", "confidence": 0.5},
        ]

        action, confidence, reasoning = strategy._resolve_debate(proposals, [])

        # Forward should win (2 votes vs 1)
        assert action.action_type in [ActionType.MOVE_FORWARD, ActionType.TURN_LEFT]