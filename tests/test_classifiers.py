"""Tests for classifiers module."""

import pytest
from unittest.mock import Mock, patch

from core.context import NavContext, NavContextBuilder, TaskType
from classifiers.rule_classifier import RuleClassifier, ClassificationResult
from classifiers.llm_classifier import LLMClassifier, LLMClassificationResult
from classifiers.task_classifier import TaskTypeClassifier


class TestRuleClassifier:
    """Tests for RuleClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = RuleClassifier()

    def test_classify_simple_turn(self):
        """Test classification of simple turn instruction."""
        context = NavContextBuilder().with_instruction("turn left").build()

        result = self.classifier.classify(context)

        assert result.task_type == TaskType.TYPE_0
        assert result.confidence > 0

    def test_classify_forward(self):
        """Test classification of forward instruction."""
        context = NavContextBuilder().with_instruction("move forward").build()

        result = self.classifier.classify(context)

        assert result.task_type == TaskType.TYPE_0

    def test_classify_path_following(self):
        """Test classification of path following."""
        context = NavContextBuilder().with_instruction(
            "walk down the hallway and turn right"
        ).build()

        result = self.classifier.classify(context)

        assert result.task_type in [TaskType.TYPE_0, TaskType.TYPE_1]

    def test_classify_object_search(self):
        """Test classification of object search."""
        context = NavContextBuilder().with_instruction(
            "find the red chair in the room"
        ).build()

        result = self.classifier.classify(context)

        # Should be Type-2 for object search
        assert result.task_type in [TaskType.TYPE_1, TaskType.TYPE_2]

    def test_classify_multi_room(self):
        """Test classification of multi-room navigation."""
        context = NavContextBuilder().with_instruction(
            "go to the kitchen through the living room"
        ).build()

        result = self.classifier.classify(context)

        # Should be Type-3 for multi-room
        assert result.task_type in [TaskType.TYPE_2, TaskType.TYPE_3]

    def test_classify_conditional(self):
        """Test classification of conditional navigation."""
        context = NavContextBuilder().with_instruction(
            "if you see a door turn left, otherwise go straight"
        ).build()

        result = self.classifier.classify(context)

        # Should be Type-4 for conditional
        assert result.task_type in [TaskType.TYPE_3, TaskType.TYPE_4]

    def test_get_complexity_score(self):
        """Test complexity score calculation."""
        context = NavContextBuilder().with_instruction("turn left").build()

        score = self.classifier.get_task_complexity_score(context)

        assert 0 <= score <= 1


class TestLLMClassifier:
    """Tests for LLMClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = LLMClassifier({"model_type": "mock"})

    def test_classify_with_mock(self):
        """Test classification with mock model."""
        context = NavContextBuilder().with_instruction(
            "find the chair"
        ).build()

        result = self.classifier.classify(context)

        assert result.task_type in [TaskType.TYPE_0, TaskType.TYPE_1, TaskType.TYPE_2, TaskType.TYPE_3, TaskType.TYPE_4]
        assert result.confidence >= 0

    def test_parse_response(self):
        """Test response parsing."""
        response = '{"task_type": "Type-2", "confidence": 0.85, "reasoning": "Object search", "subtasks": [], "complexity_factors": {}}'

        result = self.classifier._parse_response(response)

        assert result.task_type == TaskType.TYPE_2
        assert result.confidence == 0.85

    def test_default_result(self):
        """Test default result when parsing fails."""
        result = self.classifier._default_result()

        assert result.task_type == TaskType.TYPE_1
        assert result.confidence == 0.5


class TestTaskTypeClassifier:
    """Tests for TaskTypeClassifier."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TaskTypeClassifier({
            "rule_threshold": 0.9,
            "use_llm_fallback": True,
            "cache_results": False,
        })

    def test_classify_simple_high_confidence(self):
        """Test that simple instructions use rule classifier."""
        context = NavContextBuilder().with_instruction("turn left").build()

        task_type = self.classifier.classify(context)

        assert task_type == TaskType.TYPE_0

    def test_classify_with_details(self):
        """Test detailed classification."""
        context = NavContextBuilder().with_instruction("go forward").build()

        result = self.classifier.classify_with_details(context)

        assert "task_type" in result
        assert "method" in result
        assert result["task_type"] is not None

    def test_statistics(self):
        """Test statistics tracking."""
        context = NavContextBuilder().with_instruction("turn left").build()

        self.classifier.classify(context)
        stats = self.classifier.get_statistics()

        assert stats["total_classifications"] == 1
        assert stats["rule_classifications"] >= 1

    def test_clear_cache(self):
        """Test cache clearing."""
        self.classifier._cache["test"] = TaskType.TYPE_0

        self.classifier.clear_cache()

        assert len(self.classifier._cache) == 0


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_create_result(self):
        """Test creating classification result."""
        result = ClassificationResult(
            task_type=TaskType.TYPE_2,
            confidence=0.85,
            reasoning="Object search task",
        )

        assert result.task_type == TaskType.TYPE_2
        assert result.confidence == 0.85
        assert "Object" in result.reasoning


class TestLLMClassificationResult:
    """Tests for LLMClassificationResult dataclass."""

    def test_create_result(self):
        """Test creating LLM classification result."""
        result = LLMClassificationResult(
            task_type=TaskType.TYPE_3,
            confidence=0.8,
            reasoning="Multi-room navigation",
            subtasks=["exit room", "navigate hallway", "enter room"],
            complexity_factors={"multi_room": True},
        )

        assert result.task_type == TaskType.TYPE_3
        assert len(result.subtasks) == 3
        assert result.complexity_factors["multi_room"] is True