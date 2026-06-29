"""Unit tests for CoT action extraction (_parse_action_suggestion method)."""

import pytest
from strategies.cot import CoTStrategy


class TestParseActionSuggestion:
    """Test cases for _parse_action_suggestion method."""

    @pytest.fixture
    def strategy(self):
        """Create a CoTStrategy instance for testing."""
        return CoTStrategy()

    def test_valid_json_parsing(self, strategy):
        """Test Layer 1: Valid JSON parsing."""
        response = '''{
            "analysis": "Looking at the scene, I need to turn left to find the kitchen.",
            "suggestion": {
                "direction": "left",
                "turn_count": {"min": 2, "max": 3},
                "forward_count": {"min": 3, "max": 5}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"
        assert result["suggestion"]["turn_count"]["min"] == 2
        assert result["suggestion"]["turn_count"]["max"] == 3
        assert result["suggestion"]["forward_count"]["min"] == 3
        assert result["suggestion"]["forward_count"]["max"] == 5
        assert result["subtask_completed"] is False

    def test_valid_json_right_direction(self, strategy):
        """Test Layer 1: Valid JSON with right direction."""
        response = '''{
            "analysis": "Turn right to face the door.",
            "suggestion": {
                "direction": "right",
                "turn_count": {"min": 1, "max": 2},
                "forward_count": {"min": 2, "max": 4}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "right"
        assert result["suggestion"]["turn_count"]["min"] == 1
        assert result["suggestion"]["turn_count"]["max"] == 2

    def test_valid_json_forward_direction(self, strategy):
        """Test Layer 1: Valid JSON with forward direction."""
        response = '''{
            "analysis": "Go straight ahead.",
            "suggestion": {
                "direction": "forward",
                "turn_count": {"min": 0, "max": 0},
                "forward_count": {"min": 5, "max": 7}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"
        assert result["suggestion"]["turn_count"]["min"] == 0
        assert result["suggestion"]["turn_count"]["max"] == 0

    def test_partial_json_missing_direction(self, strategy):
        """Test Layer 1: JSON missing direction field - should default to forward."""
        response = '''{
            "analysis": "Continue moving forward.",
            "suggestion": {
                "turn_count": {"min": 2, "max": 3},
                "forward_count": {"min": 3, "max": 5}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"
        assert result["suggestion"]["turn_count"]["min"] == 2
        assert result["suggestion"]["turn_count"]["max"] == 3

    def test_partial_json_missing_turn_count(self, strategy):
        """Test Layer 1: JSON missing turn_count - should use default."""
        response = '''{
            "analysis": "Turn left at the corner.",
            "suggestion": {
                "direction": "left",
                "forward_count": {"min": 3, "max": 5}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"
        assert result["suggestion"]["turn_count"]["min"] == 2
        assert result["suggestion"]["turn_count"]["max"] == 3

    def test_partial_json_missing_forward_count(self, strategy):
        """Test Layer 1: JSON missing forward_count - should use default."""
        response = '''{
            "analysis": "Turn right to the hallway.",
            "suggestion": {
                "direction": "right",
                "turn_count": {"min": 1, "max": 2}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "right"
        assert result["suggestion"]["forward_count"]["min"] == 3
        assert result["suggestion"]["forward_count"]["max"] == 5

    def test_regex_extract_direction_left(self, strategy):
        """Test Layer 2: Regex extraction for left direction."""
        response = '''Some text before. "direction": "left" some text after.'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"
        assert result["suggestion"]["turn_count"]["min"] == 2
        assert result["suggestion"]["turn_count"]["max"] == 3

    def test_regex_extract_direction_right(self, strategy):
        """Test Layer 2: Regex extraction for right direction."""
        response = '''The analysis shows: "direction":  "right" is needed.'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "right"

    def test_regex_extract_direction_forward(self, strategy):
        """Test Layer 2: Regex extraction for forward direction."""
        response = '''Based on analysis: "direction": "forward" is the choice.'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"

    def test_text_keyword_turn_left(self, strategy):
        """Test Layer 3: Text keyword inference for 'turn left'."""
        response = "Based on my analysis, I should turn left to find the kitchen."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_text_keyword_turn_right(self, strategy):
        """Test Layer 3: Text keyword inference for 'turn right'."""
        response = "The door is on the right side, so turn right."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "right"

    def test_text_keyword_forward(self, strategy):
        """Test Layer 3: Text keyword inference for 'forward'."""
        response = "The path ahead is clear, move forward."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"

    def test_text_keyword_straight(self, strategy):
        """Test Layer 3: Text keyword inference for 'straight'."""
        response = "Go straight ahead to reach the goal."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"

    def test_chinese_keyword_left(self, strategy):
        """Test Layer 3: Chinese keyword inference for left turn."""
        response = "分析结果表明应该左转进入厨房。"
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_chinese_keyword_right(self, strategy):
        """Test Layer 3: Chinese keyword inference for right turn."""
        response = "门在右边，需要右转。"
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "right"

    def test_chinese_keyword_forward(self, strategy):
        """Test Layer 3: Chinese keyword inference for forward movement."""
        response = "前方道路通畅，继续前进。"
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"

    def test_chinese_english_mixed(self, strategy):
        """Test Layer 3: Mixed Chinese and English keywords."""
        response = "According to the analysis, 应该左转 to find the exit."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_default_forward_fallback(self, strategy):
        """Test Layer 4: Default forward when no recognizable pattern."""
        response = "The situation is unclear and no direction can be determined."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"
        assert result["suggestion"]["turn_count"]["min"] == 0
        assert result["suggestion"]["turn_count"]["max"] == 0
        assert result["suggestion"]["forward_count"]["min"] == 3
        assert result["suggestion"]["forward_count"]["max"] == 5

    def test_empty_response(self, strategy):
        """Test Layer 4: Empty response defaults to forward."""
        response = ""
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"
        assert result["analysis"] == ""

    def test_malformed_json(self, strategy):
        """Test Layer 2-4: Malformed JSON falls back to regex/text."""
        response = '''{"analysis": "Turn left", "suggestion": {"direction": "left"'''
        # Should fall through to Layer 2 (regex) which should catch "direction": "left"
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_invalid_json_with_keywords(self, strategy):
        """Test Layer 3: Invalid JSON but contains text keywords."""
        response = '''{broken json} turn left to reach destination'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_priority_json_over_keywords(self, strategy):
        """Test that valid JSON takes priority over text keywords."""
        # JSON says left, but text says right - JSON should win
        response = '''{
            "analysis": "Turn right to find the door.",
            "suggestion": {
                "direction": "left",
                "turn_count": {"min": 2, "max": 3},
                "forward_count": {"min": 3, "max": 5}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_priority_regex_over_keywords(self, strategy):
        """Test that regex extraction takes priority over text keywords."""
        # Regex finds "left", text says "right" - regex should win
        response = '''The analysis suggests "direction": "left" but text says turn right.'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_analysis_truncation(self, strategy):
        """Test that analysis is truncated to 100 characters."""
        long_analysis = "A" * 200
        response = f"{{\"analysis\": \"{long_analysis}\", \"suggestion\": {{\"direction\": \"forward\"}}}}"
        result = strategy._parse_action_suggestion(response)

        # For non-JSON paths, analysis is truncated to first 100 chars
        response_text = "Turn left to find the door. " + "X" * 150
        result = strategy._parse_action_suggestion(response_text)

        assert len(result["analysis"]) == 100

    def test_subtask_completed_default(self, strategy):
        """Test that subtask_completed defaults to False."""
        response = "Turn left at the corridor."
        result = strategy._parse_action_suggestion(response)

        assert result["subtask_completed"] is False


class TestParseActionSuggestionEdgeCases:
    """Additional edge case tests for _parse_action_suggestion."""

    @pytest.fixture
    def strategy(self):
        """Create a CoTStrategy instance for testing."""
        return CoTStrategy()

    def test_whitespace_variations_in_json(self, strategy):
        """Test JSON with various whitespace patterns."""
        response = '''
        {
            "analysis" : "Go forward" ,
            "suggestion" : {
                "direction" : "forward" ,
                "turn_count" : { "min" : 0 , "max" : 0 } ,
                "forward_count" : { "min" : 3 , "max" : 5 }
            } ,
            "subtask_completed" : false
        }
        '''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "forward"

    def test_regex_case_insensitive_direction(self, strategy):
        """Test that regex handles various JSON formats."""
        # The regex looks for "direction": "value", test exact match
        response = '''Result: "direction":"left"'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_keyword_case_insensitivity(self, strategy):
        """Test that text keywords are case-insensitive."""
        response = "I should TURN LEFT to proceed."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_multiple_keywords_left_priority(self, strategy):
        """Test that 'turn left' keyword is detected when multiple directions mentioned."""
        response = "Maybe turn right, but actually turn left is better."
        result = strategy._parse_action_suggestion(response)

        # Should match 'turn left' which comes after 'turn right'
        assert result["suggestion"]["direction"] == "left"

    def test_multiple_keywords_right_priority(self, strategy):
        """Test that 'turn right' keyword is detected."""
        response = "The path on the left is blocked, turn right instead."
        result = strategy._parse_action_suggestion(response)

        # Should match 'turn right'
        assert result["suggestion"]["direction"] == "right"

    def test_unicode_in_response(self, strategy):
        """Test response with Unicode characters."""
        response = "Turn left ← to find the kitchen."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"

    def test_nested_json_structure(self, strategy):
        """Test response with nested JSON structure."""
        response = '''{
            "thinking": {
                "analysis": "Complex nested structure"
            },
            "suggestion": {
                "direction": "right",
                "turn_count": {"min": 1, "max": 2},
                "forward_count": {"min": 3, "max": 5}
            },
            "subtask_completed": false
        }'''
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "right"

    def test_special_characters_in_analysis(self, strategy):
        """Test response with special characters."""
        response = "Turn left! @#$% The path is clear."
        result = strategy._parse_action_suggestion(response)

        assert result["suggestion"]["direction"] == "left"


class TestConvertSuggestionToActions:
    """Test cases for DecisionAgent._convert_suggestion_to_actions method."""

    @pytest.fixture
    def agent(self):
        """Create a DecisionAgent instance for testing."""
        from agents.decision_agent import DecisionAgent
        return DecisionAgent({})

    def test_convert_suggestion_left(self, agent):
        """Test converting suggestion with left direction."""
        from core.action import ActionType
        suggestion = {"direction": "left", "turn_count": {"min": 2, "max": 3}, "forward_count": {"min": 3, "max": 5}}
        actions = agent._convert_suggestion_to_actions(suggestion)
        assert len(actions) >= 5
        assert actions[0][0] == ActionType.TURN_LEFT

    def test_convert_suggestion_forward(self, agent):
        """Test converting suggestion with forward direction."""
        from core.action import ActionType
        suggestion = {"direction": "forward", "turn_count": {"min": 0, "max": 0}, "forward_count": {"min": 3, "max": 5}}
        actions = agent._convert_suggestion_to_actions(suggestion)
        assert len(actions) >= 3
        for action_type, _ in actions:
            assert action_type == ActionType.MOVE_FORWARD

    def test_convert_suggestion_default(self, agent):
        """Test converting suggestion with default values."""
        from core.action import ActionType
        suggestion = {"direction": "right"}
        actions = agent._convert_suggestion_to_actions(suggestion)
        assert len(actions) >= 5
        assert actions[0][0] == ActionType.TURN_RIGHT

    def test_turn_count_middle_value(self, agent):
        """Test that turn_count uses middle value of min and max."""
        # min=2, max=4 -> middle = (2+4)//2 = 3
        suggestion = {"direction": "left", "turn_count": {"min": 2, "max": 4}, "forward_count": {"min": 3, "max": 5}}
        actions = agent._convert_suggestion_to_actions(suggestion)
        # Count TURN_LEFT actions
        turn_left_count = sum(1 for action_type, _ in actions if action_type.name == "TURN_LEFT")
        assert turn_left_count == 3, f"Expected 3 turn_left actions, got {turn_left_count}"

    def test_forward_count_middle_value(self, agent):
        """Test that forward_count uses middle value of min and max."""
        # min=3, max=7 -> middle = (3+7)//2 = 5
        suggestion = {"direction": "forward", "turn_count": {"min": 0, "max": 0}, "forward_count": {"min": 3, "max": 7}}
        actions = agent._convert_suggestion_to_actions(suggestion)
        # Count MOVE_FORWARD actions
        forward_count = sum(1 for action_type, _ in actions if action_type.name == "MOVE_FORWARD")
        assert forward_count == 5, f"Expected 5 forward actions, got {forward_count}"

    def test_turn_count_middle_value_respects_minimum(self, agent):
        """Test that turn_count respects minimum of 2 even if middle is lower."""
        # min=0, max=1 -> middle = (0+1)//2 = 0, but minimum is 2
        suggestion = {"direction": "left", "turn_count": {"min": 0, "max": 1}, "forward_count": {"min": 3, "max": 5}}
        actions = agent._convert_suggestion_to_actions(suggestion)
        turn_left_count = sum(1 for action_type, _ in actions if action_type.name == "TURN_LEFT")
        assert turn_left_count == 2, f"Expected 2 turn_left actions (minimum), got {turn_left_count}"

    def test_forward_count_middle_value_respects_minimum(self, agent):
        """Test that forward_count respects minimum of 3 even if middle is lower."""
        # min=0, max=2 -> middle = (0+2)//2 = 1, but minimum is 3
        suggestion = {"direction": "forward", "turn_count": {"min": 0, "max": 0}, "forward_count": {"min": 0, "max": 2}}
        actions = agent._convert_suggestion_to_actions(suggestion)
        forward_count = sum(1 for action_type, _ in actions if action_type.name == "MOVE_FORWARD")
        assert forward_count == 3, f"Expected 3 forward actions (minimum), got {forward_count}"


class TestIntegrationFullFlow:
    """Integration tests for CoT -> DecisionAgent data flow."""

    def test_full_flow_mock(self):
        """Test full flow: CoT execute -> DecisionAgent generate_sequence."""
        from strategies.cot import CoTStrategy
        from agents.decision_agent import DecisionAgent
        from core.action import ActionType
        from strategies.base_strategy import StrategyResult

        # Simulate parsed suggestion from CoT
        parsed = {
            "analysis": "Stairs on left, go down",
            "suggestion": {
                "direction": "left",
                "turn_count": {"min": 2, "max": 3},
                "forward_count": {"min": 3, "max": 5}
            },
            "subtask_completed": False
        }

        # Verify parsing structure
        assert parsed["suggestion"]["direction"] == "left"

        # Convert to actions
        agent = DecisionAgent({})
        actions = agent._convert_suggestion_to_actions(parsed["suggestion"])

        # Verify actions
        assert len(actions) >= 5, f"Expected at least 5 actions, got {len(actions)}"
        assert actions[0][0] == ActionType.TURN_LEFT, "First action should be turn_left"

        # Count action types
        turn_count = sum(1 for a in actions if a[0] == ActionType.TURN_LEFT)
        forward_count = sum(1 for a in actions if a[0] == ActionType.MOVE_FORWARD)

        assert turn_count >= 2, f"Expected at least 2 turns, got {turn_count}"
        assert forward_count >= 3, f"Expected at least 3 forwards, got {forward_count}"

    def test_suggestion_in_strategy_result(self):
        """Test that StrategyResult contains suggestion field."""
        strategy = CoTStrategy({})

        # Parse a mock LLM response
        mock_response = '''{
            "analysis": "Goal: stairs down left",
            "suggestion": {
                "direction": "left",
                "turn_count": {"min": 2, "max": 4},
                "forward_count": {"min": 3, "max": 6}
            },
            "subtask_completed": false
        }'''

        parsed = strategy._parse_action_suggestion(mock_response)

        # Verify structure
        assert "suggestion" in parsed
        assert "subtask_completed" in parsed
        assert parsed["suggestion"]["direction"] == "left"
        assert parsed["subtask_completed"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])