#!/usr/bin/env python3
"""Test DecisionAgent action sequence generation directly."""

import sys
sys.path.insert(0, '/home/WZ/MA_VLN/habitat_vln/src')

from agents.decision_agent import DecisionAgent
from core.action import ActionSequence
import json

# Mock strategy data (similar to what experiment produces)
strategy_data = {
    "perception": {
        "room_type": "走廊",
        "objects": [{"name": "glass door"}, {"name": "wall"}],
        "scene_description": "走廊，有玻璃门和墙壁",
        "walkable_analysis": {
            "left": {"clear": False},
            "center": {"clear": True},
            "right": {"clear": True}
        },
        "obstacle_ahead": {"blocked": False, "min_distance": 5.0},
        "nav_hint": "前方是走廊"
    },
    "trajectory": {
        "distance_traveled": 0.0,
        "heading": "朝南"
    },
    "instruction": {
        "directions": ["向下", "向右"],
        "instruction_analysis": {
            "landmarks": ["楼梯", "扶手", "地毯", "长椅", "钢琴"],
            "goals": ["楼下区域", "有地毯的地方"]
        }
    },
    "reflection": "当前策略正确，需确认楼梯位置。潜在问题：未识别楼梯入口，可能误判前方玻璃门为楼梯。建议：扫描右侧墙壁寻找楼梯结构，避免直接走向玻璃门。下一步导航重点：寻找楼梯入口并确认下降路径。"
}

# Mock subtask
class MockSubTask:
    def __init__(self):
        self.id = 0
        self.description = "Walk down the stairs"
        self.level = "中等"
        self.completion_condition = {
            "type": "y_change",
            "direction": "down",
            "min_change": 1.5,
            "description": "walk down the stairs"
        }

# Mock context
class MockContext:
    def __init__(self):
        self.position = (5.58, -1.62, 2.81)
        self.metadata = {"goal_position": (11.63, -3.16, 1.99)}

    def get_distance_to_goal(self):
        return 6.3

# Create agent with REMOTE mode enabled
from models.model_manager import get_model_manager
model_manager = get_model_manager({
    "use_remote": True,
    "use_remote_llm": True,
    "llm_server_url": "http://localhost:8000",
})

agent = DecisionAgent({"model_manager": model_manager})
# Don't set _initialized = True - let the agent initialize itself

print("=" * 60)
print("Testing DecisionAgent action sequence generation")
print("=" * 60)

context = MockContext()
subtask = MockSubTask()

# Build prompt
prompt = agent._build_sequence_prompt_v2(context, subtask, strategy_data, "中等")
print("\n=== Prompt (first 500 chars) ===")
print(prompt[:500] + "...")

# Call generate_action_sequence
print("\n=== Calling generate_action_sequence ===")
try:
    sequence = agent.generate_action_sequence(context, type('obj', (object,), {'metadata': strategy_data})(), subtask)
    print(f"SUCCESS: Got {len(sequence.actions)} actions")
    print(f"Reasoning: {sequence.reasoning[:200] if sequence.reasoning else 'None'}...")
    print(f"Subtask completed: {sequence.subtask_completed}")
    for i, (action_type, steps) in enumerate(sequence.actions[:5]):
        print(f"  Action {i+1}: {action_type}, steps={steps}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Also test raw LLM call
print("\n=== Testing raw LLM call ===")
try:
    response = model_manager.generate(
        "qwen-9b-decision",
        prompt,
        max_new_tokens=300,
        temperature=0.1,
    )
    print(f"Response:\n{response[:500] if response else 'None'}...")

    # Parse response
    actions, reasoning, subtask_completed = agent._parse_sequence_response(response)
    print(f"\nParsed: {len(actions)} actions")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
