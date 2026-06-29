#!/usr/bin/env python3
"""Test DecisionAgent LLM response directly."""

import sys
sys.path.insert(0, '/home/WZ/MA_VLN/habitat_vln/src')

from agents.decision_agent import DecisionAgent
from core.context import NavContext
from core.action import ActionSequence
import json

# Mock strategy data
strategy_data = {
    "perception": {
        "room_type": "走廊",
        "objects": [{"name": "stairs"}, {"name": "railing"}],
        "scene_description": "向下的楼梯，有扶手",
        "walkable_analysis": {
            "left": {"clear": False},
            "center": {"clear": True},
            "right": {"clear": True}
        },
        "obstacle_ahead": False,
        "nav_hint": "前方是向下的楼梯"
    },
    "trajectory": {
        "distance_traveled": 0.0,
        "heading": "朝南"
    },
    "instruction": {
        "directions": ["向下", "向右"],
        "instruction_analysis": {
            "landmarks": ["楼梯", "扶手"],
            "goals": ["楼下区域"]
        }
    },
    "analysis": "需要走下楼梯，然后右转"
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

    def get_distance_to_goal(self):
        return 5.0

# Create agent and test
agent = DecisionAgent()
agent._initialized = True

# Initialize model manager with REMOTE mode enabled
from models.model_manager import get_model_manager
agent._model_manager = get_model_manager({
    "use_remote": True,
    "use_remote_llm": True,
    "llm_server_url": "http://localhost:8000",
})
# Don't load local models - we use remote vLLM server

print("=" * 60)
print("Testing DecisionAgent LLM response")
print("=" * 60)

context = MockContext()
subtask = MockSubTask()

# Build prompt
prompt = agent._build_sequence_prompt_v2(context, subtask, strategy_data, "中等")
print("\n=== Prompt ===")
print(prompt[:500] + "...")

# Call LLM
print("\n=== Calling LLM ===")
try:
    response = agent._model_manager.generate(
        "qwen-9b-decision",
        prompt,
        max_new_tokens=300,
        temperature=0.1,
    )
    print(f"Response:\n{response}")

    # Parse response
    print("\n=== Parsing Response ===")
    actions, reasoning, subtask_completed = agent._parse_sequence_response(response)
    print(f"Reasoning: {reasoning}")
    print(f"Actions: {len(actions)} steps")
    print(f"Subtask completed: {subtask_completed}")

    if len(actions) < 5:
        print(f"\nERROR: Only got {len(actions)} actions, need at least 5")
    else:
        print(f"\nSUCCESS: Got {len(actions)} valid actions")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
