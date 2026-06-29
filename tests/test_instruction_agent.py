#!/usr/bin/env python3
"""Test InstructionAgent LLM response directly."""

import sys
sys.path.insert(0, '/home/WZ/MA_VLN/habitat_vln/src')

from agents.instruction_agent import InstructionAgent
import json

# Mock context
class MockContext:
    def __init__(self):
        self.instruction = "Walk down the stairs, turn right, and walk towards place with a rug. Wait near the bench and piano along the right side of the wall."
        self.metadata = {"goal_position": (11.63, -3.16, 1.99)}

# Create agent with REMOTE mode enabled
from models.model_manager import get_model_manager
model_manager = get_model_manager({
    "use_remote": True,
    "use_remote_llm": True,
    "llm_server_url": "http://localhost:8000",
})

agent = InstructionAgent({"model_manager": model_manager})

print("=" * 60)
print("Testing InstructionAgent LLM response")
print("=" * 60)

context = MockContext()

# Test semantic decomposition
print("\n=== Testing Semantic Decomposition ===")
subtasks = agent._semantic_decompose_with_llm(context.instruction, context.metadata.get("goal_position"))

if subtasks:
    print(f"SUCCESS: Got {len(subtasks)} subtasks")
    for st in subtasks:
        print(f"  - {st.description} (level: {st.level})")
else:
    print("FAILED: LLM decomposition returned None, will fallback to rule-based")

# Test full process
print("\n=== Testing Full Process ===")
result = agent.process(context)
print(f"Result success: {result.success}")
if result.success:
    data = result.data
    print(f"Subtasks: {len(data.get('subtasks', []))}")
    print(f"Task level: {data.get('task_level', 'unknown')}")
    print(f"Decomposition method: {data.get('decomposition_method', 'unknown')}")
else:
    print(f"Errors: {result.errors}")
