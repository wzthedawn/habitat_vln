#!/usr/bin/env python3
"""
Test script for generating paired normal samples.

Validates:
1. Format consistency with existing normal samples
2. original_instruction extraction
3. Actions generation logic
"""

import json
import random
import os
from typing import Dict, List, Tuple, Any

# Paths
EMERGENCY_DATA_PATH = "/data/WZ/Dataset/emergency_dataset/emergency_instructions.json"
R2R_PATH = "/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json"
OUTPUT_PATH = "/data/WZ/Dataset/test_normal_samples.jsonl"


def format_normal_sample(
    goal_position: List[float],
    instruction: str,
    actions: List[str] = None,
    reasoning_suffix: str = ""
) -> str:
    """Format a normal navigation sample following the exact template."""

    # Generate actions if not provided
    if actions is None:
        actions = generate_actions(len(instruction), instruction)

    # Truncate instruction for reasoning
    inst_short = instruction[:50] + "..." if len(instruction) > 50 else instruction

    # Build reasoning
    reasoning = f"Normal navigation task. Clear path to goal. Following instruction: {inst_short}"

    sample = f"""<|im_start|>system
You are a navigation assistant. Given the navigation context, determine the best action sequence to reach the goal safely.
<|im_end|>
<|im_start|>user
Navigation Context:
- Current Position: Starting point
- Goal Position: {json.dumps(goal_position)}
- No obstacles detected
- Clear path ahead

Instruction: {instruction}

Provide navigation actions to reach the goal.
<|im_end|>
<|im_start|>assistant
{{
  "risk_score": 0.1,
  "reasoning": "{reasoning}",
  "actions": {json.dumps(actions)},
  "emergency_mode": false
}}
<|im_end|>"""

    return sample


def generate_actions(geodesic_distance: float, instruction: str) -> List[str]:
    """Generate reasonable actions based on distance and instruction."""

    actions = []
    inst_lower = instruction.lower()

    # Base forward count (assuming ~0.25m per step)
    base_forward = max(3, min(6, int(geodesic_distance / 3)))

    # Detect turn instructions
    turn_count = inst_lower.count('turn')

    # Build action sequence
    if 'turn left' in inst_lower:
        actions.append('turn_left')
    elif 'turn right' in inst_lower:
        actions.append('turn_right')

    # Add forward movements
    for _ in range(base_forward):
        actions.append('forward')

    # Add second turn if instruction has multiple turns
    if turn_count > 1:
        remaining = inst_lower[inst_lower.find('turn') + 10:]
        if 'left' in remaining:
            actions.append('turn_left')
        else:
            actions.append('turn_right')
        actions.append('forward')

    # Ensure length in 3-8 range
    while len(actions) < 3:
        actions.append('forward')
    while len(actions) > 8:
        actions = actions[:8]

    return actions


def test_paired_samples():
    """Test generating paired normal samples from emergency data."""

    print("=" * 70)
    print("Test 1: Paired Normal Samples (from emergency original_instruction)")
    print("=" * 70)

    # Load emergency data
    with open(EMERGENCY_DATA_PATH, 'r') as f:
        emergency_data = json.load(f)

    episodes = emergency_data['episodes']

    # Get unique goals with their original instructions
    goal_info = {}
    for ep in episodes:
        goal = tuple(round(x, 2) for x in ep['goal_position'])
        if goal not in goal_info:
            goal_info[goal] = {
                'goal_position': ep['goal_position'],
                'original_instruction': ep['original_instruction'],
                'geodesic_distance': ep.get('geodesic_distance', 10),
                'scene_id': ep['scene_id'],
            }

    print(f"\nFound {len(goal_info)} unique goals in emergency data")

    # Generate 5 test samples
    test_samples = []
    goals = list(goal_info.keys())[:5]

    for i, goal in enumerate(goals):
        info = goal_info[goal]

        sample_text = format_normal_sample(
            goal_position=info['goal_position'],
            instruction=info['original_instruction'],
            actions=generate_actions(info['geodesic_distance'], info['original_instruction'])
        )

        test_samples.append({
            'text': sample_text,
            'metadata': {
                'source': 'paired',
                'goal_position': info['goal_position'],
                'scene_id': info['scene_id'],
                'original_instruction': info['original_instruction'][:60] + '...',
            }
        })

        print(f"\nSample {i+1}:")
        print(f"  Goal: {goal}")
        print(f"  Instruction: {info['original_instruction'][:60]}...")

    return test_samples


def test_extra_samples():
    """Test generating extra normal samples from R2R."""

    print("\n" + "=" * 70)
    print("Test 2: Extra Normal Samples (from R2R val_seen)")
    print("=" * 70)

    # Load R2R data
    with open(R2R_PATH, 'r') as f:
        r2r_data = json.load(f)

    episodes = r2r_data.get('episodes', r2r_data)

    # Load emergency goals for filtering
    with open(EMERGENCY_DATA_PATH, 'r') as f:
        emergency_data = json.load(f)

    emergency_goals = set()
    for ep in emergency_data['episodes']:
        goal = tuple(round(x, 2) for x in ep['goal_position'])
        emergency_goals.add(goal)

    # Filter R2R episodes
    available = []
    for ep in episodes:
        goals = ep.get('goals', [])
        if goals:
            goal_pos = tuple(round(x, 2) for x in goals[0].get('position', [0,0,0]))
            if goal_pos not in emergency_goals:
                inst = ep.get('instruction', {})
                inst_text = inst.get('instruction_text', '') if isinstance(inst, dict) else str(inst)
                available.append({
                    'goal_position': goals[0].get('position'),
                    'instruction': inst_text,
                    'geodesic_distance': ep.get('info', {}).get('geodesic_distance', 10),
                    'scene_id': ep.get('scene_id', ''),
                })

    print(f"\nFound {len(available)} R2R episodes not overlapping with emergency goals")

    # Generate 5 test samples
    test_samples = []

    for i, ep_info in enumerate(available[:5]):
        sample_text = format_normal_sample(
            goal_position=ep_info['goal_position'],
            instruction=ep_info['instruction'],
            actions=generate_actions(ep_info['geodesic_distance'], ep_info['instruction'])
        )

        test_samples.append({
            'text': sample_text,
            'metadata': {
                'source': 'r2r_extra',
                'goal_position': ep_info['goal_position'],
                'scene_id': ep_info['scene_id'],
                'instruction': ep_info['instruction'][:60] + '...',
            }
        })

        print(f"\nSample {i+1}:")
        print(f"  Goal: {tuple(round(x, 2) for x in ep_info['goal_position'])}")
        print(f"  Instruction: {ep_info['instruction'][:60]}...")

    return test_samples


def validate_format(test_samples: List[Dict], existing_samples: List[Dict]) -> bool:
    """Validate test samples match existing format."""

    print("\n" + "=" * 70)
    print("Test 3: Format Validation")
    print("=" * 70)

    if not existing_samples:
        print("No existing samples to compare against")
        return False

    existing = existing_samples[0]['text']
    test = test_samples[0]['text']

    # Check key markers
    checks = [
        ('<|im_start|>system', 'System marker'),
        ('<|im_start|>user', 'User marker'),
        ('<|im_start|>assistant', 'Assistant marker'),
        ('Navigation Context:', 'Context header'),
        ('Goal Position:', 'Goal field'),
        ('No obstacles detected', 'Obstacle field'),
        ('Clear path ahead', 'Path field'),
        ('risk_score', 'Risk score field'),
        ('reasoning', 'Reasoning field'),
        ('actions', 'Actions field'),
        ('emergency_mode', 'Emergency mode field'),
    ]

    all_passed = True
    for marker, desc in checks:
        existing_has = marker in existing
        test_has = marker in test

        status = "✓" if test_has else "✗"
        if not test_has:
            all_passed = False

        print(f"  {status} {desc}: {'present' if test_has else 'MISSING'}")

    return all_passed


def main():
    print("=" * 70)
    print("Normal Sample Generation - Small Scale Test")
    print("=" * 70)

    # Load existing normal samples for comparison
    with open('/data/WZ/Dataset/qlora_mixed/decision_train.jsonl', 'r') as f:
        samples = [json.loads(line) for line in f]

    existing_normal = [s for s in samples if 'Obstacle Detected:' not in s['text']]
    print(f"\nLoaded {len(existing_normal)} existing normal samples for comparison")

    # Test 1: Paired samples
    paired_samples = test_paired_samples()

    # Test 2: Extra samples
    extra_samples = test_extra_samples()

    # Test 3: Format validation
    all_samples = paired_samples + extra_samples
    format_ok = validate_format(all_samples, existing_normal)

    # Save test output
    print("\n" + "=" * 70)
    print("Saving Test Output")
    print("=" * 70)

    os.makedirs('data', exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps({'text': sample['text']}, ensure_ascii=False) + '\n')

    print(f"\nSaved {len(all_samples)} test samples to {OUTPUT_PATH}")

    # Print sample comparison
    print("\n" + "=" * 70)
    print("Sample Comparison (Test vs Existing)")
    print("=" * 70)

    print("\n【Existing Normal Sample】:")
    print(existing_normal[0]['text'][:500] + "...")

    print("\n【Generated Test Sample】:")
    print(all_samples[0]['text'][:500] + "...")

    # Final status
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"\n  Paired samples: {len(paired_samples)} generated ✓")
    print(f"  Extra samples: {len(extra_samples)} generated ✓")
    print(f"  Format validation: {'PASSED' if format_ok else 'FAILED'} {'✓' if format_ok else '✗'}")

    if format_ok:
        print("\n✓ All tests passed. Ready for full-scale generation.")
        return True
    else:
        print("\n✗ Some tests failed. Please review the output.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)