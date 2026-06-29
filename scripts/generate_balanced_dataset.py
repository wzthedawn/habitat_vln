#!/usr/bin/env python3
"""
Generate balanced training dataset with paired normal + emergency samples.

Strategy 3 (Recommended):
- Generate 127 paired normal samples (one per emergency goal, using original_instruction)
- Generate 273 extra normal samples (from R2R val_seen, non-overlapping)
- Split by obstacle type groups (same goal + obstacle_type stays together)

Output: /data/WZ/Dataset/qlora_balanced/
  - train.jsonl (~640 samples)
  - val.jsonl (~80 samples)
  - test.jsonl (~80 samples)
  - stats.json
"""

import json
import random
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Paths
EMERGENCY_DATA_PATH = "/data/WZ/Dataset/emergency_dataset/emergency_instructions.json"
R2R_PATH = "/data/WZ/Dataset/r2r/v1/val_seen/val_seen.json"
OUTPUT_DIR = "/data/WZ/Dataset/qlora_balanced"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


def format_normal_sample(
    goal_position: List[float],
    instruction: str,
    actions: List[str] = None,
) -> str:
    """Format a normal navigation sample following the exact template."""

    if actions is None:
        actions = generate_actions(len(instruction), instruction)

    inst_short = instruction[:50] + "..." if len(instruction) > 50 else instruction
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


def format_emergency_sample(
    goal_position: List[float],
    instruction: str,
    obstacle_type: str,
    obstacle_position: List[float],
    trigger_step: int,
    actions: List[str] = None,
) -> str:
    """Format an emergency navigation sample."""

    if actions is None:
        actions = generate_emergency_actions(obstacle_type, len(instruction))

    inst_short = instruction[:40] + "..." if len(instruction) > 40 else instruction
    reasoning = f"Emergency: {obstacle_type} detected at step {trigger_step}. {inst_short}"

    sample = f"""<|im_start|>system
You are an emergency navigation assistant. Given the navigation context and detected obstacle, determine the best action sequence to safely reach the goal.
<|im_end|>
<|im_start|>user
Navigation Context:
- Current Position: Starting point
- Goal Position: {json.dumps(goal_position)}
- Obstacle Detected: {obstacle_type} at step {trigger_step}
- Obstacle Position: {json.dumps(obstacle_position)}

Instruction: {instruction}
<|im_end|>
<|im_start|>assistant
{{
  "risk_score": 0.7,
  "reasoning": "{reasoning}",
  "actions": {json.dumps(actions)},
  "emergency_mode": true
}}
<|im_end|>"""

    return sample


def generate_actions(geodesic_distance: float, instruction: str) -> List[str]:
    """Generate reasonable actions for normal navigation."""

    actions = []
    inst_lower = instruction.lower()

    base_forward = max(3, min(6, int(geodesic_distance / 3)))

    if 'turn left' in inst_lower:
        actions.append('turn_left')
    elif 'turn right' in inst_lower:
        actions.append('turn_right')

    for _ in range(base_forward):
        actions.append('forward')

    turn_count = inst_lower.count('turn')
    if turn_count > 1:
        remaining = inst_lower[inst_lower.find('turn') + 10:]
        if 'left' in remaining:
            actions.append('turn_left')
        else:
            actions.append('turn_right')
        actions.append('forward')

    while len(actions) < 3:
        actions.append('forward')
    while len(actions) > 8:
        actions = actions[:8]

    return actions


def generate_emergency_actions(obstacle_type: str, complexity: int) -> List[str]:
    """Generate emergency response actions."""

    base_actions = {
        'blocked_path': ['turn_left', 'forward', 'forward', 'turn_right', 'forward', 'forward'],
        'dynamic_obstacle': ['forward', 'turn_right', 'forward', 'forward', 'turn_left', 'forward'],
        'emergency_evacuation': ['turn_right', 'turn_right', 'forward', 'forward', 'forward', 'forward'],
    }

    actions = base_actions.get(obstacle_type, ['forward', 'turn_left', 'forward', 'turn_right', 'forward'])
    return actions[:min(len(actions), 8)]


def load_emergency_data() -> Tuple[Dict, Dict]:
    """Load emergency data and organize by goal."""

    with open(EMERGENCY_DATA_PATH, 'r') as f:
        data = json.load(f)

    episodes = data['episodes']

    # Group by goal
    goal_info = {}
    for ep in episodes:
        goal = tuple(round(x, 2) for x in ep['goal_position'])

        if goal not in goal_info:
            goal_info[goal] = {
                'goal_position': ep['goal_position'],
                'scene_id': ep['scene_id'],
                'start_position': ep['start_position'],
                'original_instruction': ep['original_instruction'],
                'geodesic_distance': ep.get('geodesic_distance', 10),
                'episodes': [],
            }
        goal_info[goal]['episodes'].append(ep)

    return goal_info, episodes


def generate_paired_normal_samples(goal_info: Dict) -> List[Dict]:
    """Generate 127 paired normal samples from emergency goals."""

    samples = []

    for goal, info in goal_info.items():
        sample_text = format_normal_sample(
            goal_position=info['goal_position'],
            instruction=info['original_instruction'],
            actions=generate_actions(info['geodesic_distance'], info['original_instruction'])
        )

        samples.append({
            'text': sample_text,
            'type': 'normal',
            'source': 'paired',
            'goal': goal,
            'scene_id': info['scene_id'],
        })

    return samples


def load_r2r_available(emergency_goals: set) -> List[Dict]:
    """Load R2R episodes not overlapping with emergency goals."""

    with open(R2R_PATH, 'r') as f:
        r2r_data = json.load(f)

    episodes = r2r_data.get('episodes', r2r_data)

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
                    'goal': goal_pos,
                })

    return available


def generate_extra_normal_samples(r2r_available: List[Dict], count: int) -> List[Dict]:
    """Generate extra normal samples from R2R."""

    random.seed(SEED)
    selected = random.sample(r2r_available, min(count, len(r2r_available)))

    samples = []
    for ep_info in selected:
        sample_text = format_normal_sample(
            goal_position=ep_info['goal_position'],
            instruction=ep_info['instruction'],
            actions=generate_actions(ep_info['geodesic_distance'], ep_info['instruction'])
        )

        samples.append({
            'text': sample_text,
            'type': 'normal',
            'source': 'r2r_extra',
            'goal': ep_info['goal'],
            'scene_id': ep_info['scene_id'],
        })

    return samples


def prepare_emergency_samples(episodes: List) -> List[Dict]:
    """Prepare all emergency samples with metadata."""

    samples = []
    for ep in episodes:
        goal = tuple(round(x, 2) for x in ep['goal_position'])
        obs_type = ep['obstacle_config']['type']

        sample_text = format_emergency_sample(
            goal_position=ep['goal_position'],
            instruction=ep['emergency_instruction'],
            obstacle_type=obs_type,
            obstacle_position=ep['obstacle_config']['position'],
            trigger_step=ep['obstacle_config']['trigger_step'],
        )

        samples.append({
            'text': sample_text,
            'type': 'emergency',
            'obstacle_type': obs_type,
            'goal': goal,
            'scene_id': ep['scene_id'],
            'episode_id': ep['episode_id'],
        })

    return samples


def split_by_obstacle_type_groups(
    emergency_samples: List[Dict],
    paired_normal_samples: List[Dict],
    extra_normal_samples: List[Dict]
) -> Tuple[List, List, List]:
    """
    Split data by obstacle type groups (Strategy 3).

    Rule: Same goal + same obstacle_type must stay together.
    """

    random.seed(SEED)

    # Group emergency samples by (goal, obstacle_type)
    emergency_groups = defaultdict(list)
    for s in emergency_samples:
        key = (s['goal'], s['obstacle_type'])
        emergency_groups[key].append(s)

    # Shuffle group keys
    group_keys = list(emergency_groups.keys())
    random.shuffle(group_keys)

    # Split emergency groups
    n_groups = len(group_keys)
    n_train = int(n_groups * TRAIN_RATIO)
    n_val = int(n_groups * VAL_RATIO)

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    test_keys = set(group_keys[n_train + n_val:])

    # Assign emergency samples
    train_emergency = []
    val_emergency = []
    test_emergency = []

    for key, samples in emergency_groups.items():
        if key in train_keys:
            train_emergency.extend(samples)
        elif key in val_keys:
            val_emergency.extend(samples)
        else:
            test_emergency.extend(samples)

    # Get goals in each split
    train_goals = set(s['goal'] for s in train_emergency)
    val_goals = set(s['goal'] for s in val_emergency)
    test_goals = set(s['goal'] for s in test_emergency)

    # Assign paired normal samples (same split as their emergency goal)
    train_paired = []
    val_paired = []
    test_paired = []

    for s in paired_normal_samples:
        if s['goal'] in train_goals:
            train_paired.append(s)
        elif s['goal'] in val_goals:
            val_paired.append(s)
        elif s['goal'] in test_goals:
            test_paired.append(s)
        else:
            # Default to train
            train_paired.append(s)

    # Split extra normal samples randomly to balance
    random.shuffle(extra_normal_samples)

    # Calculate needed extra for each split
    train_normal_needed = 320 - len(train_paired)
    val_normal_needed = 40 - len(val_paired)
    test_normal_needed = 40 - len(test_paired)

    train_extra = extra_normal_samples[:train_normal_needed]
    val_extra = extra_normal_samples[train_normal_needed:train_normal_needed + val_normal_needed]
    test_extra = extra_normal_samples[train_normal_needed + val_normal_needed:train_normal_needed + val_normal_needed + test_normal_needed]

    # Combine
    train_samples = train_emergency + train_paired + train_extra
    val_samples = val_emergency + val_paired + val_extra
    test_samples = test_emergency + test_paired + test_extra

    return train_samples, val_samples, test_samples


def save_splits(train: List, val: List, test: List, stats: Dict):
    """Save splits to files."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save JSONL files
    for split_name, samples in [('train', train), ('val', val), ('test', test)]:
        filepath = os.path.join(OUTPUT_DIR, f'{split_name}.jsonl')
        with open(filepath, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps({'text': s['text']}, ensure_ascii=False) + '\n')
        print(f"Saved {split_name}: {len(samples)} samples")

    # Save stats
    stats_path = os.path.join(OUTPUT_DIR, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved stats: {stats_path}")


def main():
    print("=" * 70)
    print("Balanced Dataset Generation (Strategy 3)")
    print("=" * 70)

    # Load emergency data
    print("\n[1/6] Loading emergency data...")
    goal_info, episodes = load_emergency_data()
    print(f"  Found {len(goal_info)} unique goals")
    print(f"  Found {len(episodes)} emergency episodes")

    # Generate paired normal samples
    print("\n[2/6] Generating paired normal samples...")
    paired_normal = generate_paired_normal_samples(goal_info)
    print(f"  Generated {len(paired_normal)} paired normal samples")

    # Load R2R and generate extra normal samples
    print("\n[3/6] Loading R2R data...")
    emergency_goals = set(goal_info.keys())
    r2r_available = load_r2r_available(emergency_goals)
    print(f"  Found {len(r2r_available)} available R2R episodes")

    print("\n[4/6] Generating extra normal samples...")
    extra_normal = generate_extra_normal_samples(r2r_available, 273)
    print(f"  Generated {len(extra_normal)} extra normal samples")

    # Prepare emergency samples
    print("\n[5/6] Preparing emergency samples...")
    emergency_samples = prepare_emergency_samples(episodes)
    print(f"  Prepared {len(emergency_samples)} emergency samples")

    # Split by strategy 3
    print("\n[6/6] Splitting by obstacle type groups (Strategy 3)...")
    train, val, test = split_by_obstacle_type_groups(
        emergency_samples,
        paired_normal,
        extra_normal
    )

    # Calculate stats
    stats = {
        'total_samples': len(train) + len(val) + len(test),
        'train': {
            'total': len(train),
            'emergency': sum(1 for s in train if s['type'] == 'emergency'),
            'normal': sum(1 for s in train if s['type'] == 'normal'),
            'paired_normal': sum(1 for s in train if s.get('source') == 'paired'),
            'extra_normal': sum(1 for s in train if s.get('source') == 'r2r_extra'),
        },
        'val': {
            'total': len(val),
            'emergency': sum(1 for s in val if s['type'] == 'emergency'),
            'normal': sum(1 for s in val if s['type'] == 'normal'),
            'paired_normal': sum(1 for s in val if s.get('source') == 'paired'),
            'extra_normal': sum(1 for s in val if s.get('source') == 'r2r_extra'),
        },
        'test': {
            'total': len(test),
            'emergency': sum(1 for s in test if s['type'] == 'emergency'),
            'normal': sum(1 for s in test if s['type'] == 'normal'),
            'paired_normal': sum(1 for s in test if s.get('source') == 'paired'),
            'extra_normal': sum(1 for s in test if s.get('source') == 'r2r_extra'),
        },
        'strategy': 'Strategy 3: Split by obstacle type groups',
        'output_dir': OUTPUT_DIR,
    }

    # Save
    print("\n" + "=" * 70)
    print("Saving splits...")
    print("=" * 70)
    save_splits(train, val, test, stats)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nTrain: {stats['train']['total']} ({stats['train']['emergency']} emergency + {stats['train']['normal']} normal)")
    print(f"  - Paired normal: {stats['train']['paired_normal']}")
    print(f"  - Extra normal: {stats['train']['extra_normal']}")

    print(f"\nVal: {stats['val']['total']} ({stats['val']['emergency']} emergency + {stats['val']['normal']} normal)")
    print(f"  - Paired normal: {stats['val']['paired_normal']}")
    print(f"  - Extra normal: {stats['val']['extra_normal']}")

    print(f"\nTest: {stats['test']['total']} ({stats['test']['emergency']} emergency + {stats['test']['normal']} normal)")
    print(f"  - Paired normal: {stats['test']['paired_normal']}")
    print(f"  - Extra normal: {stats['test']['extra_normal']}")

    print(f"\nTotal: {stats['total_samples']} samples")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()