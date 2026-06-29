# Stair Direction Extraction Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify RGB+Depth VLM prompt to force output of stair_direction JSON field

**Architecture:** Single prompt text modification in perception_agent.py - change conditional phrasing to mandatory field specification

**Tech Stack:** Python, VLM prompt engineering

---

## File Structure

| File | Responsibility | Change Type |
|------|---------------|-------------|
| `agents/perception_agent.py` | VLM perception prompt | Modify lines 285, 299-302 |

---

### Task 1: Modify RGB+Depth Prompt for Mandatory stair_direction

**Files:**
- Modify: `agents/perception_agent.py:285,299-302`

- [ ] **Step 1: Modify Required Output header (line 285)**

Change from:
```python
## Required Output (JSON only)
```

To:
```python
## Required Output (ALL fields mandatory, no fields can be omitted)
```

- [ ] **Step 2: Modify stair_direction guideline (lines 299-302)**

Change from:
```python
4. stair_direction: ONLY when stairs are visible:
   - "up" if stairs ascend away from you
   - "down" if stairs descend away from you
   - "none" if no stairs visible
```

To:
```python
4. stair_direction: MANDATORY field - always include in JSON:
   - "up" if stairs ascend away from viewer
   - "down" if stairs descend away from viewer
   - "none" if no stairs in scene (output this value, do not omit the field)
```

- [ ] **Step 3: Verify the complete modified prompt section**

The modified prompt section (lines 285-304) should look like:

```python
## Required Output (ALL fields mandatory, no fields can be omitted)
{{"room_type":"hallway/bedroom/living_room/kitchen/bathroom/stairs",
"scene_description":"detailed description in 30-50 words",
"objects":[{{"name":"object name","distance":1.0}}],
"stair_direction":"up/down/none"}

## Guidelines
1. room_type: Use simple names (hallway, bedroom, living_room, kitchen, bathroom, stairs)
2. scene_description: Detailed description including:
   - Room layout and size
   - Visible furniture and objects
   - Open paths and doors
   - Stairs if visible (direction)
3. objects: List 3-5 visible objects with distance in meters
4. stair_direction: MANDATORY field - always include in JSON:
   - "up" if stairs ascend away from viewer
   - "down" if stairs descend away from viewer
   - "none" if no stairs in scene (output this value, do not omit the field)

Output JSON only, no explanation:
```

- [ ] **Step 4: Commit the changes**

```bash
git add agents/perception_agent.py
git commit -m "fix: make stair_direction a mandatory field in VLM prompt

- Change 'ONLY when stairs are visible' to 'MANDATORY field'
- Emphasize 'always include in JSON'
- Clarify 'none' must be output even when no stairs
- Fixes VLM not outputting stair_direction JSON field"
```

---

### Task 2: Verify Changes with Manual Test

**Files:**
- Test: Run single step experiment on stairs scene

- [ ] **Step 1: Start vLLM server (if not running)**

```bash
bash scripts/run_vln_experiment_with_vllm.sh start_vllm
```

- [ ] **Step 2: Run single step experiment on episode 1**

```bash
python run_vln_experiment.py \
    --use-remote-llm \
    --llm-server http://localhost:8000 \
    --episodes 1 \
    --max-steps 5 \
    --use-sequence-mode \
    --output-dir results/stair-test
```

- [ ] **Step 3: Check perception_output contains stair_direction**

```bash
python3 -c "
import json
with open('results/stair-test/episode*/episode1/agent_outputs.json') as f:
    data = json.load(f)
    # Find first stairs detection
    for key in data:
        if 'perception_output' in str(key):
            po = data[key] if isinstance(data[key], dict) else {}
            if po.get('room_type') == 'stairs':
                print('Found stairs perception:')
                print('stair_direction:', po.get('stair_direction', 'MISSING'))
                break
"
```

Expected output: `stair_direction: up` or `stair_direction: down` (not 'MISSING')

---

## Self-Review

**1. Spec coverage:** ✅
- Spec requirement: Modify prompt to make stair_direction mandatory
- Task 1 covers this exactly

**2. Placeholder scan:** ✅
- No TBD, TODO, or vague descriptions
- All code changes shown explicitly

**3. Type consistency:** ✅
- Single file change, no type dependencies across tasks