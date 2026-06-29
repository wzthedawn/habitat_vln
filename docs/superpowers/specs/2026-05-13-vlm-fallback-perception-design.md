# VLM Fallback Perception Design

## Problem Statement

When ObservationAgent fails to detect subtask-related objects (e.g., stairs not visible due to wrong initial orientation), VLM outputs become unhelpful ("white object blocking view"). The agent loses navigation context and repeats similar actions without progress.

## Solution Overview

Add a fallback mechanism: when subtask objects are not detected, VLM falls back to describing objects related to the original instruction, providing exploration hints for navigation continuity.

---

## Design Details

### 1. Data Structure Changes

**ObservationOutput new fields**:

```python
@dataclass
class ObservationOutput:
    subtask_relevant: bool        # Subtask objects visible (renamed from task_relevant)
    instruction_relevant: bool    # NEW: Original instruction objects visible
    fallback_mode: bool           # NEW: Currently in fallback mode
    objects: List[Dict[str, Any]] # Object list with features (unchanged)
    exploration_hint: str         # NEW: Exploration direction suggestion
    target_direction: str         # (unchanged)
    target_distance: str          # (unchanged)
    path_blocked: bool            # (unchanged)
    navigation_cues: List[str]    # (unchanged)
    scene_description: str        # (unchanged)
```

**Objects format** (unchanged):
```python
objects: [
    {
        "name": "stairs",
        "direction": "forward_left",
        "location": "left_side",
        "distance": "medium",
        "features": "stairs going down, metal railings"
    }
]
```

---

### 2. Prompt Structure

**Template**:

```
You are a navigation robot analyzing images.

## Images
- First: RGB image (visual scene)
- Second: Depth image (distance visualization):
  * RED = CLOSE (near you)
  * BLUE = FAR (far from you)
  * Gradient: red → yellow → green → cyan → blue (near → far)

## Navigation Context
### Current Subtask
- Description: {subtask_description}
- Target Objects: {relevant_objects_list}

### Original Instruction
{full_instruction}

## Detection Priority
1. Priority: Subtask target objects → if visible, subtask_relevant=true
2. Fallback: Objects from original instruction → if subtask objects NOT visible, check instruction-related objects
3. Last fallback: Report blocking situation + suggest exploration direction

## Output Format (JSON only)
{
  "subtask_relevant": true/false,
  "instruction_relevant": true/false,
  "fallback_mode": true/false,
  "objects": [{"name": "...", "direction": "...", "location": "...", "distance": "...", "features": "..."}],
  "exploration_hint": "e.g., 'turn_left to find stairs' or empty if target visible",
  "target_direction": "left/right/forward/backward/unknown",
  "target_distance": "close/medium/far/unknown",
  "path_blocked": true/false,
  "navigation_cues": ["..."],
  "scene_description": "..."
}

## Rules
- If subtask objects visible: subtask_relevant=true, fallback_mode=false, fill objects with subtask targets
- If subtask objects NOT visible but instruction objects visible: subtask_relevant=false, instruction_relevant=true, fallback_mode=true, fill objects with instruction-related items
- If neither visible: subtask_relevant=false, instruction_relevant=false, fallback_mode=true, objects=[], provide exploration_hint

JSON output only:
```

---

### 3. Data Passing Adjustment

**Navigator needs to pass instruction**:

```python
# navigator.py - save instruction in initialize_episode()
def initialize_episode(self, instruction: str, ...):
    self._instruction = instruction  # NEW: save original instruction

# navigator.py - pass to ObservationAgent
observation_output = self._registry.call(
    "observation",
    subtask=subtask_obj,
    instruction=self._instruction,  # NEW: pass original instruction
    position=self._position,
    rotation=self._rotation,
    rgb_image=rgb,
    depth_image=depth,
)
```

---

### 4. Parsing Logic Adjustment

**ObservationAgent._parse_response() handles new fields**:

```python
defaults = {
    "subtask_relevant": False,
    "instruction_relevant": False,
    "fallback_mode": False,
    "objects": [],
    "exploration_hint": "",
    "target_direction": "unknown",
    "target_distance": "unknown",
    "path_blocked": False,
    "navigation_cues": [],
    "scene_description": "",
}

# Compatibility: map task_relevant → subtask_relevant
if "task_relevant" in data and "subtask_relevant" not in data:
    result["subtask_relevant"] = data["task_relevant"]
```

---

### 5. AnalysisAgent Usage

**Adjust prompt based on fallback_mode**:

```python
def _build_cot_prompt(...):
    if observation.fallback_mode:
        fallback_note = """
NOTE: Subtask objects NOT visible. Currently in fallback mode.
- Focus on instruction-related objects or use exploration_hint
- exploration_hint: {observation.exploration_hint}
"""
    else:
        fallback_note = ""
    
    prompt = f"""
...
## Current Observation
- Subtask relevant: {observation.subtask_relevant}
- Instruction relevant: {observation.instruction_relevant}
- Fallback mode: {observation.fallback_mode}
- Objects: {objects_text}
- Exploration hint: {observation.exploration_hint}
{fallback_note}
...
"""
```

**Decision logic**:
```
if subtask_relevant=true → move toward target
elif instruction_relevant=true → move toward instruction-related objects (intermediate goal)
elif fallback_mode=true → use exploration_hint for turning/exploration
else → default forward or turn
```

---

### 6. ReviewAgent Compatibility

**Use subtask_relevant instead of task_relevant**:

```python
def _verify_near_object(...):
    task_relevant = observation.subtask_relevant  # renamed field
    ...
```

**Field mapping**:

| Old Field | New Field | Note |
|-----------|-----------|------|
| task_relevant | subtask_relevant | Primary detection field |
| - | instruction_relevant | NEW |
| - | fallback_mode | NEW |
| - | exploration_hint | NEW |
| objects | objects | Unchanged (List[Dict]) |

---

## Files to Modify

1. `agents/pipeline/base_pipeline_agent.py` - ObservationOutput dataclass
2. `agents/pipeline/observation_agent.py` - Prompt + parsing logic
3. `agents/pipeline/navigator.py` - Save + pass instruction
4. `agents/pipeline/analysis_agent.py` - Use new fields in prompt
5. `agents/pipeline/review_agent.py` - Use subtask_relevant field

---

## Expected Outcome

- When subtask objects not visible, VLM falls back to instruction-related objects
- Agent maintains navigation context even when current subtask target not found
- exploration_hint provides explicit direction suggestion for search behavior