# VLM Fallback Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fallback mechanism to ObservationAgent - when subtask objects not detected, VLM falls back to describing instruction-related objects.

**Architecture:** Modify ObservationOutput dataclass with new fields (subtask_relevant, instruction_relevant, fallback_mode, exploration_hint), update prompt to include original instruction, modify Navigator to pass instruction.

**Tech Stack:** Python dataclasses, VLM (Qwen3.6-35B), Pipeline Agent Architecture

---

## File Structure

| File | Responsibility | Changes |
|------|----------------|---------|
| `agents/pipeline/base_pipeline_agent.py` | ObservationOutput dataclass | Add 4 new fields |
| `agents/pipeline/observation_agent.py` | VLM prompt + parsing | New prompt with instruction, updated defaults |
| `agents/pipeline/navigator.py` | Orchestrator | Save `_instruction`, pass to ObservationAgent |
| `agents/pipeline/analysis_agent.py` | LLM reasoning | Use new fields in prompt |
| `agents/pipeline/review_agent.py` | Completion verification | Use subtask_relevant field |

---

### Task 1: Modify ObservationOutput dataclass

**Files:**
- Modify: `agents/pipeline/base_pipeline_agent.py:16-29`

- [ ] **Step 1: Update ObservationOutput dataclass**

Replace the existing ObservationOutput (lines 16-29) with:

```python
@dataclass
class ObservationOutput:
    """Observation Agent output - visual scene analysis result.

    Contains information about what the agent observes in the environment,
    including subtask-relevant objects, instruction-related objects, and navigation cues.
    """

    subtask_relevant: bool  # Whether subtask target objects are visible (renamed from task_relevant)
    instruction_relevant: bool  # Whether original instruction objects are visible (fallback)
    fallback_mode: bool  # Whether currently in fallback mode
    objects: List[Dict[str, Any]]  # List of objects with features: {name, direction, location, distance, features}
    exploration_hint: str  # Exploration direction suggestion (e.g., "turn_left to find stairs")
    target_direction: str  # Target direction (left/right/forward/backward)
    target_distance: str  # Target distance (close/medium/far/unknown)
    path_blocked: bool  # Whether forward path is blocked
    navigation_cues: List[str]  # Navigation cues for decision making
    scene_description: str = ""  # Scene description text

    # Compatibility alias for legacy code
    @property
    def task_relevant(self) -> bool:
        """Alias for backward compatibility."""
        return self.subtask_relevant
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/base_pipeline_agent.py
git commit -m "feat: add fallback fields to ObservationOutput dataclass"
```

---

### Task 2: Update ObservationAgent defaults and parsing

**Files:**
- Modify: `agents/pipeline/observation_agent.py:245-254`

- [ ] **Step 1: Update defaults dict in _parse_response**

Replace the defaults dict (lines 245-254) with:

```python
# 默认值 - 包含新字段
defaults = {
    "subtask_relevant": False,  # 新字段名（替代原task_relevant）
    "instruction_relevant": False,  # 新增
    "fallback_mode": False,  # 新增
    "objects": [],  # List[Dict[str, Any]]
    "exploration_hint": "",  # 新增
    "target_direction": "unknown",
    "target_distance": "unknown",
    "path_blocked": False,
    "navigation_cues": [],
    "scene_description": "",
}

# Compatibility: map task_relevant → subtask_relevant if VLM uses old field name
if "task_relevant" in data and "subtask_relevant" not in data:
    result["subtask_relevant"] = data["task_relevant"]
```

- [ ] **Step 2: Add compatibility mapping**

Insert after line 324 (after `result[key] = data[key]`), before the return statement:

```python
# Compatibility: map task_relevant → subtask_relevant
if "task_relevant" in data and "subtask_relevant" not in data:
    result["subtask_relevant"] = data["task_relevant"]
```

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/observation_agent.py
git commit -m "feat: add fallback fields to ObservationAgent parsing"
```

---

### Task 3: Update ObservationAgent prompt

**Files:**
- Modify: `agents/pipeline/observation_agent.py:126-230`

- [ ] **Step 1: Modify process() to accept instruction parameter**

Update the process() method signature (lines 49-56) and add instruction extraction:

```python
def process(
    self,
    subtask,
    position: List[float],
    rotation: float,
    rgb_image: np.ndarray,
    depth_image: np.ndarray,
    instruction: str = "",  # NEW: original instruction for fallback
) -> ObservationOutput:
```

Update line 77 to pass instruction:

```python
# 构建任务导向prompt
prompt = self._build_prompt(subtask, position, rotation, instruction)  # NEW: pass instruction
```

- [ ] **Step 2: Update _build_prompt signature and content**

Replace entire `_build_prompt` method (lines 126-231) with:

```python
def _build_prompt(
    self,
    subtask,
    position: List[float],
    rotation: float,
    instruction: str = "",  # NEW: original instruction
) -> str:
    """构建任务导向prompt with fallback机制.

    Args:
        subtask: 当前子任务
        position: 当前位置
        rotation: 当前朝向
        instruction: 原始导航指令（用于fallback）

    Returns:
        任务导向的prompt字符串
    """
    subtask_desc = getattr(subtask, 'description', str(subtask))
    completion_cond = getattr(subtask, 'completion_condition', {})

    # 解析完成条件
    cond_type = completion_cond.get("type", "unknown")
    cond_direction = completion_cond.get("direction", "")
    cond_target = completion_cond.get("target", "")
    min_change = completion_cond.get("min_change", "")

    # 获取 relevant_objects
    relevant_objects = getattr(subtask, 'relevant_objects', [])
    if not relevant_objects:
        target = completion_cond.get("target", "")
        if target:
            relevant_objects = [target]

    target_list = ", ".join(relevant_objects) if relevant_objects else "none specified"

    prompt = f"""You are a navigation robot analyzing images.

## Images
- First: RGB image (visual scene)
- Second: Depth image (distance visualization):
  * RED = CLOSE (near you)
  * BLUE = FAR (far from you)
  * Gradient: red → yellow → green → cyan → blue (near → far)

## Navigation Context
### Current Subtask
- Description: {subtask_desc}
- Target Objects: {target_list}

### Original Instruction
{instruction}

## Detection Priority
1. Priority: Subtask target objects → if visible, subtask_relevant=true
2. Fallback: Objects from original instruction → if subtask objects NOT visible, check instruction-related objects
3. Last fallback: Report blocking situation + suggest exploration direction

## Output Format (JSON only)
{{
  "subtask_relevant": true/false,
  "instruction_relevant": true/false,
  "fallback_mode": true/false,
  "objects": [
    {{
      "name": "object_name",
      "direction": "forward_left/forward_right/left/right/forward/backward",
      "location": "center/left_side/right_side/foreground/background",
      "distance": "close/medium/far",
      "features": "specific characteristics"
    }}
  ],
  "exploration_hint": "turn_left to find stairs" or "" if target visible,
  "target_direction": "left/right/forward/backward/unknown",
  "target_distance": "close/medium/far/unknown",
  "path_blocked": true/false,
  "navigation_cues": ["cue1", "cue2"],
  "scene_description": "brief description"
}}

## Rules
- If subtask objects visible: subtask_relevant=true, fallback_mode=false, fill objects with subtask targets
- If subtask objects NOT visible but instruction objects visible: subtask_relevant=false, instruction_relevant=true, fallback_mode=true, fill objects with instruction-related items
- If neither visible: subtask_relevant=false, instruction_relevant=false, fallback_mode=true, objects=[], provide exploration_hint

JSON output only:"""

    return prompt
```

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/observation_agent.py
git commit -m "feat: update ObservationAgent prompt with instruction fallback"
```

---

### Task 4: Update Navigator to save and pass instruction

**Files:**
- Modify: `agents/pipeline/navigator.py:85-92` (add `_instruction` field)
- Modify: `agents/pipeline/navigator.py:187-225` (save instruction in initialize_episode)
- Modify: `agents/pipeline/navigator.py:318-325` (pass instruction to ObservationAgent)

- [ ] **Step 1: Add `_instruction` field in Navigator.__init__**

Add after line 92 (after `_step_count = 0`):

```python
self._instruction = ""  # NEW: save original instruction for fallback
```

- [ ] **Step 2: Save instruction in initialize_episode**

Add after line 202 (after `self._step_count = 0`):

```python
self._instruction = instruction  # NEW: save original instruction
```

- [ ] **Step 3: Pass instruction to ObservationAgent**

Modify the ObservationAgent call (lines 318-325):

```python
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

- [ ] **Step 4: Update fallback ObservationOutput creation**

Modify lines 330-338:

```python
observation_output = ObservationOutput(
    subtask_relevant=False,  # renamed field
    instruction_relevant=False,  # new field
    fallback_mode=True,  # new field
    objects=[],
    exploration_hint="",  # new field
    target_direction="forward",
    target_distance="unknown",
    path_blocked=False,
    navigation_cues=[],
    scene_description="",
)
```

- [ ] **Step 5: Commit**

```bash
git add agents/pipeline/navigator.py
git commit -m "feat: Navigator saves and passes instruction for fallback"
```

---

### Task 5: Update AnalysisAgent to use new fields

**Files:**
- Modify: `agents/pipeline/analysis_agent.py` (multiple locations)

- [ ] **Step 1: Update _build_cot_prompt to include new fields**

Find the `_build_cot_prompt` method and modify the observation section. Add after line 200 (in the prompt building):

```python
# Add fallback note if in fallback mode
fallback_note = ""
if hasattr(observation, 'fallback_mode') and observation.fallback_mode:
    fallback_note = f"""
NOTE: Subtask objects NOT visible. Currently in fallback mode.
- Focus on instruction-related objects or use exploration_hint
- exploration_hint: {observation.exploration_hint}
"""

# Update observation section in prompt
```

Update the prompt's observation section to include new fields:

```python
## Current Observation
- Subtask relevant: {observation.subtask_relevant if hasattr(observation, 'subtask_relevant') else observation.task_relevant}
- Instruction relevant: {observation.instruction_relevant if hasattr(observation, 'instruction_relevant') else False}
- Fallback mode: {observation.fallback_mode if hasattr(observation, 'fallback_mode') else False}
- Detected objects (with details): {objects_text}
- Exploration hint: {observation.exploration_hint if hasattr(observation, 'exploration_hint') else ''}
{fallback_note}
- Target direction: {observation.target_direction}
- Target distance: {observation.target_distance}
- Path blocked: {observation.path_blocked}
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/analysis_agent.py
git commit -m "feat: AnalysisAgent uses fallback fields in prompt"
```

---

### Task 6: Update ReviewAgent compatibility

**Files:**
- Modify: `agents/pipeline/review_agent.py:257-279`

- [ ] **Step 1: Update _verify_near_object to use subtask_relevant**

Modify line 259 (the task_relevant usage):

```python
task_relevant = observation.subtask_relevant if hasattr(observation, 'subtask_relevant') else observation.task_relevant
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/review_agent.py
git commit -m "feat: ReviewAgent uses subtask_relevant field"
```

---

### Task 7: Integration test

**Files:**
- Test: Run experiment to verify fallback works

- [ ] **Step 1: Run short experiment**

Run: `python run_vln_experiment.py --use-remote-llm --llm-server http://localhost:8000 --episodes 1 --max-steps 10 --use-pipeline`

Expected: VLM should fallback to instruction objects when subtask objects not visible

- [ ] **Step 2: Check logs for fallback behavior**

Run: `grep "fallback_mode\|instruction_relevant" results/episode-*/episode1/agent_outputs.json`

Expected: Should see `fallback_mode: true` when subtask objects not detected

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: VLM fallback perception mechanism complete"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ Data structure changes → Task 1
- ✅ Prompt structure → Task 3
- ✅ Parsing logic → Task 2
- ✅ Navigator instruction passing → Task 4
- ✅ AnalysisAgent usage → Task 5
- ✅ ReviewAgent compatibility → Task 6
- ✅ Integration test → Task 7

**2. Placeholder scan:**
- No TBD, TODO, or vague descriptions
- All code blocks contain complete implementation

**3. Type consistency:**
- `subtask_relevant` used consistently across all tasks
- Compatibility alias `task_relevant` property provided
- All new fields have consistent default values