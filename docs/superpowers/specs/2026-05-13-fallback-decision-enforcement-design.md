# Fallback Decision Enforcement Design

## Problem Statement

When ObservationAgent fails to detect subtask-related objects (e.g., stairs not visible), it correctly outputs `fallback_mode=true` with `exploration_hint="turn_left to find stairs"`. However, AnalysisAgent and PlanningAgent ignore this information and continue generating `forward` actions, causing the agent to walk blindly without turning to explore.

**Root cause:** The fallback hint exists as descriptive text in the prompt, but there's no explicit decision rule telling LLMs what to do in fallback mode.

## Solution Overview

Add explicit decision rules in both AnalysisAgent and PlanningAgent prompts, and pass observation information to PlanningAgent so it can understand fallback state.

---

## Design Details

### 1. AnalysisAgent Prompt Enhancement

**Current state:**
- `fallback_note` is a descriptive text block in the middle of observation section
- `## Rules` section has no fallback-specific rules

**Changes:**

1. **Add Fallback Decision Rule in Rules section:**
```python
## Rules
- recommended_action must be one of: forward, turn_left, turn_right
- confidence must be between 0.0 and 1.0

## CRITICAL: Fallback Mode Decision Rule
If fallback_mode=true:
  - Target objects are NOT visible in current view
  - MUST follow exploration_hint direction if provided
  - Example: hint="turn_left to find stairs" -> recommended_action="turn_left"
  - DO NOT output "forward" when target is not visible and you need to explore
```

2. **Strengthen fallback_note visibility:**
- Keep existing fallback_note structure
- Add explicit instruction: "When in fallback mode, your recommended_action should match the exploration direction."

### 2. PlanningAgent Enhancement

**Current state:**
- `process()` only receives `AnalysisOutput`, not `ObservationOutput`
- Prompt has no fallback-related information

**Changes:**

1. **Add observation parameter:**
```python
def process(
    self,
    analysis: AnalysisOutput,
    topology: TopologyGraph,
    position: List[float],
    goal: Optional[List[float]] = None,
    observation: Optional[ObservationOutput] = None,  # NEW
) -> PlanningOutput:
```

2. **Add fallback planning note in prompt:**
```python
## Planning Context
{fallback_planning_note}

## CRITICAL: Fallback Mode Planning Rule
If agent is in fallback mode (cannot see target objects):
  - exploration_hint indicates where to search
  - Generate actions that explore in that direction
  - Example: hint="turn_left to find stairs"
    -> actions should include turn_left followed by forward steps
  - DO NOT generate all "forward" actions blindly
  - Preferred pattern for search: ["turn_left/right", "forward", "forward", "turn_left/right", "forward"]
```

3. **Build fallback planning note:**
```python
fallback_planning_note = ""
if observation and hasattr(observation, 'fallback_mode') and observation.fallback_mode:
    fallback_planning_note = f"""
FALLBACK MODE ACTIVE:
- Cannot see target objects
- Exploration hint: {observation.exploration_hint}
- Recommended action from analysis: {analysis.recommended_action}
"""
```

### 3. Navigator Integration

**Current state:**
- Navigator calls PlanningAgent without observation parameter

**Change:**
```python
# navigator.py step() method
planning_output = self._registry.call(
    "planning",
    analysis=analysis_output,
    topology=self._topology,
    position=self._position,
    observation=observation_output,  # NEW: pass observation
)
```

---

## Data Flow After Changes

```
Navigator.step()
    ↓
ObservationAgent → ObservationOutput(fallback_mode=True, exploration_hint="turn_left to find stairs")
    ↓
AnalysisAgent.process(observation)
    ↓ Prompt contains CRITICAL fallback decision rule
    → AnalysisOutput(recommended_action="turn_left")
    ↓
PlanningAgent.process(analysis, observation)  ← NEW: receives observation
    ↓ Prompt contains CRITICAL fallback planning rule
    → PlanningOutput(actions=["turn_left", "forward", "forward", "turn_left", "forward"])
    ↓
Navigator executes actions (agent turns to explore)
```

---

## Expected Behavior

| Scenario | Before | After |
|----------|--------|-------|
| fallback_mode=true, hint="turn_left" | AnalysisAgent outputs "forward", PlanningAgent generates 5 "forward" | AnalysisAgent outputs "turn_left", PlanningAgent generates exploration pattern |
| fallback_mode=false, stairs visible | Normal navigation | Normal navigation (unchanged) |

---

## Files to Modify

1. `agents/pipeline/analysis_agent.py` - Add fallback decision rule in prompt
2. `agents/pipeline/planning_agent.py` - Add observation parameter and fallback planning rule
3. `agents/pipeline/navigator.py` - Pass observation to PlanningAgent call

---

## Verification

1. Run experiment with episode where initial view doesn't show stairs
2. Check agent_outputs.json for:
   - `analysis_output.recommended_action` should be "turn_left" (not "forward") when fallback_mode=true
   - `planning_output.actions` should include turn_left actions
3. Verify agent trajectory includes turning behavior