# Fallback Decision Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit fallback decision rules in AnalysisAgent and PlanningAgent prompts, and pass observation to PlanningAgent so LLMs prioritize exploration_hint when in fallback mode.

**Architecture:** Modify prompt content in AnalysisAgent._build_cot_prompt() and PlanningAgent._build_llm_prompt(), add observation parameter to PlanningAgent.process(), and update Navigator to pass observation.

**Tech Stack:** Python, LLM prompt engineering, Pipeline Agent Architecture

---

## File Structure

| File | Responsibility | Changes |
|------|----------------|---------|
| `agents/pipeline/analysis_agent.py` | LLM reasoning | Add fallback decision rule in Rules section |
| `agents/pipeline/planning_agent.py` | Action planning | Add observation parameter, add fallback planning rule |
| `agents/pipeline/navigator.py` | Agent coordination | Pass observation to PlanningAgent call |

---

### Task 1: Add Fallback Decision Rule in AnalysisAgent

**Files:**
- Modify: `agents/pipeline/analysis_agent.py:247-251` (Rules section)

- [ ] **Step 1: Modify _build_cot_prompt() Rules section**

Replace lines 247-251 in `_build_cot_prompt()` with enhanced Rules section:

```python
## Rules
- recommended_action must be one of: forward, turn_left, turn_right
- confidence must be between 0.0 and 1.0
- Pay attention to object features (direction, distance) when deciding action
- Output ONLY the JSON, no additional text

## CRITICAL: Fallback Mode Decision Rule
If fallback_mode=true and objects=none:
  - Target objects are NOT visible in current view
  - MUST follow exploration_hint direction if provided
  - Example: hint="turn_left to find stairs" -> recommended_action="turn_left"
  - DO NOT output "forward" when target is not visible and you need to explore

When fallback_mode=true, your recommended_action MUST align with exploration_hint direction.
"""
```

- [ ] **Step 2: Verify prompt structure**

Run: `grep -A10 "CRITICAL: Fallback" agents/pipeline/analysis_agent.py`
Expected: Should see the fallback decision rule added after Rules section

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/analysis_agent.py
git commit -m "feat: add fallback decision rule in AnalysisAgent prompt"
```

---

### Task 2: Add observation parameter to PlanningAgent.process()

**Files:**
- Modify: `agents/pipeline/planning_agent.py:60-94` (process method)

- [ ] **Step 1: Update process() signature**

Replace lines 60-66 with updated signature including observation parameter:

```python
def process(
    self,
    analysis: AnalysisOutput,
    topology: TopologyGraph,
    position: List[float],
    goal: Optional[List[float]] = None,
    observation: Optional[ObservationOutput] = None,  # NEW
) -> PlanningOutput:
    """Path planning.

    Args:
        analysis: AnalysisAgent output with recommendations
        topology: Topology graph with key nodes
        position: Current position [x, y, z]
        goal: Optional goal position [x, y, z]
        observation: Optional ObservationOutput for fallback context (NEW)

    Returns:
        PlanningOutput with action sequence and expected result
    """
```

- [ ] **Step 2: Add ObservationOutput import**

Add at line 16-20, update the imports:

```python
from agents.pipeline.base_pipeline_agent import (
    AnalysisOutput,
    ObservationOutput,  # NEW
    PlanningOutput,
    SubAgent,
)
```

- [ ] **Step 3: Pass observation to _llm_planning**

Modify lines 84-89 to pass observation:

```python
# Execute corresponding algorithm
if algorithm == "topology":
    result = self._topology_planning(topology, position, analysis)
elif algorithm == "astar":
    result = self._astar_planning(position, goal)
else:
    result = self._llm_planning(analysis, position, goal, observation)  # NEW: pass observation
```

- [ ] **Step 4: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: add observation parameter to PlanningAgent.process()"
```

---

### Task 3: Add fallback planning rule in PlanningAgent prompt

**Files:**
- Modify: `agents/pipeline/planning_agent.py:122-143` (_llm_planning method)
- Modify: `agents/pipeline/planning_agent.py:145-195` (_build_llm_prompt method)

- [ ] **Step 1: Update _llm_planning signature**

Replace lines 122-127 with updated signature:

```python
def _llm_planning(
    self,
    analysis: AnalysisOutput,
    position: List[float],
    goal: Optional[List[float]],
    observation: Optional[ObservationOutput] = None,  # NEW
) -> dict:
    """LLM semantic planning.

    Uses LLM to generate action sequence based on analysis recommendations.

    Args:
        analysis: Analysis output with recommendations
        position: Current position
        goal: Goal position (optional)
        observation: Observation output for fallback context (NEW)

    Returns:
        Planning result dictionary
    """
    prompt = self._build_llm_prompt(analysis, position, goal, observation)  # NEW: pass observation
    response = self._call_llm(prompt, max_tokens=400, temperature=0.3)

    return self._parse_response(response)
```

- [ ] **Step 2: Update _build_llm_prompt signature and content**

Replace lines 145-195 with updated prompt including fallback planning rule:

```python
def _build_llm_prompt(
    self,
    analysis: AnalysisOutput,
    position: List[float],
    goal: Optional[List[float]],
    observation: Optional[ObservationOutput] = None,  # NEW
) -> str:
    """Build LLM planning prompt.

    Args:
        analysis: Analysis output
        position: Current position
        goal: Goal position (optional)
        observation: Observation output for fallback context (NEW)

    Returns:
        Prompt string for LLM
    """
    goal_info = f"Goal position: {goal}" if goal else "Goal position: unknown (semantic goal)"

    # Build fallback planning note
    fallback_planning_note = ""
    if observation and hasattr(observation, 'fallback_mode') and observation.fallback_mode:
        hint = observation.exploration_hint if hasattr(observation, 'exploration_hint') else ""
        fallback_planning_note = f"""
## FALLBACK MODE ACTIVE (CRITICAL)
- Cannot see target objects in current view
- Exploration hint: {hint}
- Recommended action from analysis: {analysis.recommended_action}
- You MUST generate actions that explore in the suggested direction
"""

    prompt = f"""You are a navigation planning expert. Generate an action sequence to navigate toward the goal.
{fallback_planning_note}
## Navigation Goal
- Goal summary: {analysis.goal_summary}
- Current gap: {analysis.current_gap}
- Recommended action: {analysis.recommended_action}
- Reasoning: {analysis.reasoning}
- Confidence: {analysis.confidence}
{goal_info}

## Current Position
Position: {position}

## Available Actions
- forward: Move forward one step
- turn_left: Turn left (about 90 degrees)
- turn_right: Turn right (about 90 degrees)

## Task
Generate exactly 5 actions that will help achieve the navigation goal.
Consider the recommended action and reasoning from the analysis.

## Output Format (JSON)
Output only valid JSON:
{{"actions": ["action1", "action2", "action3", "action4", "action5"], "expected_result": "description of expected result after executing actions"}}

## Rules
- actions must contain exactly 5 items
- Each action must be one of: forward, turn_left, turn_right
- expected_result should describe what happens after executing the actions
- Output ONLY the JSON, no additional text

## CRITICAL: Fallback Mode Planning Rule
If in fallback mode (cannot see target objects):
  - exploration_hint indicates where to search for targets
  - Generate actions that explore in that direction
  - Example: hint="turn_left to find stairs" -> include turn_left in actions
  - DO NOT generate all "forward" actions blindly when target not visible
  - Preferred search pattern: ["turn_left/right", "forward", "forward", "turn_left/right", "forward"]
"""

    return prompt
```

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: add fallback planning rule in PlanningAgent prompt"
```

---

### Task 4: Pass observation to PlanningAgent in Navigator

**Files:**
- Modify: `agents/pipeline/navigator.py:391-397` (PlanningAgent call)

- [ ] **Step 1: Update PlanningAgent call to include observation**

Replace lines 392-397 with updated call:

```python
planning_output = self._registry.call(
    "planning",
    analysis=analysis_output,
    topology=self._topology,
    position=self._position,
    observation=observation_output,  # NEW: pass observation for fallback context
)
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/navigator.py
git commit -m "feat: pass observation to PlanningAgent for fallback context"
```

---

### Task 5: Integration test

**Files:**
- Test: Run experiment to verify fallback decision enforcement

- [ ] **Step 1: Run short experiment**

Run: `bash scripts/run_vln_experiment_with_vllm.sh run_exp`

Expected: When fallback_mode=true, agent should turn to explore instead of walking forward blindly

- [ ] **Step 2: Check logs for fallback behavior**

Run: `grep -A3 "recommended_action" results/episode-*/episode*/agent_outputs.json | head -20`

Expected: Should see recommended_action matching exploration_hint when fallback_mode=true

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: fallback decision enforcement complete"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ AnalysisAgent fallback decision rule → Task 1
- ✅ PlanningAgent observation parameter → Task 2
- ✅ PlanningAgent fallback planning rule → Task 3
- ✅ Navigator observation passing → Task 4
- ✅ Integration test → Task 5

**2. Placeholder scan:**
- No TBD, TODO, or vague descriptions
- All code blocks contain complete implementation

**3. Type consistency:**
- `observation: Optional[ObservationOutput]` used consistently in PlanningAgent.process() and _llm_planning()
- `ObservationOutput` import added in Task 2
- hasattr() checks used for backward compatibility