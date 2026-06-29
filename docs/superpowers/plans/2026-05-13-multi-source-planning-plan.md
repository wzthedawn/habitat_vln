# Multi-Source Navigation Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate A* direction calculation and topology information into LLM prompt, enabling multi-source collaborative planning where LLM synthesizes mathematical suggestions, topology nodes, and perception status.

**Architecture:** Add `_compute_astar_direction()` method for pure math direction calculation, modify `_llm_planning()` to receive topology parameter, rebuild `_build_llm_prompt()` with multi-source context sections, and pass topology through Navigator.

**Tech Stack:** Python, math.atan2 for direction, TopologyGraph NodeType filtering, LLM prompt engineering

---

## File Structure

| File | Responsibility | Changes |
|------|----------------|---------|
| `agents/pipeline/planning_agent.py` | Path planning | Add `_compute_astar_direction()`, modify `_llm_planning()` and `_build_llm_prompt()` with topology and astar_info |
| `agents/pipeline/navigator.py` | Agent coordination | Pass topology to registry.call for planning |

---

### Task 1: Add _compute_astar_direction() method

**Files:**
- Modify: `agents/pipeline/planning_agent.py` (add new method after `_select_algorithm`)

- [ ] **Step 1: Add NodeType import**

Add at top of file (around line 22):

```python
from agents.pipeline.tools.topology_graph import TopologyGraph, KeyNode, NodeType
```

NodeType is already imported, verify it exists.

- [ ] **Step 2: Add _compute_astar_direction() method**

Add new method after `_select_algorithm()` (around line 124):

```python
def _compute_astar_direction(
    self,
    position: List[float],
    goal: Optional[List[float]],
    topology: Optional[TopologyGraph] = None,
) -> Dict[str, Any]:
    """Calculate A* direction suggestion (pure math, no LLM).
    
    When goal is None, tries to find nearest valuable topology node.
    
    Args:
        position: Current position [x, y, z]
        goal: Goal position (optional)
        topology: Topology graph (optional, for node-based subgoal)
    
    Returns:
        {
            "available": True/False,
            "direction": "left|right|forward",
            "angle": float,  # Relative to forward
            "distance": float,
            "suggested_actions": List[str]
        }
    """
    import math
    
    # No goal: try nearest valuable node from topology
    if goal is None and topology and topology.has_key_nodes():
        candidates = [
            node for node in topology.nodes
            if node.visited_count < 2  # Avoid over-visited nodes
            and node.type in [NodeType.STAIRS_ENTRY, NodeType.EXIT, NodeType.ROOM_ENTRY]
        ]
        if candidates:
            nearest = min(candidates, key=lambda n: self._compute_distance(position, n.position))
            goal = nearest.position
    
    if goal is None:
        return {
            "available": False,
            "direction": "unknown",
            "angle": 0.0,
            "distance": 0.0,
            "suggested_actions": []
        }
    
    # Calculate direction
    dx = goal[0] - position[0]
    dz = goal[2] - position[2]
    angle = math.degrees(math.atan2(dx, dz))
    dist = math.sqrt(dx*dx + dz*dz)
    
    # Convert to suggestion
    if abs(angle) < 30:
        direction = "forward"
        suggested = ["forward"] * 3
    elif angle > 0:
        direction = "right"
        suggested = ["turn_right", "forward", "forward"]
    else:
        direction = "left"
        suggested = ["turn_left", "forward", "forward"]
    
    return {
        "available": True,
        "direction": direction,
        "angle": round(angle, 1),
        "distance": round(dist, 2),
        "suggested_actions": suggested
    }
```

- [ ] **Step 3: Verify method added**

Run: `grep -n "_compute_astar_direction" agents/pipeline/planning_agent.py`
Expected: Should find the new method

- [ ] **Step 4: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: add _compute_astar_direction method for multi-source planning"
```

---

### Task 2: Modify _llm_planning() to receive topology

**Files:**
- Modify: `agents/pipeline/planning_agent.py:125-148` (_llm_planning method)

- [ ] **Step 1: Update _llm_planning signature and implementation**

Replace lines 125-148 with:

```python
def _llm_planning(
    self,
    analysis: AnalysisOutput,
    position: List[float],
    goal: Optional[List[float]],
    observation: Optional[ObservationOutput] = None,
    topology: Optional[TopologyGraph] = None,  # NEW
) -> dict:
    """LLM semantic planning with multi-source context.

    Uses LLM to generate action sequence based on analysis recommendations,
    A* direction suggestions, and topology key nodes.

    Args:
        analysis: Analysis output with recommendations
        position: Current position
        goal: Goal position (optional)
        observation: Optional ObservationOutput for fallback context
        topology: Optional TopologyGraph for key nodes (NEW)

    Returns:
        Planning result dictionary with actions and decision_source
    """
    # Calculate A* direction suggestion
    astar_info = self._compute_astar_direction(position, goal, topology)
    
    # Get nearby topology nodes
    nearby_nodes = []
    if topology and topology.has_key_nodes():
        nearby_nodes = topology.get_nearby_key_nodes(position, radius=10.0)
    
    # Build integrated prompt
    prompt = self._build_llm_prompt(
        analysis, position, goal, observation,
        astar_info, nearby_nodes  # NEW parameters
    )
    
    response = self._call_llm(prompt, max_tokens=400, temperature=0.3)
    return self._parse_response(response)
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: add topology parameter to _llm_planning and compute astar_info"
```

---

### Task 3: Rebuild _build_llm_prompt() with multi-source context

**Files:**
- Modify: `agents/pipeline/planning_agent.py:150-223` (_build_llm_prompt method)

- [ ] **Step 1: Update _build_llm_prompt signature and content**

Replace lines 150-223 with:

```python
def _build_llm_prompt(
    self,
    analysis: AnalysisOutput,
    position: List[float],
    goal: Optional[List[float]],
    observation: Optional[ObservationOutput] = None,
    astar_info: Dict[str, Any] = None,  # NEW
    nearby_nodes: List[KeyNode] = None,  # NEW
) -> str:
    """Build LLM planning prompt with multi-source navigation context.

    Args:
        analysis: Analysis output
        position: Current position
        goal: Goal position (optional)
        observation: Observation output for fallback context
        astar_info: A* direction calculation result (NEW)
        nearby_nodes: Nearby topology key nodes (NEW)

    Returns:
        Prompt string for LLM with integrated navigation context
    """
    # Initialize defaults
    if astar_info is None:
        astar_info = {"available": False}
    if nearby_nodes is None:
        nearby_nodes = []
    
    goal_info = f"Goal position: {goal}" if goal else "Goal position: unknown (semantic goal)"
    
    # Build A* direction section
    astar_section = ""
    if astar_info.get("available"):
        astar_section = f"""
### A* Direction Reference (数学建议)
- Available: true
- Direction: {astar_info['direction']}
- Angle: {astar_info['angle']}° (relative to forward)
- Distance: {astar_info['distance']}m
- Suggested actions: {', '.join(astar_info['suggested_actions'])}
"""
    else:
        astar_section = """
### A* Direction Reference (数学建议)
- Available: false (no goal or topology target)
"""
    
    # Build topology nodes section
    nodes_section = ""
    if nearby_nodes:
        nodes_text = ""
        for i, node in enumerate(nearby_nodes[:5]):  # Show top 5
            dist = self._compute_distance(position, node.position)
            nodes_text += f"- Node {i+1}: {node.type.value} at {node.position}, {dist:.1f}m (visited: {node.visited_count})\n"
        nodes_section = f"""
### Nearby Key Nodes (拓扑结构)
{nodes_text}
"""
    else:
        nodes_section = """
### Nearby Key Nodes (拓扑结构)
- No nearby key nodes found
"""
    
    # Build observation status section
    fallback_mode = False
    exploration_hint = ""
    visible_objects = "none"
    
    if observation:
        fallback_mode = hasattr(observation, 'fallback_mode') and observation.fallback_mode
        if hasattr(observation, 'exploration_hint'):
            exploration_hint = observation.exploration_hint
        if hasattr(observation, 'objects') and observation.objects:
            visible_objects = ", ".join([obj.get('name', 'unknown') if isinstance(obj, dict) else obj for obj in observation.objects])
    
    observation_section = f"""
### Observation Status (感知状态)
- Fallback mode: {fallback_mode}
- Exploration hint: {exploration_hint if exploration_hint else 'none'}
- Visible objects: {visible_objects}
"""
    
    # Build analysis reasoning section
    analysis_section = f"""
### Analysis Reasoning (LLM推理)
- Recommended action: {analysis.recommended_action}
- Current gap: {analysis.current_gap}
- Confidence: {analysis.confidence}
"""
    
    # Build full prompt with multi-source context
    prompt = f"""You are a navigation planning expert. Generate an action sequence using multi-source navigation context.

## Multi-Source Navigation Context
{astar_section}
{nodes_section}
{observation_section}
{analysis_section}

## Navigation Goal
- Goal summary: {analysis.goal_summary}
{goal_info}

## Current Position
Position: {position}

## Available Actions
- forward: Move forward one step
- turn_left: Turn left (about 90 degrees)
- turn_right: Turn right (about 90 degrees)

## Decision Priority (决策优先级)
1. If fallback_mode=true and objects=none → MUST follow exploration_hint direction
2. If nearby key nodes exist → Consider navigating through valuable nodes (stairs_entry, room_entry, exit)
3. If A* direction available → Reference math suggestion for direction
4. Synthesize analysis reasoning with above sources

## Task
Generate exactly 5 actions that best achieve the navigation goal.
Consider ALL sources above and prioritize according to decision priority rules.

## Output Format (JSON)
Output only valid JSON:
{{"actions": ["action1", "action2", "action3", "action4", "action5"], "expected_result": "description", "decision_source": "fallback|topology|astar|analysis|hybrid"}}

## Rules
- actions must contain exactly 5 items
- Each action must be one of: forward, turn_left, turn_right
- decision_source indicates which information source influenced your decision most
- Output ONLY the JSON, no additional text
"""

    return prompt
```

- [ ] **Step 2: Verify prompt structure**

Run: `grep -n "Multi-Source Navigation Context" agents/pipeline/planning_agent.py`
Expected: Should find the new prompt section

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: rebuild _build_llm_prompt with multi-source navigation context"
```

---

### Task 4: Update process() to pass topology to _llm_planning

**Files:**
- Modify: `agents/pipeline/planning_agent.py:86-92` (process method algorithm dispatch)

- [ ] **Step 1: Update the llm algorithm dispatch**

Replace lines 86-92 with:

```python
# Execute corresponding algorithm
if algorithm == "topology":
    result = self._topology_planning(topology, position, analysis)
elif algorithm == "astar":
    result = self._astar_planning(position, goal)
else:
    result = self._llm_planning(analysis, position, goal, observation, topology)  # NEW: pass topology
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: pass topology to _llm_planning in process method"
```

---

### Task 5: Update Navigator to pass topology

**Files:**
- Modify: `agents/pipeline/navigator.py:391-398` (registry.call for planning)

- [ ] **Step 1: Add topology to registry.call**

Current call passes observation. Add topology parameter:

```python
planning_output = self._registry.call(
    "planning",
    analysis=analysis_output,
    topology=self._topology,
    position=self._position,
    observation=observation_output,  # Already exists
)
```

Note: The registry.call passes kwargs, so topology will be passed through.

- [ ] **Step 2: Verify topology is passed**

Run: `grep -A5 "planning_output = self._registry.call" agents/pipeline/navigator.py`
Expected: Should see topology=self._topology in the call

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/navigator.py
git commit -m "feat: pass topology to PlanningAgent in Navigator registry.call"
```

---

### Task 6: Update _parse_response to handle decision_source

**Files:**
- Modify: `agents/pipeline/planning_agent.py:384-469` (_parse_response method)

- [ ] **Step 1: Add decision_source to default and parsing**

In `_parse_response()` around line 396-399, update default:

```python
# Default result
default = {
    "actions": ["forward"] * self.ACTION_COUNT,
    "expected_result": "unknown result",
    "decision_source": "llm",  # NEW
}
```

In `_validate_and_fill()` around line 480-484, add:

```python
required_fields = {
    "actions": ["forward"] * self.ACTION_COUNT,
    "expected_result": "unknown result",
    "decision_source": "llm",  # NEW
}
```

- [ ] **Step 2: Commit**

```bash
git add agents/pipeline/planning_agent.py
git commit -m "feat: add decision_source field to planning output parsing"
```

---

### Task 7: Integration test

**Files:**
- Test: Run experiment to verify multi-source planning works

- [ ] **Step 1: Run short experiment**

Run: `bash scripts/run_vln_experiment_with_vllm.sh run_exp`

Expected: When fallback_mode=true, agent should receive multi-source context and generate turn actions

- [ ] **Step 2: Check logs for multi-source context**

Run: `grep "A\* Direction Reference\|Nearby Key Nodes\|decision_source" results/episode-*/episode*/agent_outputs.json | head -10`

Expected: Should see astar_info and decision_source fields in output

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: multi-source navigation planning complete"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ `_compute_astar_direction()` method → Task 1
- ✅ `_llm_planning()` topology parameter → Task 2
- ✅ `_build_llm_prompt()` multi-source sections → Task 3
- ✅ process() topology dispatch → Task 4
- ✅ Navigator topology passing → Task 5
- ✅ decision_source field → Task 6
- ✅ Integration test → Task 7

**2. Placeholder scan:**
- No TBD, TODO, or vague descriptions
- All code blocks contain complete implementation

**3. Type consistency:**
- `topology: Optional[TopologyGraph]` used consistently
- `astar_info: Dict[str, Any]` used consistently
- `nearby_nodes: List[KeyNode]` used consistently
- NodeType imported for node filtering