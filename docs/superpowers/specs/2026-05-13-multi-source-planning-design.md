# Multi-Source Navigation Planning Design

## Problem Statement

Current PlanningAgent uses mutually exclusive algorithm selection:
- Topology OR A* OR LLM (only one at a time)
- LLM planning only receives AnalysisOutput, missing mathematical direction guidance and topology structure
- When fallback_mode=true, LLM generates 5 "forward" actions blindly without spatial context

## Solution Overview

Integrate A* direction calculation and topology information into LLM prompt, enabling multi-source collaborative planning. LLM synthesizes mathematical suggestions, topology nodes, and perception status to generate optimal action sequences.

---

## Design Details

### 1. A* Direction Calculation (Pure Math, No LLM)

**New method `_compute_astar_direction()`:**

```python
def _compute_astar_direction(
    self,
    position: List[float],
    goal: Optional[List[float]],
    topology: Optional[TopologyGraph] = None,
) -> Dict[str, Any]:
    """Calculate A* direction suggestion (math only).
    
    Returns:
        {
            "available": True/False,
            "direction": "left|right|forward",
            "angle": -45.0,  # Relative to current heading
            "distance": 5.2,
            "suggested_actions": ["turn_left", "forward", "forward"]
        }
    """
    # No goal: try nearest valuable node from topology
    if goal is None and topology and topology.has_key_nodes():
        candidates = [
            node for node in topology.nodes
            if node.visited_count < 2  # Avoid over-visited nodes
            and node.type in [NodeType.STAIRS_ENTRY, NodeType.EXIT, NodeType.ROOM_ENTRY]
        ]
        if candidates:
            nearest = min(candidates, key=lambda n: distance(position, n.position))
            goal = nearest.position
    
    if goal is None:
        return {"available": False, "direction": "unknown", "distance": 0, "suggested_actions": []}
    
    # Calculate direction
    dx = goal[0] - position[0]
    dz = goal[2] - position[2]
    angle = math.degrees(math.atan2(dx, dz))
    distance = math.sqrt(dx*dx + dz*dz)
    
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
    
    return {"available": True, "direction": direction, "angle": round(angle,1), 
            "distance": round(distance,2), "suggested_actions": suggested}
```

**Key design:**

| Scenario | Handling |
|----------|----------|
| goal=None, has topology nodes | Select nearest unvisited valuable node (STAIRS_ENTRY, EXIT, ROOM_ENTRY) |
| goal=None, no topology | Return `available=False` |
| Has goal | Calculate direction and distance |
| visited_count >= 2 | Skip to avoid loops |

### 2. Prompt Structure Enhancement

**New prompt layout:**

```
┌─────────────────────────────────────────────────────┐
│  ## Multi-Source Navigation Context                 │
│                                                     │
│  ### A* Direction Reference (数学建议)              │
│  - Available: true/false                            │
│  - Direction: left/right/forward                    │
│  - Angle: -45.0°                                    │
│  - Distance: 5.2m                                   │
│  - Suggested: turn_left, forward, forward           │
│                                                     │
│  ### Nearby Key Nodes (拓扑结构)                    │
│  - Node 1: stairs_entry at [x,y,z], 3.5m            │
│  - Node 2: room_entry at [x,y,z], 2.1m              │
│  - (visited_count info for loop avoidance)          │
│                                                     │
│  ### Observation Status (感知状态)                  │
│  - Fallback mode: true/false                        │
│  - Exploration hint: "turn_left to find stairs"     │
│  - Visible objects: none / stairs(forward)          │
│                                                     │
│  ### Analysis Reasoning (LLM推理)                   │
│  - Recommended action: turn_left                    │
│  - Gap: stairs not visible                          │
│                                                     │
├─────────────────────────────────────────────────────┤
│  ## Decision Priority (决策优先级)                  │
│  1. Fallback mode → Follow exploration_hint         │
│  2. Key node nearby → Navigate through node         │
│  3. A* direction available → Reference math guide   │
│  4. Analysis reasoning → Synthesize all sources     │
│                                                     │
│  RULE: If fallback_mode=true and objects=none,      │
│  MUST prioritize exploration_hint direction.        │
├─────────────────────────────────────────────────────┤
│  ## Output Format                                   │
│  {"actions": ["..."], "expected_result": "...",     │
│   "decision_source": "fallback|topology|astar|llm"} │
└─────────────────────────────────────────────────────┘
```

### 3. Method Signature Changes

**PlanningAgent._llm_planning():**

```python
def _llm_planning(
    self,
    analysis: AnalysisOutput,
    position: List[float],
    goal: Optional[List[float]],
    observation: Optional[ObservationOutput] = None,
    topology: Optional[TopologyGraph] = None,  # NEW
) -> dict:
    # Calculate A* direction
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

**PlanningAgent.process():**

```python
def process(self, analysis, topology, position, goal=None, observation=None):
    ...
    else:
        result = self._llm_planning(analysis, position, goal, observation, topology)
```

### 4. Navigator Integration

**Current call (line 392-398):**

```python
result = self._llm_planning(analysis, position, goal, observation)
```

**Change:**

```python
result = self._llm_planning(analysis, position, goal, observation, topology)
```

---

## Data Flow After Changes

```
Navigator.step()
    ↓
ObservationAgent → ObservationOutput(fallback_mode, exploration_hint, objects)
    ↓
AnalysisAgent → AnalysisOutput(recommended_action, reasoning)
    ↓
PlanningAgent.process(analysis, topology, position, goal, observation)
    ↓
    ├─ _compute_astar_direction(position, goal, topology)
    │    → astar_info (direction, angle, distance, suggested_actions)
    │
    ├─ topology.get_nearby_key_nodes(position)
    │    → nearby_nodes (key navigation points)
    │
    ↓
    _build_llm_prompt(analysis, position, goal, observation, astar_info, nearby_nodes)
    ↓
    LLM synthesizes all sources → PlanningOutput(actions, decision_source)
    ↓
Navigator executes actions
```

---

## Expected Behavior

| Scenario | Before | After |
|----------|--------|-------|
| fallback_mode=true, no goal | 5 "forward" blindly | Turn based on exploration_hint + A* suggestion |
| goal available, no topology | Only analysis info | A* direction as math reference |
| topology nodes available | Ignored by LLM planning | Nodes shown to LLM as navigation options |
| Near stairs_entry, visited once | May walk past | Navigate toward node |
| Near turn_point, visited 3 times | May loop back | Node filtered out (visited_count >= 2) |

---

## Files to Modify

1. `agents/pipeline/planning_agent.py`
   - Add `_compute_astar_direction()` method
   - Modify `_llm_planning()` to receive topology and compute astar_info
   - Modify `_build_llm_prompt()` to include astar_info and nearby_nodes

2. `agents/pipeline/navigator.py`
   - Pass topology to _llm_planning call (through registry.call)

---

## Verification

1. Run experiment with episode where fallback_mode=true
2. Check agent_outputs.json:
   - `planning_output.actions` should include turn actions (not just forward)
   - `decision_source` field should indicate which source influenced decision
3. Verify no navigation loops (visited_count filtering works)