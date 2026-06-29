"""PlanningAgent - LLM core, path planning.

This agent performs path planning and generates action sequences using
multiple algorithms:
- LLM: Semantic planning based on analysis recommendations (default)
- Topology: Planning using key nodes (turn points, room entries, etc.)
- A*: Shortest path planning when goal coordinates are available
"""

import json
import logging
import math
import re
from typing import Dict, Any, List, Optional

from agents.pipeline.base_pipeline_agent import (
    AnalysisOutput,
    ObservationOutput,
    PlanningOutput,
    SubAgent,
)
from agents.pipeline.tools.topology_graph import TopologyGraph, KeyNode, NodeType
from agents.base_agent import AgentRole


class PlanningAgent(SubAgent):
    """Planning Agent - LLM core, path planning.

    Responsibilities:
    - Generate action sequences for navigation
    - Select appropriate planning algorithm
    - Output expected results and path information

    Algorithms:
    - topology: When key nodes exist in topology graph (highest priority)
    - astar: When goal coordinates are available (no key nodes)
    - llm: Default semantic planning (fallback)
    """

    name = "planning_agent"

    ALGORITHMS = ["llm", "topology", "astar"]

    # Number of actions to generate per plan
    ACTION_COUNT = 5

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PlanningAgent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("PlanningAgent")

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def process(
        self,
        analysis: AnalysisOutput,
        topology: TopologyGraph,
        position: List[float],
        goal: Optional[List[float]] = None,
        observation: Optional[ObservationOutput] = None,
    ) -> PlanningOutput:
        """Path planning.

        Args:
            analysis: AnalysisAgent output with recommendations
            topology: Topology graph with key nodes
            position: Current position [x, y, z]
            goal: Optional goal position [x, y, z]
            observation: Optional ObservationOutput for fallback context

        Returns:
            PlanningOutput with action sequence and expected result
        """
        # Algorithm selection (rule-based, no LLM)
        algorithm = self._select_algorithm(topology, goal, observation)

        self.logger.info(f"[PlanningAgent] Selected algorithm: {algorithm}")

        # Execute corresponding algorithm
        if algorithm == "topology":
            result = self._topology_planning(topology, position, analysis)
        elif algorithm == "astar":
            result = self._astar_planning(position, goal)
        else:
            result = self._llm_planning(analysis, position, goal, observation, topology)  # NEW: pass topology

        # Add algorithm_used to result
        result["algorithm_used"] = algorithm

        print(f"[PlanningAgent] algo={algorithm} actions={result.get('actions', [])}")

        return PlanningOutput(**result)

    def _select_algorithm(self, topology: TopologyGraph, goal: Optional[List[float]],
                          observation: Optional[ObservationOutput] = None) -> str:
        """Select planning algorithm based on available information.

        Rule-based selection logic (no LLM):
        - llm: when stairs are detected (needs semantic reasoning)
        - topology: when topology has key nodes and no stairs
        - astar: when goal coordinates available
        - llm: default fallback

        Args:
            topology: Topology graph
            goal: Goal position (optional)
            observation: Observation output for stair detection

        Returns:
            Algorithm name: "topology", "astar", or "llm"
        """
        # Stair navigation needs LLM semantic reasoning
        if observation and hasattr(observation, 'stair_position'):
            if observation.stair_position in ("top", "bottom"):
                return "llm"

        # Topology planning: when key nodes exist
        if topology.has_key_nodes():
            return "topology"

        # A* planning: when goal coordinates are available
        if goal is not None:
            return "astar"

        # LLM planning: default fallback
        return "llm"

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

    def _llm_planning(
        self,
        analysis: AnalysisOutput,
        position: List[float],
        goal: Optional[List[float]],
        observation: Optional[ObservationOutput] = None,
        topology: Optional[TopologyGraph] = None,
    ) -> dict:
        """LLM semantic planning with multi-source context.

        Uses LLM to generate action sequence based on analysis recommendations,
        A* direction suggestions, and topology key nodes.

        Args:
            analysis: Analysis output with recommendations
            position: Current position
            goal: Goal position (optional)
            observation: Optional ObservationOutput for fallback context
            topology: Optional TopologyGraph for key nodes

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
            astar_info, nearby_nodes
        )

        response = self._call_llm(prompt, max_tokens=400, temperature=0.3)
        return self._parse_response(response)

    def _build_llm_prompt(
        self,
        analysis: AnalysisOutput,
        position: List[float],
        goal: Optional[List[float]],
        observation: Optional[ObservationOutput] = None,
        astar_info: Dict[str, Any] = None,  # NEW
        nearby_nodes: List = None,  # NEW (List[KeyNode] but KeyNode not in typing imports)
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

## CRITICAL: Alignment-First Rule
Before going forward, you MUST align with the target. Check visible objects:
- If a target object (stairs, door, etc.) has direction "forward_left" → start with 1x turn_left
- If direction is "forward_right" → start with 1x turn_right
- If direction is "left" → start with 2x turn_left
- If direction is "right" → start with 2x turn_right
- ONLY if all target objects show direction="forward" → all forward actions

## Task
Generate exactly 5 actions. Apply the Alignment-First Rule above.
First 1-2 actions: turn to face the target. Remaining actions: move forward.

## Output Format (JSON)
Output only valid JSON:
{{"actions": ["action1", "action2", "action3", "action4", "action5"], "expected_result": "description", "decision_source": "fallback|topology|astar|analysis|hybrid"}}

## Rules
- actions must contain exactly 5 items
- Each action must be one of: forward, turn_left, turn_right
- First action(s) MUST be turns if target is not directly forward
- decision_source indicates which information source influenced your decision most
- Output ONLY the JSON, no additional text
"""

        return prompt

    def _topology_planning(
        self,
        topology: TopologyGraph,
        position: List[float],
        analysis: AnalysisOutput,
    ) -> dict:
        """Topology-based planning.

        Uses key nodes (turn points, room entries, etc.) for planning.

        Args:
            topology: Topology graph with key nodes
            position: Current position
            analysis: Analysis output

        Returns:
            Planning result dictionary with path information
        """
        # Get nearby key nodes
        nearby_nodes = topology.get_nearby_key_nodes(position, radius=10.0)

        # Build prompt with topology information
        prompt = self._build_topology_prompt(analysis, nearby_nodes, position)
        response = self._call_llm(prompt, max_tokens=400, temperature=0.3)

        result = self._parse_response(response)

        # Add path information (list of nearby key nodes)
        result["path"] = [
            {
                "id": node.id,
                "position": node.position,
                "type": node.type.value,
            }
            for node in nearby_nodes[:3]  # Top 3 nearby nodes
        ]

        return result

    def _build_topology_prompt(
        self,
        analysis: AnalysisOutput,
        nearby_nodes: List[KeyNode],
        position: List[float],
    ) -> str:
        """Build topology planning prompt.

        Args:
            analysis: Analysis output
            nearby_nodes: List of nearby key nodes
            position: Current position

        Returns:
            Prompt string for LLM
        """
        # Format nearby nodes for prompt
        nodes_text = ""
        for i, node in enumerate(nearby_nodes[:5]):  # Show top 5
            distance = self._compute_distance(position, node.position)
            nodes_text += f"""
### Node {i+1}
- Type: {node.type.value}
- Position: {node.position}
- Distance: {distance:.2f}m
- Rotation: {node.rotation:.1f} degrees
"""

        prompt = f"""You are a navigation planning expert with topology information. Use nearby key nodes to plan navigation.

## Navigation Goal
- Goal: {analysis.goal_summary}
- Current gap: {analysis.current_gap}
- Recommended action: {analysis.recommended_action}

## Current Position
Position: {position}

## Nearby Key Nodes (Navigation landmarks)
{nodes_text if nodes_text else "No nearby key nodes found"}

## CRITICAL: Alignment-First Rule
Before moving forward, turn to face the target direction:
- If recommended_action is "turn_left" → start with turn_left
- If recommended_action is "turn_right" → start with turn_right
- If recommended_action is "forward" BUT the nearest valuable node is to the left/right → turn toward it first
- Stairs nodes (stairs_entry) with direction "descend" → must walk forward toward them
- ONLY all forward if target is directly ahead

## Task
Generate exactly 5 actions. Apply the Alignment-First Rule above.

## Output Format (JSON)
Output only valid JSON:
{{"actions": ["action1", "action2", "action3", "action4", "action5"], "expected_result": "description of expected result"}}

## Rules
- actions must contain exactly 5 items
- Each action must be one of: forward, turn_left, turn_right
- First action MUST turn toward the target if not directly forward
- Choose actions that navigate toward the most useful key node
- Output ONLY the JSON"""

        return prompt

    def _astar_planning(
        self,
        position: List[float],
        goal: List[float],
    ) -> dict:
        """A* planning (simplified version).

        Generates direction-based actions toward goal coordinates.

        Args:
            position: Current position [x, y, z]
            goal: Goal position [x, y, z]

        Returns:
            Planning result dictionary
        """
        # Calculate direction to goal
        dx = goal[0] - position[0]
        dz = goal[2] - position[2]

        # Generate action sequence based on direction
        actions = self._generate_direction_actions(dx, dz)

        # Calculate expected result
        distance = self._compute_distance(position, goal)
        expected_result = f"reach goal at {goal} (distance: {distance:.2f}m)"

        return {
            "actions": actions,
            "expected_result": expected_result,
            "path": None,
        }

    def _generate_direction_actions(self, dx: float, dz: float) -> List[str]:
        """Generate action sequence based on direction.

        Args:
            dx: X-axis delta (positive = goal is to the right)
            dz: Z-axis delta (positive = goal is ahead)

        Returns:
            List of 5 actions
        """
        actions = []

        # Determine turn direction based on dx
        turn_threshold = 1.0  # Need at least 1m lateral distance to turn

        if abs(dx) > turn_threshold:
            if dx > 0:
                # Goal is to the right
                actions.append("turn_right")
            else:
                # Goal is to the left
                actions.append("turn_left")

        # Fill remaining slots with forward actions
        remaining = self.ACTION_COUNT - len(actions)
        for _ in range(remaining):
            actions.append("forward")

        return actions

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response to structured dictionary.

        Multi-layer fallback for robust parsing.

        Args:
            response: LLM response text

        Returns:
            Parsed dictionary with required fields
        """
        # Default result
        default = {
            "actions": ["forward"] * self.ACTION_COUNT,
            "expected_result": "unknown result",
            "decision_source": "llm",
        }

        if not response:
            return default

        # Layer 1: Try direct JSON parsing
        try:
            result = json.loads(response.strip())
            return self._validate_and_fill(result)
        except (json.JSONDecodeError, ValueError):
            pass

        # Layer 2: Extract JSON from markdown code block
        if "```json" in response:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    return self._validate_and_fill(result)
                except (json.JSONDecodeError, ValueError):
                    pass
        elif "```" in response:
            match = re.search(r"```\s*([\s\S]*?)\s*```", response)
            if match:
                try:
                    result = json.loads(match.group(1).strip())
                    return self._validate_and_fill(result)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Layer 3: Regex extract actions
        actions_match = re.search(r'"actions":\s*\[([^\]]+)\]', response)
        if actions_match:
            actions_str = actions_match.group(1)
            # Extract individual actions
            actions = re.findall(r'"(\w+)"', actions_str)
            if actions:
                # Validate actions
                valid_actions = ["forward", "turn_left", "turn_right"]
                actions = [a for a in actions if a in valid_actions]
                # Fill to exactly 5
                while len(actions) < self.ACTION_COUNT:
                    actions.append("forward")

                result = default.copy()
                result["actions"] = actions[:self.ACTION_COUNT]

                # Try to extract expected_result
                result_match = re.search(r'"expected_result":\s*"([^"]+)"', response)
                if result_match:
                    result["expected_result"] = result_match.group(1)

                return result

        # Layer 4: Keyword inference
        response_lower = response.lower()
        actions = []
        if "turn_left" in response_lower or "左转" in response:
            actions.append("turn_left")
        elif "turn_right" in response_lower or "右转" in response:
            actions.append("turn_right")

        # Fill remaining with forward
        while len(actions) < self.ACTION_COUNT:
            actions.append("forward")

        result = default.copy()
        result["actions"] = actions
        result["expected_result"] = response[:100] if len(response) > 100 else response

        return result

    def _validate_and_fill(self, result: dict) -> dict:
        """Validate parsed result and fill missing fields.

        Args:
            result: Parsed result dictionary

        Returns:
            Validated and filled dictionary
        """
        # Ensure all required fields exist
        required_fields = {
            "actions": ["forward"] * self.ACTION_COUNT,
            "expected_result": "unknown result",
            "decision_source": "llm",
        }

        for field, default_value in required_fields.items():
            if field not in result:
                result[field] = default_value

        # Validate actions
        valid_actions = ["forward", "turn_left", "turn_right"]
        if "actions" in result:
            # Filter invalid actions
            result["actions"] = [a for a in result["actions"] if a in valid_actions]
            # Ensure exactly 5 actions
            while len(result["actions"]) < self.ACTION_COUNT:
                result["actions"].append("forward")
            result["actions"] = result["actions"][:self.ACTION_COUNT]

        # Ensure path field exists for topology planning
        if "path" not in result:
            result["path"] = None

        return result

    def _compute_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Compute Euclidean distance between two positions.

        Args:
            pos1: First position [x, y, z]
            pos2: Second position [x, y, z]

        Returns:
            Euclidean distance
        """
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )