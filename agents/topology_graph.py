"""TopologyGraph - Topological memory for navigation trajectory compression.

This module implements a topology graph that compresses navigation trajectories
into 10-20 key nodes, enabling LLM to understand spatial context without
being overwhelmed by raw position lists.

Key components:
- GraphNode: Represents a key position (junction, room entrance, stairs, etc.)
- GraphEdge: Represents a traversable path between nodes
- TopologyGraph: Manages the graph structure and provides LLM-readable summaries
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math


@dataclass
class GraphNode:
    """Represents a key position in the navigation topology.

    Attributes:
        node_id: Unique identifier, e.g., "node_0", "node_1"
        position: 3D coordinates (x, y, z)
        node_type: Type of key position - junction/room_entrance/stairs/stuck_region
        timestamp: Navigation step when this node was created
        visit_count: Number of times this node was revisited
        semantic_info: Optional semantic description, e.g., "厨房入口", "楼梯底部"
        is_key_position: Whether this is a critical node (prioritized during pruning)
    """
    node_id: str
    position: Tuple[float, float, float]
    node_type: str  # junction/room_entrance/stairs/stuck_region
    timestamp: int
    visit_count: int = 1
    semantic_info: Optional[str] = None
    is_key_position: bool = False


@dataclass
class GraphEdge:
    """Represents a connection between two nodes in the topology.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        distance: Euclidean distance between nodes
        action_sequence: List of actions to traverse this edge, e.g., ["forward", "turn_left"]
        traversable: Whether this path is currently passable (False for blocked_path)
    """
    source_id: str
    target_id: str
    distance: float
    action_sequence: List[str] = field(default_factory=list)
    traversable: bool = True


@dataclass
class TopologyGraph:
    """Topology graph for compressing navigation trajectories.

    The topology graph maintains a compact representation of the navigation
    history by storing only key positions (junctions, room entrances, etc.)
    and their connections.

    Attributes:
        nodes: Dictionary mapping node_id to GraphNode
        edges: Dictionary mapping edge_key (e.g., "edge_0_1") to GraphEdge
        current_node_id: ID of the node closest to current position
        goal_node_id: ID of the node representing the goal location
    """
    # Pruning constants
    MAX_NODES: int = field(default=30, repr=False)  # LLM context limit
    PRUNE_CHECK_INTERVAL: int = field(default=50, repr=False)  # Check every 50 steps

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)
    current_node_id: Optional[str] = None
    goal_node_id: Optional[str] = None

    # Internal counter for generating unique node IDs
    _node_counter: int = field(default=0, repr=False)

    def add_node(
        self,
        position: Tuple[float, float, float],
        node_type: str,
        semantic_info: Optional[str] = None,
        timestamp: Optional[int] = None
    ) -> str:
        """Add a new node to the topology graph.

        Args:
            position: 3D coordinates of the node
            node_type: Type of node (junction/room_entrance/stairs/stuck_region)
            semantic_info: Optional semantic description
            timestamp: Navigation step when created (defaults to current node count)

        Returns:
            The node_id of the newly created node
        """
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1

        if timestamp is None:
            timestamp = len(self.nodes)

        node = GraphNode(
            node_id=node_id,
            position=position,
            node_type=node_type,
            timestamp=timestamp,
            visit_count=1,
            semantic_info=semantic_info,
            is_key_position=False
        )

        self.nodes[node_id] = node
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        action_sequence: List[str],
        traversable: bool = True
    ) -> str:
        """Add an edge connecting two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            action_sequence: List of actions to traverse this edge
            traversable: Whether this path is passable

        Returns:
            The edge_key of the newly created edge

        Raises:
            ValueError: If either source or target node doesn't exist
        """
        if source_id not in self.nodes:
            raise ValueError(f"Source node '{source_id}' does not exist")
        if target_id not in self.nodes:
            raise ValueError(f"Target node '{target_id}' does not exist")

        # Calculate distance between nodes
        source_pos = self.nodes[source_id].position
        target_pos = self.nodes[target_id].position
        distance = self._calculate_distance(source_pos, target_pos)

        # Create edge key (bidirectional, use sorted order for consistency)
        edge_key = f"edge_{source_id}_{target_id}"

        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            distance=distance,
            action_sequence=action_sequence,
            traversable=traversable
        )

        self.edges[edge_key] = edge
        return edge_key

    def update_current_node(self, position: Tuple[float, float, float]) -> Optional[str]:
        """Update the current node based on position.

        Finds the nearest node to the given position and updates current_node_id.
        If within a threshold distance, increments that node's visit count.

        Args:
            position: Current 3D position

        Returns:
            The current_node_id after update, or None if no nodes exist
        """
        if not self.nodes:
            return None

        # Find nearest node
        nearest_id = None
        nearest_distance = float('inf')

        for node_id, node in self.nodes.items():
            dist = self._calculate_distance(position, node.position)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_id = node_id

        # Update current node if within reasonable distance
        if nearest_distance < 2.0:  # 2 meters threshold
            self.current_node_id = nearest_id
            self.nodes[nearest_id].visit_count += 1
        else:
            self.current_node_id = nearest_id

        return self.current_node_id

    def set_goal_node(self, position: Tuple[float, float, float]) -> Optional[str]:
        """Set the goal node based on position.

        Finds the nearest node to the goal position and marks it as the goal.
        Also marks it as a key position for pruning priority.

        Args:
            position: Goal 3D position

        Returns:
            The goal_node_id after update, or None if no nodes exist
        """
        if not self.nodes:
            return None

        # Find nearest node to goal position
        nearest_id = None
        nearest_distance = float('inf')

        for node_id, node in self.nodes.items():
            dist = self._calculate_distance(position, node.position)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_id = node_id

        self.goal_node_id = nearest_id

        # Mark goal node as key position
        if nearest_id:
            self.nodes[nearest_id].is_key_position = True

        return self.goal_node_id

    def _calculate_distance(
        self,
        pos1: Tuple[float, float, float],
        pos2: Tuple[float, float, float]
    ) -> float:
        """Calculate Euclidean distance between two 3D positions.

        Args:
            pos1: First position (x, y, z)
            pos2: Second position (x, y, z)

        Returns:
            Euclidean distance
        """
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def prune_nodes(self) -> None:
        """Compress node count to MAX_NODES or fewer.

        Pruning strategy:
        1. Mark key positions (goal, stuck_region, room_entrance, high visit count)
        2. Merge non-key adjacent nodes into path_segment nodes
        3. Remove orphan nodes (no connected edges)

        Iteratively applies pruning until target is reached or no more merges possible.
        """
        # Iteratively prune until target reached
        while len(self.nodes) > self.MAX_NODES:
            # Step 1: Mark key positions (prioritized for retention)
            self._mark_key_positions()

            # Step 2: Merge non-key adjacent nodes
            merged = self._merge_non_key_nodes()
            if not merged:
                # No more non-key nodes to merge, stop
                break

            # Step 3: Remove orphan nodes (no connected edges)
            self._remove_orphan_nodes()

    def _mark_key_positions(self) -> None:
        """Mark key positions that should be prioritized during pruning.

        Key positions include:
        - Goal node
        - stuck_region nodes
        - room_entrance nodes
        - Nodes with high visit count (>=3)
        """
        for node in self.nodes.values():
            node.is_key_position = (
                node.node_id == self.goal_node_id or
                node.node_type in ["stuck_region", "room_entrance"] or
                node.visit_count >= 3
            )

    def _merge_non_key_nodes(self) -> bool:
        """Merge adjacent non-key nodes into path_segment nodes.

        Finds pairs of adjacent non-key nodes and merges them into a single
        path_segment node, reducing total node count while preserving connectivity.

        Returns:
            True if any nodes were merged, False otherwise.
        """
        # Build adjacency map for efficient neighbor lookup
        adjacency = self._build_adjacency_map()

        # Find pairs of non-key nodes that are adjacent
        merged = set()
        nodes_to_remove = []
        nodes_to_add = []
        merged_any = False

        for node_id, node in list(self.nodes.items()):
            if node_id in merged or node.is_key_position:
                continue

            # Find adjacent non-key nodes
            neighbors = adjacency.get(node_id, [])
            for neighbor_id in neighbors:
                if neighbor_id in merged:
                    continue
                neighbor = self.nodes.get(neighbor_id)
                if neighbor is None or neighbor.is_key_position:
                    continue

                # Merge these two non-key nodes
                new_node = self._create_merged_node(node, neighbor)
                nodes_to_add.append(new_node)
                nodes_to_remove.extend([node_id, neighbor_id])
                merged.add(node_id)
                merged.add(neighbor_id)
                merged_any = True

                # Update edges to point to merged node
                self._update_edges_for_merge(node_id, neighbor_id, new_node.node_id, adjacency)
                break  # Only merge one pair per iteration to maintain graph integrity

        # Apply changes
        for node_id in nodes_to_remove:
            if node_id in self.nodes:
                del self.nodes[node_id]

        for node in nodes_to_add:
            self.nodes[node.node_id] = node

        return merged_any

    def _create_merged_node(self, node1: GraphNode, node2: GraphNode) -> GraphNode:
        """Create a merged path_segment node from two nodes.

        Args:
            node1: First node to merge
            node2: Second node to merge

        Returns:
            New GraphNode with node_type='path_segment' combining both nodes
        """
        new_node_id = f"node_{self._node_counter}"
        self._node_counter += 1

        # Use midpoint of positions
        new_position = (
            (node1.position[0] + node2.position[0]) / 2,
            (node1.position[1] + node2.position[1]) / 2,
            (node1.position[2] + node2.position[2]) / 2
        )

        # Combine visit counts and use earlier timestamp
        new_node = GraphNode(
            node_id=new_node_id,
            position=new_position,
            node_type="path_segment",
            timestamp=min(node1.timestamp, node2.timestamp),
            visit_count=node1.visit_count + node2.visit_count,
            semantic_info=None,
            is_key_position=False
        )

        return new_node

    def _update_edges_for_merge(
        self,
        old_id1: str,
        old_id2: str,
        new_id: str,
        adjacency: Dict[str, List[str]]
    ) -> None:
        """Update edges after merging two nodes.

        Args:
            old_id1: First merged node ID
            old_id2: Second merged node ID
            new_id: New merged node ID
            adjacency: Adjacency map for edge updates
        """
        edges_to_remove = []
        edges_to_add = []

        for edge_key, edge in list(self.edges.items()):
            source = edge.source_id
            target = edge.target_id

            # Check if this edge involves either of the merged nodes
            if source in (old_id1, old_id2) or target in (old_id1, old_id2):
                edges_to_remove.append(edge_key)

                # Create new edge if not connecting the two merged nodes to each other
                if source in (old_id1, old_id2) and target in (old_id1, old_id2):
                    # Edge between the two merged nodes - skip, no longer needed
                    continue
                elif source in (old_id1, old_id2):
                    # Source was merged, update to new node
                    new_edge_key = f"edge_{new_id}_{target}"
                    new_edge = GraphEdge(
                        source_id=new_id,
                        target_id=target,
                        distance=self._calculate_distance(
                            self.nodes[new_id].position,
                            self.nodes[target].position
                        ) if new_id in self.nodes and target in self.nodes else edge.distance,
                        action_sequence=edge.action_sequence.copy(),
                        traversable=edge.traversable
                    )
                    edges_to_add.append((new_edge_key, new_edge))
                elif target in (old_id1, old_id2):
                    # Target was merged, update to new node
                    new_edge_key = f"edge_{source}_{new_id}"
                    new_edge = GraphEdge(
                        source_id=source,
                        target_id=new_id,
                        distance=self._calculate_distance(
                            self.nodes[source].position,
                            self.nodes[new_id].position
                        ) if source in self.nodes and new_id in self.nodes else edge.distance,
                        action_sequence=edge.action_sequence.copy(),
                        traversable=edge.traversable
                    )
                    edges_to_add.append((new_edge_key, new_edge))

        # Apply edge changes
        for edge_key in edges_to_remove:
            if edge_key in self.edges:
                del self.edges[edge_key]

        for edge_key, edge in edges_to_add:
            self.edges[edge_key] = edge

    def _build_adjacency_map(self) -> Dict[str, List[str]]:
        """Build an adjacency map from edges for efficient neighbor lookup.

        Returns:
            Dictionary mapping node_id to list of adjacent node_ids
        """
        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges.values():
            if not edge.traversable:
                continue
            source = edge.source_id
            target = edge.target_id
            if source not in adjacency:
                adjacency[source] = []
            if target not in adjacency:
                adjacency[target] = []
            if target not in adjacency[source]:
                adjacency[source].append(target)
            if source not in adjacency[target]:
                adjacency[target].append(source)
        return adjacency

    def _remove_orphan_nodes(self) -> None:
        """Remove orphan nodes that have no connected edges.

        Orphan nodes provide no navigation value and can be safely removed.
        Preserves the goal node even if orphaned.
        """
        # Find all nodes that have at least one connected edge
        connected_nodes = set()
        for edge in self.edges.values():
            connected_nodes.add(edge.source_id)
            connected_nodes.add(edge.target_id)

        # Remove nodes that are not connected and not the goal
        nodes_to_remove = []
        for node_id, node in self.nodes.items():
            if node_id not in connected_nodes and node_id != self.goal_node_id:
                nodes_to_remove.append(node_id)

        for node_id in nodes_to_remove:
            del self.nodes[node_id]

            # Clear current_node_id if it was the removed node
            if self.current_node_id == node_id:
                self.current_node_id = None

    def get_summary(self) -> Dict[str, Any]:
        """Return LLM-readable topology summary.

        This method provides a structured summary of the topology graph
        that can be easily understood by DecisionAgent's LLM reasoning.

        Returns:
            Dictionary containing:
            - total_nodes: Total number of nodes in the graph
            - key_nodes: List of key position nodes (id, type, position)
            - current_node: ID of current position node
            - goal_node: ID of goal position node
            - path_to_goal: List of node IDs forming path from current to goal
            - visited_rooms: Unique list of visited room names
            - stuck_regions: List of stuck region nodes (id, position)
        """
        return {
            "total_nodes": len(self.nodes),
            "key_nodes": [
                {
                    "id": n.node_id,
                    "type": n.node_type,
                    "position": list(n.position)  # Convert tuple to list for JSON
                }
                for n in self.nodes.values() if n.is_key_position
            ],
            "current_node": self.current_node_id,
            "goal_node": self.goal_node_id,
            "path_to_goal": self._find_path_to_goal(),
            "visited_rooms": sorted(set(
                n.semantic_info for n in self.nodes.values()
                if n.node_type == "room_entrance" and n.semantic_info
            )),
            "stuck_regions": [
                {
                    "id": n.node_id,
                    "position": list(n.position)  # Convert tuple to list for JSON
                }
                for n in self.nodes.values() if n.node_type == "stuck_region"
            ],
        }

    def _find_path_to_goal(self) -> List[str]:
        """Find path from current node to goal node using BFS.

        Returns:
            List of node IDs forming the shortest path from current_node to goal_node.
            Returns empty list if:
            - No current node or goal node set
            - No path exists
            - Current node is the goal node
        """
        # Handle edge cases
        if self.current_node_id is None or self.goal_node_id is None:
            return []

        if self.current_node_id == self.goal_node_id:
            return [self.current_node_id]

        if self.current_node_id not in self.nodes or self.goal_node_id not in self.nodes:
            return []

        # Build adjacency map from traversable edges
        adjacency = self._build_adjacency_map()

        # BFS to find shortest path
        queue = deque([(self.current_node_id, [self.current_node_id])])
        visited = {self.current_node_id}

        while queue:
            current_id, path = queue.popleft()

            # Check neighbors
            neighbors = adjacency.get(current_id, [])
            for neighbor_id in neighbors:
                if neighbor_id == self.goal_node_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited and neighbor_id in self.nodes:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        # No path found
        return []


class KeyPositionDetector:
    """Detects key positions in navigation to trigger node creation.

    Key positions are critical navigation landmarks:
    - junction: Turns or direction changes indicating decision points
    - room_entrance: Semantic transitions between rooms
    - stairs: Significant height changes
    - stuck_region: Areas with repetitive actions (potential navigation issues)
    """

    # Detection thresholds with rationale
    STAIRS_HEIGHT_THRESHOLD = 0.5  # meters - typical single step height
    JUNCTION_HISTORY_WINDOW = 3    # minimum actions to detect direction change
    STUCK_REGION_WINDOW = 10       # actions to analyze for stuck detection
    STUCK_UNIQUE_ACTIONS_MAX = 2   # max unique actions indicating stuck state

    def detect(
        self,
        current_pos: Tuple[float, float, float],
        prev_pos: Tuple[float, float, float],
        action_history: List[str],
        semantic_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Detect if current position is a key position.

        Checks for key positions in priority order:
        1. stairs - significant height change
        2. room_entrance - semantic room change
        3. junction - turn actions in recent history
        4. stuck_region - highly repetitive actions

        Args:
            current_pos: Current 3D position (x, y, z)
            prev_pos: Previous 3D position (x, y, z)
            action_history: List of recent action names
            semantic_info: Optional dict with semantic information like room_changed

        Returns:
            Key position type string or None if not a key position
        """
        # Priority 1: Check for stairs (height change)
        if self._detect_stairs(current_pos, prev_pos):
            return "stairs"

        # Priority 2: Check for room entrance (semantic change)
        if self._detect_room_entrance(semantic_info):
            return "room_entrance"

        # Priority 3: Check for junction (turn actions)
        if self._detect_junction(action_history):
            return "junction"

        # Priority 4: Check for stuck region (repetitive actions)
        if self._detect_stuck_region(action_history):
            return "stuck_region"

        return None

    def _detect_stairs(
        self,
        current_pos: Tuple[float, float, float],
        prev_pos: Tuple[float, float, float]
    ) -> bool:
        """Detect stairs based on significant height change.

        Args:
            current_pos: Current 3D position
            prev_pos: Previous 3D position

        Returns:
            True if stairs detected (height change > STAIRS_HEIGHT_THRESHOLD)
        """
        height_change = abs(current_pos[1] - prev_pos[1])
        return height_change > self.STAIRS_HEIGHT_THRESHOLD

    def _detect_room_entrance(self, semantic_info: Optional[Dict[str, Any]]) -> bool:
        """Detect room entrance based on semantic information.

        Args:
            semantic_info: Dict with semantic information

        Returns:
            True if room entrance detected
        """
        if semantic_info and semantic_info.get("room_changed"):
            return True
        return False

    def _detect_junction(self, action_history: List[str]) -> bool:
        """Detect junction based on turn actions in recent history.

        A junction is detected when:
        - At least JUNCTION_HISTORY_WINDOW recent actions exist
        - Contains at least one turn action
        - Actions are not all the same (indicating direction change)

        Args:
            action_history: List of action names

        Returns:
            True if junction detected
        """
        if len(action_history) >= self.JUNCTION_HISTORY_WINDOW:
            recent = action_history[-self.JUNCTION_HISTORY_WINDOW:]
            has_turn = any("turn" in a for a in recent)
            has_variation = any(a != recent[0] for a in recent)
            return has_turn and has_variation
        return False

    def _detect_stuck_region(self, action_history: List[str]) -> bool:
        """Detect stuck region based on highly repetitive actions.

        A stuck region is detected when:
        - At least STUCK_REGION_WINDOW recent actions exist
        - Only STUCK_UNIQUE_ACTIONS_MAX or fewer unique actions in recent history

        Args:
            action_history: List of action names

        Returns:
            True if stuck region detected
        """
        if action_history and len(action_history) >= self.STUCK_REGION_WINDOW:
            recent_actions = set(action_history[-self.STUCK_REGION_WINDOW:])
            return len(recent_actions) <= self.STUCK_UNIQUE_ACTIONS_MAX
        return False