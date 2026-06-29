"""TopologyGraph - Key node management for navigation.

This module provides a pure logic tool (no LLM) for managing key nodes
during navigation, including turn points, room entries, stairs, doors, and exits.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import math


class NodeType(Enum):
    """Types of key nodes in navigation topology."""

    TURN_POINT = "turn_point"  # Turn/decision point
    ROOM_ENTRY = "room_entry"  # Room entrance
    STAIRS_ENTRY = "stairs_entry"  # Stairs entrance
    DOOR = "door"  # Door
    EXIT = "exit"  # Exit point


@dataclass
class KeyNode:
    """Represents a key node in the navigation topology.

    Attributes:
        id: Unique identifier for the node
        position: 3D position (x, y, z)
        type: Type of the node
        rotation: Rotation angle at this position
        visited_count: Number of times this node was visited
        metadata: Additional metadata about the node
    """

    id: int
    position: List[float]
    type: NodeType
    rotation: float
    visited_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Represents an edge between two key nodes.

    Attributes:
        source_id: ID of the source node
        target_id: ID of the target node
        distance: Distance between nodes
    """

    source_id: int
    target_id: int
    distance: float


@dataclass
class TopologyGraph:
    """Topology graph for managing key nodes in navigation.

    This is a pure logic tool (no LLM) that manages critical navigation
    nodes such as turn points, room entries, stairs, doors, and exits.

    Attributes:
        nodes: List of KeyNode objects
        edges: List of Edge objects
        visited_positions: List of visited positions (grid-aligned)
        _node_counter: Internal counter for generating unique node IDs
        _grid_size: Grid size for position alignment
        _max_visited_positions: Maximum number of visited positions to keep
    """

    nodes: List[KeyNode] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    visited_positions: List[List[float]] = field(default_factory=list)
    _node_counter: int = field(default=0, repr=False)
    _grid_size: float = field(default=0.5, repr=False)
    _max_visited_positions: int = field(default=50, repr=False)

    def add_key_node(
        self,
        position: List[float],
        node_type: NodeType,
        rotation: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KeyNode:
        """Add a new key node to the topology graph.

        Args:
            position: 3D position (x, y, z)
            node_type: Type of the node
            rotation: Rotation angle at this position
            metadata: Optional additional metadata

        Returns:
            The newly created KeyNode
        """
        node = KeyNode(
            id=self._node_counter,
            position=position.copy() if isinstance(position, list) else list(position),
            type=node_type,
            rotation=rotation,
            visited_count=1,
            metadata=metadata if metadata is not None else {}
        )
        self._node_counter += 1
        self.nodes.append(node)
        return node

    def add_visited_position(self, position: List[float]) -> bool:
        """Add a visited position to the history.

        Positions are grid-aligned to avoid duplicates. Only the most recent
        50 positions are kept.

        Args:
            position: 3D position (x, y, z)

        Returns:
            True if the position was added, False if it was a duplicate
        """
        # Grid-align the position
        aligned_position = self._align_to_grid(position)

        # Check if already exists
        for existing_pos in self.visited_positions:
            if self._positions_equal(existing_pos, aligned_position):
                return False

        # Add new position
        self.visited_positions.append(aligned_position)

        # Keep only the most recent 50 positions
        if len(self.visited_positions) > self._max_visited_positions:
            self.visited_positions = self.visited_positions[-self._max_visited_positions:]

        return True

    def get_nearby_key_nodes(
        self,
        position: List[float],
        radius: float = 5.0
    ) -> List[KeyNode]:
        """Get key nodes within a specified radius of a position.

        Args:
            position: Center position (x, y, z)
            radius: Search radius in meters

        Returns:
            List of KeyNodes within the radius, sorted by distance (nearest first)
        """
        nearby = []
        for node in self.nodes:
            distance = self._compute_distance(position, node.position)
            if distance <= radius:
                nearby.append((node, distance))

        # Sort by distance (nearest first)
        nearby.sort(key=lambda x: x[1])
        return [node for node, _ in nearby]

    def find_nearest_exit(self, position: List[float]) -> Optional[KeyNode]:
        """Find the nearest exit node from a given position.

        Args:
            position: Current position (x, y, z)

        Returns:
            The nearest exit KeyNode, or None if no exits exist
        """
        exits = [node for node in self.nodes if node.type == NodeType.EXIT]
        if not exits:
            return None

        # Find the nearest exit
        nearest = None
        min_distance = float('inf')
        for exit_node in exits:
            distance = self._compute_distance(position, exit_node.position)
            if distance < min_distance:
                min_distance = distance
                nearest = exit_node

        return nearest

    def has_key_nodes(self) -> bool:
        """Check if there are any key nodes in the graph.

        Returns:
            True if there is at least one key node
        """
        return len(self.nodes) > 0

    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the topology graph.

        Returns:
            Dictionary containing graph summary information
        """
        # Count nodes by type
        node_types: Dict[str, int] = {}
        for node in self.nodes:
            type_name = node.type.value
            node_types[type_name] = node_types.get(type_name, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "visited_positions_count": len(self.visited_positions),
            "node_types": node_types,
            "nodes": [
                {
                    "id": node.id,
                    "position": node.position,
                    "type": node.type.value,
                    "rotation": node.rotation,
                    "visited_count": node.visited_count
                }
                for node in self.nodes
            ]
        }

    def _align_to_grid(self, position: List[float]) -> List[float]:
        """Align a position to the grid.

        Args:
            position: Original position (x, y, z)

        Returns:
            Grid-aligned position
        """
        return [
            round(coord / self._grid_size) * self._grid_size
            for coord in position
        ]

    def _positions_equal(self, pos1: List[float], pos2: List[float]) -> bool:
        """Check if two positions are equal.

        Args:
            pos1: First position
            pos2: Second position

        Returns:
            True if positions are equal
        """
        if len(pos1) != len(pos2):
            return False
        return all(abs(a - b) < 1e-9 for a, b in zip(pos1, pos2))

    def _compute_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Compute Euclidean distance between two positions.

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