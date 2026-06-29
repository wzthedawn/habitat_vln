"""Tests for topology_graph module.

This module tests the TopologyGraph implementation including:
- GraphNode and GraphEdge dataclass creation
- TopologyGraph.add_node functionality
- KeyPositionDetector.detect for junction detection
- TopologyGraph.prune_nodes for node compression
- TopologyGraph.get_summary output format
"""

import pytest
from agents.topology_graph import (
    GraphNode,
    GraphEdge,
    TopologyGraph,
    KeyPositionDetector,
)


class TestGraphNodeCreation:
    """Tests for GraphNode and GraphEdge creation."""

    def test_graph_node_creation(self):
        """Test creating a GraphNode with all required attributes."""
        node = GraphNode(
            node_id="node_0",
            position=(0.0, 0.0, 0.0),
            node_type="junction",
            timestamp=0,
            visit_count=1,
            semantic_info="test junction",
            is_key_position=False,
        )

        assert node.node_id == "node_0"
        assert node.position == (0.0, 0.0, 0.0)
        assert node.node_type == "junction"
        assert node.timestamp == 0
        assert node.visit_count == 1
        assert node.semantic_info == "test junction"
        assert node.is_key_position is False

    def test_graph_node_minimal(self):
        """Test creating GraphNode with minimal required fields."""
        node = GraphNode(
            node_id="node_1",
            position=(1.0, 2.0, 3.0),
            node_type="room_entrance",
            timestamp=5,
        )

        assert node.node_id == "node_1"
        assert node.position == (1.0, 2.0, 3.0)
        assert node.node_type == "room_entrance"
        assert node.visit_count == 1  # default value
        assert node.semantic_info is None  # default value
        assert node.is_key_position is False  # default value

    def test_graph_node_types(self):
        """Test creating GraphNodes with different node types."""
        node_types = ["junction", "room_entrance", "stairs", "stuck_region"]

        for node_type in node_types:
            node = GraphNode(
                node_id=f"node_{node_type}",
                position=(0.0, 0.0, 0.0),
                node_type=node_type,
                timestamp=0,
            )
            assert node.node_type == node_type

    def test_graph_edge_creation(self):
        """Test creating a GraphEdge with all attributes."""
        edge = GraphEdge(
            source_id="node_0",
            target_id="node_1",
            distance=5.0,
            action_sequence=["forward", "turn_left", "forward"],
            traversable=True,
        )

        assert edge.source_id == "node_0"
        assert edge.target_id == "node_1"
        assert edge.distance == 5.0
        assert edge.action_sequence == ["forward", "turn_left", "forward"]
        assert edge.traversable is True

    def test_graph_edge_minimal(self):
        """Test creating GraphEdge with minimal required fields."""
        edge = GraphEdge(
            source_id="node_0",
            target_id="node_1",
            distance=3.5,
        )

        assert edge.source_id == "node_0"
        assert edge.target_id == "node_1"
        assert edge.distance == 3.5
        assert edge.action_sequence == []  # default value
        assert edge.traversable is True  # default value

    def test_graph_edge_non_traversable(self):
        """Test creating a non-traversable GraphEdge."""
        edge = GraphEdge(
            source_id="node_0",
            target_id="node_1",
            distance=2.0,
            action_sequence=["forward"],
            traversable=False,
        )

        assert edge.traversable is False


class TestTopologyGraphAddNode:
    """Tests for TopologyGraph.add_node functionality."""

    def test_topology_graph_add_node(self):
        """Test adding a single node to the topology graph."""
        graph = TopologyGraph()
        node_id = graph.add_node(position=(1.0, 0.0, 2.0), node_type="junction")

        assert len(graph.nodes) == 1
        assert node_id == "node_0"
        assert "node_0" in graph.nodes

    def test_topology_graph_add_multiple_nodes(self):
        """Test adding multiple nodes to the topology graph."""
        graph = TopologyGraph()

        node_id_0 = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        node_id_1 = graph.add_node(position=(1.0, 0.0, 1.0), node_type="room_entrance")
        node_id_2 = graph.add_node(position=(2.0, 0.0, 2.0), node_type="stairs")

        assert len(graph.nodes) == 3
        assert node_id_0 == "node_0"
        assert node_id_1 == "node_1"
        assert node_id_2 == "node_2"

    def test_topology_graph_add_node_with_semantic_info(self):
        """Test adding a node with semantic information."""
        graph = TopologyGraph()
        node_id = graph.add_node(
            position=(0.0, 0.0, 0.0),
            node_type="room_entrance",
            semantic_info="Kitchen entrance",
        )

        node = graph.nodes[node_id]
        assert node.semantic_info == "Kitchen entrance"

    def test_topology_graph_add_node_with_timestamp(self):
        """Test adding a node with a specific timestamp."""
        graph = TopologyGraph()
        node_id = graph.add_node(
            position=(0.0, 0.0, 0.0),
            node_type="junction",
            timestamp=42,
        )

        node = graph.nodes[node_id]
        assert node.timestamp == 42

    def test_topology_graph_add_edge(self):
        """Test adding an edge between two nodes."""
        graph = TopologyGraph()

        source_id = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        target_id = graph.add_node(position=(1.0, 0.0, 0.0), node_type="junction")

        edge_key = graph.add_edge(
            source_id=source_id,
            target_id=target_id,
            action_sequence=["forward", "forward"],
        )

        assert len(graph.edges) == 1
        assert edge_key == f"edge_{source_id}_{target_id}"
        assert graph.edges[edge_key].distance > 0

    def test_topology_graph_add_edge_nonexistent_node(self):
        """Test that adding an edge with nonexistent node raises ValueError."""
        graph = TopologyGraph()
        node_id = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")

        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge(
                source_id=node_id,
                target_id="nonexistent_node",
                action_sequence=["forward"],
            )

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge(
                source_id="nonexistent_node",
                target_id=node_id,
                action_sequence=["forward"],
            )


class TestKeyPositionDetector:
    """Tests for KeyPositionDetector.detect functionality."""

    def test_key_detector_junction(self):
        """Test detecting a junction from action history with turns."""
        detector = KeyPositionDetector()
        result = detector.detect(
            current_pos=(1, 0, 1),
            prev_pos=(0, 0, 0),
            action_history=["forward", "turn_left", "forward", "turn_right"],
            semantic_info={},
        )

        assert result == "junction"

    def test_key_detector_junction_with_turn_actions(self):
        """Test junction detection requires both turn and variation."""
        detector = KeyPositionDetector()

        # Turn action with variation -> junction
        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=["forward", "turn_left", "forward"],
            semantic_info={},
        )
        assert result == "junction"

    def test_key_detector_no_junction_without_variation(self):
        """Test that uniform actions don't trigger junction detection."""
        detector = KeyPositionDetector()

        # All same actions - no variation
        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=["forward", "forward", "forward"],
            semantic_info={},
        )
        assert result != "junction"

    def test_key_detector_stairs(self):
        """Test detecting stairs from height change."""
        detector = KeyPositionDetector()

        # Significant height change (>0.5m)
        result = detector.detect(
            current_pos=(0, 1.0, 0),  # Y changed by 1.0m
            prev_pos=(0, 0.0, 0),
            action_history=["forward"],
            semantic_info={},
        )

        assert result == "stairs"

    def test_key_detector_stairs_negative_height(self):
        """Test detecting stairs when going down."""
        detector = KeyPositionDetector()

        result = detector.detect(
            current_pos=(0, 0.0, 0),
            prev_pos=(0, 1.0, 0),  # Went down 1.0m
            action_history=["forward"],
            semantic_info={},
        )

        assert result == "stairs"

    def test_key_detector_no_stairs_small_change(self):
        """Test that small height changes don't trigger stairs detection."""
        detector = KeyPositionDetector()

        # Height change < 0.5m threshold
        result = detector.detect(
            current_pos=(0, 0.3, 0),
            prev_pos=(0, 0.0, 0),
            action_history=["forward"],
            semantic_info={},
        )

        assert result != "stairs"

    def test_key_detector_room_entrance(self):
        """Test detecting room entrance from semantic info."""
        detector = KeyPositionDetector()

        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=["forward"],
            semantic_info={"room_changed": True},
        )

        assert result == "room_entrance"

    def test_key_detector_stuck_region(self):
        """Test detecting stuck region from repetitive actions."""
        detector = KeyPositionDetector()

        # 10 actions with only 2 unique types, no turn actions to avoid junction detection
        # stuck_region has lower priority than junction, so we need to avoid junction triggers
        stuck_actions = ["forward", "stop"] * 5  # 10 actions, 2 unique, no turns

        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=stuck_actions,
            semantic_info={},
        )

        assert result == "stuck_region"

    def test_key_detector_no_stuck_with_variety(self):
        """Test that varied actions don't trigger stuck detection."""
        detector = KeyPositionDetector()

        # 10 actions with 5 unique types (not stuck)
        varied_actions = ["forward", "turn_left", "turn_right", "look_up", "look_down"] * 2

        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=varied_actions,
            semantic_info={},
        )

        assert result != "stuck_region"

    def test_key_detector_priority_stairs_over_room(self):
        """Test that stairs has higher priority than room_entrance."""
        detector = KeyPositionDetector()

        # Both stairs and room_entrance conditions met
        result = detector.detect(
            current_pos=(0, 1.0, 0),
            prev_pos=(0, 0.0, 0),
            action_history=["forward"],
            semantic_info={"room_changed": True},
        )

        assert result == "stairs"

    def test_key_detector_priority_room_over_junction(self):
        """Test that room_entrance has higher priority than junction."""
        detector = KeyPositionDetector()

        # Both room_entrance and junction conditions met
        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=["forward", "turn_left", "forward"],
            semantic_info={"room_changed": True},
        )

        assert result == "room_entrance"

    def test_key_detector_no_key_position(self):
        """Test returning None when no key position is detected."""
        detector = KeyPositionDetector()

        result = detector.detect(
            current_pos=(0, 0, 0),
            prev_pos=(0, 0, 0),
            action_history=["forward"],
            semantic_info={},
        )

        assert result is None


class TestTopologyGraphPrune:
    """Tests for TopologyGraph.prune_nodes functionality."""

    def test_topology_graph_prune(self):
        """Test pruning reduces nodes to MAX_NODES or fewer."""
        graph = TopologyGraph()

        # Add 35 nodes (exceeds MAX_NODES=30)
        for i in range(35):
            graph.add_node(
                position=(float(i), 0.0, 0.0),
                node_type="junction",
            )

        # Add edges to make nodes connected (prevent orphan removal)
        node_ids = list(graph.nodes.keys())
        for i in range(len(node_ids) - 1):
            graph.add_edge(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                action_sequence=["forward"],
            )

        graph.prune_nodes()

        assert len(graph.nodes) <= 30

    def test_topology_graph_prune_preserves_key_positions(self):
        """Test that key positions are preserved during pruning."""
        graph = TopologyGraph()

        # Add nodes with different types
        for i in range(35):
            node_type = "junction" if i < 30 else "room_entrance"
            graph.add_node(
                position=(float(i), 0.0, 0.0),
                node_type=node_type,
            )

        # Add edges
        node_ids = list(graph.nodes.keys())
        for i in range(len(node_ids) - 1):
            graph.add_edge(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                action_sequence=["forward"],
            )

        # Set goal node (makes it a key position)
        graph.goal_node_id = node_ids[0]

        graph.prune_nodes()

        # Goal node should be preserved
        assert node_ids[0] in graph.nodes
        # Room entrance nodes should be preserved (key positions)
        room_entrance_nodes = [
            nid for nid, n in graph.nodes.items() if n.node_type == "room_entrance"
        ]
        assert len(room_entrance_nodes) == 5

    def test_topology_graph_prune_no_op_when_within_limit(self):
        """Test that pruning does nothing when nodes <= MAX_NODES."""
        graph = TopologyGraph()

        # Add only 10 nodes (within limit)
        for i in range(10):
            graph.add_node(
                position=(float(i), 0.0, 0.0),
                node_type="junction",
            )

        graph.prune_nodes()

        # All nodes should still be present
        assert len(graph.nodes) == 10

    def test_topology_graph_prune_high_visit_count_nodes(self):
        """Test that nodes with high visit count become key positions."""
        graph = TopologyGraph()

        # Add nodes
        for i in range(35):
            graph.add_node(
                position=(float(i), 0.0, 0.0),
                node_type="junction",
            )

        # Add edges
        node_ids = list(graph.nodes.keys())
        for i in range(len(node_ids) - 1):
            graph.add_edge(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                action_sequence=["forward"],
            )

        # Set high visit count on specific node
        graph.nodes[node_ids[5]].visit_count = 5

        graph.prune_nodes()

        # Node with high visit count should be preserved and marked as key
        assert node_ids[5] in graph.nodes, "High-visit-count node should not be pruned"
        assert graph.nodes[node_ids[5]].is_key_position is True


class TestTopologyGraphGetSummary:
    """Tests for TopologyGraph.get_summary output format."""

    def test_topology_graph_get_summary(self):
        """Test get_summary returns correct format."""
        graph = TopologyGraph()
        summary = graph.get_summary()

        assert "total_nodes" in summary
        assert "key_nodes" in summary
        assert isinstance(summary["key_nodes"], list)

    def test_topology_graph_get_summary_fields(self):
        """Test get_summary contains all required fields."""
        graph = TopologyGraph()
        summary = graph.get_summary()

        # Required fields
        assert "total_nodes" in summary
        assert "key_nodes" in summary
        assert "current_node" in summary
        assert "goal_node" in summary
        assert "path_to_goal" in summary
        assert "visited_rooms" in summary
        assert "stuck_regions" in summary

    def test_topology_graph_get_summary_with_nodes(self):
        """Test get_summary with actual nodes."""
        graph = TopologyGraph()

        # Add a key position node
        node_id = graph.add_node(
            position=(1.0, 0.0, 0.0),
            node_type="junction",
            semantic_info="test room",
        )
        graph.nodes[node_id].is_key_position = True
        graph.current_node_id = node_id

        summary = graph.get_summary()

        assert summary["total_nodes"] == 1
        assert len(summary["key_nodes"]) == 1
        assert summary["key_nodes"][0]["id"] == node_id
        assert summary["current_node"] == node_id

    def test_topology_graph_get_summary_stuck_regions(self):
        """Test get_summary correctly identifies stuck regions."""
        graph = TopologyGraph()

        # Add stuck region nodes
        stuck_id = graph.add_node(
            position=(0.0, 0.0, 0.0),
            node_type="stuck_region",
        )

        # Add regular node
        junction_id = graph.add_node(
            position=(1.0, 0.0, 0.0),
            node_type="junction",
        )

        summary = graph.get_summary()

        assert len(summary["stuck_regions"]) == 1
        assert summary["stuck_regions"][0]["id"] == stuck_id

    def test_topology_graph_get_summary_visited_rooms(self):
        """Test get_summary correctly extracts visited rooms."""
        graph = TopologyGraph()

        # Add room entrance nodes with semantic info
        graph.add_node(
            position=(0.0, 0.0, 0.0),
            node_type="room_entrance",
            semantic_info="Kitchen",
        )
        graph.add_node(
            position=(1.0, 0.0, 0.0),
            node_type="room_entrance",
            semantic_info="Living Room",
        )
        # Regular junction without room info
        graph.add_node(
            position=(2.0, 0.0, 0.0),
            node_type="junction",
        )

        summary = graph.get_summary()

        assert "Kitchen" in summary["visited_rooms"]
        assert "Living Room" in summary["visited_rooms"]

    def test_topology_graph_get_summary_path_to_goal(self):
        """Test get_summary returns path_to_goal."""
        graph = TopologyGraph()

        # Add connected nodes
        node_0 = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        node_1 = graph.add_node(position=(1.0, 0.0, 0.0), node_type="junction")
        node_2 = graph.add_node(position=(2.0, 0.0, 0.0), node_type="junction")

        # Add edges
        graph.add_edge(node_0, node_1, ["forward"])
        graph.add_edge(node_1, node_2, ["forward"])

        # Set current and goal
        graph.current_node_id = node_0
        graph.goal_node_id = node_2

        summary = graph.get_summary()

        assert summary["path_to_goal"] == [node_0, node_1, node_2]

    def test_topology_graph_get_summary_json_serializable(self):
        """Test that get_summary output is JSON serializable."""
        import json

        graph = TopologyGraph()

        # Add various nodes
        graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        graph.add_node(
            position=(1.0, 0.0, 0.0),
            node_type="room_entrance",
            semantic_info="Test Room",
        )

        summary = graph.get_summary()

        # Should not raise exception
        json_str = json.dumps(summary)
        assert json_str is not None


class TestTopologyGraphIntegration:
    """Integration tests for TopologyGraph with KeyPositionDetector."""

    def test_full_workflow(self):
        """Test complete workflow of node creation and detection."""
        graph = TopologyGraph()
        detector = KeyPositionDetector()

        # Simulate navigation with position changes
        positions = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), ["forward"]),
            ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), ["forward", "turn_left"]),
            ((1.0, 0.0, 1.0), (1.0, 0.0, 0.0), ["forward", "turn_left", "forward"]),
        ]

        for curr, prev, actions in positions:
            node_type = detector.detect(
                current_pos=curr,
                prev_pos=prev,
                action_history=actions,
                semantic_info={},
            )
            if node_type:
                graph.add_node(position=curr, node_type=node_type)

        # Should have detected some key positions
        assert len(graph.nodes) >= 1
        assert graph.get_summary()["total_nodes"] >= 1

    def test_update_current_node(self):
        """Test updating current node based on position."""
        graph = TopologyGraph()

        # Add nodes
        node_0 = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        node_1 = graph.add_node(position=(5.0, 0.0, 5.0), node_type="junction")

        # Update to position near node_0
        result = graph.update_current_node((0.5, 0.0, 0.5))

        assert result == node_0
        assert graph.current_node_id == node_0
        assert graph.nodes[node_0].visit_count == 2  # Incremented

    def test_update_current_node_threshold(self):
        """Test update_current_node threshold behavior at 2.0m."""
        graph = TopologyGraph()
        n0 = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        graph.add_node(position=(10.0, 0.0, 10.0), node_type="junction")

        # Within threshold (2m) - should increment visit_count
        graph.update_current_node((1.0, 0.0, 0.0))  # 1m from n0, < 2m threshold
        assert graph.nodes[n0].visit_count == 2

        # Outside threshold - should not increment
        graph.update_current_node((5.0, 0.0, 5.0))  # > 2m from any node
        assert graph.nodes[n0].visit_count == 2

    def test_set_goal_node(self):
        """Test setting goal node based on position."""
        graph = TopologyGraph()

        # Add nodes
        node_0 = graph.add_node(position=(0.0, 0.0, 0.0), node_type="junction")
        node_1 = graph.add_node(position=(10.0, 0.0, 10.0), node_type="junction")

        # Set goal near node_1
        result = graph.set_goal_node((9.5, 0.0, 9.5))

        assert result == node_1
        assert graph.goal_node_id == node_1
        assert graph.nodes[node_1].is_key_position is True


class TestUpdateTopologyOnly:
    """Tests for TrajectoryAgent.update_topology_only method."""

    @pytest.fixture
    def trajectory_agent(self):
        """Create TrajectoryAgent instance for testing."""
        from agents.trajectory_agent import TrajectoryAgent
        return TrajectoryAgent(config={})

    def test_stairs_up_detection(self, trajectory_agent):
        """Test stairs detection when y increases > 0.5m."""
        current_pos = (0.0, 1.0, 0.0)  # y increased by 1.0
        prev_pos = (0.0, 0.0, 0.0)

        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=0.0,
            prev_rot=0.0,
            step_count=10
        )

        assert new_node_id is not None
        assert new_node_id in trajectory_agent.topology_graph.nodes
        node = trajectory_agent.topology_graph.nodes[new_node_id]
        assert node.node_type == "stairs"
        assert "上升" in node.semantic_info

    def test_stairs_down_detection(self, trajectory_agent):
        """Test stairs detection when y decreases > 0.5m."""
        current_pos = (0.0, 0.0, 0.0)  # y decreased by 1.0
        prev_pos = (0.0, 1.0, 0.0)

        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=0.0,
            prev_rot=0.0,
            step_count=10
        )

        assert new_node_id is not None
        node = trajectory_agent.topology_graph.nodes[new_node_id]
        assert node.node_type == "stairs"
        assert "下降" in node.semantic_info

    def test_junction_detection(self, trajectory_agent):
        """Test junction detection when rotation > 28 degrees."""
        import math
        current_pos = (1.0, 0.0, 1.0)
        prev_pos = (0.0, 0.0, 0.0)
        current_rot = math.pi / 2  # 90 degrees
        prev_rot = 0.0

        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=current_rot,
            prev_rot=prev_rot,
            step_count=10
        )

        assert new_node_id is not None
        node = trajectory_agent.topology_graph.nodes[new_node_id]
        assert node.node_type == "junction"

    def test_rotation_boundary_handling(self, trajectory_agent):
        """Test rotation handles -pi/pi boundary correctly."""
        import math
        current_pos = (1.0, 0.0, 0.0)
        prev_pos = (0.0, 0.0, 0.0)
        # Near boundary: -170 deg to 170 deg = 20 deg actual difference
        current_rot = -170 * math.pi / 180
        prev_rot = 170 * math.pi / 180

        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=current_rot,
            prev_rot=prev_rot,
            step_count=10
        )

        # Should NOT detect junction (20 deg < 28 deg threshold)
        assert new_node_id is None

    def test_no_key_position(self, trajectory_agent):
        """Test returns None when no key position change."""
        current_pos = (0.5, 0.0, 0.5)  # Small movement, no y change
        prev_pos = (0.0, 0.0, 0.0)

        new_node_id = trajectory_agent.update_topology_only(
            current_pos=current_pos,
            prev_pos=prev_pos,
            current_rot=0.0,
            prev_rot=0.0,
            step_count=10
        )

        assert new_node_id is None