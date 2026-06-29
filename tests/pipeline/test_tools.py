"""Tests for pipeline tools."""

import pytest
from agents.pipeline.tools.topology_graph import TopologyGraph, NodeType, KeyNode
from agents.pipeline.tools.action_converter import ActionConverter
from core.action import ActionType


class TestTopologyGraph:
    """Tests for TopologyGraph class."""

    def test_topology_graph_init(self):
        """Test topology graph initialization."""
        graph = TopologyGraph()
        assert graph.nodes == []
        assert graph.edges == []
        assert len(graph.visited_positions) == 0

    def test_add_key_node(self):
        """Test adding key node."""
        graph = TopologyGraph()
        node = graph.add_key_node(
            position=[1.0, 0.0, 2.0],
            node_type=NodeType.TURN_POINT,
            rotation=0.5
        )
        assert len(graph.nodes) == 1
        assert node.position == [1.0, 0.0, 2.0]
        assert node.type == NodeType.TURN_POINT

    def test_get_nearby_key_nodes(self):
        """Test getting nearby key nodes."""
        graph = TopologyGraph()
        graph.add_key_node([0.0, 0.0, 0.0], NodeType.TURN_POINT, 0.0)
        graph.add_key_node([10.0, 0.0, 10.0], NodeType.ROOM_ENTRY, 0.0)

        nearby = graph.get_nearby_key_nodes([1.0, 0.0, 1.0], radius=5.0)
        assert len(nearby) == 1
        assert nearby[0].position == [0.0, 0.0, 0.0]

    def test_has_key_nodes(self):
        """Test checking if there are key nodes."""
        graph = TopologyGraph()
        assert not graph.has_key_nodes()
        graph.add_key_node([0.0, 0.0, 0.0], NodeType.TURN_POINT, 0.0)
        assert graph.has_key_nodes()

    def test_add_visited_position(self):
        """Test adding visited position."""
        graph = TopologyGraph()
        graph.add_visited_position([1.0, 0.0, 2.0])
        assert len(graph.visited_positions) == 1

        # Same position (grid aligned) should not be added again
        graph.add_visited_position([1.1, 0.0, 2.1])
        assert len(graph.visited_positions) == 1

    def test_find_nearest_exit(self):
        """Test finding nearest exit."""
        graph = TopologyGraph()
        # No exits initially
        assert graph.find_nearest_exit([0.0, 0.0, 0.0]) is None

        # Add an exit
        graph.add_key_node([5.0, 0.0, 5.0], NodeType.EXIT, 0.0)
        graph.add_key_node([10.0, 0.0, 10.0], NodeType.EXIT, 0.0)

        nearest = graph.find_nearest_exit([0.0, 0.0, 0.0])
        assert nearest is not None
        assert nearest.position == [5.0, 0.0, 5.0]

    def test_summary(self):
        """Test summary generation."""
        graph = TopologyGraph()
        graph.add_key_node([0.0, 0.0, 0.0], NodeType.TURN_POINT, 0.0)
        graph.add_key_node([5.0, 0.0, 5.0], NodeType.EXIT, 1.0)

        summary = graph.summary()
        assert "total_nodes" in summary
        assert summary["total_nodes"] == 2
        assert "node_types" in summary


class TestNodeType:
    """Tests for NodeType enum."""

    def test_node_types_exist(self):
        """Test all required node types exist."""
        assert NodeType.TURN_POINT.value == "turn_point"
        assert NodeType.ROOM_ENTRY.value == "room_entry"
        assert NodeType.STAIRS_ENTRY.value == "stairs_entry"
        assert NodeType.DOOR.value == "door"
        assert NodeType.EXIT.value == "exit"


class TestKeyNode:
    """Tests for KeyNode dataclass."""

    def test_key_node_creation(self):
        """Test creating a KeyNode."""
        node = KeyNode(
            id=1,
            position=[1.0, 2.0, 3.0],
            type=NodeType.TURN_POINT,
            rotation=0.5
        )
        assert node.id == 1
        assert node.position == [1.0, 2.0, 3.0]
        assert node.type == NodeType.TURN_POINT
        assert node.rotation == 0.5
        assert node.visited_count == 1
        assert node.metadata == {}

    def test_key_node_with_metadata(self):
        """Test KeyNode with custom metadata."""
        metadata = {"label": "corner", "confidence": 0.9}
        node = KeyNode(
            id=2,
            position=[0.0, 0.0, 0.0],
            type=NodeType.DOOR,
            rotation=0.0,
            metadata=metadata
        )
        assert node.metadata == metadata


class TestActionConverter:
    """Tests for ActionConverter class."""

    def test_action_converter_convert(self):
        """Test action name conversion."""
        converter = ActionConverter()
        actions = converter.convert(["forward", "turn_left", "forward"])

        assert len(actions) == 3
        assert actions[0] == (ActionType.MOVE_FORWARD, 1)
        assert actions[1] == (ActionType.TURN_LEFT, 1)
        assert actions[2] == (ActionType.MOVE_FORWARD, 1)

    def test_action_converter_ensure_5(self):
        """Test padding to 5 actions."""
        converter = ActionConverter()

        # Less than 5, pad with forward
        actions = converter.convert(["forward", "turn_left"])
        padded = converter.ensure_5_actions(actions)
        assert len(padded) == 5
        assert padded[0] == (ActionType.MOVE_FORWARD, 1)
        assert padded[1] == (ActionType.TURN_LEFT, 1)
        assert padded[2] == (ActionType.MOVE_FORWARD, 1)  # padded

        # More than 5, truncate
        actions = converter.convert(["forward"] * 7)
        truncated = converter.ensure_5_actions(actions)
        assert len(truncated) == 5

    def test_action_converter_generate_stop(self):
        """Test generating STOP action."""
        converter = ActionConverter()
        stop_actions = converter.generate_stop()

        assert len(stop_actions) == 1
        assert stop_actions[0] == (ActionType.STOP, 1)

    def test_action_converter_unknown_action(self):
        """Test unknown action maps to forward."""
        converter = ActionConverter()
        actions = converter.convert(["unknown_action", "forward"])

        assert actions[0] == (ActionType.MOVE_FORWARD, 1)  # fallback


class MockSubAgent:
    """Mock SubAgent for testing."""
    def __init__(self, name):
        self.name = name

    def process(self, input_data):
        return {"output": f"processed by {self.name}"}


class TestSubAgentRegistry:
    """Tests for SubAgentRegistry class."""

    def test_subagent_registry_register(self):
        """Test registering SubAgent."""
        from agents.pipeline.tools.subagent_registry import SubAgentRegistry
        registry = SubAgentRegistry()
        agent = MockSubAgent("test_agent")

        registry.register("test", agent)
        assert "test" in registry._agents
        assert registry._agents["test"].name == "test_agent"

    def test_subagent_registry_get(self):
        """Test getting SubAgent."""
        from agents.pipeline.tools.subagent_registry import SubAgentRegistry
        registry = SubAgentRegistry()
        agent = MockSubAgent("test_agent")
        registry.register("test", agent)

        retrieved = registry.get("test")
        assert retrieved.name == "test_agent"

        # Non-existent returns None
        assert registry.get("nonexistent") is None

    def test_subagent_registry_call(self):
        """Test calling SubAgent."""
        from agents.pipeline.tools.subagent_registry import SubAgentRegistry
        registry = SubAgentRegistry()
        agent = MockSubAgent("test_agent")
        registry.register("test", agent)

        result = registry.call("test", input_data="test_input")
        assert result["output"] == "processed by test_agent"

    def test_subagent_registry_list(self):
        """Test listing all SubAgents."""
        from agents.pipeline.tools.subagent_registry import SubAgentRegistry
        registry = SubAgentRegistry()
        registry.register("agent1", MockSubAgent("agent1"))
        registry.register("agent2", MockSubAgent("agent2"))

        agents = registry.list_all()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents

    def test_subagent_registry_clear(self):
        """Test clearing all SubAgents."""
        from agents.pipeline.tools.subagent_registry import SubAgentRegistry
        registry = SubAgentRegistry()
        registry.register("agent1", MockSubAgent("agent1"))
        registry.register("agent2", MockSubAgent("agent2"))

        registry.clear()
        assert len(registry.list_all()) == 0


class TestStateCalculator:
    """Tests for StateCalculator class."""

    def test_state_calculator_position_change(self):
        """Test position change calculation."""
        import math
        from agents.pipeline.tools.state_calculator import StateCalculator

        calc = StateCalculator()

        change = calc.compute_position_change(
            start_pos=[0.0, 0.0, 0.0],
            current_pos=[1.0, -1.5, 2.0]
        )

        assert change["dx"] == 1.0
        assert change["dy"] == -1.5
        assert change["dz"] == 2.0
        assert change["horizontal_dist"] == math.sqrt(1.0 * 1.0 + 2.0 * 2.0)

    def test_state_calculator_rotation_change(self):
        """Test rotation change calculation."""
        import math
        from agents.pipeline.tools.state_calculator import StateCalculator

        calc = StateCalculator()

        # Normal change
        delta = calc.compute_rotation_change(start_rot=0.0, current_rot=math.pi / 4)
        assert abs(delta - 45) < 1

        # Boundary handling (-180/180)
        delta = calc.compute_rotation_change(start_rot=math.pi, current_rot=-math.pi * 0.9)
        assert abs(delta) < 30

    def test_state_calculator_distance(self):
        """Test distance calculation."""
        from agents.pipeline.tools.state_calculator import StateCalculator

        calc = StateCalculator()

        dist = calc.compute_distance([0, 0, 0], [3, 0, 4])
        assert dist == 5.0

    def test_state_calculator_y_change_direction(self):
        """Test Y change direction judgment."""
        from agents.pipeline.tools.state_calculator import StateCalculator

        calc = StateCalculator()

        # Going down
        dy = -2.0
        assert calc.is_y_down(dy, threshold=1.5)
        assert not calc.is_y_up(dy, threshold=1.5)

        # Going up
        dy = 2.0
        assert calc.is_y_up(dy, threshold=1.5)
        assert not calc.is_y_down(dy, threshold=1.5)

    def test_state_calculator_distance_to_goal(self):
        """Test distance to goal calculation."""
        from agents.pipeline.tools.state_calculator import StateCalculator

        calc = StateCalculator()

        dist = calc.compute_distance_to_goal(
            current_pos=[0.0, 0.0, 0.0],
            goal_pos=[3.0, 0.0, 4.0]
        )
        assert dist == 5.0