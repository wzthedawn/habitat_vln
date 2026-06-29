"""Navigation context definitions for VLN system."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import time
import json

from .action import Action


# Precondition types for subtask verification
PRECONDITION_TYPES = {
    "height_reached": "Reached specified height",        # {"type": "height_reached", "target_y": -3.0, "tolerance": 1.0}
    "floor_changed": "Floor has changed",           # {"type": "floor_changed", "direction": "down"}
    "object_visible": "Target object visible",        # {"type": "object_visible", "object": "stairs"}
    "position_reached": "Reached specified position",      # {"type": "position_reached", "near": "stairs_bottom"}
    "rotation_completed": "Rotation completed",        # {"type": "rotation_completed", "direction": "right"}
}

# Completion condition types
COMPLETION_TYPES = {
    "y_change": "Height change",                  # {"type": "y_change", "min_change": 2.0, "direction": "down"}
    "rotation": "Rotation action",                  # {"type": "rotation", "direction": "right", "min_degrees": 60}
    "distance": "Distance moved",                  # {"type": "distance", "min_meters": 3.0}
    "object_near": "Approaching object",               # {"type": "object_near", "object": "bench", "max_distance": 2.0}
    "room_type": "Room type",                 # {"type": "room_type", "expected": "hallway"}
    "at_goal": "Reached goal",                   # {"type": "at_goal", "max_distance": 3.0}
}


class TaskType(Enum):
    """Task type classification based on complexity."""
    TYPE_0 = "Type-0"  # Simple navigation - single step
    TYPE_1 = "Type-1"  # Path following - corridor navigation
    TYPE_2 = "Type-2"  # Target search - object finding
    TYPE_3 = "Type-3"  # Spatial reasoning - cross-room
    TYPE_4 = "Type-4"  # Complex decision - ambiguous scenes


@dataclass
class StuckRegion:
    """Records a region where the agent got stuck."""
    position: Tuple[float, float, float]
    radius: float = 1.0
    entry_step: int = 0
    exit_step: Optional[int] = None
    escape_actions: List[str] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)


@dataclass
class PathOpinion:
    """Path planning opinion from an agent."""
    direction: str  # "left", "right", "forward", "backward"
    confidence: float = 0.5
    reason: str = ""
    stop_condition: str = ""  # When to stop (e.g., "move 2 meters")
    agent_source: str = ""  # Source agent name


@dataclass
class SubTask:
    """Represents a subtask decomposed from the main instruction."""

    id: int
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    level: str = "medium"  # Subtask difficulty level (easy/medium/hard)
    required_agents: List[str] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    result: Optional[str] = None

    # === New fields for semantic decomposition ===
    # 前置条件 (precondition for execution)
    precondition: Optional[Dict[str, Any]] = None
    # 示例: {"type": "height_change", "direction": "down", "min_change": 1.0}

    # 完成条件 (condition for completion verification)
    completion_condition: Optional[Dict[str, Any]] = None
    # 示例: {"type": "position", "check": "at_target_floor", "tolerance": 1.0}

    # 空间约束 (spatial constraint for execution)
    spatial_constraint: Optional[Dict[str, Any]] = None
    # 示例: {"floor": "lower", "near": "stairs_bottom"}

    # 执行上下文（运行时填充）
    start_context: Optional[Dict[str, Any]] = None  # 开始时的状态
    end_context: Optional[Dict[str, Any]] = None    # 完成时的状态

    def __str__(self) -> str:
        precondition_str = f", pre={self.precondition}" if self.precondition else ""
        completion_str = f", comp={self.completion_condition}" if self.completion_condition else ""
        return f"SubTask({self.id}: {self.description[:30]}... [{self.status}] [{self.level}]{precondition_str}{completion_str})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert subtask to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "level": self.level,
            "required_agents": self.required_agents,
            "dependencies": self.dependencies,
            "result": self.result,
            "precondition": self.precondition,
            "completion_condition": self.completion_condition,
            "spatial_constraint": self.spatial_constraint,
            "start_context": self.start_context,
            "end_context": self.end_context,
        }


@dataclass
class VisualFeatures:
    """Visual features extracted from observation."""

    rgb_embedding: Optional[Any] = None
    depth_embedding: Optional[Any] = None
    panorama_features: Optional[Dict[str, Any]] = None
    object_detections: Optional[List[Dict]] = None
    room_classification: Optional[str] = None
    scene_description: Optional[str] = None
    # Raw images for YOLO processing
    rgb_image: Optional[Any] = None
    depth_image: Optional[Any] = None


@dataclass
class NavContext:
    """Navigation context - core data structure of the system."""

    # Input
    instruction: str
    visual_features: VisualFeatures = field(default_factory=VisualFeatures)

    # Raw images (for YOLO processing)
    rgb_image: Optional[Any] = None
    depth_image: Optional[Any] = None

    # State
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: float = 0.0
    room_type: str = "unknown"
    step_count: int = 0

    # Task related
    task_type: TaskType = TaskType.TYPE_0
    subtasks: List[SubTask] = field(default_factory=list)
    current_subtask_idx: int = 0

    # History
    action_history: List[Action] = field(default_factory=list)
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)

    # NEW: Observation history for stuck analysis
    rgb_history: List[Any] = field(default_factory=list)
    depth_history: List[Any] = field(default_factory=list)
    stuck_regions: List[Dict] = field(default_factory=list)

    # Current stuck state
    is_stuck: bool = False
    stuck_counter: int = 0

    # Output
    current_action: Optional[Action] = None
    confidence: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_action(self, action: Action) -> None:
        """Add action to history and update state."""
        self.action_history.append(action)
        self.step_count += 1

    def add_trajectory_point(self, position: Tuple[float, float, float]) -> None:
        """Add position to trajectory."""
        self.trajectory.append(position)

    def add_decision(self, decision: Dict[str, Any]) -> None:
        """Add decision to history."""
        self.decision_history.append({
            "step": self.step_count,
            "decision": decision,
            "position": self.position,
        })

    def add_observation(self, rgb: Any, depth: Any, max_history: int = 20) -> None:
        """Add observation (RGB and depth) to history.

        Args:
            rgb: RGB image
            depth: Depth image
            max_history: Maximum history length (default 20)
        """
        self.rgb_history.append(rgb)
        self.depth_history.append(depth)

        # Limit history length
        if len(self.rgb_history) > max_history:
            self.rgb_history = self.rgb_history[-max_history:]
            self.depth_history = self.depth_history[-max_history:]

    def record_stuck_region(
        self,
        position: Tuple[float, float, float],
        radius: float = 1.0,
        escape_actions: List[str] = None,
        failed_attempts: List[str] = None
    ) -> None:
        """Record a stuck region.

        Args:
            position: Position where stuck occurred
            radius: Radius of stuck region
            escape_actions: Actions that successfully escaped
            failed_attempts: Actions that failed to escape
        """
        stuck_record = {
            "position": position,
            "radius": radius,
            "entry_step": self.step_count,
            "exit_step": None,
            "escape_actions": escape_actions or [],
            "failed_attempts": failed_attempts or [],
        }
        self.stuck_regions.append(stuck_record)

    def is_in_stuck_region(self, position: Tuple[float, float, float]) -> bool:
        """Check if position is within any known stuck region.

        Args:
            position: Position to check

        Returns:
            True if position is within a stuck region
        """
        import math
        for region in self.stuck_regions:
            dx = position[0] - region["position"][0]
            dz = position[2] - region["position"][2]
            dist = math.sqrt(dx*dx + dz*dz)
            if dist < region["radius"]:
                return True
        return False

    def get_current_subtask(self) -> Optional[SubTask]:
        """Get current subtask."""
        if 0 <= self.current_subtask_idx < len(self.subtasks):
            return self.subtasks[self.current_subtask_idx]
        return None

    def advance_subtask(self) -> bool:
        """Advance to next subtask. Returns True if successful."""
        if self.current_subtask_idx < len(self.subtasks) - 1:
            # Complete current subtask first
            current = self.get_current_subtask()
            if current and current.status == "in_progress":
                current.status = "completed"

            self.current_subtask_idx += 1

            # Start new subtask (record start state)
            new_subtask = self.get_current_subtask()
            if new_subtask:
                new_subtask.start_context = {
                    "position": self.position,
                    "rotation": self.rotation,
                    "y": self.position[1],
                    "step": self.step_count,
                    "timestamp": time.time(),
                }
                new_subtask.status = "in_progress"

            return True
        return False

    def get_action_summary(self, last_n: int = 5) -> str:
        """Get summary of recent actions."""
        if not self.action_history:
            return "No actions taken yet."

        recent = self.action_history[-last_n:]
        summary = [f"Step {i}: {a.action_type.name}" for i, a in enumerate(recent, 1)]
        return "\n".join(summary)

    # === Subtask lifecycle methods ===

    def start_subtask(self) -> None:
        """Record current subtask start state."""
        current = self.get_current_subtask()
        if current:
            current.start_context = {
                "position": self.position,
                "rotation": self.rotation,
                "y": self.position[1],
                "step": self.step_count,
                "timestamp": time.time(),
            }
            current.status = "in_progress"

    def complete_subtask(self) -> None:
        """Record current subtask completion state."""
        current = self.get_current_subtask()
        if current:
            current.end_context = {
                "position": self.position,
                "rotation": self.rotation,
                "y": self.position[1],
                "step": self.step_count,
            }
            current.status = "completed"

            # Calculate state changes
            if current.start_context:
                current.result = json.dumps({
                    "y_change": current.end_context["y"] - current.start_context["y"],
                    "rotation_change": current.end_context["rotation"] - current.start_context["rotation"],
                    "steps_taken": current.end_context["step"] - current.start_context["step"],
                })

    def get_subtask_y_change(self) -> float:
        """Get current subtask's height change."""
        current = self.get_current_subtask()
        if current and current.start_context:
            return self.position[1] - current.start_context["y"]
        return 0.0

    def get_subtask_rotation_change(self) -> float:
        """Get current subtask's rotation change in radians."""
        current = self.get_current_subtask()
        if current and current.start_context:
            return self.rotation - current.start_context["rotation"]
        return 0.0

    def get_subtask_distance_moved(self) -> float:
        """Get current subtask's horizontal distance moved."""
        current = self.get_current_subtask()
        if current and current.start_context:
            start_pos = current.start_context["position"]
            import math
            return math.sqrt(
                (self.position[0] - start_pos[0])**2 +
                (self.position[2] - start_pos[2])**2
            )
        return 0.0


class NavContextBuilder:
    """Builder for creating navigation context."""

    def __init__(self):
        self._instruction: str = ""
        self._visual_features: VisualFeatures = VisualFeatures()
        self._position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._rotation: float = 0.0
        self._metadata: Dict[str, Any] = {}
        self._rgb_image: Optional[Any] = None
        self._depth_image: Optional[Any] = None

    def with_instruction(self, instruction: str) -> "NavContextBuilder":
        """Set instruction."""
        self._instruction = instruction
        return self

    def with_visual_features(self, features: VisualFeatures) -> "NavContextBuilder":
        """Set visual features."""
        self._visual_features = features
        return self

    def with_position(self, position: Tuple[float, float, float]) -> "NavContextBuilder":
        """Set position."""
        self._position = position
        return self

    def with_rotation(self, rotation: float) -> "NavContextBuilder":
        """Set rotation."""
        self._rotation = rotation
        return self

    def with_metadata(self, metadata: Dict[str, Any]) -> "NavContextBuilder":
        """Set metadata."""
        self._metadata = metadata
        return self

    def with_rgb_image(self, rgb_image: Any) -> "NavContextBuilder":
        """Set RGB image."""
        self._rgb_image = rgb_image
        return self

    def with_depth_image(self, depth_image: Any) -> "NavContextBuilder":
        """Set depth image."""
        self._depth_image = depth_image
        return self

    def build(self) -> NavContext:
        """Build the navigation context."""
        context = NavContext(
            instruction=self._instruction,
            visual_features=self._visual_features,
            position=self._position,
            rotation=self._rotation,
            metadata=self._metadata,
            rgb_image=self._rgb_image,
            depth_image=self._depth_image,
        )
        context.add_trajectory_point(self._position)
        return context