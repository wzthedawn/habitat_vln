"""Navigation context definitions for VLN system."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np

from .action import Action


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
    level: str = "中等"  # 子任务难度等级 (简单/中等/困难)
    required_agents: List[str] = field(default_factory=list)
    dependencies: List[int] = field(default_factory=list)
    result: Optional[str] = None

    def __str__(self) -> str:
        return f"SubTask({self.id}: {self.description[:30]}... [{self.status}] [{self.level}])"


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
            self.current_subtask_idx += 1
            return True
        return False

    def get_action_summary(self, last_n: int = 5) -> str:
        """Get summary of recent actions."""
        if not self.action_history:
            return "No actions taken yet."

        recent = self.action_history[-last_n:]
        summary = [f"Step {i}: {a.action_type.name}" for i, a in enumerate(recent, 1)]
        return "\n".join(summary)


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