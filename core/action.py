"""Action definitions for VLN navigation system."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


class ActionType(Enum):
    """Navigation action types."""
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


@dataclass
class Action:
    """Represents a navigation action with metadata."""

    action_type: ActionType
    confidence: float = 1.0
    reasoning: Optional[str] = None
    predicted_position: Optional[Tuple[float, float, float]] = None
    predicted_rotation: Optional[float] = None

    def __str__(self) -> str:
        return f"Action({self.action_type.name}, conf={self.confidence:.2f})"

    def to_habitat_action(self) -> str:
        """Convert to Habitat-compatible action string."""
        action_map = {
            ActionType.STOP: "stop",
            ActionType.MOVE_FORWARD: "move_forward",
            ActionType.TURN_LEFT: "turn_left",
            ActionType.TURN_RIGHT: "turn_right",
            ActionType.LOOK_UP: "look_up",
            ActionType.LOOK_DOWN: "look_down",
        }
        return action_map[self.action_type]

    @classmethod
    def stop(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a stop action."""
        return cls(ActionType.STOP, confidence=confidence, reasoning=reasoning)

    @classmethod
    def forward(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a forward action."""
        return cls(ActionType.MOVE_FORWARD, confidence=confidence, reasoning=reasoning)

    @classmethod
    def turn_left(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a turn left action."""
        return cls(ActionType.TURN_LEFT, confidence=confidence, reasoning=reasoning)

    @classmethod
    def turn_right(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a turn right action."""
        return cls(ActionType.TURN_RIGHT, confidence=confidence, reasoning=reasoning)

    @classmethod
    def look_up(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a look up action."""
        return cls(ActionType.LOOK_UP, confidence=confidence, reasoning=reasoning)

    @classmethod
    def look_down(cls, confidence: float = 1.0, reasoning: Optional[str] = None) -> "Action":
        """Create a look down action."""
        return cls(ActionType.LOOK_DOWN, confidence=confidence, reasoning=reasoning)


@dataclass
class ActionSequence:
    """动作序列，用于子任务级别的规划。

    每个 ActionSequence 对应一个子任务，包含一系列动作及其重复次数。
    执行时按顺序执行每个动作块，直到完成或触发中断条件。
    """

    subtask_id: int                      # 对应的子任务ID
    subtask_description: str             # 子任务描述
    actions: List[Tuple[ActionType, int]]  # [(动作, 重复次数), ...]
    estimated_steps: int                 # 预估总步数

    # 执行状态
    current_index: int = 0               # 当前执行到第几个动作块
    current_repeat: int = 0              # 当前动作块内的重复次数
    executed_steps: int = 0              # 已执行步数

    # 中断条件
    abort_conditions: Dict[str, Any] = field(default_factory=dict)
    # 例如: {"obstacle_ahead": True, "max_stuck_steps": 3}

    # 检查点
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)

    # 生成信息
    reasoning: str = ""                  # LLM 生成的推理过程
    confidence: float = 0.8              # 置信度
    subtask_completed: bool = False      # LLM 判断子任务是否完成

    def get_next_action(self) -> Optional[ActionType]:
        """获取下一个动作，返回 None 表示序列完成"""
        if self.current_index >= len(self.actions):
            return None

        action_type, repeat_count = self.actions[self.current_index]

        if self.current_repeat >= repeat_count:
            # 当前动作块完成，移动到下一个
            self.current_index += 1
            self.current_repeat = 0
            return self.get_next_action()

        self.current_repeat += 1
        self.executed_steps += 1
        return action_type

    def peek_next_action(self) -> Optional[ActionType]:
        """查看下一个动作但不推进状态"""
        if self.current_index >= len(self.actions):
            return None

        action_type, repeat_count = self.actions[self.current_index]

        if self.current_repeat >= repeat_count:
            # 当前动作块完成，查看下一个
            next_index = self.current_index + 1
            if next_index >= len(self.actions):
                return None
            return self.actions[next_index][0]

        return action_type

    def is_complete(self) -> bool:
        """序列是否完成"""
        return self.current_index >= len(self.actions)

    def get_progress(self) -> float:
        """获取执行进度 (0.0 - 1.0)"""
        if self.estimated_steps == 0:
            return 0.0
        return min(1.0, self.executed_steps / self.estimated_steps)

    def reset(self) -> None:
        """重置序列状态"""
        self.current_index = 0
        self.current_repeat = 0
        self.executed_steps = 0

    def get_remaining_steps(self) -> int:
        """获取剩余步数"""
        total = sum(count for _, count in self.actions)
        return total - self.executed_steps

    def get_current_action_info(self) -> Dict[str, Any]:
        """获取当前动作信息"""
        if self.current_index >= len(self.actions):
            return {"action": None, "remaining_in_block": 0}

        action_type, repeat_count = self.actions[self.current_index]
        remaining = repeat_count - self.current_repeat
        return {
            "action": action_type.name,
            "remaining_in_block": remaining,
            "block_index": self.current_index,
            "total_blocks": len(self.actions)
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "subtask_id": self.subtask_id,
            "subtask_description": self.subtask_description,
            "actions": [(a.name, c) for a, c in self.actions],
            "estimated_steps": self.estimated_steps,
            "executed_steps": self.executed_steps,
            "progress": self.get_progress(),
            "is_complete": self.is_complete(),
            "abort_conditions": self.abort_conditions
        }

    def __str__(self) -> str:
        actions_str = ", ".join(f"{a.name}x{c}" for a, c in self.actions)
        return f"ActionSequence({self.subtask_description[:30]}...: [{actions_str}], progress={self.get_progress():.0%})"