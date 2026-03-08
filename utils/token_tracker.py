"""Token usage tracking and reporting for VLN system."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging


@dataclass
class TokenUsage:
    """Token usage record for a single API call."""
    agent_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    task_type: str = ""
    prompt_type: str = ""  # e.g., "classify", "perception", "decision"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp,
            "task_type": self.task_type,
            "prompt_type": self.prompt_type,
        }


@dataclass
class TaskTokenSummary:
    """Token usage summary for a single navigation task."""
    task_id: str
    task_type: str
    instruction: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    usage_records: List[TokenUsage] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.usage_records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.usage_records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.usage_records)

    @property
    def num_api_calls(self) -> int:
        return len(self.usage_records)

    def get_usage_by_agent(self) -> Dict[str, Dict[str, int]]:
        """Get token usage grouped by agent."""
        result = {}
        for record in self.usage_records:
            if record.agent_name not in result:
                result[record.agent_name] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                }
            result[record.agent_name]["input_tokens"] += record.input_tokens
            result[record.agent_name]["output_tokens"] += record.output_tokens
            result[record.agent_name]["total_tokens"] += record.total_tokens
            result[record.agent_name]["calls"] += 1
        return result

    def get_usage_by_model(self) -> Dict[str, Dict[str, int]]:
        """Get token usage grouped by model."""
        result = {}
        for record in self.usage_records:
            if record.model_name not in result:
                result[record.model_name] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                }
            result[record.model_name]["input_tokens"] += record.input_tokens
            result[record.model_name]["output_tokens"] += record.output_tokens
            result[record.model_name]["total_tokens"] += record.total_tokens
            result[record.model_name]["calls"] += 1
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "instruction": self.instruction,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "summary": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "num_api_calls": self.num_api_calls,
            },
            "by_agent": self.get_usage_by_agent(),
            "by_model": self.get_usage_by_model(),
            "records": [r.to_dict() for r in self.usage_records],
        }


class TokenTracker:
    """
    Global token usage tracker.

    Tracks token usage across all navigation tasks and provides
    reporting capabilities.
    """

    _instance: Optional["TokenTracker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logging.getLogger("TokenTracker")

        # Current task being tracked
        self._current_task: Optional[TaskTokenSummary] = None

        # Historical tasks
        self._task_history: List[TaskTokenSummary] = []

        # Global totals
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        # Task counter for generating IDs
        self._task_counter = 0

    def start_task(self, instruction: str, task_type: str = "unknown") -> str:
        """
        Start tracking a new navigation task.

        Args:
            instruction: Navigation instruction
            task_type: Task type (Type-0 to Type-4)

        Returns:
            Task ID
        """
        # End previous task if exists
        if self._current_task is not None:
            self.end_task()

        self._task_counter += 1
        task_id = f"task_{self._task_counter:04d}"

        self._current_task = TaskTokenSummary(
            task_id=task_id,
            task_type=task_type,
            instruction=instruction,
        )

        self.logger.info(f"Started tracking task: {task_id} (Type: {task_type})")
        return task_id

    def end_task(self) -> Optional[TaskTokenSummary]:
        """
        End current task tracking.

        Returns:
            Completed task summary
        """
        if self._current_task is None:
            return None

        self._current_task.end_time = datetime.now().isoformat()
        completed = self._current_task

        # Update global totals
        self._total_tokens += completed.total_tokens
        self._total_input_tokens += completed.total_input_tokens
        self._total_output_tokens += completed.total_output_tokens

        # Add to history
        self._task_history.append(completed)

        self.logger.info(
            f"Task {completed.task_id} completed: "
            f"{completed.total_tokens} tokens "
            f"({completed.num_api_calls} calls)"
        )

        self._current_task = None
        return completed

    def record_usage(
        self,
        agent_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        prompt_type: str = "",
    ) -> Optional[TokenUsage]:
        """
        Record token usage for an API call.

        Args:
            agent_name: Name of the agent making the call
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            prompt_type: Type of prompt (e.g., "classify", "perception")

        Returns:
            TokenUsage record
        """
        usage = TokenUsage(
            agent_name=agent_name,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            task_type=self._current_task.task_type if self._current_task else "",
            prompt_type=prompt_type,
        )

        if self._current_task is not None:
            self._current_task.usage_records.append(usage)

        self.logger.debug(
            f"Token usage: {agent_name}/{model_name} - "
            f"input={input_tokens}, output={output_tokens}, total={usage.total_tokens}"
        )

        return usage

    def get_current_task_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current task."""
        if self._current_task is None:
            return None
        return self._current_task.to_dict()

    def get_global_summary(self) -> Dict[str, Any]:
        """Get global token usage summary."""
        return {
            "total_tasks": len(self._task_history),
            "total_tokens": self._total_tokens,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "average_tokens_per_task": (
                self._total_tokens / len(self._task_history)
                if self._task_history else 0
            ),
        }

    def get_task_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task history."""
        tasks = self._task_history[-limit:]
        return [t.to_dict() for t in tasks]

    def print_current_task_report(self) -> str:
        """Print a formatted report of the current task."""
        if self._current_task is None:
            return "No task currently being tracked."

        task = self._current_task
        lines = [
            "=" * 60,
            f"TOKEN USAGE REPORT - {task.task_id}",
            "=" * 60,
            f"Task Type: {task.task_type}",
            f"Instruction: {task.instruction[:50]}...",
            f"Duration: {task.start_time}",
            "-" * 60,
            "TOKEN SUMMARY:",
            f"  Input Tokens:  {task.total_input_tokens:,}",
            f"  Output Tokens: {task.total_output_tokens:,}",
            f"  Total Tokens:  {task.total_tokens:,}",
            f"  API Calls:     {task.num_api_calls}",
            "-" * 60,
            "USAGE BY AGENT:",
        ]

        for agent, stats in task.get_usage_by_agent().items():
            lines.append(
                f"  {agent}: {stats['total_tokens']:,} tokens "
                f"({stats['calls']} calls)"
            )

        lines.append("-" * 60)
        lines.append("USAGE BY MODEL:")

        for model, stats in task.get_usage_by_model().items():
            lines.append(
                f"  {model}: {stats['total_tokens']:,} tokens "
                f"({stats['calls']} calls)"
            )

        lines.append("-" * 60)
        lines.append("DETAILED RECORDS:")

        for record in task.usage_records:
            lines.append(
                f"  [{record.agent_name}] {record.prompt_type or 'general'}: "
                f"in={record.input_tokens}, out={record.output_tokens}"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def print_global_report(self) -> str:
        """Print a formatted global report."""
        summary = self.get_global_summary()

        lines = [
            "=" * 60,
            "GLOBAL TOKEN USAGE SUMMARY",
            "=" * 60,
            f"Total Tasks Tracked: {summary['total_tasks']}",
            f"Total Tokens: {summary['total_tokens']:,}",
            f"  - Input:  {summary['total_input_tokens']:,}",
            f"  - Output: {summary['total_output_tokens']:,}",
            f"Average per Task: {summary['average_tokens_per_task']:,.1f}",
            "=" * 60,
        ]

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracking data."""
        self._current_task = None
        self._task_history = []
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._task_counter = 0
        self.logger.info("Token tracker reset")


# Global singleton instance
def get_token_tracker() -> TokenTracker:
    """Get the global token tracker instance."""
    return TokenTracker()