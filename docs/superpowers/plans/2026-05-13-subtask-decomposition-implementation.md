# SubtaskDecompositionAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现导航 instruction 子任务分解功能，将长指令分解为多个可验证的子任务

**Architecture:** 新增 SubtaskDecompositionAgent（LLM 驱动），Navigator 在 initialize_episode 时调用，分解后存入 _subtasks 列表，ReviewAgent 判断完成后自动切换下一个子任务

**Tech Stack:** Python, LLM (RemoteClient via ModelManager), dataclasses

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `agents/pipeline/base_pipeline_agent.py` | 修改 | 添加 DecompositionOutput dataclass |
| `agents/pipeline/subtask_decomposition_agent.py` | 新增 | LLM 驱动的子任务分解 Agent |
| `agents/pipeline/navigator.py` | 修改 | 集成分解调用 + 子任务切换逻辑 |
| `run_vln_experiment.py` | 修改 | 传递 goal_position 给 Navigator |
| `tests/pipeline/test_subtask_decomposition_agent.py` | 新增 | 单元测试 |

---

### Task 1: 添加 DecompositionOutput dataclass

**Files:**
- Modify: `agents/pipeline/base_pipeline_agent.py:75-88` (在 EmergencyEvent 后添加)

- [ ] **Step 1: 在 base_pipeline_agent.py 添加 DecompositionOutput**

在 EmergencyEvent dataclass 后添加新 dataclass:

```python
@dataclass
class DecompositionOutput:
    """Subtask Decomposition Agent output - instruction decomposition result.

    Contains the list of decomposed subtasks and reasoning for the decomposition.
    """

    subtasks: List[Dict[str, Any]]  # List of subtask dicts
    reasoning: str = ""  # Decomposition reasoning explanation
```

- [ ] **Step 2: 验证语法正确**

Run: `python -c "from agents.pipeline.base_pipeline_agent import DecompositionOutput; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agents/pipeline/base_pipeline_agent.py
git commit -m "feat: add DecompositionOutput dataclass for subtask decomposition"
```

---

### Task 2: 创建 SubtaskDecompositionAgent

**Files:**
- Create: `agents/pipeline/subtask_decomposition_agent.py`
- Test: `tests/pipeline/test_subtask_decomposition_agent.py`

- [ ] **Step 1: 创建测试文件 tests/pipeline/test_subtask_decomposition_agent.py**

```python
"""Unit tests for SubtaskDecompositionAgent."""

import pytest
from unittest.mock import Mock, patch
from agents.pipeline.subtask_decomposition_agent import SubtaskDecompositionAgent
from agents.pipeline.base_pipeline_agent import DecompositionOutput


class TestSubtaskDecompositionAgent:
    """Test SubtaskDecompositionAgent functionality."""

    def test_agent_creation(self):
        """Test agent can be created."""
        agent = SubtaskDecompositionAgent()
        assert agent.name == "subtask_decomposition_agent"

    def test_process_returns_decomposition_output(self):
        """Test process returns DecompositionOutput."""
        agent = SubtaskDecompositionAgent()
        # Mock model manager
        mock_mm = Mock()
        mock_mm.generate_sync.return_value = """
        {
            "subtasks": [
                {"id": 1, "description": "Find stairs and go down", "completion_condition": {"type": "y_change", "direction": "down", "threshold": 1.0}},
                {"id": 2, "description": "Turn right", "completion_condition": {"type": "rotation", "direction": "right", "threshold": 45}}
            ],
            "reasoning": "Instruction has 2 stages"
        }
        """
        agent.set_model_manager(mock_mm)

        result = agent.process(
            instruction="Walk down the stairs and turn right",
            goal_position=[10.0, -3.0, 2.0],
            start_position=[5.0, -1.0, 3.0],
        )

        assert isinstance(result, DecompositionOutput)
        assert len(result.subtasks) == 2
        assert result.subtasks[0]["id"] == 1
        assert result.subtasks[0]["completion_condition"]["type"] == "y_change"

    def test_fallback_on_llm_failure(self):
        """Test fallback when LLM fails."""
        agent = SubtaskDecompositionAgent()
        agent.set_model_manager(None)  # No model manager

        result = agent.process(
            instruction="Walk down the stairs",
            goal_position=[10.0, -3.0, 2.0],
            start_position=[5.0, -1.0, 3.0],
        )

        assert isinstance(result, DecompositionOutput)
        assert len(result.subtasks) == 1
        assert result.subtasks[0]["completion_condition"]["type"] == "distance_to_goal"

    def test_parse_valid_json_response(self):
        """Test parsing valid JSON response."""
        agent = SubtaskDecompositionAgent()

        response = """
        ```json
        {
            "subtasks": [
                {"id": 1, "description": "desc1", "completion_condition": {"type": "y_change"}},
                {"id": 2, "description": "desc2", "completion_condition": {"type": "rotation"}}
            ],
            "reasoning": "test"
        }
        ```
        """

        result = agent._parse_response(response)
        assert len(result["subtasks"]) == 2
        assert result["reasoning"] == "test"

    def test_parse_malformed_response_fallback(self):
        """Test fallback for malformed response."""
        agent = SubtaskDecompositionAgent()

        response = "This is not JSON at all"

        result = agent._parse_response(response, fallback_instruction="Walk forward")
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["description"] == "Walk forward"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/pipeline/test_subtask_decomposition_agent.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'agents.pipeline.subtask_decomposition_agent')

- [ ] **Step 3: 创建 agents/pipeline/subtask_decomposition_agent.py**

```python
"""SubtaskDecompositionAgent - LLM-driven instruction decomposition.

Decomposes navigation instruction into multiple subtasks with
verifiable completion conditions.

Responsibilities:
- Parse instruction text
- Identify subtask boundaries (stairs, turns, object approach)
- Generate completion_condition for each subtask
"""

import json
import re
import logging
from typing import Dict, Any, List

from agents.pipeline.base_pipeline_agent import SubAgent, DecompositionOutput
from agents.base_agent import AgentRole


class SubtaskDecompositionAgent(SubAgent):
    """Subtask Decomposition Agent - LLM core.

    Uses LLM to decompose navigation instruction into multiple subtasks.
    Each subtask has:
    - id: Sequential number
    - description: Clear action description with implicit direction
    - completion_condition: Verifiable condition type
    """

    name = "subtask_decomposition_agent"

    @property
    def role(self) -> AgentRole:
        """Get agent role."""
        return AgentRole.DECISION

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SubtaskDecompositionAgent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.logger = logging.getLogger("SubtaskDecompositionAgent")

    def process(
        self,
        instruction: str,
        goal_position: List[float],
        start_position: List[float],
    ) -> DecompositionOutput:
        """Decompose instruction into subtasks.

        Args:
            instruction: Full navigation instruction text
            goal_position: Final goal position [x, y, z]
            start_position: Starting position [x, y, z]

        Returns:
            DecompositionOutput with subtasks list and reasoning
        """
        # Fallback if no model manager
        if self._model_manager is None:
            self.logger.warning("[SubtaskDecompositionAgent] No ModelManager, using fallback")
            return self._fallback_decomposition(instruction, goal_position)

        # Build prompt
        prompt = self._build_prompt(instruction, goal_position, start_position)

        # Call LLM
        try:
            response = self._call_llm(prompt, max_tokens=500, temperature=0.3)
            self.logger.info(f"[SubtaskDecompositionAgent] LLM response: {response[:200]}")

            # Parse response
            result = self._parse_response(response, fallback_instruction=instruction)

            return DecompositionOutput(
                subtasks=result["subtasks"],
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            self.logger.warning(f"[SubtaskDecompositionAgent] LLM call failed: {e}")
            return self._fallback_decomposition(instruction, goal_position)

    def _build_prompt(
        self,
        instruction: str,
        goal_position: List[float],
        start_position: List[float],
    ) -> str:
        """Build decomposition prompt.

        Args:
            instruction: Navigation instruction
            goal_position: Goal position
            start_position: Start position

        Returns:
            Prompt string for LLM
        """
        # Calculate initial elevation difference to help LLM understand context
        elevation_diff = goal_position[1] - start_position[1]

        prompt = f"""You are decomposing a navigation instruction into subtasks for a robot.

## Input
- Instruction: {instruction}
- Start position: {start_position} (x, y, z - y is elevation)
- Goal position: {goal_position}
- Elevation difference: {elevation_diff:.2f} (positive means goal is higher)

## Task
Break the instruction into 2-6 subtasks. Each subtask should:
1. Have a clear description with implicit action direction (e.g., "Find stairs and go down" not just "stairs")
2. Have a verifiable completion_condition

## Completion Condition Types
1. y_change: Elevation change (stairs/ramps)
   Example: {"type": "y_change", "direction": "down", "threshold": 1.0}

2. rotation: Turning action
   Example: {"type": "rotation", "direction": "right", "threshold": 45}

3. near_object: Approaching an object
   Example: {"type": "near_object", "target": "rug", "distance": 3.0}

4. distance_to_goal: Final distance to goal
   Example: {"type": "distance_to_goal", "threshold": 3.0}

5. combined: Multiple conditions (for final subtask)
   Example: {"type": "combined", "conditions": [{"type": "near_object", "target": "piano"}, {"type": "distance_to_goal", "threshold": 3.0}]}

## Output Format (JSON only, no other text)
```json
{
  "subtasks": [
    {"id": 1, "description": "...", "completion_condition": {...}},
    {"id": 2, "description": "...", "completion_condition": {...}}
  ],
  "reasoning": "Brief explanation of decomposition logic"
}
```

JSON output only:"""

        return prompt

    def _parse_response(
        self,
        response: str,
        fallback_instruction: str = None,
    ) -> Dict[str, Any]:
        """Parse LLM response into structured result.

        Args:
            response: LLM response text
            fallback_instruction: Instruction for fallback

        Returns:
            Dict with subtasks and reasoning
        """
        if not response:
            return self._fallback_dict(fallback_instruction)

        # Extract JSON from response
        json_str = response.strip()

        # Handle markdown code blocks
        if "```json" in json_str:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", json_str)
            if match:
                json_str = match.group(1).strip()
        elif "```" in json_str:
            match = re.search(r"```\s*([\s\S]*?)\s*```", json_str)
            if match:
                json_str = match.group(1).strip()

        # Find JSON object
        start = json_str.find("{")
        if start == -1:
            return self._fallback_dict(fallback_instruction)

        # Match braces
        brace_count = 0
        end = -1
        for i in range(start, len(json_str)):
            if json_str[i] == "{":
                brace_count += 1
            elif json_str[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if end == -1:
            return self._fallback_dict(fallback_instruction)

        try:
            data = json.loads(json_str[start:end+1])

            # Validate subtasks
            if "subtasks" not in data or not isinstance(data["subtasks"], list):
                return self._fallback_dict(fallback_instruction)

            # Ensure each subtask has required fields
            validated_subtasks = []
            for i, subtask in enumerate(data["subtasks"]):
                validated = {
                    "id": subtask.get("id", i + 1),
                    "description": subtask.get("description", f"Subtask {i+1}"),
                    "completion_condition": subtask.get("completion_condition", {"type": "distance_to_goal", "threshold": 3.0}),
                }
                validated_subtasks.append(validated)

            return {
                "subtasks": validated_subtasks,
                "reasoning": data.get("reasoning", ""),
            }
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parse failed: {e}")
            return self._fallback_dict(fallback_instruction)

    def _fallback_decomposition(
        self,
        instruction: str,
        goal_position: List[float],
    ) -> DecompositionOutput:
        """Create fallback single-subtask decomposition.

        Args:
            instruction: Original instruction
            goal_position: Goal position

        Returns:
            DecompositionOutput with single subtask
        """
        return DecompositionOutput(
            subtasks=[
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {
                        "type": "distance_to_goal",
                        "threshold": 3.0,
                        "goal_position": goal_position,
                    },
                }
            ],
            reasoning="Fallback: No LLM available or LLM failed",
        )

    def _fallback_dict(self, instruction: str) -> Dict[str, Any]:
        """Create fallback dict for parse failure.

        Args:
            instruction: Original instruction

        Returns:
            Dict with single subtask
        """
        return {
            "subtasks": [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {"type": "distance_to_goal", "threshold": 3.0},
                }
            ],
            "reasoning": "Fallback: JSON parsing failed",
        }
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/pipeline/test_subtask_decomposition_agent.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/pipeline/subtask_decomposition_agent.py tests/pipeline/test_subtask_decomposition_agent.py
git commit -m "feat: add SubtaskDecompositionAgent for instruction decomposition

- LLM-driven instruction decomposition into 2-6 subtasks
- Supports y_change, rotation, near_object, distance_to_goal, combined condition types
- Fallback to single subtask when LLM unavailable
- Unit tests for decomposition, parsing, fallback"
```

---

### Task 3: 修改 Navigator 集成分解和切换逻辑

**Files:**
- Modify: `agents/pipeline/navigator.py:85-100` (新增字段)
- Modify: `agents/pipeline/navigator.py:115-128` (register_subagents)
- Modify: `agents/pipeline/navigator.py:129-159` (initialize_episode)
- Modify: `agents/pipeline/navigator.py:441-499` (_check_completion - 添加切换)

- [ ] **Step 1: 新增 import 和字段**

在 navigator.py 顶部添加 import:
```python
from agents.pipeline.subtask_decomposition_agent import SubtaskDecompositionAgent
from agents.pipeline.base_pipeline_agent import DecompositionOutput
```

在 `__init__` 方法中新增字段（约第 85-90 行，在现有字段后）:
```python
        # Subtask management
        self._current_subtask_index: int = 0  # Current subtask index
```

- [ ] **Step 2: 修改 register_subagents() 注册新 Agent**

将第 115-128 行的 `register_subagents()` 方法改为:
```python
    def register_subagents(self) -> None:
        """Register all SubAgents to the registry."""
        self._registry.register("decomposition", SubtaskDecompositionAgent())
        self._registry.register("observation", ObservationAgent())
        self._registry.register("analysis", AnalysisAgent())
        self._registry.register("planning", PlanningAgent())
        self._registry.register("review", ReviewAgent())
        self._registry.register("emergency", EmergencyAgent())

        # Set model_manager for newly registered agents
        if self._model_manager is not None:
            for agent in self._registry.list_all().values():
                if hasattr(agent, "set_model_manager"):
                    agent.set_model_manager(self._model_manager)
```

- [ ] **Step 3: 新增 _decompose_instruction() 方法**

在 `register_subagents()` 方法后添加新方法:
```python
    def _decompose_instruction(
        self,
        instruction: str,
        goal_position: List[float],
    ) -> List[Dict[str, Any]]:
        """Decompose instruction into subtasks using LLM.

        Args:
            instruction: Navigation instruction text
            goal_position: Goal position [x, y, z]

        Returns:
            List of subtask dicts
        """
        if self._model_manager is None:
            self.logger.warning("[Navigator] No ModelManager, using fallback decomposition")
            return [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {
                        "type": "distance_to_goal",
                        "threshold": 3.0,
                        "goal_position": goal_position,
                    },
                }
            ]

        try:
            decomposition_output = self._registry.call(
                "decomposition",
                instruction=instruction,
                goal_position=goal_position,
                start_position=self._position,
            )

            self.logger.info(f"[Navigator] Decomposed into {len(decomposition_output.subtasks)} subtasks")
            for i, subtask in enumerate(decomposition_output.subtasks):
                self.logger.info(f"  Subtask {i+1}: {subtask['description']}")

            return decomposition_output.subtasks
        except Exception as e:
            self.logger.warning(f"[Navigator] Decomposition failed: {e}")
            return [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": {
                        "type": "distance_to_goal",
                        "threshold": 3.0,
                        "goal_position": goal_position,
                    },
                }
            ]
```

- [ ] **Step 4: 修改 initialize_episode() 方法**

将第 129-159 行的 `initialize_episode()` 方法改为:
```python
    def initialize_episode(
        self,
        instruction: str,
        start_position: List[float],
        goal_position: List[float] = None,
    ) -> None:
        """Initialize episode with instruction and starting position.

        Args:
            instruction: Navigation instruction text
            start_position: Starting position [x, y, z]
            goal_position: Goal position [x, y, z] (optional, for decomposition)
        """
        self._position = start_position.copy() if isinstance(start_position, list) else list(start_position)
        self._rotation = 0.0
        self._history = [{"position": self._position, "rotation": self._rotation, "step": 0}]
        self._step_count = 0

        # Decompose instruction into subtasks
        if goal_position is not None:
            self._subtasks = self._decompose_instruction(instruction, goal_position)
        else:
            # Fallback if no goal_position
            self._subtasks = [
                {
                    "id": 1,
                    "description": instruction,
                    "completion_condition": None,
                }
            ]

        self._current_subtask_index = 0
        self._current_subtask = self._subtasks[0]

        # Clear topology for new episode
        self._topology = TopologyGraph()

        self.logger.info(f"[Navigator] Episode initialized: instruction='{instruction[:50]}...', position={start_position}")
        self.logger.info(f"[Navigator] Total subtasks: {len(self._subtasks)}, starting with subtask 1")
```

- [ ] **Step 5: 修改 _check_completion() 添加子任务切换逻辑**

在 `_check_completion()` 方法中（约第 493-496 行），将:
```python
            if review_output.completed:
                self.logger.info(f"[Navigator] Subtask completed: {review_output.reason}")

            return review_output.completed
```

改为:
```python
            if review_output.completed:
                self.logger.info(f"[Navigator] Subtask {self._current_subtask_index + 1} completed: {review_output.reason}")

                # Check if there's next subtask
                if self._current_subtask_index < len(self._subtasks) - 1:
                    self._current_subtask_index += 1
                    self._current_subtask = self._subtasks[self._current_subtask_index]
                    self.logger.info(f"[Navigator] Switching to subtask {self._current_subtask_index + 1}: {self._current_subtask['description']}")
                    return False  # Not fully complete, just switched
                else:
                    self.logger.info(f"[Navigator] All {len(self._subtasks)} subtasks completed!")
                    return True  # All subtasks complete

            return False
```

- [ ] **Step 6: 验证语法正确**

Run: `python -c "from agents.pipeline.navigator import Navigator; n = Navigator(); print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add agents/pipeline/navigator.py
git commit -m "feat: integrate SubtaskDecompositionAgent into Navigator

- Register SubtaskDecompositionAgent in registry
- Add _decompose_instruction() method for LLM decomposition
- Modify initialize_episode() to accept goal_position and decompose
- Add _current_subtask_index field for tracking progress
- Add subtask switching logic in _check_completion()"
```

---

### Task 4: 修改 run_vln_experiment.py 传递 goal_position

**Files:**
- Modify: `run_vln_experiment.py:1900-1910`

- [ ] **Step 1: 修改 initialize_episode 调用**

将第 1900-1910 行的调用改为:
```python
        # 初始化 Navigator
        self.navigator.initialize_episode(
            instruction=episode.instruction,
            start_position=list(start_pos),
            goal_position=list(episode.goal_position),
        )

        # Note: completion_condition is now set by SubtaskDecompositionAgent
        # No need to manually set it here
```

- [ ] **Step 2: 验证语法正确**

Run: `python -c "from run_vln_experiment import MultiAgentVLNEvaluator; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add run_vln_experiment.py
git commit -m "feat: pass goal_position to Navigator for subtask decomposition"
```

---

### Task 5: 集成测试验证

**Files:**
- No new files (run existing pipeline test)

- [ ] **Step 1: 运行 Pipeline 实验**

Run: `bash scripts/run_vln_experiment_with_vllm.sh run_exp`
Expected:
- Logs show "Decomposed into X subtasks"
- Logs show "Subtask 1: ..."
- Logs show "Switching to subtask 2" when subtask completes

- [ ] **Step 2: 检查 agent_outputs.json 中的 subtasks**

Run: `cat results/episode-*/episode1/agent_outputs.json | python -c "import json,sys; d=json.load(sys.stdin); print('subtasks:', d.get('subtasks'))"`
Expected: Non-empty subtasks list with multiple subtask entries

- [ ] **Step 3: Final commit if needed**

```bash
git add -A
git commit -m "test: verify SubtaskDecompositionAgent integration works"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - DecompositionOutput dataclass: Task 1 ✓
   - SubtaskDecompositionAgent: Task 2 ✓
   - Navigator integration: Task 3 ✓
   - goal_position passing: Task 4 ✓
   - Testing: Task 5 ✓

2. **Placeholder scan:** No "TBD", "TODO", or vague steps - all code shown

3. **Type consistency:**
   - `DecompositionOutput` used in Task 1 and Task 2
   - `goal_position: List[float]` consistent across all tasks
   - `_current_subtask_index: int` defined in Task 3 and used in Task 3