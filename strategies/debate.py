"""Debate strategy implementation.

This strategy collects opinions from all agents, synthesizes them,
and generates consensus for action sequence generation.

Phase 2: EvaluationAgent as Judge with dynamic weight calculation.
"""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class DebateStrategy(BaseStrategy):
    """
    Debate strategy for difficult tasks.

    Collects opinions from all agents (Perception, Trajectory, Instruction, Evaluation),
    synthesizes them through LLM-based debate, and generates consensus.

    Pattern: Collect Opinions → EvaluationAgent Judge → Provide to DecisionAgent

    EvaluationAgent acts as the judge:
    1. Evaluates quality of other agents' opinions
    2. Detects conflicts between opinions
    3. Uses dynamic weights from PerformanceTracker
    4. Provides final arbitration
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Configuration
        self.weight_perception = self.config.get("weight_perception", 1.2)
        self.weight_instruction = self.config.get("weight_instruction", 1.0)
        self.weight_trajectory = self.config.get("weight_trajectory", 0.8)
        self.weight_evaluation = self.config.get("weight_evaluation", 1.0)

        # Performance tracker for dynamic weights
        self._performance_tracker = None

    def _get_performance_tracker(self):
        """Get or create PerformanceTracker."""
        if self._performance_tracker is None:
            from agents.evaluation_agent import get_performance_tracker
            self._performance_tracker = get_performance_tracker(self.config)
        return self._performance_tracker

    @property
    def name(self) -> str:
        return "Debate"

    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.DEBATE

    def execute(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        prev_result: Optional[StrategyResult] = None,
    ) -> StrategyResult:
        """
        Execute Debate strategy - collect opinions and synthesize.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with opinions and consensus
        """
        self.initialize()

        steps = []
        tracker = self._get_performance_tracker()

        try:
            # Step 1: Collect opinions from Perception, Trajectory, Instruction agents
            opinions = {}
            evaluation_agent = None

            for agent in agents:
                if agent is None:
                    continue

                agent_name = agent.name if hasattr(agent, 'name') else str(type(agent))

                # Separate EvaluationAgent for judge role
                if "evaluation" in agent_name:
                    evaluation_agent = agent
                    continue

                # Try to get debate opinion from agent
                if hasattr(agent, 'build_debate_opinion'):
                    try:
                        opinion = agent.build_debate_opinion(context)
                        opinions[agent_name] = opinion
                        steps.append({"type": "opinion", "agent": agent_name, "data": opinion})
                    except Exception as e:
                        self.logger.warning(f"[Debate] Failed to get opinion from {agent_name}: {e}")
                else:
                    # Fallback: collect info from context
                    info = self._collect_agent_info(context, agent_name)
                    if info:
                        opinions[agent_name] = info
                        steps.append({"type": "info", "agent": agent_name, "data": info})

            # Store opinions in context for EvaluationAgent
            context.metadata["debate_opinions"] = opinions

            # Step 2: EvaluationAgent as judge - evaluates and arbitrates
            if evaluation_agent and opinions:
                try:
                    eval_opinion = evaluation_agent.build_debate_opinion(
                        context,
                        opinions=opinions,
                        performance_tracker=tracker
                    )
                    opinions["evaluation_agent"] = eval_opinion
                    steps.append({"type": "judge", "agent": "evaluation_agent", "data": eval_opinion})
                    self.logger.info(f"[Debate] Evaluation judge: {eval_opinion.primary_action} (conf={eval_opinion.confidence:.2f})")
                except Exception as e:
                    self.logger.warning(f"[Debate] EvaluationAgent judge failed: {e}")

            # Step 3: Synthesize opinions (now includes evaluation judgment)
            consensus = self._synthesize_opinions(context, opinions)
            steps.append({"type": "consensus", "data": consensus})

            # Step 4: Record opinion outcomes for performance tracking
            self._record_opinion_outcomes(context, opinions, consensus, tracker)

            return StrategyResult(
                success=True,
                action=None,  # No single action - will be used for sequence generation
                reasoning=consensus.get("reasoning", ""),
                steps=steps,
                confidence=consensus.get("confidence", 0.7),
                metadata={
                    "opinions": opinions,
                    "consensus": consensus,
                    "weights": tracker.get_summary() if tracker else {},
                },
            )

        except Exception as e:
            self.logger.error(f"[Debate] Execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"Debate failed: {str(e)}",
                steps=steps,
            )

    def _collect_agent_info(self, context: NavContext, agent_name: str) -> Optional[Dict[str, Any]]:
        """Collect info from context for agents without build_debate_opinion method."""
        if "perception" in agent_name:
            return {
                "agent": "perception",
                "primary_action": "unknown",
                "confidence": 0.5,
                "evidence": context.metadata.get("perception_output", {}),
                "reasoning": "感知信息",
            }
        elif "trajectory" in agent_name:
            return {
                "agent": "trajectory",
                "primary_action": "unknown",
                "confidence": 0.5,
                "evidence": context.metadata.get("trajectory_output", {}),
                "reasoning": "轨迹信息",
            }
        elif "instruction" in agent_name:
            return {
                "agent": "instruction",
                "primary_action": "unknown",
                "confidence": 0.5,
                "evidence": context.metadata.get("instruction_output", {}),
                "reasoning": "指令信息",
            }
        elif "evaluation" in agent_name:
            return {
                "agent": "evaluation",
                "primary_action": "unknown",
                "confidence": 0.5,
                "evidence": context.metadata.get("evaluation_output", {}),
                "reasoning": "评估信息",
            }
        return None

    def _synthesize_opinions(self, context: NavContext, opinions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to synthesize opinions and generate consensus.

        If evaluation_agent provided a judgment, it takes priority
        as the judge's decision.

        Returns:
            Dict with consensus reasoning and confidence
        """
        # Check if evaluation agent provided a judgment
        eval_opinion = opinions.get("evaluation_agent")
        if eval_opinion:
            # Extract judgment from evaluation
            if hasattr(eval_opinion, 'primary_action'):
                action = eval_opinion.primary_action
                confidence = eval_opinion.confidence
                reasoning = eval_opinion.reasoning
                evidence = getattr(eval_opinion, 'evidence', {})
            elif isinstance(eval_opinion, dict):
                action = eval_opinion.get("primary_action", "")
                confidence = eval_opinion.get("confidence", 0.5)
                reasoning = eval_opinion.get("reasoning", "")
                evidence = eval_opinion.get("evidence", {})
            else:
                action = ""
                confidence = 0.5
                reasoning = ""
                evidence = {}

            # If evaluation has high confidence, use it directly
            if confidence >= 0.7 and action:
                return {
                    "analysis": f"裁判决策: {reasoning[:100]}",
                    "conflicts": evidence.get("conflicts", []),
                    "resolution": "使用评估Agent仲裁结果",
                    "recommended_focus": action,
                    "reasoning": reasoning[:100],
                    "confidence": confidence,
                    "from_judge": True,
                }

        # Fall back to LLM synthesis for lower confidence cases
        prompt = self._build_synthesis_prompt(context, opinions)

        # Call LLM
        response = self._call_llm(
            prompt,
            max_tokens=500,
            temperature=0.2
        )

        # Parse response
        consensus = self._parse_synthesis_response(response)

        return consensus

    def _build_synthesis_prompt(
        self,
        context: NavContext,
        opinions: Dict[str, Any],
    ) -> str:
        """Build the synthesis prompt for LLM."""
        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description if current_subtask else "无"

        # Format opinions
        opinions_str = ""
        for agent_name, opinion in opinions.items():
            # Handle DebateOpinion objects
            if hasattr(opinion, 'primary_action'):
                action = opinion.primary_action
                confidence = opinion.confidence
                reasoning = opinion.reasoning[:100] if opinion.reasoning else ""
            # Handle dict opinions
            elif isinstance(opinion, dict):
                reasoning = opinion.get("reasoning", "")[:100]
                action = opinion.get("primary_action", "unknown")
                confidence = opinion.get("confidence", 0.5)
            else:
                continue

            opinions_str += f"\n### {agent_name}\n"
            opinions_str += f"- 推荐动作: {action}\n"
            opinions_str += f"- 置信度: {confidence:.2f}\n"
            opinions_str += f"- 理由: {reasoning}\n"

        prompt = f"""你是一个导航辩论综合专家。请分析各Agent的意见，综合得出最佳导航策略。

## 导航指令
{context.instruction}

## 当前子任务
{subtask_desc}

## Agent意见
{opinions_str}

## 当前状态
- 步数: {context.step_count}
- 位置: ({context.position[0]:.1f}, {context.position[1]:.1f}, {context.position[2]:.1f})
- 房间: {context.room_type}

## 分析要求
1. 分析各Agent意见的一致性和冲突点
2. 权衡不同意见的重要性
3. 解决冲突，给出综合建议
4. 明确下一步导航重点

## 输出格式
请严格按照以下JSON格式输出:
```json
{{
  "analysis": "各意见的综合分析...",
  "conflicts": ["冲突点1", "冲突点2"],
  "resolution": "冲突解决方案...",
  "recommended_focus": "导航重点（20字以内）",
  "reasoning": "综合理由（50字以内）",
  "confidence": 0.0-1.0
}}
```

直接输出JSON："""

        return prompt

    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM synthesis response."""
        import json
        import re

        # Default result
        result = {
            "analysis": "",
            "conflicts": [],
            "resolution": "",
            "recommended_focus": "继续导航",
            "reasoning": "综合各Agent意见",
            "confidence": 0.6,
        }

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                result["analysis"] = data.get("analysis", "")
                result["conflicts"] = data.get("conflicts", [])
                result["resolution"] = data.get("resolution", "")
                result["recommended_focus"] = data.get("recommended_focus", "继续导航")
                result["reasoning"] = data.get("reasoning", "")
                result["confidence"] = float(data.get("confidence", 0.6))

                # Clamp confidence
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"[Debate] Failed to parse JSON response: {e}")
            result["reasoning"] = response[:200]

        return result

    def _record_opinion_outcomes(
        self,
        context: NavContext,
        opinions: Dict[str, Any],
        consensus: Dict[str, Any],
        tracker: 'PerformanceTracker'
    ) -> None:
        """
        Record opinion outcomes for performance tracking.

        This is called after consensus is reached to update
        the performance tracker with whether each agent's opinion
        was correct (aligned with final decision).

        Args:
            context: Navigation context
            opinions: All agent opinions
            consensus: Final consensus result
            tracker: PerformanceTracker instance
        """
        if not tracker or not opinions:
            return

        # Get the recommended action from evaluation or consensus
        eval_opinion = opinions.get("evaluation_agent", {})
        if hasattr(eval_opinion, 'primary_action'):
            final_action = eval_opinion.primary_action
        elif isinstance(eval_opinion, dict):
            final_action = eval_opinion.get("primary_action", "")
        else:
            final_action = ""

        if not final_action:
            final_action = consensus.get("recommended_focus", "")

        # Normalize action name (use "forward" for consistency)
        action_map = {
            "move_forward": "forward",
            "left": "turn_left",
            "right": "turn_right",
        }
        final_action = action_map.get(final_action, final_action)

        # Record each agent's opinion outcome
        for agent_name, opinion in opinions.items():
            if agent_name == "evaluation_agent":
                continue  # Don't record judge's opinion

            # Handle DebateOpinion objects
            if hasattr(opinion, 'primary_action'):
                predicted_action = opinion.primary_action
                evidence = getattr(opinion, 'evidence', {})
            elif isinstance(opinion, dict):
                predicted_action = opinion.get("primary_action", "")
                evidence = opinion.get("evidence", {})
            else:
                continue

            predicted_action = action_map.get(predicted_action, predicted_action)

            # Check if opinion aligned with final decision
            was_correct = predicted_action == final_action

            # Check if it was a critical contribution (e.g., obstacle detected)
            was_critical = False
            if isinstance(evidence, dict):
                # Critical if detected obstacle or key landmark
                if evidence.get("obstacle_ahead") or evidence.get("landmarks"):
                    was_critical = True

            # Record to tracker
            tracker.record_opinion(
                agent_name=agent_name,
                was_correct=was_correct,
                was_critical=was_critical
            )

        # Save tracker state periodically
        tracker.save()

    def reset_episode(self) -> None:
        """Reset episode statistics for new episode."""
        if self._performance_tracker:
            self._performance_tracker.reset_episode()