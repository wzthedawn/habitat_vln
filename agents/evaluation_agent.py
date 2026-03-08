"""Evaluation Agent for assessing navigation decisions.

This agent uses Qwen3.5-9B (local, INT8) for:
1. Decision evaluation and scoring
2. Feedback generation
3. Re-planning trigger logic
"""

from typing import Dict, Any, Optional, List
import logging
import json
import re

from .base_agent import BaseAgent, AgentOutput, AgentRole
from core.context import NavContext


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for evaluating navigation decisions.

    Uses Qwen3.5-9B (local, INT8) for evaluation.

    Key responsibilities:
    1. Evaluate decision quality (score 0.0-1.0)
    2. Generate feedback for improvement
    3. Trigger re-planning when needed

    Called based on task level:
    - 简单: Not called
    - 中等: Every 5 steps
    - 困难: Every step
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("EvaluationAgent")

        # Evaluation parameters
        self.low_score_threshold = self.config.get("low_score_threshold", 0.4)
        self.medium_score_threshold = self.config.get("medium_score_threshold", 0.7)
        self.replan_consecutive_low = self.config.get("replan_consecutive_low", 3)
        self.replan_total_low = self.config.get("replan_total_low", 5)

        # History tracking
        self._evaluation_history: List[Dict[str, Any]] = []
        self._recent_scores: List[float] = []

        # Model reference
        self._model_manager = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "evaluation_agent"

    @property
    def role(self) -> AgentRole:
        return AgentRole.DECISION  # Reusing DECISION role

    def get_required_inputs(self) -> List[str]:
        return ["context", "decision"]

    def get_output_keys(self) -> List[str]:
        return ["score", "feedback", "replan_needed", "suggestions"]

    def initialize(self) -> None:
        """Initialize model manager and load Qwen3.5-9B."""
        if self._initialized:
            return

        try:
            from models.model_manager import get_model_manager
            self._model_manager = get_model_manager(self.config)

            # Check if using remote LLM
            use_remote = self.config.get("use_remote", False)

            if use_remote:
                self.logger.info("Using remote LLM service for evaluation")
                self._model_manager.load_all_models()  # Only loads YOLO locally
            else:
                self._model_manager.load_all_models()

                # Load Qwen3.5-9B for evaluation
                self.logger.info("Loading Qwen3.5-9B for evaluation...")
                if self._model_manager.load_llm("qwen-9b"):
                    self.logger.info("Qwen3.5-9B loaded successfully")
                else:
                    self.logger.warning("Failed to load Qwen3.5-9B, using fallback evaluation")

            self._initialized = True
            self.logger.info("EvaluationAgent initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize model manager: {e}")
            self._initialized = True

    def process(
        self,
        context: NavContext,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """
        Evaluate the current navigation decision.

        Args:
            context: Navigation context
            strategy_result: Decision output from DecisionAgent

        Returns:
            AgentOutput with evaluation score and feedback
        """
        self.initialize()

        try:
            # Get decision from strategy_result or context
            decision = strategy_result or context.metadata.get("decision_output", {})

            # Evaluate using Qwen9B
            evaluation = self._evaluate_decision(context, decision)

            score = evaluation.get("score", 0.5)
            feedback = evaluation.get("feedback", "")
            suggestions = evaluation.get("suggestions", [])

            # Track history
            self._evaluation_history.append({
                "step": context.step_count,
                "score": score,
                "feedback": feedback,
            })
            self._recent_scores.append(score)

            # Keep only recent history
            max_history = 20
            if len(self._evaluation_history) > max_history:
                self._evaluation_history = self._evaluation_history[-max_history:]
            if len(self._recent_scores) > max_history:
                self._recent_scores = self._recent_scores[-max_history:]

            # Check if re-planning is needed
            replan_needed = self._check_replan_needed()

            # Store evaluation in context
            context.metadata["evaluation_output"] = {
                "score": score,
                "feedback": feedback,
                "suggestions": suggestions,
                "replan_needed": replan_needed,
            }
            context.metadata["last_evaluation_score"] = score

            return AgentOutput.success_output(
                data={
                    "score": score,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "replan_needed": replan_needed,
                    "score_level": self._get_score_level(score),
                    "recent_avg_score": sum(self._recent_scores[-5:]) / max(len(self._recent_scores[-5:]), 1),
                },
                confidence=score,
                reasoning=f"Evaluation score: {score:.2f} - {feedback}",
            )

        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return AgentOutput.failure_output([str(e)], "Evaluation failed")

    def _evaluate_decision(
        self,
        context: NavContext,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate decision using Qwen9B."""
        prompt = self._build_evaluation_prompt(context, decision)

        try:
            if self._model_manager:
                response = self._model_manager.generate(
                    "qwen-9b",
                    prompt,
                    max_new_tokens=400,
                    temperature=0.3  # Lower temperature for more consistent evaluation
                )
                return self._parse_evaluation_response(response)
            else:
                return self._fallback_evaluation(context, decision)

        except Exception as e:
            self.logger.warning(f"LLM evaluation failed: {e}")
            return self._fallback_evaluation(context, decision)

    def _build_evaluation_prompt(
        self,
        context: NavContext,
        decision: Dict[str, Any]
    ) -> str:
        """Build prompt for evaluation."""
        # Get instruction info
        instruction = context.instruction
        task_level = context.metadata.get("task_level", "中等")

        # Get current subtask
        current_subtask = context.get_current_subtask()
        subtask_desc = current_subtask.description if current_subtask else "无"

        # Get perception info
        perception = context.metadata.get("perception_output", {})
        room_type = perception.get("room_type", "未知")
        objects = perception.get("objects", [])[:3]
        landmarks = perception.get("landmarks", [])

        # Get trajectory info
        trajectory = context.metadata.get("trajectory", {})
        heading = trajectory.get("heading", "未知")
        distance = trajectory.get("distance_traveled", 0)
        visited = trajectory.get("visited", False)
        corrections = trajectory.get("corrections", [])

        # Get decision info
        action = decision.get("action", "unknown")
        decision_reasoning = decision.get("reasoning", "")

        # Build history summary
        history_summary = self._build_history_summary(context)

        prompt = f"""你是一个导航决策评估专家。请评估当前的导航决策是否合理。

## 导航指令
{instruction}

## 任务等级
{task_level}

## 当前子任务
{subtask_desc}

## 视觉感知
房间: {room_type}
物体: {', '.join([o.get('name', '') for o in objects]) if objects else '无'}
地标: {', '.join([lm.get('name', '') for lm in landmarks]) if landmarks else '无'}

## 轨迹状态
朝向: {heading}
已走: {distance:.1f}米
重复访问: {"是" if visited else "否"}
路径问题: {len(corrections)} 个

## 当前决策
动作: {action}
理由: {decision_reasoning}

## 历史评估
{history_summary}

请评估这个决策(0.0-1.0分):
- 0.0-0.4: 决策不佳，需要调整
- 0.4-0.7: 决策一般，可以改进
- 0.7-1.0: 决策良好

输出JSON格式:
{{
  "score": 0.0-1.0,
  "feedback": "评估反馈",
  "suggestions": ["建议1", "建议2"]
}}

只输出JSON。"""

        return prompt

    def _build_history_summary(self, context: NavContext) -> str:
        """Build summary of recent evaluation history."""
        if not self._evaluation_history:
            return "无历史评估"

        recent = self._evaluation_history[-5:]
        lines = []

        for eval_record in recent:
            step = eval_record.get("step", "?")
            score = eval_record.get("score", 0)
            lines.append(f"步骤{step}: {score:.2f}分")

        return "\n".join(lines)

    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse evaluation from LLM response."""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return {
                    "score": float(evaluation.get("score", 0.5)),
                    "feedback": evaluation.get("feedback", ""),
                    "suggestions": evaluation.get("suggestions", []),
                }
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse JSON: {e}")

        # Fallback: extract score from text
        score_match = re.search(r'(\d+\.?\d*)', response)
        if score_match:
            score = float(score_match.group(1))
            if score > 1:
                score = score / 10 if score <= 10 else score / 100
            return {
                "score": min(max(score, 0), 1),
                "feedback": response[:100],
                "suggestions": [],
            }

        return {
            "score": 0.5,
            "feedback": "无法解析评估结果",
            "suggestions": [],
        }

    def _fallback_evaluation(
        self,
        context: NavContext,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback evaluation when LLM is unavailable."""
        score = 0.5
        feedback = ""
        suggestions = []

        # Check if decision action makes sense
        action = decision.get("action", "forward")
        perception = context.metadata.get("perception_output", {})
        trajectory = context.metadata.get("trajectory", {})

        # Check for landmark proximity
        landmarks = perception.get("landmarks", [])
        if landmarks:
            closest = min(landmarks, key=lambda x: x.get("distance", 999))
            dist = closest.get("distance", 999)

            if dist < 2.0:
                score += 0.2
                feedback = f"接近目标地标: {closest.get('name')}"
            elif dist < 5.0:
                score += 0.1
                feedback = f"正在接近地标: {closest.get('name')}"

        # Check for repeated visits
        if trajectory.get("visited", False):
            score -= 0.2
            feedback = "重复访问相同区域"
            suggestions.append("尝试不同方向")

        # Check for corrections
        corrections = trajectory.get("corrections", [])
        if corrections:
            score -= 0.1 * len(corrections)
            for correction in corrections:
                suggestions.append(correction.get("message", ""))

        # Check for stuck
        if any(c.get("type") == "stuck" for c in corrections):
            score -= 0.3
            feedback = "导航似乎卡住了"
            suggestions.append("考虑回头或转向")

        # Normalize score
        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "feedback": feedback or "评估完成",
            "suggestions": suggestions,
        }

    def _check_replan_needed(self) -> bool:
        """Check if re-planning is needed based on scores."""
        if len(self._recent_scores) < self.replan_consecutive_low:
            return False

        # Check for consecutive low scores
        recent = self._recent_scores[-self.replan_consecutive_low:]
        if all(score < self.low_score_threshold for score in recent):
            self.logger.warning(f"Re-planning triggered: {self.replan_consecutive_low} consecutive low scores")
            return True

        # Check for total low scores
        if len(self._recent_scores) >= self.replan_total_low:
            low_count = sum(1 for score in self._recent_scores[-self.replan_total_low:]
                          if score < self.medium_score_threshold)
            if low_count >= self.replan_total_low:
                self.logger.warning(f"Re-planning triggered: {low_count} low scores in {self.replan_total_low} steps")
                return True

        return False

    def _get_score_level(self, score: float) -> str:
        """Get score level description."""
        if score >= self.medium_score_threshold:
            return "good"
        elif score >= self.low_score_threshold:
            return "medium"
        else:
            return "low"

    def should_call_evaluation(self, task_level: str, step_count: int) -> bool:
        """
        Determine if evaluation should be called based on task level.

        Args:
            task_level: Task difficulty level
            step_count: Current step count

        Returns:
            True if evaluation should be called
        """
        if task_level == "简单":
            return False  # Never call for simple tasks

        elif task_level == "中等":
            return step_count % 5 == 0  # Every 5 steps

        elif task_level == "困难":
            return True  # Every step

        return False

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self._evaluation_history:
            return {
                "total_evaluations": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }

        scores = [e.get("score", 0) for e in self._evaluation_history]

        return {
            "total_evaluations": len(self._evaluation_history),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "recent_scores": self._recent_scores[-5:],
        }

    def reset_history(self) -> None:
        """Reset evaluation history."""
        self._evaluation_history.clear()
        self._recent_scores.clear()