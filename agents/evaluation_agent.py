"""Evaluation Agent for assessing navigation decisions.

This agent uses Qwen3.5-4B (local, INT8) for:
1. Decision evaluation and scoring
2. Feedback generation
3. Re-planning trigger logic

Note: Originally designed for Qwen3.5-9B, but uses Qwen3.5-4B for efficiency
and compatibility with the current model server configuration.
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

    Uses Qwen3.5-4B (local, INT8) for evaluation.
    This allows sharing the model with DecisionAgent while maintaining
    independent conversation contexts.

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

                # Load Qwen3.5-4B for evaluation (shares with DecisionAgent)
                self.logger.info("Loading Qwen3.5-4B for evaluation...")
                if self._model_manager.load_llm("qwen-4b"):
                    self.logger.info("Qwen3.5-4B loaded successfully for evaluation")
                else:
                    self.logger.warning("Failed to load Qwen3.5-4B, using fallback evaluation")

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
            vertical_nav_ok = evaluation.get("vertical_nav_ok", True)  # NEW

            # Track history
            self._evaluation_history.append({
                "step": context.step_count,
                "score": score,
                "feedback": feedback,
                "vertical_nav_ok": vertical_nav_ok,
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
                "vertical_nav_ok": vertical_nav_ok,  # NEW
            }
            context.metadata["last_evaluation_score"] = score

            return AgentOutput.success_output(
                data={
                    "score": score,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "replan_needed": replan_needed,
                    "vertical_nav_ok": vertical_nav_ok,  # NEW
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
        """Evaluate decision using Qwen4B (shared with DecisionAgent but independent context)."""
        prompt = self._build_evaluation_prompt(context, decision)

        try:
            if self._model_manager:
                # Get episode_id for conversation context isolation
                # Note: Uses independent conversation_id from DecisionAgent
                episode_id = context.metadata.get("episode_id", 0)
                conversation_id = f"evaluation_ep{episode_id}"

                response = self._model_manager.generate(
                    "qwen-4b-evaluation",  # Use dedicated evaluation model config
                    prompt,
                    max_new_tokens=200,  # Reduced from 400 for efficiency
                    temperature=0.3,  # Lower temperature for more consistent evaluation
                    conversation_id=conversation_id,
                    keep_context=True,
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
        """Build prompt for evaluation with vertical navigation awareness."""
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
        y_change = trajectory.get("y_change", 0.0)
        y_direction = trajectory.get("y_direction", "稳定")

        # Get decision info
        action = decision.get("action", "unknown")
        decision_reasoning = decision.get("reasoning", "")

        # Build history summary
        history_summary = self._build_history_summary(context)

        # === NEW: Spatial evaluation information ===
        import math
        current_pos = context.position
        goal_pos = context.metadata.get("goal_position")

        if goal_pos:
            vert_dist = goal_pos[1] - current_pos[1]
            horiz_dist = math.sqrt((goal_pos[0]-current_pos[0])**2 + (goal_pos[2]-current_pos[2])**2)
            floor_relation = "目标在下层" if vert_dist < -0.5 else "目标在上层" if vert_dist > 0.5 else "同层"
        else:
            vert_dist = 0
            horiz_dist = 0
            floor_relation = "未知"

        # Height change trend evaluation
        y_trend = "稳定"
        if len(context.trajectory) >= 3:
            recent_y = [p[1] for p in context.trajectory[-3:]]
            y_change_recent = recent_y[-1] - recent_y[0]
            if abs(y_change_recent) > 0.1:
                y_trend = f"最近{'上升' if y_change_recent > 0 else '下降'}{abs(y_change_recent):.2f}米"

        # === NEW: Enhanced prompt template ===
        prompt = f"""你是导航决策评估专家。评估决策合理性。

## 导航目标
- 指令: {instruction[:80]}
- 子任务: {subtask_desc[:60] if subtask_desc else "无"}

## 空间关系 (关键评估维度)
- 高度差: {vert_dist:+.2f}米
- 楼层关系: {floor_relation}
- 水平距离: {horiz_dist:.1f}米
- 高度趋势: {y_trend}
- 总高度变化: {y_change:+.2f}米 ({y_direction})

## 视觉感知
- 房间: {room_type}
- 物体: {', '.join([o.get('name', '') for o in objects]) if objects else '无'}
- 地标: {', '.join([lm.get('name', '') for lm in landmarks]) if landmarks else '无'}

## 轨迹状态
- 朝向: {heading}
- 已走: {distance:.1f}米
- 重复访问: {"是" if visited else "否"}
- 路径问题: {len(corrections)} 个

## 当前决策
- 动作: {action}
- 理由: {decision_reasoning[:80] if decision_reasoning else "无"}

## 历史评估
{history_summary}

## 评估要点
1. 如果需要垂直导航（高度差>0.5m），决策是否在寻找楼梯？
2. 高度变化方向是否与目标方向一致？
   - 目标在下层 + 高度下降 = 正确
   - 目标在上层 + 高度上升 = 正确
   - 目标在下层 + 高度上升 = 错误，需要转向
   - 目标在上层 + 高度下降 = 错误，需要转向
3. 动作是否合理推进子任务？

输出JSON:
{{
  "score": 0.0-1.0,
  "feedback": "评估反馈",
  "vertical_nav_ok": true/false,
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
            # Find JSON in response - use a more robust regex for multiline JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return {
                    "score": float(evaluation.get("score", 0.5)),
                    "feedback": evaluation.get("feedback", ""),
                    "vertical_nav_ok": evaluation.get("vertical_nav_ok", True),  # NEW
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
                "vertical_nav_ok": True,  # Default
                "suggestions": [],
            }

        return {
            "score": 0.5,
            "feedback": "无法解析评估结果",
            "vertical_nav_ok": True,  # Default
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

    def penalize_stuck_decisions(
        self,
        context: NavContext,
        decision: Dict[str, Any]
    ) -> float:
        """Penalize decisions that led to stuck situations.

        Args:
            context: Navigation context
            decision: Decision to evaluate

        Returns:
            Penalty factor (0.0-1.0, lower is worse)
        """
        penalty = 1.0

        # Check if position is in known stuck region
        if hasattr(context, 'is_in_stuck_region') and context.is_in_stuck_region(context.position):
            penalty *= 0.7

        # Check recent action patterns that led to stuck
        if context.action_history:
            recent = context.action_history[-6:]
            turn_count = sum(1 for a in recent if a.action_type.name in ["TURN_LEFT", "TURN_RIGHT"])
            forward_count = sum(1 for a in recent if a.action_type.name == "MOVE_FORWARD")

            # Penalize excessive turning without forward progress
            if turn_count > 4 and forward_count < 2:
                penalty *= 0.6

        return penalty

    def get_stuck_escape_opinion(
        self,
        decision_history: List[Dict[str, Any]],
        evaluation_history: List[Dict[str, Any]],
        context: NavContext = None
    ) -> Dict[str, Any]:
        """Provide stuck escape opinion based on decision history.

        Analyzes past decisions and their outcomes to suggest escape routes.

        Args:
            decision_history: List of past decisions
            evaluation_history: List of past evaluations
            context: Navigation context (optional)

        Returns:
            Dict with escape direction, confidence, and reasoning
        """
        opinion = {
            "direction": "right",
            "confidence": 0.5,
            "reason": "默认建议",
            "stop_condition": "",
            "agent_source": "evaluation",
        }

        if not decision_history or not evaluation_history:
            opinion["reason"] = "决策历史不足"
            return opinion

        # Analyze recent decisions and their scores
        recent_decisions = decision_history[-10:]
        recent_evals = evaluation_history[-10:] if evaluation_history else []

        # Find patterns of successful vs failed decisions
        successful_actions = []
        failed_actions = []

        for i, decision_record in enumerate(recent_decisions):
            decision = decision_record.get("decision", {})
            action = decision.get("action", "unknown")

            # Get corresponding evaluation if available
            if i < len(recent_evals):
                eval_score = recent_evals[i].get("score", 0.5)
                if eval_score >= 0.6:
                    successful_actions.append(action)
                elif eval_score < 0.4:
                    failed_actions.append(action)

        # Count action frequencies
        from collections import Counter
        success_counts = Counter(successful_actions)
        failed_counts = Counter(failed_actions)

        # Find actions that worked vs didn't work
        good_forward = success_counts.get("forward", 0) + success_counts.get("move_forward", 0)
        bad_forward = failed_counts.get("forward", 0) + failed_counts.get("move_forward", 0)
        good_left = success_counts.get("turn_left", 0) + success_counts.get("left", 0)
        bad_left = failed_counts.get("turn_left", 0) + failed_counts.get("left", 0)
        good_right = success_counts.get("turn_right", 0) + success_counts.get("right", 0)
        bad_right = failed_counts.get("turn_right", 0) + failed_counts.get("right", 0)

        # Recommend direction with best success rate
        if good_forward > bad_forward and good_forward > 0:
            opinion["direction"] = "forward"
            opinion["confidence"] = 0.7 + (good_forward - bad_forward) * 0.05
            opinion["reason"] = f"前进决策成功率较高({good_forward}/{good_forward + bad_forward})"
        elif good_left > bad_left and good_left > good_right:
            opinion["direction"] = "left"
            opinion["confidence"] = 0.65 + (good_left - bad_left) * 0.05
            opinion["reason"] = f"左转决策成功率较高({good_left}/{good_left + bad_left})"
        elif good_right > bad_right and good_right > good_left:
            opinion["direction"] = "right"
            opinion["confidence"] = 0.65 + (good_right - bad_right) * 0.05
            opinion["reason"] = f"右转决策成功率较高({good_right}/{good_right + bad_right})"
        else:
            # No clear pattern - suggest trying different direction
            opinion["reason"] = "决策历史无明显模式，建议探索"

            # Avoid recently failed directions
            if failed_actions:
                recent_failed = [a for a in failed_actions[-3:]]
                if "turn_left" in recent_failed or "left" in recent_failed:
                    opinion["direction"] = "right"
                    opinion["reason"] += "，避免最近失败的左转"
                elif "turn_right" in recent_failed or "right" in recent_failed:
                    opinion["direction"] = "left"
                    opinion["reason"] += "，避免最近失败的右转"

        opinion["stop_condition"] = "决策得分>0.6或移动2米"

        return opinion

    def reset_history(self) -> None:
        """Reset evaluation history."""
        self._evaluation_history.clear()
        self._recent_scores.clear()