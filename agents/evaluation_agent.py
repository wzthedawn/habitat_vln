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

                # Load Qwen3.5-4B for evaluation (uses dedicated model config)
                self.logger.info("Loading Qwen3.5-4B for evaluation...")
                if self._model_manager.load_llm("qwen-4b-evaluation"):
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
            self.logger.info(f"[Evaluation] 评分:{score:.2f}, 反馈:{feedback[:30]}...")

            return AgentOutput.success_output(
                data={
                    "evaluation": {
                        "score": score,
                        "feedback": feedback,
                        "score_level": self._get_score_level(score),
                    },
                    "recommendation": {
                        "action": suggestions[0] if suggestions else "continue",
                        "replan_needed": replan_needed,
                        "vertical_nav_ok": vertical_nav_ok,
                    },
                    "score": score,
                    "feedback": feedback,
                    "suggestions": suggestions,
                    "replan_needed": replan_needed,
                    "vertical_nav_ok": vertical_nav_ok,
                    "score_level": self._get_score_level(score),
                    "recent_avg_score": sum(self._recent_scores[-5:]) / max(len(self._recent_scores[-5:]), 1),
                },
                confidence=score,
                reasoning=f"Evaluation score: {score:.2f} - {feedback}",
            )

        except Exception as e:
            self.logger.error(f"[Evaluation] 错误: {e}")
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
            self.logger.warning(f"[Evaluation] LLM失败: {e}")
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
            # Find JSON in response - support nested objects up to 2 levels
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                return {
                    "score": float(evaluation.get("score", 0.5)),
                    "feedback": evaluation.get("feedback", ""),
                    "vertical_nav_ok": evaluation.get("vertical_nav_ok", True),  # NEW
                    "suggestions": evaluation.get("suggestions", []),
                }
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"[Evaluation] JSON解析失败: {e}")

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
        # 禁用重新规划功能
        return False

        # if len(self._recent_scores) < self.replan_consecutive_low:
        #     return False

        # # Check for consecutive low scores
        # recent = self._recent_scores[-self.replan_consecutive_low:]
        # if all(score < self.low_score_threshold for score in recent):
        #     self.logger.warning(f"Re-planning triggered: {self.replan_consecutive_low} consecutive low scores")
        #     return True

        # # Check for total low scores
        # if len(self._recent_scores) >= self.replan_total_low:
        #     low_count = sum(1 for score in self._recent_scores[-self.replan_total_low:]
        #                   if score < self.medium_score_threshold)
        #     if low_count >= self.replan_total_low:
        #         self.logger.warning(f"Re-planning triggered: {low_count} low scores in {self.replan_total_low} steps")
        #         return True

        # return False

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

    # ========== Debate Judge Role ==========

    def build_debate_opinion(
        self,
        context: NavContext,
        opinions: Dict[str, Any] = None,
        performance_tracker: 'PerformanceTracker' = None
    ) -> 'DebateOpinion':
        """
        Build evaluation opinion for debate strategy.

        As the judge, this method:
        1. Evaluates quality of other agents' opinions
        2. Detects conflicts between opinions
        3. Provides final arbitration

        Args:
            context: Navigation context
            opinions: Dict of agent_name -> DebateOpinion from other agents
            performance_tracker: Tracker for dynamic weights

        Returns:
            DebateOpinion with evaluation results
        """
        from core.debate_types import DebateOpinion, ActionConstraint

        # Initialize performance tracker if not provided
        if performance_tracker is None:
            performance_tracker = get_performance_tracker(self.config)

        # Get current agent opinions from context metadata if not provided
        if opinions is None:
            opinions = context.metadata.get("debate_opinions", {})

        # Calculate dynamic weights based on historical performance
        weights = self._calculate_debate_weights(performance_tracker)

        # Evaluate each opinion
        opinion_scores = {}
        conflicts = []

        for agent_name, opinion in opinions.items():
            # Handle both DebateOpinion objects and dicts
            if hasattr(opinion, 'primary_action') or isinstance(opinion, dict):
                score = self._evaluate_opinion_quality(context, agent_name, opinion)
                opinion_scores[agent_name] = score

        # Detect conflicts between opinions
        conflicts = self._detect_opinion_conflicts(opinions)

        # Generate arbitration result
        arbitration = self._arbitrate_opinions(context, opinions, weights, opinion_scores, conflicts)

        # Build constraints based on evaluation
        constraints = {"hard": [], "soft": []}

        # Add hard constraints for blocked actions
        if arbitration.get("blocked_actions"):
            for action in arbitration["blocked_actions"]:
                constraints["hard"].append(ActionConstraint(
                    action=action,
                    blocked=True,
                    reason="评估结果禁止"
                ))

        # Add soft constraints for weight adjustments
        if arbitration.get("weight_adjustments"):
            for action, multiplier in arbitration["weight_adjustments"].items():
                constraints["soft"].append(ActionConstraint(
                    action=action,
                    weight_multiplier=multiplier,
                    reason="评估权重调整"
                ))

        # Store evaluation result in context
        context.metadata["evaluation_output"] = {
            "weights": weights,
            "opinion_scores": opinion_scores,
            "conflicts": conflicts,
            "arbitration": arbitration,
        }

        return DebateOpinion(
            agent="evaluation",
            primary_action=arbitration.get("recommended_action", "move_forward"),
            confidence=arbitration.get("confidence", 0.5),
            evidence={
                "opinion_scores": opinion_scores,
                "weights": weights,
                "conflicts": conflicts,
            },
            reasoning=arbitration.get("reasoning", "评估完成"),
            constraints=constraints,
        )

    def _calculate_debate_weights(self, tracker: 'PerformanceTracker') -> Dict[str, float]:
        """Calculate weights for each agent based on performance."""
        return {
            "perception": tracker.get_accuracy("perception_agent"),
            "trajectory": tracker.get_accuracy("trajectory_agent"),
            "instruction": tracker.get_accuracy("instruction_agent"),
        }

    def _evaluate_opinion_quality(
        self,
        context: NavContext,
        agent_name: str,
        opinion: Dict[str, Any]
    ) -> float:
        """Evaluate quality of a single opinion."""
        score = 0.5

        # Handle DebateOpinion objects
        if hasattr(opinion, 'reasoning'):
            reasoning = opinion.reasoning or ""
            confidence = opinion.confidence
            evidence = opinion.evidence if hasattr(opinion, 'evidence') else {}
        elif isinstance(opinion, dict):
            reasoning = opinion.get("reasoning", "")
            confidence = opinion.get("confidence", 0.5)
            evidence = opinion.get("evidence", {})
        else:
            return score

        # Check if opinion has good reasoning
        if len(reasoning) > 50:
            score += 0.1

        # Check confidence alignment with evidence
        confidence = confidence if isinstance(confidence, (int, float)) else 0.5

        # Perception opinion quality
        if "perception" in agent_name:
            # Check for detected objects/obstacles or landmarks
            if evidence.get("objects") or evidence.get("obstacles") or evidence.get("obstacle_ahead"):
                score += 0.2
            if evidence.get("landmarks"):
                score += 0.1
            if evidence.get("nav_hint"):
                score += 0.1

        # Trajectory opinion quality
        elif "trajectory" in agent_name:
            if evidence.get("distance_traveled", 0) > 1.0:
                score += 0.1
            if evidence.get("corrections"):
                score -= 0.1

        # Instruction opinion quality
        elif "instruction" in agent_name:
            if evidence.get("subtasks"):
                score += 0.1
            if evidence.get("current_subtask"):
                score += 0.1

        return min(1.0, max(0.0, score))

    def _detect_opinion_conflicts(self, opinions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between agent opinions."""
        conflicts = []

        if not opinions:
            return conflicts

        # Action name normalization
        action_normalize = {
            "move_forward": "forward",
            "left": "turn_left",
            "right": "turn_right",
        }

        # Extract recommended actions
        actions = {}
        for agent_name, opinion in opinions.items():
            # Handle DebateOpinion objects
            if hasattr(opinion, 'primary_action'):
                action = opinion.primary_action
            elif isinstance(opinion, dict):
                action = opinion.get("primary_action", "unknown")
            else:
                action = "unknown"
            # Normalize
            action = action_normalize.get(action, action)
            actions[agent_name] = action

        # Check for opposing recommendations
        action_set = set(actions.values())

        # Forward vs Stop conflict
        if "forward" in action_set and "stop" in action_set:
            conflicts.append({
                "type": "action_conflict",
                "agents": [k for k, v in actions.items() if v in ["forward", "stop"]],
                "description": "前进与停止建议冲突",
            })

        # Turn left vs Turn right conflict
        if "turn_left" in action_set and "turn_right" in action_set:
            conflicts.append({
                "type": "direction_conflict",
                "agents": [k for k, v in actions.items() if v in ["turn_left", "turn_right"]],
                "description": "左右转向建议冲突",
            })

        return conflicts

    def _arbitrate_opinions(
        self,
        context: NavContext,
        opinions: Dict[str, Any],
        weights: Dict[str, float],
        opinion_scores: Dict[str, float],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Arbitrate between conflicting opinions and generate final recommendation."""

        # Score each action (use "forward" for consistency)
        action_scores = {
            "forward": 0.0,
            "turn_left": 0.0,
            "turn_right": 0.0,
            "stop": 0.0,
        }

        # Action name normalization
        action_normalize = {
            "move_forward": "forward",
            "left": "turn_left",
            "right": "turn_right",
        }

        for agent_name, opinion in opinions.items():
            # Handle DebateOpinion objects
            if hasattr(opinion, 'primary_action'):
                action = opinion.primary_action
                confidence = opinion.confidence
                weight = weights.get(agent_name.replace("_agent", ""), 0.5)
                quality = opinion_scores.get(agent_name, 0.5)
            # Handle dict opinions
            elif isinstance(opinion, dict):
                action = opinion.get("primary_action", "forward")
                confidence = opinion.get("confidence", 0.5)
                weight = weights.get(agent_name.replace("_agent", ""), 0.5)
                quality = opinion_scores.get(agent_name, 0.5)
            else:
                continue

            # Normalize action name
            action = action_normalize.get(action, action)

            if action in action_scores:
                action_scores[action] += confidence * weight * quality

        # Find best action
        best_action = max(action_scores, key=action_scores.get)
        total_score = sum(action_scores.values())
        confidence = action_scores[best_action] / total_score if total_score > 0 else 0.5

        # Determine blocked actions (low score actions)
        blocked_actions = []
        for action, score in action_scores.items():
            if score == 0.0 and action != best_action:
                blocked_actions.append(action)

        # Generate reasoning
        conflict_str = ""
        if conflicts:
            conflict_str = f"，解决{len(conflicts)}个冲突"

        reasoning = f"综合{len(opinions)}个意见{conflict_str}，推荐{best_action}"

        return {
            "recommended_action": best_action,
            "confidence": confidence,
            "reasoning": reasoning,
            "action_scores": action_scores,
            "blocked_actions": blocked_actions,
            "weight_adjustments": {},
        }


# ============================================================================
# Phase 2: Performance Tracking and Dynamic Scoring for Debate Strategy
# ============================================================================

import os
from dataclasses import dataclass, asdict
from typing import Dict as TypingDict


@dataclass
class AgentPerformance:
    """Historical performance record for an agent.

    Tracks both persistent (cross-episode) and current episode statistics.
    """
    agent_name: str

    # Persistent storage (cross-episode)
    total_opinions: int = 0
    correct_predictions: int = 0        # Predicted action succeeded
    critical_contributions: int = 0     # Key contributions (e.g., obstacle detection)

    # Current episode
    episode_opinions: int = 0
    episode_correct: int = 0
    episode_critical: int = 0

    def get_accuracy(
        self,
        use_persistent: bool = True,
        use_episode: bool = True,
        persistent_weight: float = 0.7
    ) -> float:
        """Calculate composite accuracy score.

        Args:
            use_persistent: Include persistent history
            use_episode: Include current episode
            persistent_weight: Weight for persistent vs episode in hybrid mode

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        scores = []

        if use_persistent and self.total_opinions > 0:
            persistent_acc = self.correct_predictions / self.total_opinions
            scores.append((persistent_acc, persistent_weight))

        if use_episode and self.episode_opinions > 0:
            episode_acc = self.episode_correct / self.episode_opinions
            scores.append((episode_acc, 1.0 - persistent_weight))

        if not scores:
            return 0.5  # Default value

        total_weight = sum(w for _, w in scores)
        return sum(s * w for s, w in scores) / total_weight

    def to_dict(self) -> TypingDict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "total_opinions": self.total_opinions,
            "correct_predictions": self.correct_predictions,
            "critical_contributions": self.critical_contributions,
            "episode_opinions": self.episode_opinions,
            "episode_correct": self.episode_correct,
            "episode_critical": self.episode_critical,
        }


class PerformanceTracker:
    """Tracks historical performance of agents for dynamic weight calculation.

    Supports three modes:
    - persistent: Use only cross-episode history
    - episode: Use only current episode history
    - hybrid: Combine both with configurable weights
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the performance tracker.

        Args:
            config: Configuration dictionary with:
                - weight_mode: "persistent", "episode", or "hybrid"
                - persistent_weight: Weight for persistent history in hybrid mode
                - performance_file: Path to save performance data
        """
        self.config = config or {}
        self.logger = logging.getLogger("PerformanceTracker")

        self.weight_mode = self.config.get("weight_mode", "hybrid")
        self.persistent_weight = self.config.get("persistent_weight", 0.7)
        self.performance_file = self.config.get("performance_file", "data/agent_performance.json")
        self.agents: TypingDict[str, AgentPerformance] = {}

    def load(self) -> None:
        """Load historical performance from file."""
        if os.path.exists(self.performance_file):
            try:
                with open(self.performance_file, 'r') as f:
                    data = json.load(f)
                for name, perf in data.items():
                    self.agents[name] = AgentPerformance(
                        agent_name=name,
                        total_opinions=perf.get("total_opinions", 0),
                        correct_predictions=perf.get("correct_predictions", 0),
                        critical_contributions=perf.get("critical_contributions", 0),
                        episode_opinions=0,  # Reset episode stats on load
                        episode_correct=0,
                        episode_critical=0,
                    )
                self.logger.info(f"[Evaluation] 加载性能数据: {len(self.agents)}个agent")
            except Exception as e:
                self.logger.warning(f"[Evaluation] 加载失败: {e}")

    def save(self) -> None:
        """Save historical performance to file."""
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(
                    {name: perf.to_dict() for name, perf in self.agents.items()},
                    f, indent=2
                )
            self.logger.debug(f"Saved performance data to {self.performance_file}")
        except Exception as e:
            self.logger.warning(f"[Evaluation] 保存失败: {e}")

    def record_opinion(
        self,
        agent_name: str,
        was_correct: bool,
        was_critical: bool = False
    ) -> None:
        """Record the outcome of an agent's opinion.

        Args:
            agent_name: Name of the agent
            was_correct: Whether the predicted action succeeded
            was_critical: Whether this was a critical contribution (e.g., obstacle detection)
        """
        if agent_name not in self.agents:
            self.agents[agent_name] = AgentPerformance(agent_name=agent_name)

        perf = self.agents[agent_name]
        perf.total_opinions += 1
        perf.episode_opinions += 1

        if was_correct:
            perf.correct_predictions += 1
            perf.episode_correct += 1

        if was_critical:
            perf.critical_contributions += 1
            perf.episode_critical += 1

    def get_performance(self, agent_name: str) -> Optional[AgentPerformance]:
        """Get performance record for an agent."""
        return self.agents.get(agent_name)

    def get_accuracy(self, agent_name: str) -> float:
        """Get accuracy for an agent."""
        perf = self.agents.get(agent_name)
        if perf is None:
            return 0.5

        return perf.get_accuracy(
            use_persistent=self.weight_mode in ["persistent", "hybrid"],
            use_episode=self.weight_mode in ["episode", "hybrid"],
            persistent_weight=self.persistent_weight,
        )

    def reset_episode(self) -> None:
        """Reset current episode statistics for all agents."""
        for perf in self.agents.values():
            perf.episode_opinions = 0
            perf.episode_correct = 0
            perf.episode_critical = 0

    def get_summary(self) -> TypingDict[str, Any]:
        """Get summary of all agent performances."""
        return {
            name: {
                "accuracy": perf.get_accuracy(),
                "total_opinions": perf.total_opinions,
                "critical_contributions": perf.critical_contributions,
            }
            for name, perf in self.agents.items()
        }


class DynamicScorer:
    """Dynamic scoring system for debate opinions.

    Calculates action scores based on:
    1. Agent weights (static + dynamic adjustment)
    2. Opinion confidence
    3. Hard constraints (blocking actions)
    4. Soft constraints (weight multipliers)
    """

    BASE_WEIGHTS = {
        "perception": 1.0,
        "instruction": 1.0,
        "trajectory": 1.0,
        "decision": 1.0,
    }

    def __init__(self, performance_tracker: PerformanceTracker):
        """Initialize the dynamic scorer.

        Args:
            performance_tracker: PerformanceTracker instance for dynamic weights
        """
        self.tracker = performance_tracker
        self.logger = logging.getLogger("DynamicScorer")

    def calculate_weights(self) -> TypingDict[str, float]:
        """Calculate dynamic weights for all agents.

        Weights are adjusted based on historical accuracy:
        - Higher accuracy -> higher weight
        - Range: 0.5 * base to 1.5 * base

        Returns:
            Dictionary of agent weights
        """
        weights = {}

        for agent, base_weight in self.BASE_WEIGHTS.items():
            accuracy = self.tracker.get_accuracy(agent)

            # Weight adjustment: 0.5 to 1.5 multiplier based on accuracy
            # accuracy 0.0 -> 0.5x, accuracy 0.5 -> 1.0x, accuracy 1.0 -> 1.5x
            adjustment = 0.5 + accuracy
            weights[agent] = base_weight * adjustment

        return weights

    def score_opinions(
        self,
        opinions: List[Any],  # List[DebateOpinion]
        weights: TypingDict[str, float]
    ) -> TypingDict[str, float]:
        """Score candidate actions based on opinions.

        Args:
            opinions: List of DebateOpinion objects
            weights: Agent weights

        Returns:
            Dictionary of action scores
        """
        # Use "forward" for consistency
        action_scores = {
            "forward": 0.0,
            "turn_left": 0.0,
            "turn_right": 0.0,
            "stop": 0.0,
        }

        # Action normalization
        action_normalize = {
            "move_forward": "forward",
            "left": "turn_left",
            "right": "turn_right",
        }

        # 1. Process hard constraints first
        hard_constraints: TypingDict[str, float] = {}
        for opinion in opinions:
            for constraint in opinion.constraints.get("hard", []):
                if constraint.blocked:
                    action = action_normalize.get(constraint.action, constraint.action)
                    hard_constraints[action] = 0.0

        # 2. Process opinions with soft constraints
        for opinion in opinions:
            agent = opinion.agent
            base_weight = weights.get(agent, 1.0)

            # Apply soft constraint adjustments
            weight = base_weight
            for constraint in opinion.constraints.get("soft", []):
                weight *= constraint.weight_multiplier

            action = opinion.primary_action

            # Normalize action name
            action = action_normalize.get(action, action)

            # Skip if blocked by hard constraint
            if action in hard_constraints:
                continue

            # Add to score
            if action in action_scores:
                action_scores[action] += opinion.confidence * weight

        # 3. Apply hard constraints
        for action, score in hard_constraints.items():
            if action in action_scores:
                action_scores[action] = score

        return action_scores

    def generate_evaluation_output(
        self,
        opinions: List[Any],
        action_scores: TypingDict[str, float],
        weights: TypingDict[str, float]
    ) -> TypingDict[str, Any]:
        """Generate structured evaluation output.

        Args:
            opinions: List of DebateOpinion objects
            action_scores: Calculated action scores
            weights: Weights used

        Returns:
            Evaluation output dictionary
        """
        # Find best action
        best_action = max(action_scores, key=action_scores.get)
        total_score = sum(action_scores.values())
        confidence = action_scores[best_action] / total_score if total_score > 0 else 0.5

        # Find hard constraints applied
        hard_constraints_applied = [
            action for action, score in action_scores.items()
            if score == 0.0
        ]

        # Generate reasoning
        reasoning = self._generate_reasoning(opinions, action_scores, best_action)

        return {
            "evaluation": {
                "weights_used": weights,
                "action_scores": action_scores,
                "best_action": best_action,
                "confidence": confidence,
                "hard_constraints_applied": hard_constraints_applied,
                "reasoning": reasoning,
            }
        }

    def _generate_reasoning(
        self,
        opinions: List[Any],
        action_scores: TypingDict[str, float],
        best_action: str
    ) -> str:
        """Generate human-readable reasoning for the decision."""
        # Find supporting opinions for best action
        supporting = []

        # Action normalization
        action_normalize = {
            "move_forward": "forward",
            "left": "turn_left",
            "right": "turn_right",
        }

        for opinion in opinions:
            action = opinion.primary_action
            action = action_normalize.get(action, action)

            if action == best_action:
                supporting.append(f"{opinion.agent}: {opinion.reasoning[:50]}")

        if supporting:
            return f"选择{best_action} - " + "; ".join(supporting[:2])

        return f"选择{best_action} (得分: {action_scores[best_action]:.2f})"


def get_performance_tracker(config: Dict[str, Any] = None) -> PerformanceTracker:
    """Factory function to get or create a PerformanceTracker instance.

    Args:
        config: Configuration dictionary

    Returns:
        PerformanceTracker instance
    """
    tracker = PerformanceTracker(config)
    tracker.load()
    return tracker