"""Debate strategy implementation."""

from typing import Dict, Any, List, Optional
import logging

from .base_strategy import BaseStrategy, StrategyResult, StrategyType
from core.context import NavContext
from core.action import Action
from agents.base_agent import BaseAgent


class DebateStrategy(BaseStrategy):
    """
    Debate strategy: Multi-agent debate for decision making.

    Pattern: Proposal → Arguments → Counter-arguments → Resolution

    This strategy enables robust decision-making through structured
    debate between different perspectives (agents).
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.logger = logging.getLogger("DebateStrategy")

        # Configuration
        self.max_rounds = self.config.get("max_rounds", 3)
        self.consensus_threshold = self.config.get("consensus_threshold", 0.7)
        self.require_unanimous = self.config.get("require_unanimous", False)

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
        Execute Debate strategy.

        Args:
            context: Navigation context
            agents: List of available agents
            prev_result: Optional previous strategy result

        Returns:
            StrategyResult with debated action
        """
        self.initialize()

        steps = []

        try:
            # Initial proposals from all agents
            proposals = self._gather_proposals(context, agents)
            steps.append({
                "type": "initial_proposals",
                "proposals": [p["action"] for p in proposals],
            })

            # Run debate rounds
            for round_num in range(self.max_rounds):
                debate_round = self._run_debate_round(
                    context, agents, proposals, round_num
                )
                steps.append({
                    "type": "debate_round",
                    "round": round_num + 1,
                    "arguments": debate_round["arguments"],
                    "updated_proposals": debate_round["proposals"],
                })

                proposals = debate_round["proposals"]

                # Check for consensus
                if self._check_consensus(proposals):
                    steps.append({
                        "type": "consensus",
                        "round": round_num + 1,
                    })
                    break

            # Final resolution
            action, confidence, reasoning = self._resolve_debate(proposals, steps)
            steps.append({
                "type": "resolution",
                "action": action.to_habitat_action(),
                "confidence": confidence,
                "reasoning": reasoning,
            })

            return StrategyResult(
                success=True,
                action=action,
                reasoning=reasoning,
                steps=steps,
                confidence=confidence,
            )

        except Exception as e:
            self.logger.error(f"Debate execution error: {e}")
            return StrategyResult(
                success=False,
                reasoning=f"Debate failed: {str(e)}",
                steps=steps,
            )

    def _gather_proposals(
        self, context: NavContext, agents: List[BaseAgent]
    ) -> List[Dict[str, Any]]:
        """Gather initial action proposals from all agents."""
        proposals = []

        for agent in agents:
            try:
                output = agent.process(context)
                if output.success:
                    action_str = output.data.get("action", "forward")
                    proposals.append({
                        "agent": agent.name,
                        "action": action_str,
                        "confidence": output.confidence,
                        "reasoning": output.reasoning,
                    })
            except Exception as e:
                self.logger.warning(f"Agent {agent.name} failed to propose: {e}")

        if not proposals:
            # Default proposal if all agents fail
            proposals.append({
                "agent": "default",
                "action": "forward",
                "confidence": 0.5,
                "reasoning": "Default action due to no proposals",
            })

        return proposals

    def _run_debate_round(
        self,
        context: NavContext,
        agents: List[BaseAgent],
        proposals: List[Dict],
        round_num: int,
    ) -> Dict[str, Any]:
        """Run a single debate round."""
        arguments = []

        # Each agent can argue for their proposal
        for proposal in proposals:
            argument = self._generate_argument(
                context, proposal, proposals, round_num
            )
            arguments.append({
                "agent": proposal["agent"],
                "argument": argument,
                "action": proposal["action"],
            })

        # Update proposals based on arguments
        updated_proposals = self._update_proposals(proposals, arguments)

        return {
            "arguments": arguments,
            "proposals": updated_proposals,
        }

    def _generate_argument(
        self,
        context: NavContext,
        proposal: Dict,
        all_proposals: List[Dict],
        round_num: int,
    ) -> str:
        """Generate argument for a proposal."""
        action = proposal["action"]
        agent = proposal["agent"]
        reasoning = proposal.get("reasoning", "")

        # Base argument
        argument = f"{agent} argues for '{action}' because: {reasoning}"

        # Add context-aware argumentation
        instruction_lower = context.instruction.lower()

        if action == "turn_left" and "left" in instruction_lower:
            argument += " This aligns with the instruction to turn left."
        elif action == "turn_right" and "right" in instruction_lower:
            argument += " This aligns with the instruction to turn right."
        elif action == "forward":
            argument += " Forward movement is safe and progresses toward goal."
        elif action == "stop":
            argument += " Current position may satisfy the goal condition."

        # Consider other proposals
        opposing = [p for p in all_proposals if p["action"] != action]
        if opposing and round_num > 0:
            argument += f" However, {len(opposing)} agent(s) suggest alternatives."

        return argument

    def _update_proposals(
        self, proposals: List[Dict], arguments: List[Dict]
    ) -> List[Dict]:
        """Update proposals based on arguments."""
        updated = []

        for proposal in proposals:
            # Find matching argument
            arg = next(
                (a for a in arguments if a["agent"] == proposal["agent"]),
                None
            )

            updated_proposal = proposal.copy()

            # Adjust confidence based on argument strength
            if arg:
                # Simple heuristic: if argument mentions alignment, boost confidence
                if "aligns with" in arg["argument"]:
                    updated_proposal["confidence"] = min(
                        proposal["confidence"] + 0.1, 1.0
                    )
                elif "However" in arg["argument"]:
                    updated_proposal["confidence"] = max(
                        proposal["confidence"] - 0.1, 0.3
                    )

            updated.append(updated_proposal)

        return updated

    def _check_consensus(self, proposals: List[Dict]) -> bool:
        """Check if proposals have reached consensus."""
        if not proposals:
            return False

        # Count action votes
        action_counts: Dict[str, float] = {}
        for proposal in proposals:
            action = proposal["action"]
            conf = proposal["confidence"]
            action_counts[action] = action_counts.get(action, 0) + conf

        # Find most voted action
        if action_counts:
            max_votes = max(action_counts.values())
            total_votes = sum(action_counts.values())

            if self.require_unanimous:
                return len(action_counts) == 1

            return max_votes / total_votes >= self.consensus_threshold

        return False

    def _resolve_debate(
        self, proposals: List[Dict], steps: List[Dict]
    ) -> tuple:
        """Resolve debate and select final action."""
        if not proposals:
            return Action.forward(), 0.5, "No proposals to resolve"

        # Weight actions by confidence
        action_weights: Dict[str, float] = {}
        action_reasons: Dict[str, List[str]] = {}

        for proposal in proposals:
            action = proposal["action"]
            weight = proposal["confidence"]

            # Agent weight based on role importance
            agent = proposal["agent"]
            if "decision" in agent:
                weight *= 1.5  # Decision agent has more weight
            elif "perception" in agent:
                weight *= 1.2

            action_weights[action] = action_weights.get(action, 0) + weight

            if action not in action_reasons:
                action_reasons[action] = []
            action_reasons[action].append(proposal.get("reasoning", ""))

        # Select action with highest weight
        best_action = max(action_weights.keys(), key=lambda a: action_weights[a])
        confidence = action_weights[best_action] / sum(action_weights.values())
        reasons = action_reasons.get(best_action, [])

        # Create action and reasoning
        action = self._action_from_string(best_action)
        reasoning = f"Debate resolved: {best_action} (confidence: {confidence:.2f})"
        if reasons:
            reasoning += f". Key reason: {reasons[0]}"

        return action, confidence, reasoning

    def _action_from_string(self, action_str: str) -> Action:
        """Convert action string to Action object."""
        action_map = {
            "stop": Action.stop(),
            "move_forward": Action.forward(),
            "forward": Action.forward(),
            "turn_left": Action.turn_left(),
            "turn_right": Action.turn_right(),
        }
        return action_map.get(action_str.lower(), Action.forward())