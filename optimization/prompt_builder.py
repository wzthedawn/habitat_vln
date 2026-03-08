"""Prompt builder for constructing LLM prompts."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from core.context import NavContext
from optimization.context_compressor import ContextCompressor


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    name: str
    system_prompt: str
    user_template: str
    examples: List[Dict[str, str]]


class PromptBuilder:
    """
    Prompt Builder for constructing LLM prompts.

    Builds optimized prompts for different navigation tasks
    with proper context inclusion and formatting.
    """

    # Pre-defined prompt templates
    TEMPLATES = {
        "navigation": PromptTemplate(
            name="navigation",
            system_prompt="You are a navigation assistant that helps agents navigate indoor environments.",
            user_template="Given the current state, determine the next action.\n\n{context}\n\nWhat action should be taken?",
            examples=[],
        ),

        "classification": PromptTemplate(
            name="classification",
            system_prompt="You are a task classifier for navigation instructions.",
            user_template='Classify the following navigation task:\n\nInstruction: "{instruction}"\n\nClassify as Type-0 (simple), Type-1 (path), Type-2 (search), Type-3 (multi-room), or Type-4 (complex).',
            examples=[],
        ),

        "reasoning": PromptTemplate(
            name="reasoning",
            system_prompt="You are a navigation reasoning engine. Think step by step about navigation decisions.",
            user_template="Analyze the navigation situation:\n\n{context}\n\nThink through what action to take and why.",
            examples=[],
        ),

        "perception": PromptTemplate(
            name="perception",
            system_prompt="You are a visual perception analyzer for navigation.",
            user_template="Analyze the following observation:\n\n{observation}\n\nWhat objects and landmarks are visible?",
            examples=[],
        ),

        "planning": PromptTemplate(
            name="planning",
            system_prompt="You are a navigation planner that breaks down complex instructions into subtasks.",
            user_template='Break down this navigation instruction into subtasks:\n\n"{instruction}"\n\nList each subtask clearly.',
            examples=[],
        ),
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize prompt builder.

        Args:
            config: Builder configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger("PromptBuilder")

        # Context compressor
        self.compressor = ContextCompressor(self.config.get("compression", {}))

        # Custom templates
        self._custom_templates: Dict[str, PromptTemplate] = {}

        # Prompt cache
        self._cache: Dict[str, str] = {}

    def build(
        self,
        template_name: str,
        context: NavContext = None,
        variables: Dict[str, Any] = None,
        compression_level: str = "standard",
    ) -> str:
        """
        Build a prompt from template.

        Args:
            template_name: Name of the template to use
            context: Navigation context (optional)
            variables: Additional template variables
            compression_level: Context compression level

        Returns:
            Built prompt string
        """
        # Get template
        template = self._get_template(template_name)

        if template is None:
            self.logger.warning(f"Template {template_name} not found")
            return ""

        # Build variables
        prompt_vars = variables or {}

        # Add compressed context if available
        if context:
            prompt_vars["context"] = self.compressor.compress(context, compression_level)
            prompt_vars["instruction"] = context.instruction
            prompt_vars["step"] = context.step_count

        # Format user template
        try:
            user_prompt = template.user_template.format(**prompt_vars)
        except KeyError as e:
            self.logger.warning(f"Missing variable in template: {e}")
            user_prompt = template.user_template

        # Combine system and user prompts
        if template.system_prompt:
            full_prompt = f"{template.system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        return full_prompt

    def build_with_examples(
        self,
        template_name: str,
        context: NavContext = None,
        variables: Dict[str, Any] = None,
        num_examples: int = 2,
    ) -> str:
        """
        Build prompt with few-shot examples.

        Args:
            template_name: Template name
            context: Navigation context
            variables: Template variables
            num_examples: Number of examples to include

        Returns:
            Built prompt with examples
        """
        template = self._get_template(template_name)

        # Build base prompt
        base_prompt = self.build(template_name, context, variables)

        if template and template.examples:
            # Add examples
            example_text = "\n\nExamples:\n"
            for i, example in enumerate(template.examples[:num_examples]):
                example_text += f"\n{i+1}. Input: {example.get('input', '')}\n   Output: {example.get('output', '')}"

            base_prompt = example_text + "\n\n" + base_prompt

        return base_prompt

    def build_chain_of_thought(
        self,
        template_name: str,
        context: NavContext,
        reasoning_steps: int = 3,
    ) -> str:
        """
        Build chain-of-thought prompt.

        Args:
            template_name: Template name
            context: Navigation context
            reasoning_steps: Number of reasoning steps to prompt

        Returns:
            Chain-of-thought prompt
        """
        base_prompt = self.build(template_name, context, compression_level="detailed")

        # Add reasoning structure
        cot_prompt = base_prompt + "\n\nLet's think step by step:"

        for i in range(reasoning_steps):
            cot_prompt += f"\n{i+1}. [Reasoning step {i+1}]"

        cot_prompt += "\n\nBased on this reasoning, the action is:"

        return cot_prompt

    def build_debate_prompt(
        self,
        context: NavContext,
        proposals: List[Dict[str, Any]],
        round_num: int,
    ) -> str:
        """
        Build prompt for debate strategy.

        Args:
            context: Navigation context
            proposals: List of action proposals
            round_num: Current debate round

        Returns:
            Debate prompt
        """
        prompt = f"Navigation Debate - Round {round_num}\n\n"
        prompt += f"Instruction: {context.instruction}\n\n"

        prompt += "Proposed actions:\n"
        for i, proposal in enumerate(proposals):
            prompt += f"{i+1}. {proposal.get('agent')}: {proposal.get('action')} "
            prompt += f"(confidence: {proposal.get('confidence', 0):.2f})\n"

        prompt += "\nAnalyze each proposal and argue for or against them. "
        prompt += "What is the best action to take?"

        return prompt

    def build_reflection_prompt(
        self,
        context: NavContext,
        recent_outcome: str,
    ) -> str:
        """
        Build prompt for reflection strategy.

        Args:
            context: Navigation context
            recent_outcome: Description of recent outcome

        Returns:
            Reflection prompt
        """
        prompt = "Navigation Reflection\n\n"
        prompt += f"Instruction: {context.instruction}\n"
        prompt += f"Current step: {context.step_count}\n\n"

        prompt += f"Recent outcome: {recent_outcome}\n\n"

        if context.action_history:
            prompt += "Recent actions:\n"
            for action in context.action_history[-5:]:
                prompt += f"  - {action.action_type.name}\n"
            prompt += "\n"

        prompt += "Reflect on the navigation so far:\n"
        prompt += "1. What went well?\n"
        prompt += "2. What could be improved?\n"
        prompt += "3. What should be done differently?"

        return prompt

    def add_template(self, template: PromptTemplate) -> None:
        """Add a custom template."""
        self._custom_templates[template.name] = template

    def _get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        # Check custom templates first
        if name in self._custom_templates:
            return self._custom_templates[name]

        # Check built-in templates
        return self.TEMPLATES.get(name)

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.TEMPLATES.keys()) + list(self._custom_templates.keys())

    def clear_cache(self) -> None:
        """Clear prompt cache."""
        self._cache.clear()


class PromptBuilderUtils:
    """Utility functions for prompt building."""

    @staticmethod
    def format_action_options(actions: List[str]) -> str:
        """Format action options for prompt."""
        return " | ".join(actions)

    @staticmethod
    def format_history(history: List, max_items: int = 5) -> str:
        """Format action history for prompt."""
        if not history:
            return "No previous actions"

        items = [h.action_type.name for h in history[-max_items:]]
        return " → ".join(items)

    @staticmethod
    def format_subtasks(subtasks: List) -> str:
        """Format subtasks for prompt."""
        if not subtasks:
            return "No subtasks"

        lines = []
        for i, st in enumerate(subtasks):
            status = "✓" if st.status == "completed" else "○"
            lines.append(f"{status} {i+1}. {st.description}")

        return "\n".join(lines)

    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."