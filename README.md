# Multi-Agent VLN Navigation System

A hierarchical multi-agent collaborative vision-language navigation system based on the ideas from DiscussNav, MSNav, and Multi-agent Architecture Search via Agentic Supernet papers.

## Features

- **Hierarchical Architecture**: Weak level (local small model) + Strong level (multi-agent collaboration)
- **Task-Driven Selection**: Automatic agent and strategy selection based on task complexity
- **Token Optimization**: Layered prompts, context compression, and cache reuse
- **Multiple Strategies**: ReAct, Chain of Thought, Debate, and Reflection strategies

## Project Structure

```
habitat_vln/
├── configs/           # Configuration files
├── core/              # Core modules (context, action, navigator)
├── classifiers/       # Task type classifiers
├── supernet/          # Agent-strategy orchestration
├── agents/            # Specialized agents
├── strategies/        # Navigation strategies
├── models/            # Model implementations
├── optimization/      # Token optimization
├── fallback/          # Failure handling
├── environment/       # Habitat integration
├── utils/             # Utilities
├── scripts/           # Training and evaluation scripts
└── tests/             # Unit tests
```

## Task Type Classification

| Type | Name | Description | Agents | Strategies |
|------|------|-------------|--------|------------|
| Type-0 | Simple Navigation | Single step instruction | None | None |
| Type-1 | Path Following | Corridor navigation | perception + decision | ReAct |
| Type-2 | Target Search | Object finding | perception + trajectory + decision | ReAct + CoT |
| Type-3 | Spatial Reasoning | Cross-room navigation | All | CoT + Reflection |
| Type-4 | Complex Decision | Ambiguous scenes | All | CoT + Debate + Reflection |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from core.navigator import VLNNavigator

# Initialize navigator
navigator = VLNNavigator()
navigator.initialize()

# Set instruction and navigate
navigator.set_instruction("turn left and go to the kitchen")
action = navigator.navigate()

print(f"Action: {action.action_type.name}")
```

### Training

```bash
python scripts/train.py --config configs/default.yaml --episodes 1000
```

### Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml --episodes 100
```

### Inference

```bash
python scripts/inference.py --instruction "find the red chair"
```

## Configuration

Configuration files are located in `configs/`:

- `default.yaml`: Main configuration
- `model_config.yaml`: Model settings
- `architecture_config.yaml`: Agent-strategy mapping

## Testing

```bash
pytest tests/
```

## Architecture

The system uses a hierarchical architecture:

1. **Task Classifier**: Determines task complexity (Type-0 to Type-4)
2. **Supernet**: Orchestrates agent-strategy combinations
3. **Agents**: Instruction, Perception, Trajectory, Decision
4. **Strategies**: ReAct, CoT, Debate, Reflection
5. **Fallback**: Cascading degradation for failures

## License

MIT License