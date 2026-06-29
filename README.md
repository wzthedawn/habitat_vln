# Pipeline VLN Navigation System

Hierarchical vision-language navigation system with multi-tier model allocation.

## Architecture

```
ObservationAgent (Qwen3-VL-8B)  →  EmergencyAgent  →  AnalysisAgent (CoT/Debate/Reflection)
        ↓                              ↓                        ↓
  Structured JSON              Depth obstacle check      Qwen3.6-35B strong LLM
        ↓                                                     ↓
  Dynamic Difficulty  ←──────────────────────────→  PlanningAgent (LLM + Topology + A*)
                                                            ↓
                                                     ActionConverter (5 actions)
                                                            ↓
                                                      ReviewAgent (rule + LLM)
```

## Project Structure

```
habitat_vln/
├── agents/                  # Pipeline Agent architecture
│   ├── base_agent.py        # Shared base class
│   └── pipeline/
│       ├── navigator.py     # Orchestrator + difficulty grading
│       ├── observation_agent.py    # VLM perception (Qwen3-VL-8B)
│       ├── analysis_agent.py      # CoT/Debate/Reflection reasoning
│       ├── planning_agent.py      # LLM + topology + A* planning
│       ├── review_agent.py        # Completion verification
│       ├── emergency_agent.py     # Obstacle detection & handling
│       ├── subtask_decomposition_agent.py
│       └── tools/                 # ActionConverter, TopologyGraph, etc.
├── core/                    # Action, Context, EscapePlanner
├── models/                  # ModelManager, RemoteLLMClient
├── environment/             # Habitat simulator integration
├── utils/                   # Metrics, logging, visualization
├── configs/                 # YAML configuration files
├── data/                    # Dataset preparation scripts
├── scripts/                 # Entry point scripts
│   ├── start_vllm.sh        # vLLM server launcher
│   ├── download_qwen.py     # Model download tool
│   └── download_r2r.py      # R2R dataset download
├── envs/                    # Conda environment files
├── docs/                    # Design docs & specifications
├── tests/                   # Unit tests
├── run_vln_experiment.py    # Main experiment runner
├── vllm_server.py           # vLLM inference server
└── setup.py                 # Package setup
```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environments
conda env create -f envs/habitat_env.yml      # Python 3.9 + Habitat
conda env create -f envs/vllm_env.yml         # Python 3.10 + vLLM

# Install package
pip install -e .
```

### 2. Download Models

```bash
python scripts/download_qwen.py --model qwen3-vl-8b --output /data/WZ/Model/Qwen/
```

### 3. Start vLLM Server

```bash
bash scripts/start_vllm.sh vl8b 8000
```

### 4. Run Experiment

```bash
conda activate Habitat
python run_vln_experiment.py \
    --use-remote-llm \
    --llm-server http://localhost:8000 \
    --episodes 10 \
    --seed 42
```

## Model Tiers

| Tier | Model | Size | Purpose |
|------|-------|------|---------|
| VLM | Qwen3-VL-8B-Instruct | 17GB | Structured scene perception |
| Fast LLM | Qwen3.5-9B-AWQ | 12GB | Decomposition, review, emergency |
| Strong LLM | Qwen3.6-35B-A3B | 67GB | CoT, Debate, Reflection, Planning |

## Difficulty-Graded Strategy

| Difficulty | Strategy | LLM Calls | Model |
|-----------|----------|-----------|-------|
| easy | Rule (skip LLM) | 0 | None |
| medium | CoT | 1 | 35B |
| hard (1st) | Debate Light | 2 | 35B |
| hard (2nd) | Debate Standard | 3 | 35B |
| hard (3rd+) | Debate Deep | 4-5 | 35B |

## License

MIT License
