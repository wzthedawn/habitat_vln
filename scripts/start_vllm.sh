#!/bin/bash
# Start vLLM server with multi-model support
# Usage: bash scripts/start_vllm.sh --model [9b|35b|vl8b] --port [8000]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_TYPE="${1:-9b}"
PORT="${2:-8000}"

case $MODEL_TYPE in
    vl8b)
        # Qwen3-VL-8B: dedicated VLM for perception
        MODEL_PATH="/data/WZ/Model/Qwen/Qwen3-VL-8B-Instruct"
        GPU_IDS="0"
        TENSOR_PARALLEL=1
        GPU_MEMORY=0.85
        MAX_LEN=8192
        ;;
    9b)
        # Qwen3.5-9B: fast LLM for simple tasks
        MODEL_PATH="/data/WZ/Model/Qwen/Qwen3___5-9B"
        GPU_IDS="0"
        TENSOR_PARALLEL=1
        GPU_MEMORY=0.85
        MAX_LEN=4096
        ;;
    35b)
        # Qwen3.6-35B-A3B: strong MoE LLM for complex reasoning
        MODEL_PATH="/data/WZ/Model/Qwen/Qwen3.6-35B-A3B"
        GPU_IDS="0,1"
        TENSOR_PARALLEL=2
        GPU_MEMORY=0.85
        MAX_LEN=4096
        ;;
    *)
        echo "Usage: $0 [vl8b|9b|35b] [port]"
        echo "  vl8b - Qwen3-VL-8B-Instruct (VLM perception)"
        echo "  9b   - Qwen3.5-9B (fast LLM)"
        echo "  35b  - Qwen3.6-35B-A3B (strong MoE, needs 2 GPUs)"
        exit 1
        ;;
esac

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: model not found at $MODEL_PATH"
    exit 1
fi

echo "=== Starting vLLM Server ==="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "GPU: $GPU_IDS | Port: $PORT"
echo "=============================="

CUDA_VISIBLE_DEVICES=$GPU_IDS vllm serve "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len $MAX_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --trust-remote-code \
    --dtype auto \
    --enforce-eager
