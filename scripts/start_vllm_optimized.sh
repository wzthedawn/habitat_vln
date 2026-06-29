#!/bin/bash
# Optimized vLLM server startup script for VLN experiments
# Usage: ./scripts/start_vllm_optimized.sh [model_path]

MODEL_PATH="${1:-/data/WZ/Model/Qwen/Qwen3.6-35B-A3B}"
PORT="${2:-8000}"
TP_SIZE="${3:-2}"

echo "Starting optimized vLLM server..."
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Tensor Parallel: $TP_SIZE"

# Kill existing server
pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null
sleep 2

# Activate environment
source ~/.bashrc
conda activate vllm_env

# Start server with optimizations
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --trust-remote-code \
    --quantization fp8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --tensor-parallel-size $TP_SIZE \
    --dtype auto \
    --enforce-eager \
    \
    `# === Optimization parameters ===` \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    \
    `# === Logging ===` \
    --disable-log-stats \
    2>&1 | tee logs/vllm_server_$PORT.log &

echo "Server starting... waiting for ready"
sleep 5

# Check if server is ready
for i in {1..30}; do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server ready at http://localhost:$PORT"
        exit 0
    fi
    sleep 1
done

echo "✗ Server failed to start within 30s"
exit 1