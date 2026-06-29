#!/bin/bash
# Multi-model vLLM launcher for 4x RTX 4090 deployment
#
# GPU allocation:
#   GPU 0: Qwen3-VL-8B-Instruct  (17GB) → port 8000  VLM perception
#   GPU 1: Qwen3.5-9B-AWQ         (12GB) → port 8001  fast LLM
#   GPU 2+3: Qwen3.6-35B-A3B FP8  (~34GB) → port 8002  strong LLM
#
# Usage: bash scripts/start_vllm_multi.sh [--no-35b]

set -e

NO_35B=false
for arg in "$@"; do
    case $arg in
        --no-35b) NO_35B=true ;;
        --no-vl) NO_VL=true ;;
    esac
done

echo "============================================"
echo "  Multi-Model vLLM Launcher"
echo "  4x RTX 4090 24GB Deployment"
echo "============================================"

# ---- Server 1: VLM (GPU 0, port 8000) ----
echo ""
echo "[1/3] Starting Qwen3-VL-8B VLM server on GPU 0:8000..."
CUDA_VISIBLE_DEVICES=0 vllm serve /data/WZ/Model/Qwen/Qwen3-VL-8B-Instruct \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --trust-remote-code --dtype auto --enforce-eager \
    --served-model-name qwen3-vl-8b \
    > logs/vllm_vl8b.log 2>&1 &
PID_VL=$!
echo "  PID: $PID_VL"

# ---- Server 2: Fast LLM (GPU 1, port 8001) ----
echo ""
echo "[2/3] Starting Qwen3.5-9B fast LLM server on GPU 1:8001..."
CUDA_VISIBLE_DEVICES=1 vllm serve /data/WZ/Model/Qwen/Qwen3___5-9b_AWQ \
    --port 8001 --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --trust-remote-code --dtype auto --enforce-eager \
    --served-model-name qwen3.5-9b-fast \
    > logs/vllm_9b.log 2>&1 &
PID_9B=$!
echo "  PID: $PID_9B"

# ---- Server 3: Strong LLM (GPU 2+3, port 8002, FP8) ----
if [ "$NO_35B" = false ]; then
    echo ""
    echo "[3/3] Starting Qwen3.6-35B strong LLM server on GPU 2,3:8002 (FP8)..."
    CUDA_VISIBLE_DEVICES=2,3 vllm serve /data/WZ/Model/Qwen/Qwen3.6-35B-A3B \
        --port 8002 --host 0.0.0.0 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --tensor-parallel-size 2 \
        --quantization fp8 \
        --trust-remote-code --dtype auto --enforce-eager \
        --served-model-name qwen3.6-35b-strong \
        > logs/vllm_35b.log 2>&1 &
    PID_35B=$!
    echo "  PID: $PID_35B"
else
    echo ""
    echo "[3/3] Skipping 35B (--no-35b)"
fi

echo ""
echo "============================================"
echo "  All servers launched. Waiting for readiness..."
echo "============================================"

# Wait for servers to be ready
for port in 8000 8001; do
    echo -n "  Port $port: "
    for i in $(seq 1 60); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "READY"
            break
        fi
        sleep 2
    done
done

if [ "$NO_35B" = false ]; then
    echo -n "  Port 8002: "
    for i in $(seq 1 120); do
        if curl -s http://localhost:8002/health > /dev/null 2>&1; then
            echo "READY"
            break
        fi
        sleep 2
    done
fi

echo ""
echo "============================================"
echo "  All servers ready!"
echo "  VLM:      http://localhost:8000 (GPU 0)"
echo "  Fast LLM: http://localhost:8001 (GPU 1)"
if [ "$NO_35B" = false ]; then
    echo "  Strong:   http://localhost:8002 (GPU 2+3, FP8)"
fi
echo "============================================"
echo ""
echo "PIDs: VL=$PID_VL 9B=$PID_9B"
[ -n "$PID_35B" ] && echo "PID: 35B=$PID_35B"
echo ""
echo "Logs: logs/vllm_vl8b.log  logs/vllm_9b.log"
[ -n "$PID_35B" ] && echo "      logs/vllm_35b.log"
echo ""
echo "To stop: kill $PID_VL $PID_9B $PID_35B"
