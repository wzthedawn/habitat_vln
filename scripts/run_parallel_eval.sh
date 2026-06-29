#!/bin/bash
# scripts/run_parallel_eval.sh
#
# Parallel evaluation script using 4 GPUs
# GPU allocation:
#   GPU 0: vLLM server + exp-c
#   GPU 1: baseline
#   GPU 2: exp-a
#   GPU 3: exp-b

set -e

# Configuration
EPISODES=${1:-10}
NO_VIDEO="--no-video --no-trajectory"
REMOTE_LLM="--use-remote-llm --llm-server http://localhost:8000"
OUTPUT_DIR="results/emergency_eval_parallel"

echo "========================================"
echo "Parallel Emergency Evaluation"
echo "========================================"
echo "Episodes per experiment: $EPISODES"
echo "Video/Trajectory: DISABLED"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Create output directory
mkdir -p $OUTPUT_DIR

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Start 4 experiments in parallel
echo "Starting experiments..."

# GPU 1: baseline
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp baseline --episodes $EPISODES --output "$OUTPUT_DIR/baseline" $NO_VIDEO $REMOTE_LLM \
    2>&1 | tee "$OUTPUT_DIR/baseline_log.txt" &
PID1=$!
echo "Started baseline on GPU 1 (PID: $PID1)"

# GPU 2: exp-a (PathReplanner only)
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp exp-a --episodes $EPISODES --output "$OUTPUT_DIR/exp-a" $NO_VIDEO $REMOTE_LLM \
    2>&1 | tee "$OUTPUT_DIR/exp-a_log.txt" &
PID2=$!
echo "Started exp-a on GPU 2 (PID: $PID2)"

# GPU 3: exp-b (PathReplanner + LoRA)
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp exp-b --episodes $EPISODES --output "$OUTPUT_DIR/exp-b" $NO_VIDEO $REMOTE_LLM \
    2>&1 | tee "$OUTPUT_DIR/exp-b_log.txt" &
PID3=$!
echo "Started exp-b on GPU 3 (PID: $PID3)"

# GPU 0: exp-c (LoRA only, shares with vLLM)
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/run_emergency_eval.py" \
    --exp exp-c --episodes $EPISODES --output "$OUTPUT_DIR/exp-c" $NO_VIDEO $REMOTE_LLM \
    2>&1 | tee "$OUTPUT_DIR/exp-c_log.txt" &
PID4=$!
echo "Started exp-c on GPU 0 (PID: $PID4)"

echo ""
echo "All 4 experiments running in parallel."
echo "Waiting for completion..."
echo ""

# Wait for all processes
wait $PID1
echo "baseline completed (exit code: $?)"

wait $PID2
echo "exp-a completed (exit code: $?)"

wait $PID3
echo "exp-b completed (exit code: $?)"

wait $PID4
echo "exp-c completed (exit code: $?)"

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No JSON summary files found"