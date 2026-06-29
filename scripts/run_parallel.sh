#!/bin/bash
# Parallel multi-GPU VLN-CE evaluation
# Splits episodes across 4 GPUs, runs independently, merges results.
#
# Usage: bash scripts/run_parallel.sh [total_episodes] [options]
#   bash scripts/run_parallel.sh 60 --max-steps 100 --seed 42 --shuffle
#   bash scripts/run_parallel.sh 30 --resolution 256 --output-dir results/exp1
#
# Each GPU runs total_episodes/4 episodes with --start-episode offset.

set -e

TOTAL_EP=${1:-60}
shift 2>/dev/null || true
COMMON_ARGS="$@"
EP_PER_GPU=$(( (TOTAL_EP + 3) / 4 ))  # ceiling division

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_DIR="${OUT_DIR:-results/parallel_$TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "  Parallel VLN-CE Evaluation"
echo "  Total: $TOTAL_EP episodes / 4 GPUs = $EP_PER_GPU/GPU"
echo "  Output: $OUT_DIR"
echo "  Args: $COMMON_ARGS"
echo "============================================"

PIDS=()
for GPU in 0 1 2 3; do
    START_EP=$(( GPU * EP_PER_GPU * 10 ))  # spread starts to avoid same episodes
    LOG="$OUT_DIR/gpu${GPU}.log"

    echo "[GPU $GPU] Starting episodes from $START_EP, $EP_PER_GPU episodes..."

    CUDA_VISIBLE_DEVICES=$GPU python run_vln_experiment.py \
        --use-remote-llm \
        --episodes $EP_PER_GPU \
        --start-episode $START_EP \
        --output "$OUT_DIR/gpu${GPU}_results.json" \
        --output-dir "$OUT_DIR/gpu${GPU}" \
        $COMMON_ARGS \
        > "$LOG" 2>&1 &

    PIDS+=($!)
    echo "[GPU $GPU] PID: ${PIDS[-1]}"
done

echo ""
echo "All GPUs launched. Waiting for completion..."
echo "Monitor: tail -f $OUT_DIR/gpu*.log"
echo ""

# Wait for all processes
FAILED=0
for i in 0 1 2 3; do
    wait ${PIDS[$i]} || { echo "[GPU $i] FAILED (exit code $?)"; FAILED=1; }
    echo "[GPU $i] Done"
done

echo ""
echo "============================================"
echo "  Merging results..."
echo "============================================"

python3 -c "
import json, glob, os

out_dir = '$OUT_DIR'
all_eps = []
summaries = []

for f in sorted(glob.glob(f'{out_dir}/gpu*_results.json')):
    if not os.path.exists(f):
        print(f'  WARNING: {f} not found, skipping')
        continue
    with open(f) as fh:
        data = json.load(fh)
    all_eps.extend(data.get('episodes', []))
    summaries.append(data.get('summary', {}))

# Deduplicate by episode_id
seen = set()
unique_eps = []
for ep in all_eps:
    if ep['episode_id'] not in seen:
        seen.add(ep['episode_id'])
        unique_eps.append(ep)

total = len(unique_eps)
successes = sum(1 for e in unique_eps if e['success'])
sr = successes / total * 100 if total > 0 else 0
oracle = sum(1 for e in unique_eps if e['oracle_success']) / total * 100 if total > 0 else 0
avg_spl = sum(e['spl'] for e in unique_eps) / total if total > 0 else 0
avg_ne = sum(e['distance_to_goal'] for e in unique_eps) / total if total > 0 else 0
avg_steps = sum(e['steps'] for e in unique_eps) / total if total > 0 else 0

merged = {
    'summary': {
        'num_episodes': total,
        'success_rate': successes/total if total>0 else 0,
        'spl': avg_spl,
        'oracle_success_rate': oracle/100,
        'nDTW': sum(e['nDTW'] for e in unique_eps)/total if total>0 else 0,
        'SDTW': sum(e.get('SDTW',0) for e in unique_eps)/total if total>0 else 0,
        'avg_distance_to_goal': avg_ne,
        'avg_steps': avg_steps,
    },
    'episodes': unique_eps,
}

out_file = f'{out_dir}/merged_results.json'
with open(out_file, 'w') as f:
    json.dump(merged, f, indent=2)

print(f'  Merged: {total} unique episodes')
print(f'  SR: {sr:.1f}% | SPL: {avg_spl:.3f} | NE: {avg_ne:.2f}m | Steps: {avg_steps:.0f}')
print(f'  Saved: {out_file}')
"

echo ""
echo "============================================"
echo "  Complete!"
echo "  Results: $OUT_DIR/merged_results.json"
echo "============================================"
