#!/bin/bash
set -e

# 切换到项目目录
cd /home/WZ/MA_VLN/habitat_vln

EPISODES="1"
MAX_STEPS=30
VLLM_PORT=8000
GPU_UTILIZATION=0.8
MODEL_PATH="/data/WZ/Model/Qwen/Qwen3.6-35B-A3B"
LOG_DIR="logs"
OUTPUT_DIR="results"
MM_PROCESSOR_KWARGS='{"min_pixels": 784, "max_pixels": 1003520}'
LIMIT_MM_PER_PROMPT='{"image": 2}'

log() { echo "[$(date '+%H:%M:%S')] $1"; }

check_gpu() {
    log "检查GPU状态..."
    nvidia-smi | grep -E "MiB /" | head -4
}

clear_gpu() {
    log "清理GPU进程..."
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
    if [ -n "$pids" ]; then
        for pid in $pids; do
            if [ "$pid" != "$$" ]; then
                log "终止进程 $pid"
                kill -9 $pid 2>/dev/null || true
            fi
        done
        sleep 5
    fi
    check_gpu
}

wait_for_vllm() {
    log "等待vLLM服务器就绪..."
    max_wait=180
    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s http://localhost:${VLLM_PORT}/v1/models 2>/dev/null | grep -q "data"; then
            log "vLLM服务器已就绪！"
            return 0
        fi
        if grep -q "ERROR" ${LOG_DIR}/vllm_server.log 2>/dev/null; then
            log "vLLM启动失败，查看日志："
            tail -20 ${LOG_DIR}/vllm_server.log | grep -E "ERROR|error|failed"
            return 1
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        log "已等待 ${elapsed}s / ${max_wait}s"
    done
    log "超时！"
    tail -50 ${LOG_DIR}/vllm_server.log
    return 1
}

start_vllm() {
    log "启动vLLM服务器..."
    mkdir -p ${LOG_DIR}
    nohup conda run -n vllm_env --no-capture-output \
        vllm serve ${MODEL_PATH} \
        --port ${VLLM_PORT} \
        --max_model_len 4096 \
        --mm_processor_kwargs "${MM_PROCESSOR_KWARGS}" \
        --limit_mm_per_prompt "${LIMIT_MM_PER_PROMPT}" \
        --trust_remote_code \
        --tensor_parallel_size 2 \
        --quantization fp8 \
        --enforce_eager \
        --served-model-name qwen-9b \
        > ${LOG_DIR}/vllm_server.log 2>&1 &
    log "vLLM服务器PID: $!"
    wait_for_vllm
}

run_vln_experiment() {
    log "运行VLN实验..."
    for ep in ${EPISODES}; do
        log "开始 Episode ${ep}..."
        python run_vln_experiment.py \
            --use-pipeline \
            --use-remote-llm \
            --llm-server http://localhost:${VLLM_PORT} \
            --episodes 1 \
            --start-episode ${ep} \
            --max-steps ${MAX_STEPS} \
            --use-sequence-mode \
            --output-dir "${OUTPUT_DIR}/episode-$(date '+%Y-%m%d-%H%M')"
        log "Episode ${ep} 完成"
    done
}

stop_vllm() {
    log "停止vLLM服务器..."
    pkill -f "vllm serve" || true
    sleep 3
    log "vLLM服务器已停止"
}

case "$1" in
    clear_gpu) clear_gpu ;;
    start_vllm) clear_gpu; start_vllm ;;
    stop_vllm) stop_vllm ;;
    run_exp) run_vln_experiment ;;
    all|"") clear_gpu; start_vllm; run_vln_experiment ;;
    *) echo "用法: $0 [clear_gpu|start_vllm|stop_vllm|run_exp|all]"; exit 1 ;;
esac


#   使用示例：

#   # 完整流程（推荐）
#   bash scripts/run_vln_experiment_with_vllm.sh all

#   # 分步执行
#   bash scripts/run_vln_experiment_with_vllm.sh clear_gpu  # 先清理GPU
#   bash scripts/run_vln_experiment_with_vllm.sh start_vllm # 启动服务器
#   bash scripts/run_vln_experiment_with_vllm.sh run_exp    # 运行实验
#   bash scripts/run_vln_experiment_with_vllm.sh stop_vllm  # 停止服务器

#   注意事项：
#   - 需要在 /home/WZ/MA_VLN/habitat_vln 目录下运行
#   - vLLM服务器启动需要约180秒等待时间
#   - 使用2卡tensor parallel（RTX 4090双卡）