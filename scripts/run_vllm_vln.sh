#!/bin/bash
# VLN 实验启动脚本
# 启动 vLLM 服务器并运行 VLN 评估实验

cd /home/WZ/MA_VLN/habitat_vln

echo "=========================================="
echo "VLN 评估实验 - vLLM 服务器启动"
echo "=========================================="

# 默认参数
EPISODES=${EPISODES:-1}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-5}
USE_STRATEGY=${USE_STRATEGY:-true}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --sequence-length)
            SEQUENCE_LENGTH="$2"
            shift 2
            ;;
        --no-strategy)
            USE_STRATEGY=false
            shift
            ;;
        --help)
            echo "用法：/run-vllm-vln [选项]"
            echo ""
            echo "选项:"
            echo "  --episodes N        运行 episode 数量 (默认：1)"
            echo "  --sequence-length N 动作序列长度 (默认：5)"
            echo "  --no-strategy       禁用策略模式"
            echo "  --help              显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项：$1"
            exit 1
            ;;
    esac
done

echo "配置:"
echo "  Episodes: $EPISODES"
echo "  Sequence Length: $SEQUENCE_LENGTH"
echo "  Strategy Mode: $USE_STRATEGY"
echo ""

# 检查 vLLM 是否已在运行
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM 服务器已在运行"
else
    echo "启动 vLLM 服务器..."
    /home/WZ/.conda/envs/vllm_env/bin/vllm serve \
        /data/WZ/Model/Qwen/Qwen3___5-9b_AWQ \
        --port 8000 \
        --host 0.0.0.0 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 4096 \
        --trust-remote-code \
        --dtype auto \
        --quantization awq \
        --enforce-eager &

    # 等待 vLLM 启动
    echo "等待 vLLM 服务器启动..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "vLLM 服务器已启动!"
            break
        fi
        sleep 1
    done

    # 检查是否启动成功
    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "错误：vLLM 服务器启动失败"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "运行 VLN 实验"
echo "=========================================="

# 构建命令
CMD="python run_vln_experiment.py"
CMD="$CMD --use-remote-llm"
CMD="$CMD --llm-server http://localhost:8000"
CMD="$CMD --episodes $EPISODES"
CMD="$CMD --sequence-length $SEQUENCE_LENGTH"
CMD="$CMD --use-sequence-mode"

if [ "$USE_STRATEGY" = true ]; then
    CMD="$CMD --use-strategy-mode"
fi

# 运行实验
eval $CMD
