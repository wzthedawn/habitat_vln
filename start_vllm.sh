#!/bin/bash
# 启动 vLLM OpenAI 兼容服务器
# 支持模型切换: --model 9b 或 --model 35b

# 默认模型
MODEL_TYPE="9b"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --model [9b|35b] --port [端口号]"
            exit 1
            ;;
    esac
done

# 模型配置
case $MODEL_TYPE in
    9b)
        MODEL_PATH="/data/WZ/Model/Qwen/Qwen3___5-9B"
        GPU_IDS="0"
        TENSOR_PARALLEL=1
        GPU_MEMORY=0.85
        MAX_LEN=4096
        QUANTIZATION=""
        EXTRA_ARGS="--mm-encoder-tp-mode data --enable-auto-tool-choice --tool-call-parser qwen3"
        ;;
    35b)
        MODEL_PATH="/data/WZ/Model/Qwen/Qwen3.6-35B-A3B"
        GPU_IDS="0,1"
        TENSOR_PARALLEL=2
        GPU_MEMORY=0.85
        MAX_LEN=4096
        QUANTIZATION="--quantization fp8"
        EXTRA_ARGS="--max-num-seqs 8"
        ;;
    *)
        echo "错误: 未知模型类型 '$MODEL_TYPE'"
        echo "可选: 9b, 35b"
        exit 1
        ;;
esac

# 默认端口
PORT=${PORT:-8000}

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "启动 vLLM OpenAI 兼容服务器"
echo "模型类型: $MODEL_TYPE"
echo "模型路径: $MODEL_PATH"
echo "GPU: $GPU_IDS"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "端口: $PORT"
echo "=========================================="

# 检查GPU状态
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv

# 启动 vLLM 服务器
CUDA_VISIBLE_DEVICES=$GPU_IDS vllm serve "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --gpu-memory-utilization $GPU_MEMORY \
    --max-model-len $MAX_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL \
    --trust-remote-code \
    --dtype auto \
    --enforce-eager \
    $QUANTIZATION \
    $EXTRA_ARGS