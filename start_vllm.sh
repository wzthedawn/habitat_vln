#!/bin/bash
# 启动 vLLM OpenAI 兼容服务器
# 用于 Qwen3.5-4B VLM 多模态推理

# 模型路径
MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "=========================================="
echo "启动 vLLM OpenAI 兼容服务器"
echo "模型: $MODEL_PATH"
echo "端口: 8000"
echo "=========================================="

# 启动 vLLM 服务器
# 关键参数:
#   --mm-encoder-tp-mode data: 数据并行视觉编码 (VLM 必需)
#   --max-model-len 4096: 上下文长度
#   --gpu-memory-utilization 0.45: 4B 模型显存控制
vllm serve "$MODEL_PATH" \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096 \
    --mm-encoder-tp-mode data \
    --trust-remote-code \
    --dtype float16