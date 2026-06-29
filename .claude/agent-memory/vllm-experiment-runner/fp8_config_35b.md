---
name: FP8 Quantization Config for 35B Model
description: FP8量化配置用于在双卡RTX 4090上加载70GB的Qwen3.6-35B-A3B MoE模型
type: reference
---

## FP8量化启动命令

```bash
conda run --no-capture-output -n vllm_env --live-stream \
    vllm serve "/data/WZ/Model/Qwen/Qwen3.6-35B-A3B" \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --quantization fp8 \
    --dtype auto \
    --enforce-eager \
    --max-num-seqs 16 \
    > /tmp/vllm_server.log 2>&1 &
```

## 关键参数说明
- `--quantization fp8`: 使用FP8量化，将模型权重压缩到8位浮点格式
- `--gpu-memory-utilization 0.85`: 显存利用率85%（经验值，避免OOM）
- `--tensor-parallel-size 2`: 双卡并行
- `--enforce-eager`: 禁用CUDAGraph优化（避免某些兼容性问题）

## 加载结果
- 模型加载时间: 约14.57秒
- 显存占用: 约16.99 GiB（FP8量化后）
- 原始模型大小: 约70GB（26个safetensors文件）
- KV cache可用: 约0.63 GiB
- 最大并发度: 8.71x (对于4096 tokens请求)

## 注意事项
1. 必须先清理GPU显存（kill残留vllm进程）
2. FP8量化在RTX 4090上有效（compute capability 8.9支持FP8）
3. 不使用FP8时，双卡48GB显存不足以加载70GB模型