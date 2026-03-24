# VLN 环境配置指南

## 环境架构

```
┌─────────────────────────────────────────────────────────┐
│                    VLN 系统架构                          │
├─────────────────────────────────────────────────────────┤
│  Habitat (Python 3.9)                                   │
│  - VLN 主进程、habitat-sim、导航环境                      │
│  - YOLO 检测、CLIP 特征提取                              │
│  - 通过 HTTP 调用 LLM 服务                               │
├─────────────────────────────────────────────────────────┤
│  vllm_env (Python 3.10)                                 │
│  - vLLM 0.18.0 推理引擎                                  │
│  - Qwen3.5-4B/2B VLM 模型                                │
│  - 端口: 8000                                            │
└─────────────────────────────────────────────────────────┘
```

## 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| 内存 | 32GB | 64GB |
| 存储 | 100GB | 150GB+ |

## 新服务器环境搭建

### Step 1: 创建 Habitat 环境

```bash
# 方法1: 从配置文件创建 (推荐)
conda env create -f habitat_env.yml

# 方法2: 手动创建
conda create -n Habitat python=3.9 -y
conda activate Habitat

# 安装 habitat-sim (需要 aihabitat 频道)
conda install habitat-sim -c aihabitat -c conda-forge

# 安装 habitat-lab (需要从源码安装)
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # 可编辑安装

# 安装其他依赖
pip install -r habitat_requirements.txt
```

**注意**: `habitat-lab` 不在 PyPI 上，必须从源码安装：
```bash
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
```

### Step 2: 创建 vllm_env 环境

```bash
# 方法1: 从配置文件创建 (推荐)
conda env create -f vllm_env.yml

# 方法2: 手动创建
conda create -n vllm_env python=3.10 -y
conda activate vllm_env

# 安装 vLLM (会自动安装 PyTorch)
pip install vllm

# 安装其他依赖
pip install -r vllm_requirements.txt
```

### Step 3: 验证安装

```bash
# 验证 Habitat 环境
conda activate Habitat
python -c "import habitat; print('Habitat:', habitat.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

# 验证 vllm_env 环境
conda activate vllm_env
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"
```

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `habitat_env.yml` | Habitat 环境 conda 配置 |
| `habitat_requirements.txt` | Habitat 环境 pip 依赖 |
| `vllm_env.yml` | vllm_env 环境 conda 配置 |
| `vllm_requirements.txt` | vllm_env 环境 pip 依赖 |

## 启动服务

### 启动 vLLM 服务器

```bash
conda activate vllm_env
python vllm_server.py --port 8000
```

### 运行 VLN 实验

```bash
conda activate Habitat
xvfb-run -a python run_vln_experiment.py \
    --use-remote-llm \
    --episodes 3 \
    --max-steps 50 \
    --output results.json
```

## 模型路径配置

模型默认路径:
- Qwen3.5-4B: `/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B`
- Qwen3.5-2B: `/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B`

修改 `vllm_server.py` 中的 `get_model_configs()` 函数以更新路径。

## 常见问题

### 1. habitat-sim 安装失败
```bash
# 尝试使用特定版本
conda install habitat-sim=0.3.0 -c conda-forge
```

### 2. vLLM GPU 内存不足
```bash
# 降低 gpu-memory-utilization 参数
python vllm_server.py --gpu-memory 0.3
```

### 3. CUDA 版本不兼容
```bash
# 检查 CUDA 版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```