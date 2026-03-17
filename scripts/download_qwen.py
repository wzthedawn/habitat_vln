#!/usr/bin/env python3
"""
从 ModelScope 下载 Qwen3.5 模型

使用方法:
    python download_qwen.py --model qwen3.5-plus --output ./models/qwen3.5-plus
"""

import argparse
import os
import sys

def download_from_modelscope(model_id, output_dir):
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download

        print(f"正在从 ModelScope 下载模型: {model_id}")
        print(f"保存目录: {output_dir}")

        model_path = snapshot_download(
            model_id,
            cache_dir=output_dir,
            revision='master'
        )

        print(f"\n✅ 模型下载完成!")
        print(f"模型路径: {model_path}")
        return model_path

    except ImportError:
        print("❌ 请先安装 modelscope:")
        print("   pip install modelscope")
        return None
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def download_with_huggingface(model_id, output_dir):
    """使用 Hugging Face 镜像下载"""
    try:
        from huggingface_hub import snapshot_download

        # 使用 HF 镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        print(f"正在从 Hugging Face 镜像下载模型: {model_id}")
        print(f"保存目录: {output_dir}")

        model_path = snapshot_download(
            model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )

        print(f"\n✅ 模型下载完成!")
        print(f"模型路径: {model_path}")
        return model_path

    except ImportError:
        print("❌ 请先安装 huggingface_hub:")
        print("   pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


# ModelScope 上的 Qwen3.5 模型列表
QWEN_MODELS = {
    # Base models
    "qwen3.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen3.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen3.5-3b": "Qwen/Qwen2.5-3B",
    "qwen3.5-7b": "Qwen/Qwen2.5-7B",
    "qwen3.5-14b": "Qwen/Qwen2.5-14B",
    "qwen3.5-32b": "Qwen/Qwen2.5-32B",
    "qwen3.5-72b": "Qwen/Qwen2.5-72B",
    # Instruct versions (推荐用于对话和推理)
    "qwen3.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen3.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "qwen3.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    # 别名 (用户友好名称)
    "local_small": "Qwen/Qwen2.5-3B-Instruct",      # 用于 Type-0/1
    "local_medium": "Qwen/Qwen2.5-7B-Instruct",    # 用于 Type-1/2
}


def main():
    parser = argparse.ArgumentParser(description="下载 Qwen3.5 模型")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen3.5-7b-instruct",
        choices=list(QWEN_MODELS.keys()),
        help="要下载的模型名称"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/",
        help="模型保存目录"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="下载源"
    )

    args = parser.parse_args()

    # 获取模型ID
    model_id = QWEN_MODELS.get(args.model)
    if not model_id:
        print(f"❌ 未知的模型: {args.model}")
        print(f"可用模型: {list(QWEN_MODELS.keys())}")
        return 1

    # 创建输出目录
    output_dir = os.path.join(args.output, args.model)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Qwen3.5 模型下载工具")
    print("=" * 60)
    print(f"模型: {args.model} ({model_id})")
    print(f"保存目录: {output_dir}")
    print(f"下载源: {args.source}")
    print("=" * 60)

    # 下载模型
    if args.source == "modelscope":
        success = download_from_modelscope(model_id, output_dir)
    else:
        success = download_with_huggingface(model_id, output_dir)

    if success:
        print("\n使用方法:")
        print(f"  模型路径: {success}")
        print("\n在代码中加载:")
        print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
        print(f'  model = AutoModelForCausalLM.from_pretrained("{success}")')
        print(f'  tokenizer = AutoTokenizer.from_pretrained("{success}")')
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())