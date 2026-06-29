#!/usr/bin/env python3
"""Start vLLM OpenAI-compatible server for Qwen3.5-4B VLM.

This script starts vLLM with OpenAI-compatible API using the LLM class
internally, which is more stable than `vllm serve` command.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--gpu-memory", type=float, default=0.85, help="GPU memory utilization")
    args = parser.parse_args()

    model_path = "/data/WZ/Model/Qwen/Qwen3___5-9b_AWQ"

    # Check model exists
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    print(f"Starting vLLM OpenAI-compatible server...")
    print(f"Model: {model_path}")
    print(f"Port: {args.port}")
    print(f"GPU Memory: {args.gpu_memory}")

    # Use uvicorn to run the OpenAI-compatible server
    import uvicorn
    from vllm.entrypoints.openai.api_server import build_app, create_server_config

    # Create engine config
    from vllm.config import EngineConfig, ModelConfig, SchedulerConfig, DeviceConfig
    from vllm import EngineArgs

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        quantization="awq",
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=4096,
        enforce_eager=True,
    )

    # Build and run the app
    import asyncio
    from vllm.entrypoints.openai.api_server import init_app

    # Use vLLM's built-in server setup
    app = init_app(engine_args, response_role="assistant")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()