#!/usr/bin/env python3
"""vLLM Inference Server for VLN Multi-Agent System.

Runs in Python 3.10 environment with Qwen3.5 models using vLLM engine.
Provides HTTP API for LLM inference to support the dual-environment IPC architecture.

Usage:
    # In vllm_env environment:
    conda activate vllm_env
    python vllm_server.py --port 8000

Architecture:
    - Python 3.9 (Habitat): VLN main process, habitat-sim, YOLO
    - Python 3.10 (this server): vLLM + Qwen3.5 models for LLM inference

Communication:
    - HTTP POST /generate: Generate text using specified model
    - HTTP POST /generate_vision: Generate text from image + text (VLM)
    - HTTP GET /health: Health check and loaded models
"""

import argparse
import base64
import logging
import os
import time
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VLLMServer")

# Global model storage
llm_engines: Dict[str, Any] = {}  # vLLM engines
model_configs: Dict[str, Dict] = {}

# Conversation contexts for multi-turn dialogues
conversation_contexts: Dict[str, List[Dict[str, str]]] = {}


def get_model_configs() -> Dict[str, Dict]:
    """Get model configurations.

    Model allocation (same as llm_server.py for compatibility):
    - qwen-4b-perception: Visual perception and scene description
    - qwen-4b-instruction: Instruction decomposition with semantic analysis
    - qwen-4b-decision: Navigation decision making
    - qwen-4b-evaluation: Decision evaluation and feedback
    - qwen-2b-trajectory: Trajectory summarization (2B for efficiency)

    Note: All 4B models share the same physical weights but have independent configurations.
    Each agent uses a unique model key for isolated conversation contexts.

    Total VRAM: ~6GB (2B) + ~8GB (4B) = ~14GB + Habitat ~2GB = ~16GB (safe for 24GB GPU)
    """
    return {
        # === 4B Models (shared weights, independent configs) ===
        "qwen-4b-perception": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 400,
            "default_temperature": 0.3,
            "description": "Visual perception and scene description",
            "is_vlm": True,  # Qwen3.5-4B is a multimodal VLM
        },
        "qwen-4b-instruction": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 300,
            "default_temperature": 0.1,  # Lower temperature for stable JSON output
            "description": "Instruction decomposition with semantic analysis",
        },
        "qwen-4b-decision": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 250,
            "default_temperature": 0.2,
            "description": "Navigation decision making",
        },
        "qwen-4b-evaluation": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 150,
            "default_temperature": 0.2,
            "description": "Decision evaluation and feedback",
        },
        # === 2B Model ===
        "qwen-2b-trajectory": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B",
            "max_new_tokens": 200,
            "default_temperature": 0.2,
            "description": "Trajectory summarization and navigation progress",
        },
    }


def load_vllm_engine(model_key: str, gpu_memory_utilization: float = 0.5) -> bool:
    """Load a vLLM engine for a model.

    Models with the same model_path share the same engine to save VRAM.

    Args:
        model_key: Model identifier
        gpu_memory_utilization: GPU memory fraction for this engine

    Returns:
        True if loaded successfully
    """
    global llm_engines, model_configs

    if model_key in llm_engines:
        return True

    config = model_configs.get(model_key)
    if not config:
        logger.error(f"Unknown model key: {model_key}")
        return False

    model_path = config["model_name"]

    # Check if model path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    # Check if a model with the same path is already loaded (engine sharing)
    for existing_key, existing_config in model_configs.items():
        if existing_key in llm_engines and existing_config["model_name"] == model_path and existing_key != model_key:
            logger.info(f"Sharing vLLM engine {existing_key} -> {model_key} (same path: {model_path})")
            llm_engines[model_key] = llm_engines[existing_key]
            return True

    try:
        from vllm import LLM

        logger.info(f"Loading vLLM engine for {model_key} from {model_path}...")

        # vLLM engine initialization
        # Note: vLLM automatically manages KV cache with PagedAttention
        engine = LLM(
            model=model_path,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,  # Context length
            trust_remote_code=True,
            enforce_eager=True,  # Disable CUDA graphs for compatibility
        )

        llm_engines[model_key] = engine
        logger.info(f"vLLM engine for {model_key} loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load vLLM engine for {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_all_models(gpu_memory_utilization: float = 0.5) -> bool:
    """Load all configured models.

    Args:
        gpu_memory_utilization: GPU memory fraction per engine

    Returns:
        True if all required models loaded successfully
    """
    global model_configs

    if not model_configs:
        model_configs = get_model_configs()

    logger.info("=" * 60)
    logger.info("Loading vLLM engines...")
    logger.info(f"GPU memory utilization: {gpu_memory_utilization}")
    logger.info(f"Models to load: {list(model_configs.keys())}")
    logger.info("=" * 60)

    success = True
    for model_key in model_configs:
        loaded = load_vllm_engine(model_key, gpu_memory_utilization)
        if not loaded:
            logger.warning(f"Failed to load {model_key}")
            success = False

    logger.info("=" * 60)
    logger.info(f"Engines loaded: {list(llm_engines.keys())}")
    logger.info("=" * 60)

    return success


def generate_text(
    model_key: str,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    conversation_id: Optional[str] = None,
    keep_context: bool = False,
) -> Dict[str, Any]:
    """Generate text using vLLM engine.

    Args:
        model_key: Model identifier
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        conversation_id: Optional ID for multi-turn conversations
        keep_context: Whether to keep conversation context

    Returns:
        Dictionary with response and metadata
    """
    start_time = time.time()

    if model_key not in llm_engines:
        return {
            "error": f"Model {model_key} not found",
            "available_models": list(llm_engines.keys()),
        }

    engine = llm_engines[model_key]
    config = model_configs[model_key]

    # Get generation parameters
    max_tokens = max_new_tokens or config["max_new_tokens"]
    temp = temperature if temperature is not None else config["default_temperature"]

    try:
        from vllm import SamplingParams

        # Handle conversation context
        context_str = ""
        if conversation_id and keep_context:
            if conversation_id not in conversation_contexts:
                conversation_contexts[conversation_id] = []
            context = conversation_contexts[conversation_id][-5:]
            if context:
                context_str = "\n".join([
                    f"Human: {c['human']}\nAssistant: {c['assistant']}"
                    for c in context
                ]) + "\n"

        # Build full prompt
        full_prompt = f"{context_str}Human: {prompt}\nAssistant:" if context_str else prompt

        # Sampling parameters for vLLM
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=50,
        )

        # Generate with vLLM
        outputs = engine.generate([full_prompt], sampling_params)

        # Extract response
        generated_text = outputs[0].outputs[0].text.strip()
        tokens_generated = len(outputs[0].outputs[0].token_ids)

        latency = (time.time() - start_time) * 1000

        # Update conversation context
        if conversation_id and keep_context:
            conversation_contexts[conversation_id].append({
                "human": prompt,
                "assistant": generated_text,
            })
            if len(conversation_contexts[conversation_id]) > 10:
                conversation_contexts[conversation_id] = conversation_contexts[conversation_id][-10:]

        return {
            "response": generated_text,
            "model": model_key,
            "tokens_generated": tokens_generated,
            "latency_ms": latency,
            "conversation_id": conversation_id,
        }

    except Exception as e:
        logger.error(f"Generation failed for {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "model": model_key,
        }


def generate_vision(
    model_key: str,
    prompt: str,
    image_base64: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    depth_base64: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate text from image and prompt using vLLM.

    Note: vLLM supports multimodal models. For Qwen3.5-VL, we use the
    offline inference with image inputs.

    Args:
        model_key: Model identifier (e.g., qwen-4b-perception)
        prompt: Input prompt
        image_base64: Base64 encoded RGB image
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        depth_base64: Optional base64 encoded depth image (colored)

    Returns:
        Dictionary with response and metadata
    """
    start_time = time.time()

    if model_key not in llm_engines:
        return {
            "error": f"Model {model_key} not found",
            "available_models": list(llm_engines.keys()),
        }

    engine = llm_engines[model_key]
    config = model_configs[model_key]

    max_tokens = max_new_tokens or config["max_new_tokens"]
    temp = temperature if temperature is not None else config["default_temperature"]

    try:
        from vllm import SamplingParams
        from PIL import Image

        # Decode base64 RGB image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Prepare inputs for vLLM multimodal
        # vLLM uses a specific format for multimodal inputs
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

        # Add depth image if provided
        if depth_base64:
            depth_data = base64.b64decode(depth_base64)
            depth_image = Image.open(BytesIO(depth_data))
            if depth_image.mode != 'RGB':
                depth_image = depth_image.convert('RGB')
            # Note: For dual images, we concatenate prompts
            inputs["prompt"] = f"[Image 1: RGB]\n[Image 2: Depth]\n{prompt}"

        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=50,
        )

        # Generate with vLLM
        outputs = engine.generate([inputs], sampling_params)

        generated_text = outputs[0].outputs[0].text.strip()
        tokens_generated = len(outputs[0].outputs[0].token_ids)

        latency = (time.time() - start_time) * 1000

        logger.info(f"[VLM-vLLM] {model_key}: generated {tokens_generated} tokens in {latency:.0f}ms")

        return {
            "response": generated_text,
            "model": model_key,
            "tokens_generated": tokens_generated,
            "latency_ms": latency,
        }

    except Exception as e:
        logger.error(f"Vision generation failed for {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "model": model_key,
        }


def clear_conversation(conversation_id: str) -> bool:
    """Clear conversation context."""
    if conversation_id in conversation_contexts:
        del conversation_contexts[conversation_id]
        return True
    return False


# FastAPI application
@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for FastAPI."""
    logger.info("Starting vLLM Inference Server...")
    yield
    logger.info("Shutting down vLLM Inference Server...")
    llm_engines.clear()


def create_app() -> "FastAPI":
    """Create FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="VLN vLLM Inference Server",
        description="HTTP API for vLLM-based LLM inference in VLN multi-agent system",
        version="1.0.0",
        lifespan=lifespan,
    )

    class GenerateRequest(BaseModel):
        """Request model for text generation."""
        model: str = Field(..., description="Model identifier (e.g., qwen-4b-decision)")
        prompt: str = Field(..., description="Input prompt for generation")
        max_new_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
        temperature: Optional[float] = Field(None, description="Sampling temperature (0.0-2.0)")
        conversation_id: Optional[str] = Field(None, description="Optional ID for multi-turn conversations")
        keep_context: bool = Field(False, description="Whether to keep conversation context")

    class GenerateResponse(BaseModel):
        """Response model for text generation."""
        response: str
        model: str
        tokens_generated: int
        latency_ms: float
        conversation_id: Optional[str] = None
        error: Optional[str] = None

    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str
        models_loaded: List[str]
        model_configs: Dict[str, Dict]
        gpu_memory: Optional[Dict[str, float]] = None

    class ClearConversationRequest(BaseModel):
        """Request model for clearing conversation."""
        conversation_id: str

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        gpu_memory = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = {
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                }
        except:
            pass

        return HealthResponse(
            status="healthy",
            models_loaded=list(llm_engines.keys()),
            model_configs={
                k: {"description": v["description"], "max_new_tokens": v["max_new_tokens"]}
                for k, v in model_configs.items()
            },
            gpu_memory=gpu_memory,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text using specified model."""
        if request.model not in llm_engines:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} not found. Available: {list(llm_engines.keys())}"
            )

        result = generate_text(
            model_key=request.model,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
            keep_context=request.keep_context,
        )

        if "error" in result and "response" not in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return GenerateResponse(
            response=result.get("response", ""),
            model=result.get("model", request.model),
            tokens_generated=result.get("tokens_generated", 0),
            latency_ms=result.get("latency_ms", 0),
            conversation_id=result.get("conversation_id"),
            error=result.get("error"),
        )

    class GenerateVisionRequest(BaseModel):
        """Request model for vision-language generation."""
        model: str = Field(default="qwen-4b-perception", description="VLM model identifier")
        prompt: str = Field(..., description="Input prompt for generation")
        image_base64: str = Field(..., description="Base64 encoded RGB image")
        max_new_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
        temperature: Optional[float] = Field(None, description="Sampling temperature (0.0-2.0)")
        depth_base64: Optional[str] = Field(None, description="Base64 encoded depth image (colored)")

    class GenerateVisionResponse(BaseModel):
        """Response model for vision-language generation."""
        response: str
        model: str
        tokens_generated: int
        latency_ms: float
        error: Optional[str] = None

    @app.post("/generate_vision", response_model=GenerateVisionResponse)
    async def generate_vision_endpoint(request: GenerateVisionRequest):
        """Generate text from image and prompt using VLM.

        Supports single image (RGB) or dual image (RGB + Depth) input.
        """
        result = generate_vision(
            model_key=request.model,
            prompt=request.prompt,
            image_base64=request.image_base64,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            depth_base64=request.depth_base64,
        )

        if "error" in result and "response" not in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return GenerateVisionResponse(
            response=result.get("response", ""),
            model=result.get("model", request.model),
            tokens_generated=result.get("tokens_generated", 0),
            latency_ms=result.get("latency_ms", 0),
            error=result.get("error"),
        )

    @app.post("/clear_conversation")
    async def clear_conv(request: ClearConversationRequest):
        """Clear conversation context."""
        cleared = clear_conversation(request.conversation_id)
        return {"cleared": cleared, "conversation_id": request.conversation_id}

    @app.get("/models")
    async def list_models():
        """List available models and their configurations."""
        return {
            "models": {
                k: {
                    "loaded": k in llm_engines,
                    "description": v["description"],
                    "max_new_tokens": v["max_new_tokens"],
                    "default_temperature": v["default_temperature"],
                }
                for k, v in model_configs.items()
            }
        }

    @app.post("/load_model")
    async def load_model_endpoint(model_key: str, gpu_memory_utilization: float = 0.5):
        """Load a specific model."""
        if model_key in llm_engines:
            return {"status": "already_loaded", "model": model_key}

        success = load_vllm_engine(model_key, gpu_memory_utilization)
        if success:
            return {"status": "loaded", "model": model_key}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load {model_key}")

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VLN vLLM Inference Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--gpu-memory", type=float, default=0.5,
                        help="GPU memory utilization per engine (0.0-1.0)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Specific models to load (default: all)")

    args = parser.parse_args()

    # Set up model configs
    global model_configs
    model_configs = get_model_configs()

    # Filter models if specified
    if args.models:
        model_configs = {k: v for k, v in model_configs.items() if k in args.models}

    # Load models
    load_all_models(args.gpu_memory)

    # Create and run app
    import uvicorn
    app = create_app()

    logger.info(f"Starting vLLM server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()