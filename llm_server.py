#!/usr/bin/env python3
"""LLM Inference Server for VLN Multi-Agent System.

Runs in Python 3.10 environment with Qwen3.5 models.
Provides HTTP API for LLM inference to support the dual-environment IPC architecture.

Usage:
    # In Python 3.10 environment:
    conda activate habitat_py310
    python llm_server.py --port 8000

Architecture:
    - Python 3.9 (Habitat): VLN main process, habitat-sim, YOLO
    - Python 3.10 (this server): Qwen3.5 models for LLM inference

Communication:
    - HTTP POST /generate: Generate text using specified model
    - HTTP POST /generate_vision: Generate text from image + text (VLM)
    - HTTP GET /health: Health check and loaded models
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMServer")

# Global model storage
models: Dict[str, Any] = {}
tokenizers: Dict[str, Any] = {}
model_configs: Dict[str, Dict] = {}

# VLM-specific storage
vlm_processors: Dict[str, Any] = {}  # VLM image processors

# Conversation contexts for multi-turn dialogues
conversation_contexts: Dict[str, List[Dict[str, str]]] = {}


def get_model_configs() -> Dict[str, Dict]:
    """Get model configurations.

    Model allocation (方案二):
    - qwen-4b-perception: Visual perception and scene description (4B for better visual understanding)
    - qwen-2b-trajectory: Trajectory summarization (2B sufficient for simple text tasks)
    - qwen-4b: Navigation decision making (4B for reasoning)
    - qwen-4b-evaluation: Decision evaluation (4B for assessment)
    - qwen2-vl-2b: Vision-Language Model for object detection and scene analysis

    Total VRAM: ~16GB + Habitat ~2GB = ~18GB (safe for 24GB GPU)
    """
    return {
        "qwen-4b-perception": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 256,
            "default_temperature": 0.3,
            "description": "Visual perception and scene description (VLM)",
            "is_vlm": True,  # Qwen3.5-4B is a multimodal VLM
        },
        "qwen-2b-trajectory": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-2B",
            "max_new_tokens": 200,
            "default_temperature": 0.2,
            "description": "Trajectory summarization and navigation progress",
        },
        "qwen-4b": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 150,
            "default_temperature": 0.1,
            "description": "Navigation decision making",
        },
        "qwen-4b-evaluation": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen3___5-4B",
            "max_new_tokens": 150,
            "default_temperature": 0.2,
            "description": "Decision evaluation and feedback",
        },
        "qwen2-vl-2b": {
            "model_name": "/root/.cache/modelscope/hub/models/Qwen/Qwen2-VL-2B-Instruct",
            "max_new_tokens": 256,
            "default_temperature": 0.3,
            "description": "Vision-Language Model for object detection and scene analysis (optional)",
            "is_vlm": True,
            "optional": True,  # VLM is optional, won't fail if not available
        },
    }


def load_model(model_key: str, use_int8: bool = True) -> bool:
    """Load a single model.

    Args:
        model_key: Model identifier
        use_int8: Whether to use INT8 quantization

    Returns:
        True if loaded successfully
    """
    global models, tokenizers, model_configs

    if model_key in models:
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

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"Loading {model_key} from {model_path}...")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Configure quantization
        if use_int8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

        model.eval()

        models[model_key] = model
        tokenizers[model_key] = tokenizer

        logger.info(f"{model_key} loaded successfully")
        return True

    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install transformers bitsandbytes accelerate")
        return False
    except Exception as e:
        logger.error(f"Failed to load {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_vlm_model(model_key: str, use_int8: bool = True) -> bool:
    """Load a Vision-Language Model.

    Supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3.5-VL models.

    Args:
        model_key: Model identifier (e.g., qwen2-vl-2b, qwen-4b-perception)
        use_int8: Whether to use INT8 quantization

    Returns:
        True if loaded successfully
    """
    global models, tokenizers, vlm_processors, model_configs

    if model_key in models:
        return True

    config = model_configs.get(model_key)
    if not config:
        logger.error(f"Unknown model key: {model_key}")
        return False

    if not config.get("is_vlm", False):
        logger.error(f"Model {model_key} is not configured as a VLM")
        return False

    model_path = config["model_name"]

    # Check if model path exists
    if not os.path.exists(model_path):
        logger.warning(f"VLM model path does not exist: {model_path}")
        logger.warning("VLM model not available, perception will use text-only fallback")
        return False

    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import Qwen2VLForConditionalGeneration, Qwen3_5ForConditionalGeneration

        logger.info(f"Loading VLM {model_key} from {model_path}...")

        # Load processor (includes tokenizer and image processor)
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Configure quantization
        if use_int8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            quantization_config = None

        # Select model class based on model key
        model = None
        try:
            if "qwen2-vl" in model_key.lower():
                # Qwen2-VL-2B uses dedicated class
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            else:
                # Qwen3.5-4B uses Qwen3_5ForConditionalGeneration
                # This properly handles both text and image inputs
                model = Qwen3_5ForConditionalGeneration.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
        except Exception as e:
            logger.warning(f"Failed to load VLM model: {e}")
            # Fallback: try without quantization
            try:
                if "qwen2-vl" in model_key.lower():
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
                else:
                    model = Qwen3_5ForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
            except Exception as e2:
                logger.error(f"Failed to load VLM model: {e2}")
                return False

        model.eval()

        models[model_key] = model
        tokenizers[model_key] = processor.tokenizer
        vlm_processors[model_key] = processor

        logger.info(f"VLM {model_key} loaded successfully")
        return True

    except ImportError as e:
        logger.error(f"Missing VLM dependencies: {e}")
        logger.error("Install with: pip install transformers>=4.45.0 qwen-vl-utils")
        return False
    except Exception as e:
        logger.error(f"Failed to load VLM {model_key}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_all_models(use_int8: bool = True) -> bool:
    """Load all configured models.

    Args:
        use_int8: Whether to use INT8 quantization

    Returns:
        True if all required models loaded successfully
    """
    global model_configs

    # Don't overwrite if model_configs already set (e.g., filtered by --models)
    if not model_configs:
        model_configs = get_model_configs()

    logger.info("=" * 60)
    logger.info("Loading LLM models...")
    logger.info(f"INT8 quantization: {use_int8}")
    logger.info(f"Models to load: {list(model_configs.keys())}")
    logger.info("=" * 60)

    success = True
    for model_key in model_configs:
        config = model_configs[model_key]
        is_optional = config.get("optional", False)
        is_vlm = config.get("is_vlm", False)

        # Try to load the model
        if is_vlm:
            loaded = load_vlm_model(model_key, use_int8)
        else:
            loaded = load_model(model_key, use_int8)

        if not loaded:
            if is_optional:
                logger.warning(f"Optional model {model_key} not available - continuing without it")
            else:
                logger.warning(f"Failed to load required model {model_key}")
                success = False

    logger.info("=" * 60)
    logger.info(f"Models loaded: {list(models.keys())}")
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
    """Generate text using specified model.

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

    if model_key not in models:
        return {
            "error": f"Model {model_key} not found",
            "available_models": list(models.keys()),
        }

    model = models[model_key]
    tokenizer = tokenizers[model_key]
    config = model_configs[model_key]

    # Get generation parameters
    max_tokens = max_new_tokens or config["max_new_tokens"]
    temp = temperature if temperature is not None else config["default_temperature"]

    try:
        import torch

        # Handle conversation context
        full_prompt = prompt
        if conversation_id and keep_context:
            if conversation_id not in conversation_contexts:
                conversation_contexts[conversation_id] = []

            # Build context from history (keep last 5 exchanges)
            context = conversation_contexts[conversation_id][-5:]
            if context:
                context_str = "\n".join([
                    f"Human: {c['human']}\nAssistant: {c['assistant']}"
                    for c in context
                ])
                full_prompt = f"{context_str}\nHuman: {prompt}\nAssistant:"
            else:
                full_prompt = prompt

        # Tokenize
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                top_k=50,
            )

        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        latency = (time.time() - start_time) * 1000
        tokens_generated = len(outputs[0]) - inputs["input_ids"].shape[1]

        # Update conversation context
        if conversation_id and keep_context:
            conversation_contexts[conversation_id].append({
                "human": prompt,
                "assistant": generated_text.strip(),
            })
            # Keep only last 10 exchanges
            if len(conversation_contexts[conversation_id]) > 10:
                conversation_contexts[conversation_id] = conversation_contexts[conversation_id][-10:]

        return {
            "response": generated_text.strip(),
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
) -> Dict[str, Any]:
    """Generate text from image and prompt using VLM.

    Supports both Qwen2-VL and Qwen3.5-VL models.

    Args:
        model_key: Model identifier (e.g., qwen2-vl-2b, qwen-4b-perception)
        prompt: Input prompt
        image_base64: Base64 encoded image
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with response and metadata
    """
    start_time = time.time()

    # Check if VLM model is available
    if model_key not in models:
        # Try to load VLM model
        if not load_vlm_model(model_key):
            return {
                "error": f"VLM model {model_key} not available",
                "available_models": list(models.keys()),
            }

    model = models[model_key]
    processor = vlm_processors.get(model_key)
    config = model_configs[model_key]

    if processor is None:
        return {
            "error": f"VLM processor not found for {model_key}",
            "model": model_key,
        }

    # Get generation parameters
    max_tokens = max_new_tokens or config["max_new_tokens"]
    temp = temperature if temperature is not None else config["default_temperature"]

    try:
        import torch
        from PIL import Image

        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Prepare messages for VLM
        # Format compatible with both Qwen2-VL and Qwen3.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs - use processor's __call__ method
        # This handles both image and text processing
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        # Move all inputs to device (including VLM-specific keys like pixel_values)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=temp > 0,
                top_p=0.9,
                top_k=50,
            )

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        latency = (time.time() - start_time) * 1000
        tokens_generated = len(generated_ids)

        logger.info(f"[VLM] {model_key}: generated {tokens_generated} tokens in {latency:.0f}ms")
        logger.info(f"[VLM] Response: {generated_text.strip()[:500]}")

        return {
            "response": generated_text.strip(),
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
    # Startup
    logger.info("Starting LLM Inference Server...")
    yield
    # Shutdown
    logger.info("Shutting down LLM Inference Server...")
    models.clear()
    tokenizers.clear()


def create_app() -> "FastAPI":
    """Create FastAPI application."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="VLN LLM Inference Server",
        description="HTTP API for LLM inference in VLN multi-agent system",
        version="1.0.0",
        lifespan=lifespan,
    )

    class GenerateRequest(BaseModel):
        """Request model for text generation."""
        model: str = Field(..., description="Model identifier (e.g., qwen-2b-perception)")
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
            models_loaded=list(models.keys()),
            model_configs={
                k: {"description": v["description"], "max_new_tokens": v["max_new_tokens"]}
                for k, v in model_configs.items()
            },
            gpu_memory=gpu_memory,
        )

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate text using specified model."""
        if request.model not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} not found. Available: {list(models.keys())}"
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
        image_base64: str = Field(..., description="Base64 encoded image")
        max_new_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
        temperature: Optional[float] = Field(None, description="Sampling temperature (0.0-2.0)")

    class GenerateVisionResponse(BaseModel):
        """Response model for vision-language generation."""
        response: str
        model: str
        tokens_generated: int
        latency_ms: float
        error: Optional[str] = None

    @app.post("/generate_vision", response_model=GenerateVisionResponse)
    async def generate_vision_endpoint(request: GenerateVisionRequest):
        """Generate text from image and prompt using VLM."""
        result = generate_vision(
            model_key=request.model,
            prompt=request.prompt,
            image_base64=request.image_base64,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
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
                    "loaded": k in models,
                    "description": v["description"],
                    "max_new_tokens": v["max_new_tokens"],
                    "default_temperature": v["default_temperature"],
                }
                for k, v in model_configs.items()
            }
        }

    @app.post("/load_model")
    async def load_model_endpoint(model_key: str, use_int8: bool = True):
        """Load a specific model."""
        if model_key in models:
            return {"status": "already_loaded", "model": model_key}

        success = load_model(model_key, use_int8)
        if success:
            return {"status": "loaded", "model": model_key}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load {model_key}")

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VLN LLM Inference Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8 quantization")
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
    use_int8 = not args.no_int8
    load_all_models(use_int8)

    # Create and run app
    import uvicorn
    app = create_app()

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()