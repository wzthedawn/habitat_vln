"""QLoRA Fine-tuning Script for Emergency Navigation.

Fine-tunes Qwen model using QLoRA (Quantized LoRA) for emergency navigation tasks.
Supports both decision and perception models.

Usage:
    python train_qlora.py --task decision --data_dir /data/WZ/Dataset/qlora_train --output_dir outputs/qlora_decision

Memory: ~15GB GPU for 7B model with 4-bit quantization
"""

import os
import sys
import json
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QLoRATrainer")


def check_dependencies():
    """Check required packages are installed."""
    required = ["transformers", "peft", "bitsandbytes", "datasets", "accelerate"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)


check_dependencies()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import Dataset
import bitsandbytes as bnb


@dataclass
class QLoRAConfig:
    """QLoRA training configuration."""

    # Model - 使用原始Qwen3.5-9B模型（非AWQ）
    model_name: str = "/data/WZ/Model/Qwen/Qwen3___5-9B"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bf16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # LoRA - Correct target modules for Qwen3.5-9B-AWQ (linear attention model)
    # Note: This model uses linear attention, not standard q/k/v/o_proj
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "in_proj_qkv", "out_proj",  # Linear attention layers
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ])

    # Training
    output_dir: str = "outputs/qlora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01

    # Optimization
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # Data
    max_seq_length: int = 1024
    packing: bool = False

    # Logging
    logging_steps: int = 5
    save_steps: int = 20
    eval_steps: int = 20
    save_total_limit: int = 5

    # Misc
    seed: int = 42


def load_model_and_tokenizer(config: QLoRAConfig):
    """Load model and tokenizer.

    For AWQ quantized models, load directly without additional quantization.
    For non-quantized models, apply 4-bit quantization.
    """

    logger.info(f"Loading model: {config.model_name}")

    # Check if model is AWQ quantized
    is_awq = "awq" in config.model_name.lower() or "AWQ" in config.model_name

    if is_awq:
        # Load AWQ model directly (already quantized)
        logger.info("Loading AWQ quantized model...")
        # Use device_map={"": 0} to prevent Trainer from trying to move model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map={"": 0},  # Stay on GPU 0, don't let Trainer move it
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        # Mark model as parallelizable to prevent Trainer device move
        model.is_parallelizable = True
        model.model_parallel = True
    else:
        # Apply 4-bit quantization for non-AWQ models
        compute_dtype = torch.bfloat16 if config.bnb_4bit_compute_dtype == "bf16" else torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.use_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},  # Stay on GPU 0
            trust_remote_code=True,
        )
        # Mark model as parallelizable to prevent Trainer device move
        model.is_parallelizable = True
        model.model_parallel = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare model for k-bit training (NOT needed for AWQ models)
    if not is_awq:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def apply_lora(model, config: QLoRAConfig):
    """Apply LoRA adapters to model."""

    logger.info("Applying LoRA adapters...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        modules_to_save=None,
    )

    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} "
                f"({100 * trainable_params / all_params:.2f}%)")

    return model


def load_dataset(data_path: str, split: str = "train") -> Dataset:
    """Load JSONL dataset."""

    file_path = os.path.join(data_path, f"{split}.jsonl") if not data_path.endswith(".jsonl") else data_path

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} samples from {file_path}")

    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_length: int):
    """Tokenize examples."""

    # Tokenize
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )

    # Labels = input_ids (for causal LM)
    model_inputs["labels"] = model_inputs["input_ids"].copy()

    return model_inputs


def train(config: QLoRAConfig, data_dir: str, task: str = "decision"):
    """Run QLoRA training."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting QLoRA Training: {task}")
    logger.info(f"{'='*60}\n")

    # Update output dir
    config.output_dir = os.path.join(config.output_dir, task)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply LoRA
    model = apply_lora(model, config)

    # Load datasets
    train_file = os.path.join(data_dir, "train.jsonl")
    val_file = os.path.join(data_dir, "val.jsonl")

    train_dataset = load_dataset(train_file) if os.path.exists(train_file) else None
    eval_dataset = load_dataset(val_file) if os.path.exists(val_file) else None

    if train_dataset is None:
        logger.error(f"Training data not found: {train_file}")
        return

    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    tokenized_eval = None
    if eval_dataset:
        tokenized_eval = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval",
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if tokenized_eval else None,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        bf16=True,
        fp16=False,
        eval_strategy="steps" if tokenized_eval else "no",
        load_best_model_at_end=True if tokenized_eval else False,
        metric_for_best_model="eval_loss" if tokenized_eval else None,
        report_to="none",
        seed=config.seed,
        # Prevent Trainer from trying to move model
        ddp_find_unused_parameters=False,
        skip_memory_metrics=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Save training metrics
    with open(os.path.join(config.output_dir, "train_metrics.json"), 'w') as f:
        json.dump(train_result.metrics, f, indent=2)

    logger.info(f"Training complete. Model saved to {config.output_dir}")

    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    """Merge LoRA weights with base model."""

    logger.info("Merging LoRA weights...")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge
    merged_model = model.merge_and_unload()

    # Save
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    logger.info(f"Merged model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for emergency navigation")
    parser.add_argument("--task", type=str, choices=["decision", "perception"],
                        default="decision", help="Task to train")
    parser.add_argument("--data_dir", type=str, default="/data/WZ/Dataset/qlora_balanced",
                        help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="outputs/qlora_balanced",
                        help="Output directory for trained model")
    parser.add_argument("--model_name", type=str, default="/data/WZ/Model/Qwen/Qwen3___5-9B",
                        help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    args = parser.parse_args()

    # Create config
    config = QLoRAConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Run training
    train(config, args.data_dir, args.task)


if __name__ == "__main__":
    main()