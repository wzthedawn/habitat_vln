"""LoRA Fine-tuning Configuration for Emergency Navigation.

This module provides configuration for fine-tuning Qwen3.5-9B-AWQ
for emergency navigation tasks.

Target: 24GB GPU memory
Tasks:
  1. Perception model: RGB+Depth -> danger_info
  2. Decision model: context -> emergency action sequence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])

    # Training parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    use_8bit_optimizer: bool = True
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # or "fp16"

    # Logging
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100

    # Paths
    output_dir: str = "outputs/lora_finetune"
    logging_dir: str = "outputs/logs"


@dataclass
class PerceptionTaskConfig:
    """Configuration for perception model fine-tuning."""

    # Task info
    task_name: str = "emergency_perception"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # or local path

    # Input/Output
    input_fields: List[str] = field(default_factory=lambda: [
        "rgb_image", "depth_image", "instruction"
    ])
    output_fields: List[str] = field(default_factory=lambda: [
        "room_type", "objects", "nav_hint", "danger_info"
    ])

    # Data
    max_samples: int = 400
    image_size: int = 224
    max_text_length: int = 512

    # LoRA config
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    # Estimated memory
    estimated_memory_gb: float = 12.0


@dataclass
class DecisionTaskConfig:
    """Configuration for decision model fine-tuning."""

    # Task info
    task_name: str = "emergency_decision"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # or local path

    # Input/Output
    input_fields: List[str] = field(default_factory=lambda: [
        "instruction", "perception_info", "trajectory_info", "emergency_signal"
    ])
    output_fields: List[str] = field(default_factory=lambda: [
        "risk_score", "reasoning", "action_sequence"
    ])

    # Data
    max_samples: int = 600
    max_text_length: int = 1024

    # LoRA config
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    # Estimated memory
    estimated_memory_gb: float = 8.0


def get_training_config(task: str = "decision") -> Dict[str, Any]:
    """Get training configuration for specified task.

    Args:
        task: "perception" or "decision"

    Returns:
        Configuration dictionary
    """
    if task == "perception":
        config = PerceptionTaskConfig()
    else:
        config = DecisionTaskConfig()

    return {
        "task_name": config.task_name,
        "model_name": config.model_name,
        "input_fields": config.input_fields,
        "output_fields": config.output_fields,
        "max_samples": config.max_samples,
        "max_text_length": config.max_text_length,
        "lora": {
            "r": config.lora_config.lora_rank,
            "lora_alpha": config.lora_config.lora_alpha,
            "lora_dropout": config.lora_config.lora_dropout,
            "target_modules": config.lora_config.target_modules,
        },
        "training": {
            "learning_rate": config.lora_config.learning_rate,
            "num_train_epochs": config.lora_config.num_epochs,
            "per_device_train_batch_size": config.lora_config.batch_size,
            "gradient_accumulation_steps": config.lora_config.gradient_accumulation_steps,
            "warmup_steps": config.lora_config.warmup_steps,
            "max_grad_norm": config.lora_config.max_grad_norm,
            "weight_decay": config.lora_config.weight_decay,
            "lr_scheduler_type": config.lora_config.lr_scheduler_type,
        },
        "optimization": {
            "use_8bit_optimizer": config.lora_config.use_8bit_optimizer,
            "use_gradient_checkpointing": config.lora_config.use_gradient_checkpointing,
            "mixed_precision": config.lora_config.mixed_precision,
        },
        "logging": {
            "logging_steps": config.lora_config.logging_steps,
            "save_steps": config.lora_config.save_steps,
            "eval_steps": config.lora_config.eval_steps,
            "output_dir": config.lora_config.output_dir,
            "logging_dir": config.lora_config.logging_dir,
        },
        "estimated_memory_gb": config.estimated_memory_gb,
    }


def save_training_config(config: Dict, output_path: str) -> None:
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Configuration saved to {output_path}")


def create_training_script(config_path: str, output_path: str) -> None:
    """Create training script from configuration.

    This generates a Python script that can be run to start training.
    """
    script_content = '''#!/usr/bin/env python
"""Auto-generated LoRA fine-tuning script for emergency navigation."""

import os
import json
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes as bnb


def load_config(config_path: str) -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_model(model_name: str, use_4bit: bool = True):
    """Load model with optional quantization."""

    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def apply_lora(model, lora_config: Dict):
    """Apply LoRA to model."""

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Create LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def load_dataset(data_path: str, max_samples: int = None):
    """Load training dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data.get("samples", [])
    if max_samples:
        samples = samples[:max_samples]

    return Dataset.from_list(samples)


def format_sample(sample: Dict, tokenizer, max_length: int = 512) -> Dict:
    """Format sample for training."""

    # Create prompt
    instruction = sample.get("instruction", "")
    expected = sample.get("expected_output", {})

    # Format as conversation
    prompt = f"""<|im_start|>system
You are an emergency navigation assistant. Help navigate around obstacles safely.
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{json.dumps(expected, ensure_ascii=False)}
<|im_end|>"""

    # Tokenize
    encoded = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"].squeeze(),
        "attention_mask": encoded["attention_mask"].squeeze(),
        "labels": encoded["input_ids"].squeeze(),
    }


def train(config_path: str, data_path: str):
    """Run fine-tuning."""

    # Load config
    config = load_config(config_path)

    print(f"Training task: {config['task_name']}")
    print(f"Estimated memory: {config['estimated_memory_gb']} GB")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(config["model_name"], use_4bit=True)

    # Apply LoRA
    print("Applying LoRA...")
    model = apply_lora(model, config["lora"])

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(data_path, config.get("max_samples"))

    # Format dataset
    def preprocess(sample):
        return format_sample(sample, tokenizer, config["max_text_length"])

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["logging"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        warmup_steps=config["training"]["warmup_steps"],
        logging_dir=config["logging"]["logging_dir"],
        logging_steps=config["logging"]["logging_steps"],
        save_steps=config["logging"]["save_steps"],
        bf16=True,
        gradient_checkpointing=config["optimization"]["use_gradient_checkpointing"],
        optim="paged_adamw_8bit" if config["optimization"]["use_8bit_optimizer"] else "adamw_torch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    final_output = os.path.join(config["logging"]["output_dir"], "final_model")
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"Model saved to {final_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    args = parser.parse_args()

    train(args.config, args.data)
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    os.chmod(output_path, 0o755)
    print(f"Training script created: {output_path}")


def main():
    """Generate configurations and scripts."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate LoRA fine-tuning configs")
    parser.add_argument("--output-dir", type=str, default="configs/finetune")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate perception config
    perception_config = get_training_config("perception")
    save_training_config(
        perception_config,
        os.path.join(args.output_dir, "perception_lora_config.json")
    )

    # Generate decision config
    decision_config = get_training_config("decision")
    save_training_config(
        decision_config,
        os.path.join(args.output_dir, "decision_lora_config.json")
    )

    # Create training script
    create_training_script(
        os.path.join(args.output_dir, "decision_lora_config.json"),
        os.path.join(args.output_dir, "run_lora_train.py")
    )

    print("\nConfiguration files generated:")
    print(f"  - {args.output_dir}/perception_lora_config.json")
    print(f"  - {args.output_dir}/decision_lora_config.json")
    print(f"  - {args.output_dir}/run_lora_train.py")

    print("\nEstimated GPU memory requirements:")
    print(f"  - Perception model: {perception_config['estimated_memory_gb']} GB")
    print(f"  - Decision model: {decision_config['estimated_memory_gb']} GB")
    print(f"  - Total: {perception_config['estimated_memory_gb'] + decision_config['estimated_memory_gb']} GB")


if __name__ == "__main__":
    main()