#!/usr/bin/env python
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
