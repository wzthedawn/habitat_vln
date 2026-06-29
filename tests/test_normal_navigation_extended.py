"""Extended validation: Test normal navigation with more samples from R2R dataset.

Compares base model vs fine-tuned model on real navigation instructions.
"""

import os
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_r2r_test_samples(r2r_path: str, num_samples: int = 50) -> list:
    """Load real navigation instructions from R2R dataset."""

    samples = []

    # Load from val_seen (unseen during training)
    file_path = os.path.join(r2r_path, "val_seen/val_seen.json")
    if not os.path.exists(file_path):
        file_path = os.path.join(r2r_path, "train/train.json")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    episodes = data.get("episodes", []) if isinstance(data, dict) else data
    random.seed(42)
    random.shuffle(episodes)

    for ep in episodes[:num_samples]:
        instr = ep.get("instruction", {})
        text = instr.get("instruction_text", "") if isinstance(instr, dict) else str(instr)

        if not text:
            continue

        goal = ep.get("goals", [{}])[0].get("position", [10, 0, 10]) if ep.get("goals") else [10, 0, 10]

        # Extract expected actions from instruction
        expected = extract_expected_keywords(text)

        samples.append({
            "instruction": text,
            "goal": goal,
            "expected_keywords": expected
        })

    print(f"Loaded {len(samples)} test samples from R2R")
    return samples


def extract_expected_keywords(instruction: str) -> list:
    """Extract expected action keywords from instruction."""

    keywords = []
    instr_lower = instruction.lower()

    # Direction keywords
    if "left" in instr_lower:
        keywords.append("left")
    if "right" in instr_lower:
        keywords.append("right")
    if "straight" in instr_lower or "forward" in instr_lower:
        keywords.append("forward")
    if "back" in instr_lower:
        keywords.append("back")

    # Room/object keywords
    rooms = ["kitchen", "bedroom", "bathroom", "hallway", "living", "dining", "door", "window", "stairs"]
    for room in rooms:
        if room in instr_lower:
            keywords.append(room)

    # Action keywords
    actions = ["turn", "walk", "go", "stop", "wait", "continue"]
    for action in actions:
        if action in instr_lower:
            keywords.append(action)

    # Ensure at least some keywords
    if not keywords:
        keywords = ["navigate", "move"]

    return keywords[:5]  # Limit to 5 keywords max


def format_prompt(instruction: str, goal: list) -> str:
    """Format navigation prompt."""

    context = f"""Navigation Context:
- Current Position: Starting point
- Goal Position: {goal}
- No obstacles detected
- Clear path ahead

Instruction: {instruction}

Provide navigation actions to reach the goal."""

    return f"""<|im_start|>system
You are a navigation assistant. Given the navigation context, provide appropriate actions to reach the goal safely.
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
"""


def test_model(model, tokenizer, samples: list, model_name: str) -> dict:
    """Test model on navigation samples."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}\n")

    results = []
    total_keywords = 0
    found_keywords = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(samples)}")

        prompt = format_prompt(sample['instruction'], sample['goal'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Reduced for speed
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        # Check keywords
        case_found = []
        for kw in sample['expected_keywords']:
            if kw.lower() in response.lower():
                case_found.append(kw)
                found_keywords += 1

        total_keywords += len(sample['expected_keywords'])

        results.append({
            "case_id": i + 1,
            "instruction": sample['instruction'][:80],
            "keywords_found": case_found,
            "expected": sample['expected_keywords'],
            "success_rate": len(case_found) / len(sample['expected_keywords']) if sample['expected_keywords'] else 0,
            "response": response[:200]
        })

    overall_rate = found_keywords / total_keywords if total_keywords > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results for {model_name}:")
    print(f"  Overall Success Rate: {overall_rate:.1%}")
    print(f"  Keywords Found: {found_keywords}/{total_keywords}")
    print(f"{'='*60}")

    return {
        "model_name": model_name,
        "overall_success_rate": overall_rate,
        "total_keywords": total_keywords,
        "found_keywords": found_keywords,
        "results": results
    }


def main():
    base_model_path = "/data/WZ/Model/Qwen/Qwen3___5-9B"
    lora_path = "outputs/qlora_decision/decision"
    r2r_path = "/data/WZ/Dataset/R2R_VLNCE_v1-3"

    num_samples = 20  # Reduced for faster validation

    print("="*60)
    print("EXTENDED NORMAL NAVIGATION VALIDATION")
    print(f"Testing with {num_samples} real R2R instructions")
    print("="*60)

    # Load test samples
    test_samples = load_r2r_test_samples(r2r_path, num_samples)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Test 1: Base model
    print("\n" + "="*60)
    print("Loading BASE MODEL...")
    print("="*60)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    base_results = test_model(base_model, tokenizer, test_samples, "Base Model")

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    # Test 2: Fine-tuned model
    print("\n" + "="*60)
    print("Loading FINE-TUNED MODEL...")
    print("="*60)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    ft_model = PeftModel.from_pretrained(base_model, lora_path)
    ft_model.eval()

    ft_results = test_model(ft_model, tokenizer, test_samples, "Fine-tuned Model")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nBase Model Success Rate:       {base_results['overall_success_rate']:.1%}")
    print(f"Fine-tuned Model Success Rate: {ft_results['overall_success_rate']:.1%}")

    delta = ft_results['overall_success_rate'] - base_results['overall_success_rate']
    print(f"\nDifference: {delta:+.1%}")

    if delta < -0.1:
        print("\n⚠️  SIGNIFICANT DEGRADATION DETECTED")
        print("   Recommendation: Use mixed data training")
    elif delta < -0.05:
        print("\n⚠️  MODERATE DEGRADATION DETECTED")
        print("   Recommendation: Consider mixed data training")
    elif delta < 0:
        print("\n⚠️  Minor degradation detected")
    else:
        print("\n✅ No degradation detected")

    # Save results
    output = {
        "config": {
            "num_samples": num_samples,
            "model_path": base_model_path,
            "lora_path": lora_path
        },
        "base_model": {
            "success_rate": base_results['overall_success_rate'],
            "found_keywords": base_results['found_keywords'],
            "total_keywords": base_results['total_keywords']
        },
        "finetuned_model": {
            "success_rate": ft_results['overall_success_rate'],
            "found_keywords": ft_results['found_keywords'],
            "total_keywords": ft_results['total_keywords']
        },
        "comparison": {
            "delta": delta,
            "degradation_detected": delta < -0.05
        },
        "detailed_results": {
            "base": base_results['results'][:10],  # First 10 samples
            "finetuned": ft_results['results'][:10]
        }
    }

    os.makedirs("outputs/qlora_decision", exist_ok=True)
    with open("outputs/qlora_decision/extended_validation.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: outputs/qlora_decision/extended_validation.json")


if __name__ == "__main__":
    main()