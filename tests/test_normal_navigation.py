"""Test fine-tuned model on normal navigation to check for catastrophic forgetting.

This script compares the model's performance on normal navigation tasks
before and after fine-tuning for emergency scenarios.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Normal navigation test cases (no emergency scenarios)
NORMAL_TEST_CASES = [
    {
        "instruction": "Walk straight ahead and turn left at the end of the hallway.",
        "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [8, 0, 5]
- No obstacles detected
- Clear path ahead

Instruction: Walk straight ahead and turn left at the end of the hallway.

Provide navigation actions.""",
        "expected_keywords": ["forward", "turn_left", "straight"]
    },
    {
        "instruction": "Go to the kitchen and find the table.",
        "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [10, 0, 12]
- No obstacles detected
- Room type: kitchen

Instruction: Go to the kitchen and find the table.

Provide navigation actions.""",
        "expected_keywords": ["forward", "kitchen", "table"]
    },
    {
        "instruction": "Turn right and walk through the door.",
        "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [5, 0, 8]
- No obstacles detected
- Door visible ahead

Instruction: Turn right and walk through the door.

Provide navigation actions.""",
        "expected_keywords": ["turn_right", "forward", "door"]
    },
    {
        "instruction": "Walk down the stairs and turn left at the bottom.",
        "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [3, -2, 6]
- No obstacles detected
- Stairs detected ahead

Instruction: Walk down the stairs and turn left at the bottom.

Provide navigation actions.""",
        "expected_keywords": ["stairs", "turn_left", "forward"]
    },
    {
        "instruction": "Navigate to the bedroom and stop at the bed.",
        "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [12, 0, 15]
- No obstacles detected
- Room type: bedroom

Instruction: Navigate to the bedroom and stop at the bed.

Provide navigation actions.""",
        "expected_keywords": ["bedroom", "bed", "forward"]
    }
]


def format_prompt(context: str) -> str:
    """Format prompt in Qwen chat format."""
    return f"""<|im_start|>system
You are a navigation assistant. Given the navigation context, provide appropriate actions to reach the goal safely.
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
"""


def test_model(model, tokenizer, test_cases: list, model_name: str) -> dict:
    """Test model on navigation tasks."""

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}\n")

    results = []
    total_keywords = 0
    found_keywords = 0

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Instruction: {test_case['instruction'][:60]}...")

        prompt = format_prompt(test_case['context'])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Clean response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        print(f"\nResponse:\n{response[:300]}...")

        # Check keywords
        case_found = []
        for kw in test_case['expected_keywords']:
            if kw.lower() in response.lower():
                case_found.append(kw)
                found_keywords += 1

        total_keywords += len(test_case['expected_keywords'])
        case_rate = len(case_found) / len(test_case['expected_keywords'])

        print(f"\nKeywords found: {case_found} ({case_rate:.0%})")

        results.append({
            "case_id": i + 1,
            "instruction": test_case['instruction'],
            "keywords_found": case_found,
            "success_rate": case_rate,
            "response": response
        })

    overall_rate = found_keywords / total_keywords if total_keywords > 0 else 0

    print(f"\n{'='*60}")
    print(f"Overall Success Rate: {overall_rate:.1%}")
    print(f"{'='*60}")

    return {
        "model_name": model_name,
        "overall_success_rate": overall_rate,
        "results": results
    }


def main():
    base_model_path = "/data/WZ/Model/Qwen/Qwen3___5-9B"
    lora_path = "outputs/qlora_decision/decision"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Test 1: Base model (before fine-tuning)
    print("\n" + "="*60)
    print("Loading BASE MODEL (before fine-tuning)...")
    print("="*60)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    base_results = test_model(base_model, tokenizer, NORMAL_TEST_CASES, "Base Model (Before Fine-tuning)")

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

    ft_results = test_model(ft_model, tokenizer, NORMAL_TEST_CASES, "Fine-tuned Model")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nBase Model Success Rate:     {base_results['overall_success_rate']:.1%}")
    print(f"Fine-tuned Model Success Rate: {ft_results['overall_success_rate']:.1%}")

    delta = ft_results['overall_success_rate'] - base_results['overall_success_rate']
    if delta < -0.1:
        print(f"\n⚠️ WARNING: Significant degradation detected ({delta:.1%})")
        print("Recommendation: Use mixed data training to prevent catastrophic forgetting")
    elif delta < 0:
        print(f"\n⚠️ Minor degradation detected ({delta:.1%})")
        print("Recommendation: Consider adding normal navigation samples to training")
    else:
        print(f"\n✅ No degradation detected ({delta:+.1%})")

    # Save results
    output = {
        "base_model": base_results,
        "finetuned_model": ft_results,
        "comparison": {
            "base_rate": base_results['overall_success_rate'],
            "finetuned_rate": ft_results['overall_success_rate'],
            "delta": delta,
            "degradation_detected": delta < -0.1
        }
    }

    with open("outputs/qlora_decision/normal_nav_comparison.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: outputs/qlora_decision/normal_nav_comparison.json")


if __name__ == "__main__":
    main()