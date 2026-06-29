"""Test fine-tuned decision model for emergency navigation."""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def test_finetuned_model():
    """Test the fine-tuned decision model."""

    base_model_path = "/data/WZ/Model/Qwen/Qwen3___5-9B"  # Use original model, not AWQ
    lora_path = "outputs/qlora_decision/decision"

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    # Test cases
    test_cases = [
        {
            "input": """Navigation Context:
- Current Position: Starting point
- Goal Position: [10, 0, 10]
- Obstacle Detected: blocked_path at step 10
- Obstacle Position: [5, 0, 5]

Instruction: Walk forward, the path is blocked, turn left and find another way to reach the destination.

Analyze the emergency situation and provide navigation actions.""",
            "expected_keywords": ["risk_score", "reasoning", "actions", "turn_left", "forward"]
        },
        {
            "input": """Navigation Context:
- Current Position: Starting point
- Goal Position: [15, 0, 8]
- Obstacle Detected: dynamic_obstacle at step 15
- Obstacle Position: [7, 0, 4]

Instruction: Proceed forward, obstacle appeared, navigate around.

Analyze the emergency situation and provide navigation actions.""",
            "expected_keywords": ["risk_score", "reasoning", "actions", "dynamic"]
        },
        {
            "input": """Navigation Context:
- Current Position: Starting point
- Goal Position: [8, 0, 12]
- Obstacle Detected: emergency_evacuation at step 8
- Obstacle Position: [4, 0, 6]

Instruction: Emergency: Navigate to the exit, obstacle detected, find safe exit immediately.

Analyze the emergency situation and provide navigation actions.""",
            "expected_keywords": ["risk_score", "emergency", "actions"]
        }
    ]

    print("\n" + "=" * 60)
    print("Testing Fine-tuned Decision Model")
    print("=" * 60 + "\n")

    results = []
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print("-" * 40)

        # Format prompt
        prompt = f"""<|im_start|>system
You are an emergency navigation assistant. Given the navigation context and detected obstacle, determine the best action sequence to safely reach the goal.
<|im_end|>
<|im_start|>user
{test_case['input']}
<|im_end|>
<|im_start|>assistant
"""

        # Generate
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

        # Extract assistant response
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        print(f"Input: {test_case['input'][:100]}...")
        print(f"\nResponse:\n{response}")

        # Check keywords
        found_keywords = []
        for kw in test_case['expected_keywords']:
            if kw.lower() in response.lower():
                found_keywords.append(kw)

        success_rate = len(found_keywords) / len(test_case['expected_keywords'])
        print(f"\nKeywords found: {found_keywords}")
        print(f"Success rate: {success_rate:.1%}")

        results.append({
            "case_id": i + 1,
            "keywords_found": found_keywords,
            "success_rate": success_rate,
            "response": response
        })

    # Summary
    avg_success = sum(r["success_rate"] for r in results) / len(results)
    print("\n" + "=" * 60)
    print(f"Average Success Rate: {avg_success:.1%}")
    print("=" * 60)

    # Save results
    with open("outputs/qlora_decision/test_results.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    test_finetuned_model()