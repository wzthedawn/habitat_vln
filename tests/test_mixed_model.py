"""Validate mixed-data trained model on both emergency and normal navigation."""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def format_prompt(context: str, is_emergency: bool = True) -> str:
    """Format prompt in Qwen chat format."""
    system = "You are an emergency navigation assistant. Given the navigation context and detected obstacle, determine the best action sequence to safely reach the goal." if is_emergency else "You are a navigation assistant. Given the navigation context, determine the best action sequence to reach the goal safely."
    return f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{context}
<|im_end|>
<|im_start|>assistant
"""


def test_model(model, tokenizer, test_cases: list):
    """Test model on given cases."""
    results = []
    for i, case in enumerate(test_cases):
        prompt = format_prompt(case['context'], case.get('is_emergency', True))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        # Check keywords
        found = [kw for kw in case['expected_keywords'] if kw.lower() in response.lower()]
        success_rate = len(found) / len(case['expected_keywords']) if case['expected_keywords'] else 0

        results.append({
            "case_id": i + 1,
            "type": case['type'],
            "success_rate": success_rate,
            "keywords_found": found,
            "response": response[:200]
        })

        print(f"[{case['type']}] Case {i+1}: {success_rate:.0%} - {found}")

    return results


def main():
    base_model_path = "/data/WZ/Model/Qwen/Qwen3___5-9B"
    lora_path = "outputs/qlora_mixed/decision"  # Updated path

    # Test cases
    emergency_cases = [
        {
            "type": "emergency",
            "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [10, 0, 10]
- Obstacle Detected: blocked_path at step 10
- Obstacle Position: [5, 0, 5]

Instruction: Walk forward, the path is blocked, turn left to find another way.

Provide navigation actions.""",
            "expected_keywords": ["risk_score", "actions", "turn_left", "forward"],
            "is_emergency": True
        },
        {
            "type": "emergency",
            "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [15, 0, 8]
- Obstacle Detected: dynamic_obstacle at step 12
- Obstacle Position: [7, 0, 4]

Instruction: Proceed forward, obstacle appeared, navigate around.

Provide navigation actions.""",
            "expected_keywords": ["risk_score", "actions", "dynamic", "forward"],
            "is_emergency": True
        }
    ]

    normal_cases = [
        {
            "type": "normal",
            "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [8, 0, 5]
- No obstacles detected
- Clear path ahead

Instruction: Walk straight ahead and turn left at the end of the hallway.

Provide navigation actions.""",
            "expected_keywords": ["forward", "turn_left", "straight"],
            "is_emergency": False
        },
        {
            "type": "normal",
            "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [10, 0, 12]
- No obstacles detected
- Room type: kitchen

Instruction: Go to the kitchen and find the table.

Provide navigation actions.""",
            "expected_keywords": ["forward", "kitchen"],
            "is_emergency": False
        },
        {
            "type": "normal",
            "context": """Navigation Context:
- Current Position: Starting point
- Goal Position: [5, 0, 8]
- No obstacles detected
- Door visible ahead

Instruction: Turn right and walk through the door.

Provide navigation actions.""",
            "expected_keywords": ["turn_right", "forward", "door"],
            "is_emergency": False
        }
    ]

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    print("\n" + "="*60)
    print("EMERGENCY NAVIGATION TEST")
    print("="*60)
    emergency_results = test_model(model, tokenizer, emergency_cases)
    emergency_avg = sum(r['success_rate'] for r in emergency_results) / len(emergency_results)

    print("\n" + "="*60)
    print("NORMAL NAVIGATION TEST")
    print("="*60)
    normal_results = test_model(model, tokenizer, normal_cases)
    normal_avg = sum(r['success_rate'] for r in normal_results) / len(normal_results)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Emergency Navigation: {emergency_avg:.1%}")
    print(f"Normal Navigation:    {normal_avg:.1%}")

    # Save results
    output = {
        "emergency_avg": emergency_avg,
        "normal_avg": normal_avg,
        "emergency_results": emergency_results,
        "normal_results": normal_results
    }
    with open("outputs/qlora_mixed/validation_results.json", 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: outputs/qlora_mixed/validation_results.json")


if __name__ == "__main__":
    main()