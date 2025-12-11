#!/usr/bin/env python3
"""
Generator Evaluation Script
============================
Evaluate the trained Phi-2 + LoRA generator on GSM8K validation set.

Usage: python eval_generator.py
"""

import torch
import re
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL = "microsoft/phi-2"
ADAPTER_PATH = "./generator_ovm"
SEED = 42

# ============================================================================
# Helper Functions
# ============================================================================
def format_gsm8k_example(example):

    question = example["question"].strip()
    answer = example["answer"].strip()

    if "####" not in answer:
        return {"prompt": None, "response": None, "question": question, "answer": None}

    ground_truth_raw = answer.split("####")[-1].strip()
    m = re.search(r"-?\d+(\.\d+)?", ground_truth_raw)
    if not m:
        return {"prompt": None, "response": None, "question": question, "answer": None}

    ground_truth = m.group(0)

    reasoning = answer.split("####")[0].strip()
    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)

    prompt = f"Question: {question}\n\nLet's solve this step by step.\n"
    response = f"{reasoning}\n\nFinal answer: {ground_truth}"

    return {
        "prompt": prompt,
        "response": response,
        "question": question,
        "answer": ground_truth
    }


def validate_example(example):
    return example["prompt"] is not None


def generate_solution(model, tokenizer, question, max_new_tokens=512):

    prompt = f"Question: {question}\n\nLet's solve this step by step.\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = generated.split("Let's solve this step by step.")[-1].strip()
    return solution


def extract_answer(solution_text):

    match = re.search(r'Final answer:\s*([^\n]+)', solution_text, re.IGNORECASE)
    if not match:
        return None

    answer_str = match.group(1).strip()
    answer_str = re.sub(r'[$,]', '', answer_str)

    num_match = re.search(r'-?\d+\.?\d*', answer_str)
    if num_match:
        return float(num_match.group())
    return None


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Generator Evaluation (Pass@1)")
    print("=" * 60)

    print(f"\nLoading model from {ADAPTER_PATH}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    print("Model loaded!")

    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
    val_set = train_val["test"]

    val_formatted = val_set.map(format_gsm8k_example)
    val_formatted = val_formatted.filter(validate_example)
    print(f"Validation examples: {len(val_formatted)}")

    print("\nEvaluating...")
    correct = 0
    total = 0
    invalid_format = 0

    for i, example in enumerate(tqdm(val_formatted, desc="Evaluating")):
        question = example["question"]
        ground_truth_str = str(example["answer"])
        ground_truth_str = re.sub(r'[$,]', '', ground_truth_str)

        try:
            ground_truth = float(ground_truth_str)
        except:
            continue

        solution = generate_solution(model, tokenizer, question)
        predicted = extract_answer(solution)

        if predicted is None:
            invalid_format += 1
            continue

        is_correct = abs(predicted - ground_truth) < 1e-4
        if is_correct:
            correct += 1
        total += 1

        if i < 3:
            print(f"\n{'─' * 50}")
            print(f"Example {i+1}: {'✅' if is_correct else '❌'}")
            print(f"Q: {question[:80]}...")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted: {predicted}")

    accuracy = correct / total if total > 0 else 0
    format_rate = (total / len(val_formatted)) * 100

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Pass@1 Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Valid Format Rate: {format_rate:.2f}%")
    print(f"Invalid Format: {invalid_format}")
    print("=" * 60)

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "format_rate": format_rate,
        "invalid_format": invalid_format
    }

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to eval_results.json")


if __name__ == "__main__":
    main()
