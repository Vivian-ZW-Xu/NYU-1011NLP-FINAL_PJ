#!/usr/bin/env python3
"""
Generate K=15 solutions for GSM8K TEST set (Batch Version)
===========================================================
用于后续所有 baseline 和 iterative G+V 实验的统一数据。

优化：使用 batch generation 加速 3-4 倍
"""

import torch
import json
import re
import os
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL = "microsoft/phi-2"
ADAPTER_PATH = "./generator_ovm"
OUTPUT_PATH = "./test_k15.json"

K = 15  # 每题生成 15 个 solutions
BATCH_SIZE = 1  # 每次只处理1题，生成15个序列，12GB显存够用
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9

# ============================================================================
# Helper Functions
# ============================================================================
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def format_prompt(question):
    """Format question into prompt for the generator."""
    return f"""Solve the following math problem step by step. Show your work and end with "Final answer: [number]".

Question: {question}

Solution:"""

def extract_answer(text):
    """Extract numerical answer from solution text."""
    patterns = [
        r"[Ff]inal\s*[Aa]nswer\s*[:=\-–]?\s*\$?([+-]?\d+\.?\d*)",
        r"[Ff]inal\s*[Aa]ns\s*[:=\-–]?\s*\$?([+-]?\d+\.?\d*)",
        r"[Tt]he\s+answer\s+is\s*[:=]?\s*\$?([+-]?\d+\.?\d*)",
        r"[Aa]nswer\s*[:=\-–]\s*\$?([+-]?\d+\.?\d*)",
        r"####\s*\$?([+-]?\d+\.?\d*)",
        r"[Ss]o\s+the\s+(?:answer|total|result)\s+is\s*\$?([+-]?\d+\.?\d*)",
        r"[Tt]hus[,]?\s+(?:the\s+)?(?:answer|total|result)\s+is\s*\$?([+-]?\d+\.?\d*)",
        r"[Tt]herefore[,]?\s+(?:the\s+)?(?:answer|total|result)\s+is\s*\$?([+-]?\d+\.?\d*)",
        r"[Tt]he\s+(?:total|result)\s+is\s*\$?([+-]?\d+\.?\d*)",
        r"=\s*\$?([+-]?\d+\.?\d*)\s*\.?\s*$",
        r"(?:is|equals?|=)\s*\$?([+-]?\d+\.?\d*)\s*\.?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except:
                continue
    return None

def extract_ground_truth(answer_text):
    """Extract ground truth from GSM8K answer format."""
    match = re.search(r"####\s*([+-]?\d+\.?\d*)", answer_text)
    if match:
        try:
            return float(match.group(1))
        except:
            pass
    return None

# ============================================================================
# Batch Generation
# ============================================================================
def generate_batch(model, tokenizer, questions, k, device):
    """
    Batch generation: 同时处理多个问题，每个生成 k 个答案

    Args:
        questions: list of questions
        k: number of solutions per question

    Returns:
        list of list: [[sol1, sol2, ...], [sol1, sol2, ...], ...]
    """
    prompts = [format_prompt(q) for q in questions]

    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    # 记录输入的 token 长度（用于后续截取生成部分）
    # 这比用字符串长度更可靠
    input_length = inputs["input_ids"].shape[1]

    batch_size = len(questions)
    all_solutions = [[] for _ in range(batch_size)]

    # 分批生成，每次5个序列，避免OOM
    seqs_per_call = 5
    num_calls = (k + seqs_per_call - 1) // seqs_per_call

    for call_idx in range(num_calls):
        current_k = min(seqs_per_call, k - call_idx * seqs_per_call)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=current_k,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # outputs shape: (batch_size * current_k, seq_len)
        # HuggingFace generate() 输出顺序: 每个input的sequences连续排列
        # 即 [input0_seq0, input0_seq1, ..., input1_seq0, input1_seq1, ...]
        for i in range(batch_size):
            for j in range(current_k):
                output_idx = i * current_k + j  # 正确索引
                generated_tokens = outputs[output_idx][input_length:]
                solution = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                all_solutions[i].append(solution)

    return all_solutions


def generate_single(model, tokenizer, question, k, device):
    """
    单个问题生成 k 个答案（fallback，用于 batch 处理不了的情况）
    """
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    solutions = []
    seqs_per_call = 5
    num_calls = (k + seqs_per_call - 1) // seqs_per_call

    for call_idx in range(num_calls):
        current_k = min(seqs_per_call, k - call_idx * seqs_per_call)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=current_k,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for i in range(current_k):
            generated_tokens = outputs[i][input_length:]
            solution = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            solutions.append(solution)

    return solutions


# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 60)
    log("Generating K=15 solutions for GSM8K TEST set (Batch Mode)")
    log("=" * 60)
    log(f"Batch size: {BATCH_SIZE}")
    log(f"Solutions per question: {K}")
    log(f"Design: 15 solutions = 3 rounds x 5 per round")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        log(f"GPU Memory: {gpu_mem:.1f} GB")

        # 根据显存调整 batch size
        if gpu_mem < 12:
            BATCH_SIZE_ACTUAL = 2
        elif gpu_mem < 24:
            BATCH_SIZE_ACTUAL = 4
        else:
            BATCH_SIZE_ACTUAL = 8
        log(f"Adjusted batch size: {BATCH_SIZE_ACTUAL}")
    else:
        BATCH_SIZE_ACTUAL = 1

    # Load model
    log(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    log(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 对于 decoder-only 模型，左 padding

    log("Model loaded!")

    # Load GSM8K test set
    log("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset["test"]
    log(f"Test set size: {len(test_set)} questions")

    # Check for existing checkpoint
    results = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r") as f:
                results = json.load(f)
            start_idx = len(results)
            if start_idx > 0:
                log(f"Found checkpoint with {start_idx} questions, resuming...")
        except:
            log("Checkpoint corrupted, starting fresh...")
            results = []
            start_idx = 0

    # Prepare data
    remaining_items = list(test_set)[start_idx:]
    log(f"\nGenerating solutions...")
    log(f"Total questions: {len(test_set)}")
    log(f"Starting from: {start_idx}")
    log(f"Remaining: {len(remaining_items)}")

    # Process in batches
    num_batches = (len(remaining_items) + BATCH_SIZE_ACTUAL - 1) // BATCH_SIZE_ACTUAL

    pbar = tqdm(total=len(remaining_items), initial=0, desc="Generating")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE_ACTUAL
        batch_end = min(batch_start + BATCH_SIZE_ACTUAL, len(remaining_items))
        batch_items = remaining_items[batch_start:batch_end]

        questions = [item["question"] for item in batch_items]

        try:
            # Batch generation
            if len(questions) > 1:
                all_solutions = generate_batch(model, tokenizer, questions, K, device)
            else:
                # 单个问题用单独处理
                all_solutions = [generate_single(model, tokenizer, questions[0], K, device)]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"OOM at batch {batch_idx}, falling back to single generation...")
                torch.cuda.empty_cache()
                all_solutions = []
                for q in questions:
                    sols = generate_single(model, tokenizer, q, K, device)
                    all_solutions.append(sols)
            else:
                raise e

        # Process results
        for i, item in enumerate(batch_items):
            global_idx = start_idx + batch_start + i
            solutions = all_solutions[i]
            ground_truth = extract_ground_truth(item["answer"])

            solution_data = []
            for sol in solutions:
                predicted = extract_answer(sol)
                is_correct = (predicted == ground_truth) if predicted is not None and ground_truth is not None else False
                solution_data.append({
                    "solution": sol,
                    "predicted_answer": predicted,
                    "is_correct": is_correct,
                    "reasoning_length": len(sol),
                })

            results.append({
                "index": global_idx,
                "question": item["question"],
                "ground_truth": ground_truth,
                "solutions": solution_data,
            })

        pbar.update(len(batch_items))

        # Save checkpoint
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)

    pbar.close()

    # Final save
    log(f"\nSaving final results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Statistics
    log("\n" + "=" * 60)
    log("GENERATION COMPLETE")
    log("=" * 60)

    total_solutions = len(results) * K
    correct_solutions = sum(
        sum(1 for s in item["solutions"] if s["is_correct"])
        for item in results
    )

    # Pass@k statistics
    for k_val in [1, 5, 10, 15]:
        questions_with_correct = sum(
            1 for item in results
            if any(s["is_correct"] for s in item["solutions"][:k_val])
        )
        log(f"Pass@{k_val}: {questions_with_correct}/{len(results)} ({100*questions_with_correct/len(results):.2f}%)")

    log(f"\nTotal solutions: {total_solutions}")
    log(f"Correct solutions: {correct_solutions} ({100*correct_solutions/total_solutions:.1f}%)")
    log("=" * 60)
    log(f"Output saved to: {OUTPUT_PATH}")
    log("Ready for evaluation with evaluate_gv_v2.py")

if __name__ == "__main__":
    main()
