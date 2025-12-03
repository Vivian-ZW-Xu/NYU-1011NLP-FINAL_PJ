#!/usr/bin/env python3
"""
Generate Verifier Training Data (with Batching)
================================================
This script generates solution candidates for Verifier training using
the fine-tuned Generator (Phi-2 + QLoRA on GSM8K).

Data splits:
- Training: 2000 questions (indices 0-1999) with k=1,3,5,7 solutions each
- Evaluation: 500 questions (indices 2000-2499) with k=1,3,5,7 solutions each

k=1 uses greedy decoding, k>1 uses sampling (temperature=0.7, top_p=0.95, top_k=50)
"""

import torch
import json
import re
import random
import os
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 50
BATCH_SIZE = 4  # Number of solutions to generate in parallel

# Data splits
TRAIN_START = 0
TRAIN_END = 2000
EVAL_START = 2000
EVAL_END = 2500

# Paths
MODEL_PATH = "./generator_ovm"
BASE_MODEL = "microsoft/phi-2"
OUTPUT_DIR = "./verifier_data"

# ============================================================================
# Set random seeds for reproducibility
# ============================================================================
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Helper Functions
# ============================================================================
def log(msg):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def format_prompt(question):
    """Format a question into a prompt."""
    return f"""Question: {question}

Let's solve this step by step.
"""

def generate_solutions_batch(model, tokenizer, question, num_solutions, use_sampling=True):
    """Generate multiple solutions for a question using batching."""
    prompt = format_prompt(question)

    # Create batch of identical prompts
    inputs = tokenizer(
        [prompt] * num_solutions,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        if use_sampling:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Decode all outputs
    solutions = []
    for i in range(num_solutions):
        generated = tokenizer.decode(outputs[i], skip_special_tokens=True)
        solution = generated.split("Let's solve this step by step.")[-1].strip()
        solutions.append(solution)

    return solutions

def generate_k_solutions(model, tokenizer, question, k):
    """Generate k solutions for a question using batching.

    k=1: greedy decoding (single solution)
    k>1: sampling with batching
    """
    if k == 1:
        # Greedy decoding for k=1
        return generate_solutions_batch(model, tokenizer, question, 1, use_sampling=False)

    # For k>1, generate in batches
    solutions = []
    remaining = k

    while remaining > 0:
        batch_size = min(remaining, BATCH_SIZE)
        batch_solutions = generate_solutions_batch(
            model, tokenizer, question, batch_size, use_sampling=True
        )
        solutions.extend(batch_solutions)
        remaining -= batch_size

    return solutions

def extract_answer(text):
    """Extract numerical answer from solution text."""
    patterns = [
        r'[Ff]inal [Aa]nswer[:\s]*([-\d\.,]+)',
        r'[Aa]nswer[:\s]*([-\d\.,]+)',
        r'[Tt]he answer is[:\s]*([-\d\.,]+)',
        r'=\s*([-\d\.,]+)\s*$',
        r'([-\d\.,]+)\s*$',
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except:
                pass
    return None

def extract_gt(answer):
    """Extract ground truth from GSM8K answer format."""
    m = re.search(r'####\s*([-\d\.,]+)', answer)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except:
            pass
    return None

def is_correct(pred, gt, tol=1e-4):
    """Check if prediction matches ground truth within tolerance."""
    if pred is None or gt is None:
        return False
    return abs(pred - gt) < tol

def process_questions(model, tokenizer, questions, split_name, k=7):
    """Process a set of questions and generate k solutions for each."""
    results = []

    log(f"Processing {len(questions)} {split_name} questions with k={k} (batch_size={BATCH_SIZE})...")

    for i, item in enumerate(tqdm(questions, desc=f"{split_name}")):
        question = item["question"]
        answer = item["answer"]
        gt = extract_gt(answer)

        # Generate k solutions using batching
        solutions = generate_k_solutions(model, tokenizer, question, k)

        # Extract predictions and check correctness for each solution
        solution_data = []
        for sol in solutions:
            pred = extract_answer(sol)
            correct = is_correct(pred, gt)
            solution_data.append({
                "solution": sol,
                "predicted_answer": pred,
                "is_correct": correct
            })

        results.append({
            "index": i,
            "question": question,
            "ground_truth": gt,
            "solutions": solution_data
        })

        # Progress logging every 100 questions
        if (i + 1) % 100 == 0:
            correct_count = sum(1 for r in results if any(s["is_correct"] for s in r["solutions"]))
            log(f"Progress: {i+1}/{len(questions)}, Pass@{k}: {correct_count}/{i+1} = {correct_count/(i+1)*100:.1f}%")

    return results

def extract_k_subset(results, k):
    """Extract k-solution subset from results with more solutions."""
    subset = []
    for item in results:
        new_item = {
            "index": item["index"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "solutions": item["solutions"][:k]  # Take first k solutions
        }
        subset.append(new_item)
    return subset

def compute_statistics(results, k_values=[1, 3, 5, 7]):
    """Compute Pass@k statistics for results."""
    stats = {}
    for k in k_values:
        if k > len(results[0]["solutions"]):
            continue

        pass_count = 0
        for item in results:
            # Check if any of first k solutions is correct
            if any(item["solutions"][i]["is_correct"] for i in range(min(k, len(item["solutions"])))):
                pass_count += 1

        stats[f"pass@{k}"] = pass_count / len(results)

    return stats

def save_results(results, filepath):
    """Save results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Saved results to {filepath}")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    log("=" * 60)
    log("Verifier Training Data Generation (with Batching)")
    log("=" * 60)
    log(f"Batch size: {BATCH_SIZE}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    log("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    log("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Set padding token (required for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    log(f"Model loaded on device: {model.device}")

    # Load dataset
    log("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    train_full = dataset["train"]

    # Split data
    train_questions = train_full.select(range(TRAIN_START, TRAIN_END))
    eval_questions = train_full.select(range(EVAL_START, EVAL_END))

    log(f"Training questions: {len(train_questions)}")
    log(f"Evaluation questions: {len(eval_questions)}")

    # ========================================================================
    # Generate Training Data (k=7, then extract k=1,3,5)
    # ========================================================================
    log("\n" + "=" * 60)
    log("Generating TRAINING data (k=7)...")
    log("=" * 60)

    train_results_k7 = process_questions(model, tokenizer, train_questions, "train", k=7)

    # Save k=7 results
    save_results(train_results_k7, os.path.join(OUTPUT_DIR, "train_k7.json"))

    # Extract and save subsets
    for k in [1, 3, 5]:
        subset = extract_k_subset(train_results_k7, k)
        save_results(subset, os.path.join(OUTPUT_DIR, f"train_k{k}.json"))

    # Compute and log training statistics
    train_stats = compute_statistics(train_results_k7)
    log(f"Training statistics: {train_stats}")

    # ========================================================================
    # Generate Evaluation Data (k=7, then extract k=1,3,5)
    # ========================================================================
    log("\n" + "=" * 60)
    log("Generating EVALUATION data (k=7)...")
    log("=" * 60)

    eval_results_k7 = process_questions(model, tokenizer, eval_questions, "eval", k=7)

    # Save k=7 results
    save_results(eval_results_k7, os.path.join(OUTPUT_DIR, "eval_k7.json"))

    # Extract and save subsets
    for k in [1, 3, 5]:
        subset = extract_k_subset(eval_results_k7, k)
        save_results(subset, os.path.join(OUTPUT_DIR, f"eval_k{k}.json"))

    # Compute and log evaluation statistics
    eval_stats = compute_statistics(eval_results_k7)
    log(f"Evaluation statistics: {eval_stats}")

    # ========================================================================
    # Final Summary
    # ========================================================================
    log("\n" + "=" * 60)
    log("GENERATION COMPLETE!")
    log("=" * 60)
    log(f"Output directory: {OUTPUT_DIR}")
    log("Generated files:")
    for split in ["train", "eval"]:
        for k in [1, 3, 5, 7]:
            filepath = os.path.join(OUTPUT_DIR, f"{split}_k{k}.json")
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                log(f"  - {split}_k{k}.json ({size:.1f} KB)")

    log("\nTraining Pass@k:")
    for k, v in train_stats.items():
        log(f"  {k}: {v*100:.2f}%")

    log("\nEvaluation Pass@k:")
    for k, v in eval_stats.items():
        log(f"  {k}: {v*100:.2f}%")

    log("\nDone!")

if __name__ == "__main__":
    main()
