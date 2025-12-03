#!/usr/bin/env python3
"""
Evaluate G+V methods on classified groups
=========================================

读取分组文件 (group_high.json, group_medium.json, group_low.json)
运行评估方法并输出结果

评估方法:
(i)   Generator-only Majority Vote (incremental batches)
(ii)  G+V Top-Score Selection
(iii) G+V Weighted Majority Vote
"""

import torch
import json
import os
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from collections import Counter
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
GROUP_DIR = "./"  # 分组文件目录
GROUP_FILES = {
    "high": "group_high.json",
    "medium": "group_medium.json",
    "low": "group_low.json",
}
METADATA_FILE = "classify_metadata.json"

VERIFIER_PATH = "./verifier_RoBERTa"
OUTPUT_PATH = "./eval_results.json"

# Verifier threshold
CLASSIFIER_THRESHOLD = 0.5

# Batch structure
BATCH_SIZE = 5  # solutions per batch
NUM_BATCHES = 3  # total 15 solutions = 3 batches x 5

# ============================================================================
# Helper Functions
# ============================================================================
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

# ============================================================================
# Verifier
# ============================================================================
class Verifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if self.model is not None:
            return
        log("Loading Verifier...")
        self.tokenizer = AutoTokenizer.from_pretrained(VERIFIER_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(VERIFIER_PATH)
        self.model = self.model.to(self.device)
        self.model.eval()
        log(f"Verifier loaded on {self.device}!")

    def score_solutions(self, question, solutions):
        """Score a batch of solutions"""
        texts = [f"Question: {question} [SEP] Solution: {s['solution']}" for s in solutions]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().tolist()

        return scores

# ============================================================================
# Evaluation Methods
# ============================================================================

def method_majority_vote_incremental(solutions, ground_truth):
    """
    Method (i): Generator-only Majority Vote (incremental batches)
    Returns: predicted_answer, is_correct, batches_used
    """
    for num_batches in range(1, NUM_BATCHES + 1):
        batch_solutions = solutions[:num_batches * BATCH_SIZE]
        answers = [s["predicted_answer"] for s in batch_solutions if s["predicted_answer"] is not None]

        if not answers:
            continue

        counter = Counter(answers)
        most_common = counter.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1] or num_batches == NUM_BATCHES:
            predicted = most_common[0][0]
            is_correct = (predicted == ground_truth)
            return predicted, is_correct, num_batches

    return None, False, NUM_BATCHES


def method_gv_top_score(solutions, ground_truth):
    """
    Method (ii): G+V with Top-Score Selection
    Returns: predicted_answer, is_correct, batches_used
    """
    all_candidates = []

    for batch_idx in range(NUM_BATCHES):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_solutions = solutions[start:end]

        candidates = [
            s for s in batch_solutions
            if s.get("verifier_score", 0) > CLASSIFIER_THRESHOLD
            and s.get("predicted_answer") is not None
        ]
        all_candidates.extend(candidates)

        if all_candidates:
            break

    if not all_candidates:
        valid_solutions = [s for s in solutions if s.get("predicted_answer") is not None]
        if valid_solutions:
            best = max(valid_solutions, key=lambda x: x.get("verifier_score", 0))
            return best.get("predicted_answer"), best.get("predicted_answer") == ground_truth, NUM_BATCHES
        return None, False, NUM_BATCHES

    best = max(all_candidates, key=lambda x: x.get("verifier_score", 0))
    batches_used = (solutions.index(best) // BATCH_SIZE) + 1
    return best.get("predicted_answer"), best.get("predicted_answer") == ground_truth, batches_used


def method_gv_weighted_mv(solutions, ground_truth):
    """
    Method (iii): G+V with Weighted Majority Vote
    Returns: predicted_answer, is_correct, batches_used
    """
    all_candidates = []
    last_batch_with_candidate = 0

    for batch_idx in range(NUM_BATCHES):
        start = batch_idx * BATCH_SIZE
        end = start + BATCH_SIZE
        batch_solutions = solutions[start:end]

        candidates = [
            s for s in batch_solutions
            if s.get("verifier_score", 0) > CLASSIFIER_THRESHOLD
            and s.get("predicted_answer") is not None
        ]

        if candidates:
            all_candidates.extend(candidates)
            last_batch_with_candidate = batch_idx + 1

    if not all_candidates:
        valid_solutions = [s for s in solutions if s.get("predicted_answer") is not None]
        if valid_solutions:
            best = max(valid_solutions, key=lambda x: x.get("verifier_score", 0))
            return best.get("predicted_answer"), best.get("predicted_answer") == ground_truth, NUM_BATCHES
        return None, False, NUM_BATCHES

    answer_weights = {}
    for s in all_candidates:
        ans = s["predicted_answer"]
        score = s.get("verifier_score", 0)
        answer_weights[ans] = answer_weights.get(ans, 0) + score

    best_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return best_answer, best_answer == ground_truth, last_batch_with_candidate if last_batch_with_candidate > 0 else NUM_BATCHES


# ============================================================================
# Main Evaluation
# ============================================================================
def main():
    log("=" * 70)
    log("G+V Evaluation on Classified Groups")
    log("=" * 70)
    log(f"Verifier: {VERIFIER_PATH}")
    log(f"Classifier threshold: {CLASSIFIER_THRESHOLD}")

    # Load metadata
    metadata_path = f"{GROUP_DIR}{METADATA_FILE}"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        log(f"\nClassification method: {metadata.get('method', 'unknown')}")
        log(f"Thresholds: τ_low={metadata.get('tau_low', 'N/A')}, τ_high={metadata.get('tau_high', 'N/A')}")
    else:
        metadata = {}
        log("\nWarning: No metadata file found")

    # Load all groups
    log("\nLoading group files...")
    all_data = []
    groups = {}

    for group_name, filename in GROUP_FILES.items():
        filepath = f"{GROUP_DIR}{filename}"
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                items = json.load(f)
            groups[group_name] = items
            all_data.extend(items)
            log(f"  Loaded {group_name}: {len(items)} questions")
        else:
            log(f"  Warning: {filename} not found")
            groups[group_name] = []

    total = len(all_data)
    log(f"Total: {total} questions")

    # Load verifier
    verifier = Verifier()
    verifier.load()

    # Score all solutions (once)
    log("\n" + "=" * 70)
    log("Scoring all solutions with verifier...")
    log("=" * 70)
    for item in tqdm(all_data, desc="Scoring"):
        question = item["question"]
        solutions = item["solutions"]
        scores = verifier.score_solutions(question, solutions)
        for sol, score in zip(solutions, scores):
            sol["verifier_score"] = score

    # Run evaluation methods
    log("\nRunning evaluation methods...")
    results = {
        "high": {"mv": [], "top_score": [], "weighted_mv": []},
        "medium": {"mv": [], "top_score": [], "weighted_mv": []},
        "low": {"mv": [], "top_score": [], "weighted_mv": []},
    }

    for group_name, items in groups.items():
        for item in tqdm(items, desc=f"Evaluating {group_name}"):
            ground_truth = item["ground_truth"]
            solutions = item["solutions"]

            # Method (i): Majority Vote
            pred_mv, correct_mv, batches_mv = method_majority_vote_incremental(solutions, ground_truth)
            results[group_name]["mv"].append({
                "index": item["index"],
                "predicted": pred_mv,
                "is_correct": correct_mv,
                "batches_used": batches_mv
            })

            # Method (ii): G+V Top Score
            pred_ts, correct_ts, batches_ts = method_gv_top_score(solutions, ground_truth)
            results[group_name]["top_score"].append({
                "index": item["index"],
                "predicted": pred_ts,
                "is_correct": correct_ts,
                "batches_used": batches_ts
            })

            # Method (iii): G+V Weighted MV
            pred_wmv, correct_wmv, batches_wmv = method_gv_weighted_mv(solutions, ground_truth)
            results[group_name]["weighted_mv"].append({
                "index": item["index"],
                "predicted": pred_wmv,
                "is_correct": correct_wmv,
                "batches_used": batches_wmv
            })

    # Compute and print metrics
    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)

    # Overall metrics
    all_results = {"mv": [], "top_score": [], "weighted_mv": []}
    for group_name in ["high", "medium", "low"]:
        for method in ["mv", "top_score", "weighted_mv"]:
            all_results[method].extend(results[group_name][method])

    method_names = {
        "mv": "Majority Vote (G-only)",
        "top_score": "G+V Top Score",
        "weighted_mv": "G+V Weighted MV"
    }

    # Header
    log(f"\n{'Method':<25} | {'Overall':>10} | {'High':>10} | {'Medium':>10} | {'Low':>10}")
    log("-" * 75)

    for method_key, method_name in method_names.items():
        overall_acc = 100 * sum(1 for r in all_results[method_key] if r["is_correct"]) / len(all_results[method_key]) if all_results[method_key] else 0

        accs = []
        for group_name in ["high", "medium", "low"]:
            group_results = results[group_name][method_key]
            if group_results:
                acc = 100 * sum(1 for r in group_results if r["is_correct"]) / len(group_results)
            else:
                acc = 0
            accs.append(acc)

        log(f"{method_name:<25} | {overall_acc:>9.2f}% | {accs[0]:>9.2f}% | {accs[1]:>9.2f}% | {accs[2]:>9.2f}%")

    # Pass@k
    log("\n" + "-" * 75)
    log("Oracle Upper Bounds (Pass@k):")
    for k in [5, 15]:
        overall_pass = sum(1 for item in all_data if any(s["is_correct"] for s in item["solutions"][:k]))
        log(f"  Pass@{k}: {100*overall_pass/total:.2f}%")

    # Average batches used
    log("\n" + "-" * 75)
    log("Average Batches Used:")
    for method_key, method_name in method_names.items():
        avg_batches = np.mean([r["batches_used"] for r in all_results[method_key]])
        log(f"  {method_name}: {avg_batches:.2f}")

    # Detailed group analysis
    log("\n" + "=" * 70)
    log("Detailed Group Analysis")
    log("=" * 70)

    for group_name in ["high", "medium", "low"]:
        group_items = groups[group_name]
        if not group_items:
            continue

        log(f"\n[{group_name.capitalize()} Consensus Group] - {len(group_items)} questions")

        # Pass@k
        for k in [5, 15]:
            pass_k = sum(1 for item in group_items if any(s["is_correct"] for s in item["solutions"][:k]))
            log(f"  Pass@{k}: {100*pass_k/len(group_items):.2f}%")

        # Method comparison
        for method_key, method_name in method_names.items():
            group_results = results[group_name][method_key]
            acc = 100 * sum(1 for r in group_results if r["is_correct"]) / len(group_results)
            avg_batches = np.mean([r["batches_used"] for r in group_results])
            log(f"  {method_name}: {acc:.2f}% (avg {avg_batches:.2f} batches)")

    # Save results
    log(f"\nSaving results to {OUTPUT_PATH}...")
    output_data = {
        "config": {
            "verifier": VERIFIER_PATH,
            "classifier_threshold": CLASSIFIER_THRESHOLD,
            "classification_metadata": metadata,
        },
        "group_sizes": {name: len(items) for name, items in groups.items()},
        "results": results,
        "all_results": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    log("\nDone!")


if __name__ == "__main__":
    main()
