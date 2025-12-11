#!/usr/bin/env python3
"""
Evaluate Ranker vs Classifier on test_k15.json
==============================================

Compare:
- (i)    Generator-only Majority Vote
- (ii)   G+Classifier Top-Score Selection
- (iii)  G+Classifier Weighted Majority Vote
- (iv)   G+Ranker Top-Score Selection
- (v)    G+Ranker Weighted Majority Vote
- (vi)   G+Classifier+Ranker Top (Classifier filter → Ranker select top-1)
- (vii)  G+Classifier+Ranker Weighted MV (Classifier filter → Ranker weighted vote)
"""

import torch
import torch.nn as nn
import json
import os
from datetime import datetime
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import Counter
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
GROUP_HIGH = "./group_high.json"
GROUP_MEDIUM = "./group_medium.json"
GROUP_LOW = "./group_low.json"

CLASSIFIER_PATH = "./verifier_RoBERTa"
RANKER_PATH = "./ranker_roberta"

OUTPUT_PATH = "./ranker_eval_results.json"

# Classifier threshold for G+V methods
CLASSIFIER_THRESHOLD = 0.5

# Batch config
BATCH_SIZE = 5  # solutions per batch
NUM_BATCHES = 3  # total 15 solutions = 3 batches x 5

# ============================================================================
# Helper Functions
# ============================================================================
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

# ============================================================================
# Ranker Model Definition
# ============================================================================
class RoBERTaRanker(nn.Module):
    """
    RoBERTa-based ranker that outputs a single score per solution.

    Architecture (must match training script exactly):
    - RoBERTa encoder (via AutoModel)
    - Mean pooling over sequence
    - Linear projection to scalar score
    """
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / sum_mask  # (batch, hidden)

        score = self.score_head(pooled).squeeze(-1)  # (batch,)
        return score

# ============================================================================
# Classifier (Verifier)
# ============================================================================
class Classifier:
    def __init__(self, path):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path

    def load(self):
        if self.model is not None:
            return
        log(f"Loading Classifier from {self.path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
        self.model = self.model.to(self.device)
        self.model.eval()
        log(f"Classifier loaded on {self.device}!")

    def score_solutions(self, question, solutions):
        """Score solutions - returns probability of being correct."""
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
# Ranker
# ============================================================================
class Ranker:
    def __init__(self, path):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path

    def load(self):
        if self.model is not None:
            return
        log(f"Loading Ranker from {self.path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

        self.model = RoBERTaRanker("roberta-large")

        checkpoint = torch.load(os.path.join(self.path, "ranker_model.pt"), map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()
        log(f"Ranker loaded on {self.device}!")

    def score_solutions(self, question, solutions):
        """Score solutions - returns ranking scores (higher = better)."""
        texts = [f"Question: {question} [SEP] Solution: {s['solution']}" for s in solutions]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            scores = self.model(inputs["input_ids"], inputs["attention_mask"])
            scores = scores.cpu().tolist()

        return scores

# ============================================================================
# Evaluation Methods
# ============================================================================

def method_majority_vote(solutions, ground_truth):
    """
    Method (i): Generator-only Majority Vote
    Returns: predicted_answer, is_correct
    """
    answers = [s["predicted_answer"] for s in solutions if s["predicted_answer"] is not None]

    if not answers:
        return None, False

    counter = Counter(answers)
    predicted = counter.most_common(1)[0][0]
    is_correct = (predicted == ground_truth)
    return predicted, is_correct


def method_top_score(solutions, ground_truth, threshold=0.5):
    """
    Method (ii)/(iv): Top-Score Selection
    Returns: predicted_answer, is_correct
    """
    candidates = [
        s for s in solutions
        if s.get("score", 0) > threshold
        and s.get("predicted_answer") is not None
    ]

    if not candidates:
        valid_solutions = [s for s in solutions if s.get("predicted_answer") is not None]
        if valid_solutions:
            best = max(valid_solutions, key=lambda x: x.get("score", 0))
            return best.get("predicted_answer"), best.get("predicted_answer") == ground_truth
        return None, False

    best = max(candidates, key=lambda x: x.get("score", 0))
    return best.get("predicted_answer"), best.get("predicted_answer") == ground_truth


def method_weighted_mv(solutions, ground_truth, threshold=0.5):
    """
    Method (iii)/(v): Weighted Majority Vote
    Returns: predicted_answer, is_correct
    """
    candidates = [
        s for s in solutions
        if s.get("score", 0) > threshold
        and s.get("predicted_answer") is not None
    ]

    if not candidates:
        candidates = [s for s in solutions if s.get("predicted_answer") is not None]

    if not candidates:
        return None, False

    answer_weights = {}
    for s in candidates:
        ans = s["predicted_answer"]
        score = max(s.get("score", 0), 0.01)
        answer_weights[ans] = answer_weights.get(ans, 0) + score

    best_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return best_answer, best_answer == ground_truth


def method_cls_rnk_top(solutions, ground_truth, cls_threshold=0.5):
    """
    Method (vi): G + Classifier filter → Ranker Top Score

    Steps:
    1. Classifier filters candidates (score > threshold)
    2. Ranker picks the top-1 from surviving candidates

    Returns: predicted_answer, is_correct
    """
    candidates = [
        s for s in solutions
        if s.get("classifier_score", 0) > cls_threshold
        and s.get("predicted_answer") is not None
    ]

    if not candidates:
        candidates = [s for s in solutions if s.get("predicted_answer") is not None]

    if not candidates:
        return None, False

    best = max(candidates, key=lambda x: x.get("ranker_score", 0))
    return best.get("predicted_answer"), best.get("predicted_answer") == ground_truth


def method_cls_rnk_wmv(solutions, ground_truth, cls_threshold=0.5):
    """
    Method (vii): G + Classifier filter → Ranker Weighted Majority Vote

    Steps:
    1. Classifier filters candidates (score > threshold)
    2. Ranker scores candidates
    3. Weighted majority vote using ranker scores as weights

    Returns: predicted_answer, is_correct
    """
    candidates = [
        s for s in solutions
        if s.get("classifier_score", 0) > cls_threshold
        and s.get("predicted_answer") is not None
    ]

    if not candidates:
        candidates = [s for s in solutions if s.get("predicted_answer") is not None]

    if not candidates:
        return None, False

    min_rnk = min(s.get("ranker_score", 0) for s in candidates)

    answer_weights = {}
    for s in candidates:
        ans = s["predicted_answer"]
        # Shift scores to be positive (add offset if min is negative)
        score = s.get("ranker_score", 0) - min_rnk + 0.01  # +0.01 to avoid zero
        answer_weights[ans] = answer_weights.get(ans, 0) + score

    best_answer = max(answer_weights.keys(), key=lambda x: answer_weights[x])
    return best_answer, best_answer == ground_truth


# ============================================================================
# Main Evaluation
# ============================================================================
def main():
    log("=" * 70)
    log("Ranker vs Classifier Evaluation")
    log("=" * 70)

    log("\nLoading test data from group files...")
    groups = {"high": [], "medium": [], "low": []}

    with open(GROUP_HIGH, "r") as f:
        groups["high"] = json.load(f)
    with open(GROUP_MEDIUM, "r") as f:
        groups["medium"] = json.load(f)
    with open(GROUP_LOW, "r") as f:
        groups["low"] = json.load(f)

    test_data = groups["high"] + groups["medium"] + groups["low"]

    log(f"Loaded {len(test_data)} questions total:")
    for g, items in groups.items():
        log(f"  {g.capitalize()}: {len(items)} questions")

    classifier = Classifier(CLASSIFIER_PATH)
    classifier.load()

    ranker = Ranker(RANKER_PATH)
    ranker.load()

    log("\n" + "=" * 70)
    log("Scoring all solutions...")
    log("=" * 70)

    for item in tqdm(test_data, desc="Scoring"):
        question = item["question"]
        solutions = item["solutions"]

        classifier_scores = classifier.score_solutions(question, solutions)
        for sol, score in zip(solutions, classifier_scores):
            sol["classifier_score"] = score

        ranker_scores = ranker.score_solutions(question, solutions)
        for sol, score in zip(solutions, ranker_scores):
            sol["ranker_score"] = score

    log("\nEvaluating methods...")

    results = {
        "high": {"mv": [], "cls_top": [], "cls_wmv": [], "rnk_top": [], "rnk_wmv": [], "cls_rnk_top": [], "cls_rnk_wmv": []},
        "medium": {"mv": [], "cls_top": [], "cls_wmv": [], "rnk_top": [], "rnk_wmv": [], "cls_rnk_top": [], "cls_rnk_wmv": []},
        "low": {"mv": [], "cls_top": [], "cls_wmv": [], "rnk_top": [], "rnk_wmv": [], "cls_rnk_top": [], "cls_rnk_wmv": []},
    }

    for group_name, items in groups.items():
        for item in tqdm(items, desc=f"Evaluating {group_name}"):
            ground_truth = item["ground_truth"]
            solutions = item["solutions"]

            pred_mv, correct_mv = method_majority_vote(solutions, ground_truth)
            results[group_name]["mv"].append(correct_mv)

            for s in solutions:
                s["score"] = s["classifier_score"]

            pred_cls_top, correct_cls_top = method_top_score(solutions, ground_truth, CLASSIFIER_THRESHOLD)
            results[group_name]["cls_top"].append(correct_cls_top)

            pred_cls_wmv, correct_cls_wmv = method_weighted_mv(solutions, ground_truth, CLASSIFIER_THRESHOLD)
            results[group_name]["cls_wmv"].append(correct_cls_wmv)

            best_by_ranker = max(solutions, key=lambda x: x["ranker_score"])
            correct_rnk_top = best_by_ranker.get("predicted_answer") == ground_truth
            results[group_name]["rnk_top"].append(correct_rnk_top)

            min_rnk = min(s["ranker_score"] for s in solutions)
            answer_weights_rnk = {}
            for s in solutions:
                if s.get("predicted_answer") is not None:
                    ans = s["predicted_answer"]
                    weight = s["ranker_score"] - min_rnk + 0.01
                    answer_weights_rnk[ans] = answer_weights_rnk.get(ans, 0) + weight
            if answer_weights_rnk:
                best_rnk_wmv = max(answer_weights_rnk.keys(), key=lambda x: answer_weights_rnk[x])
                correct_rnk_wmv = (best_rnk_wmv == ground_truth)
            else:
                correct_rnk_wmv = False
            results[group_name]["rnk_wmv"].append(correct_rnk_wmv)

            pred_cls_rnk_top, correct_cls_rnk_top = method_cls_rnk_top(solutions, ground_truth, CLASSIFIER_THRESHOLD)
            results[group_name]["cls_rnk_top"].append(correct_cls_rnk_top)

            pred_cls_rnk_wmv, correct_cls_rnk_wmv = method_cls_rnk_wmv(solutions, ground_truth, CLASSIFIER_THRESHOLD)
            results[group_name]["cls_rnk_wmv"].append(correct_cls_rnk_wmv)

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)

    method_names = {
        "mv": "Majority Vote (G-only)",
        "cls_top": "G+Classifier Top",
        "cls_wmv": "G+Classifier Weighted",
        "rnk_top": "G+Ranker Top",
        "rnk_wmv": "G+Ranker Weighted",
        "cls_rnk_top": "G+C+R Top",
        "cls_rnk_wmv": "G+C+R Weighted MV"
    }

    all_results = {method: [] for method in method_names.keys()}
    for group_name in ["high", "medium", "low"]:
        for method in method_names.keys():
            all_results[method].extend(results[group_name][method])

    log(f"\n{'Method':<25} | {'Overall':>10} | {'High':>10} | {'Medium':>10} | {'Low':>10}")
    log("-" * 75)

    for method_key, method_name in method_names.items():
        overall_acc = 100 * sum(all_results[method_key]) / len(all_results[method_key]) if all_results[method_key] else 0

        accs = []
        for group_name in ["high", "medium", "low"]:
            group_results = results[group_name][method_key]
            if group_results:
                acc = 100 * sum(group_results) / len(group_results)
            else:
                acc = 0
            accs.append(acc)

        log(f"{method_name:<25} | {overall_acc:>9.2f}% | {accs[0]:>9.2f}% | {accs[1]:>9.2f}% | {accs[2]:>9.2f}%")

    # Pass@k
    log("\n" + "-" * 75)
    log("Oracle Upper Bounds (Pass@k):")
    total = len(test_data)
    for k in [5, 15]:
        overall_pass = sum(1 for item in test_data if any(s["is_correct"] for s in item["solutions"][:k]))
        log(f"  Pass@{k}: {100*overall_pass/total:.2f}%")

    log("\n" + "=" * 70)
    log("IMPROVEMENT ANALYSIS")
    log("=" * 70)

    log("\n[Classifier vs Ranker (Top Score)]")
    for group_name in ["high", "medium", "low"]:
        cls_acc = 100 * sum(results[group_name]["cls_top"]) / len(results[group_name]["cls_top"]) if results[group_name]["cls_top"] else 0
        rnk_acc = 100 * sum(results[group_name]["rnk_top"]) / len(results[group_name]["rnk_top"]) if results[group_name]["rnk_top"] else 0
        diff = rnk_acc - cls_acc
        log(f"  {group_name.capitalize():>8}: Classifier={cls_acc:.2f}%, Ranker={rnk_acc:.2f}%, Diff={diff:+.2f}%")

    overall_cls = 100 * sum(all_results["cls_top"]) / len(all_results["cls_top"])
    overall_rnk = 100 * sum(all_results["rnk_top"]) / len(all_results["rnk_top"])
    log(f"  {'Overall':>8}: Classifier={overall_cls:.2f}%, Ranker={overall_rnk:.2f}%, Diff={overall_rnk - overall_cls:+.2f}%")

    log("\n[G+C+R (Classifier filter + Ranker) vs Classifier alone]")
    for group_name in ["high", "medium", "low"]:
        cls_acc = 100 * sum(results[group_name]["cls_top"]) / len(results[group_name]["cls_top"]) if results[group_name]["cls_top"] else 0
        cr_acc = 100 * sum(results[group_name]["cls_rnk_top"]) / len(results[group_name]["cls_rnk_top"]) if results[group_name]["cls_rnk_top"] else 0
        diff = cr_acc - cls_acc
        log(f"  {group_name.capitalize():>8}: Cls={cls_acc:.2f}%, C+R={cr_acc:.2f}%, Diff={diff:+.2f}%")

    overall_cls = 100 * sum(all_results["cls_top"]) / len(all_results["cls_top"])
    overall_cr = 100 * sum(all_results["cls_rnk_top"]) / len(all_results["cls_rnk_top"])
    log(f"  {'Overall':>8}: Cls={overall_cls:.2f}%, C+R={overall_cr:.2f}%, Diff={overall_cr - overall_cls:+.2f}%")

    log(f"\nSaving results to {OUTPUT_PATH}...")
    output_data = {
        "config": {
            "classifier_path": CLASSIFIER_PATH,
            "ranker_path": RANKER_PATH,
            "classifier_threshold": CLASSIFIER_THRESHOLD,
            "test_data": {"high": GROUP_HIGH, "medium": GROUP_MEDIUM, "low": GROUP_LOW},
        },
        "group_sizes": {name: len(items) for name, items in groups.items()},
        "accuracy": {
            method_key: {
                "overall": sum(all_results[method_key]) / len(all_results[method_key]) if all_results[method_key] else 0,
                "high": sum(results["high"][method_key]) / len(results["high"][method_key]) if results["high"][method_key] else 0,
                "medium": sum(results["medium"][method_key]) / len(results["medium"][method_key]) if results["medium"][method_key] else 0,
                "low": sum(results["low"][method_key]) / len(results["low"][method_key]) if results["low"][method_key] else 0,
            }
            for method_key in method_names.keys()
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)

    log("\nDone!")


if __name__ == "__main__":
    main()
