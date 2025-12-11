#!/usr/bin/env python3
"""
Classify questions by margin (r1 - r2)
======================================

import json
import numpy as np
from datetime import datetime
from collections import Counter

# ============================================================================
# Configuration
# ============================================================================
INPUT_PATH = "./test_k15.json"
OUTPUT_DIR = "./"

# ============================================================================
# Helper Functions
# ============================================================================
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def calculate_margin(solutions):
    """Calculate r1, r2, and margin (r1 - r2)"""
    answers = [s["predicted_answer"] for s in solutions if s["predicted_answer"] is not None]
    if not answers:
        return 0, 0, 0
    counter = Counter(answers)
    most_common = counter.most_common()
    r1 = most_common[0][1] / 15
    r2 = most_common[1][1] / 15 if len(most_common) > 1 else 0
    margin = r1 - r2
    return r1, r2, margin

# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 70)
    log("Classification by Margin (r1 - r2)")
    log("=" * 70)
    log(f"Input: {INPUT_PATH}")

    log("\nLoading data...")
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    log(f"Loaded {len(data)} questions")

    log("\nCalculating margins...")
    margins = []
    for item in data:
        r1, r2, margin = calculate_margin(item["solutions"])
        item["r1"] = r1
        item["r2"] = r2
        item["margin"] = margin
        margins.append(margin)

    margins_arr = np.array(margins)
    tau_low = float(np.percentile(margins_arr, 33))
    tau_high = float(np.percentile(margins_arr, 66))

    log(f"\nMargin distribution:")
    log(f"  Min: {margins_arr.min():.3f}, Max: {margins_arr.max():.3f}")
    log(f"  Mean: {margins_arr.mean():.3f}, Median: {np.median(margins_arr):.3f}")
    log(f"\nGrouping thresholds (33rd/66th percentiles):")
    log(f"  τ_low (33rd): {tau_low:.3f}")
    log(f"  τ_high (66th): {tau_high:.3f}")
    log(f"  Low: margin < {tau_low:.3f}")
    log(f"  Medium: {tau_low:.3f} <= margin < {tau_high:.3f}")
    log(f"  High: margin >= {tau_high:.3f}")

    groups = {"high": [], "medium": [], "low": []}
    for item in data:
        margin = item["margin"]
        if margin >= tau_high:
            item["consensus_group"] = "high"
            groups["high"].append(item)
        elif margin >= tau_low:
            item["consensus_group"] = "medium"
            groups["medium"].append(item)
        else:
            item["consensus_group"] = "low"
            groups["low"].append(item)

    log("\nGroup distribution:")
    for name, items in groups.items():
        avg_margin = np.mean([item["margin"] for item in items]) if items else 0
        log(f"  {name.capitalize()}: {len(items)} questions ({100*len(items)/len(data):.1f}%), avg margin: {avg_margin:.3f}")

    log("\nSaving group files...")


    metadata = {
        "method": "margin (r1-r2) with percentile thresholds",
        "input": INPUT_PATH,
        "total_questions": len(data),
        "tau_low": tau_low,
        "tau_high": tau_high,
        "margin_stats": {
            "min": float(margins_arr.min()),
            "max": float(margins_arr.max()),
            "mean": float(margins_arr.mean()),
            "median": float(np.median(margins_arr)),
        },
        "group_sizes": {name: len(items) for name, items in groups.items()},
    }

    with open(f"{OUTPUT_DIR}classify_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"  Saved: classify_metadata.json")

    for name, items in groups.items():
        output_path = f"{OUTPUT_DIR}group_{name}.json"
        with open(output_path, "w") as f:
            json.dump(items, f, indent=2)
        log(f"  Saved: group_{name}.json ({len(items)} questions)")

    log("\nDone!")


if __name__ == "__main__":
    main()
