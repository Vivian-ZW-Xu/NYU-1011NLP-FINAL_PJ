# README

## Project Overview

This project implements a modular Generator–Verifier–Ranker framework for GSM8K-style mathematical reasoning. The goal is to (1) generate multiple reasoning candidates, (2) study generator stability through consensus analysis, and (3) evaluate whether Verifiers or Rankers can improve final answer reliability.

The full system contains:

1. A LoRA-fine-tuned Phi-2 Generator  
2. A RoBERTa-large Verifier (binary classifier)  
3. A RoBERTa-large Ranker (scalar scoring model)  
4. Scripts for:
   - Verifier-data generation  
   - Ranker training  
   - 15-sample test-set generation  
   - Consensus-based grouping  
   - Evaluation of Generator-only baselines  
   - Evaluation of Verifier, Ranker, and combined multi-stage pipelines  

All components are fully modular. Each model can be trained, replaced, or evaluated independently.

---

## File Descriptions

### **Generator-related**

#### `generator_train.py`
Trains the Generator (Phi-2 + LoRA) on GSM8K.  
Outputs a checkpoint directory:  
`generator_checkpoint/`

#### `generator_eval.py`
Evaluates the fine-tuned Generator (greedy decoding or sampling).  
Used as a baseline before introducing Verifier or Ranker.

#### `generate_verifier_data.py`
Creates supervised datasets for Verifier training.  
For each training question, the Generator produces k = 1, 3, 5, 7 solutions.  
Each solution includes:
- reasoning text  
- predicted answer  
- weak correctness label  

Outputs saved in:
`verifier_data/`

#### `generate_test_k15.py`
**Main test-time generation script.**  
Produces:
`test_k15.json`  
containing:
- 1319 GSM8K test questions  
- 15 sampled Generator solutions per question  

This file is required for:  
• consensus analysis  
• all GV and Ranker-based evaluation methods  

---

### **Verifier-related**

#### `verifier_train_RoBERTa.py`
Trains a RoBERTa-large binary classifier for outcome-level correctness.  
Saves to:  
`verifier_RoBERTa/`

#### `evaluate_groups.py`
Legacy GV evaluation script using only the Classifier.  
Runs:
- Majority Vote baseline  
- Classifier Top-Score selection  
- Classifier Weighted Majority Vote  

Outputs:
`eval_results.json`

---

### **Ranker-related (New)**

#### `verifier_train_ranker_RoBERTa.py`
Trains a RoBERTa-large Ranker with a margin-ranking loss.  
Outputs a scalar score for each solution.  
The trained model is saved in:  
`ranker_roberta/`

#### `evaluate_ranker.py` (old)
Early experiment version (kept for reference).

#### `evaluate_ranker.py` (new main script)
**Main evaluation script comparing 7 methods**, including Ranker and Classifier combinations:

1. Majority Vote (Generator only)  
2. G + Classifier (Top Score)  
3. G + Classifier (Weighted MV)  
4. G + Ranker (Top Score)  
5. G + Ranker (Weighted MV)  
6. G + Classifier → Ranker (Top-1)  
7. G + Classifier → Ranker (Weighted MV)

Evaluates separately on:
- group_high.json  
- group_medium.json  
- group_low.json  

Outputs final results to:
`ranker_eval_results.json`

---

### **Consensus Analysis**

#### `classify_by_margin.py`
Computes answer-frequency statistics from 15 Generator samples:

- r1 = most common answer frequency  
- r2 = second most common  
- r3 = third most common  
- margin = r1 - r2  

Uses two learned thresholds (tau_low, tau_high) to partition the 1319 test questions into:

- High-consensus (stable generator)  
- Medium-consensus  
- Low-consensus (unstable, multi-modal generator behavior)  

Outputs:
- `group_high.json`  
- `group_medium.json`  
- `group_low.json`  
- `classify_metadata.json` (stores thresholds and distribution curves)

---

### **Data files**

#### `test_k15.json`
Main test dataset:  
1319 questions × 15 Generator samples each  
Used by all Verifier–Ranker evaluations.

#### `group_high.json`, `group_medium.json`, `group_low.json`
Consensus-based partitions of the test problems.

#### `classify_metadata.json`
Stores tau_low, tau_high, and distribution summaries.

#### `ranker_eval_results.json`
Final evaluation metrics for all 7 methods.

---

### **Support files**

#### `requirements.txt`
Python package requirements.

#### `README.md`
This file.

---

## Execution Workflow

### **1. Train the Generator**

### **2. (Optional) Evaluate the Generator**

### **3. Generate Verifier training data**

### **4. Train the Verifier**

### **5. Train the Ranker (New)**

### **6. Generate 15 test-set samples per problem**

### **7. Classify test-set questions into consensus groups**

### **8. Run all evaluation methods**

Outputs:
`ranker_eval_results.json`

---

## Baseline Results

### **Pass@k Summary**
From Generator-only sampling (Phi-2 + LoRA):

| k | Training (2000 Q) | Evaluation (500 Q) | Test (1319 Q) |
|---|------------------|--------------------|----------------|
| 1 | 46.85% | 51.60% | 45.56% |
| 3 | 67.50% | 69.20% | – |
| 5 | 73.70% | 76.20% | – |
| 7 | 77.45% | 79.60% | – |
| 10 | – | – | 84.91% |
| 15 | – | – | 88.55% |

---

## Final Evaluation (High / Medium / Low Consensus)

Results from 7 methods evaluated across the three consensus groups:

| Method | Overall | High | Medium | Low |
|-------|---------|-------|---------|------|
| (i) Majority Vote | **68.61%** | **96.04%** | **72.04%** | 32.13% |
| (ii) G+Classifier Top | 41.17% | 65.15% | 35.77% | 17.27% |
| (iii) G+Classifier Weighted | 64.90% | 95.05% | 64.99% | 28.30% |
| (iv) G+Ranker Top | 36.54% | 59.41% | 30.73% | 14.39% |
| (v) G+Ranker Weighted | **68.16%** | 95.84% | 70.53% | **32.37%** |
| (vi) G+C+R Top | 38.06% | 60.79% | 33.00% | 15.35% |
| (vii) G+C+R Weighted | 61.64% | 93.07% | 60.96% | 24.22% |

---

## Summary

This repository provides a full, modular pipeline for studying:

- Generator consistency (via 15-sample analysis)  
- Verifier classification performance  
- Ranker fine-grained scoring ability  
- Combined multi-stage pipelines (Classifier → Ranker)  
- Behavior across high / medium / low consensus subsets  

Key findings:
- Majority Vote remains a very strong baseline.  
- Binary-classification Verifier is not reliable as a selector.  
- Ranker-weighted voting nearly matches MV and sometimes improves medium-consensus cases.  
- Multi-stage pipelines (Classifier → Ranker) do not improve performance and often degrade it.

All modules are replaceable, making the system suitable for controlled experiments, ablation studies, and further exploration into small-model reasoning reliability.
