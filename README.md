# README

## Project Overview

This project implements a modular Generator–Verifier (G+V) framework for GSM8K-style mathematical reasoning. The core goal is to (1) generate multiple reasoning samples per problem, (2) evaluate generator stability via consensus analysis, and (3) determine whether a separately trained Verifier can improve final answer reliability. The workflow is fully modular: each stage—data generation, model training, test sampling, consensus grouping, and final evaluation—can be executed independently.

The system consists of:
1. A LoRA-fine-tuned Phi-2 Generator
2. A RoBERTa-large Verifier trained for outcome-level correctness classification
3. Scripts for data preparation, 15-sample test generation, consensus-grouping, and final evaluation across high/medium/low consensus categories

---

## File Descriptions

### generator_train.py  
Trains the Generator model on the GSM8K training set using LoRA.  
The resulting checkpoint is saved in the `generator_checkpoint/` directory and is used by all inference scripts.

### generator_eval.py  
Evaluates the trained Generator.  
Provides greedy-decoding accuracy and optional sampling-based metrics.  
This script serves as a baseline before integrating the Verifier.

### generate_verifier_data.py  
Generates all datasets used to train the Verifier.  
For each question, the trained Generator produces multiple solutions (k = 1, 3, 5, 7).  
Each solution is parsed into:
- a reasoning chain  
- a predicted answer  
- a weak-supervision correctness label (`is_correct`)  

All datasets are stored in the `verifier_data/` directory.  
The files `train_k5.json` and `eval_k5.json` are used for final Verifier training.

### verifier_train_RoBERTa.py  
(Previously named `train_verifier_RoBERTa.py`)  
Trains a RoBERTa-large model as an outcome-level Verifier.  
The model performs binary classification, predicting whether a generated solution is correct.  
The fully trained checkpoint and tokenizer are saved in the `verifier_RoBERTa/` directory.

### generate_test_solutions.py  
Legacy script for generating test-set solutions (non-batched, smaller k).  
Kept for compatibility but no longer used in the final pipeline.

### generate_test_k15.py  
**New, primary script for test-time data generation.**  
Generates 15 solutions per GSM8K test problem (three rounds × five samples).  
The output file `test_k15.json` is the main dataset used for:
- consensus analysis  
- all Generator-only baselines  
- all Generator–Verifier evaluations  

### classify_by_margin.py  
Classifies each test-set problem into a consensus group using the 15 generated solutions.  
The margin is defined from the solution distribution (e.g., top-frequency minus second-highest frequency).  
Outputs:

- `group_high.json`  
- `group_medium.json`  
- `group_low.json`  
- `classify_metadata.json` (stores thresholds and statistics)

### evaluate_groups.py  
(Previously named `evaluate_gv_iterative_v2.py`)  
Runs the complete set of evaluation experiments using the grouped data.  
This includes:

- generator-only majority vote  
- pass@k statistics  
- Verifier-guided filtering  
- Verifier-weighted aggregation  
- comparisons across high/medium/low consensus groups  

Loads:
- the grouped files (`group_*.json`)  
- the Verifier (`verifier_RoBERTa/`)  
- the 15-solution dataset (`test_k15.json`)  

Outputs final metrics to:

- `eval_results.json`

### generator_checkpoint/  
Directory containing the LoRA-fine-tuned Generator model produced by `generator_train.py`.  
Used by all sampling and data-generation scripts.

### verifier_data/  
Contains all automatically generated datasets for Verifier training.  
Produced by `generate_verifier_data.py`.  
Includes multiple k-versions (k = 1, 3, 5, 7) for both train and eval splits.

### verifier_RoBERTa/  
Directory containing the fully trained RoBERTa Verifier.  
Produced by `verifier_train_RoBERTa.py`.

### test_k15.json  
The main test-set sampling file:  
1319 GSM8K problems × 15 solutions = 19,785 total samples.  
Required by all consensus and G+V evaluations.

### group_high.json / group_medium.json / group_low.json  
Consensus-based partitions of the test set, produced by `classify_by_margin.py`.

### classify_metadata.json  
Stores consensus thresholds and preprocessing statistics.

---

## Execution Workflow

The recommended pipeline is:

1. **Train the Generator**  
   ```
   python generator_train.py
   ```

2. **(Optional) Evaluate the Generator**  
   ```
   python generator_eval.py
   ```

3. **Generate Verifier training data**  
   ```
   python generate_verifier_data.py
   ```

4. **Train the Verifier**  
   ```
   python verifier_train_RoBERTa.py
   ```

5. **Generate 15 solutions per test-set problem**  
   ```
   python generate_test_k15.py
   ```

6. **Classify test problems into consensus groups**  
   ```
   python classify_by_margin.py
   ```

7. **Run all Generator–Verifier evaluations**  
   ```
   python evaluate_groups.py
   ```

These steps produce:
- the trained Generator checkpoint  
- the trained Verifier checkpoint  
- the 15-sample test dataset  
- consensus partitions  
- the final `eval_results.json` summarizing all performance metrics  

---

## Summary

This repository provides a complete, modular pipeline for studying Generator stability and Verifier effectiveness in long-form mathematical reasoning tasks. It isolates each component—generation, verification, consensus modeling, and evaluation—making it easy to analyze failure cases and the limits of binary-classification verification on reasoning chains.

