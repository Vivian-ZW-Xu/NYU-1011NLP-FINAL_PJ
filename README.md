# README

## Project Overview

This project implements a Generator–Verifier framework for GSM8K-style mathematical reasoning. The main objective is to analyze how a fine-tuned Generator behaves under multi-sample inference and to determine whether a separately trained Verifier can improve answer reliability. The project also evaluates how different verification strategies perform under varying levels of generator stability (high-consensus, polarized, and low-consensus problems).

The pipeline includes:
1. A fine-tuned Generator model (Phi-2 with LoRA)
2. A RoBERTa-based Verifier for outcome-level correctness classification
3. Scripts for generating training data, producing 15-sample test outputs, and running full Generator–Verifier evaluation

The repository is structured so that each stage of the workflow is modular and can be executed independently.

## File Descriptions

generator_train.py:
Trains the Generator model using the GSM8K training set with LoRA fine-tuning. The resulting checkpoint is saved in the generator_checkpoint/ directory and is used by all subsequent inference scripts.

generator_eval.py:
Evaluates the trained Generator model. It computes greedy decoding accuracy and can optionally evaluate sampling-based decoding. This serves as a baseline before integrating the Verifier.

generate_verifier_data.py:
Generates the training and evaluation datasets for the Verifier. It uses the trained Generator to produce multiple solutions per question (k = 1, 3, 5, 7), extracts predicted answers, labels correctness using exact match, and saves all datasets in the verifier_data/ directory. The files train_k5.json and eval_k5.json are used to train the final Verifier.

train_verifier_RoBERTa.py:
Trains a RoBERTa-based Verifier model on the datasets in verifier_data/. The model performs binary classification on reasoning chains, predicting whether each solution is correct. The trained Verifier checkpoint and configuration are stored in verifier_RoBERTa/.

generate_test_solutions.py:
Generates 15 solutions for every problem in the GSM8K test set using the trained Generator. Each solution includes a reasoning paragraph and a predicted answer. All results are saved into test_k15.json. This file is the foundation for consensus analysis and all Generator–Verifier evaluation experiments.

evaluate_gv_iterative_v2.py:
Runs the final evaluation comparing three methods: generator-only majority vote, generator + verifier with top-score selection, and generator + verifier with weighted majority vote. It loads the 15-solution dataset from test_k15.json, applies each evaluation strategy, and outputs accuracy results across different consensus groups.

generator_checkpoint/:
The directory containing the trained Generator model produced by generator_train.py. All inference and data-generation scripts depend on this checkpoint.

verifier_data/:
A directory containing automatically generated datasets for Verifier training. These files are produced by generate_verifier_data.py and include versions for several values of k.

verifier_RoBERTa/:
The directory containing the trained RoBERTa Verifier produced by train_verifier_RoBERTa.py.

## Execution Workflow

The recommended order for running the entire project is:

1. Train the Generator:
   python generator_train.py

2. (Optional) Evaluate the Generator baseline:
   python generator_eval.py

3. Generate Verifier training data:
   python generate_verifier_data.py

4. Train the Verifier:
   python train_verifier_RoBERTa.py

5. Generate 15 solutions per test-set question:
   python generate_test_solutions.py

6. Run the full Generator–Verifier evaluation:
   python evaluate_gv_iterative_v2.py

The outputs of these scripts include the generator checkpoint, the verifier checkpoint, multi-sample test solutions, and the final accuracy results for all evaluation methods.

This completes the documentation for the repository.
