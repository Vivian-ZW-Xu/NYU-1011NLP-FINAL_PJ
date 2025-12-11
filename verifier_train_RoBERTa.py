#!/usr/bin/env python3
"""
RoBERTa Verifier Training Script (OVM - Outcome-level Verification)
====================================================================
Train RoBERTa-large to verify if a math solution is correct.

Why RoBERTa (Encoder-only) over decoder LLMs:
1. Encoder-only = bidirectional attention = better for classification
2. ~355M params = full finetune possible, no LoRA needed
3. Much faster training and inference
4. More stable with noisy CoT solutions
"""

import torch
import torch.nn as nn
import json
import random
import os
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
SEED = 42

# Model choices (uncomment one):
# BASE_MODEL = "microsoft/deberta-v3-base"      # ~180M params (tokenizer bug in transformers 4.57)
# BASE_MODEL = "microsoft/deberta-v3-large"    # ~430M params
# BASE_MODEL = "roberta-base"                   # ~125M params, fast
BASE_MODEL = "roberta-large"                    # ~355M params, strong encoder

DATA_DIR = "./verifier_data"
OUTPUT_DIR = "./verifier_deberta"
MAX_LENGTH = 512  # RoBERTa handles this well

LEARNING_RATE = 2e-5  # Lower LR for full finetune (no LoRA)
BATCH_SIZE = 8        # Can be larger since model is smaller
GRADIENT_ACCUMULATION = 4  # Effective batch size = 32
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
TRAIN_VAL_SPLIT = 0.1
WEIGHT_DECAY = 0.01

# ============================================================================
# Set random seeds
# ============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Helper Functions
# ============================================================================
def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def load_data(filepath):

    with open(filepath, 'r') as f:
        data = json.load(f)

    examples = []
    for item in data:
        question = item["question"]
        for sol_data in item["solutions"]:
            solution = sol_data["solution"]
            label = 1 if sol_data["is_correct"] else 0

            text = f"Question: {question} [SEP] Solution: {solution}"
            examples.append({
                "text": text,
                "label": label,
                "question_idx": item["index"]
            })

    return examples

def balance_dataset(examples):

    pos_examples = [ex for ex in examples if ex["label"] == 1]
    neg_examples = [ex for ex in examples if ex["label"] == 0]

    min_count = min(len(pos_examples), len(neg_examples))

    random.shuffle(pos_examples)
    random.shuffle(neg_examples)

    balanced = pos_examples[:min_count] + neg_examples[:min_count]
    random.shuffle(balanced)

    return balanced

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

def compute_pairwise_accuracy(model, tokenizer, eval_data, device, batch_size=32):

    model.eval()

    from collections import defaultdict
    question_groups = defaultdict(lambda: {"pos": [], "neg": []})

    for ex in eval_data:
        key = ex["question_idx"]
        if ex["label"] == 1:
            question_groups[key]["pos"].append(ex["text"])
        else:
            question_groups[key]["neg"].append(ex["text"])

    def get_scores_batched(texts):
        if len(texts) == 0:
            return []

        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                all_scores.extend(probs.cpu().tolist())

        return all_scores

    correct_pairs = 0
    total_pairs = 0

    for q_idx, group in question_groups.items():
        pos_texts = group["pos"]
        neg_texts = group["neg"]

        if len(pos_texts) == 0 or len(neg_texts) == 0:
            continue

        pos_scores = get_scores_batched(pos_texts)
        neg_scores = get_scores_batched(neg_texts)

        for ps in pos_scores:
            for ns in neg_scores:
                total_pairs += 1
                if ps > ns:
                    correct_pairs += 1

    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0
    return pairwise_acc, total_pairs

# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 60)
    log("RoBERTa Verifier Training (OVM)")
    log("=" * 60)

    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ========================================================================
    # 1. Load Data
    # ========================================================================
    log("\nLoading training data...")

    train_examples = load_data(os.path.join(DATA_DIR, "train_k5.json"))

    orig_pos = sum(1 for ex in train_examples if ex["label"] == 1)
    orig_neg = len(train_examples) - orig_pos
    log(f"Original: {len(train_examples)} examples (pos: {orig_pos}, neg: {orig_neg})")

    train_examples = balance_dataset(train_examples)
    bal_pos = sum(1 for ex in train_examples if ex["label"] == 1)
    bal_neg = len(train_examples) - bal_pos
    log(f"After balancing: {len(train_examples)} examples (pos: {bal_pos}, neg: {bal_neg})")

    random.shuffle(train_examples)
    val_size = int(len(train_examples) * TRAIN_VAL_SPLIT)
    val_examples = train_examples[:val_size]
    train_examples = train_examples[val_size:]

    log(f"Training: {len(train_examples)}, Validation: {len(val_examples)}")

    eval_examples_final = load_data(os.path.join(DATA_DIR, "eval_k5.json"))
    log(f"Final evaluation set: {len(eval_examples_final)} examples")

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    # ========================================================================
    # 2. Load Tokenizer & Model
    # ========================================================================
    log(f"\nLoading {BASE_MODEL}...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        problem_type="single_label_classification",
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total params: {total_params:,}")
    log(f"Trainable params: {trainable_params:,} (100.00%)")

    # ========================================================================
    # 3. Tokenize
    # ========================================================================
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    log("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function, batched=True,
        remove_columns=["text", "question_idx"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True,
        remove_columns=["text", "question_idx"]
    )

    # ========================================================================
    # 4. Training Configuration
    # ========================================================================
    log("\nSetting up training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # Can be larger for eval
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",

        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,

        fp16=True,
        dataloader_num_workers=4,

        logging_steps=25,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ========================================================================
    # 5. Train
    # ========================================================================
    log("\n" + "=" * 60)
    log("STARTING TRAINING")
    log("=" * 60)
    log(f"Model: {BASE_MODEL}")
    log(f"Epochs: {NUM_EPOCHS}")
    log(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    log(f"Learning rate: {LEARNING_RATE}")
    log(f"MAX_LENGTH: {MAX_LENGTH}")
    log(f"Full finetune (no LoRA)")
    log("=" * 60 + "\n")

    trainer.train()

    log("\n" + "=" * 60)
    log("TRAINING COMPLETE!")
    log("=" * 60)

    # ========================================================================
    # 6. Final Evaluation
    # ========================================================================
    log("\nRunning final evaluation...")

    val_results = trainer.evaluate()
    log("\nValidation Results:")
    for key, value in val_results.items():
        if isinstance(value, float):
            log(f"  {key}: {value:.4f}")

    log("\nComputing Pairwise Accuracy on eval set...")
    pairwise_acc, total_pairs = compute_pairwise_accuracy(
        model, tokenizer, eval_examples_final, device
    )
    log(f"Pairwise Accuracy: {pairwise_acc:.4f} ({total_pairs} pairs)")

    # ========================================================================
    # 7. Save Model
    # ========================================================================
    log("\nSaving model...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    training_info = {
        "base_model": BASE_MODEL,
        "model_type": "encoder-only (RoBERTa)",
        "training_examples": len(train_examples),
        "val_examples": len(val_examples),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "full_finetune": True,
        "validation_results": {k: float(v) if isinstance(v, float) else v
                              for k, v in val_results.items()},
        "pairwise_accuracy": pairwise_acc,
        "pairwise_total_pairs": total_pairs,
    }

    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    log(f"\nModel saved to {OUTPUT_DIR}")

    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    log(f"Model: {BASE_MODEL}")
    log(f"Validation Accuracy: {val_results.get('eval_accuracy', 0):.4f}")
    log(f"Validation AUC: {val_results.get('eval_auc', 0):.4f}")
    log(f"Pairwise Accuracy: {pairwise_acc:.4f}")
    log("=" * 60)
    log("Done!")

if __name__ == "__main__":
    main()
