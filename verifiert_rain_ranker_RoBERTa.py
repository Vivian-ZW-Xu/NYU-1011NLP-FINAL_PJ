#!/usr/bin/env python3
"""
DeBERTa Ranker Training Script (Pairwise Margin Ranking)
=========================================================
Train DeBERTa-v3-base to RANK math solutions, not just classify.

Key differences from classifier:
1. Uses Margin Ranking Loss instead of Cross-Entropy
2. Trains on (correct, wrong) pairs
3. Outputs a single score per solution (not class probabilities)
4. Optimized for ranking, not classification

This should perform better on Low consensus group where
we need to distinguish between multiple plausible wrong answers.
"""

import torch
import torch.nn as nn
import json
import random
import os
from datetime import datetime
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
SEED = 42

# Model - RoBERTa-large is stable and strong for this task
# (DeBERTa has tokenizer bugs in recent transformers versions)
BASE_MODEL = "roberta-large"  # ~355M params, stable encoder

DATA_DIR = "./verifier_data"
OUTPUT_DIR = "./ranker_roberta"
MAX_LENGTH = 512

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 2  # pairs per batch (reduced for 12GB GPU)
GRADIENT_ACCUMULATION = 16  # effective batch = 32 pairs
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
TRAIN_VAL_SPLIT = 0.1
WEIGHT_DECAY = 0.01
MARGIN = 0.5  # margin for ranking loss (0.5 is more stable than 1.0)

# Use K=7 data for maximum diversity
TRAIN_DATA = "train_k7.json"
EVAL_DATA = "eval_k7.json"

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


# ============================================================================
# Ranker Model
# ============================================================================
class RoBERTaRanker(nn.Module):
    """
    RoBERTa-based ranker that outputs a single score per solution.

    Architecture:
    - RoBERTa encoder
    - Mean pooling over sequence
    - Linear projection to scalar score
    """
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Score head: hidden -> 1
        self.score_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Mean pooling (excluding padding)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_hidden / sum_mask  # (batch, hidden)

        # Score
        score = self.score_head(pooled).squeeze(-1)  # (batch,)
        return score


# ============================================================================
# Dataset
# ============================================================================
class PairwiseRankingDataset(Dataset):
    """
    Dataset that yields (positive_text, negative_text) pairs.

    For each question, we create pairs of (correct_solution, wrong_solution).
    This is the key to training a ranker vs a classifier.
    """
    def __init__(self, pairs, tokenizer, max_length):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_text, neg_text = self.pairs[idx]

        pos_enc = self.tokenizer(
            pos_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        neg_enc = self.tokenizer(
            neg_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'pos_input_ids': pos_enc['input_ids'].squeeze(0),
            'pos_attention_mask': pos_enc['attention_mask'].squeeze(0),
            'neg_input_ids': neg_enc['input_ids'].squeeze(0),
            'neg_attention_mask': neg_enc['attention_mask'].squeeze(0),
        }


class SingleSolutionDataset(Dataset):
    """Dataset for evaluation - single solutions with labels."""
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': ex['label'],
            'question_idx': ex['question_idx'],
        }


# ============================================================================
# Data Loading
# ============================================================================
def load_data_with_split(filepath, val_ratio=0.1):
    """
    Load data and split by QUESTIONS (not pairs) to avoid leakage.

    Returns:
        train_pairs: list of (pos_text, neg_text) for training
        val_items: list of question items for validation (SingleSolution format)

    This ensures validation questions are completely held out.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Shuffle questions first
    random.shuffle(data)

    # Split by questions
    val_size = int(len(data) * val_ratio)
    val_questions = data[:val_size]
    train_questions = data[val_size:]

    # Create training pairs from train_questions only
    question_pairs = []
    for item in train_questions:
        question = item["question"]

        correct_solutions = []
        wrong_solutions = []

        for sol_data in item["solutions"]:
            solution = sol_data["solution"]
            text = f"Question: {question} [SEP] Solution: {solution}"

            if sol_data["is_correct"]:
                correct_solutions.append(text)
            else:
                wrong_solutions.append(text)

        # Create all (correct, wrong) pairs for this question
        q_pairs = []
        for correct_text in correct_solutions:
            for wrong_text in wrong_solutions:
                q_pairs.append((correct_text, wrong_text))

        if q_pairs:
            random.shuffle(q_pairs)
            question_pairs.append(q_pairs)

    # Shuffle order of questions, flatten, shuffle again
    random.shuffle(question_pairs)
    train_pairs = [p for q_pairs in question_pairs for p in q_pairs]
    random.shuffle(train_pairs)

    # Create validation examples (single solutions with labels)
    val_examples = []
    for item in val_questions:
        question = item["question"]
        for sol_data in item["solutions"]:
            solution = sol_data["solution"]
            label = 1 if sol_data["is_correct"] else 0
            text = f"Question: {question} [SEP] Solution: {solution}"
            val_examples.append({
                "text": text,
                "label": label,
                "question_idx": item["index"]
            })

    return train_pairs, val_examples, len(train_questions), len(val_questions)


def load_single_examples(filepath):
    """Load data as single examples for evaluation."""
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


# ============================================================================
# Evaluation
# ============================================================================
def compute_pairwise_accuracy(model, dataloader, device):
    """
    Compute pairwise accuracy: for each question, check if correct solutions
    are scored higher than incorrect solutions.
    """
    model.eval()

    # Collect all scores grouped by question
    from collections import defaultdict
    question_scores = defaultdict(lambda: {"pos": [], "neg": []})

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            question_idxs = batch['question_idx']

            scores = model(input_ids, attention_mask).cpu().tolist()

            for score, label, q_idx in zip(scores, labels, question_idxs):
                q_idx = q_idx.item() if hasattr(q_idx, 'item') else q_idx
                if label == 1:
                    question_scores[q_idx]["pos"].append(score)
                else:
                    question_scores[q_idx]["neg"].append(score)

    # Compute pairwise accuracy
    correct_pairs = 0
    total_pairs = 0

    for q_idx, scores in question_scores.items():
        pos_scores = scores["pos"]
        neg_scores = scores["neg"]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue

        for ps in pos_scores:
            for ns in neg_scores:
                total_pairs += 1
                if ps > ns:
                    correct_pairs += 1

    pairwise_acc = correct_pairs / total_pairs if total_pairs > 0 else 0
    return pairwise_acc, total_pairs


def compute_top1_accuracy(model, dataloader, device):
    """
    Compute top-1 accuracy: for each question, check if the highest-scored
    solution is correct.
    """
    model.eval()

    from collections import defaultdict
    question_solutions = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Top-1 Eval", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            question_idxs = batch['question_idx']

            scores = model(input_ids, attention_mask).cpu().tolist()

            for score, label, q_idx in zip(scores, labels, question_idxs):
                q_idx = q_idx.item() if hasattr(q_idx, 'item') else q_idx
                label = label.item() if hasattr(label, 'item') else label
                question_solutions[q_idx].append((score, label))

    correct = 0
    total = 0

    for q_idx, solutions in question_solutions.items():
        if not solutions:
            continue
        # Find solution with highest score
        _, best_label = max(solutions, key=lambda x: x[0])
        total += 1
        if best_label == 1:
            correct += 1

    return correct / total if total > 0 else 0, total


# ============================================================================
# Training
# ============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, margin, scaler):
    """Train for one epoch with Margin Ranking Loss and mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Margin Ranking Loss: wants pos_score > neg_score by at least margin
    # y=1 means first input should be ranked higher
    ranking_loss = nn.MarginRankingLoss(margin=margin)

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Get positive and negative samples
        pos_input_ids = batch['pos_input_ids'].to(device)
        pos_attention_mask = batch['pos_attention_mask'].to(device)
        neg_input_ids = batch['neg_input_ids'].to(device)
        neg_attention_mask = batch['neg_attention_mask'].to(device)

        # Forward pass with mixed precision
        with autocast('cuda'):
            pos_scores = model(pos_input_ids, pos_attention_mask)
            neg_scores = model(neg_input_ids, neg_attention_mask)

            # Compute loss: we want pos_scores > neg_scores
            target = torch.ones(pos_scores.size(0), device=device)
            loss = ranking_loss(pos_scores, neg_scores, target)
            loss = loss / GRADIENT_ACCUMULATION

        # Backward pass with scaler
        scaler.scale(loss).backward()

        total_loss += loss.item() * GRADIENT_ACCUMULATION
        num_batches += 1

        if num_batches % GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix({'loss': total_loss / num_batches})

    return total_loss / num_batches


# ============================================================================
# Main
# ============================================================================
def main():
    log("=" * 60)
    log("DeBERTa Ranker Training (Pairwise Margin Ranking)")
    log("=" * 60)

    # Check GPU
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        log("WARNING: Training on CPU will be slow!")

    # ========================================================================
    # 1. Load Data (with proper train/val/test split)
    # ========================================================================
    log("\n" + "=" * 60)
    log("Loading and preparing data...")
    log("=" * 60)

    train_path = os.path.join(DATA_DIR, TRAIN_DATA)
    test_path = os.path.join(DATA_DIR, EVAL_DATA)

    log(f"Training data: {train_path}")
    log(f"Test data (final eval only): {test_path}")

    # Split train_k7 into train + validation BY QUESTIONS (not pairs)
    # This avoids data leakage - validation questions are completely held out
    train_pairs, val_examples, n_train_q, n_val_q = load_data_with_split(
        train_path, val_ratio=TRAIN_VAL_SPLIT
    )

    log(f"Train questions: {n_train_q}, Validation questions: {n_val_q}")
    log(f"Training pairs: {len(train_pairs):,}")
    log(f"Validation examples: {len(val_examples)}")
    if len(train_pairs) > 50000:
        log(f"  NOTE: Large dataset - training may take a while")

    # Load test data (eval_k7.json) - ONLY used for final evaluation
    test_examples = load_single_examples(test_path)
    log(f"Test examples (held out): {len(test_examples)}")

    # ========================================================================
    # 2. Load Tokenizer & Model
    # ========================================================================
    log("\n" + "=" * 60)
    log(f"Loading {BASE_MODEL}...")
    log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = RoBERTaRanker(BASE_MODEL)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total params: {total_params:,}")
    log(f"Trainable params: {trainable_params:,}")

    # ========================================================================
    # 3. Create Datasets
    # ========================================================================
    log("\nCreating datasets...")

    train_dataset = PairwiseRankingDataset(train_pairs, tokenizer, MAX_LENGTH)
    val_dataset = SingleSolutionDataset(val_examples, tokenizer, MAX_LENGTH)
    test_dataset = SingleSolutionDataset(test_examples, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 4,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 4,
        num_workers=4,
        pin_memory=True
    )

    # ========================================================================
    # 4. Setup Optimizer & Scheduler
    # ========================================================================
    log("\nSetting up optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    log(f"Total training steps: {total_steps}")
    log(f"Warmup steps: {warmup_steps}")

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
    log(f"Margin: {MARGIN}")
    log(f"Loss: Margin Ranking Loss")
    log(f"Mixed Precision: FP16 enabled")
    log("=" * 60 + "\n")

    # Create gradient scaler for mixed precision training
    scaler = GradScaler('cuda')

    best_pairwise_acc = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        log(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, MARGIN, scaler)
        log(f"Training loss: {train_loss:.4f}")

        # Evaluate on VALIDATION set (not test!)
        val_pairwise_acc, num_pairs = compute_pairwise_accuracy(model, val_loader, device)
        log(f"Val pairwise accuracy: {val_pairwise_acc:.4f} ({num_pairs} pairs)")

        val_top1_acc, num_questions = compute_top1_accuracy(model, val_loader, device)
        log(f"Val top-1 accuracy: {val_top1_acc:.4f} ({num_questions} questions)")

        # Save checkpoint for this epoch
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_pairwise_acc': val_pairwise_acc,
            'val_top1_acc': val_top1_acc,
            'train_loss': train_loss,
        }, checkpoint_path)
        log(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model based on validation performance
        if val_pairwise_acc > best_pairwise_acc:
            best_pairwise_acc = val_pairwise_acc
            best_epoch = epoch + 1

            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            tokenizer.save_pretrained(OUTPUT_DIR)
            log(f"  -> New best model saved! (val_pairwise_acc={val_pairwise_acc:.4f})")

    # ========================================================================
    # 6. Final Evaluation on TEST set (eval_k7.json - never seen during training)
    # ========================================================================
    log("\n" + "=" * 60)
    log("TRAINING COMPLETE!")
    log("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt")))

    log("\nFinal evaluation on held-out TEST set (eval_k7.json):")
    test_pairwise_acc, num_pairs = compute_pairwise_accuracy(model, test_loader, device)
    test_top1_acc, num_questions = compute_top1_accuracy(model, test_loader, device)

    log(f"  Test pairwise accuracy: {test_pairwise_acc:.4f} ({num_pairs} pairs)")
    log(f"  Test top-1 accuracy: {test_top1_acc:.4f} ({num_questions} questions)")

    # ========================================================================
    # 7. Save Final Model
    # ========================================================================
    log("\nSaving final model...")

    # Save full model for easy loading
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'base_model': BASE_MODEL,
            'max_length': MAX_LENGTH,
        }
    }, os.path.join(OUTPUT_DIR, "ranker_model.pt"))

    # Save training info
    training_info = {
        "base_model": BASE_MODEL,
        "model_type": "RoBERTa Ranker (Pairwise Margin)",
        "train_questions": n_train_q,
        "val_questions": n_val_q,
        "training_pairs": len(train_pairs),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "epochs": NUM_EPOCHS,
        "best_epoch": best_epoch,
        "batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
        "margin": MARGIN,
        "max_length": MAX_LENGTH,
        "train_data": TRAIN_DATA,
        "test_data": EVAL_DATA,
        "best_val_pairwise_accuracy": best_pairwise_acc,
        "final_test_pairwise_accuracy": test_pairwise_acc,
        "final_test_top1_accuracy": test_top1_acc,
    }

    with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)

    log(f"\nModel saved to {OUTPUT_DIR}")

    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    log(f"Model: {BASE_MODEL}")
    log(f"Loss: Margin Ranking Loss (margin={MARGIN})")
    log(f"Training data: {TRAIN_DATA} ({len(train_pairs):,} pairs from {n_train_q} questions)")
    log(f"Validation: {n_val_q} questions (held out from train)")
    log(f"Test: {EVAL_DATA} (completely held out)")
    log(f"Best epoch: {best_epoch} (based on val pairwise acc)")
    log(f"Best Val Pairwise Acc: {best_pairwise_acc:.4f}")
    log(f"Test Pairwise Acc: {test_pairwise_acc:.4f}")
    log(f"Test Top-1 Acc: {test_top1_acc:.4f}")
    log("=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
