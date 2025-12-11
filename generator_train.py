#!/usr/bin/env python3
"""
Generator Training Script (Phi-2 + QLoRA)
==========================================
Train Phi-2 with LoRA on GSM8K for math reasoning.

Output: ./generator_ovm (LoRA adapter + tokenizer)
"""

import torch
import random
import numpy as np
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
MODEL_NAME = "microsoft/phi-2"
OUTPUT_DIR = "./generator_ovm"
CHECKPOINT_DIR = "./generator_checkpoints"

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

NUM_EPOCHS = 2
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 32  # Effective batch size = 32
LEARNING_RATE = 1e-4
MAX_LENGTH = 1024

# ============================================================================
# Set random seeds
# ============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================================
# Data Formatting
# ============================================================================
def format_gsm8k_example(example):

    question = example["question"].strip()
    answer = example["answer"].strip()

    if "####" not in answer:
        return {"prompt": None, "response": None}

    ground_truth_raw = answer.split("####")[-1].strip()

    m = re.search(r"-?\d+(\.\d+)?", ground_truth_raw)
    if not m:
        return {"prompt": None, "response": None}
    ground_truth = m.group(0)

    reasoning = answer.split("####")[0].strip()

    reasoning = re.sub(r'<<[^>]+>>', '', reasoning)
    reasoning = reasoning.strip()

    prompt = (
        f"Question: {question}\n\n"
        f"Let's solve this step by step.\n"
    )

    response = (
        f"{reasoning}\n\n"
        f"Final answer: {ground_truth}"
    )

    return {
        "prompt": prompt,
        "response": response,
        "question": question,
        "answer": ground_truth
    }


def validate_example(example):
    """Filter out invalid entries."""
    return example["prompt"] is not None and example["response"] is not None


# ============================================================================
# Main Training
# ============================================================================
def main():
    print("=" * 60)
    print("Generator Training (Phi-2 + QLoRA)")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available!")

    # ========================================================================
    # 1. Load and prepare data
    # ========================================================================
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")

    train_val = dataset["train"].train_test_split(test_size=0.1, seed=SEED)
    train_set = train_val["train"]
    val_set = train_val["test"]

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(dataset['test'])}")

    print("\nFormatting training set...")
    train_formatted = train_set.map(format_gsm8k_example)
    train_formatted = train_formatted.filter(validate_example)

    print("Formatting validation set...")
    val_formatted = val_set.map(format_gsm8k_example)
    val_formatted = val_formatted.filter(validate_example)

    print(f"Training examples: {len(train_formatted)}")
    print(f"Validation examples: {len(val_formatted)}")

    # ========================================================================
    # 2. Load model with QLoRA
    # ========================================================================
    print(f"\nLoading {MODEL_NAME} with QLoRA...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ========================================================================
    # 3. Tokenize data
    # ========================================================================
    def format_and_tokenize(batch):

        inputs = []
        attn = []
        labels = []

        for p, r in zip(batch["prompt"], batch["response"]):
            full = p + r
            tok_full = tokenizer(full, truncation=True, max_length=MAX_LENGTH, padding=False)
            tok_prompt = tokenizer(p, truncation=True, max_length=MAX_LENGTH, padding=False)

            lab = tok_full["input_ids"][:]
            prompt_len = len(tok_prompt["input_ids"])
            lab[:prompt_len] = [-100] * prompt_len

            inputs.append(tok_full["input_ids"])
            attn.append(tok_full["attention_mask"])
            labels.append(lab)

        return {"input_ids": inputs, "attention_mask": attn, "labels": labels}

    print("\nTokenizing datasets...")
    tokenized_train = train_formatted.map(
        format_and_tokenize,
        batched=True,
        remove_columns=train_formatted.column_names,
        desc="Tokenizing training set"
    )
    tokenized_val = val_formatted.map(
        format_and_tokenize,
        batched=True,
        remove_columns=val_formatted.column_names,
        desc="Tokenizing validation set"
    )

    # ========================================================================
    # 4. Training
    # ========================================================================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,

        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Training examples: {len(tokenized_train)}")
    print(f"Validation examples: {len(tokenized_val)}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Total epochs: {NUM_EPOCHS}")
    print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print("=" * 60)

    trainer.train()

    print("\nTraining complete!")

    # ========================================================================
    # 5. Save model
    # ========================================================================
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
