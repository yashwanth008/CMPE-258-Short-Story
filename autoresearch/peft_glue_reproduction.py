"""
PEFT A2Z - Autoresearch Reproduction Script
============================================
Reproduces GLUE benchmark results from Section 6.1 of:
  Prottasha et al. (2025), "PEFT A2Z: Parameter-Efficient Fine-Tuning Survey"
  arXiv:2504.14117v1, Table 1

Compares LoRA, BitFit, and Prefix Tuning on RoBERTa-Base
across SST-2 (sentiment) and RTE (textual entailment) tasks.

Based on autoresearch template:
  https://github.com/dlmastery/autoresearch/tree/master/generalized_ml_autoresearch

Usage:
    python peft_glue_reproduction.py --method lora --task sst2
    python peft_glue_reproduction.py --method bitfit --task sst2
    python peft_glue_reproduction.py --method prefix --task sst2
    python peft_glue_reproduction.py --method lora --task rte
    python peft_glue_reproduction.py --method bitfit --task rte
    python peft_glue_reproduction.py --method prefix --task rte
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
)
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ─────────────────────────────────────────────
# Paper reference values (Table 1, RoBERTa-Base)
# ─────────────────────────────────────────────
PAPER_RESULTS = {
    "lora":   {"sst2": 93.31, "rte": 74.92},
    "bitfit": {"sst2": 93.12, "rte": 77.98},
    "prefix": {"sst2": 93.81, "rte": 54.51},
    "full":   {"sst2": 92.89, "rte": 72.43},
}

PAPER_PARAMS = {
    "lora":   0.89,   # millions
    "bitfit": 0.083,
    "prefix": 0.96,
    "full":   124.6,
}

# ─────────────────────────────────────────────
# Task configuration
# ─────────────────────────────────────────────
TASK_CONFIG = {
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "text_col": "sentence",
        "label_col": "label",
        "num_labels": 2,
        "metric": "accuracy",
        "max_length": 128,
    },
    "rte": {
        "dataset": "glue",
        "subset": "rte",
        "text_col": ["sentence1", "sentence2"],
        "label_col": "label",
        "num_labels": 2,
        "metric": "accuracy",
        "max_length": 256,
    },
}


def count_trainable_params(model):
    """Count trainable parameters in millions."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable / 1e6, total / 1e6


def apply_bitfit(model):
    """
    BitFit: freeze everything except bias terms.
    From: Zaken et al. (2021), arXiv:2106.10199
    """
    for name, param in model.named_parameters():
        if "bias" not in name:
            param.requires_grad = False
    return model


def apply_lora(model, task_type=TaskType.SEQ_CLS):
    """
    LoRA: Low-Rank Adaptation.
    Config matches paper: rank=8, alpha=16 (standard settings for RoBERTa GLUE tasks).
    From: Hu et al. (2021), arXiv:2106.09685
    """
    config = LoraConfig(
        task_type=task_type,
        r=8,                        # rank — controls parameter budget
        lora_alpha=16,              # scaling factor
        lora_dropout=0.1,
        target_modules=["query", "value"],  # attention matrices
        bias="none",
    )
    model = get_peft_model(model, config)
    return model


def apply_prefix_tuning(model, task_type=TaskType.SEQ_CLS, num_virtual_tokens=20):
    """
    Prefix Tuning: prepend learnable virtual tokens.
    From: Li & Liang (2021), arXiv:2101.00190
    """
    config = PrefixTuningConfig(
        task_type=task_type,
        num_virtual_tokens=num_virtual_tokens,
        encoder_hidden_size=768,    # RoBERTa-base hidden size
    )
    model = get_peft_model(model, config)
    return model


def load_and_tokenize(task_name, tokenizer, max_length):
    """Load GLUE dataset and tokenize."""
    cfg = TASK_CONFIG[task_name]
    dataset = load_dataset(cfg["dataset"], cfg["subset"])

    def tokenize_single(examples):
        return tokenizer(
            examples[cfg["text_col"]],
            truncation=True,
            max_length=max_length,
        )

    def tokenize_pair(examples):
        return tokenizer(
            examples[cfg["text_col"][0]],
            examples[cfg["text_col"][1]],
            truncation=True,
            max_length=max_length,
        )

    tokenize_fn = tokenize_pair if isinstance(cfg["text_col"], list) else tokenize_single

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column(cfg["label_col"], "labels")
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names
         if c not in ["input_ids", "attention_mask", "labels"]]
    )
    tokenized.set_format("torch")
    return tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


def run_experiment(method, task, output_dir="results", seed=42):
    """
    Main experiment runner.
    Loads RoBERTa-Base, applies PEFT method, trains on GLUE task,
    evaluates on validation set, and saves results.
    """
    print(f"\n{'='*60}")
    print(f"Running: {method.upper()} on {task.upper()}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    cfg = TASK_CONFIG[task]
    model_name = "roberta-base"

    # ── Load tokenizer and model ──
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=cfg["num_labels"]
    )

    # ── Apply PEFT method ──
    print(f"Applying {method} fine-tuning...")
    if method == "lora":
        model = apply_lora(model)
    elif method == "bitfit":
        model = apply_bitfit(model)
    elif method == "prefix":
        model = apply_prefix_tuning(model)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: lora, bitfit, prefix")

    trainable_m, total_m = count_trainable_params(model)
    print(f"Trainable params: {trainable_m:.3f}M / {total_m:.1f}M total "
          f"({100*trainable_m/total_m:.2f}%)")
    print(f"Paper reports: {PAPER_PARAMS[method]}M trainable params")

    # ── Load and tokenize data ──
    print(f"Loading {task} dataset...")
    tokenized = load_and_tokenize(task, tokenizer, cfg["max_length"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training arguments ──
    # These settings mirror typical PEFT paper configurations
    run_dir = os.path.join(output_dir, f"{method}_{task}")
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=10 if task == "rte" else 5,   # RTE needs more epochs (small dataset)
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-4 if method == "lora" else 1e-3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=seed,
        report_to="none",           # disable wandb/tensorboard for clean output
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ── Train ──
    print(f"\nTraining {method} on {task}...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # ── Evaluate ──
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    accuracy = eval_results["eval_accuracy"] * 100

    paper_acc = PAPER_RESULTS[method][task]
    diff = accuracy - paper_acc

    print(f"\n{'─'*40}")
    print(f"  Our result:    {accuracy:.2f}%")
    print(f"  Paper result:  {paper_acc:.2f}%")
    print(f"  Difference:    {diff:+.2f}%")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    print(f"{'─'*40}")

    # ── Save results ──
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "glue_results.csv")
    file_exists = os.path.isfile(results_file)

    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "method", "task", "accuracy_ours",
            "accuracy_paper", "diff", "trainable_params_M",
            "paper_params_M", "training_time_min", "seed"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "method": method,
            "task": task,
            "accuracy_ours": round(accuracy, 2),
            "accuracy_paper": paper_acc,
            "diff": round(diff, 2),
            "trainable_params_M": round(trainable_m, 3),
            "paper_params_M": PAPER_PARAMS[method],
            "training_time_min": round(elapsed / 60, 1),
            "seed": seed,
        })

    print(f"\nResults saved to {results_file}")
    return accuracy


def print_summary_table(results_file="results/glue_results.csv"):
    """Print a formatted summary table comparing our results to the paper."""
    if not os.path.isfile(results_file):
        print("No results file found. Run experiments first.")
        return

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY: Our Reproduction vs. Paper (Table 1, RoBERTa-Base)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Task':<8} {'Ours':>8} {'Paper':>8} {'Diff':>8} {'Params(M)':>12}")
    print(f"{'─'*70}")

    with open(results_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"{row['method']:<12} {row['task']:<8} "
                  f"{row['accuracy_ours']:>8} {row['accuracy_paper']:>8} "
                  f"{row['diff']:>+8} {row['trainable_params_M']:>12}")

    print(f"{'─'*70}")
    print("\nFull Fine-Tuning (from paper, not reproduced):")
    print(f"  SST-2: {PAPER_RESULTS['full']['sst2']}%  |  RTE: {PAPER_RESULTS['full']['rte']}%  |  Params: {PAPER_PARAMS['full']}M")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce PEFT GLUE results from Prottasha et al. (2025)"
    )
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["lora", "bitfit", "prefix"],
        help="PEFT method to run"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["sst2", "rte"],
        help="GLUE task"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary table of all completed runs"
    )

    args = parser.parse_args()

    if args.summary:
        print_summary_table(os.path.join(args.output_dir, "glue_results.csv"))
    else:
        run_experiment(
            method=args.method,
            task=args.task,
            output_dir=args.output_dir,
            seed=args.seed,
        )
