"""
Training Script for RoBERTa Named Entity Recognition
======================================================
Data source: data/bio_dataset.json
  - Pre-tokenised BIO-tagged dataset with tokens + labels per sample.
  - All 11 entity types with explicit BIO label list.

Features:
  - Per-epoch metrics recording (train/val loss, precision, recall, F1)
  - Comprehensive visualization generation
  - Hyperparameter tuning mode with grid search
  - All metrics and graphs saved to docs/graphs/entity/
"""

import json
import torch
import csv
import numpy as np
from transformers import (
    RobertaTokenizerFast, RobertaForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForTokenClassification,
)
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import NERDataset, load_bio_data, load_data_from_json
from utils import (
    post_process_entities,
    post_process_entities_with_confidence,
    compute_metrics,
    evaluate,
    create_visualizations,
)


# ─────────────────────────────────────────────────────────────────────────────
# Label definitions – explicit BIO tags for all entity types
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "COURSE", "LOCATION", "COLLEGE_TYPE",
    "RANK", "BUDGET", "HOSTEL",
    "COLLEGE_NAME", "COLLEGE_NAME_1", "COLLEGE_NAME_2",
    "ATTRIBUTE", "RATING",
]

LABELS = ["O"]
for _etype in ENTITY_TYPES:
    LABELS.append(f"B-{_etype}")
    LABELS.append(f"I-{_etype}")

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

print(f"Total labels: {len(LABELS)}")


# ─────────────────────────────────────────────────────────────────────────────
# NER Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NERTrainer:
    """RoBERTa token-classification trainer.

    Reads from ``bio_dataset.json`` – a pre-tokenised BIO-tagged dataset
    with ``tokens`` and ``labels`` per sample.
    """

    def __init__(self, model_name: str = "roberta-base", max_length: int = 128):
        self.model_name  = model_name
        self.max_length  = max_length
        self.tokenizer   = RobertaTokenizerFast.from_pretrained(
            model_name, add_prefix_space=True
        )
        self.model       = None
        self.label2id    = LABEL2ID
        self.id2label    = ID2LABEL

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_datasets(self, samples: list,
                         test_size: float = 0.10,
                         val_size:  float = 0.10) -> tuple:
        """Split samples into train / val / test and wrap as NERDataset.

        Default proportions: 80 % train · 10 % val · 10 % test.
        """
        tmp, test_samples = train_test_split(
            samples, test_size=test_size, random_state=42
        )
        val_ratio = val_size / (1 - test_size)
        train_samples, val_samples = train_test_split(
            tmp, test_size=val_ratio, random_state=42
        )

        train_ds = NERDataset(train_samples, self.tokenizer, self.label2id, self.max_length)
        val_ds   = NERDataset(val_samples,   self.tokenizer, self.label2id, self.max_length)
        test_ds  = NERDataset(test_samples,  self.tokenizer, self.label2id, self.max_length)

        print(f"Split → Train: {len(train_ds):,}  "
              f"Val: {len(val_ds):,}  Test: {len(test_ds):,}")
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_dataset, val_dataset, output_dir: str,
              num_epochs: int = 15, batch_size: int = 16,
              learning_rate: float = 2e-5) -> Trainer:
        """Fine-tune RoBERTa for token classification (NER)."""

        self.model = RobertaForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )

        training_args = TrainingArguments(
            output_dir                  = output_dir,
            num_train_epochs            = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            warmup_ratio                = 0.1,
            weight_decay                = 0.01,
            logging_dir                 = os.path.join(output_dir, "logs"),
            logging_steps               = 10,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1",
            greater_is_better           = True,
            learning_rate               = learning_rate,
            save_total_limit            = 2,
            fp16                        = torch.cuda.is_available(),
            dataloader_num_workers      = 0,
            report_to                   = "none",
        )

        trainer = Trainer(
            model           = self.model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            data_collator   = data_collator,
            compute_metrics = lambda p: compute_metrics(p, self.id2label),
            callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Starting RoBERTa NER training …")
        trainer.train()

        # Persist model + tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Persist label mappings
        with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
            json.dump(
                {"label2id": self.label2id,
                 "id2label": {str(k): v for k, v in self.id2label.items()}},
                f, indent=2,
            )

        print(f"Model saved → {output_dir}")
        return trainer

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str, threshold: float = 0.30) -> list:
        """Extract entities from a single text string."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Load / train a model before calling predict().")

        self.model.eval()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids      = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits

        probs  = torch.softmax(logits, dim=-1)[0]
        preds  = torch.argmax(logits, dim=-1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [self.id2label[p.item()] for p in preds]

        filtered = [
            (t, l, probs[i][preds[i]].item())
            for i, (t, l) in enumerate(zip(tokens, labels))
            if t not in [self.tokenizer.cls_token,
                         self.tokenizer.sep_token,
                         self.tokenizer.pad_token]
        ]
        if not filtered:
            return []

        tokens, labels, scores = zip(*filtered)
        spans = post_process_entities_with_confidence(tokens, labels, scores)
        return [e for e in spans if e["confidence"] >= threshold]


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions for Metrics Recording
# ─────────────────────────────────────────────────────────────────────────────

def save_epoch_metrics_csv(graphics_dir: str, trainer_logs: list):
    """Save per-epoch metrics to CSV file for NER training."""
    epoch_data = {}

    for log in trainer_logs:
        if 'eval_loss' in log:
            epoch = int(log.get('epoch', len(epoch_data) + 1))
            epoch_data[epoch] = {
                'epoch': epoch,
                'train_loss': 'N/A',
                'val_loss': log['eval_loss'],
                'precision': log.get('eval_precision', 0),
                'recall': log.get('eval_recall', 0),
                'f1_score': log.get('eval_f1', 0),
                'accuracy': log.get('eval_accuracy', 0),
            }

    train_losses_by_epoch = {}
    current_epoch = 0
    for log in trainer_logs:
        if 'epoch' in log:
            current_epoch = int(log['epoch'])
        if 'loss' in log and 'eval_loss' not in log:
            if current_epoch not in train_losses_by_epoch:
                train_losses_by_epoch[current_epoch] = []
            train_losses_by_epoch[current_epoch].append(log['loss'])

    for epoch, losses in train_losses_by_epoch.items():
        if epoch in epoch_data:
            epoch_data[epoch]['train_loss'] = sum(losses) / len(losses)

    csv_path = os.path.join(graphics_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss',
                                                'precision', 'recall', 'f1_score', 'accuracy'])
        writer.writeheader()
        for epoch in sorted(epoch_data.keys()):
            writer.writerow(epoch_data[epoch])

    print(f"✅ Epoch metrics saved to {csv_path}")
    return epoch_data


def create_hyperparameter_heatmap_ner(graphics_dir: str, tuning_results: list):
    """Create heatmap of LR vs Batch Size vs Seqeval F1."""
    if not tuning_results:
        return

    lrs = sorted(set(r['learning_rate'] for r in tuning_results))
    batches = sorted(set(r['batch_size'] for r in tuning_results))

    f1_matrix = np.zeros((len(lrs), len(batches)))
    for r in tuning_results:
        i = lrs.index(r['learning_rate'])
        j = batches.index(r['batch_size'])
        f1_matrix[i, j] = r['seqeval_f1']

    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[str(b) for b in batches],
                yticklabels=[f'{lr:.0e}' for lr in lrs])
    plt.title('Hyperparameter Tuning: LR vs Batch Size vs Seqeval F1', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'hyperparameter_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Hyperparameter heatmap saved")


def create_per_entity_grouped_bar(graphics_dir: str, per_entity_metrics: dict):
    """Create grouped bar chart for Precision, Recall, F1 per entity type."""
    entities = []
    precisions = []
    recalls = []
    f1_scores = []

    for entity, metrics in per_entity_metrics.items():
        if entity not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'O'] and isinstance(metrics, dict):
            entities.append(entity)
            precisions.append(metrics.get('precision', 0))
            recalls.append(metrics.get('recall', 0))
            f1_scores.append(metrics.get('f1-score', 0))

    if not entities:
        return

    x = np.arange(len(entities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#FF6B6B')
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#4ECDC4')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#45B7D1')

    ax.set_xlabel('Entity Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, F1 per Entity Type', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(entities, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'per_entity_grouped_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Per-entity grouped bar chart saved")


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_tuning_ner(data_path: str, output_base_dir: str, graphics_dir: str,
                                   max_length: int = 128):
    """Run grid search over hyperparameters for NER and record all results."""

    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes = [16, 32]
    epoch_configs = [5, 10]

    results = []
    best_result = None
    best_f1 = 0

    # Load BIO data
    samples = load_bio_data(data_path)

    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING - Grid Search (NER)")
    print("="*60)
    print(f"  Learning Rates: {learning_rates}")
    print(f"  Batch Sizes   : {batch_sizes}")
    print(f"  Epochs        : {epoch_configs}")
    print(f"  Total runs    : {len(learning_rates) * len(batch_sizes) * len(epoch_configs)}")
    print("="*60 + "\n")

    run_idx = 0
    total_runs = len(learning_rates) * len(batch_sizes) * len(epoch_configs)

    for lr in learning_rates:
        for batch in batch_sizes:
            for epochs in epoch_configs:
                run_idx += 1
                print(f"\n[Run {run_idx}/{total_runs}] LR={lr}, Batch={batch}, Epochs={epochs}")

                run_trainer = NERTrainer(max_length=max_length)

                train_ds, val_ds, test_ds = run_trainer.prepare_datasets(
                    samples, test_size=0.10, val_size=0.10
                )

                run_output_dir = os.path.join(output_base_dir, f"run_lr{lr}_b{batch}_e{epochs}")
                os.makedirs(run_output_dir, exist_ok=True)

                model_trainer = run_trainer.train(
                    train_ds, val_ds, run_output_dir,
                    num_epochs=epochs, batch_size=batch, learning_rate=lr
                )

                eval_results = model_trainer.evaluate(val_ds)

                result = {
                    'learning_rate': lr,
                    'batch_size': batch,
                    'epochs': epochs,
                    'seqeval_f1': eval_results.get('eval_f1', 0),
                    'precision': eval_results.get('eval_precision', 0),
                    'recall': eval_results.get('eval_recall', 0),
                    'val_loss': eval_results.get('eval_loss', 0),
                }
                results.append(result)

                print(f"  → Seqeval F1: {result['seqeval_f1']:.4f}")

                if result['seqeval_f1'] > best_f1:
                    best_f1 = result['seqeval_f1']
                    best_result = result.copy()
                    best_result['output_dir'] = run_output_dir

    os.makedirs(graphics_dir, exist_ok=True)

    csv_path = os.path.join(graphics_dir, 'hyperparameter_tuning_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['learning_rate', 'batch_size', 'epochs',
                                                'seqeval_f1', 'precision', 'recall', 'val_loss'])
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.join(graphics_dir, 'hyperparameter_tuning_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_result,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    create_hyperparameter_heatmap_ner(graphics_dir, results)

    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING COMPLETE (NER)")
    print("="*60)
    print(f"  Best Config: LR={best_result['learning_rate']}, Batch={best_result['batch_size']}, Epochs={best_result['epochs']}")
    print(f"  Best Seqeval F1: {best_result['seqeval_f1']:.4f}")
    print(f"  Results saved: {csv_path}")
    print("="*60)

    return results, best_result


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))

    print(f"Project root: {project_root}")

    parser = argparse.ArgumentParser(
        description="Train RoBERTa NER Model from bio_dataset.json"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(project_root, "data/bio_dataset.json"),
        help="Path to BIO-tagged JSON dataset (tokens + labels)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_root, "models/roberta_entity_model"),
        help="Output directory for saved model and artefacts",
    )
    parser.add_argument(
        "--graphics_dir",
        default=os.path.join(project_root, "docs/graphs/entity"),
        help="Directory for evaluation visualisations",
    )
    parser.add_argument("--epochs",        type=int,   default=15,   help="Training epochs")
    parser.add_argument("--batch_size",    type=int,   default=16,   help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--max_length",    type=int,   default=128,  help="Max token length")
    parser.add_argument("--test_size",     type=float, default=0.10, help="Test split fraction")
    parser.add_argument("--val_size",      type=float, default=0.10, help="Val split fraction")
    parser.add_argument("--tune",          action="store_true",      help="Run hyperparameter tuning")

    args = parser.parse_args()

    # Hyperparameter tuning mode
    if args.tune:
        run_hyperparameter_tuning_ner(
            args.data_path, args.output_dir, args.graphics_dir, args.max_length
        )
        return

    print("\n" + "="*60)
    print("  RoBERTa NER — Training (BIO Dataset)")
    print("="*60)
    print(f"  Data       : {args.data_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Graphics   : {args.graphics_dir}")
    print(f"  Epochs     : {args.epochs}  |  LR: {args.learning_rate}")
    print(f"  Batch size : {args.batch_size}  |  Max len: {args.max_length}")
    print(f"  Test/Val   : {args.test_size:.0%} / {args.val_size:.0%}")
    print(f"  Num labels : {len(LABELS)}")
    print("="*60 + "\n")

    os.makedirs(args.output_dir,   exist_ok=True)
    os.makedirs(args.graphics_dir, exist_ok=True)

    # Load BIO data ──────────────────────────────────────────────────────
    samples = load_bio_data(args.data_path)

    # Prepare ────────────────────────────────────────────────────────────
    ner_trainer = NERTrainer(max_length=args.max_length)
    train_ds, val_ds, test_ds = ner_trainer.prepare_datasets(
        samples,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    # Train ──────────────────────────────────────────────────────────────
    model_trainer = ner_trainer.train(
        train_ds, val_ds, args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Evaluate ───────────────────────────────────────────────────────────
    results, report, eval_metrics, true_lbls, pred_lbls = evaluate(
        test_ds, model_trainer, ner_trainer.id2label
    )

    # Visualisations ─────────────────────────────────────────────────────
    # Extract texts/labels from samples for visualization functions
    all_texts  = [" ".join(s["tokens"]) for s in samples]
    all_labels = [s["labels"] for s in samples]

    trainer_logs = (model_trainer.state.log_history
                    if hasattr(model_trainer.state, "log_history") else None)
    create_visualizations(args.graphics_dir, all_texts, all_labels, eval_metrics, trainer_logs)

    # Save per-epoch metrics CSV
    if trainer_logs:
        save_epoch_metrics_csv(args.graphics_dir, trainer_logs)

    # Create per-entity grouped bar chart (Precision, Recall, F1)
    if 'per_entity_metrics' in eval_metrics:
        create_per_entity_grouped_bar(args.graphics_dir, eval_metrics['per_entity_metrics'])

    # Save artefacts ─────────────────────────────────────────────────────
    detailed_results = {
        "training_results": results,
        "evaluation_metrics": {
            "entity_precision":  eval_metrics["entity_precision"],
            "entity_recall":     eval_metrics["entity_recall"],
            "entity_f1":         eval_metrics["entity_f1"],
            "bio_validity_rate": eval_metrics["bio_validity_rate"],
            "per_entity_f1": {
                k: (v.get("f1-score", 0) if isinstance(v, dict) else 0)
                for k, v in eval_metrics.get("per_entity_metrics", {}).items()
                if k not in ("accuracy", "macro avg", "weighted avg", "micro avg")
            },
        },
        "timestamp":    datetime.now().isoformat(),
        "data_source":  "bio_dataset.json",
        "model_config": vars(args),
    }

    with open(os.path.join(args.output_dir, "comprehensive_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save per-entity metrics CSV
    per_entity_csv = os.path.join(args.graphics_dir, "per_entity_metrics.csv")
    with open(per_entity_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entity', 'precision', 'recall', 'f1_score', 'support'])
        for entity, metrics in eval_metrics.get('per_entity_metrics', {}).items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                if entity not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']:
                    writer.writerow([
                        entity,
                        f"{metrics['precision']:.4f}",
                        f"{metrics['recall']:.4f}",
                        f"{metrics['f1-score']:.4f}",
                        metrics.get('support', 0)
                    ])

    # Summary ────────────────────────────────────────────────────────────
    ep  = eval_metrics["entity_precision"]
    er  = eval_metrics["entity_recall"]
    ef1 = eval_metrics["entity_f1"]
    bvr = eval_metrics["bio_validity_rate"]

    print("\n" + "="*60)
    print("  Training complete!")
    print(f"  Model saved         : {args.output_dir}")
    print(f"  Entity Precision    : {ep:.4f}")
    print(f"  Entity Recall       : {er:.4f}")
    print(f"  Entity F1           : {ef1:.4f}")
    print(f"  BIO Validity Rate   : {bvr:.1%}")
    print(f"  Visualisations      : {args.graphics_dir}/")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ Entity Precision / Recall / F1")
    print("  ✓ Per-entity F1 scores")
    print("  ✓ Per-entity grouped bar chart (P/R/F1)")
    print("  ✓ BIO validity rate pie")
    print("  ✓ Confusion matrix")
    print("  ✓ Training vs Validation Loss")
    print("  ✓ Validation Accuracy / F1 progression")
    print("  ✓ Learning Rate schedule")
    print("  ✓ Combined Training Dashboard")
    print("  ✓ Label distribution & sentence statistics")
    print("  ✓ training_metrics.csv - Per-epoch metrics")
    print("  ✓ per_entity_metrics.csv - Per-entity metrics")


if __name__ == "__main__":
    main()
