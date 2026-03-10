"""
Training Script for BERT Intent Classification
================================================
Data source: data/intent_entity.json
  - Combined dataset providing intent labels AND entity annotations for every
    sentence.
  - This trainer consumes only the `text` + `intent` fields.
  - Every record is guaranteed to carry a single string intent label.

Features:
  - Per-epoch metrics recording (train/val loss, accuracy, F1)
  - Comprehensive visualization generation (loss curves, confusion matrix, etc.)
  - Hyperparameter tuning mode with grid search
  - All metrics and graphs saved to docs/graphs/intent/
"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix
)
import numpy as np
import os
from datetime import datetime
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import csv


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class IntentDataset(Dataset):
    """PyTorch Dataset for BERT intent-classification fine-tuning."""

    def __init__(self, texts: list, labels: list,
                 tokenizer: BertTokenizer, max_length: int = 128):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────────────

def create_intent_visualizations(graphics_dir: str, trainer_logs: list,
                                  y_true: np.ndarray, y_pred: np.ndarray,
                                  id_to_label: dict, report_dict: dict):
    """Create all visualizations for intent classification training."""
    os.makedirs(graphics_dir, exist_ok=True)
    
    # Extract metrics from trainer logs
    train_losses, eval_losses = [], []
    eval_accuracy, eval_f1 = [], []
    eval_epochs, learning_rates = [], []
    
    for log in trainer_logs:
        if 'loss' in log and 'eval_loss' not in log:
            train_losses.append(log['loss'])
            if 'learning_rate' in log:
                learning_rates.append(log['learning_rate'])
        elif 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
            eval_accuracy.append(log.get('eval_accuracy', 0))
            eval_f1.append(log.get('eval_f1', 0))
            eval_epochs.append(log.get('epoch', len(eval_losses)))
    
    # 1. Training vs Validation Loss Curve
    if train_losses or eval_losses:
        plt.figure(figsize=(12, 8))
        if train_losses:
            steps = list(range(1, len(train_losses) + 1))
            plt.plot(steps, train_losses, label='Training Loss', linewidth=2, 
                    color='#FF6B6B', alpha=0.8)
        if eval_losses and eval_epochs:
            steps_per_epoch = len(train_losses) / max(eval_epochs) if train_losses and eval_epochs else 1
            eval_steps = [int(e * steps_per_epoch) for e in eval_epochs]
            plt.plot(eval_steps, eval_losses, 'o-', label='Validation Loss',
                    linewidth=3, color='#4ECDC4', markersize=8)
        plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Training vs Validation Accuracy Curve
    if eval_accuracy and eval_epochs:
        plt.figure(figsize=(12, 8))
        plt.plot(eval_epochs, eval_accuracy, 'o-', label='Validation Accuracy',
                linewidth=3, color='#45B7D1', markersize=10, markerfacecolor='white',
                markeredgewidth=2)
        plt.title('Validation Accuracy Over Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        for i, (x, y) in enumerate(zip(eval_epochs, eval_accuracy)):
            plt.annotate(f'{y:.1%}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Confusion Matrix
    target_names = [id_to_label[i] for i in range(len(id_to_label))]
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Intent Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Intent', fontsize=12)
    plt.ylabel('True Intent', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. F1 Score Per Intent Class (Bar Chart)
    intent_f1_scores = {}
    for intent, scores in report_dict.items():
        if intent not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(scores, dict):
            intent_f1_scores[intent] = scores.get('f1-score', 0)
    
    if intent_f1_scores:
        plt.figure(figsize=(14, 8))
        intents = list(intent_f1_scores.keys())
        f1_scores = list(intent_f1_scores.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(intents)))
        bars = plt.bar(intents, f1_scores, color=colors)
        plt.title('F1 Score Per Intent Class', fontsize=16, fontweight='bold')
        plt.xlabel('Intent', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'f1_per_intent.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Combined Training Dashboard
    if eval_losses and eval_accuracy and eval_f1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss
        if train_losses:
            steps = list(range(1, len(train_losses) + 1))
            axes[0, 0].plot(steps, train_losses, label='Train Loss', color='#FF6B6B', alpha=0.8)
        if eval_losses:
            steps_per_epoch = len(train_losses) / max(eval_epochs) if train_losses else 1
            eval_steps = [int(e * steps_per_epoch) for e in eval_epochs]
            axes[0, 0].plot(eval_steps, eval_losses, 'o-', label='Val Loss', color='#4ECDC4')
        axes[0, 0].set_title('Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(eval_epochs, eval_accuracy, 'o-', color='#45B7D1', markersize=8)
        axes[0, 1].set_title('Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(eval_epochs, eval_f1, 'o-', color='#96CEB4', markersize=8)
        axes[1, 0].set_title('F1 Score', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if learning_rates:
            lr_steps = list(range(1, len(learning_rates) + 1))
            axes[1, 1].plot(lr_steps, learning_rates, color='#FFA07A')
            axes[1, 1].set_title('Learning Rate', fontweight='bold')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'LR Data\nUnavailable', ha='center', va='center',
                          transform=axes[1, 1].transAxes, fontsize=12)
        
        plt.suptitle('BERT Intent Classification - Training Dashboard', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(os.path.join(graphics_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to {graphics_dir}")


def create_hyperparameter_heatmap(graphics_dir: str, tuning_results: list):
    """Create heatmap of LR vs Batch Size vs Accuracy."""
    if not tuning_results:
        return
    
    # Extract unique values
    lrs = sorted(set(r['learning_rate'] for r in tuning_results))
    batches = sorted(set(r['batch_size'] for r in tuning_results))
    
    # Build accuracy matrix
    acc_matrix = np.zeros((len(lrs), len(batches)))
    for r in tuning_results:
        i = lrs.index(r['learning_rate'])
        j = batches.index(r['batch_size'])
        acc_matrix[i, j] = r['val_accuracy']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(acc_matrix, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=[str(b) for b in batches],
                yticklabels=[f'{lr:.0e}' for lr in lrs])
    plt.title('Hyperparameter Tuning: LR vs Batch Size vs Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, 'hyperparameter_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Hyperparameter heatmap saved")


def save_epoch_metrics_csv(graphics_dir: str, trainer_logs: list):
    """Save per-epoch metrics to CSV file."""
    rows = []
    epoch_data = {}
    
    for log in trainer_logs:
        if 'eval_loss' in log:
            epoch = int(log.get('epoch', len(epoch_data) + 1))
            epoch_data[epoch] = {
                'epoch': epoch,
                'train_loss': log.get('loss', 'N/A'),
                'val_loss': log['eval_loss'],
                'accuracy': log.get('eval_accuracy', 0),
                'f1_score': log.get('eval_f1', 0),
                'precision': log.get('eval_precision', 0),
                'recall': log.get('eval_recall', 0),
            }
    
    # Also capture train loss from non-eval logs
    train_losses_by_epoch = {}
    current_epoch = 0
    for log in trainer_logs:
        if 'epoch' in log:
            current_epoch = int(log['epoch'])
        if 'loss' in log and 'eval_loss' not in log:
            if current_epoch not in train_losses_by_epoch:
                train_losses_by_epoch[current_epoch] = []
            train_losses_by_epoch[current_epoch].append(log['loss'])
    
    # Average train loss per epoch
    for epoch, losses in train_losses_by_epoch.items():
        if epoch in epoch_data:
            epoch_data[epoch]['train_loss'] = sum(losses) / len(losses)
    
    csv_path = os.path.join(graphics_dir, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 
                                                'accuracy', 'f1_score', 'precision', 'recall'])
        writer.writeheader()
        for epoch in sorted(epoch_data.keys()):
            writer.writerow(epoch_data[epoch])
    
    print(f"✅ Epoch metrics saved to {csv_path}")
    return epoch_data


# ─────────────────────────────────────────────────────────────────────────────
# Trainer class
# ─────────────────────────────────────────────────────────────────────────────

class IntentTrainer:
    """BERT-based intent classification trainer.

    Reads from the unified ``intent_entity.json`` that provides intent labels,
    entity annotations, and rich metadata in a single source of truth.
    """

    def __init__(self, model_name: str = "bert-base-uncased",
                 max_length: int = 128):
        self.model_name    = model_name
        self.max_length    = max_length
        self.tokenizer     = None
        self.model         = None
        self.label_mapping: dict = {}   # intent → id
        self.id_to_label:   dict = {}   # id → intent

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_path: str) -> tuple:
        """Load training data from intent_entity.json.

        Expected JSON structure::

            {
              "metadata": {
                "total_sentences": 5374,
                "unique_intents": 23,
                "intent_distribution": { ... },
                ...
              },
              "sentences": [
                {
                  "text":     "What is the fee of ABC College in Kathmandu?",
                  "intent":   "GET_fee_info",
                  "entities": { "COLLEGE_NAME": "ABC College", "LOCATION": "Kathmandu" }
                },
                ...
              ]
            }

        Only ``text`` and ``intent`` fields are consumed here; ``entities``
        are used by the NER model (train_roberta_ner.py).
        """
        print(f"Loading intent data from: {data_path}")

        ext = os.path.splitext(data_path)[1].lower()

        if ext == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            # ── Print dataset-level metadata ──────────────────────────
            meta = raw.get("metadata", {})
            if meta:
                print("\n" + "="*60)
                print(f"  Dataset    : {os.path.basename(data_path)}")
                print(f"  Total rows : {meta.get('total_sentences', 'N/A'):,} sentences")
                print(f"  Intents    : {meta.get('unique_intents', 'N/A')}")
                print(f"  Entity types: {meta.get('unique_entity_types', 'N/A')}")
                src = meta.get('source', meta.get('sources', []))
                if isinstance(src, list): src = ', '.join(src)
                print(f"  Sources    : {src}")
                print("="*60 + "\n")

            sentences = raw.get("sentences", raw if isinstance(raw, list) else [])

            texts, intents, skipped = [], [], 0
            for item in sentences:
                if not isinstance(item, dict):
                    skipped += 1
                    continue
                text   = item.get("text", "").strip()
                intent = item.get("intent", "")

                # Normalise legacy list format (not present in intent_entity.json)
                if isinstance(intent, list):
                    intent = intent[0] if intent else "Unknown"

                if not text or not intent:
                    skipped += 1
                    continue

                texts.append(text)
                intents.append(intent)

            if skipped:
                print(f"  ⚠ Skipped {skipped} malformed records.")

        elif ext == ".csv":
            df      = pd.read_csv(data_path)
            texts   = df["text"].tolist()
            intents = df["intent"].tolist()

        else:
            raise ValueError(f"Unsupported format '{ext}'. Use .json or .csv")

        # ── Build sorted, reproducible label mappings ──────────────────
        unique_intents     = sorted(set(intents))
        self.label_mapping = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.id_to_label   = {idx: intent for intent, idx in self.label_mapping.items()}
        labels             = [self.label_mapping[i] for i in intents]

        print(f"Loaded {len(texts):,} examples across {len(unique_intents)} intents.\n")
        print("Intent distribution:")
        for intent, cnt in sorted(Counter(intents).items(), key=lambda x: -x[1]):
            bar = "█" * (cnt // 15)
            print(f"  {intent:<45s} {cnt:5d}  {bar}")
        print()

        return texts, labels

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_datasets(self, texts: list, labels: list,
                         test_size: float = 0.15,
                         val_size:  float = 0.10) -> tuple:
        """Stratified split into train / val / test and tokenise.

        Default proportions: 75 % train · 10 % val · 15 % test.
        """
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            texts, labels,
            test_size=test_size, stratify=labels, random_state=42
        )
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=val_ratio, stratify=y_tmp, random_state=42
        )

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        train_ds = IntentDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_ds   = IntentDataset(X_val,   y_val,   self.tokenizer, self.max_length)
        test_ds  = IntentDataset(X_test,  y_test,  self.tokenizer, self.max_length)

        print(f"Split → Train: {len(train_ds):,}  "
              f"Val: {len(val_ds):,}  Test: {len(test_ds):,}")
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, eval_pred) -> dict:
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        return {
            "accuracy":  accuracy_score(labels, preds),
            "f1":        f1,
            "precision": precision,
            "recall":    recall,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_dataset, val_dataset, output_dir: str,
              num_epochs: int = 10, batch_size: int = 16,
              learning_rate: float = 2e-5) -> Trainer:
        """Fine-tune BERT for multi-class intent classification."""

        num_labels = len(self.label_mapping)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir                  = output_dir,
            num_train_epochs            = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            warmup_ratio                = 0.1,
            weight_decay                = 0.01,
            logging_dir                 = os.path.join(output_dir, "logs"),
            logging_steps               = 20,
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
            compute_metrics = self.compute_metrics,
            callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Starting BERT intent training …")
        trainer.train()

        # Persist model + tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Persist label mappings
        with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
            json.dump(
                {"label_to_id": self.label_mapping,
                 "id_to_label": self.id_to_label},
                f, indent=2
            )

        print(f"Model saved → {output_dir}")
        return trainer

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_dataset, trainer: Trainer) -> tuple:
        """Detailed evaluation on the held-out test split."""

        print("Evaluating on test set …")
        results  = trainer.evaluate(test_dataset)
        pout     = trainer.predict(test_dataset)
        y_pred   = np.argmax(pout.predictions, axis=1)
        y_true   = pout.label_ids

        target_names = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        report = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
        report_dict = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True
        )

        print("\nClassification Report:\n" + report)
        return results, report, report_dict, y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_tuning(data_path: str, output_base_dir: str, graphics_dir: str,
                               max_length: int = 128):
    """Run grid search over hyperparameters and record all results."""
    
    # Hyperparameter grid
    learning_rates = [2e-5, 3e-5, 5e-5]
    batch_sizes = [16, 32]
    epoch_configs = [3, 5]
    
    results = []
    best_result = None
    best_f1 = 0
    
    # Load data once
    trainer_obj = IntentTrainer(max_length=max_length)
    texts, labels = trainer_obj.load_data(data_path)
    
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING - Grid Search")
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
                
                # Fresh trainer for each run
                run_trainer = IntentTrainer(max_length=max_length)
                run_trainer.label_mapping = trainer_obj.label_mapping
                run_trainer.id_to_label = trainer_obj.id_to_label
                
                # Prepare datasets
                train_ds, val_ds, test_ds = run_trainer.prepare_datasets(
                    texts, labels, test_size=0.15, val_size=0.10
                )
                
                # Train
                run_output_dir = os.path.join(output_base_dir, f"run_lr{lr}_b{batch}_e{epochs}")
                os.makedirs(run_output_dir, exist_ok=True)
                
                model_trainer = run_trainer.train(
                    train_ds, val_ds, run_output_dir,
                    num_epochs=epochs, batch_size=batch, learning_rate=lr
                )
                
                # Evaluate
                eval_results = model_trainer.evaluate(val_ds)
                
                result = {
                    'learning_rate': lr,
                    'batch_size': batch,
                    'epochs': epochs,
                    'val_accuracy': eval_results.get('eval_accuracy', 0),
                    'val_f1': eval_results.get('eval_f1', 0),
                    'val_loss': eval_results.get('eval_loss', 0),
                }
                results.append(result)
                
                print(f"  → Val Accuracy: {result['val_accuracy']:.4f}, Val F1: {result['val_f1']:.4f}")
                
                if result['val_f1'] > best_f1:
                    best_f1 = result['val_f1']
                    best_result = result.copy()
                    best_result['output_dir'] = run_output_dir
    
    # Save tuning results
    os.makedirs(graphics_dir, exist_ok=True)
    
    # CSV table
    csv_path = os.path.join(graphics_dir, 'hyperparameter_tuning_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['learning_rate', 'batch_size', 'epochs',
                                                'val_accuracy', 'val_f1', 'val_loss'])
        writer.writeheader()
        writer.writerows(results)
    
    # JSON results
    json_path = os.path.join(graphics_dir, 'hyperparameter_tuning_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_result,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Generate heatmap
    create_hyperparameter_heatmap(graphics_dir, results)
    
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    print(f"  Best Config: LR={best_result['learning_rate']}, Batch={best_result['batch_size']}, Epochs={best_result['epochs']}")
    print(f"  Best Val F1: {best_result['val_f1']:.4f}")
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
        description="Train BERT Intent Classifier from intent_entity.json"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(project_root, "data/intent_entity.json"),
        help="Path to training JSON dataset (intent+entity format)",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_root, "models/bert_intent_model"),
        help="Output directory for saved model and artefacts",
    )
    parser.add_argument(
        "--graphics_dir",
        default=os.path.join(project_root, "docs/graphs/intent"),
        help="Directory for training visualizations and metrics",
    )
    parser.add_argument("--epochs",        type=int,   default=8,    help="Training epochs")
    parser.add_argument("--batch_size",    type=int,   default=16,   help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--max_length",    type=int,   default=128,  help="Max token length")
    parser.add_argument("--test_size",     type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--val_size",      type=float, default=0.10, help="Val split fraction")
    parser.add_argument("--tune",          action="store_true",      help="Run hyperparameter tuning")

    args = parser.parse_args()

    # Hyperparameter tuning mode
    if args.tune:
        run_hyperparameter_tuning(
            args.data_path, args.output_dir, args.graphics_dir, args.max_length
        )
        return

    print("\n" + "="*60)
    print("  BERT Intent Classifier — Training")
    print("="*60)
    print(f"  Data       : {args.data_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Graphics   : {args.graphics_dir}")
    print(f"  Epochs     : {args.epochs}  |  LR: {args.learning_rate}")
    print(f"  Batch size : {args.batch_size}  |  Max len: {args.max_length}")
    print(f"  Test/Val   : {args.test_size:.0%} / {args.val_size:.0%}")
    print("="*60 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.graphics_dir, exist_ok=True)

    # Load ───────────────────────────────────────────────────────────────
    trainer_obj = IntentTrainer(max_length=args.max_length)
    texts, labels = trainer_obj.load_data(args.data_path)

    # Prepare ────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = trainer_obj.prepare_datasets(
        texts, labels, test_size=args.test_size, val_size=args.val_size
    )

    # Train ──────────────────────────────────────────────────────────────
    model_trainer = trainer_obj.train(
        train_ds, val_ds, args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    # Evaluate ───────────────────────────────────────────────────────────
    results, report, report_dict, y_true, y_pred = trainer_obj.evaluate(test_ds, model_trainer)

    # Generate visualizations ────────────────────────────────────────────
    trainer_logs = model_trainer.state.log_history if hasattr(model_trainer.state, 'log_history') else []
    
    create_intent_visualizations(
        args.graphics_dir, trainer_logs,
        y_true, y_pred, trainer_obj.id_to_label, report_dict
    )
    
    epoch_metrics = save_epoch_metrics_csv(args.graphics_dir, trainer_logs)

    # Save artefacts ─────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        json.dump({
            "results":     results,
            "timestamp":   datetime.now().isoformat(),
            "args":        vars(args),
            "data_source": "intent_entity.json",
            "num_intents": len(trainer_obj.label_mapping),
            "intents":     trainer_obj.label_mapping,
        }, f, indent=2)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Save classification report as JSON for per-class metrics
    with open(os.path.join(args.graphics_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)
    
    # Save per-class metrics CSV
    per_class_csv = os.path.join(args.graphics_dir, "per_class_metrics.csv")
    with open(per_class_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['intent', 'precision', 'recall', 'f1_score', 'support'])
        for intent, metrics in report_dict.items():
            if isinstance(metrics, dict) and 'precision' in metrics:
                writer.writerow([
                    intent,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}",
                    metrics.get('support', 0)
                ])

    # Summary ────────────────────────────────────────────────────────────
    acc = results.get("eval_accuracy", float("nan"))
    f1  = results.get("eval_f1",       float("nan"))
    print("\n" + "="*60)
    print("  Training complete!")
    print(f"  Model saved       : {args.output_dir}")
    print(f"  Visualizations    : {args.graphics_dir}")
    print(f"  Test Accuracy     : {acc:.4f}")
    print(f"  Test F1           : {f1:.4f}")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ training_curves.png      - Loss curves")
    print("  ✓ accuracy_curve.png       - Accuracy over epochs")
    print("  ✓ confusion_matrix.png     - Confusion matrix")
    print("  ✓ f1_per_intent.png        - F1 per intent class")
    print("  ✓ training_dashboard.png   - Combined dashboard")
    print("  ✓ training_metrics.csv     - Per-epoch metrics")
    print("  ✓ per_class_metrics.csv    - Per-intent metrics")
    print("  ✓ classification_report.json")


if __name__ == "__main__":
    main()
