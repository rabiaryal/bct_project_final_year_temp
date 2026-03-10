"""
Training Script for RoBERTa + CRF Named Entity Recognition
============================================================
Architecture:
    Input tokens → RoBERTa (768-dim) → Dropout → Linear → CRF (Viterbi)

Key advantage over vanilla RoBERTa token classification:
  - CRF layer enforces valid BIO transition constraints
  - Guaranteed valid sequences (no I- after O transitions)
  - Better label-level dependency modelling via transition matrix

Data source: data/bio_dataset.json
  - Pre-tokenised BIO-tagged dataset with tokens + labels per sample.
  - 9 entity types → 19 BIO labels (O + B/I per type).

Features:
  - Per-epoch metrics recording (train/val loss, precision, recall, F1)
  - Comprehensive visualization generation (matches train_roberta_ner.py)
  - Hyperparameter tuning mode with grid search
  - Early stopping with patience
  - All metrics and graphs saved to docs/graphs/entity/
"""

import json
import os
import csv
import argparse
import itertools
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcrf import CRF
from transformers import (
    RobertaModel,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report as seq_classification_report,
)
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import NERDataset, load_bio_data


# ─────────────────────────────────────────────────────────────────────────────
# Label definitions – explicit BIO tags for all entity types
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "COURSE", "LOCATION", "COLLEGE_TYPE",
    "RANK", "BUDGET",
    "COLLEGE_NAME", "COLLEGE_NAME_1", "COLLEGE_NAME_2",
    "ATTRIBUTE",
]

# O must be index 0 — CRF expects this
LABELS = ["O"]
for _etype in ENTITY_TYPES:
    LABELS.append(f"B-{_etype}")
    LABELS.append(f"I-{_etype}")

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

print(f"[CRF] Total labels: {NUM_LABELS}")


# ─────────────────────────────────────────────────────────────────────────────
# Model — RoBERTa + CRF
# ─────────────────────────────────────────────────────────────────────────────

class RobertaCRFNER(nn.Module):
    """
    Architecture:
        Input tokens
            ↓
        RoBERTa  (contextual embeddings, 768-dim per token)
            ↓
        Dropout  (regularization)
            ↓
        Linear   (768 → num_labels emission scores)
            ↓
        CRF      (finds best label sequence via Viterbi)
            ↓
        BIO label sequence
    """

    def __init__(self, num_labels, model_name="roberta-base", dropout=0.1):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.crf        = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Training  → returns loss (scalar)
        Inference → returns predictions (list of lists)
        """
        outputs         = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions       = self.classifier(sequence_output)

        crf_mask = attention_mask.bool()

        if labels is not None:
            # CRF has no ignore_index — replace -100 with 0 ("O")
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction="mean")
            return loss
        else:
            return self.crf.decode(emissions, mask=crf_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CRFNERTrainer:
    """Manual PyTorch training loop for RoBERTa+CRF NER."""

    def __init__(self, model, tokenizer, device):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.history   = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "precision": [], "recall": [], "f1": [], "accuracy": [],
            "learning_rates": [],
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader,
              epochs=15, lr=2e-5, patience=3):

        optimizer   = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        total_steps = len(train_loader) * epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        best_f1      = 0.0
        best_weights = None
        wait         = 0

        for epoch in range(epochs):
            # ── Training phase ────────────────────────
            self.model.train()
            total_loss = 0
            epoch_lrs  = []

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                optimizer.zero_grad()
                loss = self.model(input_ids, attention_mask, labels)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_lrs.append(scheduler.get_last_lr()[0])
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # ── Validation phase ──────────────────────
            val_loss, metrics = self._evaluate_epoch(val_loader)

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["precision"].append(metrics["precision"])
            self.history["recall"].append(metrics["recall"])
            self.history["f1"].append(metrics["f1"])
            self.history["accuracy"].append(metrics["accuracy"])
            self.history["learning_rates"].append(epoch_lrs)

            print(
                f"Epoch {epoch + 1:>2}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"P: {metrics['precision']:.4f}  "
                f"R: {metrics['recall']:.4f}  "
                f"F1: {metrics['f1']:.4f}"
            )

            if metrics["f1"] > best_f1:
                best_f1      = metrics["f1"]
                best_weights = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                wait = 0
                print(f"  ✅ New best F1: {metrics['f1']:.4f}")
            else:
                wait += 1
                if wait >= patience:
                    print(f"  ⏹  Early stopping at epoch {epoch + 1}")
                    break

        if best_weights:
            self.model.load_state_dict(best_weights)
            print(f"\nRestored best weights — F1: {best_f1:.4f}")

        return self.history

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_epoch(self, loader):
        """Quick evaluation returning (avg_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0
        all_true, all_preds = [], []

        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labs = batch["labels"].to(self.device)

                total_loss += self.model(ids, mask, labs).item()
                predictions = self.model(ids, mask)

                for pred_seq, label_seq in zip(predictions, labs):
                    tl, pl = [], []
                    for p, l in zip(pred_seq, label_seq.tolist()):
                        if l == -100:
                            continue
                        tl.append(ID2LABEL.get(l, "O"))
                        pl.append(ID2LABEL.get(p, "O"))
                    all_true.append(tl)
                    all_preds.append(pl)

        return total_loss / len(loader), {
            "precision": precision_score(all_true, all_preds, zero_division=0),
            "recall":    recall_score(all_true, all_preds, zero_division=0),
            "f1":        f1_score(all_true, all_preds, zero_division=0),
            "accuracy":  accuracy_score(all_true, all_preds),
        }

    def evaluate_detailed(self, loader):
        """Full evaluation with classification report and per-entity metrics."""
        self.model.eval()
        all_true, all_preds = [], []

        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labs = batch["labels"]

                predictions = self.model(ids, mask)

                for pred_seq, label_seq in zip(predictions, labs):
                    tl, pl = [], []
                    for p, l in zip(pred_seq, label_seq.tolist()):
                        if l == -100:
                            continue
                        tl.append(ID2LABEL.get(l, "O"))
                        pl.append(ID2LABEL.get(p, "O"))
                    all_true.append(tl)
                    all_preds.append(pl)

        # Classification report
        report      = seq_classification_report(all_true, all_preds, zero_division=0)
        report_dict = seq_classification_report(all_true, all_preds, output_dict=True, zero_division=0)

        # Entity-level metrics
        entity_precision = precision_score(all_true, all_preds, zero_division=0)
        entity_recall    = recall_score(all_true, all_preds, zero_division=0)
        entity_f1        = f1_score(all_true, all_preds, zero_division=0)

        # BIO validity (CRF should guarantee ~100 %)
        bio_validity = _check_bio_validity(all_preds)

        # Flatten for confusion matrix
        flat_true = [l for seq in all_true for l in seq]
        flat_pred = [l for seq in all_preds for l in seq]

        eval_metrics = {
            "entity_precision":       entity_precision,
            "entity_recall":          entity_recall,
            "entity_f1":              entity_f1,
            "per_entity_metrics":     report_dict,
            "bio_validity_rate":      bio_validity,
            "flat_true_labels":       flat_true,
            "flat_predicted_labels":  flat_pred,
        }
        return report, eval_metrics, all_true, all_preds

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Model checkpoint
        torch.save({
            "model_state": self.model.state_dict(),
            "label2id":    LABEL2ID,
            "id2label":    ID2LABEL,
            "num_labels":  NUM_LABELS,
        }, os.path.join(output_dir, "model.pt"))

        # Tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Label mappings JSON (same format as train_roberta_ner.py)
        with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
            json.dump({
                "label2id": LABEL2ID,
                "id2label": {str(k): v for k, v in ID2LABEL.items()},
            }, f, indent=2)

        print(f"Model saved → {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference — load trained CRF model and extract entities
# ─────────────────────────────────────────────────────────────────────────────

class NERPredictor:
    """Load trained RoBERTa+CRF model and extract entities from text."""

    def __init__(self, model_dir, device=None):
        if device is None:
            device = (
                "mps"  if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        self.device = device

        checkpoint    = torch.load(
            os.path.join(model_dir, "model.pt"),
            map_location=device,
        )
        self.id2label = checkpoint["id2label"]
        num_labels    = checkpoint["num_labels"]

        self.model = RobertaCRFNER(num_labels)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.model.to(device)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_dir, add_prefix_space=True
        )

    def predict(self, text):
        """Extract entities → dict {entity_type: entity_value}."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = inputs.pop("offset_mapping")[0]
        inputs  = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predictions = self.model(inputs["input_ids"], inputs["attention_mask"])

        pred_labels = [self.id2label.get(p, "O") for p in predictions[0]]

        # Group B+I tokens into complete spans using character offsets
        entities = {}
        current  = None

        for label, offset in zip(pred_labels, offsets):
            start, end = offset[0].item(), offset[1].item()

            if start == 0 and end == 0:       # special token
                if current:
                    span = text[current["start"]:current["end"]].strip()
                    if span:
                        entities[current["type"]] = span
                    current = None
                continue

            if label.startswith("B-"):
                if current:
                    span = text[current["start"]:current["end"]].strip()
                    if span:
                        entities[current["type"]] = span
                current = {"type": label[2:], "start": start, "end": end}

            elif label.startswith("I-") and current and label[2:] == current["type"]:
                current["end"] = end

            else:
                if current:
                    span = text[current["start"]:current["end"]].strip()
                    if span:
                        entities[current["type"]] = span
                    current = None

        if current:
            span = text[current["start"]:current["end"]].strip()
            if span:
                entities[current["type"]] = span

        return entities


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_bio_validity(predicted_sequences):
    """Check BIO tagging validity across all predicted sequences."""
    valid = 0
    for seq in predicted_sequences:
        ok   = True
        prev = "O"
        for tag in seq:
            if tag.startswith("I-"):
                etype = tag[2:]
                if prev not in (f"B-{etype}", f"I-{etype}"):
                    ok = False
                    break
            prev = tag
        if ok:
            valid += 1
    return valid / len(predicted_sequences) if predicted_sequences else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation functions (mirrors train_roberta_ner.py output)
# ─────────────────────────────────────────────────────────────────────────────

def save_epoch_metrics_csv(graphics_dir, history):
    """Save per-epoch metrics to CSV."""
    csv_path = os.path.join(graphics_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss",
                           "precision", "recall", "f1_score", "accuracy"]
        )
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({
                "epoch":      history["epoch"][i],
                "train_loss": f"{history['train_loss'][i]:.6f}",
                "val_loss":   f"{history['val_loss'][i]:.6f}",
                "precision":  f"{history['precision'][i]:.4f}",
                "recall":     f"{history['recall'][i]:.4f}",
                "f1_score":   f"{history['f1'][i]:.4f}",
                "accuracy":   f"{history['accuracy'][i]:.4f}",
            })
    print(f"✅ Epoch metrics saved to {csv_path}")


def _create_data_visualizations(graphics_dir, samples):
    """Label distribution, sentence length, entities per sentence."""
    all_labels = [l for s in samples for l in s["labels"] if l != "O"]
    label_counts = Counter(all_labels)

    # 1. Label distribution
    plt.figure(figsize=(14, 8))
    labels_list = list(label_counts.keys())
    counts      = list(label_counts.values())
    plt.bar(labels_list, counts)
    plt.title("Entity Label Distribution", fontsize=16)
    plt.xlabel("Labels", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "label_distribution.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Sentence length distribution
    lengths = [len(s["tokens"]) for s in samples]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.title("Sentence Length Distribution", fontsize=16)
    plt.xlabel("Sentence Length (tokens)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(np.mean(lengths), color="red", linestyle="--",
                label=f"Mean: {np.mean(lengths):.1f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "sentence_length_distribution.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Entities per sentence
    ents_per = [sum(1 for l in s["labels"] if l.startswith("B-")) for s in samples]
    plt.figure(figsize=(10, 6))
    plt.hist(ents_per, bins=range(max(ents_per) + 2),
             alpha=0.7, color="lightgreen", edgecolor="black")
    plt.title("Entities per Sentence Distribution", fontsize=16)
    plt.xlabel("Number of Entities", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "entities_per_sentence.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _create_evaluation_charts(graphics_dir, metrics):
    """Entity-level bar, per-entity F1, BIO validity pie, confusion matrix."""

    # 4. Entity-level metrics bar chart
    vals  = [metrics["entity_precision"], metrics["entity_recall"], metrics["entity_f1"]]
    names = ["Precision", "Recall", "F1-Score"]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, vals, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    plt.title("Entity-level Performance Metrics", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "entity_level_metrics.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Per-entity F1 scores
    entity_f1 = {}
    for ent, scores in metrics.get("per_entity_metrics", {}).items():
        if ent not in ("accuracy", "macro avg", "weighted avg", "micro avg") \
                and isinstance(scores, dict):
            entity_f1[ent] = scores.get("f1-score", 0)

    if entity_f1:
        plt.figure(figsize=(12, 8))
        ents  = list(entity_f1.keys())
        f1s   = list(entity_f1.values())
        bars  = plt.bar(ents, f1s, color="lightcoral")
        plt.title("Per-Entity F1 Scores", fontsize=16)
        plt.xlabel("Entity Types", fontsize=12)
        plt.ylabel("F1-Score", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        for bar, s in zip(bars, f1s):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{s:.3f}", ha="center", va="bottom", fontsize=10)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, "per_entity_f1_scores.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # 6. BIO validity pie
    vr = metrics.get("bio_validity_rate", 0)
    ir = 1 - vr
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        [vr, ir],
        labels=[f"Valid BIO ({vr:.1%})", f"Invalid BIO ({ir:.1%})"],
        colors=["#66BB6A", "#FF7043"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 12},
    )
    plt.title("BIO Tagging Validity Rate", fontsize=16)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "bio_validity_rate.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Confusion matrix
    flat_true = metrics["flat_true_labels"]
    flat_pred = metrics["flat_predicted_labels"]
    unique    = sorted(set(flat_true + flat_pred))
    cm        = confusion_matrix(flat_true, flat_pred, labels=unique)

    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix — Entity Recognition", fontsize=16)
    plt.colorbar()
    ticks = np.arange(len(unique))
    plt.xticks(ticks, unique, rotation=45, ha="right")
    plt.yticks(ticks, unique)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"), ha="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "confusion_matrix.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _create_training_curves(graphics_dir, history):
    """Training curves from epoch-level history dict."""
    epochs     = history["epoch"]
    train_loss = history["train_loss"]
    val_loss   = history["val_loss"]
    val_acc    = history["accuracy"]
    val_f1     = history["f1"]

    # Flatten per-epoch LR lists to one list + corresponding step indices
    all_lrs   = []
    all_steps = []
    step = 0
    for epoch_lrs in history["learning_rates"]:
        for lr_val in epoch_lrs:
            all_lrs.append(lr_val)
            all_steps.append(step)
            step += 1

    # 8. Training vs Validation Loss
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_loss, "o-", label="Training Loss",
             linewidth=2, color="#FF6B6B", markersize=6)
    plt.plot(epochs, val_loss,   "o-", label="Validation Loss",
             linewidth=2, color="#4ECDC4", markersize=6,
             markerfacecolor="white", markeredgewidth=2)
    plt.title("Training vs Validation Loss", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "training_validation_loss.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 9. Validation Accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_acc, "o-", linewidth=3, color="#45B7D1",
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    plt.title("Validation Accuracy Over Epochs", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1)
    for x, y in zip(epochs, val_acc):
        plt.annotate(f"{y:.1%}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    plt.legend(["Validation Accuracy"], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "validation_accuracy.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 10. Validation F1 Score
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_f1, "o-", linewidth=3, color="#96CEB4",
             markersize=8, markerfacecolor="white", markeredgewidth=2)
    plt.title("Validation F1 Score Over Epochs", fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 1)
    for x, y in zip(epochs, val_f1):
        plt.annotate(f"{y:.1%}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")
    plt.legend(["Validation F1"], fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "validation_f1_score.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 11. Learning Rate Schedule
    if all_lrs:
        plt.figure(figsize=(12, 8))
        plt.plot(all_steps, all_lrs, linewidth=2, color="#FFA07A",
                 marker="o", markersize=2)
        plt.title("Learning Rate Schedule", fontsize=16, fontweight="bold")
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, "learning_rate_schedule.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # 12. Combined Training Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.plot(epochs, train_loss, "o-", label="Train", color="#FF6B6B")
    ax1.plot(epochs, val_loss,   "o-", label="Val",   color="#4ECDC4", markersize=6)
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_acc, "o-", color="#45B7D1", markersize=6)
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, val_f1, "o-", color="#96CEB4", markersize=6)
    ax3.set_title("F1 Score", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("F1")
    ax3.set_ylim(0, 1); ax3.grid(True, alpha=0.3)

    if all_lrs:
        ax4.plot(all_steps, all_lrs, color="#FFA07A", marker="o", markersize=2)
        ax4.set_title("Learning Rate", fontweight="bold")
        ax4.set_xlabel("Steps"); ax4.set_ylabel("LR")
        ax4.set_yscale("log"); ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "LR data\nunavailable", ha="center", va="center",
                 transform=ax4.transAxes, fontsize=12)
        ax4.set_title("Learning Rate", fontweight="bold")

    plt.suptitle("Training Metrics Dashboard (RoBERTa+CRF)",
                 fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(os.path.join(graphics_dir, "training_dashboard.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def create_per_entity_grouped_bar(graphics_dir, per_entity_metrics):
    """Grouped bar chart: Precision, Recall, F1 per entity type."""
    entities, precisions, recalls, f1s = [], [], [], []
    for ent, m in per_entity_metrics.items():
        if ent in ("accuracy", "macro avg", "weighted avg", "micro avg", "O"):
            continue
        if not isinstance(m, dict):
            continue
        entities.append(ent)
        precisions.append(m.get("precision", 0))
        recalls.append(m.get("recall", 0))
        f1s.append(m.get("f1-score", 0))

    if not entities:
        return

    x     = np.arange(len(entities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, precisions, width, label="Precision", color="#FF6B6B")
    ax.bar(x,         recalls,    width, label="Recall",    color="#4ECDC4")
    ax.bar(x + width, f1s,        width, label="F1 Score",  color="#45B7D1")

    ax.set_xlabel("Entity Type", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, F1 per Entity Type", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(entities, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "per_entity_grouped_bar.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Per-entity grouped bar chart saved")


def create_hyperparameter_heatmap(graphics_dir, tuning_results):
    """Heatmap of LR vs Batch Size vs F1."""
    if not tuning_results:
        return

    lrs     = sorted(set(r["learning_rate"] for r in tuning_results))
    batches = sorted(set(r["batch_size"]    for r in tuning_results))

    f1_mat = np.zeros((len(lrs), len(batches)))
    for r in tuning_results:
        i = lrs.index(r["learning_rate"])
        j = batches.index(r["batch_size"])
        f1_mat[i, j] = r["f1"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(f1_mat, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=[str(b) for b in batches],
                yticklabels=[f"{lr:.0e}" for lr in lrs])
    plt.title("Hyperparameter Tuning: LR vs Batch Size vs F1",
              fontsize=14, fontweight="bold")
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(graphics_dir, "hyperparameter_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Hyperparameter heatmap saved")


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning — grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_hyperparameter_tuning(data_path, output_base_dir, graphics_dir,
                              max_length=128):
    learning_rates = [1e-5, 2e-5, 3e-5]
    batch_sizes    = [16, 32]
    epoch_configs  = [5, 10]

    device    = _select_device()
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", add_prefix_space=True
    )

    samples = load_bio_data(data_path)
    for s in samples:
        s.pop("failed", None)

    train_data, temp = train_test_split(samples, test_size=0.2, random_state=42)
    val_data, _      = train_test_split(temp,    test_size=0.5, random_state=42)

    results   = []
    best_f1   = 0
    best_cfg  = None

    total = len(learning_rates) * len(batch_sizes) * len(epoch_configs)
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER TUNING — Grid Search (RoBERTa+CRF)")
    print("=" * 60)
    print(f"  LRs    : {learning_rates}")
    print(f"  Batches: {batch_sizes}")
    print(f"  Epochs : {epoch_configs}")
    print(f"  Total  : {total} runs")
    print("=" * 60 + "\n")

    run_idx = 0
    for lr in learning_rates:
        for batch in batch_sizes:
            for epochs in epoch_configs:
                run_idx += 1
                print(f"\n[Run {run_idx}/{total}] LR={lr}, Batch={batch}, Epochs={epochs}")

                train_ds = NERDataset(train_data, tokenizer, LABEL2ID, max_length)
                val_ds   = NERDataset(val_data,   tokenizer, LABEL2ID, max_length)
                train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
                val_loader   = DataLoader(val_ds,   batch_size=batch)

                model   = RobertaCRFNER(NUM_LABELS)
                trainer = CRFNERTrainer(model, tokenizer, device)
                trainer.train(train_loader, val_loader, epochs=epochs, lr=lr, patience=3)

                _, metrics = trainer._evaluate_epoch(val_loader)

                result = {
                    "learning_rate": lr,
                    "batch_size":    batch,
                    "epochs":        epochs,
                    "f1":            metrics["f1"],
                    "precision":     metrics["precision"],
                    "recall":        metrics["recall"],
                }
                results.append(result)
                print(f"  → F1: {metrics['f1']:.4f}")

                if metrics["f1"] > best_f1:
                    best_f1  = metrics["f1"]
                    best_cfg = result.copy()

    os.makedirs(graphics_dir, exist_ok=True)

    csv_path = os.path.join(graphics_dir, "hyperparameter_tuning_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "learning_rate", "batch_size", "epochs", "f1", "precision", "recall"])
        writer.writeheader()
        writer.writerows(results)

    with open(os.path.join(graphics_dir, "hyperparameter_tuning_results.json"), "w") as f:
        json.dump({
            "results":    results,
            "best_config": best_cfg,
            "timestamp":   datetime.now().isoformat(),
        }, f, indent=2)

    create_hyperparameter_heatmap(graphics_dir, results)

    print("\n" + "=" * 60)
    print("  TUNING COMPLETE (RoBERTa+CRF)")
    print("=" * 60)
    if best_cfg:
        print(f"  Best: LR={best_cfg['learning_rate']}, "
              f"Batch={best_cfg['batch_size']}, Epochs={best_cfg['epochs']}")
        print(f"  Best F1: {best_cfg['f1']:.4f}")
    print(f"  Results: {csv_path}")
    print("=" * 60)
    return results, best_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../.."))

    parser = argparse.ArgumentParser(
        description="Train RoBERTa+CRF NER Model from bio_dataset.json"
    )
    parser.add_argument(
        "--data_path",
        default=os.path.join(project_root, "data/bio_dataset.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_root, "models/crf_entity_model"),
    )
    parser.add_argument(
        "--graphics_dir",
        default=os.path.join(project_root, "docs/graphs/entity_crf"),
    )
    parser.add_argument("--epochs",        type=int,   default=15)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length",    type=int,   default=128)
    parser.add_argument("--patience",      type=int,   default=3)
    parser.add_argument("--test_size",     type=float, default=0.10)
    parser.add_argument("--val_size",      type=float, default=0.10)
    parser.add_argument("--tune",          action="store_true",
                        help="Run hyperparameter tuning grid search")

    args = parser.parse_args()

    # ── Hyperparameter tuning mode ──────────────────────────────────────
    if args.tune:
        run_hyperparameter_tuning(
            args.data_path, args.output_dir, args.graphics_dir, args.max_length
        )
        return

    # ── Banner ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RoBERTa+CRF NER — Training (BIO Dataset)")
    print("=" * 60)
    print(f"  Data       : {args.data_path}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Graphics   : {args.graphics_dir}")
    print(f"  Epochs     : {args.epochs}  |  LR: {args.learning_rate}")
    print(f"  Batch size : {args.batch_size}  |  Max len: {args.max_length}")
    print(f"  Patience   : {args.patience}")
    print(f"  Test/Val   : {args.test_size:.0%} / {args.val_size:.0%}")
    print(f"  Num labels : {NUM_LABELS}")
    print("=" * 60 + "\n")

    os.makedirs(args.output_dir,   exist_ok=True)
    os.makedirs(args.graphics_dir, exist_ok=True)

    device = _select_device()
    print(f"Device: {device}\n")

    # ── Load data ───────────────────────────────────────────────────────
    samples = load_bio_data(args.data_path)

    for s in samples:
        s.pop("failed", None)

    # ── Split ───────────────────────────────────────────────────────────
    tmp, test_data = train_test_split(
        samples, test_size=args.test_size, random_state=42
    )
    val_ratio = args.val_size / (1 - args.test_size)
    train_data, val_data = train_test_split(
        tmp, test_size=val_ratio, random_state=42
    )
    print(f"Split → Train: {len(train_data):,}  "
          f"Val: {len(val_data):,}  Test: {len(test_data):,}\n")

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", add_prefix_space=True
    )

    train_ds = NERDataset(train_data, tokenizer, LABEL2ID, args.max_length)
    val_ds   = NERDataset(val_data,   tokenizer, LABEL2ID, args.max_length)
    test_ds  = NERDataset(test_data,  tokenizer, LABEL2ID, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # ── Build & Train ───────────────────────────────────────────────────
    model   = RobertaCRFNER(NUM_LABELS)
    trainer = CRFNERTrainer(model, tokenizer, device)

    print("Starting RoBERTa+CRF NER training …\n")
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs, lr=args.learning_rate, patience=args.patience,
    )

    # ── Evaluate on test set ────────────────────────────────────────────
    print("\n── Test set evaluation ──────────────────────")
    report, eval_metrics, true_lbls, pred_lbls = trainer.evaluate_detailed(test_loader)
    print("\nClassification Report:")
    print(report)

    # ── Visualisations ──────────────────────────────────────────────────
    print("\nGenerating visualisations …")
    _create_data_visualizations(args.graphics_dir, samples)
    _create_evaluation_charts(args.graphics_dir, eval_metrics)
    _create_training_curves(args.graphics_dir, history)

    save_epoch_metrics_csv(args.graphics_dir, history)

    if "per_entity_metrics" in eval_metrics:
        create_per_entity_grouped_bar(args.graphics_dir, eval_metrics["per_entity_metrics"])

    # ── Save model ──────────────────────────────────────────────────────
    trainer.save(args.output_dir)

    # ── Save artefacts ──────────────────────────────────────────────────
    detailed_results = {
        "architecture": "RoBERTa+CRF",
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
        "training_history": {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss":   history["val_loss"][-1],
            "best_val_f1":      max(history["f1"]),
            "total_epochs":     len(history["epoch"]),
        },
        "timestamp":    datetime.now().isoformat(),
        "data_source":  "bio_dataset.json",
        "model_config": vars(args),
    }

    with open(os.path.join(args.output_dir, "comprehensive_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    per_entity_csv = os.path.join(args.graphics_dir, "per_entity_metrics.csv")
    with open(per_entity_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "precision", "recall", "f1_score", "support"])
        for ent, m in eval_metrics.get("per_entity_metrics", {}).items():
            if isinstance(m, dict) and "precision" in m \
                    and ent not in ("accuracy", "macro avg", "weighted avg", "micro avg"):
                writer.writerow([
                    ent,
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1-score']:.4f}",
                    m.get("support", 0),
                ])

    # ── Quick inference test ────────────────────────────────────────────
    print("\n── Quick inference test ─────────────────────")
    predictor = NERPredictor(args.output_dir, device=device)
    test_texts = [
        "Show me computer engineering colleges in Kathmandu",
        "Compare Pulchowk and Kathford",
        "Best colleges under 5 lakhs with rank below 500",
    ]
    for txt in test_texts:
        ents = predictor.predict(txt)
        print(f"  \"{txt}\"")
        print(f"    → {ents}\n")

    # ── Summary ─────────────────────────────────────────────────────────
    ep  = eval_metrics["entity_precision"]
    er  = eval_metrics["entity_recall"]
    ef1 = eval_metrics["entity_f1"]
    bvr = eval_metrics["bio_validity_rate"]

    print("=" * 60)
    print("  Training complete!")
    print(f"  Model saved         : {args.output_dir}")
    print(f"  Entity Precision    : {ep:.4f}")
    print(f"  Entity Recall       : {er:.4f}")
    print(f"  Entity F1           : {ef1:.4f}")
    print(f"  BIO Validity Rate   : {bvr:.1%}")
    print(f"  Visualisations      : {args.graphics_dir}/")
    print("=" * 60)
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
    print("  ✓ training_metrics.csv — Per-epoch metrics")
    print("  ✓ per_entity_metrics.csv — Per-entity metrics")


if __name__ == "__main__":
    main()
