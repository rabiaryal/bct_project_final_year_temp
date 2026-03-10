"""
Data loading utilities for the NER model.

Supports two dataset formats:
  1. BIO dataset (bio_dataset.json) — pre-tokenized tokens + BIO labels per sample
  2. Legacy JSON (intent_entity.json) — text + entities dict, BIO conversion done here
"""
import json
import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    """PyTorch Dataset for BIO-tagged NER samples.

    Each sample is a dict with ``tokens`` (list[str]) and ``labels``
    (list[str] in BIO format).  Subword alignment is handled here:
    first subword gets the real label, continuation subwords get ``-100``
    (ignored in loss) except B- → I- promotion for entity continuations.
    """

    def __init__(self, samples, tokenizer, label2id, max_len=128):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.label2id  = label2id
        self.max_len   = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        labels = sample["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids       = encoding.word_ids()
        aligned_labels = []
        prev_word_id   = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != prev_word_id:
                label = labels[word_id] if word_id < len(labels) else "O"
                aligned_labels.append(self.label2id.get(label, 0))
            else:
                label = labels[word_id] if word_id < len(labels) else "O"
                if label.startswith("B-"):
                    cont_label = "I-" + label[2:]
                    aligned_labels.append(self.label2id.get(cont_label, -100))
                elif label.startswith("I-"):
                    aligned_labels.append(self.label2id.get(label, -100))
                else:
                    aligned_labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_bio_data(json_path):
    """Load pre-tokenized BIO dataset (bio_dataset.json).

    Returns a list of sample dicts, each with ``tokens`` and ``labels``.
    """
    print(f"Loading BIO data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Loaded {len(samples)} BIO-tagged samples")
    return samples


def load_data_from_json(json_path):
    """Load NER data from legacy JSON (text + entities dict) and convert to BIO."""
    print(f"Loading data from {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts  = []
    labels = []

    sentences_data = data.get("sentences", data if isinstance(data, list) else [])

    for sentence_data in sentences_data:
        if not isinstance(sentence_data, dict):
            continue

        text     = sentence_data.get("text", "")
        entities = sentence_data.get("entities", {})

        clean_text = text.replace("**", "")
        words      = clean_text.split()
        word_labels = ["O"] * len(words)

        entity_items = []
        for e_type, e_vals in entities.items():
            if isinstance(e_vals, list):
                for v in e_vals:
                    entity_items.append((e_type, v))
            else:
                entity_items.append((e_type, e_vals))

        entity_items.sort(key=lambda x: len(str(x[1]).split()), reverse=True)

        for entity_type, entity_value in entity_items:
            if not entity_value or not isinstance(entity_value, str):
                continue
            entity_words = entity_value.split()
            entity_len   = len(entity_words)
            for i in range(len(words) - entity_len + 1):
                if words[i:i + entity_len] == entity_words:
                    if all(lbl == "O" for lbl in word_labels[i:i + entity_len]):
                        word_labels[i] = f"B-{entity_type}"
                        for j in range(1, entity_len):
                            word_labels[i + j] = f"I-{entity_type}"

        if words:
            texts.append(" ".join(words))
            labels.append(word_labels)

    print(f"Successfully loaded {len(texts)} sentences with multi-entity support.")
    return texts, labels
