"""RoBERTa + CRF Named Entity Recognition – Inference"""

import os
import json
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import RobertaModel, RobertaTokenizerFast
from typing import List, Dict, Any, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Model architecture (must match train_with_crf.py) ─────────────────────
class _RobertaCRFNER(nn.Module):
    def __init__(self, num_labels, dropout=0.1):
        super().__init__()
        self.roberta    = RobertaModel.from_pretrained("roberta-base")
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.crf        = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs         = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions       = self.classifier(sequence_output)
        crf_mask        = attention_mask.bool()

        if labels is not None:
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            return -self.crf(emissions, crf_labels, mask=crf_mask, reduction="mean")
        return self.crf.decode(emissions, mask=crf_mask)


class RoBERTaEntityExtractor:
    """RoBERTa+CRF entity extractor (drop-in replacement)."""

    def __init__(self, model_path: str = None):
        from app.utils.config import config

        self.model_path = model_path or config.models.entity_model_path
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading RoBERTa+CRF NER model from {self.model_path}")

            # Load checkpoint on CPU first to avoid MPS unaligned blit errors
            checkpoint = torch.load(
                os.path.join(self.model_path, "model.pt"),
                map_location="cpu",
            )
            self.id2label = {int(k): v for k, v in checkpoint["id2label"].items()}
            num_labels    = checkpoint["num_labels"]

            self.model = _RobertaCRFNER(num_labels)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                self.model_path, add_prefix_space=True
            )

            logger.info(
                f"RoBERTa+CRF NER model loaded with {num_labels} labels on {self.device}"
            )
        except Exception as exc:
            logger.error(f"Failed to load RoBERTa+CRF NER model: {exc}")
            raise

    async def predict(self, text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract named entities from *text*.

        Returns:
            (entities, metadata)
            entities – list of dicts: {type, value, confidence}
            metadata – model info dict
        """
        try:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
                return_offsets_mapping=True,
            )
            offset_mapping = encoding.pop("offset_mapping")[0].tolist()
            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                predictions = self.model(input_ids, attention_mask)  # CRF Viterbi decode

            pred_labels = [self.id2label.get(p, "O") for p in predictions[0]]

            entities = self._decode_bio(text, pred_labels, offset_mapping)

            metadata = {
                "model": "roberta-crf",
                "device": str(self.device),
                "entity_count": len(entities),
            }
            return entities, metadata

        except Exception as exc:
            logger.error(f"Entity extraction error: {exc}")
            return [], {"error": str(exc)}

    def _decode_bio(
        self,
        text: str,
        pred_labels: list,
        offset_mapping: list,
    ) -> List[Dict[str, Any]]:
        """Convert CRF BIO label sequence to entity spans via character offsets."""
        entities = []
        current: Dict[str, Any] = {}

        def _flush():
            nonlocal current
            if current:
                span = text[current["_s"]:current["_e"]].strip()
                if span:
                    current["value"] = span
                    entities.append(
                        {k: v for k, v in current.items() if not k.startswith("_")}
                    )
                current = {}

        for label, (start, end) in zip(pred_labels, offset_mapping):
            if start == end:          # special token ([CLS], [SEP], [PAD])
                _flush()
                continue

            if label.startswith("B-"):
                _flush()
                current = {
                    "type": label[2:],
                    "confidence": 0.95,
                    "_s": start,
                    "_e": end,
                }
            elif label.startswith("I-") and current.get("type") == label[2:]:
                current["_e"] = end
            else:
                _flush()

        _flush()
        return entities
