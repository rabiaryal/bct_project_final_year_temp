"""
NLU Test Script
Independently tests the BERT Intent classifier and RoBERTa Entity (NER) model.

Run directly (recommended – avoids full app import chain):
    python backend/app/nlu/test_nlu.py
    python backend/app/nlu/test_nlu.py "show me BCA colleges in Kathmandu"
    python backend/app/nlu/test_nlu.py --intent-only
    python backend/app/nlu/test_nlu.py --entity-only
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

# ── Make this script importable without triggering app/__init__.py ────────────
# Insert the backend/ directory into sys.path so that direct `from transformers`
# and model-path resolution work without needing the full app package.
_THIS_FILE = Path(__file__).resolve()
_BACKEND_DIR = _THIS_FILE.parents[2]          # .../backend/
_PROJECT_DIR = _THIS_FILE.parents[3]          # .../bct_final_year_project/
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ─────────────────────────────────────────────
# Paths (relative to this file → models/ folder)
# ─────────────────────────────────────────────
INTENT_MODEL_PATH = _PROJECT_DIR / "models" / "bert_intent_model"
ENTITY_MODEL_PATH = _PROJECT_DIR / "models" / "roberta_entity_model"


# ══════════════════════════════════════════════
#  INTENT PREDICTOR
# ══════════════════════════════════════════════
class IntentPredictor:
    def __init__(self, model_path: str):
        from transformers import BertTokenizer, BertForSequenceClassification

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"[Intent] Loading model from: {model_path}")
        print(f"[Intent] Device: {self.device}")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        label_path = os.path.join(model_path, "label_mapping.json")
        with open(label_path, "r") as f:
            mapping = json.load(f)
        self.id_to_label = mapping["id_to_label"]

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()

        intent = self.id_to_label.get(str(pred_id), "unknown")

        top_k = torch.topk(probs, min(5, len(probs)))
        top_predictions = [
            (self.id_to_label.get(str(i.item()), "unknown"), round(p.item(), 4))
            for i, p in zip(top_k.indices, top_k.values)
        ]
        return intent, round(confidence, 4), top_predictions


# ══════════════════════════════════════════════
#  ENTITY PREDICTOR
# ══════════════════════════════════════════════
class EntityPredictor:
    def __init__(self, model_path: str):
        from transformers import RobertaTokenizerFast, RobertaForTokenClassification

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"[Entity]  Loading model from: {model_path}")
        print(f"[Entity]  Device: {self.device}")

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = RobertaForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        label_path = os.path.join(model_path, "label_mappings.json")
        with open(label_path, "r") as f:
            mapping = json.load(f)
        self.id2label = mapping["id2label"]

    def predict(self, text: str):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_ids = logits.argmax(dim=-1)[0].tolist()
            scores = probs[0].max(dim=-1).values.tolist()

        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        # ── BIO → entity spans ──────────────────
        entities = []
        current = None

        for token, pred_id, score, (start, end) in zip(tokens, pred_ids, scores, offset_mapping):
            # Skip special tokens
            if start == end:
                current = None
                continue

            label = self.id2label.get(str(pred_id), "O")

            if label.startswith("B-"):
                if current:
                    current["text"] = text[current["start"]:current["end"]].strip()
                    entities.append(current)
                current = {"type": label[2:], "start": start, "end": end, "score": round(score, 4)}

            elif label.startswith("I-") and current and label[2:] == current["type"]:
                current["end"] = end
                current["score"] = round((current["score"] + score) / 2, 4)

            else:
                if current:
                    current["text"] = text[current["start"]:current["end"]].strip()
                    entities.append(current)
                current = None

        if current:
            current["text"] = text[current["start"]:current["end"]].strip()
            entities.append(current)

        return entities


# ══════════════════════════════════════════════
#  PRETTY PRINT HELPERS
# ══════════════════════════════════════════════
def print_intent_results(text: str, intent: str, confidence: float, top_preds: list):
    print("\n" + "═" * 60)
    print("  INTENT PREDICTION")
    print("═" * 60)
    print(f"  Input      : {text}")
    print(f"  Intent     : {intent}")
    print(f"  Confidence : {confidence:.2%}")
    print("\n  Top-5 predictions:")
    for rank, (label, prob) in enumerate(top_preds, 1):
        bar = "█" * int(prob * 30)
        print(f"    {rank}. {label:<35} {prob:.4f}  {bar}")
    print("═" * 60)


def print_entity_results(text: str, entities: list):
    print("\n" + "═" * 60)
    print("  ENTITY PREDICTION")
    print("═" * 60)
    print(f"  Input   : {text}")
    if not entities:
        print("  Entities: (none detected)")
    else:
        print(f"  Entities detected: {len(entities)}")
        print()
        for ent in entities:
            print(f"    ▸ [{ent['type']}]  \"{ent['text']}\"  (confidence: {ent['score']:.2%})")
    print("═" * 60)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════
DEFAULT_TEST_SENTENCES = [
    # Greeting / small-talk
    "Hello, can you help me find a college?",
    "Thank you so much for the information!",
    "Yes, that sounds good.",
    "No, that is not what I was looking for.",
    "Can you clarify what you mean by affiliated?",
    "Goodbye, have a great day!",

    # Location-based search
    "Find me a private BE Civil college in Lalitpur affiliated with TU.",
    "Show me government colleges for BME in Dharan.",
    "List the Purbanchal University affiliated colleges in Biratnagar.",

    # Program-based search
    "I am looking for a Pokhara University affiliated college for BE Software.",
    "Which departments does Western Region Campus have?",
    "What master's programs does Pulchowk Campus offer?",

    # Fee-based search
    "Give me colleges offering B.Arch in Lalitpur with fees less than 10 Lakhs.",
    "Find BE Electrical colleges in Kathmandu with high rating and low fee.",
    "List government colleges for BE Electrical with low fee.",

    # Seat-based search
    "How many seats are available for BE Mechanical at Thapathali Campus?",
    "Are there available seats for BE Electronics at Western Region Campus?",

    # College type search
    "Is Kathmandu Engineering College a private institution?",
    "Show me all government engineering colleges in Pokhara.",

    # Recommend college
    "Find a high-rated private college for BE Software in Kathmandu.",
    "Recommend a BE Civil college near Thapathali with a good rating.",

    # Fee info
    "What is the fee structure for B.Arch at Nepal Engineering College?",
    "What's the full fee for the BE Electronics program at Lalitpur Engineering College?",

    # College general info
    "What are the lab facilities at Universal Engineering & Science College for the Civil Department?",
    "Is Gandaki College of Engineering a TU or PU affiliated college?",

    # Program info
    "Show me details on the BE Electrical program at Eastern Region Campus.",

    # Contact info
    "What is the contact number for Kathford International College?",
    "I need the website for Kathmandu Engineering College.",

    # Location of college
    "Where is Khwopa Engineering College (PU) located?",
    "Give me the address of Chitwan Engineering Campus.",

    # Admission info
    "When is the application deadline for BE Computer at Western Region Campus?",
    "I need information on the admission process for BE Civil at Khwopa College of Engineering.",

    # Admission process
    "What are the steps to apply for admission at Pulchowk Campus?",

    # Hostel availability
    "Show me colleges with hostel facility for the BE Computer program.",
    "Does Nepal Engineering College have a hostel facility for females?",

    # Scholarship info
    "Does Pokhara Engineering College offer scholarships for B.Arch students?",
    "Show me affiliated colleges with scholarship programs.",

    # Pass percentage info
    "What is the pass percentage for BE Civil students at Thapathali Campus?",
    "Which college has the highest pass rate for BE Computer graduates?",
]


def main():
    parser = argparse.ArgumentParser(description="Test NLU intent and entity models")
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to analyse (if omitted, default test sentences are used)",
    )
    parser.add_argument(
        "--intent-only", action="store_true", help="Run only the intent model"
    )
    parser.add_argument(
        "--entity-only", action="store_true", help="Run only the entity model"
    )
    args = parser.parse_args()

    sentences = [" ".join(args.text)] if args.text else DEFAULT_TEST_SENTENCES

    # ── Load models ──────────────────────────
    intent_predictor = None
    entity_predictor = None

    if not args.entity_only:
        try:
            intent_predictor = IntentPredictor(str(INTENT_MODEL_PATH))
        except Exception as exc:
            print(f"[ERROR] Could not load intent model: {exc}")

    if not args.intent_only:
        try:
            entity_predictor = EntityPredictor(str(ENTITY_MODEL_PATH))
        except Exception as exc:
            print(f"[ERROR] Could not load entity model: {exc}")

    print(f"\n{'─'*60}")
    print(f"  Running NLU tests on {len(sentences)} sentence(s)")
    print(f"{'─'*60}")

    # ── Run predictions ───────────────────────
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if intent_predictor:
            intent, confidence, top_preds = intent_predictor.predict(sentence)
            print_intent_results(sentence, intent, confidence, top_preds)

        if entity_predictor:
            entities = entity_predictor.predict(sentence)
            print_entity_results(sentence, entities)

        print()


if __name__ == "__main__":
    main()
