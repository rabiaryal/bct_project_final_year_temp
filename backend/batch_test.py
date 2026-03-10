"""
batch_test.py — Batch evaluation pipeline for NLU + dialogue system
====================================================================

Feed a JSON file with test cases and get a detailed output JSON with
predicted vs actual intent/entities plus the system's full response.

Metrics:
  - Intent accuracy  (exact match)
  - Entity type accuracy  (same set of entity types detected)
  - Entity value accuracy (type match + value substring match)
  - Per-entity-type precision / recall
  - Per-case breakdown

INPUT format  (test_input.json):
[
  {
    "text": "Show me colleges in Kathmandu",
    "actual_intent": "search_college",
    "actual_entities": {"LOCATION": "Kathmandu"}
  },
  ...
]

OUTPUT format  (test_output.json):
[
  {
    "text": "Show me colleges in Kathmandu",
    "actual_intent": "search_college",
    "actual_entities": {"LOCATION": "Kathmandu"},
    "predicted_intent": "search_college",
    "predicted_entities": {"LOCATION": "Kathmandu"},
    "intent_match": true,
    "entity_type_match": true,
    "entity_value_match": true,
    "system_response": "Found 4 result(s): ..."
  },
  ...
]

Usage:
    cd backend/
    conda activate bctproject
    python batch_test.py                              # uses default test_input.json
    python batch_test.py -i my_tests.json             # custom input file
    python batch_test.py -i my_tests.json -o out.json # custom output file
"""

import sys
import os
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent  # backend/
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from app.dialogue_manager import dialogue_manager
from app.schemas import ChatRequest

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "bold": "\033[1m", "reset": "\033[0m",
    "cyan": "\033[96m", "green": "\033[92m",
    "yellow": "\033[93m", "red": "\033[91m",
    "dim": "\033[2m",
}

def c(clr, txt):
    return f"{C[clr]}{txt}{C['reset']}"


# ── Entity matching ───────────────────────────────────────────────────────────

def _normalize(v: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(str(v).lower().split())


def _value_match(actual: str, predicted: str) -> bool:
    """Fuzzy value match: substring containment in either direction."""
    a, p = _normalize(actual), _normalize(predicted)
    return a == p or a in p or p in a


def compare_entities(actual: dict, predicted: dict):
    """
    Compare entity dicts and return:
      - type_match:  set of predicted types == set of actual types
      - value_match: type_match AND every value matches (fuzzy)
      - per_type:    {type: {actual, predicted, type_hit, value_hit}}
    """
    actual_types    = set(actual.keys())
    predicted_types = set(predicted.keys())

    type_match  = actual_types == predicted_types
    all_values  = True
    per_type    = {}

    all_types = actual_types | predicted_types
    for etype in sorted(all_types):
        a_val = actual.get(etype)
        p_val = predicted.get(etype)
        t_hit = (etype in actual_types) and (etype in predicted_types)
        v_hit = t_hit and _value_match(a_val, p_val) if (a_val and p_val) else False

        if not v_hit:
            all_values = False

        per_type[etype] = {
            "actual":    a_val,
            "predicted": p_val,
            "type_hit":  t_hit,
            "value_hit": v_hit,
        }

    value_match = type_match and all_values
    return type_match, value_match, per_type


# ── Core runner ───────────────────────────────────────────────────────────────

async def run_batch(test_cases: list) -> list:
    """Run all test cases through the dialogue manager and collect results."""

    print(f"\n{c('bold', '='*65)}")
    print(f"  Batch Test — {len(test_cases)} case(s)  |  Model: RoBERTa+CRF")
    print(f"{c('bold', '='*65)}\n")

    await dialogue_manager.initialize()

    results       = []
    n_intent      = 0
    n_type_match  = 0
    n_value_match = 0

    # Per-entity-type tracking
    tp = defaultdict(int)   # true positive (type detected where expected)
    fp = defaultdict(int)   # false positive (type detected where not expected)
    fn = defaultdict(int)   # false negative (type expected but not detected)

    for i, case in enumerate(test_cases, 1):
        text            = case["text"]
        actual_intent   = case.get("actual_intent", "")
        actual_entities = case.get("actual_entities", {})

        # Each test case gets its own session to prevent context bleed
        session_id = f"batch-test-{i}-{datetime.now().strftime('%H%M%S%f')}"
        request = ChatRequest(message=text, session_id=session_id)

        response = await dialogue_manager.process_message(request)

        predicted_intent   = response.intent
        predicted_entities = response.entities
        system_response    = response.message

        # ── Match checks ──────────────────────────────────────────────
        intent_match = predicted_intent == actual_intent
        type_match, value_match, per_type = compare_entities(
            actual_entities, predicted_entities
        )

        if intent_match:
            n_intent += 1
        if type_match:
            n_type_match += 1
        if value_match:
            n_value_match += 1

        # Per-entity-type stats
        for etype in set(actual_entities) | set(predicted_entities):
            in_actual = etype in actual_entities
            in_pred   = etype in predicted_entities
            if in_actual and in_pred:
                tp[etype] += 1
            elif in_pred and not in_actual:
                fp[etype] += 1
            elif in_actual and not in_pred:
                fn[etype] += 1

        # ── Print per-case summary ────────────────────────────────────
        i_icon = c("green", "✓") if intent_match    else c("red", "✗")
        t_icon = c("green", "✓") if type_match      else c("red", "✗")
        v_icon = c("green", "✓") if value_match     else c("red", "✗")

        print(f"[{i:>3}/{len(test_cases)}] {c('cyan', text[:75])}")
        print(f"  Intent  {i_icon}  actual={actual_intent}  pred={predicted_intent}")
        print(f"  E-Type  {t_icon}  actual={set(actual_entities.keys())}  pred={set(predicted_entities.keys())}")
        print(f"  E-Val   {v_icon}  actual={actual_entities}")
        print(f"               pred={predicted_entities}")

        # Show per-type detail for mismatches
        if not value_match:
            for et, info in per_type.items():
                mark = c("green", "✓") if info["value_hit"] else (
                    c("yellow", "~") if info["type_hit"] else c("red", "✗"))
                print(f"          {mark} {et}: {info['actual']} → {info['predicted']}")
        print()

        results.append({
            "text":               text,
            "actual_intent":      actual_intent,
            "actual_entities":    actual_entities,
            "predicted_intent":   predicted_intent,
            "predicted_entities": predicted_entities,
            "intent_match":       intent_match,
            "entity_type_match":  type_match,
            "entity_value_match": value_match,
            "system_response":    system_response,
        })

    await dialogue_manager.shutdown()

    # ── Summary ───────────────────────────────────────────────────────
    total = len(test_cases)
    print(f"{c('bold', '='*65)}")
    print(f"  RESULTS  ({total} cases)")
    print(f"  {'─'*61}")
    print(f"  Intent accuracy      : {n_intent}/{total}  ({n_intent/total:.1%})")
    print(f"  Entity type accuracy : {n_type_match}/{total}  ({n_type_match/total:.1%})")
    print(f"  Entity value accuracy: {n_value_match}/{total}  ({n_value_match/total:.1%})")
    print(f"  {'─'*61}")

    # Per-entity-type precision/recall
    all_types = sorted(set(tp) | set(fp) | set(fn))
    if all_types:
        print(f"  {'Entity Type':<20} {'Prec':>6} {'Recall':>7} {'TP':>4} {'FP':>4} {'FN':>4}")
        print(f"  {'─'*52}")
        for et in all_types:
            prec = tp[et] / (tp[et] + fp[et]) if (tp[et] + fp[et]) else 0
            rec  = tp[et] / (tp[et] + fn[et]) if (tp[et] + fn[et]) else 0
            print(f"  {et:<20} {prec:>5.1%} {rec:>6.1%} {tp[et]:>4} {fp[et]:>4} {fn[et]:>4}")
        print(f"  {'─'*52}")

    print(f"{c('bold', '='*65)}\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch NLU + dialogue test")
    parser.add_argument(
        "-i", "--input",
        default=os.path.join(_THIS_DIR, "test_input.json"),
        help="Path to input JSON test file (default: backend/test_input.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output JSON results (default: test_output_<timestamp>.json)",
    )
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────
    with open(args.input, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Loaded {len(test_cases)} test cases from {args.input}")

    # ── Run ───────────────────────────────────────────────────────────
    results = asyncio.run(run_batch(test_cases))

    # ── Save output ───────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(_THIS_DIR, f"test_output_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
