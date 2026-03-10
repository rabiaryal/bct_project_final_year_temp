"""
Recommend Handler — recommend_with_constraints intent  (HYBRID)

Trigger: "suggest colleges for BE Civil with rank 3500 and budget 7 lakhs"

HYBRID FLOW:
  1. Check missing slots → ask if needed
  2. MongoDB hard filter → up to 15 candidates
  3. Python scoring (RecommendScorer) → score each candidate
  4. Rerank → top 3
  5. Format with constraint-focused tone
"""

from typing import Dict, List, Any
import logging

from app.core.slot_filler import get_missing_slot
from app.core.query_builder import build_candidate_pipeline, deduplicate
from app.core.scorer import RecommendScorer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# FORMAT — constraint tone: "matching your constraints"
# ─────────────────────────────────────────────────────

def _format_response(top_colleges: List[Dict[str, Any]], slots: Dict[str, Any]) -> str:
    if not top_colleges:
        msg = "Sorry, no colleges match your constraints.\n\n"
        msg += "Possible reasons:\n"
        if slots.get("rank"):
            msg += f"  • Rank {slots['rank']} may be too high\n"
        if slots.get("budget"):
            msg += f"  • Budget Rs.{int(slots['budget']):,} may be too low\n"
        msg += "\nTry:\n"
        msg += "  • Increasing your budget\n"
        msg += "  • Removing location filter\n"
        msg += "  • Checking other courses"
        return msg

    constraints = []
    if slots.get("course"):
        constraints.append(f"Course: {slots['course']}")
    if slots.get("rank"):
        constraints.append(f"Rank: {slots['rank']}")
    if slots.get("budget"):
        constraints.append(f"Budget: Rs.{int(slots['budget']):,}")
    if slots.get("location"):
        constraints.append(f"Location: {slots['location']}")
    if slots.get("college_type"):
        constraints.append(f"Type: {slots['college_type']}")

    lines = [
        f"Here are the top {len(top_colleges)} colleges "
        f"matching your constraints",
        f"({' | '.join(constraints)}):\n",
    ]

    for i, c in enumerate(top_colleges, 1):
        hostel_str = "Yes" if c.get("hostel") else "No"
        lines += [
            f"{'─' * 55}",
            f"  #{i}  {c.get('college_name', 'Unknown')}",
            f"       📍 Location : {c.get('location', 'N/A')}",
            f"       🏛️  Type     : {c.get('college_type', 'N/A')}  |  🏠 Hostel: {hostel_str}",
            f"       📚 Course   : {c.get('course', 'N/A')}",
            f"       💰 Fee      : Rs. {c.get('fee', 0):,.0f}",
            f"       🎯 Cutoff   : Rank {c.get('cutoff_rank', 'N/A')}",
            f"       ⭐ Rating   : {c.get('rating', 0)} / 5.0",
            f"       📊 Match    : {c.get('Score', 0):.1f} / 100",
            f"       ✅ Why this fits:",
        ]
        for reason in c.get("Reasons", []):
            lines.append(f"            • {reason}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# MAIN ENTRY POINT — called by router.py
# ─────────────────────────────────────────────────────

async def handle(
    intent: str,
    slots: Dict[str, Any],
    collection,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Full hybrid pipeline for recommend_with_constraints.

    Steps:
      1. Check missing slots → ask if needed
      2. MongoDB hard filter → 15 candidates
      3. Python scoring → score each candidate
      4. Rerank → top 3
      5. Format response
    """

    # ── Slot check ────────────────────────────────
    missing_slot, question = get_missing_slot(intent, slots)
    if missing_slot:
        return {
            "intent": intent,
            "query": {},
            "results": [],
            "count": 0,
            "response": question,
            "action": "ask",
            "missing_slot": missing_slot,
        }

    # ── Step 1: MongoDB hard filter ───────────────
    pipeline = build_candidate_pipeline(slots, limit=15)
    logger.info(f"[recommend] candidate pipeline: {pipeline}")

    cursor = collection.aggregate(pipeline)
    candidates = await cursor.to_list(length=15)
    candidates = deduplicate(candidates)

    if not candidates:
        return {
            "intent": intent,
            "query": {"pipeline": pipeline},
            "results": [],
            "count": 0,
            "response": _format_response([], slots),
        }

    # ── Step 2 + 3: Python scoring + reranking ────
    top_colleges = RecommendScorer.rerank(candidates, slots, top_n=min(top_k, 3))

    # ── Step 4: Format ────────────────────────────
    message = _format_response(top_colleges, slots)

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": top_colleges,
        "count": len(top_colleges),
        "response": message,
    }
