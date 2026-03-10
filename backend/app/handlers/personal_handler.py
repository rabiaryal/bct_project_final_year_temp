"""
Personal Handler — personalized_recommendation intent  (HYBRID)

Trigger: "my rank is 2500 and budget 7 lakhs suggest BE Civil"

KEY DIFFERENCE from recommend_handler:
  - Rank safety weighted HIGHER (45 vs 35)
  - Response tone is personal ("I recommend FOR YOU")
  - Adds admission safety category (SAFE / MODERATE / RISKY)

HYBRID FLOW:
  1. Check missing slots → ask one at a time
  2. MongoDB hard filter → 15 candidates
  3. Python scoring (PersonalScorer) → score + safety label
  4. Rerank → top 3
  5. Format with personal tone + safety labels
"""

from typing import Dict, List, Any
import logging

from app.core.slot_filler import get_missing_slot
from app.core.query_builder import build_candidate_pipeline, deduplicate
from app.core.scorer import PersonalScorer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# FORMAT — personal tone: "Based on YOUR profile"
# ─────────────────────────────────────────────────────

def _format_response(top_colleges: List[Dict[str, Any]], slots: Dict[str, Any]) -> str:
    if not top_colleges:
        msg = "Sorry, I could not find colleges that fit your profile.\n\n"
        if slots.get("rank"):
            msg += (
                f"With rank {slots['rank']}, you may need to:\n"
                f"  • Increase budget if targeting private colleges\n"
                f"  • Consider colleges in other locations\n"
            )
        if slots.get("budget"):
            msg += (
                f"\nWith budget Rs.{int(slots['budget']):,}:\n"
                f"  • Public colleges are well within range\n"
                f"  • Try removing location filter"
            )
        return msg

    user_rank = slots.get("rank")
    user_budget = slots.get("budget")
    course = slots.get("course", "")

    safe_count = sum(1 for c in top_colleges if c.get("SafetyLabel") == "SAFE")
    moderate_count = sum(1 for c in top_colleges if c.get("SafetyLabel") == "MODERATE")
    risky_count = sum(1 for c in top_colleges if c.get("SafetyLabel") == "RISKY")

    lines = [
        "Based on your profile:",
    ]
    if user_rank:
        lines.append(f"  🎯 IOE Rank : {user_rank}")
    if user_budget:
        lines.append(f"  💰 Budget   : Rs.{int(user_budget):,}")
    if course:
        lines.append(f"  📚 Course   : {course}")

    lines += [
        f"\nI personally recommend these {len(top_colleges)} colleges:\n",
        f"  🟢 Safe: {safe_count}  "
        f"🟡 Moderate: {moderate_count}  "
        f"🔴 Risky: {risky_count}\n",
    ]

    badge_map = {
        "SAFE": "🟢 SAFE CHOICE",
        "MODERATE": "🟡 GOOD CHOICE",
        "RISKY": "🔴 RISKY CHOICE",
    }

    for i, c in enumerate(top_colleges, 1):
        badge = badge_map.get(c.get("SafetyLabel", ""), "")
        hostel_str = "Yes" if c.get("hostel") else "No"

        lines += [
            f"{'─' * 55}",
            f"  #{i}  {c.get('college_name', 'Unknown')}  {badge}",
            f"       📍 {c.get('location', 'N/A')}",
            f"       🏛️  {c.get('college_type', 'N/A')}  |  🏠 Hostel: {hostel_str}",
            f"       📚 {c.get('course', 'N/A')}",
            f"       💰 Fee      : Rs. {c.get('fee', 0):,.0f}",
            f"       🎯 Cutoff   : Rank {c.get('cutoff_rank', 'N/A')}",
            f"       ⭐ Rating   : {c.get('rating', 0)} / 5.0",
            f"       📊 Score    : {c.get('Score', 0):.1f} / 100",
            f"       💡 Why for you:",
        ]
        for reason in c.get("Reasons", []):
            lines.append(f"            • {reason}")
        lines.append("")

    if risky_count > 0:
        lines.append(
            "⚠️  Note: 🔴 RISKY choices have very low rank gap. "
            "Apply to 🟢 SAFE colleges first."
        )

    return "\n".join(line for line in lines if line is not None)


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
    Full hybrid pipeline for personalized_recommendation.

    Steps:
      1. Check missing slots → ask one at a time
      2. MongoDB hard filter → 15 candidates
      3. Python score each candidate
      4. Rerank → top 3
      5. Format with personal tone + safety labels
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
    logger.info(f"[personal] candidate pipeline: {pipeline}")

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
    top_colleges = PersonalScorer.rerank(candidates, slots, top_n=min(top_k, 3))

    # ── Step 4: Format with personal tone ─────────
    message = _format_response(top_colleges, slots)

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": top_colleges,
        "count": len(top_colleges),
        "response": message,
    }
