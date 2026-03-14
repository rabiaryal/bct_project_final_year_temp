"""
Best Handler — best_items_search intent

Trigger: "best colleges for computer engineering", "top rated colleges"
Pipeline: $unwind → $match → $project → $sort(rating DESC) → $limit
"""

from typing import Dict, List, Any
import logging

from app.templates.intent_templates import get_template
from app.core.query_builder import build_pipeline, deduplicate
from app.utils.formatter import format_best_items_search

logger = logging.getLogger(__name__)


async def handle(intent: str, slots: Dict[str, Any], collection, top_k: int = 5) -> Dict[str, Any]:
    template = get_template(intent)
    working_slots = dict(slots)

    pipeline = build_pipeline(intent, working_slots, template, limit=top_k * 3)
    logger.info(f"[best_items_search] pipeline: {pipeline}")

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=top_k * 3)
    results = deduplicate(docs)[:top_k]

    # Fallback: if strict course filter caused no hits, retry with rating-first filters.
    # This keeps location/college_type/budget filters while dropping noisy course terms.
    if not results and working_slots.get("course") and any(
        working_slots.get(k) is not None for k in ("location", "college_type", "budget")
    ):
        relaxed_slots = dict(working_slots)
        relaxed_slots.pop("course", None)
        relaxed_pipeline = build_pipeline(intent, relaxed_slots, template, limit=top_k * 3)
        logger.info(f"[best_items_search] retry without course filter: {relaxed_pipeline}")
        relaxed_cursor = collection.aggregate(relaxed_pipeline)
        relaxed_docs = await relaxed_cursor.to_list(length=top_k * 3)
        relaxed_results = deduplicate(relaxed_docs)[:top_k]
        if relaxed_results:
            results = relaxed_results
            pipeline = relaxed_pipeline
            working_slots = relaxed_slots

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": results,
        "count": len(results),
        "response": format_best_items_search(results, working_slots),
    }
