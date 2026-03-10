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
    pipeline = build_pipeline(intent, slots, template, limit=top_k * 3)
    logger.info(f"[best_items_search] pipeline: {pipeline}")

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=top_k * 3)
    results = deduplicate(docs)[:top_k]

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": results,
        "count": len(results),
        "response": format_best_items_search(results, slots),
    }
