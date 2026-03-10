"""
Compare Handler — compare_colleges intent

Trigger: "compare Pulchowk and Thapathali"
Pipeline: $match($or on two names) → $unwind → $project → $limit
"""

from typing import Dict, List, Any
import logging

from app.templates.intent_templates import get_template
from app.core.query_builder import build_pipeline, deduplicate
from app.utils.formatter import format_compare_colleges

logger = logging.getLogger(__name__)


async def handle(intent: str, slots: Dict[str, Any], collection, top_k: int = 5) -> Dict[str, Any]:
    template = get_template(intent)
    pipeline = build_pipeline(intent, slots, template, limit=top_k * 3)
    logger.info(f"[compare_colleges] pipeline: {pipeline}")

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=top_k * 3)
    results = docs[:top_k * 2]  # keep both colleges

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": results,
        "count": len(results),
        "response": format_compare_colleges(results, slots),
    }
