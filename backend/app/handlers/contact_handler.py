"""
Contact Handler — contact_query intent

Trigger: "contact of Pulchowk Campus", "phone number of KEC"
Pipeline: $match → $project (NO $unwind — top-level fields only)
"""

from typing import Dict, List, Any
import logging

from app.templates.intent_templates import get_template
from app.core.query_builder import build_pipeline, deduplicate
from app.utils.formatter import format_contact_query

logger = logging.getLogger(__name__)


async def handle(intent: str, slots: Dict[str, Any], collection, top_k: int = 5) -> Dict[str, Any]:
    template = get_template(intent)
    pipeline = build_pipeline(intent, slots, template, limit=top_k * 3)
    logger.info(f"[contact_query] pipeline: {pipeline}")

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=top_k * 3)
    results = deduplicate(docs)[:top_k]

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": results,
        "count": len(results),
        "response": format_contact_query(results, slots),
    }
