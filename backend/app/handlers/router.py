"""
Router — dispatches intent to the correct handler.

Static intents (greeting, goodbye, unknown) are handled inline.
DB intents are delegated to their dedicated handler module.
Hybrid intents (recommend, personal) use Python-side scoring.
"""

from typing import Dict, Any
import logging

from app.utils.formatter import format_greeting, format_goodbye, format_unknown

# Handler modules — each exposes  async handle(intent, slots, collection, top_k)
from app.handlers import (
    search_handler,
    best_handler,
    recommend_handler,
    personal_handler,
    compare_handler,
    details_handler,
    attribute_handler,
    hostel_handler,
    contact_handler,
    admission_handler,
)

logger = logging.getLogger(__name__)

# Intent → handler module mapping
_HANDLER_MAP = {
    "search_college":               search_handler,
    "best_items_search":            best_handler,
    "recommend_with_constraints":   recommend_handler,
    "personalized_recommendation":  personal_handler,
    "compare_colleges":             compare_handler,
    "college_details":              details_handler,
    "college_attribute_query":      attribute_handler,
    "hostel_query":                 hostel_handler,
    "contact_query":                contact_handler,
    "admission_process":             admission_handler,
}

# Static intents — no DB access
_STATIC_INTENTS = {
    "greeting": format_greeting,
    "goodbye":  format_goodbye,
    "unknown":  format_unknown,
}


async def route_intent(
    intent: str,
    slots: Dict[str, Any],
    collection,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Route an intent to the appropriate handler.

    Returns a dict with keys: intent, query, results, count, response
    (and optionally: action, missing_slot for hybrid follow-ups).
    """

    # ── Static / conversational ───────────────────
    if intent in _STATIC_INTENTS:
        formatter = _STATIC_INTENTS[intent]
        return {
            "intent": intent,
            "query": {},
            "results": [],
            "count": 0,
            "response": formatter([], slots),
        }

    # ── DB / Hybrid handlers ─────────────────────
    handler_module = _HANDLER_MAP.get(intent)
    if handler_module:
        return await handler_module.handle(intent, slots, collection, top_k)

    # ── Fallback ─────────────────────────────────
    logger.warning(f"No handler for intent '{intent}', falling back to unknown")
    return {
        "intent": intent,
        "query": {},
        "results": [],
        "count": 0,
        "response": format_unknown([], slots),
    }
