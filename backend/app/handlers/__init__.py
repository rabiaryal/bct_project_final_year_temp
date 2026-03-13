"""Handlers module — one handler per intent, plus a router."""

from app.handlers import (
    search_handler,
    best_handler,
    recommend_handler,
    personal_handler,
    compare_handler,
    details_handler,
    hostel_handler,
    contact_handler,
    admission_handler,
)
from app.handlers.router import route_intent

__all__ = [
    "route_intent",
    "search_handler",
    "best_handler",
    "recommend_handler",
    "personal_handler",
    "compare_handler",
    "details_handler",
    "hostel_handler",
    "contact_handler",
    "admission_handler",
]
