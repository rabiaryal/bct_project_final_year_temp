"""
Context Manager — thin wrapper over SlotManager for handler use.

Handlers interact with this to read/update conversation state.
The underlying SlotManager (in app.context.slot_manager) does the heavy lifting.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from app.context.slot_manager import SlotManager, DialogueContext


class ContextManager:
    """Convenience API around SlotManager for use inside handlers."""

    def __init__(self, slot_manager: SlotManager):
        self._sm = slot_manager

    # -- read ----------------------------------------------------------------

    def get(self, session_id: str) -> Optional[DialogueContext]:
        return self._sm.get_context(session_id)

    def get_slots(self, session_id: str) -> Dict[str, Any]:
        ctx = self._sm.get_context(session_id)
        return dict(ctx.slots) if ctx else {}

    def get_last_results(self, session_id: str) -> List[str]:
        ctx = self._sm.get_context(session_id)
        return list(ctx.last_results) if ctx else []

    # -- write ---------------------------------------------------------------

    def store_results(
        self,
        session_id: str,
        results: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> None:
        self._sm.update_with_results(session_id, results, query)

    def clear(self, session_id: str) -> None:
        self._sm.clear_context(session_id)
