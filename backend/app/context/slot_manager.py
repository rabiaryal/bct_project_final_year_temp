"""
Slot Manager — 10-Intent Aggregation-Based System

Entity → Slot mapping (11 entity types):
  COURSE, LOCATION, COLLEGE_TYPE, RANK, BUDGET, HOSTEL,
  COLLEGE_NAME, COLLEGE_NAME_1, COLLEGE_NAME_2, RATING, ATTRIBUTE

Normalization:
  - Course aliases  → official DB department/course names
  - Budget strings  → integer rupees  ("7 lakhs" → 700000)
  - Rank strings    → integer         ("2500" → 2500)
  - Hostel strings  → bool            ("yes" → True)
  - College type    → "public" | "private"

Context:
  - Simplified DialogueContext (no rec_context / ready_to_query)
  - Slot carryover across related intent families
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re
import logging

from rapidfuzz import fuzz, process as rfprocess

from app.templates.intent_templates import (
    SLOT_SCHEMA,
    get_template,
    validate_slots,
    get_follow_up_question,
    get_intent_family,
    should_carry_slots,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENTITY → SLOT MAPPING
# ============================================================================

ENTITY_TO_SLOT: Dict[str, str] = {
    # Primary entity types (11)
    "COURSE":          "course",
    "LOCATION":        "location",
    "COLLEGE_TYPE":    "college_type",
    "RANK":            "rank",
    "BUDGET":          "budget",
    "HOSTEL":          "hostel",
    "COLLEGE_NAME":    "college_name",
    "COLLEGE_NAME_1":  "college_name_1",
    "COLLEGE_NAME_2":  "college_name_2",
    "RATING":          "rating",
    "ATTRIBUTE":       "attribute",

    # Legacy / alias mappings
    "COLLEGE":         "college_name",
    "NAME":            "college_name",
    "DEPARTMENT":      "course",
    "DEPARTMENT_NAME": "course",
    "PROGRAM":         "course",
    "PROGRAM_NAME":    "course",
    "COURSE_NAME":     "course",
    "FEE":             "budget",
    "FEE_LIMIT":       "budget",
    "MAX_FEE":         "budget",
    "TYPE":            "college_type",
    "CUTOFF_RANK":     "rank",
    "PROVINCE":        "location",
    "CITY":            "location",
    "HOSTEL_AVAILABILITY": "hostel",
    "SEATS":           "attribute",
    "SCHOLARSHIP":     "attribute",
}


# ============================================================================
# VALUE NORMALIZERS
# ============================================================================

_COURSE_ALIASES: Dict[str, str] = {
    "computer":             "BE Computer",
    "computer engineering": "BE Computer",
    "computer science":     "BE Computer",
    "software":             "BE Computer",
    "software engineering": "BE Computer",
    "it":                   "BE Computer",
    "information technology": "BE Computer",
    "civil":                "BE Civil",
    "civil engineering":    "BE Civil",
    "electrical":           "BE Electrical",
    "electrical engineering": "BE Electrical",
    "electronics":          "BE Electronics",
    "electronics engineering": "BE Electronics",
    "mechanical":           "BE Mechanical",
    "mechanical engineering": "BE Mechanical",
    "biomedical":           "BE Biomedical",
    "biomedical engineering": "BE Biomedical",
    "architecture":         "B.Arch",
    "aerospace":            "BE Aerospace",
    "aerospace engineering": "BE Aerospace",
    "agricultural":         "BE Agricultural",
    "agricultural engineering": "BE Agricultural",
    "automobile":           "BE Automobile",
    "automobile engineering": "BE Automobile",
    "industrial":           "BE Industrial",
    "industrial engineering": "BE Industrial",
    "geomatics":            "BE Geomatics",
    "geomatics engineering": "BE Geomatics",
    "chemical":             "BE Chemical",
    "chemical engineering": "BE Chemical",
}


def normalize_course(value: str) -> Optional[str]:
    """Normalize course name to official DB value.

    Returns None for non-course ranking words (e.g. "best", "top rated").
    """
    key = re.sub(r"[^a-z0-9\s\-/]", "", value.lower().strip())
    noisy_course_words = {
        "best", "top", "top rated", "best rated", "highest rated", "rated",
        "college", "colleges",
    }
    if key in noisy_course_words:
        return None
    return _COURSE_ALIASES.get(key, value.strip())


def normalize_budget(value: Any) -> Optional[int]:
    """Parse budget string -> integer rupees.

    Handles: "700000", "7 lakhs", "700k", "7,00,000", plain int/float.
    Bare small numbers (< 1000) are assumed to be in lakhs because
    the cheapest Nepal engineering fee is ~490,000.
    """
    if isinstance(value, (int, float)):
        v = int(value)
        if v < 1000:
            return v * 100_000
        return v
    text = str(value).lower().strip().replace(",", "")
    # "7 lakhs" / "7 lakh"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh)", text)
    if m:
        return int(float(m.group(1)) * 100_000)
    # "700k"
    m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text)
    if m:
        return int(float(m.group(1)) * 1_000)
    # plain digits
    m = re.search(r"(\d+)", text)
    if m:
        v = int(m.group(1))
        if v < 1000:
            return v * 100_000
        return v
    return None


def normalize_rank(value: Any) -> Optional[int]:
    """Extract integer rank."""
    if isinstance(value, int):
        return value
    m = re.search(r"\d+", str(value))
    return int(m.group()) if m else None


def normalize_hostel(value: Any) -> Optional[bool]:
    """Normalize hostel to boolean."""
    if isinstance(value, bool):
        return value
    text = str(value).lower().strip()
    if text in ("yes", "true", "1", "available", "needed", "required"):
        return True
    if text in ("no", "false", "0", "not needed"):
        return False
    return True  # default: user asked about hostel -> filter for available


def normalize_college_type(value: str) -> str:
    """Normalize to 'public' or 'private'."""
    v = value.lower().strip()
    if v in ("public", "government", "govt", "constituent"):
        return "public"
    if v in ("private", "pvt"):
        return "private"
    return v


# Known college names for fuzzy matching (lowercase)
_KNOWN_COLLEGES = [
    "pulchowk engineering campus",
    "thapathali engineering campus",
    "western regional campus",
    "eastern regional campus",
    "kathmandu engineering college",
    "kathford international college of engineering and management",
    "sagarmatha engineering college",
    "himalaya college of engineering",
    "kantipur engineering college",
    "advanced college of engineering and management",
    "national college of engineering",
    "khwopa college of engineering",
    "nepal engineering college",
    "lumbini engineering college",
    "institute of engineering",
]

_COLLEGE_ABBREVIATIONS = {
    "ioe": "Institute of Engineering",
    "tu": "Tribhuvan University",
    "pu": "Pokhara University",
    "ku": "Kathmandu University",
    "wrc": "Western Regional Campus",
    "erc": "Eastern Regional Campus",
    "pulchowk": "Pulchowk Engineering Campus",
    "thapathali": "Thapathali Engineering Campus",
    "kathford": "Kathford International College of Engineering and Management",
    "khwopa": "Khwopa College of Engineering",
}


def normalize_college_name(value: str) -> str:
    """Expand abbreviations, then fuzzy-match against known college list."""
    v = value.lower().strip()

    # 1. Check abbreviations
    if v in _COLLEGE_ABBREVIATIONS:
        return _COLLEGE_ABBREVIATIONS[v]

    # 2. Fuzzy match against known colleges (threshold 70)
    match = rfprocess.extractOne(
        v, _KNOWN_COLLEGES, scorer=fuzz.WRatio, score_cutoff=70
    )
    if match:
        matched_name, score, _idx = match
        logger.info(f"Fuzzy college match: '{v}' -> '{matched_name}' (score={score})")
        return matched_name

    return value.strip()


# Common misspellings / aliases → canonical location name
_LOCATION_CORRECTIONS: Dict[str, str] = {
    "kathamandu":    "kathmandu",
    "kathamndu":     "kathmandu",
    "kathmandu":     "kathmandu",
    "ktm":           "kathmandu",
    "lalitpur":      "lalitpur",
    "patan":         "lalitpur",
    "pokhara":       "pokhara",
    "pokhra":        "pokhara",
    "bhaktapur":     "bhaktapur",
    "bhadgaon":      "bhaktapur",
    "chitwan":       "chitwan",
    "dharan":        "dharan",
    "sunsari":       "sunsari",
}

def normalize_location(value: str) -> str:
    """Normalize location names (fix typos, province aliases)."""
    v = re.sub(r"[^a-z0-9\s\-]", "", value.lower().strip())
    provinces = {
        "province 1": "koshi",
        "province 2": "madhesh",
        "province 3": "bagmati",
        "province 4": "gandaki",
        "province 5": "lumbini",
        "province 6": "karnali",
        "province 7": "sudurpashchim",
    }
    if v in provinces:
        return provinces[v]
    if v in _LOCATION_CORRECTIONS:
        return _LOCATION_CORRECTIONS[v]
    return v if v else value.strip()


def normalize_value(slot_name: str, value: Any) -> Any:
    """Route to the right normalizer."""
    if value is None:
        return None
    if slot_name == "course":
        return normalize_course(str(value))
    if slot_name == "location":
        return normalize_location(str(value))
    if slot_name == "college_type":
        return normalize_college_type(str(value))
    if slot_name in ("college_name", "college_name_1", "college_name_2"):
        return normalize_college_name(str(value))
    if slot_name == "budget":
        return normalize_budget(value)
    if slot_name == "rank":
        return normalize_rank(value)
    if slot_name == "hostel":
        return normalize_hostel(value)
    if slot_name == "rating":
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return str(value).lower().strip() if isinstance(value, str) else value


# ============================================================================
# DIALOGUE CONTEXT
# ============================================================================

@dataclass
class DialogueContext:
    """Context for a single conversation session."""

    session_id: str

    # Slots (accumulated across turns)
    slots: Dict[str, Any] = field(default_factory=dict)
    current_intent: str = ""
    previous_intent: str = ""
    intent_family: str = ""
    turn_count: int = 0
    missing_slots: List[str] = field(default_factory=list)
    is_actionable: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Slot-filling lock: when the bot asked a follow-up, this records
    # which slot it expects the next message to fill.
    pending_slot: Optional[str] = None

    # Result tracking
    last_results: List[str] = field(default_factory=list)
    last_result_count: int = 0
    last_query: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":      self.session_id,
            "slots":           self.slots,
            "current_intent":  self.current_intent,
            "previous_intent": self.previous_intent,
            "intent_family":   self.intent_family,
            "turn_count":      self.turn_count,
            "missing_slots":   self.missing_slots,
            "is_actionable":   self.is_actionable,
            "pending_slot":    self.pending_slot,
            "last_results":    self.last_results,
            "last_result_count": self.last_result_count,
        }


# ============================================================================
# FALLBACK TEXT EXTRACTORS (rank / budget from raw text)
# ============================================================================

_RANK_PATTERN = re.compile(
    r"(?:rank|position)\s*(?:is|was|:|-|=)?\s*(\d{1,5})"
    r"|(\d{1,5})\s*(?:rank|position)",
    re.IGNORECASE,
)

_LAKH_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh)", re.IGNORECASE
)

_BUDGET_KEYWORD = re.compile(
    r"\b(budget|afford|spend|pay|fee|tuition|cost)\b", re.IGNORECASE
)

_BUDGET_AMOUNT = re.compile(
    r"(\d[\d,]*)\s*(lakhs?|lakh|rupees?|rs\.?|nrs\.?|k\b|thousand)?",
    re.IGNORECASE,
)


def _extract_from_text(text: str, existing_slots: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback: extract rank / budget from raw text when NLU missed them."""
    found: Dict[str, Any] = {}

    # Rank
    if "rank" not in existing_slots:
        m = _RANK_PATTERN.search(text)
        if m:
            raw = m.group(1) or m.group(2)
            try:
                found["rank"] = int(raw)
            except (TypeError, ValueError):
                pass

    # Budget
    if "budget" not in existing_slots:
        m = _LAKH_PATTERN.search(text)
        if m:
            try:
                found["budget"] = int(float(m.group(1).replace(",", "")) * 100_000)
            except (TypeError, ValueError):
                pass
        elif _BUDGET_KEYWORD.search(text):
            m2 = _BUDGET_AMOUNT.search(text)
            if m2:
                raw = m2.group(1).replace(",", "")
                suffix = (m2.group(2) or "").lower()
                try:
                    amount = float(raw)
                    if "lakh" in suffix:
                        amount *= 100_000
                    elif suffix in ("k", "thousand"):
                        amount *= 1_000
                    found["budget"] = int(amount)
                except (TypeError, ValueError):
                    pass

    return found


# ============================================================================
# SLOT MANAGER
# ============================================================================

class SlotManager:
    """Manages slot filling and context updates for the 10-intent system."""

    def __init__(self):
        self.contexts: Dict[str, DialogueContext] = {}

    def get_context(self, session_id: str) -> Optional[DialogueContext]:
        return self.contexts.get(session_id)

    def get_or_create_context(self, session_id: str) -> DialogueContext:
        if session_id not in self.contexts:
            self.contexts[session_id] = DialogueContext(session_id=session_id)
        return self.contexts[session_id]

    def process_turn(
        self,
        session_id: str,
        intent: str,
        entities: Dict[str, str],
        raw_text: str = "",
    ) -> DialogueContext:
        """Process a single turn: entities -> slots -> actionability check."""
        context = self.get_or_create_context(session_id)

        # Turn tracking
        context.turn_count += 1
        context.previous_intent = context.current_intent
        context.current_intent = intent
        context.intent_family = get_intent_family(intent)
        context.last_updated = datetime.now()

        # ── Fix #3: Greeting should NOT reset slots ──────────────────
        if intent in ("greeting", "goodbye"):
            # Keep accumulated slots; just update intent tracking above.
            is_valid, missing = validate_slots(intent, context.slots)
            context.missing_slots = missing
            context.is_actionable = is_valid
            context.pending_slot = None
            return context

        # ── Fix #2: Expected-slot override ────────────────────────────
        # If the bot asked "Which course?" last turn and the user just
        # replied "computer", NER may fail.  Force-assign the raw text
        # to the expected slot when NER produced nothing useful for it.
        if context.pending_slot and raw_text:
            slot_name = context.pending_slot
            # Only force-fill if NER didn't already supply it
            already_filled = any(
                ENTITY_TO_SLOT.get(et.upper()) == slot_name
                for et in entities
            )
            if not already_filled:
                normalized = normalize_value(slot_name, raw_text.strip())
                if normalized is not None:
                    context.slots[slot_name] = normalized
                    logger.info(f"Pending-slot override: {slot_name} = {normalized}")
            context.pending_slot = None  # consumed

        # Clear slots on family change (unless families are compatible)
        if context.previous_intent and not should_carry_slots(context.previous_intent, intent):
            context.slots = {}
            logger.info(f"Cleared slots (family change: {context.previous_intent} -> {intent})")
        elif context.previous_intent and context.previous_intent != intent:
            # Fix #4: Even when carrying across compatible families,
            # prune slots that are irrelevant to the new intent.
            template = get_template(intent)
            relevant = set(template.required_slots) | set(template.optional_slots)
            pruned = {k: v for k, v in context.slots.items() if k in relevant}
            removed = set(context.slots) - set(pruned)
            if removed:
                logger.info(f"Pruned irrelevant slots for {intent}: {removed}")
                context.slots = pruned

        # Merge entity-extracted slots
        new_slots = self._entities_to_slots(entities)
        for slot_name, slot_value in new_slots.items():
            if slot_value is not None:
                context.slots[slot_name] = slot_value
                logger.info(f"Slot updated: {slot_name} = {slot_value}")

        # Fallback: extract rank/budget from raw text
        if raw_text:
            text_slots = _extract_from_text(raw_text, context.slots)
            for k, v in text_slots.items():
                if v is not None and k not in context.slots:
                    context.slots[k] = v
                    logger.info(f"Text-extracted slot: {k} = {v}")

        # Evaluate actionability
        is_valid, missing = validate_slots(intent, context.slots)
        context.missing_slots = missing
        context.is_actionable = is_valid

        return context

    def _entities_to_slots(self, entities: Dict[str, str]) -> Dict[str, Any]:
        """Convert NLU entities to normalized slots."""
        slots = {}
        for entity_type, entity_value in entities.items():
            slot_name = ENTITY_TO_SLOT.get(entity_type.upper())
            if slot_name:
                normalized = normalize_value(slot_name, entity_value)
                if normalized is not None:
                    slots[slot_name] = normalized
        return slots

    def update_with_results(
        self,
        session_id: str,
        result_dicts: List[Dict[str, Any]],
        query: Dict[str, Any],
    ) -> None:
        """Persist DB fetch results into the session context."""
        ctx = self.contexts.get(session_id)
        if not ctx:
            return
        names = [
            r.get("college_name", "") for r in result_dicts
            if r.get("college_name")
        ]
        ctx.last_results = names
        ctx.last_result_count = len(result_dicts)
        ctx.last_query = query

        # Auto-fill college_name when exactly one result
        if len(names) == 1 and "college_name" not in ctx.slots:
            ctx.slots["college_name"] = names[0].lower()
            logger.info(f"Auto-filled college_name = '{names[0]}' (single result)")

    def get_follow_up(self, context: DialogueContext) -> str:
        """Return a follow-up question and record which slot we expect."""
        template = get_template(context.current_intent)
        missing = template.get_missing_slots(context.slots)
        if missing:
            context.pending_slot = missing[0]
            logger.info(f"Setting pending_slot = {missing[0]}")
        return get_follow_up_question(context.current_intent, context.slots)

    def is_actionable(self, context: DialogueContext) -> bool:
        return context.is_actionable

    def clear_context(self, session_id: str):
        if session_id in self.contexts:
            del self.contexts[session_id]

    def get_debug_info(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.contexts:
            return {"error": "Session not found"}
        ctx = self.contexts[session_id]
        template = get_template(ctx.current_intent)
        return {
            "session_id":      session_id,
            "turn_count":      ctx.turn_count,
            "current_intent":  ctx.current_intent,
            "previous_intent": ctx.previous_intent,
            "intent_family":   ctx.intent_family,
            "slots":           ctx.slots,
            "required_slots":  template.required_slots,
            "optional_slots":  template.optional_slots,
            "missing_slots":   ctx.missing_slots,
            "is_actionable":   ctx.is_actionable,
            "last_result_count": ctx.last_result_count,
            "last_results":    ctx.last_results,
            "last_query":      ctx.last_query,
        }
