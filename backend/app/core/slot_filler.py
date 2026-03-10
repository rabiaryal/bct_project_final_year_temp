"""
Slot Filler — checks missing slots and generates follow-up questions.

Used by hybrid handlers to ask for missing information one slot at a time.
"""

from typing import Dict, Any, Optional, Tuple

from app.templates.intent_templates import get_template, validate_slots


# Required slots per intent (only intents that need slot checking)
_REQUIRED = {
    "recommend_with_constraints":    ["course", "rank", "budget"],
    "personalized_recommendation":   ["rank", "budget", "course"],
    "compare_colleges":              ["college_name_1", "college_name_2"],
    "college_details":               ["college_name"],
    "contact_query":                 ["college_name"],
}

# Custom follow-up questions
_QUESTIONS = {
    "course":          "Which engineering course are you interested in? (e.g. Computer, Civil, Electrical)",
    "rank":            "What is your IOE entrance rank?",
    "budget":          "What is your fee budget? (e.g. 700000 or 7 lakhs)",
    "location":        "Which location? (e.g. Kathmandu, Pokhara, Lalitpur)",
    "college_name":    "Which college would you like to know about?",
    "college_name_1":  "Which is the first college you want to compare?",
    "college_name_2":  "And the second college?",
    "college_type":    "Public or private?",
}


def get_missing_slot(
    intent: str,
    slots: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Check if any required slot is missing for the given intent.

    Returns:
        (slot_name, question)  if a slot is missing
        (None, None)           if all required slots are filled
    """
    required = _REQUIRED.get(intent, [])
    for slot in required:
        if slot not in slots or slots[slot] is None:
            question = _QUESTIONS.get(
                slot, f"Could you please specify the {slot.replace('_', ' ')}?"
            )
            return slot, question
    return None, None


class SlotFiller:
    """Stateless helper — checks slot readiness for any intent."""

    @staticmethod
    def is_ready(intent: str, slots: Dict[str, Any]) -> bool:
        missing, _ = get_missing_slot(intent, slots)
        return missing is None

    @staticmethod
    def next_question(intent: str, slots: Dict[str, Any]) -> Optional[str]:
        _, question = get_missing_slot(intent, slots)
        return question
