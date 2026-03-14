"""
Intent Templates — 10-Intent Aggregation-Based System

Intents:
  1. search_college          →  COURSE + LOCATION + COLLEGE_TYPE
  2. best_items_search       →  COURSE + LOCATION  (sort by Rating)
  3. recommend_with_constraints → COURSE + RANK + BUDGET + LOCATION
  4. personalized_recommendation → RANK + BUDGET + COURSE + LOCATION
  5. compare_colleges        →  COLLEGE_NAME_1 + COLLEGE_NAME_2
  6. college_details         →  COLLEGE_NAME
  7. hostel_query            →  LOCATION + COLLEGE_TYPE + HOSTEL  (no $unwind)
  8. contact_query           →  COLLEGE_NAME  (no $unwind)
  9. admission_process       →  COLLEGE_NAME  (no $unwind, contact-oriented)
 10. greeting                →  static
 11. goodbye                 →  static

DB fields (15-college Nepal engineering DB):
  Top-level: Name, Location, Type, ContactNumber, Email, HostelAvailability
  Departments[]: Name
  Departments[].Courses[]: CourseId, Name, AverageCutoffRank, Fee, Rating
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# ============================================================================
# SLOT SCHEMA (single source of truth)
# ============================================================================

SLOT_SCHEMA = {
    "course":          {"type": str,   "desc": "Course name (e.g. BE Computer)"},
    "location":        {"type": str,   "desc": "City or district"},
    "college_type":    {"type": str,   "desc": "public or private"},
    "rank":            {"type": int,   "desc": "IOE entrance rank (lower = better)"},
    "budget":          {"type": int,   "desc": "Max fee in rupees"},
    "hostel":          {"type": bool,  "desc": "Hostel availability filter"},
    "college_name":    {"type": str,   "desc": "College name"},
    "college_name_1":  {"type": str,   "desc": "First college for comparison"},
    "college_name_2":  {"type": str,   "desc": "Second college for comparison"},
    "rating":          {"type": float, "desc": "Sort trigger only — never a filter"},
    "attribute":       {"type": str,   "desc": "Informational attribute"},
}


# ============================================================================
# INTENT TEMPLATE DATACLASS
# ============================================================================

@dataclass
class IntentTemplate:
    """Template for a single intent."""

    intent_name: str
    description: str

    # Slot requirements
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)

    # slot_name → DB field path
    query_fields: Dict[str, str] = field(default_factory=dict)

    # Pipeline behaviour
    uses_unwind: bool = True        # False for hostel_query, contact_query
    sort_field: str = ""            # DB path to sort on (after $unwind)
    sort_order: int = -1            # -1 = descending, 1 = ascending
    compute_score: str = ""         # "MatchScore" | "PersonalScore" | ""

    # Response
    response_template: str = ""
    no_results_message: str = "No results found."

    # Follow-up questions for missing slots
    follow_up_questions: Dict[str, str] = field(default_factory=dict)

    def get_missing_slots(self, filled_slots: Dict[str, Any]) -> List[str]:
        return [
            s for s in self.required_slots
            if s not in filled_slots or filled_slots[s] is None
        ]

    def is_actionable(self, filled_slots: Dict[str, Any]) -> bool:
        return len(self.get_missing_slots(filled_slots)) == 0

    def get_follow_up(self, filled_slots: Dict[str, Any]) -> str:
        missing = self.get_missing_slots(filled_slots)
        if missing and missing[0] in self.follow_up_questions:
            return self.follow_up_questions[missing[0]]
        if missing:
            return f"Could you please specify the {missing[0].replace('_', ' ')}?"
        return "Could you tell me more about what you're looking for?"


# ============================================================================
# INTENT TEMPLATES REGISTRY
# ============================================================================

INTENT_TEMPLATES: Dict[str, IntentTemplate] = {

    # 1 — search_college
    "search_college": IntentTemplate(
        intent_name="search_college",
        description="Search colleges by course, location, or type",
        required_slots=[],
        optional_slots=["course", "location", "college_type"],
        query_fields={
            "course":       "Departments.Courses.Name",
            "location":     "Location",
            "college_type": "Type",
        },
        uses_unwind=True,
        no_results_message="No colleges found matching your criteria.",
        follow_up_questions={
            "course":       "Which course are you looking for? (e.g. Computer Engineering, Civil)",
            "location":     "Which location? (e.g. Kathmandu, Pokhara, Lalitpur)",
            "college_type": "Public or private college?",
        },
    ),

    # 2 — best_items_search (sorted by Rating desc)
    "best_items_search": IntentTemplate(
        intent_name="best_items_search",
        description="Find best-rated colleges for a course or location",
        required_slots=[],
        optional_slots=["course", "location", "college_type", "budget"],
        query_fields={
            "course":       "Departments.Courses.Name",
            "location":     "Location",
            "college_type": "Type",
            "budget":       "Departments.Courses.Fee",
        },
        uses_unwind=True,
        sort_field="rating",
        sort_order=-1,
        no_results_message="No colleges found. Try broadening your search.",
        follow_up_questions={
            "course":       "Which course are you interested in?",
            "location":     "Looking in a specific location?",
            "college_type": "Public or private colleges?",
        },
    ),

    # 3 — recommend_with_constraints
    "recommend_with_constraints": IntentTemplate(
        intent_name="recommend_with_constraints",
        description="Recommend colleges matching course + rank + budget",
        required_slots=["course"],
        optional_slots=["rank", "budget", "location"],
        query_fields={
            "course":   "Departments.Courses.Name",
            "rank":     "Departments.Courses.Rank",
            "budget":   "Departments.Courses.Fee",
            "location": "Location",
        },
        uses_unwind=True,
        compute_score="MatchScore",
        no_results_message="No colleges match your criteria. Try a higher budget or relax the rank.",
        follow_up_questions={
            "course":  "Which engineering course are you interested in?",
            "rank":    "What is your IOE entrance rank?",
            "budget":  "What is your fee budget? (e.g. 700000 or 7 lakhs)",
        },
    ),

    # 4 — personalized_recommendation
    "personalized_recommendation": IntentTemplate(
        intent_name="personalized_recommendation",
        description="Personalized suggestion based on rank + budget",
        required_slots=["rank", "budget"],
        optional_slots=["course", "location"],
        query_fields={
            "rank":     "Departments.Courses.Rank",
            "budget":   "Departments.Courses.Fee",
            "course":   "Departments.Courses.Name",
            "location": "Location",
        },
        uses_unwind=True,
        compute_score="PersonalScore",
        no_results_message="No colleges match your rank and budget. Try relaxing your criteria.",
        follow_up_questions={
            "rank":    "What is your IOE entrance rank?",
            "budget":  "What is your fee budget in rupees?",
            "course":  "Any preferred course? (optional)",
        },
    ),

    # 5 — compare_colleges
    "compare_colleges": IntentTemplate(
        intent_name="compare_colleges",
        description="Side-by-side comparison of two colleges",
        required_slots=["college_name_1", "college_name_2"],
        optional_slots=[],
        query_fields={
            "college_name_1": "Name",
            "college_name_2": "Name",
        },
        uses_unwind=True,
        no_results_message="Could not find one or both colleges. Please check the names.",
        follow_up_questions={
            "college_name_1": "Which is the first college you want to compare?",
            "college_name_2": "And the second college?",
        },
    ),

    # 6 — college_details
    "college_details": IntentTemplate(
        intent_name="college_details",
        description="Detailed information about a specific college",
        required_slots=["college_name"],
        optional_slots=[],
        query_fields={
            "college_name": "Name",
        },
        uses_unwind=True,
        no_results_message="I couldn't find that college. Please check the spelling.",
        follow_up_questions={
            "college_name": "Which college would you like details about?",
        },
    ),

    # 6b — college_attribute_query (specific attribute about a college)
    "college_attribute_query": IntentTemplate(
        intent_name="college_attribute_query",
        description="Answer a specific question about a college attribute",
        required_slots=["college_name"],
        optional_slots=["attribute"],
        query_fields={
            "college_name": "Name",
        },
        uses_unwind=True,
        no_results_message="I couldn't find that college. Please check the spelling.",
        follow_up_questions={
            "college_name": "Which college are you asking about?",
        },
    ),

    # 7 — hostel_query (NO $unwind — top-level fields only)
    "hostel_query": IntentTemplate(
        intent_name="hostel_query",
        description="Find colleges with hostel availability",
        required_slots=[],
        optional_slots=["location", "college_type", "hostel"],
        query_fields={
            "location":     "Location",
            "college_type": "Type",
            "hostel":       "HostelAvailability",
        },
        uses_unwind=False,
        no_results_message="No hostel information found for your criteria.",
        follow_up_questions={
            "location":     "Which location are you checking for hostel?",
            "college_type": "Public or private?",
        },
    ),

    # 8 — contact_query (NO $unwind — top-level fields only)
    "contact_query": IntentTemplate(
        intent_name="contact_query",
        description="Get contact information for a college",
        required_slots=["college_name"],
        optional_slots=[],
        query_fields={
            "college_name": "Name",
        },
        uses_unwind=False,
        no_results_message="No contact information found for that college.",
        follow_up_questions={
            "college_name": "Which college's contact info do you need?",
        },
    ),

    # 9 — admission_process (NO $unwind — top-level contact info)
    "admission_process": IntentTemplate(
        intent_name="admission_process",
        description="Admission enquiry — returns contact info with admission guidance",
        required_slots=["college_name"],
        optional_slots=[],
        query_fields={
            "college_name": "Name",
        },
        uses_unwind=False,
        no_results_message="No information found for that college.",
        follow_up_questions={
            "college_name": "Which college are you asking about admission for?",
        },
    ),

    # 10 — greeting (static)
    "greeting": IntentTemplate(
        intent_name="greeting",
        description="User greeting",
    ),

    # 10 — goodbye (static)
    "goodbye": IntentTemplate(
        intent_name="goodbye",
        description="User saying goodbye",
    ),

    # Fallback
    "unknown": IntentTemplate(
        intent_name="unknown",
        description="Unrecognized intent",
    ),
}


# ============================================================================
# INTENT FAMILIES (for context / slot carryover)
# ============================================================================

INTENT_FAMILIES = {
    "search_family": [
        "search_college", "best_items_search",
        "recommend_with_constraints", "personalized_recommendation",
    ],
    "info_family": [
        "college_details", "college_attribute_query", "compare_colleges",
        "hostel_query", "contact_query", "admission_process",
    ],
    "conversational": ["greeting", "goodbye"],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_template(intent: str) -> IntentTemplate:
    return INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["unknown"])


def get_required_slots(intent: str) -> List[str]:
    return get_template(intent).required_slots


def validate_slots(intent: str, slots: Dict[str, Any]) -> tuple:
    template = get_template(intent)
    missing = template.get_missing_slots(slots)
    return len(missing) == 0, missing


def get_follow_up_question(intent: str, slots: Dict[str, Any]) -> str:
    return get_template(intent).get_follow_up(slots)


def get_intent_family(intent: str) -> str:
    for family, intents in INTENT_FAMILIES.items():
        if intent in intents:
            return family
    return "unknown"


def should_carry_slots(prev_intent: str, new_intent: str) -> bool:
    prev_fam = get_intent_family(prev_intent)
    new_fam = get_intent_family(new_intent)
    if prev_fam == new_fam and prev_fam != "conversational":
        return True
    if {prev_fam, new_fam} == {"info_family", "search_family"}:
        return True
    return False
