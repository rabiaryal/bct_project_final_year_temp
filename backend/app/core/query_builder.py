"""
Query Builder — builds MongoDB aggregation pipelines.

Extracted from intent_handlers.py so handlers can build pipelines
without duplicating low-level MongoDB logic.
"""

from typing import Dict, List, Any
import json
import logging

from app.templates.intent_templates import IntentTemplate, get_template

logger = logging.getLogger(__name__)


# ============================================================================
# PROJECTION HELPERS
# ============================================================================

def _unwind_projection() -> Dict[str, Any]:
    return {
        "_id": 0,
        "college_name":  "$Name",
        "location":      "$Location",
        "college_type":  "$Type",
        "hostel":        "$HostelAvailability",
        "contact":       "$ContactNumber",
        "email":         "$Email",
        "department":    "$Departments.Name",
        "course":        "$Departments.Courses.Name",
        "fee":           "$Departments.Courses.Fee",
        "rating":        "$Departments.Courses.Rating",
        "cutoff_rank":   "$Departments.Courses.AverageCutoffRank",
    }


def _top_level_projection() -> Dict[str, Any]:
    return {
        "_id": 0,
        "college_name":  "$Name",
        "location":      "$Location",
        "college_type":  "$Type",
        "hostel":        "$HostelAvailability",
        "contact":       "$ContactNumber",
        "email":         "$Email",
    }


# ============================================================================
# $match BUILDER
# ============================================================================

def _build_match(slots: Dict[str, Any], template: IntentTemplate) -> Dict[str, Any]:
    match: Dict[str, Any] = {}
    for slot_name, db_field in template.query_fields.items():
        value = slots.get(slot_name)
        if value is None:
            continue
        if slot_name == "rating":
            continue
        if slot_name == "rank":
            match[db_field] = {"$gte": int(value)}
        elif slot_name == "budget":
            match[db_field] = {"$lte": int(value)}
        elif slot_name == "hostel":
            match[db_field] = bool(value)
        else:
            match[db_field] = {"$regex": str(value), "$options": "i"}
    return match


# ============================================================================
# PIPELINE BUILDERS
# ============================================================================

def _build_unwind_pipeline(
    match: Dict[str, Any],
    template: IntentTemplate,
    slots: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    pipeline: List[Dict[str, Any]] = [
        {"$unwind": "$Departments"},
        {"$unwind": "$Departments.Courses"},
    ]
    if match:
        pipeline.append({"$match": match})
    pipeline.append({"$project": _unwind_projection()})
    if template.sort_field:
        pipeline.append({"$sort": {template.sort_field: template.sort_order}})
    pipeline.append({"$limit": limit})
    return pipeline


def _build_compare_pipeline(
    slots: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    name_1 = slots.get("college_name_1", "")
    name_2 = slots.get("college_name_2", "")
    match = {"$or": [
        {"Name": {"$regex": str(name_1), "$options": "i"}},
        {"Name": {"$regex": str(name_2), "$options": "i"}},
    ]}
    return [
        {"$match": match},
        {"$unwind": "$Departments"},
        {"$unwind": "$Departments.Courses"},
        {"$project": _unwind_projection()},
        {"$limit": limit},
    ]


def _build_top_level_pipeline(
    match: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    pipeline: List[Dict[str, Any]] = []
    if match:
        pipeline.append({"$match": match})
    pipeline.append({"$project": _top_level_projection()})
    pipeline.append({"$limit": limit})
    return pipeline


def build_pipeline(
    intent: str,
    slots: Dict[str, Any],
    template: IntentTemplate,
    limit: int = 15,
) -> List[Dict[str, Any]]:
    """Dispatch to the correct pipeline builder."""
    match = _build_match(slots, template)
    if intent == "compare_colleges":
        return _build_compare_pipeline(slots, limit)
    if not template.uses_unwind:
        pipeline = _build_top_level_pipeline(match, limit)
    else:
        pipeline = _build_unwind_pipeline(match, template, slots, limit)
    logger.debug("[build_pipeline] intent=%s  pipeline=%s", intent, json.dumps(pipeline, default=str))
    return pipeline


def build_candidate_pipeline(slots: Dict[str, Any], limit: int = 15) -> List[Dict[str, Any]]:
    """
    Build a hard-filter pipeline for hybrid handlers.

    Applies: course regex, location regex, college_type exact,
             budget (fee <= user_budget).
    Rank uses a lenient buffer (50% below user rank) so near-miss
    candidates reach the Python scorer, which handles fine-grained
    rank weighting and safety labels (SAFE/MODERATE/RISKY).
    Returns up to `limit` candidates for Python-side scoring.
    """
    match: Dict[str, Any] = {}

    if slots.get("course"):
        match["Departments.Courses.Name"] = {
            "$regex": str(slots["course"]),
            "$options": "i",
        }
    if slots.get("location"):
        match["Location"] = {
            "$regex": str(slots["location"]),
            "$options": "i",
        }
    if slots.get("college_type"):
        match["Type"] = str(slots["college_type"]).upper()
    if slots.get("rank"):
        # Lenient filter: accept courses whose cutoff is at least 50%
        # of the user's rank.  The Python scorer handles exact ranking.
        lenient_rank = max(1, int(int(slots["rank"]) * 0.5))
        match["Departments.Courses.AverageCutoffRank"] = {
            "$gte": lenient_rank,
        }
    if slots.get("budget"):
        match["Departments.Courses.Fee"] = {
            "$lte": int(slots["budget"]),
        }

    pipeline = [
        {"$unwind": "$Departments"},
        {"$unwind": "$Departments.Courses"},
        {"$match": match} if match else None,
        {"$project": _unwind_projection()},
        {"$sort": {"rating": -1}},
        {"$limit": limit},
    ]
    result = [stage for stage in pipeline if stage is not None]
    logger.debug("[build_candidate_pipeline] pipeline=%s", json.dumps(result, default=str))
    return result


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in results:
        key = str(r.get("college_name", "")).lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


class QueryBuilder:
    """Stateless helper — builds pipelines from intent + slots."""

    @staticmethod
    def build(intent: str, slots: Dict[str, Any], limit: int = 15) -> List[Dict[str, Any]]:
        template = get_template(intent)
        return build_pipeline(intent, slots, template, limit)

    @staticmethod
    def build_candidates(slots: Dict[str, Any], limit: int = 15) -> List[Dict[str, Any]]:
        return build_candidate_pipeline(slots, limit)
