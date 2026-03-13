"""
Admission Handler — admission_process intent

Trigger: "how to get admission in Sagarmatha Engineering College"
Pipeline: $match → $project (NO $unwind — top-level fields only)

Returns college contact info wrapped in an admission-oriented message.
"""

from typing import Dict, List, Any
import logging

from app.templates.intent_templates import get_template
from app.core.query_builder import build_pipeline, deduplicate

logger = logging.getLogger(__name__)


def _format_admission(results: List[Dict], slots: Dict) -> str:
    if not results:
        college = slots.get("college_name", "that college")
        return (
            f"I couldn't find information for '{college}'.\n"
            "Please check the spelling and try again."
        )

    r = results[0]
    name = r.get("college_name", "College")
    location = r.get("location", "")
    phone = r.get("contact", "")
    email = r.get("email", "")

    lines = [f"**Admission Information — {name}**\n"]

    lines.append(
        "Admissions to engineering colleges in Nepal are conducted through "
        "the **IOE Entrance Examination**. After receiving your entrance rank, "
        "you can apply to this college during the official counseling rounds.\n"
    )

    lines.append("**Steps:**")
    lines.append("1. Appear for the IOE Entrance Exam")
    lines.append("2. Receive your entrance rank")
    lines.append("3. Apply during the counseling / admission window")
    lines.append("4. Contact the college directly for seat availability and deadlines\n")

    lines.append("**Contact the college for further details:**")
    if location:
        lines.append(f"  📍 Location : {location}")
    if phone:
        lines.append(f"  📞 Phone    : {phone}")
    if email:
        lines.append(f"  📧 Email    : {email}")

    if not phone and not email:
        lines.append("  Contact details not available — please visit the college office directly.")

    return "\n".join(lines)


async def handle(
    intent: str,
    slots: Dict[str, Any],
    collection,
    top_k: int = 5,
) -> Dict[str, Any]:
    template = get_template(intent)
    pipeline = build_pipeline(intent, slots, template, limit=top_k * 3)
    logger.info(f"[admission_process] pipeline: {pipeline}")

    cursor = collection.aggregate(pipeline)
    docs = await cursor.to_list(length=top_k * 3)
    results = deduplicate(docs)[:top_k]

    return {
        "intent": intent,
        "query": {"pipeline": pipeline},
        "results": results,
        "count": len(results),
        "response": _format_admission(results, slots),
    }
