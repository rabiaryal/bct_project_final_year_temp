"""
Formatter — per-intent response formatters for non-hybrid handlers.

Hybrid handlers (recommend_handler, personal_handler) have their own
inline formatters; the functions here serve the remaining 8 intents.
"""

from typing import Dict, List, Any


def format_search_college(results: List[Dict], slots: Dict) -> str:
    if not results:
        return "No colleges found matching your criteria. Try broadening your search."

    parts = []
    course = slots.get("course", "")
    location = slots.get("location", "")
    header = "Colleges"
    if course:
        header += f" offering {course}"
    if location:
        header += f" in {location}"

    parts.append(f"**{header}**\n")
    parts.append(f"Found {len(results)} result(s):\n")

    for i, r in enumerate(results[:5], 1):
        name = r.get("college_name", "Unknown")
        loc = r.get("location", "")
        ctype = r.get("college_type", "")
        c = r.get("course", "")
        fee = r.get("fee", 0)
        rating = r.get("rating", 0)

        parts.append(f"**{i}. {name}**")
        info = []
        if loc:
            info.append(f"Location: {loc}")
        if ctype:
            info.append(f"Type: {ctype.title()}")
        if c:
            info.append(f"Course: {c}")
        if fee:
            info.append(f"Fee: NPR {fee:,.0f}")
        if rating:
            info.append(f"Rating: {rating}/5")
        parts.append("   " + " | ".join(info))
        parts.append("")

    return "\n".join(parts)


def format_best_items_search(results: List[Dict], slots: Dict) -> str:
    if not results:
        return "No colleges found. Try broadening your search."

    course = slots.get("course", "")
    location = slots.get("location", "")
    header = "Best-Rated Colleges"
    if course:
        header += f" for {course}"
    if location:
        header += f" in {location}"

    parts = [f"**{header}**\n"]

    for i, r in enumerate(results[:5], 1):
        medal = ["1st", "2nd", "3rd"][i - 1] if i <= 3 else f"{i}th"
        name = r.get("college_name", "Unknown")
        rating = r.get("rating", 0)
        fee = r.get("fee", 0)
        loc = r.get("location", "")
        c = r.get("course", "")

        parts.append(f"**{medal}. {name}**")
        info = []
        if rating:
            info.append(f"Rating: {rating}/5")
        if c:
            info.append(f"Course: {c}")
        if fee:
            info.append(f"Fee: NPR {fee:,.0f}")
        if loc:
            info.append(f"Location: {loc}")
        parts.append("   " + " | ".join(info))
        parts.append("")

    return "\n".join(parts)


def format_compare_colleges(results: List[Dict], slots: Dict) -> str:
    if not results:
        return "Could not find one or both colleges. Please check the names."

    name_1 = str(slots.get("college_name_1", "")).lower()
    name_2 = str(slots.get("college_name_2", "")).lower()

    group_a: List[Dict] = []
    group_b: List[Dict] = []
    for r in results:
        cn = str(r.get("college_name", "")).lower()
        if name_1 and name_1 in cn:
            group_a.append(r)
        elif name_2 and name_2 in cn:
            group_b.append(r)

    parts = [f"**Comparison: {slots.get('college_name_1', '?')} vs {slots.get('college_name_2', '?')}**\n"]

    def _block(label: str, rows: List[Dict]) -> List[str]:
        lines = [f"**{label}**"]
        if not rows:
            lines.append("   (no data found)")
            return lines
        first = rows[0]
        lines.append(f"   Location: {first.get('location', 'N/A')} | Type: {first.get('college_type', 'N/A')}")
        hostel_str = "Yes" if first.get("hostel") else "No"
        lines.append(f"   Hostel: {hostel_str} | Contact: {first.get('contact', 'N/A')}")
        lines.append("   Courses:")
        for r in rows[:5]:
            c = r.get("course", "N/A")
            fee = r.get("fee", 0)
            rating = r.get("rating", 0)
            cutoff = r.get("cutoff_rank", 0)
            info = []
            if fee:
                info.append(f"Fee: NPR {fee:,.0f}")
            if rating:
                info.append(f"Rating: {rating}/5")
            if cutoff:
                info.append(f"Cutoff: {cutoff}")
            detail = " | ".join(info) if info else ""
            lines.append(f"     - {c}  {detail}")
        return lines

    parts.extend(_block(f"College A: {slots.get('college_name_1', '?')}", group_a))
    parts.append("")
    parts.extend(_block(f"College B: {slots.get('college_name_2', '?')}", group_b))
    return "\n".join(parts)


def format_college_details(results: List[Dict], slots: Dict) -> str:
    if not results:
        college = slots.get("college_name", "that college")
        return f"I couldn't find information about '{college}'. Please check the spelling."

    first = results[0]
    name = first.get("college_name", "College")
    parts = [f"**{name}**\n"]

    if first.get("location"):
        parts.append(f"Location: {first['location']}")
    if first.get("college_type"):
        parts.append(f"Type: {first['college_type'].title()}")
    hostel_str = "Available" if first.get("hostel") else "Not Available"
    parts.append(f"Hostel: {hostel_str}")
    if first.get("contact"):
        parts.append(f"Contact: {first['contact']}")
    if first.get("email"):
        parts.append(f"Email: {first['email']}")

    seen: set = set()
    course_lines = []
    for r in results[:10]:
        c = r.get("course", "")
        if c and c not in seen:
            seen.add(c)
            fee = r.get("fee", 0)
            rating = r.get("rating", 0)
            seats = r.get("total_seats", 0)
            info = []
            if fee:
                info.append(f"NPR {fee:,.0f}")
            if rating:
                info.append(f"Rating {rating}/5")
            if seats:
                info.append(f"{seats} seats")
            detail = " | ".join(info) if info else ""
            course_lines.append(f"   - {c}  {detail}")

    if course_lines:
        parts.append("\nPrograms Offered:")
        parts.extend(course_lines[:8])
        if len(course_lines) > 8:
            parts.append(f"   ... and {len(course_lines) - 8} more")

    return "\n".join(parts)


def format_hostel_query(results: List[Dict], slots: Dict) -> str:
    if not results:
        return "No hostel information found for your criteria."

    parts = ["**Hostel Availability**\n"]
    for i, r in enumerate(results[:5], 1):
        name = r.get("college_name", "Unknown")
        loc = r.get("location", "")
        hostel = r.get("hostel", False)
        status = "Available" if hostel else "Not Available"
        parts.append(f"**{i}. {name}**")
        info = [f"Hostel: {status}"]
        if loc:
            info.append(f"Location: {loc}")
        parts.append("   " + " | ".join(info))
        parts.append("")

    return "\n".join(parts)


def format_contact_query(results: List[Dict], slots: Dict) -> str:
    if not results:
        college = slots.get("college_name", "that college")
        return f"No contact information found for '{college}'."

    r = results[0]
    name = r.get("college_name", "College")
    parts = [f"**Contact Information for {name}**\n"]
    if r.get("location"):
        parts.append(f"Location: {r['location']}")
    if r.get("contact"):
        parts.append(f"Phone: {r['contact']}")
    if r.get("email"):
        parts.append(f"Email: {r['email']}")
    if not r.get("contact") and not r.get("email"):
        parts.append("Contact details not available. Please visit the college website.")

    return "\n".join(parts)


def format_greeting(results: List[Dict], slots: Dict) -> str:
    return (
        "**Hello! Welcome to the College Recommendation System!**\n\n"
        "I can help you with:\n"
        "- Finding colleges by location, course, or type\n"
        "- Getting fee and rating information\n"
        "- Personalized recommendations based on rank and budget\n"
        "- Comparing two colleges side by side\n"
        "- Hostel availability and contact information\n\n"
        "**How can I help you today?**"
    )


def format_goodbye(results: List[Dict], slots: Dict) -> str:
    return (
        "**Goodbye!**\n\n"
        "Thank you for using the College Recommendation System.\n"
        "Good luck with your college search!\n\n"
        "Feel free to come back anytime."
    )


# ── Attribute-level answer ─────────────────────────────────────

_ATTRIBUTE_KEYWORDS = {
    "location":  ["location", "located", "where", "address", "place", "city"],
    "contact":   ["contact", "phone", "number", "call"],
    "email":     ["email", "mail"],
    "hostel":    ["hostel", "accommodation", "dorm", "dormitory", "stay"],
    "type":      ["type", "public", "private", "government"],
    "course":    ["course", "program", "offer", "department", "faculty"],
    "fee":       ["fee", "cost", "price", "tuition", "charge", "expensive", "cheap"],
    "rating":    ["rating", "rated", "rank", "ranking", "best"],
    "seats":     ["seat", "seats", "intake", "capacity"],
}


def _resolve_attribute(attr: str) -> str:
    """Map the raw attribute text to a canonical key."""
    attr_lower = attr.lower()
    for key, keywords in _ATTRIBUTE_KEYWORDS.items():
        if any(kw in attr_lower for kw in keywords):
            return key
    return "general"


def format_attribute_query(results: List[Dict], slots: Dict) -> str:
    college = slots.get("college_name", "that college")
    if not results:
        return f"I couldn't find information about '{college}'. Please check the spelling."

    first = results[0]
    name = first.get("college_name", college)
    attr = slots.get("attribute", "")
    key = _resolve_attribute(attr) if attr else "general"

    if key == "location":
        loc = first.get("location", "N/A")
        return f"**{name}** is located at **{loc}**."

    if key == "contact":
        phone = first.get("contact", "N/A")
        return f"The contact number for **{name}** is **{phone}**."

    if key == "email":
        email = first.get("email", "N/A")
        return f"The email for **{name}** is **{email}**."

    if key == "hostel":
        avail = "available" if first.get("hostel") else "not available"
        return f"Hostel at **{name}** is **{avail}**."

    if key == "type":
        ctype = first.get("college_type", "N/A")
        return f"**{name}** is a **{ctype.title()}** college."

    if key == "fee":
        seen = set()
        lines = [f"**Fee structure for {name}:**\n"]
        for r in results[:8]:
            c = r.get("course", "")
            if c and c not in seen:
                seen.add(c)
                fee = r.get("fee", 0)
                lines.append(f"- {c}: NPR {fee:,.0f}" if fee else f"- {c}: N/A")
        return "\n".join(lines) if len(lines) > 1 else f"Fee information for **{name}** is not available."

    if key == "rating":
        seen = set()
        lines = [f"**Ratings for {name}:**\n"]
        for r in results[:8]:
            c = r.get("course", "")
            if c and c not in seen:
                seen.add(c)
                rating = r.get("rating", 0)
                lines.append(f"- {c}: {rating}/5" if rating else f"- {c}: N/A")
        return "\n".join(lines) if len(lines) > 1 else f"Rating info for **{name}** is not available."

    if key == "course":
        seen = set()
        lines = [f"**Programs offered at {name}:**\n"]
        for r in results[:10]:
            c = r.get("course", "")
            if c and c not in seen:
                seen.add(c)
                lines.append(f"- {c}")
        return "\n".join(lines) if len(lines) > 1 else f"No course information found for **{name}**."

    if key == "seats":
        seen = set()
        lines = [f"**Seat capacity at {name}:**\n"]
        for r in results[:8]:
            c = r.get("course", "")
            if c and c not in seen:
                seen.add(c)
                seats = r.get("total_seats", 0)
                lines.append(f"- {c}: {seats} seats" if seats else f"- {c}: N/A")
        return "\n".join(lines) if len(lines) > 1 else f"Seat info for **{name}** is not available."

    # General fallback — return full details
    return format_college_details(results, slots)


def format_unknown(results: List[Dict], slots: Dict) -> str:
    return (
        "I'm not sure I understood that.\n\n"
        "Could you please rephrase? Here are some examples:\n"
        '- "Find colleges in Pokhara"\n'
        '- "Best colleges for computer engineering"\n'
        '- "Recommend me a college with rank 2500 and budget 7 lakhs"\n'
        '- "Compare Pulchowk and Thapathali"\n'
        '- "Does Pulchowk Campus have hostel?"'
    )


# ============================================================================
# DISPATCHER  (used by router.py for static / conversational intents)
# ============================================================================

_FORMATTERS = {
    "search_college":           format_search_college,
    "best_items_search":        format_best_items_search,
    "compare_colleges":         format_compare_colleges,
    "college_details":          format_college_details,
    "college_attribute_query":  format_attribute_query,
    "hostel_query":             format_hostel_query,
    "contact_query":            format_contact_query,
    "greeting":                 format_greeting,
    "goodbye":                  format_goodbye,
    "unknown":                  format_unknown,
}


def format_response(intent: str, results: List[Dict], slots: Dict) -> str:
    """Dispatch to the correct per-intent formatter."""
    fn = _FORMATTERS.get(intent)
    if fn:
        return fn(results, slots)
    # Generic fallback
    if not results:
        return "No results found for your query."
    parts = [f"**Results for {intent.replace('_', ' ')}**\n"]
    for i, r in enumerate(results[:5], 1):
        name = r.get("college_name", "Unknown")
        parts.append(f"**{i}. {name}**")
        for key, value in r.items():
            if key != "college_name" and value:
                parts.append(f"   {key}: {value}")
        parts.append("")
    return "\n".join(parts)
