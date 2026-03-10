# bio_converter.py
#
# Changes from previous version:
#   1. ATTRIBUTE no longer skipped — gets its own span finder
#      using surface form variants ("located" → "location" etc.)
#   2. HOSTEL and RATING still skipped (boolean / numeric flags)
#   3. Everything else unchanged

import json
import re

# ─────────────────────────────────────────────────────────────
# KNOWN VALUE TABLES
# ─────────────────────────────────────────────────────────────

KNOWN_LOCATIONS = {
    "lalitpur", "kathmandu", "bhaktapur",
    "pokhara", "chitwan", "dharan", "sunsari",
    "kalanki", "dhapakhel", "balkumari", "sanepa",
    "chyasal", "thapathali", "chakupat", "tathali",
    "libali", "rampur", "kalimati"
}

_SUBAREA_TO_DISTRICT = {
    "balkumari": "lalitpur", "dhapakhel": "lalitpur",
    "sanepa":    "lalitpur", "chyasal":   "lalitpur",
    "talsikhel": "lalitpur", "pulchowk":  "lalitpur",
    "chakupat":  "lalitpur",
    "kalanki":   "kathmandu", "thapathali": "kathmandu",
    "kalimati":  "kathmandu",
    "tathali":   "bhaktapur", "libali": "bhaktapur",
    "lamachaur": "pokhara",
    "rampur":    "chitwan",
}

_DISTRICT_SUBAREAS = {}
for _sub, _dist in _SUBAREA_TO_DISTRICT.items():
    _DISTRICT_SUBAREAS.setdefault(_dist, set()).add(_sub)

COLLEGE_KEYWORDS = {
    "college", "campus", "engineering", "institute",
    "university", "management", "international",
    "polytechnic", "academy", "technical"
}

KNOWN_COLLEGE_ALIASES = {
    "pulchowk", "kathford", "sagarmatha", "khwopa",
    "himalaya", "kantipur", "thapathali", "janakpur",
    "purwanchal", "pashchimanchal",
}

COURSE_MAP = {
    "be civil":       "BE Civil Engineering",
    "be computer":    "BE Computer Engineering",
    "be electrical":  "BE Electrical Engineering",
    "be electronics": "BE Electronics Engineering",
    "be mechanical":  "BE Mechanical Engineering",
}

# ── ATTRIBUTE surface forms ───────────────────────────────────
# Maps annotation value → words/phrases that represent it in text
# Order matters: longer phrases listed first for greedy matching
ATTRIBUTE_SURFACE_FORMS = {
    "location": [
        "located", "location", "situated", "address",
        "where", "place", "area", "district", "region"
    ],
    "fee": [
        "tuition fee", "total fee", "course fee",
        "fee", "fees", "cost", "tuition", "charge",
        "price", "expensive", "cheap", "how much"
    ],
    "cutoff": [
        "cutoff rank", "cut-off rank", "minimum rank",
        "required rank", "entrance rank", "rank required",
        "rank needed", "rank do i need",
        "cutoff", "cut-off", "entrance", "minimum", "required"
    ],
    "rating": [
        "rating", "rated", "score", "review", "rank"
    ],
    "seats": [
        "total seats", "number of seats",
        "seats", "seat", "intake", "capacity"
    ],
    "contact": [
        "contact number", "phone number",
        "contact", "phone", "email", "telephone", "number"
    ],
    "hostel": [
        "hostel facility", "hostel available",
        "hostel", "accommodation", "boarding", "dormitory"
    ],
}

# Entities whose values are flags/numbers that never appear
# verbatim in text → skip BIO labeling entirely
SKIP_ENTITIES = {"RATING", "HOSTEL"}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def clean_token(token):
    """Strip punctuation, lowercase — for matching only."""
    return re.sub(r'^[^\w]+|[^\w]+$', '', token).lower()


def is_valid_college_name(value):
    """
    Return True only if value looks like a real college name.
    Bare city names return False.
    """
    v = value.strip().lower()
    if not v or v in KNOWN_LOCATIONS:
        return False
    if v in KNOWN_COLLEGE_ALIASES:
        return True
    return bool(set(v.split()) & COLLEGE_KEYWORDS)


def fix_entity_type(entity_type, value):
    """
    Correct mislabeled entity types before BIO conversion.

    Rules:
      COLLEGE_NAME whose value is a bare city → LOCATION
      LOCATION whose value is a college name  → COLLEGE_NAME
    """
    v = value.strip().lower()

    if entity_type.startswith("COLLEGE_NAME"):
        if v in KNOWN_LOCATIONS and not is_valid_college_name(value):
            return "LOCATION"

    if entity_type == "LOCATION":
        if is_valid_college_name(value) and v not in KNOWN_LOCATIONS:
            return "COLLEGE_NAME"

    return entity_type


# ─────────────────────────────────────────────────────────────
# VALUE NORMALIZER
# Makes entity value match the form that appears in text
# ─────────────────────────────────────────────────────────────

def normalize_value(entity_type, value, text):
    """
    Return the version of value that actually appears in text.

    Key fixes:
      "700000"               → "7 lakhs"   (BUDGET)
      "BE Computer Engg"     → "BE Computer Engineering" (COURSE)
      "Lalitpur" not in text → try sub-areas like "Balkumari" (LOCATION)
    """
    text_lower = text.lower()
    v = str(value).strip()

    # Already in text — no change needed
    if v.lower() in text_lower:
        return v

    # ── LOCATION ─────────────────────────────────────────
    if entity_type == "LOCATION":
        subareas = _DISTRICT_SUBAREAS.get(v.lower(), set())
        for sub in subareas:
            if sub in text_lower:
                idx = text_lower.find(sub)
                return text[idx: idx + len(sub)]

    # ── BUDGET ───────────────────────────────────────────
    if entity_type == "BUDGET":
        m = re.search(r'(\d+\.?\d*)\s*(?:lakh|lakhs|l)\b', text_lower)
        if m:
            return text[m.start():m.end()].strip()
        m = re.search(r'\b(\d{4,7})\b', text)
        if m:
            return m.group(1)

    # ── RANK ─────────────────────────────────────────────
    if entity_type == "RANK":
        m = re.search(r'\b(\d{1,5})\b', text)
        if m and m.group(1) in text:
            return m.group(1)

    # ── COURSE ───────────────────────────────────────────
    if entity_type == "COURSE":
        # Short form → full form
        full = COURSE_MAP.get(v.lower())
        if full and full.lower() in text_lower:
            return full
        # Progressively shorter prefix
        parts = v.split()
        for length in range(len(parts), 0, -1):
            candidate = " ".join(parts[:length])
            if candidate.lower() in text_lower:
                return candidate
        # Subject word alone: "BE Computer Engineering" → "Computer"
        for word in reversed(parts):
            if word.lower() in ("be", "b"):
                continue
            pattern = re.compile(
                r'\b' + re.escape(word) +
                r'(?:\s+(?:engineering|science))?', re.I
            )
            m = pattern.search(text)
            if m:
                return text[m.start():m.end()]

    # ── COLLEGE_NAME ──────────────────────────────────────
    if entity_type in ("COLLEGE_NAME", "COLLEGE_NAME_1", "COLLEGE_NAME_2"):
        parts = v.split()
        for length in range(len(parts), 0, -1):
            candidate = " ".join(parts[:length])
            if candidate.lower() in text_lower:
                idx = text_lower.find(candidate.lower())
                return text[idx: idx + len(candidate)]

    return v   # fallback


# ─────────────────────────────────────────────────────────────
# SPAN FINDERS
# ─────────────────────────────────────────────────────────────

def find_span(tokens, entity_value):
    """
    Find entity span using punctuation-stripped window matching.
    Falls back to first-word match for long multi-token values.
    """
    entity_tokens = [clean_token(t) for t in entity_value.split()]
    entity_tokens = [t for t in entity_tokens if t]
    cleaned       = [clean_token(t) for t in tokens]
    length        = len(entity_tokens)

    if not length:
        return None

    # Full window match
    for i in range(len(cleaned) - length + 1):
        if cleaned[i: i + length] == entity_tokens:
            return i, i + length

    # First-word fallback (long college names)
    if length > 1:
        for i, ct in enumerate(cleaned):
            if ct == entity_tokens[0]:
                return i, i + 1

    return None


def find_span_by_offset(text, tokens, entity_value):
    """
    Character-offset fallback when token matching fails.
    """
    match = re.search(re.escape(entity_value), text, re.IGNORECASE)
    if not match and entity_value.split():
        match = re.search(
            re.escape(entity_value.split()[0]), text, re.IGNORECASE
        )
    if not match:
        return None

    char_start  = match.start()
    char_end    = match.end()
    token_start = None
    token_end   = None
    pos         = 0

    for i, token in enumerate(tokens):
        s = text.find(token, pos)
        if s == -1:
            continue
        e = s + len(token)
        if token_start is None and e > char_start:
            token_start = i
        if e <= char_end:
            token_end = i + 1
        pos = e

    if token_start is not None and token_end is not None:
        return token_start, token_end
    return None


def find_attribute_span(tokens, attribute_value, text):
    """
    Find ATTRIBUTE span using surface form matching.

    ATTRIBUTE annotation values ("location", "fee", "cutoff")
    often don't appear verbatim in text:
      annotation: "location"  text: "located" / "where" / "address"
      annotation: "fee"       text: "fee" / "cost" / "how much"
      annotation: "cutoff"    text: "cutoff rank" / "minimum rank"

    Strategy:
      1. Exact value match         ("fee" → "fee" in text)
      2. Multi-word surface forms  ("minimum rank" → 2-token span)
      3. Single-word surface forms ("located" → 1-token span)

    Returns (start_idx, end_idx) or None.
    """
    attr_lower    = attribute_value.lower().strip()
    text_lower    = text.lower()
    cleaned       = [clean_token(t) for t in tokens]

    # Strategy 1: exact value already in text
    if attr_lower in text_lower:
        for i, ct in enumerate(cleaned):
            if ct == attr_lower:
                return i, i + 1

    # Strategy 2 + 3: surface form variants
    surface_forms = ATTRIBUTE_SURFACE_FORMS.get(attr_lower, [])

    # Sort DESC by word count — try longer phrases first
    # "cutoff rank" before "cutoff" to get the best span
    surface_forms_sorted = sorted(
        surface_forms,
        key=lambda x: len(x.split()),
        reverse=True
    )

    for form in surface_forms_sorted:
        form_tokens = [clean_token(t) for t in form.split()]
        form_len    = len(form_tokens)

        if form_len == 1:
            # Single word
            for i, ct in enumerate(cleaned):
                if ct == form_tokens[0]:
                    return i, i + 1
        else:
            # Multi-word phrase
            for i in range(len(cleaned) - form_len + 1):
                if cleaned[i: i + form_len] == form_tokens:
                    return i, i + form_len

    return None


# ─────────────────────────────────────────────────────────────
# SAMPLE CONVERTER
# ─────────────────────────────────────────────────────────────

def convert_sample(sample):
    """
    Convert one sample to BIO token-label format.

    Handles both dataset formats:
      Training : "intent"        + "entities"
      Test     : "actual_intent" + "actual_entities"

    Entity processing order:
      1. Skip RATING, HOSTEL  (boolean/numeric flags, not spans)
      2. ATTRIBUTE             → find_attribute_span() with surface forms
      3. All other entities    → fix_entity_type() → normalize_value()
                                 → find_span() → find_span_by_offset()

    Overlap guard prevents two entities from labeling the same token.
    Longer entities matched first to prevent partial overlaps.
    """
    text   = sample["text"]
    intent = sample.get("intent") or sample.get("actual_intent", "")
    entities = (
        sample.get("entities")
        or sample.get("actual_entities")
        or {}
    )

    tokens  = text.split()
    labels  = ["O"] * len(tokens)
    labeled = set()
    failed  = []

    # Sort by value length DESC — longer entities matched first
    sorted_entities = sorted(
        entities.items(),
        key=lambda x: len(str(x[1]).split()),
        reverse=True
    )

    for entity_type, value in sorted_entities:
        if not value:
            continue

        # ── Skip boolean/numeric flag entities ────────
        if entity_type in SKIP_ENTITIES:
            continue

        # ── ATTRIBUTE: special surface form matching ──
        if entity_type == "ATTRIBUTE":
            span = find_attribute_span(tokens, str(value), text)
            if span is None:
                failed.append(f"ATTRIBUTE:'{value}'")
                continue

            start, end = span
            positions  = set(range(start, end))

            if positions & labeled:
                continue   # overlap — skip

            labels[start] = "B-ATTRIBUTE"
            for i in range(start + 1, end):
                labels[i] = "I-ATTRIBUTE"
            labeled.update(positions)
            continue       # done — don't fall through

        # ── All other entities ────────────────────────

        # Step 1: Correct mislabeled entity types
        entity_type = fix_entity_type(entity_type, str(value))

        # Step 2: Normalize value to match text form
        value = normalize_value(entity_type, str(value), text)

        # Step 3: Find span
        span = find_span(tokens, value)
        if span is None:
            span = find_span_by_offset(text, tokens, value)
        if span is None:
            failed.append(f"{entity_type}:'{value}'")
            continue

        start, end = span

        # Overlap guard
        positions = set(range(start, end))
        if positions & labeled:
            continue

        # Assign BIO labels
        labels[start] = f"B-{entity_type}"
        for i in range(start + 1, end):
            labels[i] = f"I-{entity_type}"
        labeled.update(positions)

    return {
        "tokens": tokens,
        "labels": labels,
        "intent": intent,
        "text":   text,
        "failed": failed
    }


# ─────────────────────────────────────────────────────────────
# DATASET CONVERTER
# ─────────────────────────────────────────────────────────────

def convert_dataset(input_file, output_file):
    """
    Convert full dataset → BIO JSON.
    Handles both list format and {"sentences": [...]} format.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw if isinstance(raw, list) else \
           raw.get("sentences", raw.get("samples", []))

    bio_samples  = []
    total_failed = []

    for sample in data:
        converted = convert_sample(sample)
        bio_samples.append(converted)
        if converted["failed"]:
            total_failed.extend(converted["failed"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"samples": bio_samples}, f, indent=2, ensure_ascii=False)

    # ── Report ────────────────────────────────────────
    from collections import Counter
    fail_counts = Counter(
        f.split(":")[0] for f in total_failed
    )

    print(f"Converted   : {len(bio_samples)} samples")
    print(f"Failed spans: {len(total_failed)}")

    if fail_counts:
        print("Failed by entity type:")
        for etype, count in fail_counts.most_common():
            print(f"  {etype:<25}: {count}")


# ─────────────────────────────────────────────────────────────
# QUICK VERIFIER
# ─────────────────────────────────────────────────────────────

def verify(sample):
    """Print token-label table for one sample."""
    result = convert_sample(sample)
    print(f"Text  : {result['text']}")
    print(f"Intent: {result['intent']}")
    print()
    print(f"  {'Token':<28} Label")
    print(f"  {'-'*45}")
    for token, label in zip(result["tokens"], result["labels"]):
        marker = "  ←" if label != "O" else ""
        print(f"  {token:<28} {label}{marker}")
    if result["failed"]:
        print(f"\n  Failed: {result['failed']}")
    print()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    convert_dataset("new_data_collection.json", "bio_dataset.json")

    # ── Verify the 3 ATTRIBUTE patterns ──────────────────
    print("\n" + "=" * 50)
    print("ATTRIBUTE verification")
    print("=" * 50 + "\n")

    verify({
        "text":   "Where is Sagarmatha Engineering College located",
        "intent": "college_attribute_query",
        "entities": {
            "COLLEGE_NAME": "Sagarmatha Engineering College",
            "ATTRIBUTE":    "location"
        }
    })

    verify({
        "text":   "What is the fee of BE Civil Engineering at Pulchowk Engineering Campus",
        "intent": "college_attribute_query",
        "entities": {
            "COLLEGE_NAME": "Pulchowk Engineering Campus",
            "COURSE":       "BE Civil Engineering",
            "ATTRIBUTE":    "fee"
        }
    })

    verify({
        "text":   "What is the cutoff rank for BE Computer at Kathford International College",
        "intent": "college_attribute_query",
        "entities": {
            "COLLEGE_NAME": "Kathford International College",
            "COURSE":       "BE Computer Engineering",
            "ATTRIBUTE":    "cutoff"
        }
    })

    verify({
        "text":   "What rank do I need for BE Civil at Kantipur Engineering College",
        "intent": "college_attribute_query",
        "entities": {
            "COLLEGE_NAME": "Kantipur Engineering College",
            "COURSE":       "BE Civil Engineering",
            "ATTRIBUTE":    "cutoff"
        }
    })

    verify({
        "text":   "How much does BE Computer Engineering cost at Himalaya College of Engineering",
        "intent": "college_attribute_query",
        "entities": {
            "COLLEGE_NAME": "Himalaya College of Engineering",
            "COURSE":       "BE Computer Engineering",
            "ATTRIBUTE":    "fee"
        }
    })