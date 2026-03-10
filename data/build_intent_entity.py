"""
build_intent_entity.py
======================
Generates intent_entity.json from scratch using the 8-intent / 8-entity
schema aligned to the Nepal IOE engineering college database.

Intents
-------
  search_college              COURSE + LOCATION + COLLEGE_TYPE
  best_items_search           COURSE + LOCATION  (sort by Rating)
  recommend_with_constraints  COURSE + RANK + BUDGET + LOCATION
  personalized_recommendation RANK + BUDGET + COURSE + LOCATION
  compare_colleges            COLLEGE_NAME_1 + COLLEGE_NAME_2
  college_details             COLLEGE_NAME  [+ COURSE]
  hostel_query                LOCATION + COLLEGE_TYPE
  contact_query               COLLEGE_NAME
  greeting                    (no entities)
  goodbye                     (no entities)

Run from project root:
    python data/build_intent_entity.py
"""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# 0.  Paths
# ─────────────────────────────────────────────
BASE = Path(__file__).parent
INTENT_FILE  = BASE / "intent" / "intent_data.json"
ENTITY_FILE  = BASE / "entity" / "balanced_data_700.json"
COLLEGE_FILE = BASE / "colleges_lower.json"
OUTPUT_FILE  = BASE / "intent_entity.json"


# ─────────────────────────────────────────────
# 1.  Load raw data
# ─────────────────────────────────────────────
with open(INTENT_FILE, encoding="utf-8")  as f: intent_raw  = json.load(f)
with open(ENTITY_FILE, encoding="utf-8")  as f: entity_raw  = json.load(f)
with open(COLLEGE_FILE, encoding="utf-8") as f: colleges     = json.load(f)

intent_sentences = intent_raw["sentences"]   # list[{text, intent:[...]}]
entity_sentences = entity_raw["sentences"]   # list[{text, entities:{...}}]


# ─────────────────────────────────────────────
# 2.  Build lookup dictionaries from real data
# ─────────────────────────────────────────────

# --- College names (from colleges_lower.json, title-cased) ----------------
COLLEGE_NAMES: list[str] = []
LOCATIONS: set[str]      = set()
PROGRAMS: set[str]       = set()
AFFILIATIONS: set[str]   = set(["TU", "Tribhuvan University", "PU",
                                  "Pokhara University", "Purbanchal University",
                                  "KU", "Kathmandu University", "CTEVT"])

for col in colleges:
    name = col.get("Name", "").title()
    if name:
        COLLEGE_NAMES.append(name)
    loc = col.get("Location", "")
    if loc:
        for part in loc.split(","):
            p = part.strip().title()
            if p and len(p) > 3:
                LOCATIONS.add(p)
    for dept in col.get("Departments", []):
        for course in dept.get("Courses", []):
            prog = course.get("Name", "")
            if prog:
                PROGRAMS.add(prog.title())

# Add well-known locations found in entity corpus
_extra_locations = [
    "Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Chitwan",
    "Biratnagar", "Dharan", "Butwal", "Nepalgunj", "Birgunj",
    "Dhankuta", "Bharatpur", "Hetauda", "Dhading", "Kavre",
    "Thapathali", "Pulchowk", "Kupondol", "Rampur", "Dang",
    "Palpa", "Bagmati Province", "Gandaki Province", "Lumbini Province",
]
LOCATIONS.update(_extra_locations)

# Sort longest-first so greedy matching picks most specific span
COLLEGE_NAMES_SORTED = sorted(set(COLLEGE_NAMES), key=len, reverse=True)
LOCATIONS_SORTED     = sorted(LOCATIONS,           key=len, reverse=True)
PROGRAMS_SORTED      = sorted(PROGRAMS,            key=len, reverse=True)

# Additional known program tokens from entity corpus
_extra_programs = [
    "BE Civil", "BE Computer", "BE Electrical", "BE Electronics",
    "BE Software", "BE Mechanical", "BE Architecture", "B.Arch",
    "BME", "BE Biomedical", "BE Geo-Informatics", "BE Industrial",
    "BE Chemical", "M.Sc.", "M.Sc. Water Resources", "Master's",
    "MBA", "BE", "B.E.",
]
for p in _extra_programs:
    PROGRAMS.add(p)
PROGRAMS_SORTED = sorted(PROGRAMS, key=len, reverse=True)

# Collect entity VALUES visible in bold from entity corpus
_entity_value_pool: dict[str, set[str]] = defaultdict(set)
for rec in entity_sentences:
    for etype, val in rec.get("entities", {}).items():
        if isinstance(val, str) and len(val) > 1:
            _entity_value_pool[etype].add(val)

# Merge college names seen in entity data
for val in _entity_value_pool.get("COLLEGE_NAME", set()):
    COLLEGE_NAMES_SORTED = sorted(
        set(COLLEGE_NAMES_SORTED) | {val}, key=len, reverse=True
    )
for val in _entity_value_pool.get("LOCATION", set()):
    if len(val) > 3 and not val.lower().startswith(("address", "yes", "no")):
        LOCATIONS_SORTED = sorted(
            set(LOCATIONS_SORTED) | {val}, key=len, reverse=True
        )
for val in _entity_value_pool.get("PROGRAM", set()):
    PROGRAMS_SORTED = sorted(
        set(PROGRAMS_SORTED) | {val}, key=len, reverse=True
    )
for val in _entity_value_pool.get("AFFILIATION", set()):
    if val.lower() not in ("yes", "no"):
        AFFILIATIONS.add(val)

AFFILIATIONS_SORTED = sorted(AFFILIATIONS, key=len, reverse=True)


# ─────────────────────────────────────────────
# 3.  Helper utilities
# ─────────────────────────────────────────────

def clean_bold(text: str) -> str:
    """Remove **…** markdown bold markers."""
    return re.sub(r"\*\*", "", text)


def normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for dedup."""
    t = unicodedata.normalize("NFKD", text).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------
# Bold-span entity extractor  (for entity-annotated records)
# ------------------------------------------------------------------
_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")

def extract_bold_entities(text: str, entities: dict) -> dict:
    """
    Return entities dict with cleaned values (no ** around values).
    Also derives compound entities that are sometimes embedded in text.
    """
    cleaned = {}
    for etype, val in entities.items():
        if isinstance(val, str):
            cleaned[etype] = val.strip()
        else:
            cleaned[etype] = val
    return cleaned


# ------------------------------------------------------------------
# Intent → entity inference (for intent-only records)
# ------------------------------------------------------------------
_FEE_KW       = re.compile(r"\b(fee|fees|cost|tuition|charges?|price|amount|rupees?|lakhs?)\b", re.I)
_SCHOLAR_KW   = re.compile(r"\b(scholarship|scholarships|financial aid|stipend|bursary)\b", re.I)
_HOSTEL_KW    = re.compile(r"\b(hostel|dormitory|accommodation|boarding|residence)\b", re.I)
_CONTACT_KW   = re.compile(r"\b(contact|phone|number|email|address|website|call|reach)\b", re.I)
_SEATS_KW     = re.compile(r"\b(seats?|vacancy|vacancies|openings?|available)\b", re.I)
_LOC_KW       = re.compile(r"\b(location|located|address|where|near|district|city|place)\b", re.I)
_PROG_KW      = re.compile(r"\b(program|programme|course|degree|curriculum|subjects?)\b", re.I)
_ADM_KW       = re.compile(r"\b(admission|apply|application|enroll|deadline|procedure|process)\b", re.I)
_PASS_KW      = re.compile(r"\b(pass(\s+percentage)?|percentage|result|success rate)\b", re.I)
_RATING_KW    = re.compile(r"\b(rating|rank(ing)?|rated|best|top|star)\b", re.I)
_RECOMMEND_KW = re.compile(r"\b(recommend|suggest|good|best|which college)\b", re.I)
_TYPE_KW      = re.compile(r"\b(private|government|public|autonomous|constituent)\b", re.I)
_AFFIL_KW     = re.compile(r"\b(affiliated|affiliation|under|university)\b", re.I)
_DEPT_KW      = re.compile(r"\b(department|dept|engineering department)\b", re.I)
_INTERN_KW    = re.compile(r"\b(internship|intern|placement|industry)\b", re.I)
_CUTOFF_KW    = re.compile(r"\b(cutoff|cut.off|rank|merit list)\b", re.I)
_FACILITY_KW  = re.compile(r"\b(facilit(y|ies)|lab(oratory)?|library|sports|canteen|wifi|infrastructure)\b", re.I)


def find_span(text: str, candidates: list[str]) -> str | None:
    """Return first candidate whose full name appears in text (case-insensitive)."""
    for c in candidates:
        if re.search(re.escape(c), text, re.IGNORECASE):
            return c
    return None


def extract_entities_from_intent_text(text: str, intent: str) -> dict:
    """
    Rule-based entity extraction from plain text using keyword patterns
    and lookup dictionaries.
    """
    ents: dict[str, str] = {}

    # ---------- structural lookups ----------
    college = find_span(text, COLLEGE_NAMES_SORTED)
    if college:
        ents["COLLEGE_NAME"] = college

    location = find_span(text, LOCATIONS_SORTED)
    if location and location.lower() not in {"address"}:
        ents["LOCATION"] = location

    program = find_span(text, PROGRAMS_SORTED)
    if program:
        ents["PROGRAM"] = program

    affil = find_span(text, AFFILIATIONS_SORTED)
    if affil:
        ents["AFFILIATION"] = affil

    # ---------- keyword-based entities ----------
    if _FEE_KW.search(text):
        fee_hint = "Yes"
        low_m  = re.search(r"\b(low|cheap|less|affordable|minimum|min)\b", text, re.I)
        high_m = re.search(r"\b(high|expensive|maximum|max|costly)\b", text, re.I)
        med_m  = re.search(r"\b(medium|moderate|average)\b", text, re.I)
        rng_m  = re.search(r"less\s+than\s+[\d\s]+[Ll]akhs?", text, re.I)
        if rng_m:   fee_hint = rng_m.group(0).title()
        elif low_m:  fee_hint = "Low"
        elif high_m: fee_hint = "High"
        elif med_m:  fee_hint = "Medium"
        ents["FEE"] = fee_hint

    if _SCHOLAR_KW.search(text):
        ents["SCHOLARSHIP"] = "Yes"

    if _HOSTEL_KW.search(text):
        ents["HOSTEL_AVAILABILITY"] = "Yes"

    if _CONTACT_KW.search(text):
        kind = "Yes"
        if re.search(r"\b(number|phone|mobile)\b", text, re.I): kind = "Number"
        elif re.search(r"\b(email|mail)\b", text, re.I):        kind = "Email"
        elif re.search(r"\b(website|web|url|site)\b", text, re.I): kind = "Website"
        elif re.search(r"\b(address)\b", text, re.I):           kind = "Address"
        ents["CONTACT_INFO"] = kind

    if _SEATS_KW.search(text):
        ents["SEATS"] = "Available"

    if _ADM_KW.search(text):
        if re.search(r"\b(deadline)\b", text, re.I):
            ents["APPLICATION_DEADLINE"] = "Yes"
        elif intent in ("Admission_process", "Get_admission_info"):
            ents["ADMISSION_PROCESS"] = "Yes"

    if _PASS_KW.search(text):
        ents["ADMISSION_PROCESS"] = ents.get("ADMISSION_PROCESS", "Pass-Percentage")

    if _RATING_KW.search(text):
        r_hint = "Yes"
        if re.search(r"\b(top|best|highest|5.star)\b", text, re.I): r_hint = "Top-Rated"
        elif re.search(r"\b(high|4.star)\b", text, re.I):           r_hint = "High"
        elif re.search(r"\b(good|3.star)\b", text, re.I):           r_hint = "Good"
        ents["RATING"] = r_hint

    if _TYPE_KW.search(text):
        m = _TYPE_KW.search(text)
        ents["COLLEGE_TYPE"] = m.group(0).title()

    if _DEPT_KW.search(text):
        dept_m = re.search(r"(civil|electrical|computer|mechanical|electronics?|software|arch(itecture)?|biomedical|chemical|geo.info)\s+(department|dept)?", text, re.I)
        ents["DEPARTMENT"] = dept_m.group(0).title() if dept_m else "Yes"

    if _INTERN_KW.search(text):
        ents["INTERNSHIP"] = "Yes"

    if _CUTOFF_KW.search(text):
        ents["CUTOFF_RANK"] = "Yes"

    if _FACILITY_KW.search(text):
        fac_m = re.search(r"\b(lab(oratory)?|library|sports?|canteen|hostel|wifi|internet|facility|facilities)\b", text, re.I)
        ents["FACILITY"] = fac_m.group(0).title() if fac_m else "Yes"

    if _RECOMMEND_KW.search(text) and intent == "Recommend_college":
        ents["RECOMMEND"] = "Yes"

    return ents


# ------------------------------------------------------------------
# Entity combination → intent inference
# ------------------------------------------------------------------
# Priority order: most-specific first
def infer_intent_from_entities(entities: dict) -> str:
    keys = set(entities.keys())

    # Hard-coded single-winner rules  (checked top-down)
    if "CONTACT_INFO"         in keys: return "Get_contact_info"
    if "HOSTEL_AVAILABILITY"  in keys: return "Get_hostel_availability_info"
    if "SCHOLARSHIP"          in keys: return "Get_scholorship_info"

    if "APPLICATION_DEADLINE" in keys:
        # If there's an ADMISSION_PROCESS entity too → Admission_process
        if "ADMISSION_PROCESS" in keys: return "Admission_process"
        return "Get_admission_info"
    if "ADMISSION_PROCESS"    in keys: return "Admission_process"
    if "CUTOFF_RANK"          in keys: return "Get_admission_info"
    if "INTERNSHIP"           in keys: return "GET_COLLEGE_INFO"

    if "FEE" in keys:
        # Has LOCATION but no COLLEGE_NAME → search by fee
        if "LOCATION" in keys and "COLLEGE_NAME" not in keys: return "Search_college_by_fee"
        return "GET_fee_info"

    if "SEATS"    in keys: return "Search_college_by_seats"
    if "RECOMMEND" in keys: return "Recommend_college"

    if "FACILITY" in keys:
        val = entities["FACILITY"].lower()
        if "hostel" in val:                      return "Get_hostel_availability_info"
        if "scholarship" in val:                 return "Get_scholorship_info"
        return "GET_COLLEGE_INFO"

    if "RATING" in keys:
        if "LOCATION" in keys or "PROGRAM" in keys: return "Recommend_college"
        return "GET_COLLEGE_INFO"

    if "COLLEGE_TYPE" in keys:
        if "LOCATION" in keys: return "Search_college_by_location"
        return "Search_college_by_type"

    if "AFFILIATION" in keys:
        if "LOCATION" in keys: return "Search_college_by_location"
        if "PROGRAM"  in keys: return "Search_college_by_program"
        return "GET_COLLEGE_INFO"

    if "DEPARTMENT" in keys: return "GET_program_info"

    if "PROGRAM" in keys:
        if "LOCATION" in keys: return "Search_college_by_program"
        if "COLLEGE_NAME" in keys: return "GET_program_info"
        return "Search_college_by_program"

    if "LOCATION" in keys:
        if "COLLEGE_NAME" in keys: return "Get_college_location"
        return "Search_college_by_location"

    if "COLLEGE_NAME" in keys: return "GET_COLLEGE_INFO"

    return "GET_COLLEGE_INFO"          # fallback


# ─────────────────────────────────────────────
# 4.  Build unified records
# ─────────────────────────────────────────────

records: list[dict] = []

# --- A) Process entity corpus (has entities, needs intent) ----------------
for rec in entity_sentences:
    raw_text  = rec["text"]
    clean_text = clean_bold(raw_text).strip()
    entities   = extract_bold_entities(raw_text, rec.get("entities", {}))
    intent     = infer_intent_from_entities(entities)

    records.append({
        "text":     clean_text,
        "intent":   intent,
        "entities": entities,
        "source":   "entity_corpus"
    })

# --- B) Process intent corpus (has intent, needs entities) ----------------
for rec in intent_sentences:
    text   = rec["text"].strip()
    intent = rec["intent"]
    if isinstance(intent, list):
        intent = intent[0] if intent else "Unknown"

    entities = extract_entities_from_intent_text(text, intent)

    records.append({
        "text":     text,
        "intent":   intent,
        "entities": entities,
        "source":   "intent_corpus"
    })


# ─────────────────────────────────────────────
# 5.  Deduplicate (keep richer annotation)
# ─────────────────────────────────────────────
seen: dict[str, dict] = {}
for rec in records:
    key = normalise(rec["text"])
    if key not in seen:
        seen[key] = rec
    else:
        # Keep the record with more entities
        existing = seen[key]
        if len(rec["entities"]) > len(existing["entities"]):
            # Merge: add missing entity keys from existing
            merged_ents = {**existing["entities"], **rec["entities"]}
            rec["entities"] = merged_ents
            seen[key] = rec
        else:
            # Augment existing with any new entity keys
            for k, v in rec["entities"].items():
                if k not in existing["entities"]:
                    existing["entities"][k] = v

unique_records = list(seen.values())

# Strip internal 'source' tracking field from final output
for r in unique_records:
    r.pop("source", None)


# ─────────────────────────────────────────────
# 6.  Build metadata
# ─────────────────────────────────────────────
intent_dist:  dict[str, int] = defaultdict(int)
entity_dist:  dict[str, int] = defaultdict(int)
both_count    = 0
intent_only   = 0
entity_only   = 0

for r in unique_records:
    has_intent  = bool(r.get("intent"))
    has_entities = bool(r.get("entities"))

    if has_intent:
        intent_dist[r["intent"]] += 1
    if has_entities:
        for etype in r["entities"]:
            entity_dist[etype] += 1

    if has_intent and has_entities: both_count  += 1
    elif has_intent:                intent_only += 1
    else:                           entity_only += 1

metadata = {
    "description": (
        "Combined intent + entity annotated dataset for BCT college "
        "recommendation chatbot. Every record has both an intent label "
        "and an entities dictionary."
    ),
    "version": "1.0.0",
    "created": "2026-03-01",
    "total_sentences":      len(unique_records),
    "records_with_both":    both_count,
    "records_intent_only":  intent_only,
    "records_entity_only":  entity_only,
    "unique_intents":        len(intent_dist),
    "unique_entity_types":  len(entity_dist),
    "intent_types":         sorted(intent_dist.keys()),
    "entity_types":         sorted(entity_dist.keys()),
    "intent_distribution":  dict(sorted(intent_dist.items(),  key=lambda x: -x[1])),
    "entity_distribution":  dict(sorted(entity_dist.items(), key=lambda x: -x[1])),
    "sources": [
        "intent/intent_data.json",
        "entity/balanced_data_700.json"
    ],
    "format": {
        "text":     "Plain natural-language query (** markers stripped).",
        "intent":   "Single intent label string.",
        "entities": "Dict[entity_type, entity_value]. Empty dict {} if no entities found."
    }
}

output = {
    "metadata":  metadata,
    "sentences": unique_records
}


# ─────────────────────────────────────────────
# 7.  Write output
# ─────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✓ Written {len(unique_records):,} records to {OUTPUT_FILE}")
print(f"  Both intent + entities : {both_count:,}")
print(f"  Intent-only            : {intent_only:,}")
print(f"  Entity-only            : {entity_only:,}")
print(f"  Unique intents         : {len(intent_dist)}")
print(f"  Unique entity types    : {len(entity_dist)}")
print("\nIntent distribution:")
for intent, count in metadata["intent_distribution"].items():
    print(f"  {intent:45s} {count:4d}")
print("\nEntity distribution:")
for etype, count in metadata["entity_distribution"].items():
    print(f"  {etype:30s} {count:4d}")
