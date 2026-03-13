# Slot Filling, Context Management & MongoDB Query Generation

A focused explanation of the three core mechanisms based on the actual source code in `backend/app/`.

---

## 1. Slot Filling

### What is a slot?

A **slot** is a named piece of information the system needs to answer a query — e.g. `course`, `budget`, `rank`, `location`. They live in `context.slots` as a plain dict and are built from the entities the NLU model extracted.

Source: `backend/app/context/slot_manager.py`

---

### Step 1 — Entity → Slot mapping

The NLU entity extractor returns typed spans like `{"BUDGET": "5 lakh", "COURSE": "computer"}`. The slot manager translates these entity type names into canonical slot names via `ENTITY_TO_SLOT`:

```python
ENTITY_TO_SLOT = {
    "COURSE":          "course",
    "LOCATION":        "location",
    "COLLEGE_TYPE":    "college_type",
    "RANK":            "rank",
    "BUDGET":          "budget",
    "HOSTEL":          "hostel",
    "COLLEGE_NAME":    "college_name",
    "COLLEGE_NAME_1":  "college_name_1",
    "COLLEGE_NAME_2":  "college_name_2",
    # Legacy / alias names the model sometimes outputs:
    "DEPARTMENT":      "course",
    "FEE":             "budget",
    "CUTOFF_RANK":     "rank",
    "CITY":            "location",
    "TYPE":            "college_type",
    ...
}
```

Any entity type the model returns — even a legacy alias — maps to the same canonical slot. This decouples model label names from the application logic.

---

### Step 2 — Normalization

Raw entity values are messy strings. Before storing, each slot passes through a type-specific normalizer in `normalize_value()`:

#### `course` → `normalize_course()`
Maps informal names to the exact DB course names:
```python
_COURSE_ALIASES = {
    "computer":             "BE Computer",
    "computer engineering": "BE Computer",
    "software":             "BE Computer",
    "civil":                "BE Civil",
    "electrical":           "BE Electrical",
    "mechanical":           "BE Mechanical",
    ...
}
# "computer" → "BE Computer"
# "software engineering" → "BE Computer"
```

#### `budget` → `normalize_budget()`
Handles every real-world budget format:
```python
"5 lakh"    → 500000
"700000"    → 700000
"700k"      → 700000
"7"         → 700000   # bare number < 1000 assumed to be lakhs
"1.5 lakh"  → 150000
```
The < 1000 rule exists because Nepal engineering fees start at ~490,000 — a user who types "7" means 7 lakhs, never 7 rupees.

#### `rank` → `normalize_rank()`
Strips any surrounding text and extracts the integer:
```python
"rank 500"  → 500
"position 1200" → 1200
500         → 500
```

#### `college_name` → `normalize_college_name()`
Two-step process:
1. **Abbreviation lookup** — instant exact match:
   ```python
   _COLLEGE_ABBREVIATIONS = {
       "kec":  "Kantipur Engineering College",
       "ioe":  "Institute of Engineering",
       "ku":   "Kathmandu University",
       "wrc":  "Western Regional Campus",
       ...
   }
   ```
2. **Fuzzy match** — if not in abbreviations, RapidFuzz WRatio against the 15 known college names (threshold 70):
   ```python
   match = rfprocess.extractOne(
       v, _KNOWN_COLLEGES, scorer=fuzz.WRatio, score_cutoff=70
   )
   # "kantipur engineering" → "Kantipur Engineering College" (score=92)
   ```

#### `location` → `normalize_location()`
Fixes typos and expands province codes:
```python
_LOCATION_CORRECTIONS = {
    "ktm":        "kathmandu",
    "pokhra":     "pokhara",     # typo
    "kathamandu": "kathmandu",   # typo
    "patan":      "lalitpur",    # alias
}
```

#### `hostel` → `normalize_hostel()`
```python
"yes" / "available" / "needed" → True
"no"  / "not needed"           → False
# default when entity present but value ambiguous → True
```

#### `college_type` → `normalize_college_type()`
```python
"govt" / "government" / "public" → "public"
"pvt"  / "private"               → "private"
```

---

### Step 3 — Pending-slot override (NER failure recovery)

When the bot asked a follow-up question last turn (e.g. "Which course?"), the expected slot name is stored in `context.pending_slot`. If the user replies with a bare word like "computer" and NER returns nothing, the slot manager force-fills from raw text:

```python
# slot_manager.py — process_turn()
if context.pending_slot and raw_text:
    slot_name = context.pending_slot
    # Only override if NER didn't already supply this slot
    already_filled = any(
        ENTITY_TO_SLOT.get(et.upper()) == slot_name
        for et in entities
    )
    if not already_filled:
        normalized = normalize_value(slot_name, raw_text.strip())
        context.slots[slot_name] = normalized
    context.pending_slot = None   # consumed after this turn
```

**Example:**
- Bot asked: *"Which engineering course are you interested in?"*
- User replied: *"computer"*
- NER returns `{}` — no entity found
- `pending_slot = "course"` → raw text "computer" is normalized → `slots["course"] = "BE Computer"`

---

### Step 4 — Text-based fallback extraction

Even without `pending_slot`, the slot manager runs regex patterns on the raw message to recover rank and budget that NER missed:

```python
_RANK_PATTERN = re.compile(
    r"(?:rank|position)\s*(?:is|was|:)?\s*(\d{1,5})|(\d{1,5})\s*(?:rank|position)",
    re.IGNORECASE
)
_LAKH_PATTERN   = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh)", re.IGNORECASE)
_BUDGET_KEYWORD = re.compile(r"\b(budget|afford|spend|pay|fee|cost)\b", re.IGNORECASE)
```

These are applied only when the slot is not already filled by NER — they never overwrite a successfully extracted entity.

---

### Step 5 — Actionability check

Every intent declares which slots it **requires** vs which are optional, in `intent_templates.py`:

```python
"recommend_with_constraints": IntentTemplate(
    required_slots=["course"],
    optional_slots=["rank", "budget", "location"],
    ...
)

"personalized_recommendation": IntentTemplate(
    required_slots=["rank", "budget"],
    optional_slots=["course", "location"],
    ...
)

"compare_colleges": IntentTemplate(
    required_slots=["college_name_1", "college_name_2"],
    optional_slots=[],
    ...
)
```

After every turn, `validate_slots()` checks whether all required slots are present:

```python
def get_missing_slots(self, filled_slots):
    return [
        s for s in self.required_slots
        if s not in filled_slots or filled_slots[s] is None
    ]

def is_actionable(self, filled_slots):
    return len(self.get_missing_slots(filled_slots)) == 0
```

If **not actionable** → the dialogue manager returns a follow-up question and records `pending_slot`:
```python
follow_up = self.slot_manager.get_follow_up(context)
# context.pending_slot is set to the first missing slot name
```

Follow-up questions are declared per intent in the template:
```python
follow_up_questions = {
    "course":  "Which engineering course are you interested in? (e.g. Computer, Civil, Electrical)",
    "rank":    "What is your IOE entrance rank?",
    "budget":  "What is your fee budget? (e.g. 700000 or 7 lakhs)",
}
```

---

---

## 2. Context Management

### The DialogueContext object

Each session has exactly one `DialogueContext` object stored in memory:

```python
@dataclass
class DialogueContext:
    session_id:        str
    turn_count:        int = 0
    current_intent:    str = ""
    previous_intent:   str = ""
    intent_family:     str = ""
    slots:             Dict[str, Any] = field(default_factory=dict)
    missing_slots:     List[str] = field(default_factory=list)
    is_actionable:     bool = False
    pending_slot:      Optional[str] = None   # slot the bot is waiting for next turn
    last_results:      List[str] = field(default_factory=list)
    last_result_count: int = 0
    last_query:        Dict[str, Any] = field(default_factory=dict)
    last_updated:      datetime = field(default_factory=datetime.now)
```

Stored in `SlotManager.contexts` — a plain Python dict keyed by `session_id`. No database, no Redis — everything lives in process memory.

---

### Intent families — the carryover rule

Slots survive across turns when intents belong to the **same family**:

```python
# intent_templates.py
INTENT_FAMILIES = {
    "search_family":     ["search_college", "best_items_search",
                          "recommend_with_constraints", "personalized_recommendation"],
    "info_family":       ["college_details", "hostel_query", "contact_query",
                          "college_attribute_query", "admission_process"],
    "comparison_family": ["compare_colleges"],
    "conversational":    ["greeting", "goodbye", "unknown"],
}
```

`should_carry_slots(prev_intent, new_intent)` returns `True` only when both intents are in the same family.

**Case 1 — Family change** → all slots cleared:
```python
# slot_manager.py — process_turn()
if not should_carry_slots(context.previous_intent, intent):
    context.slots = {}
```
Example: user was doing a recommendation (`search_family`), then asks "does KEC have hostel?" (`info_family`) — budget and rank are wiped.

**Case 2 — Same family, different intent** → irrelevant slots pruned:
```python
template = get_template(intent)
relevant = set(template.required_slots) | set(template.optional_slots)
context.slots = {k: v for k, v in context.slots.items() if k in relevant}
```
Example: switching from `recommend_with_constraints` to `personalized_recommendation` — `course` is optional in both, so it survives. Any slot not in the new template's list is dropped.

**Case 3 — Greeting / goodbye** → slots never touched:
```python
if intent in ("greeting", "goodbye"):
    # only update intent tracking — leave slots exactly as they are
    return context
```
A user can say "hi" mid-conversation and their budget and rank are still remembered.

---

### Intent lock

If `pending_slot` is set (bot asked a follow-up), the next message's NLU classification is overridden:

```python
# dialogue_manager.py — process_message()
prev_context = self.slot_manager.get_context(session_id)
if prev_context and prev_context.pending_slot:
    locked_intent = prev_context.current_intent
    if intent != locked_intent:
        intent = locked_intent   # force back
```

Without this, "computer" as a reply to "which course?" gets classified as `greeting` (46% confidence) and context is broken.

Also a low-confidence fallback: if NLU confidence < 45% and there is no `pending_slot`, the previous intent is kept:
```python
elif prev_context and confidence < 0.45:
    intent = prev_context.current_intent
```

---

### Result persistence

After a successful DB fetch, the results are saved back into the session context:

```python
# slot_manager.py — update_with_results()
ctx.last_results      = [r["college_name"] for r in result_dicts]
ctx.last_result_count = len(result_dicts)
ctx.last_query        = query

# When exactly one result → auto-fill college_name slot
if len(names) == 1 and "college_name" not in ctx.slots:
    ctx.slots["college_name"] = names[0].lower()
```

This enables follow-ups like "what is the fee?" after a single-college search — `college_name` is already in slots without the user re-stating it.

---

### Resetting context

User sends `"clear"` → immediate wipe:
```python
# dialogue_manager.py
if request.message.strip().lower() == "clear":
    self.slot_manager.clear_context(session_id)
    # Returns: "🔄 Conversation cleared! ..."
```

`clear_context()` simply deletes the session entry from the dict:
```python
def clear_context(self, session_id: str):
    if session_id in self.contexts:
        del self.contexts[session_id]
```

The next message for that session_id creates a fresh `DialogueContext`.

---

---

## 3. MongoDB Query Generation

Source: `backend/app/core/query_builder.py`

Once all required slots are filled, `build_pipeline()` translates the slot dict into a MongoDB aggregation pipeline.

---

### MongoDB document structure

Understanding the pipeline requires knowing how data is stored. Each college document looks like:

```json
{
  "Name": "Kantipur Engineering College",
  "Location": "Kathmandu",
  "Type": "PRIVATE",
  "HostelAvailability": true,
  "ContactNumber": "01-4911422",
  "Email": "info@kec.edu.np",
  "Departments": [
    {
      "Name": "Computer Engineering",
      "Courses": [
        {
          "CourseId": 101,
          "Name": "BE Computer",
          "AverageCutoffRank": 2800,
          "Fee": 650000,
          "Rating": 4.2
        }
      ]
    }
  ]
}
```

`Departments` and `Departments.Courses` are **nested arrays**. MongoDB cannot filter or sort on fields inside nested arrays without first flattening them — which is what `$unwind` does.

---

### Intent → pipeline type

```
compare_colleges        →  _build_compare_pipeline()
uses_unwind = False     →  _build_top_level_pipeline()   hostel_query, contact_query, admission_process
uses_unwind = True      →  _build_unwind_pipeline()      search, best_items, recommendations
hybrid intents          →  build_candidate_pipeline()    personalized, recommend_with_constraints
```

---

### Step 1 — Building `$match` from slots

Each intent template maps slot names to their MongoDB field paths:

```python
# intent_templates.py — recommend_with_constraints
query_fields = {
    "course":   "Departments.Courses.Name",
    "rank":     "Departments.Courses.AverageCutoffRank",
    "budget":   "Departments.Courses.Fee",
    "location": "Location",
}

# hostel_query
query_fields = {
    "location":     "Location",
    "college_type": "Type",
    "hostel":       "HostelAvailability",
}
```

`_build_match()` iterates those mappings and applies the correct MongoDB operator per slot type:

```python
def _build_match(slots, template):
    match = {}
    for slot_name, db_field in template.query_fields.items():
        value = slots.get(slot_name)
        if value is None:
            continue
        if slot_name == "rank":
            match[db_field] = {"$gte": int(value)}              # cutoff rank >= user rank
        elif slot_name == "budget":
            match[db_field] = {"$lte": int(value)}              # fee <= user budget
        elif slot_name == "hostel":
            match[db_field] = bool(value)                        # exact boolean
        else:
            match[db_field] = {"$regex": str(value), "$options": "i"}  # case-insensitive text
    return match
```

**Example** — slots `{course: "BE Computer", budget: 500000}`:
```json
{
  "Departments.Courses.Name": { "$regex": "BE Computer", "$options": "i" },
  "Departments.Courses.Fee":  { "$lte": 500000 }
}
```

**Example** — slots `{rank: 800, budget: 700000, course: "BE Civil"}`:
```json
{
  "Departments.Courses.AverageCutoffRank": { "$gte": 800 },
  "Departments.Courses.Fee":              { "$lte": 700000 },
  "Departments.Courses.Name":             { "$regex": "BE Civil", "$options": "i" }
}
```

---

### Step 2a — Unwind pipeline (search/recommendation intents)

Used whenever `template.uses_unwind = True`. This is the standard pipeline for search and best-items intents:

```python
pipeline = [
    {"$unwind": "$Departments"},          # flatten departments array
    {"$unwind": "$Departments.Courses"},  # flatten courses array within each dept
    {"$match": match},                    # apply slot filters at course level
    {"$project": {                        # rename/select fields
        "_id": 0,
        "college_name": "$Name",
        "location":     "$Location",
        "college_type": "$Type",
        "hostel":       "$HostelAvailability",
        "contact":      "$ContactNumber",
        "email":        "$Email",
        "department":   "$Departments.Name",
        "course":       "$Departments.Courses.Name",
        "fee":          "$Departments.Courses.Fee",
        "rating":       "$Departments.Courses.Rating",
        "cutoff_rank":  "$Departments.Courses.AverageCutoffRank",
    }},
    {"$sort": {"rating": -1}},            # only added when template.sort_field is set
    {"$limit": 15},
]
```

Why `$unwind` before `$match`? Because `Departments.Courses.Fee` is buried inside a nested array — MongoDB can only evaluate `$lte` on it after the documents have been flattened into one row per course.

---

### Step 2b — Top-level pipeline (hostel, contact, admission intents)

`uses_unwind = False` — filters only at the college level, no course-level data needed:

```python
pipeline = [
    {"$match": match},       # e.g. {"HostelAvailability": true, "Location": {$regex: "kathmandu"}}
    {"$project": {
        "_id": 0,
        "college_name": "$Name",
        "location":     "$Location",
        "college_type": "$Type",
        "hostel":       "$HostelAvailability",
        "contact":      "$ContactNumber",
        "email":        "$Email",
    }},
    {"$limit": 15},
]
```

No unwinding needed because these intents don't filter on course-level fields.

---

### Step 2c — Compare pipeline

Fetches both colleges in one query using `$or`, then unwinds to get all their courses:

```python
match = {"$or": [
    {"Name": {"$regex": slots["college_name_1"], "$options": "i"}},
    {"Name": {"$regex": slots["college_name_2"], "$options": "i"}},
]}
pipeline = [
    {"$match": match},
    {"$unwind": "$Departments"},
    {"$unwind": "$Departments.Courses"},
    {"$project": _unwind_projection()},
    {"$limit": 15},
]
```

---

### Step 2d — Candidate pipeline (hybrid recommendation intents)

`personalized_recommendation` and `recommend_with_constraints` use `build_candidate_pipeline()`. The key difference: **lenient rank filter (50% buffer)**:

```python
def build_candidate_pipeline(slots, limit=15):
    match = {}

    if slots.get("course"):
        match["Departments.Courses.Name"] = {
            "$regex": slots["course"], "$options": "i"
        }
    if slots.get("budget"):
        match["Departments.Courses.Fee"] = {"$lte": int(slots["budget"])}
    if slots.get("location"):
        match["Location"] = {"$regex": slots["location"], "$options": "i"}

    if slots.get("rank"):
        lenient_rank = max(1, int(int(slots["rank"]) * 0.5))
        match["Departments.Courses.AverageCutoffRank"] = {"$gte": lenient_rank}
        #                                                          ↑
        #                              50% of user rank, not user rank itself

    return [
        {"$unwind": "$Departments"},
        {"$unwind": "$Departments.Courses"},
        {"$match": match},
        {"$project": _unwind_projection()},
        {"$sort": {"rating": -1}},
        {"$limit": limit},
    ]
```

**Why lenient?** If a user's rank is 500, a college with cutoff 300 is very reachable (SAFE). A strict `$gte: 500` filter would exclude those SAFE colleges entirely from results. The 50% buffer (`$gte: 250`) lets MongoDB coarsely pre-filter, and the Python scorer (outside MongoDB) then assigns exact SAFE / MODERATE / RISKY labels based on the precise cutoff difference.

---

### Full example: end-to-end trace

**Conversation:**

> Turn 1: "show me colleges under 5 lakh"
> Turn 2: "computer"

**After Turn 1 — slot filling:**
```
NLU entities:  {BUDGET: "5 lakh"}
normalize_budget("5 lakh") → 500000
context.slots = {budget: 500000}

Intent: recommend_with_constraints
required_slots: ["course"]
missing: ["course"]   →  is_actionable = False

Response:  "Which engineering course are you interested in?"
context.pending_slot = "course"
```

**After Turn 2 — slot filling:**
```
NLU intent: greeting (46%) — BLOCKED by intent lock (pending_slot="course")
intent forced back to: recommend_with_constraints

NLU entities: {}  — NER found nothing

pending_slot override:
  slot_name = "course"
  normalize_value("course", "computer") → "BE Computer"
  context.slots["course"] = "BE Computer"
  context.pending_slot = None

context.slots = {budget: 500000, course: "BE Computer"}
missing: []   →  is_actionable = True
```

**Query generation:**
```python
# build_candidate_pipeline(slots={budget: 500000, course: "BE Computer"})
match = {
    "Departments.Courses.Name": {"$regex": "BE Computer", "$options": "i"},
    "Departments.Courses.Fee":  {"$lte": 500000},
    # no rank → no rank filter
}

pipeline = [
    {"$unwind": "$Departments"},
    {"$unwind": "$Departments.Courses"},
    {"$match": {
        "Departments.Courses.Name": {"$regex": "BE Computer", "$options": "i"},
        "Departments.Courses.Fee":  {"$lte": 500000},
    }},
    {"$project": {all fields}},
    {"$sort":  {"rating": -1}},
    {"$limit": 15},
]
```

MongoDB returns matching colleges → Python scorer ranks them → formatted response sent back.

---

### Slot-to-operator reference

| Slot | MongoDB operator | Reason |
|---|---|---|
| `rank` | `$gte` | College cutoff must be ≥ user rank (user qualifies) |
| `budget` | `$lte` | Fee must be ≤ user budget |
| `hostel` | exact `bool` | Boolean field — no range needed |
| `course` | `$regex` case-insensitive | Partial name matching ("Computer" matches "BE Computer") |
| `location` | `$regex` case-insensitive | "kathmandu" matches "Kathmandu" |
| `college_type` | `$regex` case-insensitive | "private" matches "PRIVATE" |
| `college_name` | `$regex` case-insensitive | After fuzzy normalization, still use regex for robustness |
