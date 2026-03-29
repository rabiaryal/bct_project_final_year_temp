# Recommendation System — Complete Step-by-Step Data Retrieval Flow

This document explains exactly how the **`recommend_with_constraints`** intent works, from user input → database query → ranked recommendations.

---

## Overview: The Hybrid Pipeline

The recommendation system uses a **two-stage hybrid approach**:
1. **Stage 1 (MongoDB)**: Hard filter to get 15 candidate colleges
2. **Stage 2 (Python)**: Score, weight, and rerank to top 3 colleges

This avoids scoring every college in the database and gives fine-grained control over "fit".

---

## Full End-to-End Flow

### **STEP 1: User Sends Message**

**Input:**
```
"Suggest colleges for BE Computer with rank 1000 and budget 10 lakhs"
session_id: "session_1773..."
```

**Location:** `backend/app/main.py` → POST `/api/v1/chat`

---

### **STEP 2: NLU (Natural Language Understanding)**

**Where:** `backend/app/dialogue_manager.py` → `process_message()` → line 192

**What happens:**

#### 2a. **Intent Classification (BERT)**
```
Model: BERT fine-tuned on 12 engineering college intents
Input: "Suggest colleges for BE Computer with rank 1000 and budget 10 lakhs"
Output:
  intent = "recommend_with_constraints"
  confidence = 0.7824   (78.24%)
  top_predictions = [
    ("recommend_with_constraints", 0.7824),
    ("personalized_recommendation", 0.1205),
    ("search_college", 0.0971)
  ]
```

**File:** `backend/app/nlu/intent/bert_intent.py`  
**Model path:** `models/bert_intent_model/`

#### 2b. **Entity Extraction (RoBERTa+CRF)**
```
Model: RoBERTa + CRF (Conditional Random Field) for sequence labeling
Input: "Suggest colleges for BE Computer with rank 1000 and budget 10 lakhs"
Predicted BIO tags: [B-COURSE, I-COURSE, O, O, B-RANK, O, B-BUDGET, ...]
Output entities:
  {
    "COURSE":  "BE Computer",
    "RANK":    "1000",
    "BUDGET":  "10 lakhs"
  }
```

**File:** `backend/app/nlu/entity/roberta_ner.py`  
**Model path:** `models/roberta_entity_model/`

**Terminal output:**
```
--- NLU ---
  Intent: recommend_with_constraints (78.24%)
    recommend_with_constraints: 78.24%
    personalized_recommendation: 12.05%
    search_college: 9.71%
  Entities: {'COURSE': 'BE Computer', 'RANK': '1000', 'BUDGET': '10 lakhs'}
```

---

### **STEP 3: Slot Normalization**

**Where:** `backend/app/context/slot_manager.py` → `process_turn()`

**What happens:**

The extracted entities are converted to normalized **slots** (canonical form for database queries).

#### 3a. **Entity → Slot Mapping**
```
COURSE  entity "BE Computer"      → "course"  slot
RANK    entity "1000"             → "rank"    slot
BUDGET  entity "10 lakhs"         → "budget"  slot
```

#### 3b. **Normalization (value transformation)**

**Course normalization:**
```python
normalize_course("BE Computer") 
  → removes "BE" prefix
  → looks up in alias dict
  → returns "BE Computer Engineering" (canonical form)
```

**Rank normalization:**
```python
normalize_rank("1000")
  → regex extract digits
  → returns 1000 (integer)
```

**Budget normalization:**
```python
normalize_budget("10 lakhs")
  → parses "lakhs" multiplier
  → converts "10 lakhs" → 1,000,000 rupees (integer)
```

**Output slots:**
```python
{
  "course": "BE Computer Engineering",
  "rank": 1000,
  "budget": 1000000
}
```

**File:** `backend/app/context/slot_manager.py` lines 50–150

**Terminal output:**
```
--- SLOTS ---
  course: BE Computer Engineering (new)
  rank: 1000 (new)
  budget: 1000000 (new)
  Missing: (none)
  Actionable: Yes
```

---

### **STEP 4: Check Required Slots**

**Where:** `backend/app/core/slot_filler.py` → `get_missing_slot()`

**Logic:**
```
Required slots for "recommend_with_constraints": ["course"]
Optional slots: ["rank", "budget", "location"]

Check:
  ✓ course is present → OK
  ✓ rank is present → OK (optional but beneficial)
  ✓ budget is present → OK (optional but beneficial)

Result: Actionable = YES (all required slots filled)
        No follow-up question needed
```

**If slots were MISSING:**
```
Example: User says "Suggest colleges"
  Missing: "course"
  Bot returns: "Which engineering course are you interested in?"
  Action: "ask"  (sets pending_slot = "course")
```

---

### **STEP 5: Build MongoDB Aggregation Pipeline**

**Where:** `backend/app/core/query_builder.py` → `build_candidate_pipeline(slots, limit=15)`

**Purpose:** Convert normalized slots into MongoDB query conditions

#### 5a. **Build Match Conditions**

```python
match = {}

# Course filter (case-insensitive regex)
if slots.get("course"):  # "BE Computer Engineering"
    match["Departments.Courses.Name"] = {
        "$regex": "BE Computer Engineering",
        "$options": "i"  # case-insensitive
    }

# Rank filter (lenient: 50% buffer for candidates)
if slots.get("rank"):  # 1000
    lenient_rank = max(1, int(1000 * 0.5))  # = 500
    # Accept courses where cutoff >= 500
    # Why lenient? Python scorer will do fine-grained ranking later
    match["Departments.Courses.Rank"] = {
        "$gte": 500  # >= lenient_rank
    }

# Budget filter (strict: must be affordable)
if slots.get("budget"):  # 1000000
    match["Departments.Courses.Fee"] = {
        "$lte": 1000000  # <= user budget (strict)
    }

# College type filter (if specified)
if slots.get("college_type"):  # "PUBLIC" or "PRIVATE"
    match["Type"] = "PRIVATE"
```

#### 5b. **Assemble Full MongoDB Aggregation Pipeline**

```json
[
  {
    "$unwind": "$Departments"
  },
  {
    "$unwind": "$Departments.Courses"
  },
  {
    "$match": {
      "Departments.Courses.Name": {
        "$regex": "BE Computer Engineering",
        "$options": "i"
      },
      "Departments.Courses.Rank": {
        "$gte": 500
      },
      "Departments.Courses.Fee": {
        "$lte": 1000000
      }
    }
  },
  {
    "$project": {
      "_id": 0,
      "college_name": "$Name",
      "location": "$Location",
      "college_type": "$Type",
      "department": "$Departments.Name",
      "course": "$Departments.Courses.Name",
      "fee": "$Departments.Courses.Fee",
      "rating": "$Departments.Courses.Rating",
      "cutoff_rank": "$Departments.Courses.Rank",
      "hostel": "$HostelAvailability",
      "contact": "$ContactNumber",
      "email": "$Email"
    }
  },
  {
    "$sort": {
      "rating": -1
    }
  },
  {
    "$limit": 15
  }
]
```

**File:** `backend/app/core/query_builder.py` lines 155–180

---

### **STEP 6: Execute MongoDB Query**

**Where:** `backend/app/handlers/recommend_handler.py` → line 112

```python
# Execute aggregation pipeline
cursor = collection.aggregate(pipeline)
candidates = await cursor.to_list(length=15)  # Fetch up to 15 docs

# Example output:
candidates = [
  {
    "college_name": "Pulchowk Engineering Campus",
    "location": "PULCHOWK, LALITPUR",
    "college_type": "PUBLIC",
    "course": "BE Computer Engineering",
    "fee": 490000,
    "rating": 4.9,
    "cutoff_rank": 250,
    "hostel": true
  },
  {
    "college_name": "Kantipur Engineering College",
    "location": "DHAPAKHEL, LALITPUR",
    "college_type": "PRIVATE",
    "course": "BE Computer Engineering",
    "fee": 900000,
    "rating": 4.2,
    "cutoff_rank": 2800,
    "hostel": false
  },
  ...  (up to 15 results, highest-rated first)
]
```

**Database:** `crs` collection `college data`  
**Total documents in MongoDB:** ~15 colleges × ~3 departments × ~2–5 courses = ~150–225 course-level records

---

### **STEP 7: Deduplicate**

**Where:** `backend/app/core/query_builder.py` → `deduplicate(candidates)`

**Purpose:** Remove duplicate colleges (each college can appear multiple times if it offers multiple courses)

```python
# Before dedup:
[
  {"college_name": "Pulchowk", "course": "BE Computer"},
  {"college_name": "Pulchowk", "course": "BE Civil"},        # Duplicate college
  {"college_name": "Kantipur", "course": "BE Computer"},
]

# After dedup (keep first occurrence):
[
  {"college_name": "Pulchowk", "course": "BE Computer"},
  {"college_name": "Kantipur", "course": "BE Computer"},
]
```

---

### **STEP 8: Python-Side Scoring & Reranking**

**Where:** `backend/app/core/scorer.py` → `RecommendScorer.rerank(candidates, slots, top_n=3)`

**Purpose:** Go beyond hard filters → score colleges on **fit** using weighted criteria

#### 8a. **Score Each Candidate**

For each candidate, call `RecommendScorer.score(college, slots)`:

```python
# Example: Score "Pulchowk Engineering Campus"
college = {
  "college_name": "Pulchowk Engineering Campus",
  "fee": 490000,
  "rating": 4.9,
  "cutoff_rank": 250,
  "hostel": true,
  "college_type": "PUBLIC"
}

slots = {
  "course": "BE Computer Engineering",
  "rank": 1000,
  "budget": 1000000
}

# Scoring logic:
user_rank = 1000
cutoff = 250
gap = cutoff - user_rank = 250 - 1000 = -750

# Rank safety scoring (0–35 points)
if gap <= 0:
    rank_score = 5  # WARNING: exact match, very competitive
    reason = "⚠️  Rank 1000 exactly meets cutoff (very competitive)"
elif gap <= 200:
    rank_score = 12
elif gap <= 500:
    rank_score = 20
elif gap <= 1000:
    rank_score = 28
else:
    rank_score = 35  # Safe
    reason = "Rank 1000 very safely qualifies (gap: 750 ranks)"

score = 0 + rank_score  # = 5

# Fee saving scoring (0–35 points)
user_budget = 1000000
fee = 490000
saving = 1000000 - 490000 = 510000
saving_pct = 510000 / 1000000 = 51%

if saving_pct >= 0.80:
    fee_score = 35
elif saving_pct >= 0.50:
    fee_score = 26  # 51% savings
    reason = "Affordable: Rs.490,000 fee (saves Rs.510,000 = 51% of budget)"
else:
    fee_score = 17

score += fee_score  # = 5 + 26 = 31

# Rating scoring (0–20 points)
rating = 4.9
rating_score = (4.9 / 5.0) * 20 = 19.6
reason = "Exceptional rating: 4.9/5.0"
score += rating_score  # = 31 + 19.6 = 50.6

# Hostel bonus (0 or 5 points)
if college.hostel:
    score += 5  # = 50.6 + 5 = 55.6
    reason = "Hostel available on campus"

# Public college bonus (0 or 5 points)
if college_type == "PUBLIC":
    score += 5  # = 55.6 + 5 = 60.6
    reason = "Government college — reputed and affordable"

# FINAL SCORE: 60.6 / 100
# Reasons: [
#   "Rank 1000 very safely qualifies (gap: 750 ranks)",
#   "Affordable: Rs.490,000 fee (saves Rs.510,000 = 51% of budget)",
#   "Exceptional rating: 4.9/5.0",
#   "Hostel available on campus",
#   "Government college — reputed and affordable"
# ]
```

**Weights:**
```
rank_safety:  35%  (How safe your rank is for admission)
fee_saving:   35%  (How much budget you save)
rating:       20%  (College reputation)
hostel_bonus:  5%  (Bonus if hostel available)
public_bonus:  5%  (Bonus if government college)
TOTAL:       100%
```

**File:** `backend/app/core/scorer.py` → `RecommendScorer` class

#### 8b. **Rerank & Keep Top 3**

```python
# After scoring all 15 candidates:
scored = [
  (college_1, 60.6, reasons_1),   # Pulchowk
  (college_2, 48.3, reasons_2),   # Kantipur
  (college_3, 42.1, reasons_3),   # Sagarmatha
  (college_4, 38.9, reasons_4),
  ...
  (college_15, 12.4, reasons_15),
]

# Sort by (score DESC, rating DESC)
sorted_candidates = sorted(scored, key=lambda x: (x[0]["Score"], x[0].get("rating", 0)), reverse=True)

# Keep top 3
top_colleges = sorted_candidates[:3]
# = [Pulchowk (60.6), Kantipur (48.3), Sagarmatha (42.1)]
```

---

### **STEP 9: Format Response**

**Where:** `backend/app/handlers/recommend_handler.py` → `_format_response(top_colleges, slots)`

**Output to user:**

```
Here are the top 3 colleges matching your constraints 
(Course: BE Computer Engineering | Rank: 1000 | Budget: Rs.10,00,000):

───────────────────────────────────────────────────────────────
  #1  Pulchowk Engineering Campus
       📍 Location : PULCHOWK, LALITPUR
       🏛️  Type     : PUBLIC  |  🏠 Hostel: Yes
       📚 Course   : BE Computer Engineering
       💰 Fee      : Rs. 490,000
       🎯 Cutoff   : Rank 250
       ⭐ Rating   : 4.9 / 5.0
       📊 Match    : 60.6 / 100
       ✅ Why this fits:
            • Rank 1000 very safely qualifies (gap: 750 ranks)
            • Affordable: Rs.490,000 fee (saves Rs.510,000 = 51% of budget)
            • Exceptional rating: 4.9/5.0
            • Hostel available on campus
            • Government college — reputed and affordable

───────────────────────────────────────────────────────────────
  #2  Kantipur Engineering College
       📍 Location : DHAPAKHEL, LALITPUR
       🏛️  Type     : PRIVATE  |  🏠 Hostel: No
       📚 Course   : BE Computer Engineering
       💰 Fee      : Rs. 900,000
       🎯 Cutoff   : Rank 2800
       ⭐ Rating   : 4.2 / 5.0
       📊 Match    : 48.3 / 100
       ✅ Why this fits:
            • Rank 1000 safely within cutoff (gap: 1800 ranks)
            • Affordable: Rs.900,000 fee (saves Rs.100,000 = 10% of budget)
            • Good rating: 4.2/5.0

───────────────────────────────────────────────────────────────
  #3  Sagarmatha Engineering College
...
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    USER MESSAGE                             │
│  "Suggest colleges for BE Computer with rank 1000 budget 10L"
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
      ┌──────────────────────────────┐
      │  BERT Intent Classifier      │
      │  → recommend_with_constraints│
      │     (confidence: 78.24%)     │
      └───────────────┬──────────────┘
                      │
                      ▼
      ┌──────────────────────────────┐
      │  RoBERTa+CRF Entity Extractor│
      │  → COURSE: "BE Computer"     │
      │  → RANK: "1000"              │
      │  → BUDGET: "10 lakhs"        │
      └───────────────┬──────────────┘
                      │
                      ▼
      ┌──────────────────────────────┐
      │  Slot Normalization          │
      │  • course → "BE Computer..."  │
      │  • rank → 1000 (int)         │
      │  • budget → 1000000 (int)    │
      └───────────────┬──────────────┘
                      │
                      ▼
      ┌──────────────────────────────┐
      │  Check Required Slots        │
      │  • course ✓                  │
      │  Actionable? YES             │
      └───────────────┬──────────────┘
                      │
                      ▼
     ┌──────────────────────────────────┐
     │ Build MongoDB Pipeline           │
     │ • $unwind Departments            │
     │ • $unwind Courses                │
     │ • $match fees ≤ 1000000         │
     │ • $match rank ≥ 500 (lenient)   │
     │ • $sort by rating DESC           │
     │ • $limit 15                      │
     └────────────┬─────────────────────┘
                  │
                  ▼
     ┌──────────────────────────────────┐
     │  MongoDB Aggregation             │
     │  ↓ (Query on college data)       │
     │  Returns: ~15 candidates         │
     │           (sorted by rating)     │
     └────────────┬─────────────────────┘
                  │
                  ▼
     ┌──────────────────────────────────┐
     │  Deduplicate                     │
     │  (remove duplicate colleges)     │
     │  Returns: ~10–12 unique colleges │
     └────────────┬─────────────────────┘
                  │
                  ▼
     ┌──────────────────────────────────┐
     │  Python Scoring (RecommendScorer)│
     │  • Rank safety (35%)             │
     │  • Fee saving (35%)              │
     │  • Rating (20%)                  │
     │  • Bonuses (10%)                 │
     │  Score 0–100 for each            │
     └────────────┬─────────────────────┘
                  │
                  ▼
     ┌──────────────────────────────────┐
     │  Rerank & Keep Top 3             │
     │  (sort by score, then rating)    │
     │  Returns: Top 3 colleges         │
     └────────────┬─────────────────────┘
                  │
                  ▼
     ┌──────────────────────────────────┐
     │  Format Response                 │
     │  • College cards with scores     │
     │  • Reasons why each fits         │
     │  • Formatted for user display    │
     └────────────┬─────────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │ 📤 BOT RESPONSE│
          │ Top 3 colleges│
          └───────────────┘
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| **Lenient rank filter (50% buffer)** | Hard cutoff on rank field might miss good candidates; Python scorer does fine-tuning |
| **Strict budget filter** | Budget is absolute constraint; can't afford means can't attend |
| **Deduplicate before scoring** | Avoid scoring same college multiple times (for different departments) |
| **Top 3 only** | User experience: too many choices is overwhelming |
| **Score-based rerank** | Holistic fit is better than single-field sort (e.g., rating alone) |
| **Explain reasons** | Interpretable AI: user understands WHY a college is recommended |

---

## Example: Complete Request-Response Cycle

**Input:**
```json
{
  "message": "Suggest colleges for BE Computer with rank 1000 and budget 10 lakhs",
  "session_id": "session_1773..."
}
```

**Processing:**
```
step 1: NLU → intent="recommend_with_constraints", entities={COURSE, RANK, BUDGET}
step 2: Normalize → slots={course: "BE Computer Engineering", rank: 1000, budget: 1000000}
step 3: Check slots → all filled, actionable=YES
step 4: Build MongoDB pipeline → matches course + lenient rank + budget
step 5: Query MongoDB → 15 candidates ordered by rating
step 6: Deduplicate → ~10 unique colleges
step 7: Score each → weighted fit (0–100)
step 8: Rerank top 3 → [Pulchowk, Kantipur, Sagarmatha] + reasoning
step 9: Format → human-readable response with cards
```

**Output:**
```json
{
  "message": "Here are the top 3 colleges matching your constraints...",
  "response_type": "recommendation",
  "results": [
    {
      "college_name": "Pulchowk Engineering Campus",
      "score": 60.6,
      "reasons": ["Rank 1000 safely qualifies...", "Affordable...", ...]
    },
    ...
  ]
}
```

---

## Files Involved

| File | Role |
|------|------|
| `backend/app/dialogue_manager.py` | Orchestrates entire pipeline |
| `backend/app/nlu/intent/bert_intent.py` | Intent classification |
| `backend/app/nlu/entity/roberta_ner.py` | Entity extraction |
| `backend/app/context/slot_manager.py` | Slot normalization + context |
| `backend/app/core/query_builder.py` | MongoDB pipeline construction |
| `backend/app/core/scorer.py` | Python-side ranking logic |
| `backend/app/handlers/recommend_handler.py` | Hybrid handler entry point |
| `backend/app/repositories/mongo_client.py` | MongoDB connection |

---

## Performance Notes

**Typical latency breakdown:**
- NLU (BERT + RoBERTa+CRF): ~400–600ms
- MongoDB query: ~50–100ms
- Scoring + dedup: ~20–50ms
- Response formatting: ~10–20ms
- **Total: ~0.5–0.8 seconds**

**Scalability:**
- MongoDB index on `Departments.Courses.Fee`, `Departments.Courses.Rank`, `Departments.Courses.Name` recommended
- Max candidates per query: 15 (configurable in `build_candidate_pipeline`)
- Max colleges in response: 3 (top_k = 3)
