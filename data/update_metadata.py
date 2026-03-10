"""Recompute and update metadata in intent_entity.json from its own sentences."""
import json
from pathlib import Path
from datetime import date

DATA_FILE = Path(__file__).parent / "new_data_collection.json"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data.get("sentences", [])

# --- Counters ---
intent_counts: dict[str, int] = {}
entity_counts: dict[str, int] = {}
records_with_both = 0
records_intent_only = 0
records_entity_only = 0

for record in sentences:
    intent = record.get("intent", "")
    entities: dict = record.get("entities", {})

    has_intent = bool(intent)
    has_entities = bool(entities)

    if has_intent:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    if has_entities:
        for etype in entities:
            entity_counts[etype] = entity_counts.get(etype, 0) + 1

    if has_intent and has_entities:
        records_with_both += 1
    elif has_intent:
        records_intent_only += 1
    elif has_entities:
        records_entity_only += 1

# --- Build updated metadata (preserve unchanged fields) ---
existing_meta = data.get("metadata", {})

updated_meta = {
    "description": existing_meta.get(
        "description",
        "Combined intent + entity annotated dataset for BCT college recommendation chatbot.",
    ),
    "version": existing_meta.get("version", "1.0.0"),
    "created": existing_meta.get("created", str(date.today())),
    "last_updated": str(date.today()),
    "total_sentences": len(sentences),
    "records_with_both": records_with_both,
    "records_intent_only": records_intent_only,
    "records_entity_only": records_entity_only,
    "unique_intents": len(intent_counts),
    "unique_entity_types": len(entity_counts),
    "intent_types": sorted(intent_counts.keys()),
    "entity_types": sorted(entity_counts.keys()),
    "intent_distribution": dict(sorted(intent_counts.items(), key=lambda x: -x[1])),
    "entity_distribution": dict(sorted(entity_counts.items(), key=lambda x: -x[1])),
    "sources": existing_meta.get("sources", []),
    "format": existing_meta.get("format", {}),
}

data["metadata"] = updated_meta

with open(DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# --- Summary ---
print(f"✅  Metadata updated in {DATA_FILE.name}")
print(f"    Total sentences      : {len(sentences)}")
print(f"    Records with both    : {records_with_both}")
print(f"    Records intent-only  : {records_intent_only}")
print(f"    Records entity-only  : {records_entity_only}")
print(f"    Unique intents       : {len(intent_counts)}")
print(f"    Unique entity types  : {len(entity_counts)}")
print()
print("  Intent distribution:")
for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
    print(f"    {intent:<40} {count}")
print()
print("  Entity distribution:")
for etype, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
    print(f"    {etype:<40} {count}")
