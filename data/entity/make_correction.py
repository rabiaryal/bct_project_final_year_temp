import json
from collections import defaultdict

INPUT_FILE = "combined_entity_data.json"          # your original file
OUTPUT_FILE = "combined_entity_data_merged.json"  # new cleaned file

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# -------------------------------
# 1️⃣ Merge entity labels in sentences
# -------------------------------
for sentence in data["sentences"]:
    entities = sentence.get("entities", {})

    # Merge DEPARTMENT_NAME → DEPARTMENT
    if "DEPARTMENT_NAME" in entities:
        entities["DEPARTMENT"] = entities["DEPARTMENT_NAME"]
        del entities["DEPARTMENT_NAME"]

    # Merge FACULTY → DEPARTMENT
    if "FACULTY" in entities:
        entities["DEPARTMENT"] = entities["FACULTY"]
        del entities["FACULTY"]

    sentence["entities"] = entities

# -------------------------------
# 2️⃣ Update entity_types metadata
# -------------------------------
old_entity_types = data["metadata"]["entity_types"]

new_entity_types = [
    e for e in old_entity_types
    if e not in ["DEPARTMENT_NAME", "FACULTY"]
]

# Ensure DEPARTMENT exists
if "DEPARTMENT" not in new_entity_types:
    new_entity_types.append("DEPARTMENT")

data["metadata"]["entity_types"] = sorted(set(new_entity_types))

# -------------------------------
# 3️⃣ Recompute entity_distribution
# -------------------------------
entity_distribution = defaultdict(int)

for sentence in data["sentences"]:
    for label in sentence.get("entities", {}).keys():
        entity_distribution[label] += 1

data["metadata"]["entity_distribution"] = dict(sorted(entity_distribution.items()))

# -------------------------------
# 4️⃣ Update unique_entity_types count
# -------------------------------
data["metadata"]["unique_entity_types"] = len(data["metadata"]["entity_types"])

# -------------------------------
# 5️⃣ Save cleaned dataset
# -------------------------------
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Dataset cleaned successfully!")
print(f"📄 Saved to: {OUTPUT_FILE}")
print(f"🔢 Total entity types: {data['metadata']['unique_entity_types']}")
