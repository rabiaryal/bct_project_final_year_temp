import json
from collections import Counter

def update_json_metadata(file_path):
    # 1. Load the existing data
    with open(file_path, 'r') as f:
        data = json.load(f)

    sentences = data.get("sentences", [])
    
    # 2. Calculate New Stats
    total_sentences = len(sentences)
    
    # Extract all entity keys from all sentences
    all_entity_keys = []
    for s in sentences:
        all_entity_keys.extend(s.get("entities", {}).keys())
    
    # Count distribution and unique types
    entity_distribution = dict(Counter(all_entity_keys))
    unique_entity_types = sorted(list(entity_distribution.keys()))
    
    # 3. Construct the Updated Metadata Object
    new_metadata = {
        "total_sentences": total_sentences,
        "unique_entity_types": len(unique_entity_types),
        "entity_types": unique_entity_types,
        "entity_distribution": entity_distribution,
        "sources": data.get("metadata", {}).get("sources", []) # Preserve sources
    }
    
    # 4. Update the object and save back to file
    data["metadata"] = new_metadata
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Metadata updated successfully! Processed {total_sentences} sentences.")

# Run the function
update_json_metadata('combined_entity_data_merged.json')