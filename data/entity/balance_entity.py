import json
from collections import Counter

def balance_dataset(input_file, output_file, max_limit=300):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_sentences = data.get("sentences", [])
    balanced_sentences = []
    
    # Track the current count for each entity type
    current_counts = Counter()

    for item in original_sentences:
        entities = item.get("entities", {})
        
        # Check if any entity in this sentence has already reached the limit
        can_add = True
        for entity_type in entities.keys():
            if current_counts[entity_type] >= max_limit:
                can_add = False
                break
        
        # If all entities in this sentence are under the limit, keep it
        if can_add:
            balanced_sentences.append(item)
            # Update the counts for all entities found in this sentence
            for entity_type in entities.keys():
                current_counts[entity_type] += 1

    # Update Metadata
    new_entity_types = sorted(list(current_counts.keys()))
    new_metadata = {
        "total_sentences": len(balanced_sentences),
        "unique_entity_types": len(new_entity_types),
        "entity_types": new_entity_types,
        "entity_distribution": dict(current_counts),
        "sources": data.get("metadata", {}).get("sources", []) + ["downsampled_result.json"]
    }

    # Construct final object
    result = {
        "metadata": new_metadata,
        "sentences": balanced_sentences
    }

    # Write to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Downsampling complete!")
    print(f"Original sentences: {len(original_sentences)}")
    print(f"New sentences: {len(balanced_sentences)}")
    print(f"Results saved to: {output_file}")

# Execute
balance_dataset('combined_entity_data_merged.json', 'balanced_data_300.json')