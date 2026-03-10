# verify_bio.py
# Run this BEFORE training to check quality

import json
from collections import defaultdict


def verify_bio_dataset(bio_path):
    """
    Check BIO dataset for common problems.
    """
    with open(bio_path) as f:
        data = json.load(f)

    samples = data["samples"]

    issues = {
        "no_entities_but_expected": [],
        "invalid_bio_sequence":     [],
        "i_without_b":              [],
        "empty_tokens":             [],
    }

    entity_counts = defaultdict(int)
    total_labeled = 0

    for i, sample in enumerate(samples):
        tokens = sample["tokens"]
        labels = sample["labels"]

        # Check length mismatch
        if len(tokens) != len(labels):
            issues["empty_tokens"].append(i)
            continue

        prev_label = "O"
        for j, (token, label) in enumerate(
            zip(tokens, labels)
        ):
            # Check I- without preceding B-
            if label.startswith("I-"):
                entity_type = label[2:]
                expected_b  = f"B-{entity_type}"
                if not (
                    prev_label == expected_b or
                    prev_label == label
                ):
                    issues["i_without_b"].append({
                        "sample": i,
                        "token":  token,
                        "label":  label,
                        "prev":   prev_label
                    })

            if label != "O":
                entity_type = label[2:]
                entity_counts[entity_type] += 1
                total_labeled += 1

            prev_label = label

    # Print report
    print("BIO Dataset Verification Report")
    print("="*50)
    print(f"Total samples   : {len(samples)}")
    print(f"Total labeled   : {total_labeled} tokens")
    print()

    print("Entity type distribution:")
    for entity, count in sorted(
        entity_counts.items(),
        key=lambda x: -x[1]
    ):
        print(f"  {entity:20s}: {count}")

    print()
    print("Issues found:")
    for issue_type, items in issues.items():
        print(f"  {issue_type}: {len(items)}")

    # Show example BIO sequences
    print("\nSample BIO sequences (first 3):")
    for sample in samples[:3]:
        print(f"\n  Text: {sample['text']}")
        for token, label in zip(
            sample["tokens"], sample["labels"]
        ):
            if label != "O":
                print(f"    {token:20s} → {label}")

    return issues


if __name__ == "__main__":
    verify_bio_dataset("bio_dataset.json")