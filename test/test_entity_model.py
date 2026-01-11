#!/usr/bin/env python3
"""
Interactive Entity Model Tester

This script allows you to test the RoBERTa entity extraction model individually
from the terminal with manual input.

Usage:
    python test_entity_model.py
"""

import sys
import os
import torch
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.nlu.entity.roberta_ner import RoBERTaEntityExtractor
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger("entity_tester")

async def test_entity_model():
    """Interactive entity model testing"""
    try:
        print("🚀 Loading RoBERTa Entity Extraction Model...")
        print("=" * 60)
        
        # Initialize model
        entity_extractor = RoBERTaEntityExtractor()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Device: {entity_extractor.device}")
        print(f"🎯 Available labels: {len(entity_extractor.label_mapping)}")
        
        # Show available entity types
        entity_types = set()
        for label in entity_extractor.label_mapping.values():
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label[2:]  # Remove B- or I- prefix
                entity_types.add(entity_type)
        
        print(f"📝 Entity types: {sorted(list(entity_types))}")
        print("\n🔍 Label mapping sample:")
        for i, (id_key, label) in enumerate(list(entity_extractor.label_mapping.items())[:10]):
            print(f"  {id_key}: {label}")
        if len(entity_extractor.label_mapping) > 10:
            print(f"  ... and {len(entity_extractor.label_mapping) - 10} more")
        
        print("=" * 60)
        print("💬 Enter text to extract entities (type 'quit' to exit):")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n➤ Enter text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not user_input:
                    print("⚠️  Please enter some text.")
                    continue
                
                print(f"\n🔍 Processing: '{user_input}'")
                print("-" * 40)
                
                # Run entity extraction
                entities = await entity_extractor.extract(user_input)
                
                # Display results
                if entities:
                    print(f"🎯 Extracted Entities ({len(entities)}):")
                    for entity_type, value in entities.items():
                        print(f"  📌 {entity_type}: '{value}'")
                    
                    # Show entity mapping to slots
                    print("\n🔗 Slot Mapping:")
                    for entity_type, value in entities.items():
                        if entity_type == "college_name_mentioned":
                            print(f"  🏛️  college_name → '{value}'")
                        elif entity_type == "program_mentioned":
                            print(f"  📚 course_name → '{value}'")
                        elif entity_type == "location_mentioned":
                            print(f"  📍 location → '{value}'")
                        elif entity_type == "facility_mentioned":
                            print(f"  🏢 facility → '{value}'")
                        elif entity_type == "fee_mentioned":
                            print(f"  💰 fee_type → '{value}'")
                        else:
                            slot_name = entity_type.replace("_mentioned", "_name") if "_mentioned" in entity_type else entity_type
                            print(f"  ⚙️  {slot_name} → '{value}'")
                else:
                    print("❌ No entities found")
                    print("💡 Try sentences with college names, locations, programs, etc.")
                    print("📝 Examples:")
                    print("   - 'Tell me about Kathmandu University'")
                    print("   - 'What programs does Tribhuvan University offer?'")
                    print("   - 'Where is Pokhara University located?'")
                
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing input: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"💡 Make sure the model path is correct and the model files exist.")
        return False
    
    return True

async def main():
    print("🤖 RoBERTa Entity Extraction Model Tester")
    print("=" * 60)
    success = await test_entity_model()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())