#!/usr/bin/env python3
"""
Interactive NLU Pipeline Tester

This script allows you to test both intent classification and entity extraction
together, simulating the complete NLU pipeline.

Usage:
    python test_nlu_pipeline.py
"""

import sys
import os
import torch
import asyncio
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.nlu.intent.bert_intent import BERTIntentClassifier
from app.nlu.entity.roberta_ner import RoBERTaEntityExtractor
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger("nlu_tester")

async def test_nlu_pipeline():
    """Interactive NLU pipeline testing"""
    try:
        print("🚀 Loading Complete NLU Pipeline...")
        print("=" * 60)
        
        # Initialize models
        print("📥 Loading BERT Intent Classifier...")
        intent_classifier = BERTIntentClassifier()
        
        print("📥 Loading RoBERTa Entity Extractor...")
        entity_extractor = RoBERTaEntityExtractor()
        
        print(f"\n✅ NLU Pipeline loaded successfully!")
        print(f"🎯 Intent Model: BERT ({len(intent_classifier.label_mapping)} classes)")
        print(f"🏷️  Entity Model: RoBERTa ({len(entity_extractor.label_mapping)} labels)")
        print(f"💻 Device: {intent_classifier.device}")
        
        print("\n📚 Available Intent Classes:")
        for i, intent in enumerate(intent_classifier.label_mapping.values(), 1):
            print(f"  {i:2d}. {intent}")
        
        print("\n🏷️  Available Entity Types:")
        entity_types = set()
        for label in entity_extractor.label_mapping.values():
            if label.startswith('B-'):
                entity_types.add(label[2:])
        for i, entity_type in enumerate(sorted(entity_types), 1):
            print(f"  {i:2d}. {entity_type}")
        
        print("=" * 60)
        print("💬 Enter text for complete NLU analysis (type 'quit' to exit):")
        print("💡 Examples:")
        print("   - 'Tell me about Kathmandu University engineering programs'")
        print("   - 'What are the admission requirements for Tribhuvan University?'")
        print("   - 'Where is Pokhara University located and what facilities do they have?'")
        print("=" * 60)
        
        session_count = 0
        
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
                
                session_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                print(f"\n🔬 NLU Analysis #{session_count} ({timestamp})")
                print(f"📝 Input: '{user_input}'")
                print("=" * 50)
                
                # Run intent classification
                print("🧠 INTENT CLASSIFICATION:")
                intent, confidence, metadata = intent_classifier.predict(user_input)
                
                print(f"  🎯 Predicted Intent: {intent}")
                print(f"  📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                
                # Show top 3 intent predictions
                top_predictions = metadata.get('top_predictions', [])
                if len(top_predictions) > 1:
                    print(f"  📋 Top 3 Predictions:")
                    for i, (pred_intent, pred_conf) in enumerate(top_predictions[:3], 1):
                        status = "👑" if i == 1 else f"{i}."
                        print(f"     {status} {pred_intent}: {pred_conf:.4f}")
                
                # Run entity extraction
                print("\n🏷️  ENTITY EXTRACTION:")
                entities = await entity_extractor.extract(user_input)
                
                if entities:
                    print(f"  📌 Found {len(entities)} entities:")
                    for entity_type, value in entities.items():
                        print(f"     • {entity_type}: '{value}'")
                    
                    # Show slot mapping
                    print(f"  🔗 Slot Mapping:")
                    slots = {}
                    for entity_type, value in entities.items():
                        if entity_type == "college_name_mentioned":
                            slots["college_name"] = value
                        elif entity_type == "program_mentioned":
                            slots["course_name"] = value
                        elif entity_type == "location_mentioned":
                            slots["location"] = value
                        elif entity_type == "facility_mentioned":
                            slots["facility"] = value
                        elif entity_type == "fee_mentioned":
                            slots["fee_type"] = value
                        else:
                            slot_name = entity_type.replace("_mentioned", "_name") if "_mentioned" in entity_type else entity_type
                            slots[slot_name] = value
                    
                    for slot_name, slot_value in slots.items():
                        print(f"     🔑 {slot_name} = '{slot_value}'")
                else:
                    print("  ❌ No entities detected")
                
                # Combined analysis
                print("\n🎨 COMBINED ANALYSIS:")
                if confidence >= 0.8:
                    conf_status = "🟢 High Confidence"
                elif confidence >= 0.6:
                    conf_status = "🟡 Medium Confidence"
                else:
                    conf_status = "🔴 Low Confidence"
                
                entity_richness = "🟢 Rich" if len(entities) >= 2 else "🟡 Moderate" if len(entities) == 1 else "🔴 Poor"
                
                print(f"  📈 Intent Confidence: {conf_status}")
                print(f"  🏷️  Entity Richness: {entity_richness}")
                
                # Suggest next action based on intent
                action_suggestions = {
                    "GET_COLLEGE_INFO": "action_search_college",
                    "GET_COURSE_INFO": "action_search_course",
                    "GET_ADMISSION_INFO": "action_get_admission_info",
                    "GET_FEE_INFO": "action_get_fee_info",
                    "Get_college_location": "action_get_location_info",
                    "Get_contact_info": "action_provide_contact",
                    "Greeting": "action_greet",
                    "Goodbye": "action_goodbye"
                }
                
                suggested_action = action_suggestions.get(intent, "action_fallback")
                print(f"  🎬 Suggested Action: {suggested_action}")
                
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing input: {e}")
                logger.error(f"NLU pipeline error: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Failed to load NLU pipeline: {e}")
        logger.error(f"Failed to initialize NLU: {e}")
        return False
    
    return True

async def main():
    print("🧠🏷️  Complete NLU Pipeline Tester")
    print("=" * 60)
    success = await test_nlu_pipeline()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())