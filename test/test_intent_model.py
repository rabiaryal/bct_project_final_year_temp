#!/usr/bin/env python3
"""
Interactive Intent Model Tester

This script allows you to test the BERT intent classification model individually
from the terminal with manual input.

Usage:
    python test_intent_model.py
"""

import sys
import os
import torch
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.nlu.intent.bert_intent import BERTIntentClassifier
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger("intent_tester")

def test_intent_model():
    """Interactive intent model testing"""
    try:
        print("🚀 Loading BERT Intent Classification Model...")
        print("=" * 60)
        
        # Initialize model
        intent_classifier = BERTIntentClassifier()
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Device: {intent_classifier.device}")
        print(f"🎯 Available intents: {len(intent_classifier.label_mapping)}")
        print(f"📝 Intent classes: {list(intent_classifier.label_mapping.values())}")
        print("=" * 60)
        print("💬 Enter text to classify intent (type 'quit' to exit):")
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
                
                # Run prediction
                intent, confidence, metadata = intent_classifier.predict(user_input)
                
                # Display results
                print(f"🎯 Predicted Intent: {intent}")
                print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
                print(f"🔧 Model: {metadata.get('model', 'unknown')}")
                print(f"💻 Device: {metadata.get('device', 'unknown')}")
                print(f"📈 Threshold: {metadata.get('confidence_threshold', 0.5)}")
                
                # Show top predictions
                top_predictions = metadata.get('top_predictions', [])
                if top_predictions:
                    print("\n📋 Top 5 Predictions:")
                    for i, (pred_intent, pred_conf) in enumerate(top_predictions[:5], 1):
                        status = "✅" if i == 1 else "  "
                        print(f"  {status} {i}. {pred_intent}: {pred_conf:.4f} ({pred_conf*100:.2f}%)")
                
                # Confidence assessment
                if confidence >= 0.8:
                    conf_status = "🟢 High Confidence"
                elif confidence >= 0.6:
                    conf_status = "🟡 Medium Confidence"
                else:
                    conf_status = "🔴 Low Confidence"
                
                print(f"\n🎨 Assessment: {conf_status}")
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

if __name__ == "__main__":
    print("🧠 BERT Intent Classification Model Tester")
    print("=" * 60)
    success = test_intent_model()
    if not success:
        sys.exit(1)