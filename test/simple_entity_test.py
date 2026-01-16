#!/usr/bin/env python3

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import sys
import os

class EntityExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        label_path = os.path.join(model_path, 'label_mappings.json')
        with open(label_path, 'r') as f:
            mappings = json.load(f)
            self.id2label = mappings['id2label']

    def extract_entities_with_confidence(self, text, threshold=0.3):
        """
        Extract entities with confidence scoring and threshold filtering
        
        Rules:
        1. Treat each entity independently - NO competition between entities
        2. Extract EVERY entity whose confidence >= threshold
        3. Apply threshold BEFORE ranking or limiting
        4. Sort by confidence ONLY for display purposes
        5. Return up to top 3 entities after sorting
        6. Preserve multi-word entities as single spans
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128,
            is_split_into_words=False
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get probabilities and predictions
        probabilities = torch.softmax(logits, dim=2)[0]
        predictions = torch.argmax(logits, dim=2)[0]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Extract ALL entities with BIO logic - NO filtering yet
        all_entities = []
        current_entity = None
        current_text = ""
        current_confidences = []
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            confidence = probabilities[i][pred_id.item()].item()
            predicted_label = self.id2label[str(pred_id.item())]
            
            # Clean token (RoBERTa specific)
            clean_token = token.replace('Ġ', ' ').strip()
            if not clean_token:
                continue
            
            if predicted_label.startswith('B-'):
                # Save previous entity (if exists)
                if current_entity and current_text.strip():
                    avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                    all_entities.append({
                        'text': current_text.strip(),
                        'type': current_entity,
                        'confidence': avg_confidence
                    })
                
                # Start new entity
                current_entity = predicted_label[2:]  # Remove B- prefix
                current_text = clean_token
                current_confidences = [confidence]
                
            elif predicted_label.startswith('I-') and current_entity and predicted_label[2:] == current_entity:
                # Continue current entity (multi-word span)
                current_text += clean_token
                current_confidences.append(confidence)
                
            else:
                # End current entity
                if current_entity and current_text.strip():
                    avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                    all_entities.append({
                        'text': current_text.strip(),
                        'type': current_entity,
                        'confidence': avg_confidence
                    })
                
                # Reset for next entity
                current_entity = None
                current_text = ""
                current_confidences = []
        
        # Add final entity if exists
        if current_entity and current_text.strip():
            avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
            all_entities.append({
                'text': current_text.strip(),
                'type': current_entity,
                'confidence': avg_confidence
            })
        
        # STEP 1: Apply threshold filter FIRST (independent per entity)
        threshold_filtered_entities = [
            entity for entity in all_entities 
            if entity['confidence'] >= threshold
        ]
        
        # STEP 2: Sort by confidence (for display only)
        threshold_filtered_entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        # STEP 3: Take top 3 for display (after threshold filtering)
        top_entities = threshold_filtered_entities[:3]
        
        return {
            'text': text,
            'entities': top_entities,
            'total_found': len(threshold_filtered_entities),  # Count of entities meeting threshold
            'all_detected': len(all_entities)  # Total entities detected (before threshold)
        }

    def post_process_entities_with_confidence(self, tokens, labels, scores):
        """
        Post-process entities with confidence scores (entity-level thresholding safe).
        
        Args:
            tokens: List of tokens from tokenizer
            labels: List of predicted BIO labels  
            scores: List of confidence scores per token
        
        Returns:
            List of entities with confidence scores calculated at entity level
        """
        if not tokens:
            return []

        # Step 1: Merge subword tokens based on RoBERTa's 'Ġ' prefix
        merged_tokens = []
        merged_labels = []
        merged_scores = []
        
        for token, label, score in zip(tokens, labels, scores):
            if token.startswith('Ġ'):
                # Start of a new word
                merged_tokens.append(token[1:])
                merged_labels.append(label)
                merged_scores.append(score)
            elif merged_tokens:
                # Continuation of the previous word - merge with previous
                merged_tokens[-1] += token
                # Keep the previous label (don't change it)
                # Average the scores for merged tokens
                merged_scores[-1] = (merged_scores[-1] + score) / 2
            else:
                # First token doesn't have a 'Ġ' prefix (unusual but handle it)
                merged_tokens.append(token)
                merged_labels.append(label)
                merged_scores.append(score)

        # Step 2: Extract entities from merged tokens using BIO logic
        entities = []
        current_entity_tokens = []
        current_entity_label = None
        current_entity_scores = []

        for token, label, score in zip(merged_tokens, merged_labels, merged_scores):
            if label.startswith('B-'):
                # If there's a current entity, save it before starting a new one
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    # Calculate entity confidence as average of token scores
                    entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
                    entities.append({
                        'text': entity_text, 
                        'type': current_entity_label,
                        'confidence': entity_confidence
                    })

                # Start a new entity
                current_entity_tokens = [token]
                current_entity_label = label[2:]  # Remove B- prefix
                current_entity_scores = [score]
                
            elif label.startswith('I-') and current_entity_label == label[2:]:
                # Continue the current entity
                current_entity_tokens.append(token)
                current_entity_scores.append(score)
                
            else:
                # Not an entity token or different entity - close current one
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
                    entities.append({
                        'text': entity_text,
                        'type': current_entity_label,
                        'confidence': entity_confidence
                    })
                current_entity_tokens = []
                current_entity_label = None
                current_entity_scores = []

        # Add the last entity if it exists
        if current_entity_tokens:
            entity_text = " ".join(current_entity_tokens)
            entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
            entities.append({
                'text': entity_text,
                'type': current_entity_label, 
                'confidence': entity_confidence
            })

        return entities

    def print_results(self, result):
        """
        Print extraction results with confidence scores
        Following the exact format specified in the prompt
        """
        text = result['text']
        entities = result['entities']
        total_found = result['total_found']
        all_detected = result['all_detected']
        
        print("")
        if entities:
            displayed = len(entities)
            print(f"✨ Found {total_found} entities (showing top {displayed}):")
            print("=" * 50)
            for i, entity in enumerate(entities, 1):
                print(f"{i}. '{entity['text']}' | {entity['type'].upper()} | Confidence: {entity['confidence']:.3f}")
            print("=" * 50)
        else:
            print("No entities found.")
            print("=" * 50)
        
        # Debug info: show total detected vs threshold filtered
        if all_detected > total_found:
            print(f"🔍 Debug: {all_detected} total entities detected, {total_found} above threshold (0.3)")
        
        print(f"\nInput Text:")
        print(f'"{text}"')

    def debug_all_entities(self, text):
        """Show all entities detected regardless of confidence threshold"""
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128,
            is_split_into_words=False
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probabilities = torch.softmax(logits, dim=2)[0]
        predictions = torch.argmax(logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        print("\n🔍 DEBUG: All detected entities (no threshold):")
        print("=" * 60)
        
        # Extract ALL entities regardless of confidence
        all_entities = []
        current_entity = None
        current_text = ""
        current_confidences = []
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            confidence = probabilities[i][pred_id.item()].item()
            predicted_label = self.id2label[str(pred_id.item())]
            
            clean_token = token.replace('Ġ', ' ').strip()
            if not clean_token:
                continue
            
            if predicted_label.startswith('B-'):
                if current_entity and current_text.strip():
                    avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                    all_entities.append({
                        'text': current_text.strip(),
                        'type': current_entity,
                        'confidence': avg_confidence
                    })
                
                current_entity = predicted_label[2:]
                current_text = clean_token
                current_confidences = [confidence]
                
            elif predicted_label.startswith('I-') and current_entity and predicted_label[2:] == current_entity:
                current_text += clean_token
                current_confidences.append(confidence)
                
            else:
                if current_entity and current_text.strip():
                    avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                    all_entities.append({
                        'text': current_text.strip(),
                        'type': current_entity,
                        'confidence': avg_confidence
                    })
                current_entity = None
                current_text = ""
                current_confidences = []
        
        if current_entity and current_text.strip():
            avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
            all_entities.append({
                'text': current_text.strip(),
                'type': current_entity,
                'confidence': avg_confidence
            })
        
        # Sort by confidence for display
        all_entities.sort(key=lambda x: x['confidence'], reverse=True)
        
        if all_entities:
            for i, entity in enumerate(all_entities, 1):
                status = "✅ ABOVE" if entity['confidence'] >= 0.3 else "❌ BELOW"
                print(f"{i}. '{entity['text']}' | {entity['type'].upper()} | Confidence: {entity['confidence']:.3f} {status} threshold")
        else:
            print("No entities detected at all")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Extract entities with confidence scores')
    parser.add_argument('--model-path', default=None, 
                       help='Path to the trained model')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Confidence threshold for entity extraction')
    
    args = parser.parse_args()
    
    # Auto-detect model path based on current working directory
    if args.model_path is None:
        if os.path.exists('models/roberta_entity_model'):
            # Running from project root
            args.model_path = 'models/roberta_entity_model'
        elif os.path.exists('../models/roberta_entity_model'):
            # Running from test directory  
            args.model_path = '../models/roberta_entity_model'
        else:
            print("❌ Could not find roberta_entity_model.")
            print("Please ensure you're running from project root or test directory.")
            sys.exit(1)
    
    print(f"📁 Using model path: {args.model_path}")
    
    try:
        # Initialize extractor
        extractor = EntityExtractor(args.model_path)
        
        print("🤖 Entity Extractor - Enter your text:")
        
        while True:
            try:
                # Get input from terminal
                text = input("\nEnter text (or 'quit' to exit): ").strip()
                
                if text.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not text:
                    print("Please enter some text.")
                    continue
                
                # Extract entities with confidence
                result = extractor.extract_entities_with_confidence(text, args.threshold)
                
                # Print results
                extractor.print_results(result)
                
                # Automatically show debug if no entities found
                if not result['entities']:
                    print("\n🔍 No entities found. Showing all detected entities (including below threshold):")
                    extractor.debug_all_entities(text)
                else:
                    # Optional debug for successful extractions
                    print(f"\n💡 Type 'debug' to see all detected entities, or press Enter to continue...")
                    debug_input = input("").strip().lower()
                    if debug_input == 'debug':
                        extractor.debug_all_entities(text)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()