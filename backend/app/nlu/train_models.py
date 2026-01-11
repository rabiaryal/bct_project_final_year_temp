"""Unified Training Script for Intent and Entity Models"""

import os
import argparse
import sys
import subprocess
import json
from datetime import datetime

def train_intent_model(data_path, output_dir, args):
    """Train BERT intent classification model"""
    print("="*50)
    print("TRAINING BERT INTENT CLASSIFICATION MODEL")
    print("="*50)
    
    # Import and run intent training
    from nlu.intent.train_bert_intent import IntentTrainer
    
    trainer = IntentTrainer(max_length=args.max_length)
    texts, labels = trainer.load_data(data_path)
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
    
    model_trainer = trainer.train(
        train_dataset, val_dataset, output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )
    
    results, report = trainer.evaluate(test_dataset, model_trainer)
    return results, report

def train_entity_model(data_path, output_dir, args):
    """Train RoBERTa NER model"""
    print("="*50)
    print("TRAINING ROBERTA NER MODEL")
    print("="*50)
    
    # Import and run entity training
    from nlu.entity.train_roberta_ner import NERTrainer
    
    trainer = NERTrainer(max_length=args.max_length)
    
    if data_path.endswith('.csv'):
        texts, labels = trainer.load_data_from_csv(data_path)
    elif data_path.endswith('.json'):
        texts, labels = trainer.load_data_from_json(data_path)
    else:
        raise ValueError("Unsupported file format for entity data")
    
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
    
    model_trainer = trainer.train(
        train_dataset, val_dataset, output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )
    
    results, report = trainer.evaluate(test_dataset, model_trainer)
    return results, report

def main():
    parser = argparse.ArgumentParser(description='Train Intent and Entity Models')
    parser.add_argument('--mode', choices=['intent', 'entity', 'both'], required=True,
                       help='What to train: intent, entity, or both')
    parser.add_argument('--intent_data', help='Path to intent training data')
    parser.add_argument('--entity_data', help='Path to entity training data')
    parser.add_argument('--intent_output', help='Output directory for intent model')
    parser.add_argument('--entity_output', help='Output directory for entity model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()
    
    # Create output directories
    if args.mode in ['intent', 'both'] and args.intent_output:
        os.makedirs(args.intent_output, exist_ok=True)
    if args.mode in ['entity', 'both'] and args.entity_output:
        os.makedirs(args.entity_output, exist_ok=True)
    
    results = {}
    
    try:
        # Train intent model
        if args.mode in ['intent', 'both']:
            if not args.intent_data or not args.intent_output:
                print("Error: Intent training requires --intent_data and --intent_output")
                return
            
            intent_results, intent_report = train_intent_model(
                args.intent_data, args.intent_output, args
            )
            results['intent'] = {
                'results': intent_results,
                'report': intent_report,
                'output_dir': args.intent_output
            }
            print(f"Intent model saved to: {args.intent_output}")
        
        # Train entity model
        if args.mode in ['entity', 'both']:
            if not args.entity_data or not args.entity_output:
                print("Error: Entity training requires --entity_data and --entity_output")
                return
            
            entity_results, entity_report = train_entity_model(
                args.entity_data, args.entity_output, args
            )
            results['entity'] = {
                'results': entity_results,
                'report': entity_report,
                'output_dir': args.entity_output
            }
            print(f"Entity model saved to: {args.entity_output}")
        
        # Save combined results
        summary_file = f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'args': vars(args),
                'results': results
            }, f, indent=2)
        
        print(f"\nTraining summary saved to: {summary_file}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()