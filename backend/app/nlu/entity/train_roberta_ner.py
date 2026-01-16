"""Training Script for RoBERTa Named Entity Recognition"""

import json
import torch
from transformers import (
    RobertaTokenizerFast, RobertaForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import argparse

from data_loader import NERDataset, load_data_from_csv, load_data_from_json
from utils import (
    post_process_entities, 
    post_process_entities_with_confidence,
    compute_metrics, 
    evaluate, 
    create_visualizations
)

class NERTrainer:
    """RoBERTa NER Trainer"""
    
    def __init__(self, model_name='roberta-base', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
    def create_label_mappings(self, all_labels):
        """Create label to ID mappings"""
        unique_labels = set()
        for label_seq in all_labels:
            unique_labels.update(label_seq)
        
        unique_labels = sorted(list(unique_labels))
        
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Ensure 'O' is in the mapping
        if 'O' not in self.label_to_id:
            new_id = len(self.label_to_id)
            self.label_to_id['O'] = new_id
            self.id_to_label[new_id] = 'O'
            print("Added 'O' to label mapping.")

        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        return self.label_to_id, self.id_to_label
    
    def prepare_datasets(self, texts, labels, test_size=0.1, val_size=0.1):
        """Split data and create datasets"""
        # Train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Train/validation split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # Create label mappings
        all_labels = y_train + y_val + y_test
        self.create_label_mappings(all_labels)
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.model_name, add_prefix_space=True)
        
        # Create datasets
        train_dataset = NERDataset(X_train, y_train, self.tokenizer, self.label_to_id, self.max_length)
        val_dataset = NERDataset(X_val, y_val, self.tokenizer, self.label_to_id, self.max_length)
        test_dataset = NERDataset(X_test, y_test, self.tokenizer, self.label_to_id, self.max_length)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset, output_dir, num_epochs=3, batch_size=16, learning_rate=5e-5):
        """Train the RoBERTa model"""
        # Initialize model
        num_labels = len(self.label_to_id)
        self.model = RobertaForTokenClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: compute_metrics(p, self.id_to_label),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        mappings_path = os.path.join(output_dir, 'label_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump({
                'label2id': self.label_to_id,
                'id2label': self.id_to_label
            }, f, indent=2)
        
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def predict(self, text, threshold=0.3):
        """Predict entities for a given text string with entity-level confidence thresholding."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before prediction.")

        self.model.eval()

        # Tokenize the input text with offset mapping for proper alignment
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True
        )

        input_ids = inputs['input_ids'].to(self.model.device)
        attention_mask = inputs['attention_mask'].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get probabilities and predictions
        probs = torch.softmax(logits, dim=-1)[0]
        preds = torch.argmax(logits, dim=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        labels = [self.id_to_label[p.item()] for p in preds]

        # Remove special tokens and get token-level confidence scores
        filtered = [
            (t, l, probs[i][preds[i]].item())
            for i, (t, l) in enumerate(zip(tokens, labels))
            if t not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]
        ]

        if not filtered:
            return []
            
        tokens, labels, scores = zip(*filtered)

        # 🔹 Build spans FIRST (without token-level thresholding)
        spans = post_process_entities_with_confidence(tokens, labels, scores)

        # 🔹 Apply threshold at ENTITY LEVEL
        final_entities = [
            e for e in spans
            if e["confidence"] >= threshold
        ]

        return final_entities

def main():
    # Get project root directory (go up from current script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '../../../..')  # Go up 4 levels: entity->nlu->app->backend->project_root
    project_root = os.path.abspath(project_root)
    
    print(f"🔍 Project root detected: {project_root}")
    
    parser = argparse.ArgumentParser(description='Train RoBERTa NER Model')
    parser.add_argument('--data_path', 
                       default=os.path.join(project_root, 'data/entity/combined_entity_data_merged.json'),
                       help='Path to training data (CSV or JSON)')
    parser.add_argument('--output_dir', 
                       default=os.path.join(project_root, 'models/roberta_entity_model'),
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--graphics_dir', 
                       default=os.path.join(project_root, 'docs/graphs/entity'),
                       help='Directory to save graphics and visualizations')
    
    args = parser.parse_args()
    
    print("🚀 Starting RoBERTa NER Training")
    print("="*50)
    print(f"📁 Data: {args.data_path}")
    print(f"💾 Output: {args.output_dir}")
    print(f"📊 Graphics: {args.graphics_dir}")
    print(f"🔢 Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print("="*50)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.graphics_dir, exist_ok=True)
    
    # Initialize trainer
    ner_trainer = NERTrainer(max_length=args.max_length)
    
    # Load data based on file extension
    if args.data_path.endswith('.csv'):
        texts, labels = load_data_from_csv(args.data_path)
    elif args.data_path.endswith('.json'):
        texts, labels = load_data_from_json(args.data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = ner_trainer.prepare_datasets(texts, labels)
    
    # Train model
    model_trainer = ner_trainer.train(
        train_dataset, val_dataset, args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )
    
    # Evaluate with comprehensive metrics
    results, report, evaluation_metrics, true_labels, predicted_labels = evaluate(test_dataset, model_trainer, ner_trainer.id_to_label)
    
    # Get trainer logs for loss curve
    trainer_logs = model_trainer.state.log_history if hasattr(model_trainer.state, 'log_history') else None
    
    # Create comprehensive visualizations
    create_visualizations(args.graphics_dir, texts, labels, evaluation_metrics, trainer_logs)
    
    # Save detailed results
    detailed_results = {
        'training_results': results,
        'evaluation_metrics': {
            'entity_precision': evaluation_metrics['entity_precision'],
            'entity_recall': evaluation_metrics['entity_recall'],
            'entity_f1': evaluation_metrics['entity_f1'],
            'bio_validity_rate': evaluation_metrics['bio_validity_rate'],
            'per_entity_f1': {k: v.get('f1-score', 0) if isinstance(v, dict) else 0 
                             for k, v in evaluation_metrics.get('per_entity_metrics', {}).items()
                             if k not in ['accuracy', 'macro avg', 'weighted avg', 'micro avg']}
        },
        'timestamp': datetime.now().isoformat(),
        'model_config': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, 'comprehensive_results.json')
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Print summary
    print(f"✅ Training complete! Results saved to {args.output_dir}")
    print(f"📊 Comprehensive graphics saved to {args.graphics_dir}/")
    print("\n📈 Summary Metrics:")
    print(f"  • Entity-level Precision: {evaluation_metrics['entity_precision']:.3f}")
    print(f"  • Entity-level Recall: {evaluation_metrics['entity_recall']:.3f}")
    print(f"  • Entity-level F1: {evaluation_metrics['entity_f1']:.3f}")
    print(f"  • BIO Validity Rate: {evaluation_metrics['bio_validity_rate']:.1%}")
    print(f"\n📋 Comprehensive visualizations created:")
    print(f"  ✓ Entity-level Precision/Recall/F1 chart")
    print(f"  ✓ Per-entity F1 table")
    print(f"  ✓ BIO validity rate pie chart")
    print(f"  ✓ Confusion matrix")
    print(f"  ✓ Training vs Validation Loss")
    print(f"  ✓ Validation Accuracy progression")
    print(f"  ✓ Validation F1 Score progression")
    print(f"  ✓ Learning Rate schedule")
    print(f"  ✓ Combined Training Dashboard")
    print(f"  ✓ Label distribution & sentence statistics")

if __name__ == "__main__":
    main()