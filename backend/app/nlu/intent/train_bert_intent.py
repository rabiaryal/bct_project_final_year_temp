"""Training Script for BERT Intent Classification"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import os
from datetime import datetime
import argparse

class IntentDataset(Dataset):
    """Custom Dataset for Intent Classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class IntentTrainer:
    """BERT Intent Classification Trainer"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_mapping = {}
        self.id_to_label = {}
        
    def load_data(self, data_path):
        """Load intent training data"""
        print(f"Loading data from {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Extract texts and intents
            texts = []
            intents = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        texts.append(item.get('text', ''))
                        intents.append(item.get('intent', ''))
            elif isinstance(data, dict):
                for intent, examples in data.items():
                    for example in examples:
                        texts.append(example)
                        intents.append(intent)
        
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            texts = df['text'].tolist()
            intents = df['intent'].tolist()
        
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        # Create label mappings
        unique_intents = list(set(intents))
        self.label_mapping = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.id_to_label = {idx: intent for intent, idx in self.label_mapping.items()}
        
        # Convert intents to labels
        labels = [self.label_mapping[intent] for intent in intents]
        
        print(f"Loaded {len(texts)} examples with {len(unique_intents)} intents")
        print(f"Intents: {unique_intents}")
        
        return texts, labels
    
    def prepare_datasets(self, texts, labels, test_size=0.2, val_size=0.1):
        """Split data and create datasets"""
        # Train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Train/validation split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42
        )
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = IntentDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = IntentDataset(X_val, y_val, self.tokenizer, self.max_length)
        test_dataset = IntentDataset(X_test, y_test, self.tokenizer, self.max_length)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_dataset, val_dataset, output_dir, num_epochs=3, batch_size=16, learning_rate=5e-5):
        """Train the BERT model"""
        # Initialize model
        num_labels = len(self.label_mapping)
        self.model = BertForSequenceClassification.from_pretrained(
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
            evaluation_strategy="epoch",
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
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        label_mapping_path = os.path.join(output_dir, 'label_mapping.json')
        with open(label_mapping_path, 'w') as f:
            json.dump({
                'label_to_id': self.label_mapping,
                'id_to_label': self.id_to_label
            }, f, indent=2)
        
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset, trainer):
        """Evaluate the trained model"""
        print("Evaluating model...")
        results = trainer.evaluate(test_dataset)
        
        # Detailed classification report
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Generate classification report
        target_names = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        print("\nClassification Report:")
        print(report)
        
        return results, report

def main():
    parser = argparse.ArgumentParser(description='Train BERT Intent Classifier')
    parser.add_argument('--data_path', required=True, help='Path to training data')
    parser.add_argument('--output_dir', required=True, help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = IntentTrainer(max_length=args.max_length)
    
    # Load data
    texts, labels = trainer.load_data(args.data_path)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
    
    # Train model
    model_trainer = trainer.train(
        train_dataset, val_dataset, args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )
    
    # Evaluate
    results, report = trainer.evaluate(test_dataset, model_trainer)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'args': vars(args)
        }, f, indent=2)
    
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()