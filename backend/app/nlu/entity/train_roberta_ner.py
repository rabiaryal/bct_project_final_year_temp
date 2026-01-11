"""Training Script for RoBERTa Named Entity Recognition"""

import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForTokenClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np
import os
from datetime import datetime
import argparse
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

class NERDataset(Dataset):
    """Custom Dataset for Named Entity Recognition"""
    
    def __init__(self, texts, labels, tokenizer, label_to_id, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text.split(),  # Pre-tokenized
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label_to_id.get(labels[word_idx] if word_idx < len(labels) else 'O', 0))
            else:
                aligned_labels.append(-100)  # Ignore sub-tokens
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class NERTrainer:
    """RoBERTa NER Trainer"""
    
    def __init__(self, model_name='roberta-base', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
    def load_data_from_csv(self, csv_path):
        """Load NER data from CSV format"""
        print(f"Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Group by sentence_id if available, otherwise treat each row as separate
        if 'sentence_id' in df.columns:
            grouped = df.groupby('sentence_id')
            texts = []
            labels = []
            
            for _, group in grouped:
                tokens = group['token'].tolist()
                tags = group['label'].tolist()
                texts.append(' '.join(tokens))
                labels.append(tags)
        else:
            # Assume format: token, label
            texts = [df['token'].iloc[i] for i in range(len(df))]
            labels = [df['label'].iloc[i] for i in range(len(df))]
        
        return texts, labels
    
    def load_data_from_json(self, json_path):
        """Load NER data from JSON format"""
        print(f"Loading data from {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            if isinstance(item, dict):
                text = item.get('text', '')
                entities = item.get('entities', [])
                
                # Convert to BIO format
                tokens = text.split()
                tags = ['O'] * len(tokens)
                
                for entity in entities:
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    label = entity.get('label', '')
                    
                    # Find token positions
                    entity_text = text[start:end]
                    entity_tokens = entity_text.split()
                    
                    # Simple token alignment (can be improved)
                    for i, token in enumerate(tokens):
                        if token in entity_tokens:
                            if i == 0 or tags[i-1] == 'O':
                                tags[i] = f'B-{label}'
                            else:
                                tags[i] = f'I-{label}'
                
                texts.append(text)
                labels.append(tags)
        
        return texts, labels
    
    def create_label_mappings(self, all_labels):
        """Create label to ID mappings"""
        unique_labels = set()
        for label_seq in all_labels:
            unique_labels.update(label_seq)
        
        unique_labels = sorted(list(unique_labels))
        
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        return self.label_to_id, self.id_to_label
    
    def prepare_datasets(self, texts, labels, test_size=0.2, val_size=0.1):
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
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = NERDataset(X_train, y_train, self.tokenizer, self.label_to_id, self.max_length)
        val_dataset = NERDataset(X_val, y_val, self.tokenizer, self.label_to_id, self.max_length)
        test_dataset = NERDataset(X_test, y_test, self.tokenizer, self.label_to_id, self.max_length)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute NER metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            'precision': precision_score(true_labels, true_predictions),
            'recall': recall_score(true_labels, true_predictions),
            'f1': f1_score(true_labels, true_predictions),
            'accuracy': accuracy_score(true_labels, true_predictions),
        }
    
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
        
        # Save label mappings
        mappings_path = os.path.join(output_dir, 'label_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump({
                'label2id': self.label_to_id,
                'id2label': self.id_to_label
            }, f, indent=2)
        
        print(f"Model saved to {output_dir}")
        
        return trainer
    
    def evaluate(self, test_dataset, trainer):
        """Evaluate the trained model"""
        print("Evaluating model...")
        results = trainer.evaluate(test_dataset)
        
        # Detailed predictions for classification report
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=2)
        y_true = predictions.label_ids
        
        # Convert to label names
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(y_pred, y_true):
            pred_labels = []
            true_labels_seq = []
            for p, l in zip(prediction, label):
                if l != -100:
                    pred_labels.append(self.id_to_label[p])
                    true_labels_seq.append(self.id_to_label[l])
            true_predictions.append(pred_labels)
            true_labels.append(true_labels_seq)
        
        # Generate classification report
        from seqeval.metrics import classification_report as seq_classification_report
        report = seq_classification_report(true_labels, true_predictions)
        
        print("\nClassification Report:")
        print(report)
        
        return results, report

def main():
    parser = argparse.ArgumentParser(description='Train RoBERTa NER Model')
    parser.add_argument('--data_path', required=True, help='Path to training data (CSV or JSON)')
    parser.add_argument('--output_dir', required=True, help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NERTrainer(max_length=args.max_length)
    
    # Load data based on file extension
    if args.data_path.endswith('.csv'):
        texts, labels = trainer.load_data_from_csv(args.data_path)
    elif args.data_path.endswith('.json'):
        texts, labels = trainer.load_data_from_json(args.data_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .json")
    
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