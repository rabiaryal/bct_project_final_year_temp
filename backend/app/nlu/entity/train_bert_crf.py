"""Training Script for BERT + CRF Named Entity Recognition"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast, BertModel, BertConfig,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from tqdm import tqdm

# CRF implementation
class CRF(nn.Module):
    """Conditional Random Field for sequence labeling"""
    
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        # Transition matrix: transitions[i][j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # Start and end tag scores
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, emissions, tags, mask):
        """Forward pass for training (negative log likelihood)"""
        return -self._log_likelihood(emissions, tags, mask)
    
    def decode(self, emissions, mask):
        """Viterbi decoding for inference"""
        return self._viterbi_decode(emissions, mask)
    
    def _log_likelihood(self, emissions, tags, mask):
        """Compute log likelihood of the given sequence"""
        seq_len, batch_size = emissions.shape[:2]
        
        # Forward pass
        forward_score = self._forward_algorithm(emissions, mask)
        
        # Gold score
        gold_score = self._score_sentence(emissions, tags, mask)
        
        return gold_score - forward_score
    
    def _forward_algorithm(self, emissions, mask):
        """Forward algorithm to compute partition function"""
        seq_len, batch_size = emissions.shape[:2]
        
        # Initialize
        alpha = emissions[0] + self.start_transitions.unsqueeze(0)
        
        for i in range(1, seq_len):
            emit_score = emissions[i].unsqueeze(1)  # [batch_size, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            next_alpha = alpha.unsqueeze(2) + trans_score + emit_score
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # Apply mask
            alpha = torch.where(mask[i].unsqueeze(1), next_alpha, alpha)
        
        # Add end transitions
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)
    
    def _score_sentence(self, emissions, tags, mask):
        """Score of the gold sequence"""
        seq_len, batch_size = emissions.shape[:2]
        
        # Initialize score
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Only process if there are valid positions
        valid_batches = mask.any(0)
        if not valid_batches.any():
            return score
        
        # Start transitions for valid first positions
        first_mask = mask[0]
        if first_mask.any():
            score += torch.where(first_mask, self.start_transitions[tags[0]], torch.zeros_like(score))
        
        # Emission and transition scores
        for i in range(seq_len - 1):
            current_mask = mask[i]
            next_mask = mask[i + 1]
            
            # Emission scores
            emission_scores = emissions[i].gather(1, tags[i].unsqueeze(1)).squeeze(1)
            score += torch.where(current_mask, emission_scores, torch.zeros_like(score))
            
            # Transition scores
            if next_mask.any():
                transition_scores = self.transitions[tags[i + 1], tags[i]]
                score += torch.where(current_mask & next_mask, transition_scores, torch.zeros_like(score))
        
        # Last emission scores
        last_mask = mask[-1]
        if last_mask.any():
            last_emission_scores = emissions[-1].gather(1, tags[-1].unsqueeze(1)).squeeze(1)
            score += torch.where(last_mask, last_emission_scores, torch.zeros_like(score))
            
            # End transitions
            end_scores = self.end_transitions[tags[-1]]
            score += torch.where(last_mask, end_scores, torch.zeros_like(score))
        
        return score
    
    def _viterbi_decode(self, emissions, mask):
        """Viterbi algorithm for finding best path"""
        seq_len, batch_size = emissions.shape[:2]
        
        # Initialize
        viterbi = emissions[0] + self.start_transitions.unsqueeze(0)
        path = []
        
        for i in range(1, seq_len):
            emit_score = emissions[i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_score = viterbi.unsqueeze(2) + trans_score
            
            viterbi, best_tags = next_score.max(dim=1)
            viterbi = viterbi + emit_score.squeeze(1)
            
            # Apply mask
            viterbi = torch.where(mask[i].unsqueeze(1), viterbi, viterbi)
            path.append(best_tags)
        
        # Backtrack
        viterbi = viterbi + self.end_transitions.unsqueeze(0)
        _, best_last_tags = viterbi.max(dim=1)
        
        best_paths = [best_last_tags.cpu().numpy()]
        for i in reversed(range(len(path))):
            best_last_tags = path[i].gather(1, best_last_tags.unsqueeze(1)).squeeze(1)
            best_paths.append(best_last_tags.cpu().numpy())
        
        return list(reversed(best_paths))

class BertCRFModel(nn.Module):
    """BERT + CRF model for NER"""
    
    def __init__(self, bert_model_name, num_labels, dropout=0.1):
        super(BertCRFModel, self).__init__()
        self.num_labels = num_labels
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass"""
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Classification
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Training mode
            # Create mask that excludes -100 labels and uses attention mask
            valid_mask = (labels != -100) & (attention_mask.bool())
            
            # Only compute loss on valid positions
            if valid_mask.sum() > 0:
                # Transpose for CRF (seq_len, batch_size, num_labels)
                emissions = emissions.transpose(0, 1)
                labels = labels.transpose(0, 1)
                mask = valid_mask.transpose(0, 1)
                
                # Replace -100 with 0 (they'll be masked out anyway)
                labels = labels.clamp(min=0)
                
                loss = self.crf(emissions, labels, mask)
                return {'loss': loss}
            else:
                # No valid labels, return zero loss
                return {'loss': torch.tensor(0.0, device=emissions.device, requires_grad=True)}
        else:
            # Inference mode
            emissions = emissions.transpose(0, 1)
            mask = attention_mask.transpose(0, 1).bool()
            predictions = self.crf.decode(emissions, mask)
            return {'predictions': predictions}
            return {'predictions': predictions}

class NERDataset(Dataset):
    """Dataset for BERT+CRF NER"""
    
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
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            is_split_into_words=True if isinstance(text, list) else False
        )
        
        # Handle labels alignment for subword tokenization
        if isinstance(text, list):
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
        else:
            # Simple alignment for full text
            aligned_labels = [self.label_to_id.get(label, 0) for label in labels[:self.max_length]]
            aligned_labels += [-100] * (self.max_length - len(aligned_labels))
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class BertCRFTrainer:
    """BERT+CRF NER Trainer"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label_to_id = {}
        self.id_to_label = {}
        
    def load_json_data(self, json_path):
        """Load data from the new_entity_json_data.json file"""
        print(f"Loading data from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels_sequences = []
        
        for sentence_data in data['sentences']:
            text = sentence_data['text']
            entities = sentence_data['entities']
            
            # Remove markdown formatting
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'\\1', text)
            
            # Tokenize into words
            words = clean_text.split()
            
            # Create BIO labels
            word_labels = ['O'] * len(words)
            
            # Map entities to BIO format
            for entity_type, entity_value in entities.items():
                # Find entity in text
                entity_words = entity_value.split()
                for i in range(len(words) - len(entity_words) + 1):
                    if words[i:i+len(entity_words)] == entity_words:
                        word_labels[i] = f'B-{entity_type}'
                        for j in range(1, len(entity_words)):
                            if i + j < len(word_labels):
                                word_labels[i + j] = f'I-{entity_type}'
                        break
            
            texts.append(words)
            labels_sequences.append(word_labels)
        
        print(f"Loaded {len(texts)} sentences")
        return texts, labels_sequences
    
    def create_label_mappings(self, all_labels):
        """Create label to ID mappings"""
        unique_labels = sorted(set(label for seq in all_labels for label in seq))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        return self.label_to_id, self.id_to_label
    
    def prepare_datasets(self, texts, labels, test_size=0.2, val_size=0.1):
        """Split data and create datasets"""
        # Create label mappings
        self.create_label_mappings(labels)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )
        
        # Train/validation split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size_adjusted, random_state=42
        )
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = NERDataset(X_train, y_train, self.tokenizer, self.label_to_id, self.max_length)
        val_dataset = NERDataset(X_val, y_val, self.tokenizer, self.label_to_id, self.max_length)
        test_dataset = NERDataset(X_test, y_test, self.tokenizer, self.label_to_id, self.max_length)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset, output_dir, num_epochs=3, batch_size=16, learning_rate=5e-5):
        """Train BERT+CRF model"""
        print("Initializing BERT+CRF model...")
        
        # Initialize model
        self.model = BertCRFModel(
            bert_model_name=self.model_name,
            num_labels=len(self.label_to_id)
        )
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'].mean()  # Average over batch
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss'].mean()
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model
        self.save_model(output_dir)
        
        return train_losses, val_losses
    
    def save_model(self, output_dir):
        """Save trained model and components"""
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'bert_crf_model.pth'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        mappings_path = os.path.join(output_dir, 'label_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump({
                'label2id': self.label_to_id,
                'id2label': self.id_to_label
            }, f, indent=2)
        
        # Save model config
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_labels': len(self.label_to_id),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {output_dir}")
    
    def create_visualizations(self, output_dir, train_losses, val_losses, texts, labels):
        """Create and save training visualizations"""
        graphics_dir = os.path.join(output_dir, 'graphics')
        os.makedirs(graphics_dir, exist_ok=True)
        
        print(f"Creating visualizations in {graphics_dir}...")
        
        # 1. Training Loss Plot
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='s')
        plt.title('BERT+CRF Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(graphics_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Label Distribution
        all_labels = [label for seq in labels for label in seq if label != 'O']
        label_counts = Counter(all_labels)
        
        if label_counts:
            plt.figure(figsize=(12, 6))
            labels_list, counts = zip(*sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
            bars = plt.bar(labels_list, counts)
            plt.title('Entity Label Distribution')
            plt.xlabel('Entity Labels')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphics_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Entity Statistics
        entity_lengths = []
        entity_types = defaultdict(int)
        
        for seq in labels:
            current_entity = []
            current_type = None
            
            for label in seq:
                if label.startswith('B-'):
                    if current_entity:
                        entity_lengths.append(len(current_entity))
                        entity_types[current_type] += 1
                    current_entity = [label]
                    current_type = label[2:]
                elif label.startswith('I-') and current_entity:
                    current_entity.append(label)
                else:
                    if current_entity:
                        entity_lengths.append(len(current_entity))
                        entity_types[current_type] += 1
                        current_entity = []
                        current_type = None
            
            if current_entity:
                entity_lengths.append(len(current_entity))
                entity_types[current_type] += 1
        
        # Entity length distribution
        if entity_lengths:
            plt.figure(figsize=(10, 6))
            plt.hist(entity_lengths, bins=range(1, max(entity_lengths) + 2), alpha=0.7, edgecolor='black')
            plt.title('Distribution of Entity Lengths')
            plt.xlabel('Entity Length (tokens)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(graphics_dir, 'entity_lengths.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Entity Type Distribution
        if entity_types:
            plt.figure(figsize=(10, 6))
            types, counts = zip(*sorted(entity_types.items(), key=lambda x: x[1], reverse=True))
            bars = plt.bar(types, counts)
            plt.title('Entity Type Distribution')
            plt.xlabel('Entity Types')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphics_dir, 'entity_types.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {graphics_dir}")

def main():
    # Get project root directory (go up from current script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '../../../..')  # Go up 4 levels: entity->nlu->app->backend->project_root
    project_root = os.path.abspath(project_root)
    
    print(f"🔍 Project root detected: {project_root}")
    
    parser = argparse.ArgumentParser(description='Train BERT+CRF NER Model')
    parser.add_argument('--data_path', 
                       default=os.path.join(project_root, 'data/entity/new_entity_json_data.json'),
                       help='Path to JSON training data')
    parser.add_argument('--output_dir', 
                       default=os.path.join(project_root, 'models/bert_crf_entity_model'),
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--graphics_dir', 
                       default=os.path.join(project_root, 'docs/graphs/entity'),
                       help='Directory to save graphics and visualizations')
    
    args = parser.parse_args()
    
    print("🚀 Starting BERT+CRF NER Training")
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
    trainer = BertCRFTrainer(max_length=args.max_length)
    
    # Load data from JSON file
    texts, labels = trainer.load_json_data(args.data_path)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = trainer.prepare_datasets(texts, labels)
    
    # Train model
    train_losses, val_losses = trainer.train(
        train_dataset, val_dataset, args.output_dir,
        num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
    )
    
    # Create visualizations
    trainer.create_visualizations(args.graphics_dir, train_losses, val_losses, texts, labels)
    
    # Save training summary
    summary = {
        'model': 'BERT+CRF',
        'data_source': args.data_path,
        'num_samples': len(texts),
        'num_labels': len(trainer.label_to_id),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'labels': list(trainer.label_to_id.keys()),
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Training complete! Results saved to {args.output_dir}")
    print(f"📊 Graphics saved to {args.graphics_dir}/")

if __name__ == "__main__":
    main()