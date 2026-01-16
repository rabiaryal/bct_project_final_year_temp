"""
Data loading utilities for the NER model.
"""
import json
import pandas as pd
import torch
from torch.utils.data import Dataset

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
                label = labels[word_idx] if word_idx < len(labels) else 'O'
                aligned_labels.append(self.label_to_id.get(label, self.label_to_id.get('O', 0)))
            else:
                # Subsequent subwords: Use the "I-" version of the label if it's an entity
                label = labels[word_idx] if word_idx < len(labels) else 'O'
                if label.startswith('B-'):
                    label = 'I-' + label[2:]  # Convert B to I for subwords
                aligned_labels.append(self.label_to_id.get(label, self.label_to_id.get('O', 0)))
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def load_data_from_json(json_path):
    """Load NER data from JSON and handle multiple entities per sentence."""
    print(f"Loading data from {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    labels = []
    
    # Handle both list format and the "sentences" key format
    sentences_data = data.get('sentences', data if isinstance(data, list) else [])
    
    for sentence_data in sentences_data:
        if isinstance(sentence_data, dict):
            text = sentence_data.get('text', '')
            entities = sentence_data.get('entities', {})
            
            # 1. Clean formatting but keep word structure consistent
            clean_text = text.replace('**', '')
            words = clean_text.split()
            
            # 2. Initialize labels as 'O' (Outside)
            word_labels = ['O'] * len(words)
            
            # 3. Sort entities by length (longest first) 
            # This prevents a short entity from 'stealing' words from a longer one
            entity_items = []
            for e_type, e_vals in entities.items():
                if isinstance(e_vals, list):
                    for v in e_vals: entity_items.append((e_type, v))
                else:
                    entity_items.append((e_type, e_vals))
            
            # Sort by number of words in the value (descending)
            entity_items.sort(key=lambda x: len(str(x[1]).split()), reverse=True)

            # 4. Map entities to BIO format
            for entity_type, entity_value in entity_items:
                if not entity_value or not isinstance(entity_value, str):
                    continue

                entity_words = entity_value.split()
                entity_len = len(entity_words)
                
                # Scan every possible position in the sentence
                for i in range(len(words) - entity_len + 1):
                    # Check if the sequence of words matches the entity value
                    if words[i:i+entity_len] == entity_words:
                        # CRITICAL: Only label if these words haven't been assigned yet
                        if all(label == 'O' for label in word_labels[i:i+entity_len]):
                            word_labels[i] = f'B-{entity_type}'
                            for j in range(1, entity_len):
                                word_labels[i + j] = f'I-{entity_type}'
                            # We don't 'break' here, allowing multiple occurrences 
                            # of the same entity value in one sentence.
            
            if len(words) > 0:
                texts.append(' '.join(words))
                labels.append(word_labels)
    
    print(f"Successfully loaded {len(texts)} sentences with multi-entity support.")
    return texts, labels

def load_data_from_csv(csv_path):
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
