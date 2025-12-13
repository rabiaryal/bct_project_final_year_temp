"""
Transformer-based Entity Extractor
Fine-tuned for extracting colleges, locations, courses, and affiliations
Uses token classification (NER-style) approach
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
import re


class EntityDataset(Dataset):
    """Dataset for entity extraction training using simple matching"""
    
    def __init__(self, texts: List[str], entity_labels: List[Dict], tokenizer, 
                 label2id: Dict, max_length: int = 128):
        self.texts = texts
        self.entity_labels = entity_labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        entities = self.entity_labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Create labels for each token
        labels = [self.label2id['O']] * self.max_length
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()
        
        # Mark entity tokens
        for entity_type, entity_text in entities.items():
            if entity_text and isinstance(entity_text, str) and entity_text.strip():
                entity_text_lower = entity_text.lower()
                text_lower = text.lower()
                
                # Find entity position in text
                start_pos = text_lower.find(entity_text_lower)
                if start_pos != -1:
                    end_pos = start_pos + len(entity_text_lower)
                    
                    # Map to token positions
                    for i, (token_start, token_end) in enumerate(offset_mapping):
                        if token_start == token_end:  # Special token
                            continue
                        if token_start >= start_pos and token_end <= end_pos:
                            # This token is part of the entity
                            if token_start == start_pos:
                                labels[i] = self.label2id[f'B-{entity_type.upper()}']
                            else:
                                labels[i] = self.label2id[f'I-{entity_type.upper()}']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class TransformerEntityExtractor:
    """
    Transformer-based entity extractor for college domain
    Extracts entities at College, Department, and Course levels
    """
    
    # Entity types with their levels and value types
    # College Level: college_name, college_type, location, hostel_availability
    # Department Level: department_name
    # Course Level: course_name, fee, rating, admission_process, cutoff_rank, 
    #               internship_opportunities, scholarship
    ENTITY_TYPES = [
        'COLLEGE_NAME',        # categorical - college names
        'COLLEGE_TYPE',        # enum - private/public/community
        'LOCATION',            # categorical - city/district names
        'HOSTEL_AVAILABILITY', # boolean - yes/no/available/not available
        'DEPARTMENT_NAME',     # categorical - department names
        'COURSE_NAME',         # categorical - course/program names
        'FEE',                 # numeric - fee amounts
        'RATING',              # numeric - ratings/rankings
        'ADMISSION_PROCESS',   # categorical - entrance/merit/quota
        'CUTOFF_RANK',         # numeric - cutoff ranks
        'INTERNSHIP',          # boolean - yes/no
        'SCHOLARSHIP'          # numeric/categorical - scholarship info
    ]
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        self.model_path = model_path
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # BIO labels: O, B-COLLEGE, I-COLLEGE, B-LOCATION, I-LOCATION, etc.
        self.labels = ['O']
        for entity_type in self.ENTITY_TYPES:
            self.labels.append(f'B-{entity_type}')
            self.labels.append(f'I-{entity_type}')
        
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        
        self.tokenizer = None
        self.model = None
        self.model_ready = False
        
        # Try to load fine-tuned model
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️  No fine-tuned entity model found. Call train() first.")
    
    def _load_model(self, model_path: str):
        """Load fine-tuned model from path"""
        try:
            print(f"📦 Loading entity model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.model_ready = True
            print(f"✅ Entity model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load entity model: {e}")
            self.model_ready = False
    
    def train(self, csv_path: str, output_dir: str = './models/entity_model',
              epochs: int = 10, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Fine-tune the model on entity extraction data
        
        CSV should have columns: text, plus entity columns matching ENTITY_TYPES
        Entity columns: college_name, college_type, location, hostel_availability,
                       department_name, course_name, fee, rating, admission_process,
                       cutoff_rank, internship, scholarship
        """
        print(f"🚀 Starting entity extraction training on {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples")
        
        texts = df['text'].tolist()
        
        # Map CSV columns to entity types
        column_to_entity = {
            'college_name': 'college_name',
            'college_type': 'college_type',
            'location': 'location',
            'hostel_availability': 'hostel_availability',
            'department_name': 'department_name',
            'course_name': 'course_name',
            'fee': 'fee',
            'rating': 'rating',
            'admission_process': 'admission_process',
            'cutoff_rank': 'cutoff_rank',
            'internship': 'internship',
            'scholarship': 'scholarship'
        }
        
        # Build entity labels for each sample
        entity_labels = []
        for _, row in df.iterrows():
            entities = {}
            for col, entity_key in column_to_entity.items():
                if col in df.columns and pd.notna(row[col]):
                    entities[entity_key] = str(row[col])
            entity_labels.append(entities)
        
        # Initialize tokenizer and model
        print(f"📦 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Create dataset
        dataset = EntityDataset(texts, entity_labels, self.tokenizer, self.label2id)
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy='epoch',
            load_best_model_at_end=False,
            report_to='none'
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        # Train
        print("🏋️ Training entity extractor...")
        trainer.train()
        
        # Save model
        print(f"💾 Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Reload for inference
        self.model_path = output_dir
        self._load_model(output_dir)
        
        print("✅ Entity training complete!")
        return output_dir
    
    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract entities from text
        
        Returns:
            Dict with all entity types extracted
        """
        if not self.model_ready:
            # Fallback to keyword-based extraction
            return self._keyword_fallback(text)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        offset_mapping = inputs.pop('offset_mapping').squeeze().tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
        
        # Initialize all entity types
        entities = {
            'college_name': None,
            'college_type': None,
            'location': None,
            'hostel_availability': None,
            'department_name': None,
            'course_name': None,
            'fee': None,
            'rating': None,
            'admission_process': None,
            'cutoff_rank': None,
            'internship': None,
            'scholarship': None
        }
        
        current_entity = None
        current_type = None
        current_start = None
        
        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue
            
            label = self.id2label[pred]
            
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity and current_type:
                    key = current_type.lower()
                    if key in entities:
                        entities[key] = current_entity
                
                # Start new entity
                current_type = label[2:]
                current_entity = text[start:end]
                current_start = start
                
            elif label.startswith('I-') and current_type == label[2:]:
                # Continue entity
                current_entity = text[current_start:end]
                
            else:
                # Save previous entity if exists
                if current_entity and current_type:
                    key = current_type.lower()
                    if key in entities:
                        entities[key] = current_entity
                
                current_entity = None
                current_type = None
                current_start = None
        
        # Don't forget the last entity
        if current_entity and current_type:
            key = current_type.lower()
            if key in entities:
                entities[key] = current_entity
        
        return entities
    
    def _keyword_fallback(self, text: str) -> Dict[str, Optional[str]]:
        """Fallback keyword-based entity extraction"""
        text_lower = text.lower()
        
        entities = {
            'college_name': None,
            'college_type': None,
            'location': None,
            'hostel_availability': None,
            'department_name': None,
            'course_name': None,
            'fee': None,
            'rating': None,
            'admission_process': None,
            'cutoff_rank': None,
            'internship': None,
            'scholarship': None
        }
        
        # College name patterns
        colleges = [
            'sagarmatha', 'pulchowk', 'thapathali', 'ncit', 'nec', 'ace',
            'kantipur', 'himalaya', 'islington', 'softwarica', 'herald',
            'apex', 'prime', 'kusom', 'wrc', 'khwopa', 'kathmandu engineering',
            'advanced college', 'national college'
        ]
        for college in colleges:
            if college in text_lower:
                entities['college_name'] = college.title()
                break
        
        # College type patterns
        if 'private' in text_lower:
            entities['college_type'] = 'private'
        elif 'public' in text_lower or 'government' in text_lower:
            entities['college_type'] = 'public'
        elif 'community' in text_lower:
            entities['college_type'] = 'community'
        
        # Location patterns
        locations = [
            'kathmandu', 'lalitpur', 'bhaktapur', 'pokhara', 'chitwan',
            'butwal', 'biratnagar', 'dharan', 'hetauda', 'birgunj'
        ]
        for location in locations:
            if location in text_lower:
                entities['location'] = location.title()
                break
        
        # Hostel availability
        if 'hostel' in text_lower:
            if 'no hostel' in text_lower or 'without hostel' in text_lower:
                entities['hostel_availability'] = 'no'
            else:
                entities['hostel_availability'] = 'yes'
        
        # Department patterns
        departments = [
            'computer', 'civil', 'electrical', 'mechanical', 'electronics',
            'architecture', 'it', 'software', 'information technology'
        ]
        for dept in departments:
            if dept in text_lower:
                entities['department_name'] = dept.title()
                break
        
        # Course patterns
        courses = [
            'computer engineering', 'civil engineering', 'electrical engineering',
            'mechanical engineering', 'electronics engineering', 'architecture',
            'bba', 'mba', 'bim', 'bca', 'bsc csit', 'be computer', 'be civil'
        ]
        for course in courses:
            if course in text_lower:
                entities['course_name'] = course.title()
                break
        
        # Fee patterns (look for numbers with lakhs/rupees)
        import re
        # Handle decimal numbers like 11.5 lakh
        fee_match = re.search(r'(\d+(?:\.\d+)?(?:,\d+)*)\s*(?:lakhs?|lakh|rupees?|rs\.?|npr)', text_lower)
        if fee_match:
            fee_str = fee_match.group(1).replace(',', '')
            fee_value = float(fee_str)
            # Check if it's in lakhs
            if 'lakh' in text_lower:
                fee_value = int(fee_value * 100000)
            entities['fee'] = fee_value
        else:
            # Try to match patterns like "under 5 lakh" or "below 10 lakh"
            fee_match = re.search(r'(?:under|below|less\s+than|within)\s*(\d+(?:\.\d+)?)\s*(?:lakhs?|lakh)', text_lower)
            if fee_match:
                fee_str = fee_match.group(1)
                fee_value = float(fee_str) * 100000
                entities['fee'] = int(fee_value)
        
        # Rating patterns
        rating_match = re.search(r'rating\s*(?:of\s*)?(\d+(?:\.\d+)?)', text_lower)
        if rating_match:
            entities['rating'] = rating_match.group(1)
        
        # Admission process
        if 'entrance' in text_lower:
            entities['admission_process'] = 'entrance'
        elif 'merit' in text_lower:
            entities['admission_process'] = 'merit'
        elif 'quota' in text_lower:
            entities['admission_process'] = 'quota'
        
        # Cutoff rank
        cutoff_match = re.search(r'(?:cutoff|cut-off|rank)\s*(?:of\s*)?(\d+)', text_lower)
        if cutoff_match:
            entities['cutoff_rank'] = cutoff_match.group(1)
        
        # Internship
        if 'internship' in text_lower:
            if 'no internship' in text_lower:
                entities['internship'] = 'no'
            else:
                entities['internship'] = 'yes'
        
        # Scholarship
        if 'scholarship' in text_lower:
            scholarship_match = re.search(r'(\d+)%?\s*scholarship', text_lower)
            if scholarship_match:
                entities['scholarship'] = scholarship_match.group(1) + '%'
            else:
                entities['scholarship'] = 'available'
        
        return entities


# Convenience function
def create_entity_extractor(model_path: Optional[str] = None) -> TransformerEntityExtractor:
    """Factory function to create entity extractor"""
    default_path = './models/entity_model'
    path = model_path or default_path
    
    if os.path.exists(path):
        return TransformerEntityExtractor(model_path=path)
    else:
        print(f"⚠️  No entity model at {path}. Use extractor.train() to fine-tune.")
        return TransformerEntityExtractor()


# Training script
if __name__ == "__main__":
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    
    print("🏷️ Transformer Entity Extractor Training")
    print("=" * 50)
    
    extractor = TransformerEntityExtractor()
    
    # Train on entity_training_data.csv
    if os.path.exists('entity_training_data.csv'):
        extractor.train(
            csv_path='entity_training_data.csv',
            output_dir='./models/entity_model',
            epochs=15,
            batch_size=4
        )
        
        # Test predictions
        print("\n🧪 Testing entity extraction:")
        test_prompts = [
            "Where is Sagarmatha Engineering College located?",
            "Best colleges in Kathmandu",
            "Computer engineering fee at Pulchowk",
            "IOE affiliated colleges in Pokhara",
            "MBA programs at KUSOM"
        ]
        
        for prompt in test_prompts:
            entities = extractor.extract_entities(prompt)
            print(f"'{prompt}'")
            print(f"  → {entities}")
            print()
    else:
        print("❌ entity_training_data.csv not found!")
        print("   Create it with columns: text, college, location, course, affiliation")
