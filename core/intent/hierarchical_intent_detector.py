"""
Hierarchical Transformer-based Intent Detector
Uses a two-level classification approach:
1. Primary intents: greeting, recommendation, comparison, direct_question
2. Sub-intents for direct_question: fee, location, hostel, admission, rating, courses, facilities, etc.
"""

import os
import torch
import pandas as pd
from typing import Dict, Tuple, List, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from torch.utils.data import Dataset
import numpy as np


class IntentDataset(Dataset):
    """Dataset for intent classification training"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class HierarchicalIntentDetector:
    """
    Hierarchical Transformer-based Intent Detector
    
    Intent Hierarchy:
    ├── greeting
    ├── recommendation  
    ├── comparison
    └── direct_question
        ├── fee
        ├── location
        ├── hostel
        ├── admission
        ├── rating
        ├── courses
        ├── facilities
        ├── cutoff
        ├── scholarship
        ├── internship
        ├── placement
        ├── eligibility
        ├── duration
        ├── seats
        ├── faculty
        ├── department
        └── contact
    """
    
    # Primary intent classes
    PRIMARY_INTENTS = ['greeting', 'recommendation', 'comparison', 'direct_question']
    
    # Sub-intents for direct_question
    SUB_INTENTS = [
        'fee', 'location', 'hostel', 'admission', 'rating', 'courses',
        'facilities', 'cutoff', 'scholarship', 'internship', 'placement',
        'eligibility', 'duration', 'seats', 'faculty', 'department', 'contact', 'general_info'
    ]
    
    def __init__(self, 
                 primary_model_path: Optional[str] = None,
                 sub_model_path: Optional[str] = None,
                 model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        self.primary_model_path = primary_model_path
        self.sub_model_path = sub_model_path
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Primary intent mappings
        self.primary_intent_to_id = {intent: i for i, intent in enumerate(self.PRIMARY_INTENTS)}
        self.primary_id_to_intent = {i: intent for i, intent in enumerate(self.PRIMARY_INTENTS)}
        
        # Sub-intent mappings
        self.sub_intent_to_id = {intent: i for i, intent in enumerate(self.SUB_INTENTS)}
        self.sub_id_to_intent = {i: intent for i, intent in enumerate(self.SUB_INTENTS)}
        
        # Models
        self.primary_tokenizer = None
        self.primary_model = None
        self.primary_ready = False
        
        self.sub_tokenizer = None
        self.sub_model = None
        self.sub_ready = False
        
        # Load models if paths provided
        if primary_model_path and os.path.exists(primary_model_path):
            self._load_primary_model(primary_model_path)
        
        if sub_model_path and os.path.exists(sub_model_path):
            self._load_sub_model(sub_model_path)
    
    def _load_primary_model(self, model_path: str):
        """Load the primary intent classifier"""
        try:
            print(f"📦 Loading primary intent model from {model_path}...")
            self.primary_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.primary_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.primary_model.to(self.device)
            self.primary_model.eval()
            self.primary_ready = True
            print(f"✅ Primary model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load primary model: {e}")
            self.primary_ready = False
    
    def _load_sub_model(self, model_path: str):
        """Load the sub-intent classifier"""
        try:
            print(f"📦 Loading sub-intent model from {model_path}...")
            self.sub_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.sub_model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.sub_model.to(self.device)
            self.sub_model.eval()
            self.sub_ready = True
            print(f"✅ Sub-intent model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load sub-intent model: {e}")
            self.sub_ready = False
    
    def train_primary(self, csv_path: str, output_dir: str = './models/primary_intent_model',
                      epochs: int = 10, batch_size: int = 8, learning_rate: float = 3e-5):
        """
        Train the primary intent classifier
        
        CSV should have columns: text, primary_intent
        primary_intent values: greeting, recommendation, comparison, direct_question
        """
        print(f"🚀 Training PRIMARY intent classifier on {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples")
        
        texts = df['text'].tolist()
        labels = [self.primary_intent_to_id[intent] for intent in df['primary_intent'].tolist()]
        
        # Initialize tokenizer and model
        print(f"📦 Loading {self.model_name}...")
        self.primary_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.primary_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.PRIMARY_INTENTS),
            id2label=self.primary_id_to_intent,
            label2id=self.primary_intent_to_id
        )
        
        # Create dataset
        dataset = IntentDataset(texts, labels, self.primary_tokenizer)
        
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
            model=self.primary_model,
            args=training_args,
            train_dataset=dataset
        )
        
        # Train
        print("🏋️ Training primary intent classifier...")
        trainer.train()
        
        # Save model
        print(f"💾 Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.primary_tokenizer.save_pretrained(output_dir)
        
        # Reload for inference
        self.primary_model_path = output_dir
        self._load_primary_model(output_dir)
        
        print("✅ Primary intent training complete!")
        return output_dir
    
    def train_sub_intent(self, csv_path: str, output_dir: str = './models/sub_intent_model',
                         epochs: int = 10, batch_size: int = 8, learning_rate: float = 3e-5):
        """
        Train the sub-intent classifier (for direct_question intents)
        
        CSV should have columns: text, sub_intent
        Only include samples where primary_intent == 'direct_question'
        """
        print(f"🚀 Training SUB-INTENT classifier on {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples")
        
        texts = df['text'].tolist()
        labels = [self.sub_intent_to_id[intent] for intent in df['sub_intent'].tolist()]
        
        # Initialize tokenizer and model
        print(f"📦 Loading {self.model_name}...")
        self.sub_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sub_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.SUB_INTENTS),
            id2label=self.sub_id_to_intent,
            label2id=self.sub_intent_to_id
        )
        
        # Create dataset
        dataset = IntentDataset(texts, labels, self.sub_tokenizer)
        
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
            model=self.sub_model,
            args=training_args,
            train_dataset=dataset
        )
        
        # Train
        print("🏋️ Training sub-intent classifier...")
        trainer.train()
        
        # Save model
        print(f"💾 Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.sub_tokenizer.save_pretrained(output_dir)
        
        # Reload for inference
        self.sub_model_path = output_dir
        self._load_sub_model(output_dir)
        
        print("✅ Sub-intent training complete!")
        return output_dir
    
    def detect_intent(self, prompt: str) -> Dict:
        """
        Detect intent hierarchically
        
        Returns:
            Dict with:
                - primary_intent: greeting/recommendation/comparison/direct_question
                - primary_confidence: float
                - sub_intent: (only if primary is direct_question) fee/location/hostel/etc.
                - sub_confidence: (only if primary is direct_question) float
                - full_intent: combined intent string (e.g., "direct_question.fee")
        """
        result = {
            'primary_intent': None,
            'primary_confidence': 0.0,
            'sub_intent': None,
            'sub_confidence': 0.0,
            'full_intent': None,
            'method': 'hierarchical_transformer'
        }
        
        # Step 1: Detect primary intent
        if not self.primary_ready:
            # Fallback to keyword-based
            result.update(self._keyword_fallback_primary(prompt))
        else:
            primary_intent, primary_conf = self._predict_primary(prompt)
            result['primary_intent'] = primary_intent
            result['primary_confidence'] = primary_conf
        
        # Step 2: If primary is direct_question, detect sub-intent
        if result['primary_intent'] == 'direct_question':
            if not self.sub_ready:
                # Fallback to keyword-based
                sub_result = self._keyword_fallback_sub(prompt)
                result['sub_intent'] = sub_result['sub_intent']
                result['sub_confidence'] = sub_result['sub_confidence']
            else:
                sub_intent, sub_conf = self._predict_sub(prompt)
                result['sub_intent'] = sub_intent
                result['sub_confidence'] = sub_conf
            
            result['full_intent'] = f"direct_question.{result['sub_intent']}"
        else:
            result['full_intent'] = result['primary_intent']
        
        return result
    
    def detect_multi_intent(self, prompt: str, threshold: float = 0.15) -> Dict:
        """
        Detect MULTIPLE sub-intents in a single query.
        
        For queries like "What is the fee and location of NCIT?" this returns
        both 'fee' and 'location' as detected intents.
        
        Args:
            prompt: User's query
            threshold: Minimum probability to consider a sub-intent (default 0.15)
            
        Returns:
            Dict with:
                - primary_intent: greeting/recommendation/comparison/direct_question
                - primary_confidence: float
                - sub_intents: List of (sub_intent, confidence) tuples
                - dominant_sub_intent: The highest confidence sub-intent
                - full_intent: Combined intent string
                - is_multi_intent: True if multiple sub-intents detected
        """
        # First get primary intent
        result = {
            'primary_intent': None,
            'primary_confidence': 0.0,
            'sub_intents': [],
            'dominant_sub_intent': None,
            'dominant_sub_confidence': 0.0,
            'full_intent': None,
            'is_multi_intent': False,
            'method': 'multi_intent_detection'
        }
        
        # Detect primary intent
        if not self.primary_ready:
            primary_result = self._keyword_fallback_primary(prompt)
            result['primary_intent'] = primary_result['primary_intent']
            result['primary_confidence'] = primary_result['primary_confidence']
        else:
            primary_intent, primary_conf = self._predict_primary(prompt)
            result['primary_intent'] = primary_intent
            result['primary_confidence'] = primary_conf
        
        # For non-direct_question, just return single intent
        if result['primary_intent'] != 'direct_question':
            result['full_intent'] = result['primary_intent']
            return result
        
        # For direct_question, detect ALL sub-intents above threshold
        if self.sub_ready:
            sub_intents = self._predict_multi_sub(prompt, threshold)
        else:
            sub_intents = self._keyword_multi_sub(prompt)
        
        result['sub_intents'] = sub_intents
        result['is_multi_intent'] = len(sub_intents) > 1
        
        if sub_intents:
            # Sort by confidence and get dominant
            sub_intents.sort(key=lambda x: x[1], reverse=True)
            result['dominant_sub_intent'] = sub_intents[0][0]
            result['dominant_sub_confidence'] = sub_intents[0][1]
            
            # Create full intent string
            if len(sub_intents) == 1:
                result['full_intent'] = f"direct_question.{sub_intents[0][0]}"
            else:
                intent_names = [s[0] for s in sub_intents]
                result['full_intent'] = f"direct_question.[{'+'.join(intent_names)}]"
        else:
            result['dominant_sub_intent'] = 'general_info'
            result['dominant_sub_confidence'] = 0.5
            result['full_intent'] = 'direct_question.general_info'
        
        return result
    
    def _predict_multi_sub(self, prompt: str, threshold: float = 0.15) -> List[Tuple[str, float]]:
        """
        Predict multiple sub-intents using transformer probabilities.
        Returns all sub-intents with probability above threshold.
        """
        inputs = self.sub_tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sub_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        
        # Get all sub-intents above threshold
        detected = []
        for idx, prob in enumerate(probs):
            if prob.item() >= threshold:
                intent_name = self.sub_id_to_intent[idx]
                detected.append((intent_name, prob.item()))
        
        # Sort by probability
        detected.sort(key=lambda x: x[1], reverse=True)
        
        return detected
    
    def _keyword_multi_sub(self, prompt: str) -> List[Tuple[str, float]]:
        """
        Keyword-based multi-intent detection fallback.
        Detects all matching sub-intents from keywords.
        """
        prompt_lower = prompt.lower()
        
        sub_intent_patterns = {
            'fee': ['fee', 'cost', 'price', 'tuition', 'charges', 'expensive', 'affordable'],
            'location': ['location', 'where', 'address', 'located', 'place', 'area'],
            'hostel': ['hostel', 'accommodation', 'dormitory', 'stay', 'residence'],
            'admission': ['admission', 'apply', 'application', 'enroll', 'entrance'],
            'rating': ['rating', 'rank', 'ranking', 'rated', 'review', 'reputation'],
            'courses': ['course', 'program', 'degree', 'branch', 'stream'],
            'facilities': ['facility', 'facilities', 'infrastructure', 'lab', 'library'],
            'cutoff': ['cutoff', 'cut-off', 'cut off', 'minimum marks'],
            'scholarship': ['scholarship', 'financial aid', 'fee waiver'],
            'internship': ['internship', 'intern', 'training', 'industrial'],
            'placement': ['placement', 'job', 'career', 'recruit', 'salary'],
            'eligibility': ['eligible', 'eligibility', 'criteria', 'requirement'],
            'duration': ['duration', 'years', 'semester', 'how long'],
            'seats': ['seats', 'capacity', 'intake', 'vacancies'],
            'faculty': ['faculty', 'teacher', 'professor', 'staff'],
            'contact': ['contact', 'phone', 'email', 'number', 'call']
        }
        
        detected = []
        for sub_intent, patterns in sub_intent_patterns.items():
            if any(p in prompt_lower for p in patterns):
                detected.append((sub_intent, 0.7))
        
        if not detected:
            detected.append(('general_info', 0.5))
        
        return detected
    
    def _predict_primary(self, prompt: str) -> Tuple[str, float]:
        """Predict primary intent using transformer"""
        inputs = self.primary_tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.primary_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()
        
        return self.primary_id_to_intent[predicted_id], confidence
    
    def _predict_sub(self, prompt: str) -> Tuple[str, float]:
        """Predict sub-intent using transformer"""
        inputs = self.sub_tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sub_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()
        
        return self.sub_id_to_intent[predicted_id], confidence
    
    def _keyword_fallback_primary(self, prompt: str) -> Dict:
        """Keyword-based fallback for primary intent"""
        prompt_lower = prompt.lower()
        
        # Greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 
                           'good evening', 'namaste', 'greetings', 'howdy']
        if any(g in prompt_lower for g in greeting_patterns):
            return {'primary_intent': 'greeting', 'primary_confidence': 0.8}
        
        # Recommendation patterns
        recommend_patterns = ['recommend', 'suggest', 'best', 'top', 'which college', 
                            'which university', 'should i join', 'good college']
        if any(r in prompt_lower for r in recommend_patterns):
            return {'primary_intent': 'recommendation', 'primary_confidence': 0.7}
        
        # Comparison patterns
        compare_patterns = ['compare', 'vs', 'versus', 'difference between', 'better than',
                          'which is better', 'comparison', 'or']
        if any(c in prompt_lower for c in compare_patterns):
            return {'primary_intent': 'comparison', 'primary_confidence': 0.7}
        
        # Default to direct_question
        return {'primary_intent': 'direct_question', 'primary_confidence': 0.6}
    
    def _keyword_fallback_sub(self, prompt: str) -> Dict:
        """Keyword-based fallback for sub-intent"""
        prompt_lower = prompt.lower()
        
        sub_intent_patterns = {
            'fee': ['fee', 'cost', 'price', 'tuition', 'charges', 'expensive', 'affordable'],
            'location': ['location', 'where', 'address', 'located', 'place', 'area', 'distance'],
            'hostel': ['hostel', 'accommodation', 'dormitory', 'stay', 'residence', 'room'],
            'admission': ['admission', 'apply', 'application', 'enroll', 'entrance', 'process'],
            'rating': ['rating', 'rank', 'ranking', 'rated', 'review', 'reputation'],
            'courses': ['course', 'program', 'degree', 'branch', 'stream', 'subject', 'syllabus'],
            'facilities': ['facility', 'facilities', 'infrastructure', 'lab', 'library', 'sports', 'canteen'],
            'cutoff': ['cutoff', 'cut-off', 'cut off', 'minimum', 'required marks'],
            'scholarship': ['scholarship', 'financial aid', 'fee waiver', 'discount'],
            'internship': ['internship', 'intern', 'training', 'industrial', 'practical'],
            'placement': ['placement', 'job', 'career', 'recruit', 'company', 'salary', 'package'],
            'eligibility': ['eligible', 'eligibility', 'criteria', 'requirement', 'qualify'],
            'duration': ['duration', 'years', 'semester', 'how long', 'time period'],
            'seats': ['seats', 'capacity', 'intake', 'how many students', 'vacancies'],
            'faculty': ['faculty', 'teacher', 'professor', 'staff', 'lecturer', 'instructor'],
            'department': ['department', 'head', 'hod', 'dean'],
            'contact': ['contact', 'phone', 'email', 'number', 'call', 'reach']
        }
        
        for sub_intent, patterns in sub_intent_patterns.items():
            if any(p in prompt_lower for p in patterns):
                return {'sub_intent': sub_intent, 'sub_confidence': 0.7}
        
        return {'sub_intent': 'general_info', 'sub_confidence': 0.5}


def create_hierarchical_detector(primary_path: str = './models/primary_intent_model',
                                  sub_path: str = './models/sub_intent_model') -> HierarchicalIntentDetector:
    """Factory function to create hierarchical intent detector"""
    return HierarchicalIntentDetector(
        primary_model_path=primary_path if os.path.exists(primary_path) else None,
        sub_model_path=sub_path if os.path.exists(sub_path) else None
    )


# Training script
if __name__ == "__main__":
    print("Hierarchical Intent Detector")
    print("=" * 50)
    print("\nTo train the models, use:")
    print("  detector = HierarchicalIntentDetector()")
    print("  detector.train_primary('primary_intent_data.csv')")
    print("  detector.train_sub_intent('sub_intent_data.csv')")
