"""
Transformer-based Intent Detector
Fine-tuned on test_data.csv for accurate intent classification
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


class TransformerIntentDetector:
    """
    Transformer-based intent detector fine-tuned on domain data
    Supports 24 intent classes from test_data.csv
    """
    
    # All 24 intents from test_data.csv
    INTENT_CLASSES = [
        'Admission_process', 'Best', 'College_basic_info', 'College_contact',
        'College_location', 'Compare_courses', 'Course_cutoff', 'Course_duration',
        'Course_fee', 'Course_list', 'Course_rating', 'Course_seats',
        'Course_specific_info', 'Department_head', 'Department_info',
        'Eligibility_criteria', 'Faculty_info', 'Greeting', 'Hostel_availability',
        'Internship_opportunities', 'Nearest', 'Placement_info', 'Recommend',
        'Scholarship_info'
    ]
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        self.model_path = model_path
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.intent_to_id = {intent: i for i, intent in enumerate(self.INTENT_CLASSES)}
        self.id_to_intent = {i: intent for i, intent in enumerate(self.INTENT_CLASSES)}
        
        self.tokenizer = None
        self.model = None
        self.model_ready = False
        
        # Entity patterns for extraction
        self._init_entity_patterns()
        
        # Try to load fine-tuned model if available
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("⚠️  No fine-tuned model found. Call train() first or use SafeIntentDetector.")
    
    def _init_entity_patterns(self):
        """Initialize entity extraction patterns"""
        self.college_keywords = [
            'sagarmatha', 'sec', 'pulchowk', 'ioe', 'kathmandu university', 'ku',
            'tribhuvan', 'tu', 'ace', 'advanced college', 'kantipur', 'himalaya',
            'thapathali', 'wrc', 'paschimanchal', 'purwanchal', 'khwopa',
            'nepal engineering college', 'nec', 'ncit', 'islington', 'softwarica',
            'herald', 'apex', 'prime', 'lumbini', 'gandaki'
        ]
        
        self.location_keywords = [
            'kathmandu', 'lalitpur', 'bhaktapur', 'pokhara', 'chitwan', 'butwal',
            'biratnagar', 'janakpur', 'dharan', 'hetauda', 'birgunj', 'nepalgunj',
            'dhangadhi', 'itahari', 'bharatpur', 'kirtipur', 'patan', 'thimi'
        ]
        
        self.course_keywords = [
            'computer engineering', 'civil engineering', 'electrical engineering',
            'mechanical engineering', 'electronics', 'architecture', 'bba', 'mba',
            'bim', 'bca', 'bsc csit', 'be computer', 'be civil', 'be electrical',
            'information technology', 'it', 'software engineering'
        ]
        
        self.affiliation_keywords = [
            'ioe', 'tribhuvan university', 'tu', 'kathmandu university', 'ku',
            'pokhara university', 'pu', 'purbanchal university'
        ]
    
    def _load_model(self, model_path: str):
        """Load fine-tuned model from path"""
        try:
            print(f"📦 Loading fine-tuned model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.model_ready = True
            print(f"✅ Model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_ready = False
    
    def train(self, csv_path: str, output_dir: str = './models/intent_model', 
              epochs: int = 5, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Fine-tune the model on intent classification data
        
        Args:
            csv_path: Path to CSV with 'text' and 'intent' columns
            output_dir: Where to save the fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print(f"🚀 Starting training on {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples")
        
        # Get column names
        text_col = df.columns[0]  # First column is text
        intent_col = df.columns[1]  # Second column is intent
        
        texts = df[text_col].tolist()
        intents = df[intent_col].tolist()
        
        # Convert intents to IDs
        labels = []
        for intent in intents:
            if intent in self.intent_to_id:
                labels.append(self.intent_to_id[intent])
            else:
                print(f"⚠️  Unknown intent: {intent}")
                labels.append(0)  # Default
        
        # Initialize tokenizer and model
        print(f"📦 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.INTENT_CLASSES)
        )
        
        # Create dataset
        dataset = IntentDataset(texts, labels, self.tokenizer)
        
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
            report_to='none',  # Disable wandb
            use_mps_device=(self.device == 'mps')
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )
        
        # Train
        print("🏋️ Training...")
        trainer.train()
        
        # Save model
        print(f"💾 Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Reload for inference
        self.model_path = output_dir
        self._load_model(output_dir)
        
        print("✅ Training complete!")
        return output_dir
    
    def detect_intent(self, prompt: str) -> Tuple[str, float, Dict]:
        """
        Detect intent using the fine-tuned transformer
        
        Returns:
            (intent, confidence, details)
        """
        if not self.model_ready:
            # Fallback to keyword-based
            from core.intent.safe_intent_detector import SafeIntentDetector
            fallback = SafeIntentDetector()
            return fallback.detect_intent(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            predicted_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_id].item()
        
        intent = self.id_to_intent[predicted_id]
        
        # Extract entities
        entities = self._extract_entities(prompt.lower())
        
        # Get top 3 predictions for debugging
        top_k = torch.topk(probs[0], k=3)
        top_intents = [
            (self.id_to_intent[idx.item()], prob.item())
            for idx, prob in zip(top_k.indices, top_k.values)
        ]
        
        return intent, confidence, {
            'entities': entities,
            'top_predictions': top_intents,
            'method': 'transformer',
            'model': self.model_name
        }
    
    def _extract_entities(self, prompt: str) -> Dict:
        """Extract entities using keyword matching"""
        entities = {
            'college_mentioned': None,
            'location_mentioned': None,
            'course_mentioned': None,
            'affiliation_mentioned': None
        }
        
        # Find matches
        for college in self.college_keywords:
            if college in prompt:
                entities['college_mentioned'] = college
                break
        
        for location in self.location_keywords:
            if location in prompt:
                entities['location_mentioned'] = location
                break
        
        for course in self.course_keywords:
            if course in prompt:
                entities['course_mentioned'] = course
                break
        
        for affiliation in self.affiliation_keywords:
            if affiliation in prompt:
                entities['affiliation_mentioned'] = affiliation
                break
        
        return entities


# Convenience function
def create_intent_detector(model_path: Optional[str] = None) -> TransformerIntentDetector:
    """Factory function to create intent detector"""
    default_path = './models/intent_model'
    path = model_path or default_path
    
    if os.path.exists(path):
        return TransformerIntentDetector(model_path=path)
    else:
        print(f"⚠️  No model at {path}. Use detector.train() to fine-tune.")
        return TransformerIntentDetector()


# Training script
if __name__ == "__main__":
    import sys
    
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    
    print("🎯 Transformer Intent Detector Training")
    print("=" * 50)
    
    detector = TransformerIntentDetector()
    
    # Train on test_data.csv
    if os.path.exists('test_data.csv'):
        detector.train(
            csv_path='test_data.csv',
            output_dir='./models/intent_model',
            epochs=5,
            batch_size=8
        )
        
        # Test predictions
        print("\n🧪 Testing predictions:")
        test_prompts = [
            "Where is Sagarmatha Engineering College located?",
            "What is the fee for computer engineering?",
            "Hello, good morning!",
            "Recommend best engineering colleges in Kathmandu",
            "Does Pulchowk have hostel facilities?",
            "What is the cutoff for civil engineering?"
        ]
        
        for prompt in test_prompts:
            intent, conf, details = detector.detect_intent(prompt)
            print(f"'{prompt}'")
            print(f"  → {intent} ({conf:.3f})")
            print(f"  → Top 3: {details['top_predictions']}")
            print()
    else:
        print("❌ test_data.csv not found!")
