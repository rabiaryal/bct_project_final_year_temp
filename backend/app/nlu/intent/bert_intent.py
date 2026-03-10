"""BERT Intent Classification"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
from typing import Tuple, Dict, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

class BERTIntentClassifier:
    """BERT-based intent classification"""
    
    def __init__(self, model_path: str = None):
        """Initialize BERT intent classifier"""
        from app.utils.config import config
        self.model_path = model_path or config.models.intent_model_path
        self.model = None
        self.tokenizer = None
        self.label_mapping = {}
        self.confidence_threshold = 0.5
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load trained BERT model"""
        try:
            logger.info(f"Loading BERT model from {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping  (file has {"label_to_id": {...}, "id_to_label": {...}})
            label_path = os.path.join(self.model_path, "label_mapping.json")
            with open(label_path, 'r') as f:
                raw = json.load(f)
            # Use id_to_label sub-dict so str(predicted_class_id) → intent name
            self.label_mapping = raw.get("id_to_label", raw)

            logger.info(f"Model loaded successfully with {len(self.label_mapping)} intents")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    async def predict(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict intent from text
        
        Returns:
            (intent, confidence, metadata)
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_id = probabilities.argmax().item()
                confidence = probabilities[0][predicted_class_id].item()
            
            # Map to intent label
            intent = self.label_mapping.get(str(predicted_class_id), "unknown")
            
            # Get top 3 predictions for metadata
            top_probs, top_indices = torch.topk(probabilities[0], 3)
            top_predictions = [
                (self.label_mapping.get(str(idx.item()), "unknown"), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
            
            metadata = {
                "confidence_threshold": self.confidence_threshold,
                "top_predictions": top_predictions,
                "model": "bert-base-uncased",
                "device": str(self.device)
            }
            
            return intent, confidence, metadata
            
        except Exception as e:
            logger.error(f"Intent prediction error: {e}")
            return "unknown", 0.0, {"error": str(e)}