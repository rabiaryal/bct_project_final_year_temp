"""RoBERTa Named Entity Recognition"""

import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification
import json
import os
from typing import Dict, List, Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

class RoBERTaEntityExtractor:
    """RoBERTa-based entity extraction"""
    
    def __init__(self, model_path: str = None):
        """Initialize RoBERTa entity extractor"""
        from app.utils.config import config
        self.model_path = model_path or config.models.entity_model_path
        self.model = None
        self.tokenizer = None
        self.label_mapping = {}
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load trained RoBERTa model"""
        try:
            logger.info(f"Loading RoBERTa model from {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path)
            self.model = RobertaForTokenClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            mapping_path = os.path.join(self.model_path, "label_mappings.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.label_mapping = mapping_data.get("id2label", {})
                logger.info(f"Loaded label mapping with {len(self.label_mapping)} labels")
            else:
                logger.warning(f"Label mapping file not found at {mapping_path}")
                self.label_mapping = {}
            
        except Exception as e:
            logger.error(f"Failed to load RoBERTa model: {e}")
            raise
    
    def predict_with_confidence(self, text: str, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Predict entities with confidence scores using entity-level thresholding
        
        Args:
            text: Input text
            threshold: Confidence threshold (default: 0.3)
            
        Returns:
            List of top 3 entities with confidence scores
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=128,
                is_split_into_words=False
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get predictions with probabilities
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)[0]
            preds = torch.argmax(logits, dim=-1)[0]
            
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            labels = [self.label_mapping.get(str(p.item()), 'O') for p in preds]
            
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
            spans = self._post_process_entities_with_confidence(tokens, labels, scores)
            
            # 🔹 Apply threshold at ENTITY LEVEL
            threshold_filtered_entities = [
                e for e in spans
                if e["confidence"] >= threshold
            ]
            
            # Sort by confidence and return top 3
            threshold_filtered_entities.sort(key=lambda x: x['confidence'], reverse=True)
            return threshold_filtered_entities[:3]
            
        except Exception as e:
            logger.error(f"Confidence prediction error: {e}")
            return []
    
    def _post_process_entities_with_confidence(self, tokens, labels, scores):
        """
        Post-process entities with confidence scores (entity-level thresholding safe).
        
        Args:
            tokens: List of tokens from tokenizer
            labels: List of predicted BIO labels  
            scores: List of confidence scores per token
        
        Returns:
            List of entities with confidence scores calculated at entity level
        """
        if not tokens:
            return []

        # Step 1: Merge subword tokens based on RoBERTa's 'Ġ' prefix
        merged_tokens = []
        merged_labels = []
        merged_scores = []
        
        for token, label, score in zip(tokens, labels, scores):
            if token.startswith('Ġ'):
                # Start of a new word
                merged_tokens.append(token[1:])
                merged_labels.append(label)
                merged_scores.append(score)
            elif merged_tokens:
                # Continuation of the previous word - merge with previous
                merged_tokens[-1] += token
                # Keep the previous label (don't change it)
                # Average the scores for merged tokens
                merged_scores[-1] = (merged_scores[-1] + score) / 2
            else:
                # First token doesn't have a 'Ġ' prefix (unusual but handle it)
                merged_tokens.append(token)
                merged_labels.append(label)
                merged_scores.append(score)

        # Step 2: Extract entities from merged tokens using BIO logic
        entities = []
        current_entity_tokens = []
        current_entity_label = None
        current_entity_scores = []

        for token, label, score in zip(merged_tokens, merged_labels, merged_scores):
            if label.startswith('B-'):
                # If there's a current entity, save it before starting a new one
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    # Calculate entity confidence as average of token scores
                    entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
                    entities.append({
                        'text': entity_text, 
                        'type': current_entity_label,
                        'confidence': entity_confidence
                    })

                # Start a new entity
                current_entity_tokens = [token]
                current_entity_label = label[2:]  # Remove B- prefix
                current_entity_scores = [score]
                
            elif label.startswith('I-') and current_entity_label == label[2:]:
                # Continue the current entity
                current_entity_tokens.append(token)
                current_entity_scores.append(score)
                
            else:
                # Not an entity token or different entity - close current one
                if current_entity_tokens:
                    entity_text = " ".join(current_entity_tokens)
                    entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
                    entities.append({
                        'text': entity_text,
                        'type': current_entity_label,
                        'confidence': entity_confidence
                    })
                current_entity_tokens = []
                current_entity_label = None
                current_entity_scores = []

        # Add the last entity if it exists
        if current_entity_tokens:
            entity_text = " ".join(current_entity_tokens)
            entity_confidence = sum(current_entity_scores) / len(current_entity_scores)
            entities.append({
                'text': entity_text,
                'type': current_entity_label, 
                'confidence': entity_confidence
            })

        return entities