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
                    # Use id2label mapping for prediction decoding
                    self.label_mapping = mapping_data.get("id2label", {})
                logger.info(f"Loaded label mapping with {len(self.label_mapping)} labels: {list(self.label_mapping.values())}")
            else:
                logger.warning(f"Label mapping file not found at {mapping_path}")
                self.label_mapping = {}
            
        except Exception as e:
            logger.error(f"Failed to load RoBERTa model: {e}")
            raise
    
    async def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text
        
        Returns:
            Dictionary of extracted entities
        """
        try:
            logger.debug(f"Extracting entities from: '{text}'")
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.debug(f"Tokenized input shape: {inputs['input_ids'].shape}")
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(-1)
            
            # Decode predictions
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entities = self._decode_entities(tokens, predicted_token_class_ids[0])
            
            logger.info(f"Entity extraction result: {entities}")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {}
    
    def _decode_entities(self, tokens: List[str], predictions: torch.Tensor) -> Dict[str, Any]:
        """Decode token predictions to entities"""
        entities = {}
        current_entity = ""
        current_type = ""
        
        logger.debug(f"Decoding {len(tokens)} tokens with {len(self.label_mapping)} labels")
        logger.debug(f"Available labels: {list(self.label_mapping.values()) if self.label_mapping else 'No labels loaded'}")
        
        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            if token in ["<s>", "</s>", "<pad>"]:
                continue
                
            label = self.label_mapping.get(str(pred_id.item()), "O")
            logger.debug(f"Token '{token}' -> prediction_id: {pred_id.item()} -> label: '{label}'")
            
            if label.startswith("B-"):
                # Begin new entity - save previous entity first
                if current_entity and current_type:
                    entity_key = current_type.lower() + "_mentioned"
                    entities[entity_key] = current_entity.strip()
                    logger.debug(f"Completed entity: {entity_key} = '{current_entity.strip()}'")
                
                # Start new entity
                entity_type = label[2:]  # Remove "B-" prefix
                current_type = entity_type
                current_entity = token.replace("Ġ", " ").strip()
                logger.debug(f"Started new entity: {current_type} with '{current_entity}'")
                
            elif label.startswith("I-") and current_type:
                # Continue current entity - check if entity type matches
                entity_type = label[2:]  # Remove "I-" prefix
                if entity_type == current_type:
                    # Add token to current entity
                    token_text = token.replace("Ġ", " ")
                    current_entity += token_text
                    logger.debug(f"Continuing entity: {current_type} with '{token}' -> '{current_entity}'")
                else:
                    # Different entity type, save previous and start new
                    if current_entity and current_type:
                        entity_key = current_type.lower() + "_mentioned"
                        entities[entity_key] = current_entity.strip()
                        logger.debug(f"Completed entity (type mismatch): {entity_key} = '{current_entity.strip()}'")
                    
                    current_type = entity_type
                    current_entity = token.replace("Ġ", " ").strip()
                    logger.debug(f"Started new entity from I-: {current_type} with '{current_entity}'")
                
            else:
                # Outside entity or O label - save current entity if exists
                if current_entity and current_type:
                    entity_key = current_type.lower() + "_mentioned"
                    entities[entity_key] = current_entity.strip()
                    logger.debug(f"Finished entity: {entity_key} = '{current_entity.strip()}'")
                current_entity = ""
                current_type = ""
        
        # Add final entity if exists
        if current_entity and current_type:
            entity_key = current_type.lower() + "_mentioned"
            entities[entity_key] = current_entity.strip()
            logger.debug(f"Final entity: {entity_key} = '{current_entity.strip()}'")
        
        logger.info(f"Final extracted entities: {entities}")
        return entities