"""
Intent Detection Module
Detects user intent and extracts entities from queries
"""

from .safe_intent_detector import SafeIntentDetector

# Try to import transformer detector
try:
    from .transformer_intent_detector import TransformerIntentDetector, create_intent_detector
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TransformerIntentDetector = None
    create_intent_detector = None

# Try to import entity extractor
try:
    from .entity_extractor import TransformerEntityExtractor, create_entity_extractor
    ENTITY_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENTITY_EXTRACTOR_AVAILABLE = False
    TransformerEntityExtractor = None
    create_entity_extractor = None

__all__ = [
    'SafeIntentDetector', 
    'TransformerIntentDetector', 
    'create_intent_detector',
    'TransformerEntityExtractor',
    'create_entity_extractor',
    'TRANSFORMER_AVAILABLE',
    'ENTITY_EXTRACTOR_AVAILABLE'
]
