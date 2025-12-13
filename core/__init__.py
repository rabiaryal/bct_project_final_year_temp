"""
Core Module: Contains all processing components for the chatbot
- intent: Intent detection and entity extraction
- recommendation: College recommendation using XGBoost and neural encoders
- qa: Question answering using structured retrieval
"""

from core.intent.safe_intent_detector import SafeIntentDetector

__all__ = [
    'SafeIntentDetector',
]
