"""College Recommendation System - Backend Package"""

from app.nlu import BERTIntentClassifier, RoBERTaEntityExtractor
from app.repositories import MongoRepository
from app.dialogue_manager import dialogue_manager, DialogueManager
from app.context import SlotManager, DialogueContext
from app.templates import INTENT_TEMPLATES, get_template
from app.handlers.router import route_intent
from app.utils.formatter import format_response
from app.utils import setup_logger, get_logger, config
from app import schemas

__version__ = "2.0.0"

__all__ = [
    # NLU
    "BERTIntentClassifier",
    "RoBERTaEntityExtractor",
    # Repository
    "MongoRepository",
    # Context
    "SlotManager",
    "DialogueContext",
    # Templates
    "INTENT_TEMPLATES",
    "get_template",
    # Handlers
    "route_intent",
    "format_response",
    # Dialogue Manager
    "dialogue_manager",
    "DialogueManager",
    # Utils
    "setup_logger",
    "get_logger",
    "config",
    "schemas",
]