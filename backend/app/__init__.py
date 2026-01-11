"""Backend App Initialization"""

from app.nlu import BERTIntentClassifier, RoBERTaEntityExtractor
from app.context import DialogueTracker
from app.policy import PolicyPlanner
from app.actions import ActionRegistry
from app.response import ResponseFormatter
from app.repositories import MongoRepository
from app.services import CollegeService
from app.dialogue_manager import dialogue_manager
from app.utils import setup_logger, get_logger, config
from app import schemas

__version__ = "1.0.0"

__all__ = [
    "BERTIntentClassifier", "RoBERTaEntityExtractor",
    "DialogueTracker", "PolicyPlanner", "ActionRegistry", "ResponseFormatter",
    "MongoRepository", "CollegeService", "dialogue_manager",
    "setup_logger", "get_logger", "config", "schemas"
]