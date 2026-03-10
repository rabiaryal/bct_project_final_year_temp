"""Application Configuration"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env.mongodb early so env vars are available for dataclass defaults
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env.mongodb")
load_dotenv(os.path.abspath(_ENV_PATH))

@dataclass
class DatabaseConfig:
    """Database configuration"""
    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    database_name: str = os.getenv("MONGODB_DB", "crs")
    collection_name: str = os.getenv("MONGODB_COLLECTION", "college data")
    connection_timeout: int = 30000
    server_selection_timeout: int = 30000

# Project root: backend/../ → project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

@dataclass
class ModelConfig:
    """Model configuration"""
    intent_model_path: str = os.path.join(_PROJECT_ROOT, "models", "bert_intent_model")
    entity_model_path: str = os.path.join(_PROJECT_ROOT, "models", "crf_entity_model")
    confidence_threshold: float = 0.5

@dataclass
class APIConfig:
    """API configuration"""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = os.getenv("RELOAD", "false").lower() == "true"
    workers: int = int(os.getenv("WORKERS", "1"))
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"])

@dataclass
class DialogueConfig:
    """Dialogue system configuration"""
    max_turns: int = 20
    session_timeout: int = 1800  # 30 minutes
    fallback_threshold: float = 0.3

class AppConfig:
    """Application configuration manager"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.models = ModelConfig()
        self.api = APIConfig()
        self.dialogue = DialogueConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "database": self.database.__dict__,
            "models": self.models.__dict__,
            "api": self.api.__dict__,
            "dialogue": self.dialogue.__dict__
        }

# Global config instance
config = AppConfig()