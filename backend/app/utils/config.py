"""Application Configuration"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class DatabaseConfig:
    """Database configuration"""
    mongodb_uri: str = os.getenv(
        "MONGODB_URI", 
        "mongodb+srv://078bct023rabichandra_db_user:R0cEQ9VLeAYeVoBA@cluster0.bnyhztw.mongodb.net/?appName=Cluster0&tlsAllowInvalidCertificates=true"
    )
    database_name: str = os.getenv("MONGODB_DB", "crs")
    collection_name: str = os.getenv("MONGODB_COLLECTION", "college data")
    connection_timeout: int = 30000
    server_selection_timeout: int = 30000

@dataclass
class ModelConfig:
    """Model configuration"""
    intent_model_path: str = "/Applications/development/ml learning/bct_final_year_project/models/bert_intent_model"
    entity_model_path: str = "/Applications/development/ml learning/bct_final_year_project/models/roberta_entity_model"
    faiss_index_path: str = "/Applications/development/ml learning/bct_final_year_project/models/faiss_index"
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