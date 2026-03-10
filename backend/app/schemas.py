"""Pydantic Schemas for API"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Where is Sagarmatha Engineering college located",
                "session_id": "session_123",
            }
        }


class ChatResponse(BaseModel):
    """Chat response schema"""
    message: str = Field(..., description="System response")
    session_id: str = Field(..., description="Session ID")
    intent: str = Field(..., description="Detected intent")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent confidence")
    timestamp: datetime = Field(default_factory=datetime.now)
    debug_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Debug information")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Kathmandu University is located in Dhulikhel...",
                "session_id": "session_123",
                "intent": "GET_COLLEGE_INFO",
                "entities": {"college_name": "Kathmandu University"},
                "confidence": 0.95,
                "timestamp": "2024-01-15T10:30:00",
                "debug_info": {}
            }
        }


class HealthStatus(BaseModel):
    """Health check status"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class DetailedHealthStatus(BaseModel):
    """Detailed health check with component status"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, Dict[str, Any]]
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now)