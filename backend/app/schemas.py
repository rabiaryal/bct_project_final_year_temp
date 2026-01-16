"""Pydantic Schemas for API"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Where is  Sagarmatha Engineering college located",
                "session_id": "session_123"
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
                "intent": "college_info",
                "entities": {"college_name": "Kathmandu University"},
                "confidence": 0.95,
                "timestamp": "2024-01-15T10:30:00",
                "debug_info": {"action": "action_college_info"}
            }
        }

class NLURequest(BaseModel):
    """NLU processing request"""
    text: str = Field(..., min_length=1, max_length=500)
    
class IntentResult(BaseModel):
    """Intent classification result"""
    intent: str = Field(..., description="Predicted intent")
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EntityResult(BaseModel):
    """Entity extraction result"""
    entities: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NLUResponse(BaseModel):
    """Complete NLU analysis result"""
    text: str = Field(..., description="Original text")
    intent: IntentResult
    entities: EntityResult
    
class DialogueState(BaseModel):
    """Dialogue state representation"""
    session_id: str
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Any] = Field(default_factory=dict)
    turn_count: int = 0
    last_action: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    """College retrieval result schema"""
    college_name: str = Field(..., description="Name of the college")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    college_data: Dict[str, Any] = Field(default_factory=dict, description="Full college information")
    match_reason: str = Field(..., description="Reason for this match")

class RetrievalData(BaseModel):
    """Complete retrieval results"""
    query: str = Field(..., description="Original query")
    results: List[RetrievalResult] = Field(default_factory=list)
    entities_found: Dict[str, Any] = Field(default_factory=dict)
    search_strategy: str = Field(..., description="Search strategy used")
    total_results: int = Field(default=0)

class ActionRequest(BaseModel):
    """Action execution request"""
    action: str
    slots: Dict[str, Any] = Field(default_factory=dict)
    session_id: str
    retrieval_data: Optional[RetrievalData] = None

class ActionResponse(BaseModel):
    """Action execution response"""
    response: str
    slots_updated: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieval_results: Optional[List[RetrievalResult]] = Field(default=None)

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