from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class ContextStatus(Enum):
    """Context status enumeration"""
    INIT = "INIT"
    READY = "READY"
    PROCESSING = "PROCESSING"
    WAITING = "WAITING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class EntitySource(Enum):
    """Source of entity information"""
    USER = "user"
    SYSTEM = "system"
    ENRICHED = "enriched"


@dataclass
class IntentInfo:
    """Intent information with confidence"""
    name: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EntityInfo:
    """Entity information with metadata"""
    value: str
    confidence: float
    source: EntitySource
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotInfo:
    """Slot management information"""
    required: List[str] = field(default_factory=list)
    filled: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)


@dataclass
class PolicyState:
    """Policy decision state"""
    action: Optional[str] = None
    status: ContextStatus = ContextStatus.INIT
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextFlags:
    """Context management flags"""
    awaiting_clarification: bool = False
    fallback_used: bool = False
    reset_context: bool = False
    session_expired: bool = False
    error_occurred: bool = False
    multi_turn_active: bool = False


@dataclass
class ConversationContext:
    """Main conversation context structure"""
    conversation_id: str
    turn_id: int = 0
    
    # Intent tracking
    intent: Dict[str, Any] = field(default_factory=lambda: {
        "current": None,
        "previous": []
    })
    
    # Entity management by role
    entities: Dict[str, Dict[str, EntityInfo]] = field(default_factory=lambda: {
        "IDENTIFIER": {},
        "FILTER": {},
        "CONSTRAINT": {},
        "ATTRIBUTE": {},
        "RELATION": {},
        "SIGNAL": {}
    })
    
    # Slot management
    slots: SlotInfo = field(default_factory=SlotInfo)
    
    # Policy and action state
    policy_state: PolicyState = field(default_factory=PolicyState)
    
    # Database interaction tracking
    last_db_query: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    
    # Context flags
    flags: ContextFlags = field(default_factory=ContextFlags)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @classmethod
    def new_conversation(cls, conversation_id: Optional[str] = None) -> 'ConversationContext':
        """Create a new conversation context"""
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
        return cls(conversation_id=conversation_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "intent": self.intent,
            "entities": {role: {k: {
                "value": v.value,
                "confidence": v.confidence,
                "source": v.source.value,
                "timestamp": v.timestamp,
                "metadata": v.metadata
            } for k, v in entities.items()} for role, entities in self.entities.items()},
            "slots": {
                "required": self.slots.required,
                "filled": self.slots.filled,
                "missing": self.slots.missing,
                "optional": self.slots.optional
            },
            "policy_state": {
                "action": self.policy_state.action,
                "status": self.policy_state.status.value,
                "confidence": self.policy_state.confidence,
                "metadata": self.policy_state.metadata
            },
            "last_db_query": self.last_db_query,
            "results": self.results,
            "flags": {
                "awaiting_clarification": self.flags.awaiting_clarification,
                "fallback_used": self.flags.fallback_used,
                "reset_context": self.flags.reset_context,
                "session_expired": self.flags.session_expired,
                "error_occurred": self.flags.error_occurred,
                "multi_turn_active": self.flags.multi_turn_active
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


@dataclass
class ContextUpdateRequest:
    """Request structure for context updates"""
    intent: Optional[IntentInfo] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    policy_action: Optional[str] = None
    db_query: Optional[Dict[str, Any]] = None
    db_results: Optional[Dict[str, Any]] = None
    flags_update: Optional[Dict[str, bool]] = None
    force_reset: bool = False