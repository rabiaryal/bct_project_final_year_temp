"""Dialogue State Tracking"""

from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from app.utils.logger import get_logger
from app.schemas import DialogueState

logger = get_logger(__name__)

@dataclass
class Turn:
    """Represents a dialogue turn"""
    user_input: str
    intent: str
    entities: Dict[str, Any]
    action: str
    response: str
    timestamp: datetime = field(default_factory=datetime.now)

class DialogueTracker:
    """
    Tracks dialogue state across conversation turns.
    Follows Rasa's tracker pattern for state management.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.slots = {}
        self.intent = None
        self.entities = {}
        self.turns = []
        self.messages = []  # Track message history
        self.last_action = None
        self.context = {}
        self.created_at = datetime.now()
        
    def update_from_nlu(self, intent: str, entities: Dict[str, Any], confidence: float):
        """Update tracker state from NLU results"""
        self.intent = intent
        self.entities = entities
        
        # Map entities to slots with enhanced mapping
        for entity_type, value in entities.items():
            if entity_type == "college_mentioned":
                self.slots["college_name"] = value
            elif entity_type == "course_mentioned":
                self.slots["course_name"] = value
            elif entity_type == "location_mentioned":
                self.slots["location"] = value
            elif entity_type == "fee_mentioned":
                self.slots["fee_type"] = value
            elif entity_type == "facility_mentioned":
                self.slots["facility"] = value
            else:
                # Direct mapping for other entities
                slot_name = entity_type.replace("_mentioned", "_name") if "_mentioned" in entity_type else entity_type
                self.slots[slot_name] = value
        
        # Store confidence in context
        self.context["last_intent_confidence"] = confidence
        
        logger.debug(f"Updated tracker - Intent: {intent}, Slots: {self.slots}")
    
    def add_turn(self, user_input: str, action: str, response: str):
        """Add a completed dialogue turn"""
        turn = Turn(
            user_input=user_input,
            intent=self.intent,
            entities=self.entities.copy(),
            action=action,
            response=response
        )
        self.turns.append(turn)
        self.last_action = action
        
        logger.debug(f"Added turn {len(self.turns)}: {action}")
    
    def get_slot(self, slot_name: str) -> Any:
        """Get slot value"""
        return self.slots.get(slot_name)
    
    def set_slot(self, slot_name: str, value: Any):
        """Set slot value"""
        self.slots[slot_name] = value
        logger.debug(f"Set slot {slot_name}={value}")
    
    def get_latest_message(self) -> Optional[Dict[str, Any]]:
        """Get latest user message with NLU results"""
        if not self.turns:
            return None
        
        latest_turn = self.turns[-1]
        return {
            "text": latest_turn.user_input,
            "intent": latest_turn.intent,
            "entities": latest_turn.entities
        }
    
    def get_turn_count(self) -> int:
        """Get number of turns in conversation"""
        return len(self.turns)
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current dialogue state"""
        return self.to_dict()
    
    def update_intent(self, intent: str, confidence: float, metadata: Dict[str, Any] = None):
        """Update intent with confidence and metadata"""
        self.intent = intent
        self.context["last_intent_confidence"] = confidence
        if metadata:
            self.context["intent_metadata"] = metadata
        logger.debug(f"Updated intent: {intent} (confidence: {confidence})")
    
    def update_entities(self, entities: Dict[str, Any]):
        """Update entities and map to slots"""
        self.entities = entities
        
        # Map entities to slots
        for entity_type, value in entities.items():
            if entity_type == "college_mentioned":
                self.slots["college_name"] = value
            elif entity_type == "course_mentioned":
                self.slots["course_name"] = value
            elif entity_type == "location_mentioned":
                self.slots["location"] = value
            elif entity_type == "fee_mentioned":
                self.slots["fee_type"] = value
            elif entity_type == "facility_mentioned":
                self.slots["facility"] = value
            else:
                # Direct mapping for other entities
                slot_name = entity_type.replace("_mentioned", "_name") if "_mentioned" in entity_type else entity_type
                self.slots[slot_name] = value
        
        logger.debug(f"Updated entities: {entities}, Slots: {self.slots}")
    
    def add_user_message(self, message: str):
        """Add user message to turn history"""
        self.context["last_user_message"] = message
        self.context["last_message_time"] = datetime.now().isoformat()
        logger.debug(f"Added user message: {message}")
    
    def add_bot_message(self, message: str, action: str):
        """Add bot response to turn history"""
        self.context["last_bot_message"] = message
        self.context["last_action"] = action
        self.last_action = action
        logger.debug(f"Added bot message: {message} (action: {action})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker to dictionary"""
        return {
            "session_id": self.session_id,
            "intent": self.intent,
            "entities": self.entities,
            "slots": self.slots,
            "turn_count": len(self.turns),
            "last_action": self.last_action,
            "context": self.context,
            "created_at": self.created_at.isoformat()
        }
    
    def to_dialogue_state(self) -> DialogueState:
        """Convert to Pydantic DialogueState"""
        return DialogueState(
            session_id=self.session_id,
            intent=self.intent or "unknown",
            entities=self.entities,
            slots=self.slots,
            turn_count=len(self.turns),
            last_action=self.last_action,
            context=self.context
        )