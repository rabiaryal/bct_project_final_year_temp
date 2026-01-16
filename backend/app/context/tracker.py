"""
Backwards Compatibility DialogueTracker
Simple wrapper around the new ContextManager for backwards compatibility
"""

from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

from .context_manager import context_manager

if TYPE_CHECKING:
    from app.policy.entity_roles import Entity


@dataclass
class Turn:
    """Represents a single turn in the dialogue"""
    turn_id: int
    user_input: str
    intent: Optional[str] = None
    entities: List[Dict[str, Any]] = None
    response: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class DialogueTracker:
    """
    Backwards compatibility wrapper around ContextManager
    Maintains the old interface while using the new context system
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: List[Turn] = []
        self.created_at = datetime.now()
        
    def add_turn(self, user_input: str, intent: Optional[str] = None, 
                entities: List[Dict[str, Any]] = None, response: Optional[str] = None) -> Turn:
        """Add a new turn to the dialogue"""
        turn_id = len(self.turns)
        turn = Turn(
            turn_id=turn_id,
            user_input=user_input,
            intent=intent,
            entities=entities or [],
            response=response
        )
        
        self.turns.append(turn)
        
        # Update context manager
        if intent:
            context_manager.update_intent(self.session_id, intent, 1.0)
        
        if entities:
            # Convert to Entity objects
            entity_objects = []
            for entity_dict in entities:
                try:
                    # Import Entity at runtime to avoid circular import
                    from app.policy.entity_roles import Entity
                    entity = Entity(
                        type=entity_dict.get("entity", entity_dict.get("type", "UNKNOWN")),
                        value=entity_dict.get("value", ""),
                        confidence=entity_dict.get("confidence", 1.0)
                    )
                    entity_objects.append(entity)
                except Exception as e:
                    print(f"Warning: Could not convert entity {entity_dict}: {e}")
            
            if entity_objects:
                context_manager.update_entities(self.session_id, entity_objects)
        
        return turn

    def add_user_message(self, message: str) -> None:
        """Add user message to tracker"""
        # Create new turn if needed or update current
        if not self.turns or (self.turns[-1].user_input and self.turns[-1].response):
             # New turn
             self.add_turn(user_input=message)
        else:
             # Update existing turn if we just created it without input? 
             # Simpler: always create new turn for user message
             self.add_turn(user_input=message)

    def update_intent(self, intent_name: str, confidence: float, metadata: Dict[str, Any] = None) -> None:
        """Update intent for the current turn"""
        if self.turns:
            self.turns[-1].intent = intent_name
            # No place to store confidence in simple Turn object currently, but that's fine for now
            # Sync with context manager
            context_manager.update_intent(self.session_id, intent_name, confidence)

    def update_entities(self, entities: Dict[str, Any]) -> None:
        """Update entities for the current turn"""
        if self.turns:
            # entities coming in as flattened dict {type: val} or {type: [vals]}
            # Convert to list of dicts for Turn object
            entity_list = []
            for etype, value in entities.items():
                if isinstance(value, list):
                    for v in value:
                        entity_list.append({"type": etype, "value": v})
                else:
                    entity_list.append({"type": etype, "value": value})
            
            self.turns[-1].entities = entity_list

    @property
    def slots(self) -> Dict[str, Any]:
        """Get current slots (compatibility property)"""
        context = context_manager.get_context(self.session_id)
        return {
            "filled": context.slots.filled,
            "missing": context.slots.missing
        }

    @property
    def messages(self) -> List[Any]:
        """Get messages list (compatibility property)"""
        return self.turns

    def add_bot_message(self, message: str, action_name: str = None) -> None:
        """Add bot response to the current turn"""
        if self.turns:
            current_turn = self.turns[-1]
            current_turn.response = message
            
    def get_current_state(self) -> Dict[str, Any]:
        """Get current conversation state"""
        return self.get_context_summary()
    
    def get_current_turn(self) -> Optional[Turn]:
        """Get the current turn"""
        return self.turns[-1] if self.turns else None
    
    def get_turn_count(self) -> int:
        """Get total number of turns"""
        return len(self.turns)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get context summary from new context manager"""
        return context_manager.get_context_summary(self.session_id)
    
    def reset(self):
        """Reset the dialogue tracker"""
        self.turns = []
        context_manager.reset_conversation(self.session_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "user_input": turn.user_input,
                    "intent": turn.intent,
                    "entities": turn.entities,
                    "response": turn.response,
                    "timestamp": turn.timestamp
                }
                for turn in self.turns
            ],
            "context_summary": self.get_context_summary()
        }