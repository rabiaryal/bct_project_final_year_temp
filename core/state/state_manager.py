"""
Conversation State Manager
Manages context memory for multi-turn conversations, enabling slot filling and context-aware responses.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import copy


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    user_message: str
    bot_response: str
    intent: str
    sub_intent: Optional[str]
    entities: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SlotState:
    """
    Slot filling state for tracking user preferences
    
    Slots help track what information the user has provided vs what's needed
    """
    # College-level slots
    college_name: Optional[str] = None
    college_type: Optional[str] = None  # public/private/constituent
    location: Optional[str] = None
    
    # Course-level slots
    course_name: Optional[str] = None
    department: Optional[str] = None
    
    # Preference slots
    max_fee: Optional[int] = None
    min_rating: Optional[float] = None
    hostel_required: Optional[bool] = None
    scholarship_required: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding None values)"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def update_from_entities(self, entities: Dict[str, Any]):
        """Update slots from extracted entities"""
        mapping = {
            'college_name': 'college_name',
            'college_type': 'college_type',
            'location': 'location',
            'course_name': 'course_name',
            'department_name': 'department',
            'fee': 'max_fee',
            'rating': 'min_rating',
            'hostel_availability': 'hostel_required',
            'scholarship': 'scholarship_required'
        }
        
        for entity_key, slot_key in mapping.items():
            if entity_key in entities and entities[entity_key]:
                value = entities[entity_key]
                
                # Type conversions
                if slot_key == 'max_fee' and isinstance(value, str):
                    # Try to parse fee
                    import re
                    numbers = re.findall(r'\d+', value.replace(',', ''))
                    if numbers:
                        value = int(numbers[0])
                
                if slot_key in ['hostel_required', 'scholarship_required']:
                    value = True  # If mentioned, assume required
                
                setattr(self, slot_key, value)
    
    def clear(self):
        """Clear all slots"""
        for key in self.__dict__:
            setattr(self, key, None)


class ConversationStateManager:
    """
    Manages conversation state including:
    - Context memory (recent turns)
    - Slot filling (accumulated preferences)
    - Topic tracking (current conversation topic)
    
    This enables:
    1. Multi-turn conversations ("What about the fees?" refers to last mentioned college)
    2. Slot filling ("I want computer engineering" + "in Kathmandu" combines both)
    3. Context-aware responses
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation state manager
        
        Args:
            max_history: Maximum number of turns to keep in memory
        """
        self.max_history = max_history
        
        # Per-session state storage
        self._sessions: Dict[str, Dict] = defaultdict(self._create_session)
    
    def _create_session(self) -> Dict:
        """Create a new session state"""
        return {
            'history': [],
            'slots': SlotState(),
            'current_topic': None,
            'last_colleges_mentioned': [],
            'last_intent': None,
            'created_at': datetime.now(),
            'last_active': datetime.now()
        }
    
    def get_session(self, session_id: str) -> Dict:
        """Get or create session state"""
        session = self._sessions[session_id]
        session['last_active'] = datetime.now()
        return session
    
    def add_turn(self, session_id: str, user_message: str, bot_response: str,
                 intent: str, sub_intent: Optional[str], entities: Dict[str, Any]):
        """
        Add a conversation turn and update state
        
        Args:
            session_id: Unique session identifier
            user_message: User's input
            bot_response: Bot's response
            intent: Detected primary intent
            sub_intent: Detected sub-intent (if any)
            entities: Extracted entities
        """
        session = self.get_session(session_id)
        
        # Create turn
        turn = ConversationTurn(
            user_message=user_message,
            bot_response=bot_response,
            intent=intent,
            sub_intent=sub_intent,
            entities=entities
        )
        
        # Add to history (keep last N turns)
        session['history'].append(turn)
        if len(session['history']) > self.max_history:
            session['history'].pop(0)
        
        # Update slots from entities
        session['slots'].update_from_entities(entities)
        
        # Update current topic
        session['last_intent'] = intent
        
        # Track mentioned colleges
        if 'college_name' in entities and entities['college_name']:
            college_name = entities['college_name']
            if college_name not in session['last_colleges_mentioned']:
                session['last_colleges_mentioned'].insert(0, college_name)
                # Keep only last 5 colleges
                session['last_colleges_mentioned'] = session['last_colleges_mentioned'][:5]
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get current conversation context for processing
        
        Returns dict with:
            - slots: Current slot state
            - last_colleges: Recently mentioned colleges
            - last_intent: Last detected intent
            - history_summary: Brief summary of recent turns
        """
        session = self.get_session(session_id)
        
        return {
            'slots': session['slots'].to_dict(),
            'last_colleges': session['last_colleges_mentioned'],
            'last_intent': session['last_intent'],
            'turn_count': len(session['history']),
            'history_summary': self._summarize_history(session['history'][-3:])  # Last 3 turns
        }
    
    def _summarize_history(self, turns: List[ConversationTurn]) -> List[Dict]:
        """Create a brief summary of conversation turns"""
        return [
            {
                'user': turn.user_message[:100],  # Truncate long messages
                'intent': turn.intent,
                'entities': turn.entities
            }
            for turn in turns
        ]
    
    def resolve_references(self, session_id: str, prompt: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve pronoun/reference mentions using context
        
        Examples:
        - "What about the fees?" -> uses last mentioned college
        - "Is there hostel?" -> uses last mentioned college
        - "Compare it with NCIT" -> "it" = last mentioned college
        
        Returns:
            Updated entities dict with resolved references
        """
        session = self.get_session(session_id)
        resolved = copy.deepcopy(entities)
        prompt_lower = prompt.lower()
        
        # Check for pronoun references that need resolution
        pronouns = ['it', 'that', 'this', 'there', 'the college', 'that college', 'this college']
        
        needs_college_reference = any(p in prompt_lower for p in pronouns)
        
        # Also check if question lacks a specific college but context has one
        question_words = ['what', 'how', 'where', 'is', 'does', 'are', 'can']
        is_question = any(prompt_lower.startswith(w) for w in question_words)
        
        # If no college in current entities but one was mentioned before
        if not resolved.get('college_name') and session['last_colleges_mentioned']:
            if needs_college_reference or (is_question and len(prompt_lower.split()) < 8):
                # Use most recently mentioned college
                resolved['college_name'] = session['last_colleges_mentioned'][0]
                resolved['_resolved_from_context'] = True
        
        # Combine with slot values for filtering
        slots = session['slots']
        
        # Fill in missing preferences from slots
        if not resolved.get('location') and slots.location:
            resolved['location'] = slots.location
        
        if not resolved.get('course_name') and slots.course_name:
            resolved['course_name'] = slots.course_name
        
        if not resolved.get('college_type') and slots.college_type:
            resolved['college_type'] = slots.college_type
        
        return resolved
    
    def clear_session(self, session_id: str):
        """Clear a session's state"""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def clear_slots(self, session_id: str):
        """Clear only the slots, keep history"""
        session = self.get_session(session_id)
        session['slots'].clear()
        session['last_colleges_mentioned'] = []
    
    def get_slots(self, session_id: str) -> Dict[str, Any]:
        """Get current slot values"""
        session = self.get_session(session_id)
        return session['slots'].to_dict()
    
    def set_slot(self, session_id: str, slot_name: str, value: Any):
        """Manually set a slot value"""
        session = self.get_session(session_id)
        if hasattr(session['slots'], slot_name):
            setattr(session['slots'], slot_name, value)
    
    def get_conversation_summary(self, session_id: str) -> str:
        """
        Get a human-readable summary of the conversation
        Useful for debugging or showing to user
        """
        session = self.get_session(session_id)
        
        summary_parts = []
        
        if session['last_colleges_mentioned']:
            summary_parts.append(f"Colleges discussed: {', '.join(session['last_colleges_mentioned'])}")
        
        slots = session['slots'].to_dict()
        if slots:
            slot_str = ', '.join(f"{k}={v}" for k, v in slots.items())
            summary_parts.append(f"Current preferences: {slot_str}")
        
        summary_parts.append(f"Conversation turns: {len(session['history'])}")
        
        return " | ".join(summary_parts) if summary_parts else "New conversation"


# Singleton instance for easy import
conversation_state_manager = ConversationStateManager()
