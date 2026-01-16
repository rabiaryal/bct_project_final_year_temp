from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime, timedelta
import logging

from .context_models import (
    ConversationContext, IntentInfo, EntityInfo, EntitySource,
    ContextStatus, ContextFlags, ContextUpdateRequest
)

if TYPE_CHECKING:
    from app.policy.entity_roles import EntityRole, Entity

logger = logging.getLogger(__name__)


class ContextRules:
    """Business rules for context management"""
    
    # Configuration
    MAX_PREVIOUS_INTENTS = 5
    MIN_ENTITY_CONFIDENCE = 0.3
    SESSION_TIMEOUT_MINUTES = 30
    MAX_TURNS_PER_SESSION = 50
    
    @staticmethod
    def should_reset_context(context: ConversationContext, new_intent: Optional[str] = None) -> bool:
        """Determine if context should be reset"""
        # Explicit reset triggers
        if context.flags.reset_context:
            return True
            
        # Reset intents
        reset_intents = ['goodbye', 'reset', 'start_over', 'new_conversation']
        if new_intent and new_intent.lower() in reset_intents:
            return True
            
        # Session timeout
        if context.updated_at:
            last_update = datetime.fromisoformat(context.updated_at.replace('Z', '+00:00'))
            if datetime.now(last_update.tzinfo) - last_update > timedelta(minutes=ContextRules.SESSION_TIMEOUT_MINUTES):
                logger.info(f"Session {context.conversation_id} timed out")
                return True
        
        # Too many turns
        if context.turn_id > ContextRules.MAX_TURNS_PER_SESSION:
            logger.info(f"Session {context.conversation_id} exceeded max turns")
            return True
            
        return False
    
    @staticmethod
    def should_update_intent(context: ConversationContext, new_intent: IntentInfo) -> bool:
        """Determine if intent should be updated"""
        if not context.intent.get("current"):
            return True
            
        current_intent = context.intent["current"]
        
        # Update if different intent
        if current_intent["name"] != new_intent.name:
            return True
            
        # Update if significantly higher confidence
        if new_intent.confidence > current_intent["confidence"] + 0.2:
            return True
            
        return False
    
    @staticmethod
    def should_update_entity(context: ConversationContext, entity_type: str, 
                           entity_role: str, new_entity: EntityInfo) -> bool:
        """Determine if entity should be updated"""
        # Minimum confidence threshold
        if new_entity.confidence < ContextRules.MIN_ENTITY_CONFIDENCE:
            return False
            
        existing_entity = context.entities.get(entity_role, {}).get(entity_type)
        
        if not existing_entity:
            return True
            
        # Update if higher confidence
        if new_entity.confidence > existing_entity.confidence:
            return True
            
        # Update if same value but from user (higher priority than system)
        if (new_entity.value == existing_entity.value and 
            new_entity.source == EntitySource.USER and 
            existing_entity.source == EntitySource.SYSTEM):
            return True
            
        return False
    
    @staticmethod
    def should_await_clarification(context: ConversationContext) -> bool:
        """Determine if system should ask for clarification"""
        # Check if required slots are missing
        if context.slots.missing:
            return True
            
        # Check if intent confidence is low
        current_intent = context.intent.get("current")
        if current_intent and current_intent.get("confidence", 0) < 0.7:
            return True
            
        # Check if no clear action can be determined
        if context.policy_state.status == ContextStatus.WAITING:
            return True
            
        return False
    
    @staticmethod
    def get_required_slots(intent_name: str) -> List[str]:
        """Get required slots for given intent"""
        slot_requirements = {
            'search_college_by_name': ['COLLEGE_NAME'],
            'search_college_by_location': ['LOCATION'],
            'search_college_by_program': ['PROGRAM'],
            'get_college_info': ['COLLEGE_NAME'],
            'compare_colleges': ['COLLEGE_NAME'],  # Need at least one
            'get_admission_requirements': ['COLLEGE_NAME'],
            'find_colleges_by_criteria': [],  # Flexible
            'get_fee_information': ['COLLEGE_NAME'],
            'ask_about_facilities': ['COLLEGE_NAME'],
            'greeting': [],
            'goodbye': [],
            'fallback': []
        }
        
        return slot_requirements.get(intent_name, [])
    
    @staticmethod
    def get_optional_slots(intent_name: str) -> List[str]:
        """Get optional slots for given intent"""
        optional_requirements = {
            'search_college_by_name': ['LOCATION', 'PROGRAM'],
            'search_college_by_location': ['PROGRAM', 'COLLEGE_TYPE'],
            'search_college_by_program': ['LOCATION', 'COLLEGE_TYPE'],
            'find_colleges_by_criteria': ['LOCATION', 'PROGRAM', 'FEE', 'FACILITY'],
            'compare_colleges': ['ATTRIBUTE'],
            'get_college_info': ['ATTRIBUTE'],
        }
        
        return optional_requirements.get(intent_name, [])
    
    @staticmethod
    def validate_context_consistency(context: ConversationContext) -> List[str]:
        """Validate context for consistency and return any issues"""
        issues = []
        
        # Check turn sequence
        if context.turn_id < 0:
            issues.append("Turn ID cannot be negative")
            
        # Check intent consistency
        current_intent = context.intent.get("current")
        if current_intent:
            if current_intent.get("confidence", 0) < 0 or current_intent.get("confidence", 0) > 1:
                issues.append("Intent confidence must be between 0 and 1")
                
        # Check entity confidence values
        for role, entities in context.entities.items():
            for entity_type, entity_info in entities.items():
                if entity_info.confidence < 0 or entity_info.confidence > 1:
                    issues.append(f"Entity {role}:{entity_type} confidence out of range")
                    
        # Check slots consistency
        all_slots = set(context.slots.required + context.slots.filled + context.slots.missing)
        if len(all_slots) != len(context.slots.required + context.slots.filled + context.slots.missing):
            issues.append("Overlapping slots in required/filled/missing")
            
        return issues


class ContextUpdateRules:
    """Rules for updating context components"""
    
    @staticmethod
    def update_intent(context: ConversationContext, new_intent: IntentInfo) -> ConversationContext:
        """Update intent following business rules"""
        # Move current intent to previous if exists
        if context.intent.get("current"):
            context.intent["previous"].append(context.intent["current"])
            
            # Keep only last N intents
            if len(context.intent["previous"]) > ContextRules.MAX_PREVIOUS_INTENTS:
                context.intent["previous"] = context.intent["previous"][-ContextRules.MAX_PREVIOUS_INTENTS:]
        
        # Set new current intent
        context.intent["current"] = {
            "name": new_intent.name,
            "confidence": new_intent.confidence,
            "timestamp": new_intent.timestamp
        }
        
        # Update turn counter
        context.turn_id += 1
        
        # Update required slots based on new intent
        required_slots = ContextRules.get_required_slots(new_intent.name)
        optional_slots = ContextRules.get_optional_slots(new_intent.name)
        
        context.slots.required = required_slots
        context.slots.optional = optional_slots
        
        return context
    
    @staticmethod
    def update_entities(context: ConversationContext, entities: List["Entity"]) -> ConversationContext:
        """Update entities following merge/replace rules"""
        for entity in entities:
            entity_info = EntityInfo(
                value=entity.value,
                confidence=entity.confidence,
                source=EntitySource.USER,
                metadata={"type": entity.type}
            )
            
            role_name = entity.role.name
            entity_type = entity.type
            
            # Apply update rules
            if ContextRules.should_update_entity(context, entity_type, role_name, entity_info):
                if role_name not in context.entities:
                    context.entities[role_name] = {}
                    
                context.entities[role_name][entity_type] = entity_info
                logger.debug(f"Updated entity {role_name}:{entity_type} = {entity.value}")
        
        # Update slot status
        ContextUpdateRules._update_slot_status(context)
        
        return context
    
    @staticmethod
    def _update_slot_status(context: ConversationContext) -> None:
        """Update filled/missing slot status based on current entities"""
        filled_slots = []
        
        # Check all entity types against required slots
        for role, entities in context.entities.items():
            for entity_type in entities.keys():
                if entity_type in context.slots.required and entity_type not in filled_slots:
                    filled_slots.append(entity_type)
        
        context.slots.filled = filled_slots
        context.slots.missing = [slot for slot in context.slots.required if slot not in filled_slots]
    
    @staticmethod
    def enrich_with_db_results(context: ConversationContext, 
                             db_query: Dict[str, Any], 
                             db_results: Dict[str, Any]) -> ConversationContext:
        """Enrich context with database results"""
        context.last_db_query = db_query
        context.results = db_results
        
        # Update flags
        context.flags.awaiting_clarification = False
        
        # Extract any system entities from results
        if db_results and "colleges" in db_results:
            colleges = db_results["colleges"]
            if len(colleges) == 1:
                # Single result - can extract attributes
                college = colleges[0]
                college_name = college.get("name")
                if college_name:
                    # Add college name as system entity if not already present
                    if "COLLEGE_NAME" not in context.entities.get("IDENTIFIER", {}):
                        context.entities["IDENTIFIER"]["COLLEGE_NAME"] = EntityInfo(
                            value=college_name,
                            confidence=1.0,
                            source=EntitySource.SYSTEM,
                            metadata={"from_db_result": True}
                        )
        
        return context
    
    @staticmethod
    def reset_context(context: ConversationContext) -> ConversationContext:
        """Reset context while preserving conversation ID"""
        conversation_id = context.conversation_id
        context = ConversationContext.new_conversation(conversation_id)
        logger.info(f"Context reset for conversation {conversation_id}")
        return context