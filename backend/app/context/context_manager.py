from typing import Dict, List, Optional, Any, TYPE_CHECKING
from datetime import datetime
import logging
import json

from .context_models import (
    ConversationContext, IntentInfo, EntityInfo, EntitySource,
    ContextStatus, ContextFlags, ContextUpdateRequest
)
from .context_rules import ContextRules, ContextUpdateRules

if TYPE_CHECKING:
    from app.policy.entity_roles import Entity, EntityRole

logger = logging.getLogger(__name__)


class ContextManager:
    """Comprehensive context manager for multi-turn conversations"""
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """Initialize context manager with optional storage backend"""
        self._contexts: Dict[str, ConversationContext] = {}
        self._storage = storage_backend
        
    def get_context(self, conversation_id: str) -> ConversationContext:
        """Get or create context for conversation"""
        if conversation_id not in self._contexts:
            # Try to load from storage if available
            if self._storage:
                context_data = self._storage.load_context(conversation_id)
                if context_data:
                    self._contexts[conversation_id] = self._deserialize_context(context_data)
                else:
                    self._contexts[conversation_id] = ConversationContext.new_conversation(conversation_id)
            else:
                self._contexts[conversation_id] = ConversationContext.new_conversation(conversation_id)
                
        return self._contexts[conversation_id]
    
    def update_context(self, conversation_id: str, update_request: ContextUpdateRequest) -> ConversationContext:
        """Update context based on request"""
        context = self.get_context(conversation_id)
        
        # Check if context should be reset
        if update_request.force_reset or ContextRules.should_reset_context(context, 
                                                                           update_request.intent.name if update_request.intent else None):
            context = ContextUpdateRules.reset_context(context)
            self._contexts[conversation_id] = context
        
        # Update intent
        if update_request.intent and ContextRules.should_update_intent(context, update_request.intent):
            context = ContextUpdateRules.update_intent(context, update_request.intent)
            logger.info(f"Updated intent to {update_request.intent.name} for {conversation_id}")
        
        # Update entities
        if update_request.entities:
            entities = self._convert_entities(update_request.entities)
            context = ContextUpdateRules.update_entities(context, entities)
            logger.info(f"Updated {len(entities)} entities for {conversation_id}")
        
        # Update policy state
        if update_request.policy_action:
            context.policy_state.action = update_request.policy_action
            context.policy_state.status = ContextStatus.PROCESSING
        
        # Enrich with DB results
        if update_request.db_query and update_request.db_results:
            context = ContextUpdateRules.enrich_with_db_results(
                context, update_request.db_query, update_request.db_results
            )
            logger.info(f"Enriched context with DB results for {conversation_id}")
        
        # Update flags
        if update_request.flags_update:
            for flag, value in update_request.flags_update.items():
                setattr(context.flags, flag, value)
        
        # Check if clarification is needed
        context.flags.awaiting_clarification = ContextRules.should_await_clarification(context)
        
        # Update timestamp
        context.updated_at = datetime.utcnow().isoformat()
        
        # Validate context consistency
        issues = ContextRules.validate_context_consistency(context)
        if issues:
            logger.warning(f"Context consistency issues for {conversation_id}: {issues}")
        
        # Store if backend available
        if self._storage:
            self._storage.save_context(conversation_id, context.to_dict())
        
        return context
    
    def update_intent(self, conversation_id: str, intent_name: str, confidence: float) -> ConversationContext:
        """Convenience method to update intent"""
        intent_info = IntentInfo(name=intent_name, confidence=confidence)
        update_request = ContextUpdateRequest(intent=intent_info)
        return self.update_context(conversation_id, update_request)
    
    def update_entities(self, conversation_id: str, entities: List["Entity"]) -> ConversationContext:
        """Convenience method to update entities"""
        entity_dicts = [{
            "type": entity.type,
            "value": entity.value,
            "confidence": entity.confidence,
            "role": entity.role.name
        } for entity in entities]
        
        update_request = ContextUpdateRequest(entities=entity_dicts)
        return self.update_context(conversation_id, update_request)
    
    def add_db_results(self, conversation_id: str, db_query: Dict[str, Any], 
                      db_results: Dict[str, Any]) -> ConversationContext:
        """Convenience method to add database results"""
        update_request = ContextUpdateRequest(db_query=db_query, db_results=db_results)
        return self.update_context(conversation_id, update_request)
    
    def reset_conversation(self, conversation_id: str) -> ConversationContext:
        """Reset conversation context"""
        update_request = ContextUpdateRequest(force_reset=True)
        return self.update_context(conversation_id, update_request)
    
    def get_missing_slots(self, conversation_id: str) -> List[str]:
        """Get missing required slots for conversation"""
        context = self.get_context(conversation_id)
        return context.slots.missing
    
    def is_awaiting_clarification(self, conversation_id: str) -> bool:
        """Check if conversation is awaiting clarification"""
        context = self.get_context(conversation_id)
        return context.flags.awaiting_clarification
    
    def get_last_db_results(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get last database results for conversation"""
        context = self.get_context(conversation_id)
        return context.results
    
    def get_context_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of the current context state"""
        context = self.get_context(conversation_id)
        
        return {
            "conversation_id": context.conversation_id,
            "turn_id": context.turn_id,
            "current_intent": context.intent.get("current"),
            "filled_slots": context.slots.filled,
            "missing_slots": context.slots.missing,
            "policy_action": context.policy_state.action,
            "awaiting_clarification": context.flags.awaiting_clarification,
            "has_results": context.results is not None,
            "entity_count": sum(len(entities) for entities in context.entities.values()),
            "updated_at": context.updated_at
        }
    
    def cleanup_old_contexts(self, max_age_hours: int = 24) -> int:
        """Clean up contexts older than specified hours"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        contexts_to_remove = []
        
        for conv_id, context in self._contexts.items():
            try:
                context_time = datetime.fromisoformat(context.updated_at.replace('Z', '+00:00')).timestamp()
                if context_time < cutoff_time:
                    contexts_to_remove.append(conv_id)
            except Exception as e:
                logger.error(f"Error parsing timestamp for {conv_id}: {e}")
                contexts_to_remove.append(conv_id)
        
        for conv_id in contexts_to_remove:
            del self._contexts[conv_id]
            if self._storage:
                self._storage.delete_context(conv_id)
        
        logger.info(f"Cleaned up {len(contexts_to_remove)} old contexts")
        return len(contexts_to_remove)
    
    def _convert_entities(self, entity_dicts: List[Dict[str, Any]]) -> List["Entity"]:
        """Convert entity dictionaries to Entity objects"""
        from app.policy.entity_roles import Entity  # Runtime import to avoid circular dependency
        
        entities = []
        for entity_dict in entity_dicts:
            try:
                entity = Entity(
                    type=entity_dict["type"],
                    value=entity_dict["value"],
                    confidence=entity_dict.get("confidence", 1.0)
                )
                entities.append(entity)
            except Exception as e:
                logger.error(f"Error converting entity {entity_dict}: {e}")
        return entities
    
    def _deserialize_context(self, context_data: Dict[str, Any]) -> ConversationContext:
        """Deserialize context from storage format"""
        # This is a simplified deserialization - in production, you'd want more robust handling
        try:
            context = ConversationContext(
                conversation_id=context_data["conversation_id"],
                turn_id=context_data.get("turn_id", 0)
            )
            
            # Restore intent
            context.intent = context_data.get("intent", {"current": None, "previous": []})
            
            # Restore entities
            entities_data = context_data.get("entities", {})
            for role, entities in entities_data.items():
                context.entities[role] = {}
                for entity_type, entity_info in entities.items():
                    context.entities[role][entity_type] = EntityInfo(
                        value=entity_info["value"],
                        confidence=entity_info["confidence"],
                        source=EntitySource(entity_info.get("source", "user")),
                        timestamp=entity_info.get("timestamp", datetime.utcnow().isoformat()),
                        metadata=entity_info.get("metadata", {})
                    )
            
            # Restore other fields
            context.updated_at = context_data.get("updated_at", datetime.utcnow().isoformat())
            context.created_at = context_data.get("created_at", datetime.utcnow().isoformat())
            
            return context
        except Exception as e:
            logger.error(f"Error deserializing context: {e}")
            return ConversationContext.new_conversation(context_data.get("conversation_id", "error"))


class InMemoryStorage:
    """Simple in-memory storage backend for testing"""
    
    def __init__(self):
        self._storage = {}
    
    def save_context(self, conversation_id: str, context_data: Dict[str, Any]) -> None:
        self._storage[conversation_id] = context_data
    
    def load_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get(conversation_id)
    
    def delete_context(self, conversation_id: str) -> None:
        if conversation_id in self._storage:
            del self._storage[conversation_id]


# Global context manager instance
context_manager = ContextManager(storage_backend=InMemoryStorage())