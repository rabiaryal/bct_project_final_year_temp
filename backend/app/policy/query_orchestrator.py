"""
Query Processing Orchestrator
Handles the correct flow: Input → NLU → Context → Query Class → Entity Roles → Policy → Execution Plan → DB
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from .query_classes import intent_query_mapper, QueryClass
from .entity_roles import EntityProcessor, Entity 
from .policy_decision_engine import policy_decision_engine
from ..context.context_manager import ContextManager
from ..context.context_models import ConversationContext, IntentInfo, EntityInfo, EntitySource
from ..utils.logger import get_logger

logger = get_logger(__name__)

class QueryProcessingOrchestrator:
    """
    Orchestrates the complete query processing pipeline with proper separation of concerns
    
    Flow: Input → NLU → Context → Query Class → Entity Roles → Policy → Execution Plan → DB
    """
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.intent_mapper = intent_query_mapper
        self.entity_processor = EntityProcessor()
        self.policy_engine = policy_decision_engine
        
        logger.info("QueryProcessingOrchestrator initialized")
    
    async def process_complete_query(self, 
                          conversation_id: str,
                          user_input: str, 
                          nlu_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete query processing pipeline (Alias for process_query for compatibility)
        """
        return await self.process_query(conversation_id, user_input, nlu_results)

    async def process_query(self, 
                          conversation_id: str,
                          user_input: str, 
                          nlu_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete query processing pipeline
        
        Args:
            conversation_id: Unique conversation identifier
            user_input: Raw user input text
            nlu_results: NLU output with intent and entities
            
        Returns:
            Complete processing result with policy decision and execution plan
        """
        
        logger.info(f"🚀 Processing query: '{user_input}'")
        
        try:
            # Step 1: Extract intent and entities from NLU results
            intent_name = nlu_results.get("intent", "UNKNOWN")
            intent_confidence = nlu_results.get("intent_confidence", 0.0)
            raw_entities = nlu_results.get("entities", [])
            
            logger.info(f"📊 NLU Results: Intent='{intent_name}' (confidence: {intent_confidence:.3f}), Entities={len(raw_entities)}")
            
            # Step 2: Update Context with intent and entities
            context = await self._update_context(
                conversation_id, 
                user_input,
                intent_name, 
                intent_confidence,
                raw_entities
            )
            
            logger.info(f"🧠 Context updated: Turn {context.turn_id}, {len(context.entities)} entity roles")
            
            # Step 3: Map Intent to Query Class 
            query_class = self.intent_mapper.get_query_class(intent_name)
            retrieval_config = self.intent_mapper.get_retrieval_config(query_class, raw_entities)
            
            logger.info(f"🗂️ Query Class: {query_class.value}")
            
            # Step 4: Process Entities into Structured Format with Roles
            structured_entities = self.entity_processor.from_nlu_output(raw_entities)
            entity_context = self.entity_processor.get_entity_context(structured_entities)
            
            logger.info(f"🏷️ Entity Processing: {entity_context['entity_count']} entities, Strategy: {entity_context['query_strategy']}")
            
            # Step 5: Make Policy Decision using processed context
            policy_decision = self.policy_engine.make_decision(
                context=context,
                query_class=query_class,
                structured_entities=structured_entities, 
                entity_context=entity_context
            )
            
            logger.info(f"🎯 Policy Decision: {policy_decision['decision']['action']} - {policy_decision['decision']['strategy']}")
            
            # Step 6: Return complete processing result
            processing_result = {
                "status": "SUCCESS",
                "conversation_id": conversation_id,
                "user_input": user_input,
                "nlu_results": nlu_results,
                "context": self._serialize_context(context),
                "query_class": query_class.value,
                "retrieval_config": retrieval_config,
                "structured_entities": [self._serialize_entity(e) for e in structured_entities],
                "entity_context": entity_context,
                "policy_decision": policy_decision,
                "processing_pipeline": {
                    "steps": [
                        "NLU Processing",
                        "Context Management", 
                        "Query Class Mapping",
                        "Entity Role Processing",
                        "Policy Decision",
                        "Execution Plan Creation"
                    ],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Query processing completed successfully")
            return processing_result
            
        except Exception as e:
            logger.error(f"❌ Error in query processing: {e}")
            return self._build_error_response(conversation_id, user_input, str(e))
    
    async def _update_context(self,
                            conversation_id: str,
                            user_input: str,
                            intent_name: str,
                            intent_confidence: float,
                            raw_entities: List[Dict]) -> ConversationContext:
        """Update conversation context with new turn information"""
        
        # Get or create context
        context = self.context_manager.get_context(conversation_id)
        
        # Update intent
        previous_intent = context.intent.get("current")
        if previous_intent:
            context.intent["previous"].append(previous_intent)
        context.intent["current"] = IntentInfo(name=intent_name, confidence=intent_confidence)
        
        # Update turn counter
        context.turn_id += 1
        
        # Process and merge entities
        for raw_entity in raw_entities:
            entity_type = raw_entity.get("entity", raw_entity.get("type", "")).upper()
            entity_value = raw_entity.get("value", raw_entity.get("text", ""))
            entity_confidence = raw_entity.get("confidence", 0.0)
            
            # Create entity info
            entity_info = EntityInfo(
                value=entity_value,
                confidence=entity_confidence,
                source=EntitySource.USER
            )
            
            # Determine role for grouping (using existing entity role system)
            entity_role = self._get_entity_role_name(entity_type)
            
            # Merge into context entities (replace if higher confidence)
            if entity_role not in context.entities:
                context.entities[entity_role] = {}
            
            existing_entity = context.entities[entity_role].get(entity_type)
            if not existing_entity or entity_confidence > existing_entity.confidence:
                context.entities[entity_role][entity_type] = entity_info
                logger.debug(f"Added/updated entity: {entity_role}.{entity_type} = '{entity_value}' ({entity_confidence:.2f})")
        
        # Update slots
        context.slots.filled = []
        context.slots.missing = []
        for role_entities in context.entities.values():
            for entity_type in role_entities.keys():
                context.slots.filled.append(entity_type)
        
        # Update timestamp
        context.timestamp = datetime.now().isoformat()
        
        return context
    
    def _get_entity_role_name(self, entity_type: str) -> str:
        """Get entity role name for context grouping"""
        from .entity_roles import ENTITY_TYPE_TO_ROLE
        
        role = ENTITY_TYPE_TO_ROLE.get(entity_type)
        return role.value if role else "UNKNOWN"
    
    def _serialize_context(self, context: ConversationContext) -> Dict[str, Any]:
        current_intent = context.intent.get("current")
        previous_intents = context.intent.get("previous", [])
        
        intent_data = {
            "current": {
                "name": current_intent.name if current_intent else None,
                "confidence": current_intent.confidence if current_intent else 0.0
            },
            "previous": [
                {"name": intent.name, "confidence": intent.confidence} 
                for intent in previous_intents
            ]
        }

        return {
            "conversation_id": context.conversation_id,
            "turn_id": context.turn_id,
            "intent": intent_data,
            "entity_roles": list(context.entities.keys()),
            "entity_count": sum(len(role_entities) for role_entities in context.entities.values()),
            "slots": {
                "filled": context.slots.filled,
                "missing": context.slots.missing
            }
        }
    
    def _serialize_entity(self, entity: Entity) -> Dict[str, Any]:
        """Serialize entity for response"""
        return {
            "type": entity.type,
            "value": entity.value,
            "confidence": entity.confidence,
            "role": entity.role.value if entity.role else "UNKNOWN",
            "db_field": entity.db_field,
            "status": "ERROR",
            "db_condition": entity.get_db_condition()
        }
    
    def _build_error_response(self, conversation_id: str, user_input: str, error_message: str) -> Dict[str, Any]:
        """Build error response"""
        return {
            "conversation_id": conversation_id,
            "user_input": user_input,
            "error": error_message,
            "policy_decision": {
                "decision": {"action": "GENERIC_FALLBACK", "reason": f"Processing error: {error_message}"}
            },
            "processing_pipeline": {
                "steps": ["Error occurred during processing"],
                "timestamp": datetime.now().isoformat()
            }
        }

# Global orchestrator instance
query_orchestrator = QueryProcessingOrchestrator()