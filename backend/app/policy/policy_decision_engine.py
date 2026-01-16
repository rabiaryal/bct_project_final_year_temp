"""
Policy Decision Engine for Conversational AI System
Analyzes conversation context and outputs structured policy decisions
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum

from .query_classes import intent_query_mapper, QueryClass
from .entity_roles import EntityProcessor, EntityRole, Entity
from ..context.context_models import ConversationContext
from ..utils.logger import get_logger

logger = get_logger(__name__)

class PolicyAction(Enum):
    """Available policy actions"""
    EXECUTE_QUERY = "EXECUTE_QUERY"
    ASK_REPHRASE = "ASK_REPHRASE"
    ASK_MISSING_INFO = "ASK_MISSING_INFO"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    CONFIRM_ENTITY = "CONFIRM_ENTITY"
    GENERIC_FALLBACK = "GENERIC_FALLBACK"

class QueryStrategy(Enum):
    """Query execution strategies"""
    IDENTIFIER_LOOKUP = "IDENTIFIER_LOOKUP"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    FILTER_SEARCH = "FILTER_SEARCH"
    RECOMMENDATION = "RECOMMENDATION"

class PolicyDecisionEngine:
    """
    Core Policy Decision Engine that analyzes conversation context
    and outputs structured policy decisions for the retrieval system
    """
    
    def __init__(self):
        self.query_mapper = intent_query_mapper
        self.entity_processor = EntityProcessor()
        
        # Confidence thresholds
        self.min_intent_confidence = 0.6
        self.min_entity_confidence = 0.7
        self.low_confidence_threshold = 0.5
        
        # Strategy mapping based on entity roles
        self.strategy_mapping = {
            "has_identifier": QueryStrategy.IDENTIFIER_LOOKUP,
            "has_filters_only": QueryStrategy.SEMANTIC_SEARCH, 
            "has_constraints": QueryStrategy.FILTER_SEARCH,
            "has_signals": QueryStrategy.RECOMMENDATION
        }
        
        logger.info("PolicyDecisionEngine initialized")
    
    def make_decision(self, 
                     context: ConversationContext,
                     query_class: QueryClass,
                     structured_entities: List[Entity],
                     entity_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make policy decision based on processed context, query class, and entity roles
        
        Args:
            context: Updated conversation context
            query_class: Mapped query class from intent
            structured_entities: Processed entities with roles
            entity_context: Entity analysis results
            
        Returns:
            Policy decision JSON following the specified schema
        """
        try:
            # Generate unique policy ID
            policy_id = f"pol_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Extract current intent from context
            current_intent_obj = context.intent.get("current")
            if current_intent_obj:
                # Convert IntentInfo object to dict
                current_intent = {
                    "name": current_intent_obj.name,
                    "confidence": current_intent_obj.confidence
                }
            else:
                current_intent = {"name": None, "confidence": 0.0}
            
            # Make core policy decision using processed inputs
            decision = self._analyze_and_decide(current_intent, structured_entities, entity_context, context, query_class)
            
            # Build execution plan if action is EXECUTE_QUERY
            execution_plan = self._build_execution_plan(decision, structured_entities, entity_context) if decision["action"] == PolicyAction.EXECUTE_QUERY else None
            
            # Build complete policy decision
            policy_decision = {
                "policy_id": policy_id,
                "timestamp": timestamp,
                "status": "PROCESSING",
                "confidence": current_intent.get("confidence", 0.0),
                "decision": {
                    "action": decision["action"].value,
                    "strategy": decision["strategy"].value if decision.get("strategy") else None,
                    "reason": decision["reason"]
                },
                "intent_context": {
                    "name": current_intent.get("name"),
                    "confidence": current_intent.get("confidence", 0.0)
                },
                "entity_context": self._build_entity_context(structured_entities),
                "slot_state": self._build_slot_state(structured_entities, context),
                "execution_plan": execution_plan,
                "fallback": {
                    "on_no_results": "ASK_CLARIFICATION",
                    "on_low_confidence": "CONFIRM_ENTITY", 
                    "on_error": "GENERIC_FALLBACK"
                }
            }
            
            logger.info(f"Policy decision made: {policy_decision['decision']['action']} - {policy_decision['decision']['reason']}")
            return policy_decision
            
        except Exception as e:
            logger.error(f"Error making policy decision: {e}")
            return self._build_error_decision(str(e))
    
    def _analyze_and_decide(self, intent: Dict[str, Any], entities: List[Entity], entity_context: Dict[str, Any], context: ConversationContext, query_class: QueryClass) -> Dict[str, Any]:
        """Core decision analysis logic using processed inputs"""
        
        intent_name = intent.get("name")
        intent_confidence = intent.get("confidence", 0.0)
        
        # Rule 1: Intent confidence check
        if intent_confidence < self.min_intent_confidence:
            return {
                "action": PolicyAction.ASK_REPHRASE,
                "reason": f"Intent confidence {intent_confidence:.2f} below threshold {self.min_intent_confidence}"
            }
        
        # Rule 2: Check for missing required slots based on query class
        missing_slots = self._get_missing_required_slots(query_class, entities, context)
        if missing_slots:
            return {
                "action": PolicyAction.ASK_MISSING_INFO,
                "reason": f"Missing required slots for {query_class.value}: {', '.join(missing_slots)}"
            }
        
        # Rule 3: Determine strategy based on query class and entity roles
        strategy, reason = self._determine_strategy_from_query_class(query_class, entities, entity_context)
        
        # Rule 4: Check entity confidence for strategy
        if strategy == QueryStrategy.IDENTIFIER_LOOKUP:
            identifier_entities = [e for e in entities if e.role == EntityRole.IDENTIFIER]
            if identifier_entities and any(e.confidence < self.min_entity_confidence for e in identifier_entities):
                return {
                    "action": PolicyAction.CONFIRM_ENTITY,
                    "reason": "Low confidence on identifier entity, need confirmation"
                }
        
        return {
            "action": PolicyAction.EXECUTE_QUERY,
            "strategy": strategy,
            "reason": reason
        }
    
    def _determine_strategy_from_query_class(self, query_class: QueryClass, entities: List[Entity], entity_context: Dict[str, Any]) -> tuple:
        """Determine query strategy based on query class and entity roles"""
        
        has_identifier = any(e.role == EntityRole.IDENTIFIER for e in entities)
        has_filters = any(e.role == EntityRole.FILTER for e in entities)
        has_constraints = any(e.role == EntityRole.CONSTRAINT for e in entities)
        has_signals = any(e.role == EntityRole.SIGNAL for e in entities)
        
        # Strategy decision based on query class + entity roles
        if query_class == QueryClass.INFO_LOOKUP:
            if has_identifier:
                return QueryStrategy.IDENTIFIER_LOOKUP, "INFO_LOOKUP with college identifier"
            else:
                return QueryStrategy.SEMANTIC_SEARCH, "INFO_LOOKUP requires semantic search"
                
        elif query_class == QueryClass.SEARCH:
            if has_constraints:
                return QueryStrategy.FILTER_SEARCH, "SEARCH with constraint-based filtering"
            elif has_filters:
                return QueryStrategy.SEMANTIC_SEARCH, "SEARCH with filter entities"
            else:
                return QueryStrategy.SEMANTIC_SEARCH, "General SEARCH query"
                
        elif query_class == QueryClass.RECOMMENDATION:
            return QueryStrategy.RECOMMENDATION, "RECOMMENDATION query with ranking signals"
            
        elif query_class == QueryClass.ADMISSION_FLOW:
            if has_identifier:
                return QueryStrategy.IDENTIFIER_LOOKUP, "ADMISSION_FLOW for specific college"
            else:
                return QueryStrategy.SEMANTIC_SEARCH, "General ADMISSION_FLOW information"
                
        elif query_class == QueryClass.SOCIAL:
            return QueryStrategy.SEMANTIC_SEARCH, "SOCIAL interaction, no specific strategy"
            
        else:  # FALLBACK
            return QueryStrategy.SEMANTIC_SEARCH, "FALLBACK to general semantic search"
    
    def _get_missing_required_slots(self, query_class: QueryClass, entities: List[Entity], context: ConversationContext) -> List[str]:
        """Determine missing required slots based on query class"""
        
        # Define slot requirements by query class
        slot_requirements = {
            QueryClass.INFO_LOOKUP: ["COLLEGE_NAME"],  # Requires specific college for info lookup
            QueryClass.SEARCH: [],  # Can work with any entities or none
            QueryClass.RECOMMENDATION: [],  # Can work with preferences or none
            QueryClass.ADMISSION_FLOW: [],  # Procedural info, no specific requirements
            QueryClass.SOCIAL: [],  # No requirements
            QueryClass.FALLBACK: []  # No requirements
        }
        
        required_slots = slot_requirements.get(query_class, [])
        entity_types = [e.type for e in entities]
        
        missing = [slot for slot in required_slots if slot not in entity_types]
        return missing
    
    def _build_entity_context(self, entities: List[Entity]) -> Dict[str, Any]:
        """Build entity context section for policy decision"""
        
        # Group entities by role
        entities_by_role = {}
        roles_detected = []
        
        for entity in entities:
            role_name = entity.role.value if entity.role else "UNKNOWN"
            if role_name not in entities_by_role:
                entities_by_role[role_name] = {}
                roles_detected.append(role_name)
            
            entities_by_role[role_name][entity.type] = {
                "value": entity.value,
                "confidence": entity.confidence
            }
        
        return {
            "roles_detected": roles_detected,
            "entities": entities_by_role
        }
    
    def _build_slot_state(self, entities: List[Entity], context: ConversationContext) -> Dict[str, Any]:
        """Build slot state section for policy decision"""
        
        # Get current slot state from context
        if context.slots:
            return {
                "required": context.slots.required,
                "filled": context.slots.filled,
                "missing": context.slots.missing
            }
        
        # Fallback: build from entities
        entity_types = [e.type for e in entities]
        return {
            "required": entity_types,
            "filled": entity_types,
            "missing": []
        }
    
    def _build_execution_plan(self, decision: Dict[str, Any], entities: List[Entity], entity_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build execution plan for EXECUTE_QUERY actions"""
        
        strategy = decision["strategy"]
        
        # Base execution plan
        execution_plan = {
            "query_type": self._get_query_type(strategy, entities),
            "data_sources": self._get_data_sources(strategy)
        }
        
        # Build FAISS configuration
        faiss_config = self._build_faiss_config(strategy, entities)
        if faiss_config["enabled"]:
            execution_plan["faiss"] = faiss_config
        
        # Build MongoDB configuration  
        mongodb_config = self._build_mongodb_config(strategy, entities)
        if mongodb_config["enabled"]:
            execution_plan["mongodb"] = mongodb_config
        
        return execution_plan
    
    def _get_query_type(self, strategy: QueryStrategy, entities: List[Entity]) -> str:
        """Determine specific query type based on strategy and entities"""
        
        if strategy == QueryStrategy.IDENTIFIER_LOOKUP:
            return "SINGLE_ENTITY_LOOKUP"
        elif strategy == QueryStrategy.FILTER_SEARCH:
            return "MULTI_FILTER_SEARCH"
        elif strategy == QueryStrategy.RECOMMENDATION:
            return "RECOMMENDATION_QUERY"
        else:
            return "SEMANTIC_SEARCH"
    
    def _get_data_sources(self, strategy: QueryStrategy) -> List[str]:
        """Determine data sources based on strategy"""
        
        source_mapping = {
            QueryStrategy.IDENTIFIER_LOOKUP: ["FAISS", "MONGODB"],
            QueryStrategy.SEMANTIC_SEARCH: ["FAISS", "MONGODB"],
            QueryStrategy.FILTER_SEARCH: ["MONGODB", "FAISS"],
            QueryStrategy.RECOMMENDATION: ["FAISS", "MONGODB"]
        }
        
        return source_mapping.get(strategy, ["FAISS"])
    
    def _build_faiss_config(self, strategy: QueryStrategy, entities: List[Entity]) -> Dict[str, Any]:
        """Build FAISS configuration based on strategy and entities"""
        
        # Check if FAISS should be enabled for this strategy
        faiss_strategies = [QueryStrategy.IDENTIFIER_LOOKUP, QueryStrategy.SEMANTIC_SEARCH, QueryStrategy.RECOMMENDATION]
        
        if strategy not in faiss_strategies:
            return {"enabled": False}
        
        # Build query text from entities
        query_parts = []
        boost_exact_match = False
        
        for entity in entities:
            if entity.role in [EntityRole.IDENTIFIER, EntityRole.FILTER]:
                query_parts.append(entity.value)
                if entity.role == EntityRole.IDENTIFIER:
                    boost_exact_match = True
        
        query_text = " ".join(query_parts) if query_parts else "general college search"
        
        # Set parameters based on strategy
        config = {
            "enabled": True,
            "query_text": query_text,
            "boost_exact_match": boost_exact_match,
            "min_score": 0.3
        }
        
        # Strategy-specific top_k settings
        if strategy == QueryStrategy.IDENTIFIER_LOOKUP:
            config["top_k"] = 3
        elif strategy == QueryStrategy.RECOMMENDATION:
            config["top_k"] = 10
        else:
            config["top_k"] = 5
        
        return config
    
    def _build_mongodb_config(self, strategy: QueryStrategy, entities: List[Entity]) -> Dict[str, Any]:
        """Build MongoDB configuration based on strategy and entities"""
        
        # Build filter from entities using existing entity role system
        db_filter = {}
        
        for entity in entities:
            if entity.role in [EntityRole.IDENTIFIER, EntityRole.FILTER, EntityRole.CONSTRAINT]:
                entity_condition = entity.get_db_condition()
                db_filter.update(entity_condition)
        
        config = {
            "enabled": True,
            "collection": "colleges"
        }
        
        if db_filter:
            config["filter"] = db_filter
        
        return config
    
    def _build_error_decision(self, error_message: str) -> Dict[str, Any]:
        """Build error policy decision"""
        
        return {
            "policy_id": f"pol_error_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ERROR",
            "confidence": 0.0,
            "decision": {
                "action": PolicyAction.GENERIC_FALLBACK.value,
                "strategy": None,
                "reason": f"Policy decision error: {error_message}"
            },
            "intent_context": {
                "name": None,
                "confidence": 0.0
            },
            "entity_context": {
                "roles_detected": [],
                "entities": {}
            },
            "slot_state": {
                "required": [],
                "filled": [],
                "missing": []
            },
            "execution_plan": None,
            "fallback": {
                "on_no_results": "ASK_CLARIFICATION",
                "on_low_confidence": "CONFIRM_ENTITY",
                "on_error": "GENERIC_FALLBACK"
            }
        }

# Global instance
policy_decision_engine = PolicyDecisionEngine()