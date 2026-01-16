"""Enhanced Rule-Based Dialogue Policy with Query Class Mapping"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from app.utils.logger import get_logger
from app.utils.constants import INTENT_TYPES
from app.policy.query_classes import intent_query_mapper, QueryClass

if TYPE_CHECKING:
    from app.context.tracker import DialogueTracker

logger = get_logger(__name__)

class PolicyPlanner:
    """
    Enhanced rule-based dialogue policy using query class architecture.
    Maps intents to query classes for cleaner, more scalable system design.
    """
    
    def __init__(self):
        self.mapper = intent_query_mapper
        
        # Enhanced confidence thresholds by query class
        self.confidence_thresholds = {
            QueryClass.INFO_LOOKUP: 0.6,      # High confidence for specific info
            QueryClass.SEARCH: 0.4,           # Lower threshold for search queries
            QueryClass.RECOMMENDATION: 0.5,    # Medium confidence for recommendations
            QueryClass.ADMISSION_FLOW: 0.7,   # High confidence for procedural info
            QueryClass.SOCIAL: 0.3,           # Low threshold for social interactions
            QueryClass.FALLBACK: 0.0          # Always accept fallback
        }
        
        logger.info("PolicyPlanner initialized with query class mapping system")
    
    def predict_action(self, tracker: "DialogueTracker") -> str:
        """
        Enhanced action prediction using query class mapping system.
        
        Flow:
        1. Map intent → query class  
        2. Check confidence against query class threshold
        3. Determine if retrieval is needed
        4. Select appropriate action
        """
        intent = tracker.intent
        confidence = tracker.context.get("last_intent_confidence", 0.0)
        entities = tracker.entities
        
        logger.info(f"🎯 Policy Planning: Intent='{intent}' (confidence: {confidence:.3f})")
        
        # Step 1: Handle unknown intents
        if not intent or intent not in INTENT_TYPES:
            logger.info("❌ Unknown intent, using fallback")
            return "action_fallback"
        
        # Step 2: Map intent to query class
        query_class = self.mapper.get_query_class(intent)
        logger.info(f"📊 Query Class: {query_class.value}")
        
        # Step 3: Check confidence threshold
        required_confidence = self.confidence_thresholds.get(query_class, 0.5)
        
        # Step 4: Entity-based confidence boost for relevant query classes
        if query_class in [QueryClass.INFO_LOOKUP, QueryClass.SEARCH]:
            entity_boost = self._calculate_entity_boost(entities)
            effective_confidence = confidence + entity_boost
            logger.debug(f"Entity boost: {entity_boost:.2f}, effective confidence: {effective_confidence:.3f}")
        else:
            effective_confidence = confidence
            entity_boost = 0.0
        
        # Step 5: Confidence validation
        if effective_confidence < required_confidence:
            logger.info(f"❌ Low confidence: {effective_confidence:.3f} < {required_confidence:.3f}")
            if confidence > 0.3:
                return "action_ask_college_name"  # Ask for clarification
            else:
                return "action_fallback"  # Complete fallback
        
        # Step 6: Get retrieval configuration
        retrieval_config = self.mapper.get_retrieval_config(query_class, list(entities.values()) if entities else [])
        
        # Step 7: Handle entity requirements
        if retrieval_config["entity_required"] and not retrieval_config["entity_satisfied"]:
            logger.info(f"⚠️ Missing required entities for {query_class.value}")
            return "action_ask_college_name"
        
        # Step 8: Get final action from query class
        predicted_action = self.mapper.get_action(query_class)
        
        # Step 9: Apply context rules for refinement
        final_action = self._apply_context_rules(tracker, predicted_action, query_class)
        
        logger.info(f"🎬 Final Action: {final_action} | Query Class: {query_class.value}")
        logger.info(f"🔍 Retrieval Config: {retrieval_config}")
        
        return final_action
    
    def _calculate_entity_boost(self, entities: Dict[str, Any]) -> float:
        """Calculate confidence boost based on relevant entities"""
        boost = 0.0
        relevant_entities = ["COLLEGE_NAME", "LOCATION", "PROGRAM", "DEPARTMENT"]
        
        for entity_type, entity_value in entities.items():
            if entity_type in relevant_entities and entity_value:
                boost += 0.15  # Boost per relevant entity
        
        return min(boost, 0.3)  # Cap at 0.3 to prevent over-boosting
    
    def _apply_context_rules(self, tracker: "DialogueTracker", action: str, query_class: QueryClass) -> str:
        """Apply context-based rules to refine action selection"""
        
        # Prevent action loops
        if tracker.last_action == action and tracker.get_turn_count() > 1:
            recent_turns = tracker.turns[-2:] if len(tracker.turns) >= 2 else tracker.turns
            if all(turn.action == action for turn in recent_turns):
                logger.info("🔄 Preventing action loop, using clarification")
                return "action_ask_college_name"
        
        # Query class specific refinements
        if query_class == QueryClass.INFO_LOOKUP:
            college_name = tracker.get_slot("COLLEGE_NAME") or tracker.get_slot("college_name")
            if not college_name:
                logger.info("🏫 INFO_LOOKUP missing college name, asking for it")
                return "action_ask_college_name"
        
        elif query_class == QueryClass.SEARCH:
            # For search queries, we can proceed even without specific college name
            # but might need location or program info
            entities_count = len(tracker.entities) if tracker.entities else 0
            if entities_count == 0:
                logger.info("🔍 SEARCH missing search criteria")
                return "action_ask_college_name"
        
        elif query_class == QueryClass.SOCIAL:
            # Social interactions don't need entity refinement
            pass
        
        elif query_class == QueryClass.ADMISSION_FLOW:
            # For admission flow, college name is helpful but not required
            pass
        
        return action
    
    def get_retrieval_strategy(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Get retrieval strategy configuration for given intent and entities"""
        query_class = self.mapper.get_query_class(intent)
        return self.mapper.get_retrieval_config(query_class, list(entities.values()) if entities else [])
    
    def should_use_retrieval(self, intent: str) -> bool:
        """Check if intent requires database retrieval"""
        query_class = self.mapper.get_query_class(intent)
        return self.mapper.should_use_retrieval(query_class)
    
    def get_fallback_action(self, tracker: "DialogueTracker") -> str:
        """Get appropriate fallback action based on context"""
        turn_count = tracker.get_turn_count()
        
        if turn_count == 0:
            return "action_greet"
        elif turn_count > 10:
            return "action_end_conversation"
        else:
            return "action_fallback"