"""
Query Class Mapping System with Entity Role Integration
Maps intents to clean, scalable query classes with structured entity processing
"""

from enum import Enum
from typing import Dict, List, Any
import logging
from app.policy.entity_roles import entity_processor, EntityRole, Entity

logger = logging.getLogger(__name__)

class QueryClass(Enum):
    """Core query classes for the dialogue system"""
    INFO_LOOKUP = "INFO_LOOKUP"          # Fetch specific info about college/program
    SEARCH = "SEARCH"                    # Search/filter colleges  
    RECOMMENDATION = "RECOMMENDATION"    # Ranking/suggestion logic
    ADMISSION_FLOW = "ADMISSION_FLOW"    # Admission-related steps
    SOCIAL = "SOCIAL"                    # Greeting, goodbye, affirmation
    FALLBACK = "FALLBACK"               # Unknown/clarification

class IntentQueryMapper:
    """Maps intents to query classes and defines DB actions"""
    
    def __init__(self):
        # Intent → Query Class Mapping (from classification report analysis)
        self.INTENT_TO_QUERY_CLASS = {
            # INFO_LOOKUP: Single or factual information requests
            "GET_COLLEGE_INFO": QueryClass.INFO_LOOKUP,
            "GET_program_info": QueryClass.INFO_LOOKUP,
            "GET_fee_info": QueryClass.INFO_LOOKUP,
            "Get_college_location": QueryClass.INFO_LOOKUP,
            "Get_contact_info": QueryClass.INFO_LOOKUP,
            "Get_hostel_availability_info": QueryClass.INFO_LOOKUP,
            "Get_pass_percentage_info": QueryClass.INFO_LOOKUP,
            "Get_scholorship_info": QueryClass.INFO_LOOKUP,

            # SEARCH: Filtering/searching colleges
            "Search_college_by_fee": QueryClass.SEARCH,
            "Search_college_by_location": QueryClass.SEARCH,
            "Search_college_by_program": QueryClass.SEARCH,
            "Search_college_by_seats": QueryClass.SEARCH,
            "Search_college_by_type": QueryClass.SEARCH,

            # RECOMMENDATION: Ranking/suggestion logic
            "Recommend_college": QueryClass.RECOMMENDATION,

            # ADMISSION_FLOW: Procedural guidance
            "Admission_process": QueryClass.ADMISSION_FLOW,
            "Get_admission_info": QueryClass.ADMISSION_FLOW,

            # SOCIAL: Conversation control (NO DB action needed)
            "Greeting": QueryClass.SOCIAL,
            "Goodbye": QueryClass.SOCIAL,
            "Thank_you": QueryClass.SOCIAL,
            "Affirmation": QueryClass.SOCIAL,
            "Negation": QueryClass.SOCIAL,

            # FALLBACK: Error handling
            "Clarification": QueryClass.FALLBACK,
            "Unknown": QueryClass.FALLBACK
        }
        
        # Query Class → DB Action Strategy
        self.QUERY_CLASS_DB_STRATEGY = {
            QueryClass.INFO_LOOKUP: {
                "db_action": "semantic_search",  # FAISS + MongoDB enhancement
                "requires_entities": True,
                "max_results": 3,
                "confidence_threshold": 0.3
            },
            QueryClass.SEARCH: {
                "db_action": "semantic_search",  # FAISS vector search + metadata filtering
                "requires_entities": True, 
                "max_results": 10,
                "confidence_threshold": 0.1
            },
            QueryClass.RECOMMENDATION: {
                "db_action": "semantic_search",  # FAISS + custom scoring
                "requires_entities": False,
                "max_results": 5,
                "confidence_threshold": 0.2
            },
            QueryClass.ADMISSION_FLOW: {
                "db_action": "metadata_filter",  # MongoDB static content only
                "requires_entities": False,
                "max_results": 1,
                "confidence_threshold": 0.0
            },
            QueryClass.SOCIAL: {
                "db_action": None,  # No DB action needed
                "requires_entities": False,
                "max_results": 0,
                "confidence_threshold": 0.0
            },
            QueryClass.FALLBACK: {
                "db_action": None,  # No DB action needed
                "requires_entities": False,
                "max_results": 0,
                "confidence_threshold": 0.0
            }
        }
        
        # Query Class → Action Mapping
        self.QUERY_CLASS_TO_ACTION = {
            QueryClass.INFO_LOOKUP: "action_get_college_info",
            QueryClass.SEARCH: "action_search_college", 
            QueryClass.RECOMMENDATION: "action_recommend_college",
            QueryClass.ADMISSION_FLOW: "action_get_admission_info",
            QueryClass.SOCIAL: "action_social_response",
            QueryClass.FALLBACK: "action_fallback"
        }
    
    def get_query_class(self, intent: str) -> QueryClass:
        """Map intent to query class"""
        return self.INTENT_TO_QUERY_CLASS.get(intent, QueryClass.FALLBACK)
    
    def get_db_strategy(self, query_class: QueryClass) -> Dict[str, Any]:
        """Get database action strategy for query class"""
        return self.QUERY_CLASS_DB_STRATEGY.get(query_class, self.QUERY_CLASS_DB_STRATEGY[QueryClass.FALLBACK])
    
    def get_action(self, query_class: QueryClass) -> str:
        """Get action name for query class"""
        return self.QUERY_CLASS_TO_ACTION.get(query_class, "action_fallback")
    
    def should_use_retrieval(self, query_class: QueryClass) -> bool:
        """Check if query class requires database retrieval"""
        strategy = self.get_db_strategy(query_class)
        return strategy["db_action"] is not None
    
    def get_retrieval_config(self, query_class: QueryClass, entities: List[Dict]) -> Dict[str, Any]:
        """Get retrieval configuration for query class with entity role processing"""
        strategy = self.get_db_strategy(query_class)
        
        # Convert NLU entities to structured Entity objects
        structured_entities = entity_processor.from_nlu_output(entities) if entities else []
        
        # Get entity context for intelligent query planning
        entity_context = entity_processor.get_entity_context(structured_entities)
        
        # Build database query from entities
        db_query_config = entity_processor.build_database_query(structured_entities)
        
        # Entity validation based on query class requirements
        entity_satisfied = True
        if strategy["requires_entities"]:
            if query_class == QueryClass.INFO_LOOKUP:
                # INFO_LOOKUP requires at least an identifier or strong filter
                entity_satisfied = (entity_context["has_identifier"] or 
                                  (entity_context["has_filters"] and len(entity_context["high_confidence_entities"]) > 0))
            elif query_class == QueryClass.SEARCH:
                # SEARCH requires filters or constraints
                entity_satisfied = entity_context["has_filters"] or entity_context["has_constraints"]
            elif query_class == QueryClass.RECOMMENDATION:
                # RECOMMENDATION can work without entities but benefits from signals
                entity_satisfied = True  # Always satisfied for recommendations
            else:
                entity_satisfied = len(structured_entities) > 0
        
        config = {
            "query_class": query_class.value,
            "db_action": strategy["db_action"],
            "max_results": strategy["max_results"],
            "confidence_threshold": strategy["confidence_threshold"],
            "should_retrieve": strategy["db_action"] is not None and entity_satisfied,
            "entity_required": strategy["requires_entities"],
            "entity_satisfied": entity_satisfied,
            
            # Enhanced entity information
            "entity_context": entity_context,
            "database_query": db_query_config,
            "structured_entities": structured_entities,
            
            # Query strategy optimization
            "query_strategy": entity_context.get("query_strategy", "general_search"),
            "optimization_hints": self._get_optimization_hints(query_class, entity_context)
        }
        
        logger.info(f"Query class: {query_class.value} | Strategy: {entity_context.get('query_strategy')} | Should retrieve: {config['should_retrieve']}")
        logger.debug(f"Entity context: {entity_context}")
        
        return config
    
    def _get_optimization_hints(self, query_class: QueryClass, entity_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get query optimization hints based on query class and entity context"""
        hints = {
            "use_semantic_search": True,  # Default to FAISS
            "apply_filters": entity_context["has_filters"] or entity_context["has_constraints"],
            "boost_exact_matches": entity_context["has_identifier"],
            "enable_ranking": entity_context["has_signals"] or query_class == QueryClass.RECOMMENDATION,
            "priority_fields": []
        }
        
        # Set priority fields based on entity roles
        if entity_context["has_identifier"]:
            hints["priority_fields"].append("Name")
        if entity_context["has_filters"]:
            hints["priority_fields"].extend(["Location", "Departments", "Type"])
        
        # Adjust search strategy based on query class
        if query_class == QueryClass.INFO_LOOKUP:
            hints["confidence_boost"] = 0.2  # Boost confidence for specific lookups
            hints["max_fuzzy_distance"] = 2   # Allow some fuzzy matching
        elif query_class == QueryClass.SEARCH:
            hints["enable_aggregation"] = True  # Enable grouping/sorting
            hints["include_similar"] = True     # Include similar results
        elif query_class == QueryClass.RECOMMENDATION:
            hints["enable_ranking"] = True
            hints["use_similarity_scoring"] = True
        
        return hints

# Global instance
intent_query_mapper = IntentQueryMapper()