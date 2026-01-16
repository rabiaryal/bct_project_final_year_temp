"""
Entity Role System for NLU Layer
Defines entity roles and mappings for database integration
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class EntityRole(Enum):
    """Entity roles for database query logic"""
    IDENTIFIER = "identifier"        # Uniquely identifies an object
    FILTER = "filter"               # Restricts result set  
    CONSTRAINT = "constraint"       # Numeric or logical condition
    ATTRIBUTE = "attribute"         # Requested property
    RELATION = "relation"           # Describes a relationship
    SIGNAL = "signal"              # Controls recommendation or ranking

@dataclass
class Entity:
    """Structured entity with role-based behavior"""
    type: str
    value: str
    confidence: float
    role: EntityRole = None
    db_field: Optional[str] = None
    
    def __post_init__(self):
        """Auto-assign role and db_field based on entity type"""
        self.role = ENTITY_TYPE_TO_ROLE.get(self.type)
        self.db_field = ENTITY_TYPE_TO_DB_FIELD.get(self.type)
    
    def __repr__(self):
        return f"<Entity {self.type}={self.value} role={self.role}>"
    
    def is_valid(self) -> bool:
        """Check if entity has valid role assignment"""
        return self.role is not None
    
    def get_db_condition(self) -> Dict[str, Any]:
        """Convert entity to database condition based on role"""
        if not self.db_field:
            return {}
            
        if self.role == EntityRole.IDENTIFIER:
            # Exact match for identifiers
            return {self.db_field: {"$regex": self.value, "$options": "i"}}
        
        elif self.role == EntityRole.FILTER:
            # Case-insensitive contains for filters
            return {self.db_field: {"$regex": self.value, "$options": "i"}}
        
        elif self.role == EntityRole.CONSTRAINT:
            # Handle numeric constraints
            if self.type == "FEE":
                return self._parse_fee_constraint()
            elif self.type in ["SEATS", "CUTOFF_RANK", "RATING"]:
                return self._parse_numeric_constraint()
            else:
                return {self.db_field: self.value}
        
        elif self.role == EntityRole.ATTRIBUTE:
            # Attributes are used for projection, not filtering
            return {}
        
        elif self.role == EntityRole.RELATION:
            # Handle relationship queries
            return {self.db_field: {"$regex": self.value, "$options": "i"}}
        
        elif self.role == EntityRole.SIGNAL:
            # Signals don't create direct DB conditions
            return {}
        
        return {}
    
    def _parse_fee_constraint(self) -> Dict[str, Any]:
        """Parse fee constraint like '<1000000' or 'below 10 lakh'"""
        value = self.value.lower()
        
        # Handle different fee formats
        if 'lakh' in value or 'lakhs' in value:
            import re
            numbers = re.findall(r'\d+', value)
            if numbers:
                amount = int(numbers[0]) * 100000
                if 'below' in value or 'under' in value or '<' in value:
                    return {self.db_field: {"$lt": amount}}
                elif 'above' in value or 'over' in value or '>' in value:
                    return {self.db_field: {"$gt": amount}}
                else:
                    return {self.db_field: {"$lte": amount}}
        
        # Handle direct numeric values
        try:
            amount = float(value.replace('<', '').replace('>', '').replace('=', ''))
            if '<' in self.value:
                return {self.db_field: {"$lt": amount}}
            elif '>' in self.value:
                return {self.db_field: {"$gt": amount}}
            else:
                return {self.db_field: {"$lte": amount}}
        except:
            return {}
    
    def _parse_numeric_constraint(self) -> Dict[str, Any]:
        """Parse numeric constraints for seats, ranks, etc."""
        try:
            value = self.value.lower()
            if 'above' in value or '>' in value:
                num = float(value.replace('above', '').replace('>', '').strip())
                return {self.db_field: {"$gt": num}}
            elif 'below' in value or '<' in value:
                num = float(value.replace('below', '').replace('<', '').strip())
                return {self.db_field: {"$lt": num}}
            else:
                num = float(value)
                return {self.db_field: num}
        except:
            return {}

# Entity Type → Role Mapping (AUTHORITATIVE)
ENTITY_TYPE_TO_ROLE = {
    # IDENTIFIERS - Uniquely identify objects
    "COLLEGE_NAME": EntityRole.IDENTIFIER,
    
    # FILTERS - Restrict result set
    "LOCATION": EntityRole.FILTER,
    "PROGRAM": EntityRole.FILTER,
    "DEPARTMENT": EntityRole.FILTER,
    "COLLEGE_TYPE": EntityRole.FILTER,
    "AFFILIATION": EntityRole.FILTER,
    "FACILITY": EntityRole.FILTER,
    "INTERNSHIP": EntityRole.FILTER,
    
    # CONSTRAINTS - Numeric or logical conditions
    "FEE": EntityRole.CONSTRAINT,
    "FEES": EntityRole.CONSTRAINT,
    "SEATS": EntityRole.CONSTRAINT,
    "CUTOFF_RANK": EntityRole.CONSTRAINT,
    "APPLICATION_DEADLINE": EntityRole.CONSTRAINT,
    "RATING": EntityRole.CONSTRAINT,
    
    # ATTRIBUTES - Requested properties
    "CONTACT_INFO": EntityRole.ATTRIBUTE,
    "HOSTEL_AVAILABILITY": EntityRole.ATTRIBUTE,
    "ADMISSION_PROCESS": EntityRole.ATTRIBUTE,
    "SCHOLARSHIP": EntityRole.ATTRIBUTE,
    "SCHOLORSHIP": EntityRole.ATTRIBUTE,  # Handle common misspelling
    
    # RELATIONS - Describe relationships
    "COURSE": EntityRole.RELATION,
    "SUBJECT": EntityRole.RELATION,
    
    # SIGNALS - Control recommendation/ranking
    "RECOMMEND": EntityRole.SIGNAL,
    "BEST": EntityRole.SIGNAL,
    "TOP": EntityRole.SIGNAL
}

# Entity Type → Database Field Mapping
ENTITY_TYPE_TO_DB_FIELD = {
    # MongoDB field names (use actual field names from your schema)
    "COLLEGE_NAME": "Name",  # MongoDB uses 'Name' with capital N
    "LOCATION": "Location",   # MongoDB uses 'Location' with capital L
    "PROGRAM": "Departments", # MongoDB uses 'Departments'
    "DEPARTMENT": "Departments",
    "COLLEGE_TYPE": "Type",   # MongoDB uses 'Type'
    "AFFILIATION": "Affiliation",
    "FACILITY": "Facilities",
    "INTERNSHIP": "InternshipAvailable",
    "FEE": "Fees",
    "FEES": "Fees", 
    "SEATS": "Seats",
    "CUTOFF_RANK": "CutoffRank",
    "APPLICATION_DEADLINE": "ApplicationDeadline",
    "RATING": "Rating",
    "CONTACT_INFO": "ContactNumber",
    "COURSE": "Departments",
    "SUBJECT": "Departments"
}

class EntityProcessor:
    """Processes entities for database query integration"""
    
    @staticmethod
    def from_nlu_output(nlu_entities: List[Dict[str, Any]]) -> List[Entity]:
        """Convert NLU output to structured Entity objects"""
        entities = []
        
        for ent_data in nlu_entities:
            entity_type = ent_data.get('type', '').upper()
            entity_value = ent_data.get('text', '')
            confidence = ent_data.get('confidence', 0.0)
            
            if entity_type in ENTITY_TYPE_TO_ROLE:
                entity = Entity(
                    type=entity_type,
                    value=entity_value,
                    confidence=confidence
                )
                if entity.is_valid():
                    entities.append(entity)
                    logger.debug(f"Created entity: {entity}")
                else:
                    logger.warning(f"Invalid entity: {entity_type}")
            else:
                logger.debug(f"Unknown entity type: {entity_type}")
        
        return entities
    
    @staticmethod
    def build_database_query(entities: List[Entity]) -> Dict[str, Any]:
        """Build MongoDB query from entities based on their roles"""
        mongo_query = {}
        projection = []
        signals = []
        
        for entity in entities:
            if not entity.is_valid():
                continue
                
            # Get database condition based on entity role
            condition = entity.get_db_condition()
            if condition:
                mongo_query.update(condition)
            
            # Collect attributes for projection
            if entity.role == EntityRole.ATTRIBUTE:
                projection.append(entity.type)
            
            # Collect signals for ranking/recommendation
            elif entity.role == EntityRole.SIGNAL:
                signals.append(entity.value)
        
        query_config = {
            "filter": mongo_query,
            "attributes": projection,
            "signals": signals,
            "entity_summary": {
                "identifiers": [e for e in entities if e.role == EntityRole.IDENTIFIER],
                "filters": [e for e in entities if e.role == EntityRole.FILTER],
                "constraints": [e for e in entities if e.role == EntityRole.CONSTRAINT],
                "attributes": [e for e in entities if e.role == EntityRole.ATTRIBUTE],
                "relations": [e for e in entities if e.role == EntityRole.RELATION],
                "signals": [e for e in entities if e.role == EntityRole.SIGNAL]
            }
        }
        
        logger.info(f"Built query: {mongo_query}")
        logger.debug(f"Entity summary: identifiers={len(query_config['entity_summary']['identifiers'])}, "
                    f"filters={len(query_config['entity_summary']['filters'])}, "
                    f"constraints={len(query_config['entity_summary']['constraints'])}")
        
        return query_config
    
    @staticmethod
    def get_entity_context(entities: List[Entity]) -> Dict[str, Any]:
        """Extract entity context for query planning"""
        context = {
            "has_identifier": any(e.role == EntityRole.IDENTIFIER for e in entities),
            "has_filters": any(e.role == EntityRole.FILTER for e in entities),
            "has_constraints": any(e.role == EntityRole.CONSTRAINT for e in entities),
            "has_attributes": any(e.role == EntityRole.ATTRIBUTE for e in entities),
            "has_signals": any(e.role == EntityRole.SIGNAL for e in entities),
            "entity_count": len(entities),
            "high_confidence_entities": [e for e in entities if e.confidence > 0.8]
        }
        
        # Determine query strategy based on entity composition
        if context["has_identifier"]:
            context["query_strategy"] = "specific_lookup"
        elif context["has_filters"] or context["has_constraints"]:
            context["query_strategy"] = "filtered_search" 
        elif context["has_signals"]:
            context["query_strategy"] = "recommendation"
        else:
            context["query_strategy"] = "general_search"
        
        return context

# Global processor instance
entity_processor = EntityProcessor()