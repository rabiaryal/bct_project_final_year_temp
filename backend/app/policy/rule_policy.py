"""Rule-based Dialogue Policy"""

from typing import Dict, Any, Optional
from app.utils.logger import get_logger
from app.utils.constants import INTENT_TYPES
from app.context.tracker import DialogueTracker

logger = get_logger(__name__)

class PolicyPlanner:
    """
    Rule-based dialogue policy for action prediction.
    Follows Rasa's policy pattern for action selection.
    """
    
    def __init__(self):
        # Intent to action mapping (updated to match actual intent names)
        self.intent_to_action = {
            "GET_COLLEGE_INFO": "action_search_college",
            "GET_COURSE_INFO": "action_search_course", 
            "GET_ADMISSION_INFO": "action_get_admission_info",
            "GET_FEE_INFO": "action_get_fee_info",
            "GET_SCHOLARSHIP_INFO": "action_get_scholarship_info",
            "GET_PLACEMENT_INFO": "action_get_placement_info",
            "GET_FACILITY_INFO": "action_get_facility_info",
            "Get_college_location": "action_get_location_info",
            "Get_contact_info": "action_provide_contact",
            "Greeting": "action_greet",
            "Goodbye": "action_goodbye",
            "THANK_YOU": "action_acknowledge",
            "Unknown": "action_fallback",
            # Legacy mappings for backward compatibility
            "college_info": "action_search_college",
            "course_info": "action_search_course", 
            "admission_info": "action_get_admission_info",
            "fee_info": "action_get_fee_info",
            "scholarship_info": "action_get_scholarship_info",
            "placement_info": "action_get_placement_info",
            "facility_info": "action_get_facility_info",
            "location_info": "action_get_location_info",
            "contact_info": "action_provide_contact",
            "greeting": "action_greet",
            "goodbye": "action_goodbye",
            "thank_you": "action_acknowledge",
            "unknown": "action_fallback"
        }
        
        # Confidence thresholds for different actions
        self.confidence_thresholds = {
            "action_search_college": 0.6,
            "action_search_course": 0.6,
            "action_get_admission_info": 0.5,
            "action_get_fee_info": 0.5,
            "action_greet": 0.4,
            "action_goodbye": 0.4,
            "action_fallback": 0.0
        }
    
    def predict_action(self, tracker: DialogueTracker) -> str:
        """
        Predict next action based on current dialogue state
        
        Args:
            tracker: Current dialogue tracker
            
        Returns:
            Action name to execute
        """
        intent = tracker.intent
        confidence = tracker.context.get("last_intent_confidence", 0.0)
        
        logger.debug(f"Predicting action for intent: {intent}, confidence: {confidence}")
        
        # Handle unknown or low confidence intents
        if not intent or intent not in INTENT_TYPES:
            return "action_fallback"
        
        # Get mapped action
        predicted_action = self.intent_to_action.get(intent, "action_fallback")
        
        # Check confidence threshold
        required_confidence = self.confidence_thresholds.get(predicted_action, 0.5)
        if confidence < required_confidence:
            logger.debug(f"Confidence {confidence} below threshold {required_confidence}")
            return "action_clarify" if confidence > 0.3 else "action_fallback"
        
        # Context-aware action selection
        predicted_action = self._apply_context_rules(tracker, predicted_action)
        
        logger.debug(f"Predicted action: {predicted_action}")
        return predicted_action
    
    def _apply_context_rules(self, tracker: DialogueTracker, action: str) -> str:
        """Apply context-based rules to refine action selection"""
        
        # If we have repeated the same action, try to clarify or fallback
        if tracker.last_action == action and tracker.get_turn_count() > 1:
            recent_turns = tracker.turns[-2:] if len(tracker.turns) >= 2 else tracker.turns
            if all(turn.action == action for turn in recent_turns):
                return "action_clarify"
        
        # College search enhancement
        if action == "action_search_college":
            college_name = tracker.get_slot("college_name")
            if not college_name:
                return "action_ask_college_name"
        
        # Course search enhancement  
        elif action == "action_search_course":
            course_name = tracker.get_slot("course_name")
            if not course_name:
                return "action_ask_course_name"
        
        # Location-based search
        elif action in ["action_get_admission_info", "action_get_fee_info"]:
            college_name = tracker.get_slot("college_name")
            if not college_name:
                # Try to infer from previous context
                if tracker.last_action == "action_search_college":
                    return action  # Continue with the info request
                else:
                    return "action_ask_college_name"
        
        return action
    
    def get_fallback_action(self, tracker: DialogueTracker) -> str:
        """Get appropriate fallback action based on context"""
        turn_count = tracker.get_turn_count()
        
        if turn_count == 0:
            return "action_greet"
        elif turn_count > 10:
            return "action_end_conversation"
        else:
            return "action_fallback"