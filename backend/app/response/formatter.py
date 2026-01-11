"""Response Generation and Formatting"""

from typing import Dict, Any, List
import random
from datetime import datetime

from app.utils.logger import get_logger
from app.utils.constants import RESPONSE_TEMPLATES

logger = get_logger(__name__)

class ResponseFormatter:
    """
    Formats dialogue responses with templates and dynamic content.
    Handles NLG (Natural Language Generation) for dialogue responses.
    """
    
    def __init__(self):
        self.templates = RESPONSE_TEMPLATES.copy()
        self._add_dynamic_templates()
    
    def format_response(
        self, 
        base_response: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Format response with dynamic content"""
        try:
            if not base_response:
                return "I apologize, but I don't have a response for that."
            
            # If base_response is a dict (action result), extract the response
            if isinstance(base_response, dict):
                base_response = base_response.get('response', str(base_response))
            
            # Simple formatting - add any dynamic content from context
            formatted_response = str(base_response)
            
            # Add timestamp if needed
            if context and context.get('add_timestamp'):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                formatted_response += f" (Generated at {timestamp})"
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            return "I apologize, but I encountered an error generating the response."
        
        try:
            # Apply entity-based formatting
            if entities:
                base_response = self._apply_entity_formatting(base_response, entities)
            
            # Apply context-based enhancements
            if context:
                base_response = self._apply_context_formatting(base_response, context)
            
            # Add intent-specific formatting
            if intent:
                base_response = self._apply_intent_formatting(base_response, intent)
            
            # Add timestamp if needed
            base_response = self._add_timestamp(base_response)
            
            return base_response
            
        except Exception as e:
            logger.error(f"Response formatting error: {e}")
            return base_response  # Return original if formatting fails
    
    def get_template_response(self, template_key: str, **kwargs) -> str:
        """Get response from template with dynamic values"""
        
        if template_key not in self.templates:
            logger.warning(f"Template {template_key} not found")
            return "I'm not sure how to respond to that."
        
        templates = self.templates[template_key]
        if isinstance(templates, list):
            template = random.choice(templates)
        else:
            template = templates
        
        # Format template with provided values
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Template formatting error: {e}")
            return template
    
    def format_college_list(self, colleges: List[Dict[str, Any]], query: str = None) -> str:
        """Format list of colleges for display"""
        
        if not colleges:
            return "I couldn't find any colleges matching your criteria."
        
        if len(colleges) == 1:
            return self._format_single_college(colleges[0])
        
        # Multiple colleges
        response = f"I found {len(colleges)} colleges"
        if query:
            response += f" related to '{query}'"
        response += ":\n\n"
        
        for i, college in enumerate(colleges[:5], 1):  # Limit to 5
            name = college.get("name", "Unknown")
            location = college.get("location", "Location not specified")
            response += f"{i}. **{name}** - {location}\n"
        
        if len(colleges) > 5:
            response += f"\n... and {len(colleges) - 5} more. Please be more specific for detailed information."
        
        response += "\n\nWould you like detailed information about any specific college?"
        
        return response
    
    def format_error_response(self, error_type: str, details: str = None) -> str:
        """Format error responses"""
        
        error_templates = {
            "college_not_found": "I couldn't find information about that college. Please check the spelling or try a different name.",
            "system_error": "I'm experiencing technical difficulties. Please try again in a moment.",
            "invalid_input": "I didn't understand that request. Could you please rephrase it?",
            "timeout": "The request took too long to process. Please try again.",
            "database_error": "I'm having trouble accessing college data right now. Please try again later."
        }
        
        base_response = error_templates.get(error_type, "An error occurred while processing your request.")
        
        if details:
            base_response += f" Details: {details}"
        
        return base_response
    
    def _apply_entity_formatting(self, response: str, entities: Dict[str, Any]) -> str:
        """Apply entity-based response formatting"""
        
        # Replace common entity placeholders
        if college_name := entities.get("college_name"):
            response = response.replace("{college}", college_name)
            response = response.replace("{college_name}", college_name)
        
        if course_name := entities.get("course_name"):
            response = response.replace("{course}", course_name)
            response = response.replace("{course_name}", course_name)
        
        if location := entities.get("location"):
            response = response.replace("{location}", location)
        
        return response
    
    def _apply_context_formatting(self, response: str, context: Dict[str, Any]) -> str:
        """Apply context-based response formatting"""
        
        # Add personalization based on conversation history
        turn_count = context.get("turn_count", 0)
        
        if turn_count > 5:
            # Add helpful hints for long conversations
            if "Is there anything else" not in response:
                response += "\n\nIs there anything else you'd like to know?"
        
        # Add confidence indicators for low confidence responses
        confidence = context.get("last_intent_confidence", 1.0)
        if confidence < 0.6:
            response = "I think you're asking about: " + response
        
        return response
    
    def _apply_intent_formatting(self, response: str, intent: str) -> str:
        """Apply intent-specific formatting"""
        
        # Add appropriate emojis or formatting based on intent
        intent_formatting = {
            "greeting": "👋 ",
            "goodbye": "👋 ",
            "college_info": "🏫 ",
            "course_info": "📚 ",
            "fee_info": "💰 ",
            "admission_info": "📝 ",
            "thank_you": "😊 "
        }
        
        if prefix := intent_formatting.get(intent):
            if not response.startswith(prefix):
                response = prefix + response
        
        return response
    
    def _add_timestamp(self, response: str) -> str:
        """Add timestamp to response if needed"""
        # Only add timestamp for certain types of information
        timestamp_keywords = ["latest", "current", "updated", "as of"]
        
        if any(keyword in response.lower() for keyword in timestamp_keywords):
            current_time = datetime.now().strftime("%B %d, %Y")
            response += f"\n\n*Information as of {current_time}*"
        
        return response
    
    def _format_single_college(self, college: Dict[str, Any]) -> str:
        """Format single college information"""
        name = college.get("name", "Unknown College")
        location = college.get("location", "Location not specified")
        
        response = f"**{name}**\n"
        response += f"📍 **Location:** {location}\n"
        
        if college_type := college.get("type"):
            response += f"🏫 **Type:** {college_type}\n"
        
        return response
    
    def _add_dynamic_templates(self):
        """Add dynamic response templates"""
        
        self.templates.update({
            "college_found": [
                "Here's what I found about {college_name}:",
                "Information for {college_name}:",
                "{college_name} details:"
            ],
            "multiple_colleges": [
                "I found several colleges matching your search:",
                "Here are the colleges I found:",
                "Multiple colleges match your criteria:"
            ],
            "clarification": [
                "Could you clarify what you're looking for?",
                "I need more information to help you better.",
                "Can you be more specific about your question?"
            ],
            "acknowledge": [
                "You're welcome! Is there anything else you'd like to know?",
                "Happy to help! Any other questions?",
                "Glad I could assist! What else can I help with?"
            ]
        })