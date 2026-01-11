"""Dialogue Action Handlers"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from app.services.college_service import CollegeService
from app.schemas import ActionRequest, ActionResponse
from app.utils.logger import get_logger
from app.utils.constants import RESPONSE_TEMPLATES
import random

logger = get_logger(__name__)

class BaseAction(ABC):
    """Base class for all dialogue actions"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute the action"""
        pass

class ActionSearchCollege(BaseAction):
    """Search for college information"""
    
    def __init__(self, college_service: Optional[CollegeService]):
        super().__init__("action_search_college")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute college search action"""
        try:
            slots = request.slots
            college_name = slots.get("college_name")
            
            if not college_name:
                return ActionResponse(
                    response="Which college would you like to know about?",
                    success=False,
                    metadata={"requires": "college_name"}
                )
            
            # Check if MongoDB service is available
            if not self.college_service:
                return ActionResponse(
                    response=f"I understand you're asking about {college_name}. However, I'm currently running in demo mode without access to the college database. The system includes AI-powered intent recognition and entity extraction, and would normally provide detailed information about colleges, courses, fees, and admission requirements when connected to the database.",
                    success=True,
                    metadata={"demo_mode": True, "college_name": college_name}
                )
            
            # Search for college
            result = await self.college_service.get_college_details(college_name)
            
            if result["success"]:
                college = result["college"]
                response = self._format_college_info(college)
                
                return ActionResponse(
                    response=response,
                    slots_updated={"college_found": True},
                    metadata={"college_data": college}
                )
            else:
                suggestions = result.get("suggestions", [])
                response = f"I couldn't find information about '{college_name}'."
                
                if suggestions:
                    response += f" Did you mean: {', '.join(suggestions[:3])}?"
                
                return ActionResponse(
                    response=response,
                    success=False,
                    metadata={"suggestions": suggestions}
                )
        
        except Exception as e:
            logger.error(f"College search action error: {e}")
            return ActionResponse(
                response="I'm sorry, I encountered an error while searching for college information.",
                success=False,
                metadata={"error": str(e)}
            )
    
    def _format_college_info(self, college: Dict[str, Any]) -> str:
        """Format college information for response"""
        name = college.get("name", "Unknown")
        location = college.get("location", "Location not specified")
        type_info = college.get("type", "")
        
        response = f"**{name}**\n\n"
        response += f"📍 **Location:** {location}\n"
        
        if type_info:
            response += f"🏫 **Type:** {type_info}\n"
        
        if affiliation := college.get("affiliation"):
            response += f"🔗 **Affiliation:** {affiliation}\n"
        
        if established := college.get("established"):
            response += f"📅 **Established:** {established}\n"
        
        if courses := college.get("courses"):
            response += f"📚 **Courses:** {courses[:200]}...\n" if len(courses) > 200 else f"📚 **Courses:** {courses}\n"
        
        if website := college.get("website"):
            response += f"🌐 **Website:** {website}\n"
        
        if phone := college.get("phone"):
            response += f"📞 **Phone:** {phone}\n"
        
        response += "\nWould you like to know more about admissions, fees, or facilities?"
        
        return response

class ActionGetAdmissionInfo(BaseAction):
    """Get admission information"""
    
    def __init__(self, college_service: Optional[CollegeService]):
        super().__init__("action_get_admission_info")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute admission info action"""
        try:
            college_name = request.slots.get("college_name")
            
            if not college_name:
                return ActionResponse(
                    response="Which college's admission information would you like?",
                    success=False
                )
            
            if not self.college_service:
                return ActionResponse(
                    response=f"I understand you want admission information for {college_name}. In demo mode, I can tell you that admission information typically includes application processes, eligibility criteria, entrance exams, deadlines, and required documents. When connected to the database, I would provide specific details for {college_name}.",
                    success=True,
                    metadata={"demo_mode": True, "college_name": college_name}
                )
            
            result = await self.college_service.get_college_admission_info(college_name)
            
            if result["success"]:
                admission_info = result["admission_info"]
                response = self._format_admission_info(admission_info)
                
                return ActionResponse(
                    response=response,
                    metadata={"admission_data": admission_info}
                )
            else:
                return ActionResponse(
                    response=f"I couldn't find admission information for {college_name}. Please check the college name.",
                    success=False
                )
        
        except Exception as e:
            logger.error(f"Admission info action error: {e}")
            return ActionResponse(
                response="Sorry, I encountered an error getting admission information.",
                success=False
            )
    
    def _format_admission_info(self, info: Dict[str, Any]) -> str:
        """Format admission information"""
        college_name = info.get("college_name", "College")
        
        response = f"**Admission Information - {college_name}**\n\n"
        
        if process := info.get("admission_process"):
            response += f"📋 **Process:** {process}\n\n"
        
        if eligibility := info.get("eligibility"):
            response += f"✅ **Eligibility:** {eligibility}\n\n"
        
        if exam := info.get("entrance_exam"):
            response += f"📝 **Entrance Exam:** {exam}\n\n"
        
        if deadline := info.get("application_deadline"):
            response += f"⏰ **Deadline:** {deadline}\n\n"
        
        contact = info.get("contact_info", {})
        if any(contact.values()):
            response += "**Contact for More Details:**\n"
            if contact.get("phone"):
                response += f"📞 {contact['phone']}\n"
            if contact.get("email"):
                response += f"📧 {contact['email']}\n"
            if contact.get("website"):
                response += f"🌐 {contact['website']}\n"
        
        return response

class ActionGreet(BaseAction):
    """Greeting action"""
    
    def __init__(self):
        super().__init__("action_greet")
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute greeting action"""
        response = random.choice(RESPONSE_TEMPLATES["greeting"])
        return ActionResponse(response=response)

class ActionGoodbye(BaseAction):
    """Goodbye action"""
    
    def __init__(self):
        super().__init__("action_goodbye")
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute goodbye action"""
        response = random.choice(RESPONSE_TEMPLATES["goodbye"])
        return ActionResponse(response=response)

class ActionGetLocationInfo(BaseAction):
    """Get college location information"""
    
    def __init__(self, college_service: Optional[CollegeService]):
        super().__init__("action_get_location_info")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute location info action"""
        try:
            slots = request.slots
            college_name = slots.get("college_name") or slots.get("college_name_name")
            
            if not college_name:
                return ActionResponse(
                    response="Which college's location would you like to know about?",
                    success=False,
                    metadata={"requires": "college_name"}
                )
            
            # Check if MongoDB service is available
            if not self.college_service:
                return ActionResponse(
                    response=f"I understand you're asking about the location of {college_name}. In demo mode, I can tell you that most engineering colleges in Nepal are located in major cities like Kathmandu, Lalitpur, Pokhara, and other urban centers. When connected to the database, I would provide the exact address, contact details, and nearby landmarks.",
                    success=True,
                    metadata={"demo_mode": True, "college_name": college_name, "action_type": "location_query"}
                )
            
            # Search for college location in database
            college_data = await self.college_service.search_by_name(college_name)
            
            if not college_data:
                return ActionResponse(
                    response=f"I couldn't find location information for {college_name}. Could you please check the college name or try a different search?",
                    success=False,
                    metadata={"error": "college_not_found", "college_name": college_name}
                )
            
            # Format location response
            location = college_data.get('location', 'Location not specified')
            address = college_data.get('address', '')
            
            response_parts = [f"{college_name} is located in {location}."]
            if address:
                response_parts.append(f"Address: {address}")
            
            return ActionResponse(
                response=" ".join(response_parts),
                success=True,
                metadata={
                    "college_name": college_name,
                    "location": location,
                    "address": address,
                    "action_type": "location_info"
                }
            )
            
        except Exception as e:
            logger.error(f"Location info action error: {e}")
            return ActionResponse(
                response="I apologize, but I encountered an error while retrieving location information. Please try again.",
                success=False,
                metadata={"error": str(e)}
            )

class ActionFallback(BaseAction):
    """Fallback action for unknown intents"""
    
    def __init__(self):
        super().__init__("action_fallback")
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute fallback action"""
        response = random.choice(RESPONSE_TEMPLATES["fallback"])
        return ActionResponse(
            response=response,
            metadata={"fallback_triggered": True}
        )

class ActionRegistry:
    """Registry for managing dialogue actions"""
    
    def __init__(self, college_service: Optional[CollegeService]):
        self.actions = {
            "action_search_college": ActionSearchCollege(college_service),
            "action_get_admission_info": ActionGetAdmissionInfo(college_service),
            "action_get_location_info": ActionGetLocationInfo(college_service),
            "action_greet": ActionGreet(),
            "action_goodbye": ActionGoodbye(),
            "action_fallback": ActionFallback()
        }
        service_status = "enabled" if college_service else "demo mode"
        logger.info(f"Initialized action registry with {len(self.actions)} actions ({service_status})")
    
    async def execute_action(self, action_name: str, request: ActionRequest) -> ActionResponse:
        """Execute a specific action"""
        if action_name not in self.actions:
            logger.warning(f"Unknown action: {action_name}")
            action_name = "action_fallback"
        
        action = self.actions[action_name]
        logger.debug(f"Executing action: {action_name}")
        
        try:
            response = await action.execute(request)
            logger.debug(f"Action {action_name} completed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            fallback_action = self.actions["action_fallback"]
            return await fallback_action.execute(request)
    
    def get_available_actions(self) -> list:
        """Get list of available actions"""
        return list(self.actions.keys())