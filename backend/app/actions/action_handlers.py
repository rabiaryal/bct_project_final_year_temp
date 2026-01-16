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
            retrieval_data = request.retrieval_data
            
            # If we have retrieval data from our intelligent agent, use it
            if retrieval_data and retrieval_data.results:
                response_parts = []
                response_parts.append(f"I found information about colleges for your query: **{retrieval_data.query}**\n")
                response_parts.append(f"*Search Strategy: {retrieval_data.search_strategy}*\n")
                
                if retrieval_data.entities_found:
                    entity_str = ", ".join([f"{k}: {v}" for k, v in retrieval_data.entities_found.items()])
                    response_parts.append(f"*Entities found: {entity_str}*\n")
                
                response_parts.append("\n📋 **College Results:**\n")
                
                for i, result in enumerate(retrieval_data.results[:3], 1):  # Show top 3 results
                    response_parts.append(f"\n**{i}. {result.college_name}** (Similarity: {result.similarity_score:.2f})")
                    response_parts.append(f"   *{result.match_reason}*")
                    
                    # Extract key info from college_data
                    college_data = result.college_data
                    if location := college_data.get("location"):
                        response_parts.append(f"   📍 **Location:** {location}")
                    if college_type := college_data.get("type"):
                        response_parts.append(f"   🏫 **Type:** {college_type}")
                    if courses := college_data.get("courses"):
                        course_preview = courses[:100] + "..." if len(courses) > 100 else courses
                        response_parts.append(f"   📚 **Courses:** {course_preview}")
                    if website := college_data.get("website"):
                        response_parts.append(f"   🌐 **Website:** {website}")
                
                if len(retrieval_data.results) > 3:
                    response_parts.append(f"\n*... and {len(retrieval_data.results) - 3} more results*")
                
                response_parts.append("\n\nWould you like more details about any specific college?")
                
                return ActionResponse(
                    response="\n".join(response_parts),
                    slots_updated={"colleges_found": len(retrieval_data.results)},
                    metadata={"retrieval_used": True, "total_results": retrieval_data.total_results},
                    retrieval_results=retrieval_data.results
                )
            
            # Fallback to traditional search if no retrieval data
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
            
            # Search for college using traditional method
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
            retrieval_data = request.retrieval_data
            
            # If we have retrieval data, use it for admission information
            if retrieval_data and retrieval_data.results:
                response_parts = []
                response_parts.append(f"Here's admission information for your query: **{retrieval_data.query}**\n")
                
                if retrieval_data.entities_found:
                    entity_str = ", ".join([f"{k}: {v}" for k, v in retrieval_data.entities_found.items()])
                    response_parts.append(f"*Entities found: {entity_str}*\n")
                
                response_parts.append("\n📚 **Admission Information:**\n")
                
                for i, result in enumerate(retrieval_data.results[:2], 1):  # Show top 2 for admission details
                    college_data = result.college_data
                    response_parts.append(f"\n**{i}. {result.college_name}**")
                    
                    # Extract admission-specific information
                    if admission_process := college_data.get("admission_process"):
                        response_parts.append(f"   📋 **Process:** {admission_process}")
                    if eligibility := college_data.get("eligibility_criteria"):
                        response_parts.append(f"   ✅ **Eligibility:** {eligibility}")
                    if entrance_exam := college_data.get("entrance_exam"):
                        response_parts.append(f"   📝 **Entrance Exam:** {entrance_exam}")
                    if application_deadline := college_data.get("application_deadline"):
                        response_parts.append(f"   ⏰ **Deadline:** {application_deadline}")
                    if fees := college_data.get("fees"):
                        response_parts.append(f"   💰 **Fees:** {fees}")
                    if documents := college_data.get("required_documents"):
                        response_parts.append(f"   📄 **Documents:** {documents}")
                
                response_parts.append("\n\nWould you like specific details about any college's admission process?")
                
                return ActionResponse(
                    response="\n".join(response_parts),
                    metadata={"retrieval_used": True, "admission_info_provided": True},
                    retrieval_results=retrieval_data.results
                )
            
            # Fallback to traditional approach
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
            retrieval_data = request.retrieval_data
            slots = request.slots
            college_name = slots.get("COLLEGE_NAME") or slots.get("college_name")
            
            # If we have retrieval data with location info, use it
            if retrieval_data and retrieval_data.results:
                response_parts = []
                
                # Check if we have a specific college match
                specific_matches = []
                if college_name:
                    for result in retrieval_data.results:
                        if college_name.lower() in result.college_name.lower():
                            specific_matches.append(result)
                
                if specific_matches:
                    # Found specific college match
                    result = specific_matches[0]  # Take best match
                    college_data = result.college_data
                    location = college_data.get('location', 'Location not specified')
                    
                    response_parts.append(f"**{result.college_name}** is located in **{location}**.")
                    
                    # Add additional location details if available
                    if phone := college_data.get('phone'):
                        response_parts.append(f"📞 **Phone:** {phone}")
                    if website := college_data.get('website'):
                        response_parts.append(f"🌐 **Website:** {website}")
                        
                    response_parts.append("\nWould you like more information about this college?")
                    
                else:
                    # Show multiple location options
                    response_parts.append(f"I found several colleges related to your query:\n")
                    
                    for i, result in enumerate(retrieval_data.results[:3], 1):
                        college_data = result.college_data
                        location = college_data.get('location', 'Location not specified')
                        response_parts.append(f"**{i}. {result.college_name}** - {location}")
                    
                    response_parts.append("\nWhich college's location would you like to know more about?")
                
                return ActionResponse(
                    response="\n".join(response_parts),
                    success=True,
                    metadata={
                        "retrieval_used": True,
                        "colleges_found": len(retrieval_data.results),
                        "specific_match": bool(specific_matches),
                        "action_type": "location_info"
                    },
                    retrieval_results=retrieval_data.results
                )
            
            # Fallback if no retrieval data or college service search
            if not college_name:
                return ActionResponse(
                    response="Which college's location would you like to know about?",
                    success=False,
                    metadata={"requires": "college_name"}
                )
            
            # Try direct college service search if available
            if self.college_service:
                try:
                    college_data = await self.college_service.search_by_name(college_name)
                    if college_data:
                        location = college_data.get('Location', 'Location not specified')
                        name = college_data.get('Name', college_name)
                        
                        response_parts = [f"**{name}** is located in **{location}**."]
                        
                        if phone := college_data.get('ContactNumber'):
                            response_parts.append(f"📞 **Phone:** {phone}")
                        if email := college_data.get('Email'):
                            response_parts.append(f"📧 **Email:** {email}")
                            
                        response_parts.append("\nWould you like more information about this college?")
                        
                        return ActionResponse(
                            response="\n".join(response_parts),
                            success=True,
                            metadata={
                                "college_name": name,
                                "location": location,
                                "action_type": "location_info",
                                "source": "direct_search"
                            }
                        )
                except Exception as e:
                    logger.error(f"Direct search error: {e}")
            
            # Final fallback response
            return ActionResponse(
                response=f"I couldn't find specific location information for {college_name}. Could you please check the college name or try asking about a different college?",
                success=False,
                metadata={"error": "college_not_found", "college_name": college_name}
            )
            
        except Exception as e:
            logger.error(f"Location info action error: {e}")
            return ActionResponse(
                response="I apologize, but I encountered an error while retrieving location information. Please try again.",
                success=False,
                metadata={"error": str(e)}
            )


class ActionGetCollegeInfo(BaseAction):
    """Get general college information action"""
    
    def __init__(self, college_service: Optional[CollegeService]):
        super().__init__("action_get_college_info")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute general college info action"""
        try:
            retrieval_data = request.retrieval_data
            slots = request.slots
            college_name = slots.get("COLLEGE_NAME") or slots.get("college_name")
            
            # If we have retrieval data, use it
            if retrieval_data and retrieval_data.results:
                response_parts = []
                
                # Check for specific college match
                specific_matches = []
                if college_name:
                    for result in retrieval_data.results:
                        if college_name.lower() in result.college_name.lower():
                            specific_matches.append(result)
                
                if specific_matches:
                    # Found specific college match
                    result = specific_matches[0]
                    college_data = result.college_data
                    
                    response_parts.append(f"**{result.college_name}** Information:")
                    response_parts.append(f"🏫 **Location:** {college_data.get('location', 'Not specified')}")
                    
                    if phone := college_data.get('phone'):
                        response_parts.append(f"📞 **Phone:** {phone}")
                    if email := college_data.get('email'):
                        response_parts.append(f"📧 **Email:** {email}")
                    if website := college_data.get('website'):
                        response_parts.append(f"🌐 **Website:** {website}")
                    if departments := college_data.get('departments'):
                        if isinstance(departments, list):
                            dept_text = ", ".join(departments)
                        else:
                            dept_text = str(departments)
                        response_parts.append(f"🎓 **Departments:** {dept_text}")
                    
                    response_parts.append("\nWould you like specific information about programs, fees, or admission requirements?")
                    
                else:
                    # Show multiple college options
                    response_parts.append("I found several colleges. Here are the details:\n")
                    
                    for i, result in enumerate(retrieval_data.results[:3], 1):
                        college_data = result.college_data
                        response_parts.append(f"**{i}. {result.college_name}**")
                        response_parts.append(f"   📍 Location: {college_data.get('location', 'Not specified')}")
                        if phone := college_data.get('phone'):
                            response_parts.append(f"   📞 Phone: {phone}")
                        response_parts.append("")
                    
                    response_parts.append("Which college would you like to know more about?")
                
                return ActionResponse(
                    response="\n".join(response_parts),
                    success=True,
                    metadata={
                        "retrieval_used": True,
                        "colleges_found": len(retrieval_data.results),
                        "specific_match": bool(specific_matches),
                        "action_type": "college_info"
                    },
                    retrieval_results=retrieval_data.results
                )
            
            # Fallback if no retrieval data
            if not college_name:
                return ActionResponse(
                    response="Which college would you like to know about? Please provide the college name.",
                    success=False,
                    metadata={"requires": "college_name"}
                )
            
            return ActionResponse(
                response=f"I couldn't find information about {college_name}. Could you please check the college name or try asking about a different college?",
                success=False,
                metadata={"error": "college_not_found", "college_name": college_name}
            )
            
        except Exception as e:
            logger.error(f"College info action error: {e}")
            return ActionResponse(
                response="I apologize, but I encountered an error while retrieving college information. Please try again.",
                success=False,
                metadata={"error": str(e)}
            )


class ActionGetFeeInfo(BaseAction):
    """Get college fee information action"""
    
    def __init__(self, college_service: Optional[CollegeService]):
        super().__init__("action_get_fee_info")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute fee info action"""
        try:
            retrieval_data = request.retrieval_data
            slots = request.slots
            college_name = slots.get("COLLEGE_NAME") or slots.get("college_name")
            
            # If we have retrieval data, use it
            if retrieval_data and retrieval_data.results:
                response_parts = []
                
                # Check for specific college match
                specific_matches = []
                if college_name:
                    for result in retrieval_data.results:
                        if college_name.lower() in result.college_name.lower():
                            specific_matches.append(result)
                
                if specific_matches:
                    result = specific_matches[0]
                    college_data = result.college_data
                    
                    response_parts.append(f"**{result.college_name}** Fee Information:")
                    
                    # Check for fee information in various possible fields
                    fee_info = None
                    for field in ['fee', 'fees', 'tuition', 'cost', 'Fee', 'Fees']:
                        if field in college_data:
                            fee_info = college_data[field]
                            break
                    
                    if fee_info:
                        response_parts.append(f"💰 **Fee:** {fee_info}")
                    else:
                        response_parts.append("💰 **Fee information is not available in our database.")
                        response_parts.append("\nI recommend contacting the college directly for current fee information:")
                        
                        if phone := college_data.get('phone'):
                            response_parts.append(f"📞 **Phone:** {phone}")
                        if email := college_data.get('email'):
                            response_parts.append(f"📧 **Email:** {email}")
                    
                    response_parts.append(f"\n📍 **Location:** {college_data.get('location', 'Not specified')}")
                    response_parts.append("\nNote: Fees may vary by program and are subject to change. Please contact the college for the most current information.")
                    
                else:
                    response_parts.append("I found several colleges. Here's their information:\n")
                    
                    for i, result in enumerate(retrieval_data.results[:3], 1):
                        college_data = result.college_data
                        response_parts.append(f"**{i}. {result.college_name}**")
                        response_parts.append(f"   📍 Location: {college_data.get('location', 'Not specified')}")
                        if phone := college_data.get('phone'):
                            response_parts.append(f"   📞 Contact: {phone}")
                        response_parts.append("")
                    
                    response_parts.append("Please specify which college's fee information you need.")
                
                return ActionResponse(
                    response="\n".join(response_parts),
                    success=True,
                    metadata={
                        "retrieval_used": True,
                        "colleges_found": len(retrieval_data.results),
                        "specific_match": bool(specific_matches),
                        "action_type": "fee_info"
                    },
                    retrieval_results=retrieval_data.results
                )
            
            # Fallback if no retrieval data
            if not college_name:
                return ActionResponse(
                    response="Which college's fee information would you like to know? Please provide the college name.",
                    success=False,
                    metadata={"requires": "college_name"}
                )
            
            return ActionResponse(
                response=f"I couldn't find fee information for {college_name}. Please contact the college directly or check their website for current fee details.",
                success=False,
                metadata={"error": "fee_info_not_found", "college_name": college_name}
            )
            
        except Exception as e:
            logger.error(f"Fee info action error: {e}")
            return ActionResponse(
                response="I apologize, but I encountered an error while retrieving fee information. Please try again.",
                success=False,
                metadata={"error": str(e)}
            )


class ActionAskCollegeName(BaseAction):
    """Ask user to provide college name when not specified"""
    
    def __init__(self, college_service: Optional[CollegeService] = None):
        super().__init__("action_ask_college_name")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute ask college name action"""
        try:
            retrieval_data = request.retrieval_data
            
            # If we have retrieval data, show available options
            if retrieval_data and retrieval_data.results:
                response_parts = ["I found several colleges that might match your query:\n"]
                
                for i, result in enumerate(retrieval_data.results[:5], 1):
                    college_data = result.college_data
                    location = college_data.get('location', 'Location not specified')
                    response_parts.append(f"**{i}. {result.college_name}** - {location}")
                
                response_parts.append("\nWhich college would you like to know more about? Please specify the college name.")
                
                return ActionResponse(
                    response="\n".join(response_parts),
                    success=True,
                    metadata={
                        "retrieval_used": True,
                        "colleges_found": len(retrieval_data.results),
                        "action_type": "ask_college_name"
                    },
                    retrieval_results=retrieval_data.results
                )
            
            # Default response when no retrieval data
            return ActionResponse(
                response="Which college would you like to know about? Please provide the specific college name.",
                success=True,
                metadata={"action_type": "ask_college_name"}
            )
            
        except Exception as e:
            logger.error(f"Ask college name action error: {e}")
            return ActionResponse(
                response="Could you please specify which college you're interested in?",
                success=True,
                metadata={"error": str(e)}
            )


class ActionSocialResponse(BaseAction):
    """Handle social interactions like greetings, goodbyes, thanks"""
    
    def __init__(self):
        super().__init__("action_social_response")
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute social response action"""
        try:
            # Get intent to determine type of social response
            intent = getattr(request, 'intent', 'unknown')
            
            social_responses = {
                "Greeting": [
                    "Hello! I'm here to help you with college information. What would you like to know?",
                    "Hi! I can assist you with college details, admission info, fees, and more. How can I help?",
                    "Welcome! Ask me anything about colleges, programs, locations, or admission processes."
                ],
                "Goodbye": [
                    "Goodbye! Feel free to ask if you need more college information.",
                    "Take care! I'm here whenever you need help with college queries.",
                    "Bye! Good luck with your college search!"
                ],
                "Thank_you": [
                    "You're welcome! Happy to help with your college search.",
                    "Glad I could help! Let me know if you need anything else.",
                    "My pleasure! Feel free to ask more questions anytime."
                ],
                "Affirmation": [
                    "Great! What would you like to know next?",
                    "Perfect! How else can I assist you?",
                    "Excellent! Any other college information you need?"
                ],
                "Negation": [
                    "No problem! Let me know if you need help with anything else.",
                    "That's okay! Is there something else I can help you with?",
                    "Alright! Feel free to ask about other colleges or topics."
                ]
            }
            
            responses = social_responses.get(intent, ["I'm here to help with college information. What can I do for you?"])
            response = random.choice(responses)
            
            return ActionResponse(
                response=response,
                success=True,
                metadata={"action_type": "social", "intent": intent}
            )
            
        except Exception as e:
            logger.error(f"Social response action error: {e}")
            return ActionResponse(
                response="Hello! How can I help you with college information today?",
                success=True,
                metadata={"error": str(e)}
            )


class ActionRecommendCollege(BaseAction):
    """Recommend colleges based on criteria"""
    
    def __init__(self, college_service: Optional[CollegeService] = None):
        super().__init__("action_recommend_college")
        self.college_service = college_service
    
    async def execute(self, request: ActionRequest) -> ActionResponse:
        """Execute college recommendation action"""
        try:
            retrieval_data = request.retrieval_data
            slots = request.slots
            
            # If we have retrieval data, use it for recommendations
            if retrieval_data and retrieval_data.results:
                response_parts = []
                response_parts.append("🎓 **College Recommendations Based on Your Query:**\\n")
                
                if retrieval_data.entities_found:
                    entity_str = ", ".join([f"{k}: {v}" for k, v in retrieval_data.entities_found.items()])
                    response_parts.append(f"*Based on: {entity_str}*\\n")
                
                # Show top recommendations with scoring explanation
                for i, result in enumerate(retrieval_data.results[:5], 1):
                    college_data = result.college_data
                    
                    # Calculate recommendation score (you can enhance this logic)
                    base_score = result.similarity_score if hasattr(result, 'similarity_score') else 0.8
                    
                    response_parts.append(f"\\n**{i}. {result.college_name}** ⭐ {base_score:.1f}/1.0")
                    
                    if location := college_data.get('location'):
                        response_parts.append(f"   📍 **Location:** {location}")
                    if college_type := college_data.get('type'):
                        response_parts.append(f"   🏫 **Type:** {college_type}")
                    if programs := college_data.get('programs') or college_data.get('departments'):
                        if isinstance(programs, list):
                            programs_text = ", ".join(programs[:3])
                        else:
                            programs_text = str(programs)[:100]
                        response_parts.append(f"   📚 **Programs:** {programs_text}")
                    if fees := college_data.get('fees'):
                        response_parts.append(f"   💰 **Fees:** {fees}")
                
                response_parts.append("\\n\\n💡 **Why these recommendations?**")
                response_parts.append("- Based on semantic similarity to your query")
                response_parts.append("- Matched your specified criteria") 
                response_parts.append("- Relevance to detected entities\\n")
                response_parts.append("Would you like detailed information about any of these colleges?")
                
                return ActionResponse(
                    response="\\n".join(response_parts),
                    success=True,
                    metadata={
                        "retrieval_used": True,
                        "recommendations_count": len(retrieval_data.results),
                        "action_type": "recommendation"
                    },
                    retrieval_results=retrieval_data.results
                )
            
            # Fallback for general recommendations
            preference_hints = []
            if location := slots.get("LOCATION") or slots.get("location"):
                preference_hints.append(f"in {location}")
            if program := slots.get("PROGRAM") or slots.get("program"):
                preference_hints.append(f"for {program}")
            
            if preference_hints:
                criteria = " ".join(preference_hints)
                response = f"I'd be happy to recommend colleges {criteria}. However, I need access to the database to provide specific recommendations. Could you tell me more about your preferences like budget, preferred location, or specific programs you're interested in?"
            else:
                response = "I'd love to help you find the right college! Could you tell me more about what you're looking for? For example:\\n\\n📍 **Location preference** (city, state)\\n📚 **Field of study** (engineering, medical, etc.)\\n💰 **Budget range**\\n🏫 **College type** (government, private)\\n\\nThis will help me give you better recommendations!"
            
            return ActionResponse(
                response=response,
                success=True,
                metadata={"action_type": "recommendation_request", "requires_criteria": True}
            )
            
        except Exception as e:
            logger.error(f"Recommendation action error: {e}")
            return ActionResponse(
                response="I'd be happy to help you find the right college! Could you provide some details about your preferences?",
                success=True,
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
            # Query Class: INFO_LOOKUP
            "action_get_college_info": ActionGetCollegeInfo(college_service),
            "action_get_location_info": ActionGetLocationInfo(college_service),
            "action_get_fee_info": ActionGetFeeInfo(college_service),
            
            # Query Class: SEARCH  
            "action_search_college": ActionSearchCollege(college_service),
            
            # Query Class: RECOMMENDATION
            "action_recommend_college": ActionRecommendCollege(college_service),
            
            # Query Class: ADMISSION_FLOW
            "action_get_admission_info": ActionGetAdmissionInfo(college_service),
            
            # Query Class: SOCIAL
            "action_social_response": ActionSocialResponse(),
            "action_greet": ActionGreet(),  # Legacy support
            "action_goodbye": ActionGoodbye(),  # Legacy support
            
            # Query Class: FALLBACK
            "action_ask_college_name": ActionAskCollegeName(college_service),
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