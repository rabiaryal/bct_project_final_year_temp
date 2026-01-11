"""Main Dialogue Manager"""

from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timedelta

from app.nlu import BERTIntentClassifier, RoBERTaEntityExtractor
from app.context.tracker import DialogueTracker
from app.policy.rule_policy import PolicyPlanner
from app.actions import ActionRegistry
from app.response.formatter import ResponseFormatter
from app.repositories.mongo_client import MongoRepository
from app.services.college_service import CollegeService
from app.schemas import ChatRequest, ChatResponse, ActionRequest
from app.utils.logger import get_logger
from app.utils.config import config

logger = get_logger(__name__)

class DialogueManager:
    """
    Main dialogue management system orchestrating:
    NLU → Context Tracking → Policy → Actions → Response Generation
    
    Follows Rasa/DeepPavlov architecture pattern
    """
    
    def __init__(self):
        self.intent_classifier = None
        self.entity_extractor = None
        self.policy_planner = None
        self.action_registry = None
        self.response_formatter = None
        self.mongo_repo = None
        self.college_service = None
        self.active_sessions: Dict[str, DialogueTracker] = {}
        self.session_timeout = timedelta(seconds=config.dialogue.session_timeout)
        
    async def initialize(self):
        """Initialize all dialogue components"""
        try:
            logger.info("Initializing Dialogue Manager...")
            
            # Initialize NLU components
            logger.info("Loading NLU models...")
            self.intent_classifier = BERTIntentClassifier()
            self.entity_extractor = RoBERTaEntityExtractor()
            
            # Initialize data layer
            logger.info("Connecting to MongoDB...")
            self.mongo_repo = MongoRepository()
            try:
                await self.mongo_repo.connect()
                logger.info("MongoDB connected successfully")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}. Running in fallback mode.")
                # Continue without MongoDB - system can still work with NLU only
                self.mongo_repo = None
            
            # Initialize services
            if self.mongo_repo:
                self.college_service = CollegeService(self.mongo_repo)
            else:
                self.college_service = None
                logger.warning("College service disabled - no MongoDB connection")
            
            # Initialize dialogue components
            self.policy_planner = PolicyPlanner()
            self.action_registry = ActionRegistry(self.college_service)
            self.response_formatter = ResponseFormatter()
            
            logger.info("Dialogue Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Dialogue Manager: {e}")
            raise
    
    async def shutdown(self):
        """Cleanup resources"""
        if self.mongo_repo:
            await self.mongo_repo.disconnect()
        self.active_sessions.clear()
        logger.info("Dialogue Manager shutdown complete")
    
    async def process_message(self, request: ChatRequest) -> ChatResponse:
        """
        Process incoming message through full dialogue pipeline:
        NLU → Tracker → Policy → Action → Response
        """
        try:
            # Get or create session
            session_id = request.session_id or self._generate_session_id()
            tracker = self._get_or_create_session(session_id)
            
            logger.info(f"Processing message: '{request.message}' in session: {session_id}")
            
            # Step 1: NLU Processing
            logger.info("Step 1: Running NLU...")
            intent, intent_confidence, intent_metadata = await self.intent_classifier.predict(request.message)
            entities = await self.entity_extractor.extract(request.message)
            
            logger.info(f"=== NLU RESULTS ===")
            logger.info(f"Intent: {intent}")
            logger.info(f"Intent Confidence: {intent_confidence}")
            logger.info(f"Intent Metadata: {intent_metadata}")
            logger.info(f"Entities: {entities}")
            logger.info(f"==================")
            
            # Step 2: Update Context
            logger.info("Step 2: Updating context...")
            old_context = tracker.get_current_state()
            logger.info(f"Previous Context: {old_context}")
            
            tracker.update_intent(intent, intent_confidence, intent_metadata)
            tracker.update_entities(entities)
            tracker.add_user_message(request.message)
            
            new_context = tracker.get_current_state()
            logger.info(f"Updated Context: {new_context}")
            
            # Step 3: Policy Decision
            logger.info("Step 3: Policy planning...")
            action_name = self.policy_planner.predict_action(tracker)
            logger.info(f"Selected Action: {action_name}")
            
            # Step 4: Execute Action
            logger.info("Step 4: Executing action...")
            action_result = await self._execute_action(action_name, tracker)
            logger.info(f"Action Result: {action_result}")
            
            # Step 5: Generate Response
            logger.info("Step 5: Generating response...")
            formatted_response = self.response_formatter.format_response(
                action_result, tracker.get_current_state()
            )
            logger.info(f"Formatted Response: {formatted_response}")
            
            # Step 6: Update tracker with bot response
            tracker.add_bot_message(formatted_response, action_name)
            
            # Create comprehensive response with debug info
            response = ChatResponse(
                message=formatted_response,
                session_id=session_id,
                intent=intent,
                entities=entities,
                confidence=intent_confidence,
                timestamp=datetime.now(),
                debug_info={
                    "intent_metadata": intent_metadata,
                    "previous_context": old_context,
                    "updated_context": new_context,
                    "selected_action": action_name,
                    "action_result": action_result,
                    "entity_extraction_details": {
                        "raw_entities": entities,
                        "entity_count": len(entities)
                    },
                    "session_info": {
                        "session_id": session_id,
                        "turn_count": len(tracker.messages)
                    }
                }
            )
            
            logger.info(f"=== DIALOGUE TURN COMPLETED ===")
            logger.info(f"Session: {session_id}")
            logger.info(f"Action: {action_name}")
            logger.info(f"Response: {formatted_response}")
            logger.info(f"===============================")
            
            return response
            
        except Exception as e:
            logger.error(f"Dialogue processing error: {e}")
            return ChatResponse(
                message="I apologize, but I encountered an error processing your message. Please try again.",
                session_id=session_id if 'session_id' in locals() else self._generate_session_id(),
                intent="error",
                entities={},
                confidence=0.0,
                timestamp=datetime.now(),
                debug_info={"error": str(e)}
            )
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session state"""
        if session_id in self.active_sessions:
            tracker = self.active_sessions[session_id]
            return tracker.to_dict()
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        health_status = {
            "dialogue_manager": "healthy",
            "active_sessions": len(self.active_sessions),
            "components": {}
        }
        
        # Check NLU components
        try:
            if self.intent_classifier and self.entity_extractor:
                health_status["components"]["nlu"] = "healthy"
            else:
                health_status["components"]["nlu"] = "not_initialized"
        except Exception as e:
            health_status["components"]["nlu"] = f"error: {e}"
        
        # Check MongoDB
        if self.mongo_repo:
            mongo_health = await self.mongo_repo.health_check()
            health_status["components"]["mongodb"] = mongo_health["status"]
        else:
            health_status["components"]["mongodb"] = "disconnected"
        
        # Check other components
        health_status["components"]["policy"] = "healthy" if self.policy_planner else "not_initialized"
        health_status["components"]["actions"] = "healthy" if self.action_registry else "not_initialized"
        health_status["components"]["response"] = "healthy" if self.response_formatter else "not_initialized"
        
        # Overall status
        if any("error" in str(status) for status in health_status["components"].values()):
            health_status["dialogue_manager"] = "degraded"
        elif any(status == "not_initialized" for status in health_status["components"].values()):
            health_status["dialogue_manager"] = "initializing"
        
        return health_status
    
    def _get_or_create_session(self, session_id: str) -> DialogueTracker:
        """Get existing session or create new one"""
        if session_id in self.active_sessions:
            tracker = self.active_sessions[session_id]
            # Check if session is not expired
            if datetime.now() - tracker.created_at < self.session_timeout:
                return tracker
            else:
                # Remove expired session
                del self.active_sessions[session_id]
        
        # Create new session
        tracker = DialogueTracker(session_id)
        self.active_sessions[session_id] = tracker
        logger.info(f"Created new session: {session_id}")
        return tracker
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"
    
    def _cleanup_old_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, tracker in self.active_sessions.items():
            if current_time - tracker.created_at > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.debug(f"Cleaned up expired session: {session_id}")
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def _execute_action(self, action_name: str, tracker: 'DialogueTracker') -> Dict[str, Any]:
        """Execute the specified action"""
        try:
            # Create action request
            action_request = ActionRequest(
                action=action_name,
                slots=tracker.slots.copy(),
                session_id=tracker.session_id,
                intent=tracker.intent,
                entities=tracker.entities
            )
            
            # Execute action through registry
            result = await self.action_registry.execute_action(action_name, action_request)
            
            return {
                "action": action_name,
                "response": result.response if hasattr(result, 'response') else str(result),
                "slots_updated": result.slots_updated if hasattr(result, 'slots_updated') else {},
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Action execution failed for {action_name}: {e}")
            return {
                "action": action_name,
                "response": "I apologize, but I encountered an issue processing your request.",
                "slots_updated": {},
                "success": False,
                "error": str(e)
            }

# Global dialogue manager instance
dialogue_manager = DialogueManager()