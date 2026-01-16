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
from app.services.college_retrieval_agent import CollegeRetrievalAgent
from app.execution import execution_system
from app.policy.query_orchestrator import query_orchestrator
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
        self.retrieval_agent = None  # New intelligent retrieval agent
        self.execution_system = execution_system  # New execution system
        self.query_orchestrator = query_orchestrator  # Complete pipeline orchestrator
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
                
                # Initialize intelligent retrieval agent
                logger.info("Initializing Intelligent Retrieval Agent...")
                try:
                    self.retrieval_agent = CollegeRetrievalAgent(mongo_repo=self.mongo_repo)
                    await self.retrieval_agent.initialize()
                    logger.info("Intelligent Retrieval Agent initialized successfully")
                except Exception as e:
                    logger.warning(f"Retrieval agent initialization failed: {e}. Using fallback service.")
                    self.retrieval_agent = None
            else:
                self.college_service = None
                self.retrieval_agent = None
                logger.warning("Services disabled - no MongoDB connection")
            
            # Initialize dialogue components
            self.policy_planner = PolicyPlanner()
            self.action_registry = ActionRegistry(self.college_service)
            self.response_formatter = ResponseFormatter()
            
            # Initialize execution system
            logger.info("Initializing Execution System...")
            try:
                await self.execution_system.initialize()
                logger.info("Execution System initialized successfully")
            except Exception as e:
                logger.warning(f"Execution system initialization failed: {e}. Using fallback mode.")
            
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
        NLU → Context → Query Class → Entity Roles → Policy → Execution → Response
        """
        start_time = datetime.now()
        try:
            # Get or create session
            session_id = request.session_id or self._generate_session_id()
            tracker = self._get_or_create_session(session_id)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"🚀 NEW DIALOGUE TURN STARTING")
            logger.info(f"{'='*80}")
            logger.info(f"💬 Message: '{request.message}'")
            logger.info(f"📱 Session: {session_id}")
            logger.info(f"⏰ Timestamp: {start_time.strftime('%H:%M:%S')}")
            logger.info(f"{'='*80}")
            
            # Step 1: NLU Processing
            logger.info(f"\n{'='*80}")
            logger.info(f"🧠 STAGE 1: NLU PROCESSING")
            logger.info(f"{'='*80}")
            
            intent, intent_confidence, intent_metadata = await self.intent_classifier.predict(request.message)
            entity_list = self.entity_extractor.predict_with_confidence(request.message, threshold=0.3)
            
            nlu_results = {
                "intent": intent,
                "intent_confidence": intent_confidence, 
                "intent_metadata": intent_metadata,
                "entities": entity_list
            }
            
            logger.info(f"📍 Intent: {intent} (confidence: {intent_confidence:.3f})")
            logger.info(f"🏷️ Entities: {len(entity_list)} found")
            for entity in entity_list:
                logger.info(f"   - {entity['type']}: '{entity['text']}' (conf: {entity['confidence']:.3f})")
            
            # Step 2: Use NEW Complete Pipeline with Execution System
            logger.info(f"\n{'='*80}")
            logger.info(f"🎯 STAGE 2: COMPLETE PIPELINE PROCESSING")
            logger.info(f"{'='*80}")
            
            try:
                # Process through complete pipeline: NLU → Context → Query Class → Entity Roles → Policy → Execution
                pipeline_result = await self.query_orchestrator.process_complete_query(
                    conversation_id=session_id,
                    user_input=request.message,
                    nlu_results=nlu_results
                )
                
                if pipeline_result["status"] == "SUCCESS":
                    logger.info(f"🎯 Query Class: {pipeline_result['query_class']}")
                    logger.info(f"🏷️ Entity Roles: {len(pipeline_result['structured_entities'])} structured")
                    logger.info(f"📋 Policy Decision: {pipeline_result['policy_decision']['decision']['action']}")
                    
                    policy_decision = pipeline_result["policy_decision"]
                    
                    # Step 3: Execute Database Queries if needed
                    execution_result = None
                    if policy_decision["decision"]["action"] == "EXECUTE_QUERY":
                        logger.info(f"\n{'='*80}")
                        logger.info(f"🚀 STAGE 3: DATABASE EXECUTION")
                        logger.info(f"{'='*80}")
                        
                        try:
                            execution_result = await self.execution_system.execute_policy_decision(policy_decision)
                            
                            logger.info(f"✅ Execution Status: {execution_result['status']}")
                            logger.info(f"📊 Results Count: {execution_result.get('results_count', 0)}")
                            logger.info(f"🎯 Criteria: {execution_result['criteria_type']}")
                            
                            if execution_result.get('results'):
                                logger.info("🏫 Top Results:")
                                for i, result in enumerate(execution_result['results'][:3], 1):
                                    name = result.get('name', 'Unknown')
                                    location = result.get('location', 'Unknown')
                                    logger.info(f"   {i}. {name} - {location}")
                        
                        except Exception as e:
                            logger.error(f"❌ Execution failed: {e}")
                            execution_result = {
                                "status": "ERROR",
                                "error": str(e),
                                "results_count": 0,
                                "results": []
                            }
                    
                    # Step 4: Update Context with pipeline results
                    logger.info(f"\n{'='*80}")
                    logger.info(f"📝 STAGE 4: CONTEXT UPDATE")
                    logger.info(f"{'='*80}")
                    
                    # Convert entities for tracker compatibility
                    entities_simplified = {}
                    for entity in entity_list:
                        entity_type = entity['type']
                        entity_text = entity['text']
                        
                        if entity_type not in entities_simplified:
                            entities_simplified[entity_type] = []
                        entities_simplified[entity_type].append(entity_text)
                    
                    # Flatten single-item lists for compatibility
                    for etype, eval_list in entities_simplified.items():
                        if len(eval_list) == 1:
                            entities_simplified[etype] = eval_list[0]
                    
                    tracker.update_intent(intent, intent_confidence, intent_metadata)
                    tracker.update_entities(entities_simplified)
                    tracker.add_user_message(request.message)
                    
                    logger.info(f"🔄 Session: {session_id}")
                    logger.info(f"📊 Slots Updated: {tracker.slots}")
                    logger.info(f"💬 Turn Count: {len(tracker.messages)}")
                    
                    # Step 5: Generate Response
                    logger.info(f"\n{'='*80}")
                    logger.info(f"💬 STAGE 5: RESPONSE GENERATION")
                    logger.info(f"{'='*80}")
                    
                    if execution_result and execution_result.get('results'):
                        # Use execution results for response
                        response_text = self._format_execution_response(execution_result)
                        action_taken = f"EXECUTED_{execution_result['criteria_type']}"
                        
                    elif policy_decision["decision"]["action"] != "EXECUTE_QUERY":
                        # Handle non-query responses
                        response_text = policy_decision["decision"].get("reason", "I understand your request.")
                        action_taken = policy_decision["decision"]["action"]
                        
                    else:
                        # Fallback response
                        response_text = "I understand you're looking for college information. Could you provide more specific criteria?"
                        action_taken = "PROVIDE_HELP"
                    
                    logger.info(f"📝 Response Type: {action_taken}")
                    logger.info(f"📏 Response Length: {len(response_text)} characters")
                    
                else:
                    # Pipeline processing failed
                    logger.error(f"❌ Pipeline processing failed: {pipeline_result.get('error', 'Unknown error')}")
                    response_text = "I'm having trouble processing your request. Could you please rephrase it?"
                    action_taken = "CLARIFICATION_REQUEST"
                
            except Exception as e:
                logger.error(f"❌ Pipeline processing error: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to legacy processing
                response_text = "I'm sorry, I'm having technical difficulties. Please try again."
                action_taken = "ERROR_FALLBACK"
            
            # Store response in tracker
            tracker.add_bot_message(response_text)
            
            # Step 6: Final Response
            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 DIALOGUE TURN COMPLETED")
            logger.info(f"{'='*80}")
            logger.info(f"📱 Session: {session_id}")
            logger.info(f"🎬 Final Action: {action_taken}")
            logger.info(f"💬 Response Sent: {len(response_text)} chars")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⏱️  Processing Time: {processing_time:.3f}s")
            logger.info(f"{'='*80}\n")
            
            # Create response
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                intent=intent,
                entities=entities_simplified if 'entities_simplified' in locals() else {},
                confidence=intent_confidence,
                timestamp=datetime.now(),
                debug_info={
                    "intent_metadata": intent_metadata,
                    "action_taken": action_taken,
                    "processing_time": processing_time,
                    "pipeline_used": "NEW_EXECUTION_PIPELINE"
                }
            )
            
        except Exception as e:
            logger.error(f"Dialogue processing error: {e}")
            import traceback
            traceback.print_exc()
            
            return ChatResponse(
                message="I apologize, but I encountered an error processing your message. Please try again.",
                session_id=session_id if 'session_id' in locals() else self._generate_session_id(),
                intent="error",
                entities={},
                confidence=0.0,
                timestamp=datetime.now(),
                debug_info={"error": str(e)}
            )
    
    def _format_execution_response(self, execution_result: Dict[str, Any]) -> str:
        """Format execution results into user-friendly response"""
        try:
            status = execution_result.get('status', 'ERROR')
            criteria_type = execution_result.get('criteria_type', 'UNKNOWN')
            results = execution_result.get('results', [])
            results_count = execution_result.get('results_count', 0)
            
            if status == "ERROR":
                return "I'm sorry, I encountered an issue while searching for colleges. Please try rephrasing your query."
            
            if results_count == 0:
                if criteria_type == "LOCATION":
                    return "I couldn't find any colleges matching your location criteria. Could you try a different location or check the spelling?"
                elif criteria_type == "FEE":
                    return "I couldn't find colleges matching your fee requirements. You might want to adjust your budget range."
                elif criteria_type == "LOCATION_AND_FEE":
                    return "I couldn't find colleges that match both your location and fee criteria. Try expanding your search criteria."
                else:
                    return "I couldn't find any colleges matching your criteria. Could you provide more details?"
            
            # Generate response based on criteria type
            if criteria_type == "LOCATION":
                location_summary = execution_result.get('location_summary', {})
                response_parts = [
                    f"I found {results_count} colleges matching your location criteria:"
                ]
                
            elif criteria_type == "FEE":
                fee_summary = execution_result.get('fee_summary', {})
                avg_fee = fee_summary.get('average_fee', 0)
                response_parts = [
                    f"I found {results_count} colleges matching your fee criteria:"
                ]
                if avg_fee > 0:
                    response_parts.append(f"Average fees: NPR {avg_fee:,.0f}")
                
            elif criteria_type == "LOCATION_AND_FEE":
                combined_insights = execution_result.get('combined_insights', {})
                response_parts = [
                    f"I found {results_count} colleges matching both your location and fee criteria:"
                ]
                
                recommendation = combined_insights.get('recommendation', '')
                if recommendation:
                    response_parts.append(f"\n📋 {recommendation}")
                
            else:
                response_parts = [f"Here are {results_count} colleges I found:"]
            
            # Add top results
            response_parts.append("\n🏫 Top Results:")
            for i, college in enumerate(results[:5], 1):
                name = college.get('name', 'Unknown College')
                location = college.get('location', 'Unknown Location')
                fees = college.get('fees', 'Contact college')
                
                if criteria_type == "LOCATION":
                    response_parts.append(f"{i}. {name} - {location}")
                elif criteria_type == "FEE":
                    response_parts.append(f"{i}. {name} - Fees: {fees}")
                elif criteria_type == "LOCATION_AND_FEE":
                    response_parts.append(f"{i}. {name}")
                    response_parts.append(f"   📍 Location: {location}")
                    response_parts.append(f"   💰 Fees: {fees}")
                else:
                    response_parts.append(f"{i}. {name} - {location}")
            
            # Add helpful suggestions
            if results_count > 5:
                response_parts.append(f"\n... and {results_count - 5} more colleges.")
                response_parts.append("Would you like me to show more results or help you refine your search?")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error formatting execution response: {e}")
            return "I found some colleges for you, but I'm having trouble formatting the results. Please try your query again."
    
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
    
    async def _execute_action(self, action_name: str, tracker: 'DialogueTracker', retrieval_results: Dict = None) -> Dict[str, Any]:
        """Execute the specified action with optional retrieval results"""
        try:
            # Create action request
            action_request = ActionRequest(
                action=action_name,
                slots=tracker.slots.copy(),
                session_id=tracker.session_id
            )
            
            # Add retrieval results to action context if available
            if retrieval_results and retrieval_results.get('retrieved_results'):
                from app.schemas import RetrievalData, RetrievalResult
                
                # Convert raw retrieval results to schema objects
                retrieval_objects = []
                for result in retrieval_results.get('retrieved_results', []):
                    # Create comprehensive college_data dict from result
                    college_data = {
                        'name': result.get('name', 'Unknown'),
                        'location': result.get('location', 'Unknown'),
                        'programs': result.get('programs', []),
                        'fees': result.get('fees', 'Contact college'),
                        'ranking': result.get('ranking', 'Not ranked'),
                        'description': result.get('description', ''),
                        'type': result.get('type', ''),
                        'affiliation': result.get('affiliation', ''),
                        'established': result.get('established', ''),
                        'courses': result.get('courses', ''),
                        'website': result.get('website', ''),
                        'phone': result.get('phone', '')
                    }
                    
                    retrieval_obj = RetrievalResult(
                        college_name=result.get('name', 'Unknown'),
                        similarity_score=result.get('confidence', 0.0),
                        college_data=college_data,
                        match_reason=f"{result.get('source', 'semantic')} search match"
                    )
                    retrieval_objects.append(retrieval_obj)
                
                # Create RetrievalData object
                retrieval_data = RetrievalData(
                    query=tracker.messages[-1] if tracker.messages else "",  # Latest user message
                    results=retrieval_objects,
                    entities_found=tracker.entities,
                    search_strategy=retrieval_results.get('policy', 'unknown'),
                    total_results=len(retrieval_objects)
                )
                
                action_request.retrieval_data = retrieval_data
            
            # Execute action through registry
            result = await self.action_registry.execute_action(action_name, action_request)
            
            return {
                "action": action_name,
                "response": result.response if hasattr(result, 'response') else str(result),
                "slots_updated": result.slots_updated if hasattr(result, 'slots_updated') else {},
                "retrieval_used": retrieval_results is not None,
                "retrieval_count": len(retrieval_results.get('retrieved_results', [])) if retrieval_results else 0,
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