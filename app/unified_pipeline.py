"""
Unified Pipeline Controller
Implements the architecture:

User → Intent Detector → [Greeting? → Fast Response]
                      → Entity Extractor → Conversation State Manager
                      → [Direct Answer | Recommendation | Comparison] → Response Generator
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import os
import uuid

# Import core components
from core.greetings.greeting_handler import GreetingHandler

# Try XGBoost recommender (new)
try:
    from core.recommendation.xgboost_recommender import (
        XGBoostRecommender, 
        ConversationState,
        ConversationStateManager as XGBStateManager,
        create_xgboost_recommender
    )
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Fallback to old state manager if XGBoost not available
if not XGBOOST_AVAILABLE:
    from core.state.state_manager import ConversationStateManager

# Try FAISS retriever for fast direct answers
try:
    from core.qa.faiss_retriever import FAISSDirectAnswerSystem
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Try Structured Retriever for accurate exact matching
try:
    from core.qa.structured_retriever import StructuredRetriever
    STRUCTURED_RETRIEVER_AVAILABLE = True
except ImportError:
    STRUCTURED_RETRIEVER_AVAILABLE = False

# Try hierarchical intent detector first
try:
    from core.intent.hierarchical_intent_detector import HierarchicalIntentDetector
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False

# Try entity extractor
try:
    from core.intent.entity_extractor import TransformerEntityExtractor
    ENTITY_EXTRACTOR_AVAILABLE = True
except ImportError:
    ENTITY_EXTRACTOR_AVAILABLE = False

# Fallback detector
from core.intent.safe_intent_detector import SafeIntentDetector


class UnifiedPipelineController:
    """
    Unified Pipeline Controller implementing the architecture:
    
    1. Intent Detection (greeting handled immediately)
    2. Entity Extraction
    3. Conversation State Management (context memory, slot filling)
    4. Route to: Direct Answer | Recommendation | Comparison
    5. Response Generation
    """
    
    def __init__(self, df: pd.DataFrame, data_path: str = 'full_data.json'):
        print("🏗️ Initializing Unified Pipeline Controller...")
        
        self.df = df
        self.data_path = data_path
        
        # Core components (lazy loading)
        self._intent_detector = None
        self._entity_extractor = None
        self._greeting_handler = None
        self._state_manager = None
        self._xgboost_recommender = None
        self._faiss_system = None
        self._structured_retriever = None
        
        # Check for models
        self._use_hierarchical = HIERARCHICAL_AVAILABLE and os.path.exists('./models/primary_intent_model')
        self._use_entity_extractor = ENTITY_EXTRACTOR_AVAILABLE and os.path.exists('./models/entity_model')
        self._use_faiss = FAISS_AVAILABLE
        self._use_structured = STRUCTURED_RETRIEVER_AVAILABLE
        self._use_xgboost = XGBOOST_AVAILABLE
        
        if self._use_hierarchical:
            print("🔥 Hierarchical Intent Detection enabled")
        else:
            print("🔒 Keyword-based Intent Detection (Fallback)")
        
        if self._use_entity_extractor:
            print("🏷️  Transformer Entity Extraction enabled")
        else:
            print("🔒 Basic Entity Extraction (Fallback)")
        
        if self._use_structured:
            print("🎯 Structured Retriever enabled (Exact Match Priority)")
        
        if self._use_faiss:
            print("⚡ FAISS Direct Answer System enabled (Semantic Fallback)")
        
        if self._use_xgboost:
            print("🌲 XGBoost Recommendation System enabled")
        
        print("✅ Unified Pipeline Controller ready")
    
    # ==================== Lazy Loading Properties ====================
    
    @property
    def intent_detector(self):
        """Load hierarchical intent detector"""
        if self._intent_detector is None:
            if self._use_hierarchical:
                try:
                    print("🔥 Loading Hierarchical Intent Detector...")
                    self._intent_detector = HierarchicalIntentDetector(
                        primary_model_path='./models/primary_intent_model',
                        sub_model_path='./models/sub_intent_model'
                    )
                    if self._intent_detector.primary_ready:
                        print("✅ Hierarchical Intent Detector loaded")
                    else:
                        raise Exception("Primary model not ready")
                except Exception as e:
                    print(f"⚠️ Hierarchical detector failed: {e}")
                    print("🔒 Falling back to keyword-based detector...")
                    self._intent_detector = SafeIntentDetector()
            else:
                print("🔒 Loading keyword-based intent detector...")
                self._intent_detector = SafeIntentDetector()
        return self._intent_detector
    
    @property
    def entity_extractor(self):
        """Load entity extractor"""
        if self._entity_extractor is None:
            if self._use_entity_extractor:
                try:
                    print("🏷️  Loading Transformer Entity Extractor...")
                    self._entity_extractor = TransformerEntityExtractor(
                        model_path='./models/entity_model'
                    )
                    print("✅ Entity Extractor loaded")
                except Exception as e:
                    print(f"⚠️ Entity extractor failed: {e}")
                    self._entity_extractor = None
            else:
                self._entity_extractor = None
        return self._entity_extractor
    
    @property
    def greeting_handler(self) -> GreetingHandler:
        """Load greeting handler"""
        if self._greeting_handler is None:
            self._greeting_handler = GreetingHandler()
        return self._greeting_handler
    
    @property
    def state_manager(self):
        """Load conversation state manager (XGBoost-based)"""
        if self._state_manager is None:
            if self._use_xgboost:
                self._state_manager = XGBStateManager()
            else:
                from core.state.state_manager import ConversationStateManager as OldStateManager
                self._state_manager = OldStateManager(max_history=10)
        return self._state_manager
    
    @property
    def xgboost_recommender(self):
        """Load XGBoost recommender"""
        if self._xgboost_recommender is None:
            if self._use_xgboost:
                try:
                    print("🌲 Loading XGBoost Recommender...")
                    self._xgboost_recommender = create_xgboost_recommender(self.data_path)
                    print("✅ XGBoost recommender loaded")
                except Exception as e:
                    print(f"⚠️ XGBoost recommender failed: {e}")
                    self._xgboost_recommender = "FAILED"
            else:
                self._xgboost_recommender = "FAILED"
        return self._xgboost_recommender
    
    @property
    def faiss_system(self):
        """Load FAISS direct answer system"""
        if self._faiss_system is None:
            if self._use_faiss:
                try:
                    print("⚡ Loading FAISS Direct Answer System...")
                    self._faiss_system = FAISSDirectAnswerSystem(data_path=self.data_path)
                    print("✅ FAISS system loaded")
                except Exception as e:
                    print(f"⚠️ FAISS system failed: {e}")
                    self._faiss_system = "FAILED"
            else:
                self._faiss_system = "FAILED"
        return self._faiss_system
    
    @property
    def structured_retriever(self):
        """Load Structured Retriever (exact match priority)"""
        if self._structured_retriever is None:
            if self._use_structured:
                try:
                    print("🎯 Loading Structured Retriever...")
                    # Pass FAISS system as fallback
                    faiss = self.faiss_system if self.faiss_system != "FAILED" else None
                    self._structured_retriever = StructuredRetriever(
                        data_path=self.data_path,
                        faiss_system=faiss
                    )
                    print("✅ Structured Retriever loaded")
                except Exception as e:
                    print(f"⚠️ Structured Retriever failed: {e}")
                    self._structured_retriever = "FAILED"
            else:
                self._structured_retriever = "FAILED"
        return self._structured_retriever
    
    # ==================== Main Entry Point ====================
    
    def handle_user_message(self, prompt: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point: Process user message through the unified pipeline
        
        Args:
            prompt: User's message
            session_id: Optional session ID for context tracking (creates new if not provided)
            
        Returns:
            Dict with response and metadata
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        print(f"\n🔄 Processing: '{prompt}' (session: {session_id[:8]}...)")
        
        # ===== Step 1: Intent Detection =====
        intent_result = self._detect_intent(prompt)
        primary_intent = intent_result['primary_intent']
        sub_intent = intent_result.get('sub_intent')
        full_intent = intent_result['full_intent']
        confidence = intent_result['primary_confidence']
        
        print(f"🎯 Intent: {full_intent} ({confidence:.1%})")
        
        # ===== Step 2: Fast Greeting Handling =====
        if primary_intent == 'greeting':
            greeting_result = self.greeting_handler.handle(prompt)
            
            if not greeting_result['needs_further_processing']:
                # Pure greeting - respond immediately
                print("👋 Pure greeting - fast response")
                return self._finalize_result(
                    greeting_result,
                    intent_result,
                    {},
                    session_id
                )
            else:
                # Greeting + task - extract task and continue processing
                print(f"👋 Greeting + task: '{greeting_result['task_extracted']}'")
                prompt = greeting_result['task_extracted']
                # Re-detect intent for the task
                intent_result = self._detect_intent(prompt)
                primary_intent = intent_result['primary_intent']
                sub_intent = intent_result.get('sub_intent')
                full_intent = intent_result['full_intent']
        
        # ===== Step 3: Entity Extraction =====
        entities = self._extract_entities(prompt)
        # Post-process to fix common extraction errors (e.g., "10 lakh" as college_name)
        entities = self._post_process_entities(entities, prompt)
        print(f"🏷️  Entities: {entities}")
        
        # ===== Step 4: Context Resolution (Conversation State) =====
        context = self.state_manager.get_context(session_id)
        resolved_entities = self.state_manager.resolve_references(session_id, prompt, entities)
        
        if resolved_entities.get('_resolved_from_context'):
            print(f"🔗 Context resolution: Added '{resolved_entities.get('college_name')}' from history")
        
        # ===== Step 5: Route to Handler =====
        if primary_intent == 'recommendation':
            result = self._handle_recommendation(prompt, intent_result, resolved_entities, session_id)
        elif primary_intent == 'comparison':
            result = self._handle_comparison(prompt, intent_result, resolved_entities)
        elif primary_intent == 'direct_question':
            result = self._handle_direct_question(prompt, intent_result, resolved_entities)
        else:
            # Fallback to direct question handler
            result = self._handle_direct_question(prompt, intent_result, resolved_entities)
        
        # ===== Step 6: Update Conversation State =====
        self.state_manager.add_turn(
            session_id=session_id,
            user_message=prompt,
            bot_response=result.get('response', ''),
            intent=primary_intent,
            sub_intent=sub_intent,
            entities=entities
        )
        
        return self._finalize_result(result, intent_result, resolved_entities, session_id)
    
    # ==================== Intent & Entity Detection ====================
    
    def _detect_intent(self, prompt: str) -> Dict[str, Any]:
        """Detect intent using hierarchical detector with multi-intent support"""
        if hasattr(self.intent_detector, 'detect_multi_intent'):
            # Use multi-intent detection for direct questions
            # Lower threshold (0.10) to catch more intents
            result = self.intent_detector.detect_multi_intent(prompt, threshold=0.10)
            
            # Also check for keyword-based multi-intent (backup)
            # This catches cases where transformer is too confident in one intent
            keyword_intents = self._keyword_multi_intent_check(prompt)
            if keyword_intents and len(keyword_intents) > 1:
                # Merge keyword-detected intents with transformer results
                existing_intents = {s[0] for s in result.get('sub_intents', [])}
                for intent_name in keyword_intents:
                    if intent_name not in existing_intents:
                        # Add with moderate confidence
                        result['sub_intents'].append((intent_name, 0.5))
                
                if len(result.get('sub_intents', [])) > 1:
                    result['is_multi_intent'] = True
                    # Update full_intent
                    intent_names = [s[0] for s in result['sub_intents']]
                    result['full_intent'] = f"direct_question.[{'+'.join(intent_names)}]"
            
            return result
        elif hasattr(self.intent_detector, 'detect_intent'):
            return self.intent_detector.detect_intent(prompt)
        else:
            # Fallback format
            intent, confidence, details = self.intent_detector.detect_intent(prompt)
            return {
                'primary_intent': intent.lower(),
                'primary_confidence': confidence,
                'sub_intent': None,
                'sub_confidence': 0.0,
                'full_intent': intent.lower()
            }
    
    def _keyword_multi_intent_check(self, prompt: str) -> List[str]:
        """
        Keyword-based multi-intent detection backup.
        Catches cases where transformer is too confident in one intent.
        """
        prompt_lower = prompt.lower()
        detected = []
        
        intent_keywords = {
            'fee': ['fee', 'cost', 'price', 'tuition', 'charges'],
            'location': ['location', 'where', 'address', 'located', 'place'],
            'hostel': ['hostel', 'accommodation', 'dormitory', 'stay'],
            'admission': ['admission', 'apply', 'application', 'enroll', 'entrance'],
            'rating': ['rating', 'rank', 'ranking', 'rated', 'review'],
            'scholarship': ['scholarship', 'financial aid', 'fee waiver'],
            'cutoff': ['cutoff', 'cut-off', 'cut off'],
            'duration': ['duration', 'years', 'semester', 'how long'],
            'seats': ['seats', 'capacity', 'intake'],
            'contact': ['contact', 'phone', 'email', 'number'],
            'internship': ['internship', 'intern', 'training'],
            'placement': ['placement', 'job', 'career', 'salary'],
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                detected.append(intent)
        
        return detected
    
    def _extract_entities(self, prompt: str) -> Dict[str, Any]:
        """Extract entities from prompt"""
        if self.entity_extractor:
            return self.entity_extractor.extract_entities(prompt)
        else:
            # Basic fallback entity extraction
            return self._basic_entity_extraction(prompt)
    
    def _post_process_entities(self, entities: Dict, prompt: str) -> Dict:
        """Clean up and fix common entity extraction errors"""
        prompt_lower = prompt.lower()
        
        # Fix: "10 la" being extracted as college_name when it's actually a fee
        if entities.get('college_name'):
            college_name = entities['college_name'].lower()
            # Check if it looks like a fee amount (contains numbers + lakh/lac/k)
            if any(char.isdigit() for char in college_name):
                # This is likely a fee, not a college name
                entities['college_name'] = None
        
        # Extract fee from prompt if not already extracted
        if not entities.get('fee'):
            import re
            # Match patterns like "10 lakh", "5 lac", "500000", "5,00,000", "under 10 lakh"
            fee_patterns = [
                r'(\d+)\s*(?:lakh|lac|lakhs|lacs)',  # 10 lakh, 5 lac
                r'under\s*(\d+)\s*(?:lakh|lac|lakhs|lacs)',  # under 10 lakh
                r'below\s*(\d+)\s*(?:lakh|lac|lakhs|lacs)',  # below 5 lakh
                r'less\s*than\s*(\d+)\s*(?:lakh|lac|lakhs|lacs)',  # less than 10 lakh
                r'within\s*(\d+)\s*(?:lakh|lac|lakhs|lacs)',  # within 10 lakh
                r'budget\s*(?:of\s*)?(\d+)\s*(?:lakh|lac|lakhs|lacs)',  # budget 10 lakh
                r'(\d+)k',  # 500k
                r'rs\.?\s*(\d[\d,]*)',  # Rs. 500000 or Rs 5,00,000
            ]
            
            for pattern in fee_patterns:
                match = re.search(pattern, prompt_lower)
                if match:
                    fee_str = match.group(1).replace(',', '')
                    fee_value = int(fee_str)
                    
                    # Convert lakh to actual number
                    if 'lakh' in pattern or 'lac' in pattern:
                        fee_value = fee_value * 100000
                    elif 'k' in pattern:
                        fee_value = fee_value * 1000
                    
                    entities['fee'] = fee_value
                    break
        
        return entities
    
    def _basic_entity_extraction(self, prompt: str) -> Dict[str, Any]:
        """Basic keyword-based entity extraction fallback"""
        entities = {}
        prompt_lower = prompt.lower()
        
        # College names (basic list)
        college_keywords = [
            'pulchowk', 'thapathali', 'ncit', 'nec', 'kantipur', 'kathmandu engineering',
            'ioe', 'tu', 'tribhuvan', 'pokhara', 'paschimanchal', 'advanced college'
        ]
        for college in college_keywords:
            if college in prompt_lower:
                entities['college_name'] = college.title()
                break
        
        # College type
        if 'private' in prompt_lower:
            entities['college_type'] = 'private'
        elif 'public' in prompt_lower or 'government' in prompt_lower:
            entities['college_type'] = 'public'
        elif 'constituent' in prompt_lower:
            entities['college_type'] = 'constituent'
        
        # Course names
        courses = ['computer', 'civil', 'mechanical', 'electrical', 'electronics', 'architecture']
        for course in courses:
            if course in prompt_lower:
                entities['course_name'] = f"{course} engineering"
                break
        
        # Location
        locations = ['kathmandu', 'lalitpur', 'bhaktapur', 'pokhara', 'chitwan', 'dharan']
        for loc in locations:
            if loc in prompt_lower:
                entities['location'] = loc.title()
                break
        
        # Fee extraction
        import re
        fee_patterns = [
            (r'(\d+)\s*(?:lakh|lac|lakhs|lacs)', 100000),
            (r'(\d+)k', 1000),
            (r'rs\.?\s*(\d[\d,]*)', 1),
        ]
        for pattern, multiplier in fee_patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                fee_str = match.group(1).replace(',', '')
                entities['fee'] = int(fee_str) * multiplier
                break
        
        return entities
    
    # ==================== Handler Methods ====================
    
    def _handle_recommendation(self, prompt: str, intent_result: Dict, entities: Dict, 
                               session_id: str = None) -> Dict[str, Any]:
        """Handle recommendation requests using XGBoost recommender"""
        print("🌲 Taking XGBoost recommendation flow...")
        
        if self.xgboost_recommender == "FAILED":
            print("⚠️ XGBoost recommender not available, using direct question handler")
            return self._handle_direct_question(prompt, intent_result, entities)
        
        try:
            # Get conversation state for this session
            if session_id and self._use_xgboost:
                conv_state = self.state_manager.get_state(session_id)
            else:
                # Create a temporary state
                conv_state = ConversationState()
            
            # Get recommendations
            ranked_colleges = self.xgboost_recommender.recommend(
                query=prompt,
                conversation_state=conv_state,
                entities=entities,
                k=5,
                exclude_previous=True
            )
            
            if not ranked_colleges:
                return {
                    'type': 'no_results',
                    'response': "I couldn't find colleges matching your criteria. Could you try with different requirements?",
                    'pipeline': 'xgboost_recommendation_empty'
                }
            
            # Format response
            response = self._format_recommendation_response(ranked_colleges, prompt)
            
            # Get preferences summary
            prefs_summary = ""
            if session_id and self._use_xgboost:
                prefs_summary = self.state_manager.get_preferences_summary(session_id)
            
            return {
                'type': 'recommendations',
                'response': response,
                'colleges': ranked_colleges,
                'pipeline': 'xgboost_recommendation',
                'preferences': prefs_summary
            }
        except Exception as e:
            print(f"❌ XGBoost recommendation error: {e}")
            import traceback
            traceback.print_exc()
            return self._handle_direct_question(prompt, intent_result, entities)
    
    def _handle_comparison(self, prompt: str, intent_result: Dict, entities: Dict) -> Dict[str, Any]:
        """Handle comparison requests between colleges"""
        print("⚖️ Taking comparison flow...")
        
        # Extract colleges to compare
        colleges_to_compare = self._extract_comparison_colleges(prompt, entities)
        
        if len(colleges_to_compare) < 2:
            return {
                'type': 'comparison_incomplete',
                'response': "I need at least two colleges to compare. Could you specify which colleges you'd like to compare?",
                'pipeline': 'comparison_incomplete'
            }
        
        # Get data for each college using structured retriever
        comparison_data = []
        for college_name in colleges_to_compare[:2]:  # Compare first 2
            # Use structured retriever if available
            if self.structured_retriever != "FAILED":
                # Use the new get_college_info method for full data
                college_info = self.structured_retriever.get_college_info(college_name)
                if college_info:
                    comparison_data.append(college_info)
                else:
                    comparison_data.append({'name': college_name, 'data': {}})
            else:
                comparison_data.append({'name': college_name, 'data': {}})
        
        # Generate comparison response
        response = self._format_comparison_response(comparison_data, prompt)
        
        return {
            'type': 'comparison',
            'response': response,
            'colleges_compared': colleges_to_compare[:2],
            'pipeline': 'comparison'
        }
    
    def _handle_direct_question(self, prompt: str, intent_result: Dict, entities: Dict) -> Dict[str, Any]:
        """
        Handle direct questions using a 2-tier retrieval strategy:
        1. Structured Retriever (exact match from JSON) - Most accurate
        2. FAISS semantic search (fallback for vague queries)
        
        Supports multi-intent queries.
        """
        # Check for multi-intent
        sub_intents = intent_result.get('sub_intents', [])
        is_multi_intent = intent_result.get('is_multi_intent', False)
        dominant_sub_intent = intent_result.get('dominant_sub_intent', 'general_info')
        full_intent = intent_result['full_intent']
        
        if is_multi_intent:
            print(f"🎯 MULTI-INTENT detected: {[s[0] for s in sub_intents]}")
        else:
            print(f"❓ Question type: {dominant_sub_intent}")
        
        # ========== TIER 1: Structured Retriever (Exact Match) ==========
        if self.structured_retriever != "FAILED":
            if is_multi_intent and len(sub_intents) > 1:
                # Multi-intent: get exact answer for each intent
                print("🎯 Structured Retriever: Multi-intent exact match...")
                
                answers = []
                all_data = []
                intents_answered = []
                
                for sub_intent, confidence in sub_intents:
                    result = self.structured_retriever.get_exact_answer(
                        query=prompt,
                        intent=sub_intent,
                        entities=entities
                    )
                    
                    if result and result.get('confidence', 0) >= 0.5:
                        answers.append(f"**{sub_intent.title()}**: {result['answer']}")
                        all_data.append(result.get('data', {}))
                        intents_answered.append(sub_intent)
                
                if answers:
                    combined_answer = "\n\n".join(answers)
                    print(f"✅ Exact match found for {len(intents_answered)} intents")
                    
                    return {
                        'type': 'direct_answer',
                        'response': combined_answer,
                        'facts_used': all_data,
                        'pipeline': 'structured_exact_multi',
                        'sub_intents': sub_intents,
                        'intents_answered': intents_answered,
                        'is_multi_intent': True,
                        'confidence': 1.0
                    }
            else:
                # Single intent: get exact answer
                print("🎯 Structured Retriever: Single intent exact match...")
                
                result = self.structured_retriever.get_exact_answer(
                    query=prompt,
                    intent=dominant_sub_intent,
                    entities=entities
                )
                
                if result and result.get('confidence', 0) >= 0.5:
                    print(f"✅ Exact match found (source: {result.get('source')})")
                    
                    return {
                        'type': 'direct_answer',
                        'response': result['answer'],
                        'facts_used': [result.get('data', {})],
                        'pipeline': 'structured_exact',
                        'sub_intent': dominant_sub_intent,
                        'confidence': result['confidence'],
                        'needs_clarification': result.get('needs_clarification', False)
                    }
                
                print("⚠️ No exact match found, falling back to semantic search...")
        
        # ========== TIER 2: FAISS Semantic Search ==========
        if self.faiss_system != "FAILED":
            if is_multi_intent and len(sub_intents) > 1:
                print("⚡ FAISS: Multi-intent semantic search...")
                
                result = self.faiss_system.get_multi_intent_answer(
                    query=prompt,
                    sub_intents=sub_intents,
                    entities=entities,
                    k_per_intent=3
                )
                
                intents_answered = result.get('intents_answered', [])
                print(f"📖 Answered {len(intents_answered)} intents: {intents_answered}")
                
                return {
                    'type': 'direct_answer',
                    'response': result['answer'],
                    'facts_used': result.get('facts', []),
                    'pipeline': 'faiss_multi_intent',
                    'sub_intents': sub_intents,
                    'intents_answered': intents_answered,
                    'is_multi_intent': True,
                    'confidence': result['confidence']
                }
            else:
                print("⚡ FAISS: Single intent semantic search...")
                
                result = self.faiss_system.get_answer(
                    query=prompt,
                    intent=dominant_sub_intent,
                    entities=entities,
                    k=5
                )
                
                print(f"📖 Retrieved {len(result.get('facts', []))} facts (confidence: {result['confidence']:.1%})")
                
                return {
                    'type': 'direct_answer',
                    'response': result['answer'],
                    'facts_used': result.get('facts', []),
                    'pipeline': 'faiss_semantic',
                    'sub_intent': dominant_sub_intent,
                    'confidence': result['confidence']
                }
        
        # ========== No Match Found ==========
        print("❌ No matching information found")
        
        return {
            'type': 'direct_answer',
            'response': "I couldn't find specific information about that. Could you please specify the college name and department/course you're asking about?",
            'facts_used': [],
            'pipeline': 'no_match',
            'sub_intent': dominant_sub_intent,
            'confidence': 0.0
        }
    
    # ==================== Helper Methods ====================
    
    def _extract_comparison_colleges(self, prompt: str, entities: Dict) -> list:
        """Extract colleges to compare from prompt"""
        colleges = []
        
        # Common college names to look for - check these first for accurate mapping
        prompt_lower = prompt.lower()
        college_keywords = {
            'pulchowk': 'PULCHOWK ENGINEERING CAMPUS',
            'thapathali': 'THAPATHALI ENGINEERING CAMPUS',
            'ku ': 'KATHMANDU UNIVERSITY SCHOOL OF ENGINEERING',  # Note: space after to avoid matching 'pulchowk'
            ' ku': 'KATHMANDU UNIVERSITY SCHOOL OF ENGINEERING',  # Space before
            'kathmandu university': 'KATHMANDU UNIVERSITY SCHOOL OF ENGINEERING',
            'ncit': 'NATIONAL COLLEGE OF IT',
            'nec': 'NEPAL ENGINEERING COLLEGE',
            'kantipur': 'KANTIPUR ENGINEERING COLLEGE',
            'kathmandu engineering': 'KATHMANDU ENGINEERING COLLEGE',
            'kec': 'KATHMANDU ENGINEERING COLLEGE',
            'advanced': 'ADVANCED COLLEGE OF ENGINEERING',
            'pu ': 'POKHARA UNIVERSITY SCHOOL OF ENGINEERING',
            ' pu': 'POKHARA UNIVERSITY SCHOOL OF ENGINEERING',
            'pokhara': 'POKHARA UNIVERSITY SCHOOL OF ENGINEERING',
            'sagarmatha': 'SAGARMATHA ENGINEERING COLLEGE',
            'khwopa': 'KHWOPA ENGINEERING COLLEGE',
            'himalaya': 'HIMALAYA COLLEGE OF ENGINEERING',
            'cosmos': 'COSMOS COLLEGE OF MANAGEMENT AND TECHNOLOGY',
            'patan': 'PATAN COLLEGE',
        }
        
        # Add padding for word boundary matching
        padded_prompt = ' ' + prompt_lower + ' '
        
        for keyword, full_name in college_keywords.items():
            if keyword in padded_prompt and full_name not in colleges:
                colleges.append(full_name)
        
        # If we didn't find enough from keywords, try entity extraction
        if len(colleges) < 2 and entities.get('college_name'):
            # Try to resolve the entity name
            entity_name = entities['college_name'].lower()
            for keyword, full_name in college_keywords.items():
                if keyword.strip() in entity_name and full_name not in colleges:
                    colleges.append(full_name)
                    break
        
        return colleges
    
    def _format_recommendation_response(self, ranked_colleges: list, prompt: str) -> str:
        """Format recommendation results from XGBoost recommender"""
        if not ranked_colleges:
            return "No colleges found matching your criteria."
        
        import re
        # Match patterns like "top 5", "5 colleges", "recommend 3" but NOT fee amounts like "10 lakh"
        count_patterns = [
            r'top\s+(\d+)',           # top 5
            r'(\d+)\s+colleges?',     # 5 colleges
            r'recommend\s+(\d+)',     # recommend 3
            r'suggest\s+(\d+)',       # suggest 5
            r'show\s+(\d+)',          # show 10
            r'list\s+(\d+)',          # list 5
        ]
        
        requested_count = 5  # Default
        for pattern in count_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                requested_count = int(match.group(1))
                break
        
        requested_count = min(requested_count, len(ranked_colleges))
        
        response_parts = [f"🎓 Found {len(ranked_colleges)} colleges matching your preferences:\n"]
        
        for i, college in enumerate(ranked_colleges[:requested_count], 1):
            # Handle both old and new data formats
            if 'college_data' in college:
                # Old format
                cd = college['college_data']
                name = cd.get('CollegeName', cd.get('college_name', 'Unknown'))
                course = cd.get('CourseName', cd.get('course', 'N/A'))
                location = cd.get('Location', cd.get('location', 'N/A'))
                col_type = cd.get('Type', cd.get('type', 'N/A'))
                fee = cd.get('Fee', cd.get('fee', 0))
                rating = cd.get('Rating', cd.get('rating', 'N/A'))
            else:
                # New XGBoost format
                name = college.get('college_name', 'Unknown')
                course = college.get('course', college.get('department', 'N/A'))
                location = college.get('location', 'N/A')
                col_type = college.get('type', 'N/A')
                fee = college.get('fee', 0)
                rating = college.get('rating', 'N/A')
            
            # Relevance score
            score = college.get('relevance_score', college.get('score', 0))
            score_pct = f"{score * 100:.0f}%" if isinstance(score, float) else ""
            
            college_info = [
                f"**{i}. {name}**",
                f"   📚 {course}",
                f"   📍 {location} | {col_type}",
                f"   💰 Rs. {fee:,.0f}" if fee else "   💰 Fee: N/A",
                f"   ⭐ Rating: {rating}",
            ]
            
            if score_pct:
                college_info.append(f"   🎯 Match: {score_pct}")
            
            # Add hostel/scholarship info if available
            if college.get('hostel'):
                college_info.append("   🏠 Hostel Available")
            if college.get('scholarship') and college.get('scholarship') > 0:
                college_info.append(f"   🎓 Scholarship: {college['scholarship']}%")
            
            response_parts.append('\n'.join(college_info))
        
        response_parts.append(f"\n💡 Type a number (1-{requested_count}) for detailed information about a college!")
        
        return '\n\n'.join(response_parts)
    
    def _format_comparison_response(self, comparison_data: list, prompt: str) -> str:
        """Format comparison response with comprehensive data"""
        if len(comparison_data) < 2:
            return "I need more information to make a comparison."
        
        college1, college2 = comparison_data[0], comparison_data[1]
        
        name1 = college1.get('name', 'College 1')
        name2 = college2.get('name', 'College 2')
        
        response = f"## ⚖️ Comparison: {name1} vs {name2}\n\n"
        
        # Create comparison table
        response += "| Aspect | " + name1[:25] + " | " + name2[:25] + " |\n"
        response += "|--------|--------|--------|\n"
        
        # Location
        loc1 = college1.get('location', 'N/A')
        loc2 = college2.get('location', 'N/A')
        response += f"| 📍 Location | {loc1[:30]} | {loc2[:30]} |\n"
        
        # Type
        type1 = college1.get('type', 'N/A')
        type2 = college2.get('type', 'N/A')
        response += f"| 🏛️ Type | {type1} | {type2} |\n"
        
        # Hostel
        hostel1 = '✅ Yes' if college1.get('hostel') else '❌ No'
        hostel2 = '✅ Yes' if college2.get('hostel') else '❌ No'
        response += f"| 🏠 Hostel | {hostel1} | {hostel2} |\n"
        
        # Average Fee
        fee1 = college1.get('avg_fee', 0)
        fee2 = college2.get('avg_fee', 0)
        response += f"| 💰 Avg Fee | Rs. {fee1:,.0f} | Rs. {fee2:,.0f} |\n"
        
        # Average Rating
        rating1 = college1.get('avg_rating', 0)
        rating2 = college2.get('avg_rating', 0)
        response += f"| ⭐ Avg Rating | {rating1:.1f}/5 | {rating2:.1f}/5 |\n"
        
        # Courses
        courses1 = college1.get('num_courses', 0)
        courses2 = college2.get('num_courses', 0)
        response += f"| 📚 Courses | {courses1} | {courses2} |\n"
        
        # Total Seats
        seats1 = college1.get('total_seats', 0)
        seats2 = college2.get('total_seats', 0)
        response += f"| 🪑 Total Seats | {seats1} | {seats2} |\n"
        
        response += "\n### 📋 Departments Available\n"
        response += f"\n**{name1}**: " + ", ".join(college1.get('departments', ['N/A'])[:5])
        response += f"\n**{name2}**: " + ", ".join(college2.get('departments', ['N/A'])[:5])
        
        response += "\n\n💡 Would you like more details about either college?"
        
        return response
    
    def _finalize_result(self, result: Dict, intent_result: Dict, entities: Dict, session_id: str) -> Dict[str, Any]:
        """Add metadata to result"""
        result.update({
            'intent': intent_result.get('primary_intent'),
            'sub_intent': intent_result.get('dominant_sub_intent', intent_result.get('sub_intent')),
            'sub_intents': intent_result.get('sub_intents', []),
            'is_multi_intent': intent_result.get('is_multi_intent', False),
            'full_intent': intent_result.get('full_intent'),
            'intent_confidence': intent_result.get('primary_confidence', 0),
            'entities': entities,
            'session_id': session_id
        })
        return result


# ==================== Factory Functions ====================

def create_unified_pipeline(df: pd.DataFrame, data_path: str = 'full_data.json') -> UnifiedPipelineController:
    """Factory function for creating unified pipeline controller"""
    return UnifiedPipelineController(df, data_path)


# For backward compatibility
PipelineController = UnifiedPipelineController
create_pipeline_controller = create_unified_pipeline
