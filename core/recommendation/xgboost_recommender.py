"""
XGBoost-Based College Recommendation System

This system uses XGBoost for ranking colleges based on:
1. User preferences (from conversation state)
2. College features (from database)
3. Match indicators between user and college

The conversation memory is maintained externally (not learned by the model).
XGBoost learns ranking behavior from synthetic training data.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")


# ===================== CONVERSATION STATE =====================

@dataclass
class ConversationState:
    """
    JSON-serializable conversation state for multi-turn conversations.
    This is maintained externally and passed to the recommender.
    """
    last_intent: Optional[str] = None
    last_entities: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=lambda: {
        "location": None,
        "fee_range": None,  # (min, max) tuple or max value
        "college_type": None,  # public/private
        "course": None,  # course name or department
        "hostel_required": None,
        "scholarship_needed": None,
        "min_rating": None,
        "max_cutoff": None,  # max acceptable cutoff rank
    })
    previous_recommendations: List[str] = field(default_factory=list)
    previous_comparisons: List[Tuple[str, str]] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create from dictionary"""
        return cls(
            last_intent=data.get('last_intent'),
            last_entities=data.get('last_entities', {}),
            preferences=data.get('preferences', cls.__dataclass_fields__['preferences'].default_factory()),
            previous_recommendations=data.get('previous_recommendations', []),
            previous_comparisons=data.get('previous_comparisons', []),
            conversation_history=data.get('conversation_history', [])
        )
    
    def update_from_entities(self, entities: Dict[str, Any], intent: str):
        """Update preferences from extracted entities"""
        self.last_intent = intent
        self.last_entities = entities
        
        # Map entities to preferences
        if entities.get('location'):
            self.preferences['location'] = entities['location']
        
        if entities.get('college_type'):
            self.preferences['college_type'] = entities['college_type']
        
        if entities.get('course_name') or entities.get('department_name'):
            self.preferences['course'] = entities.get('course_name') or entities.get('department_name')
        
        if entities.get('fee'):
            fee_val = entities['fee']
            # Handle if fee is already a number
            if isinstance(fee_val, (int, float)):
                self.preferences['fee_range'] = int(fee_val)
            else:
                # Try to parse fee from string
                fee_str = str(fee_val).replace(',', '')
                import re
                numbers = re.findall(r'\d+', fee_str)
                if numbers:
                    self.preferences['fee_range'] = int(numbers[0])
        
        if entities.get('hostel_availability'):
            self.preferences['hostel_required'] = True
        
        if entities.get('scholarship'):
            self.preferences['scholarship_needed'] = True
    
    def add_recommendation(self, college_name: str):
        """Track recommended colleges"""
        if college_name not in self.previous_recommendations:
            self.previous_recommendations.append(college_name)
    
    def add_turn(self, user_msg: str, bot_response: str):
        """Add conversation turn"""
        self.conversation_history.append({
            'user': user_msg,
            'bot': bot_response
        })
        # Keep last 10 turns
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def clear_recommendations(self):
        """Clear previous recommendations for fresh results"""
        self.previous_recommendations = []


# ===================== XGBOOST RECOMMENDER =====================

class XGBoostRecommender:
    """
    XGBoost-based recommendation system for college ranking.
    
    Features:
    - User features (from conversation state preferences)
    - College features (from database)
    - Match features (user-college compatibility)
    
    The model predicts a relevance score (0.0 - 1.0) for each college.
    """
    
    # Feature schema
    USER_FEATURES = [
        'user_fee_limit_norm',
        'user_hostel_required',
        'user_scholarship_needed',
        'user_min_rating_norm',
    ]
    
    COLLEGE_FEATURES = [
        'college_fee_norm',
        'college_seats_norm',
        'college_rating_norm',
        'college_cutoff_norm',
        'college_pass_rate_norm',
        'college_internship',
        'college_scholarship_norm',
        'college_hostel',
    ]
    
    MATCH_FEATURES = [
        'location_match',
        'fee_within_range',
        'hostel_match',
        'course_match',
        'college_type_match',
        'scholarship_match',
    ]
    
    def __init__(self, data_path: str = 'full_data.json', 
                 model_path: str = './models/xgboost_recommender'):
        self.data_path = data_path
        self.model_path = model_path
        
        # Model components
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.location_encoder: Optional[OneHotEncoder] = None
        self.type_encoder: Optional[OneHotEncoder] = None
        self.course_encoder: Optional[OneHotEncoder] = None
        
        # Data
        self.colleges_df: Optional[pd.DataFrame] = None
        self.raw_colleges: List[Dict] = []
        
        # Feature info
        self.all_locations: List[str] = []
        self.all_types: List[str] = []
        self.all_courses: List[str] = []
        
        # Load data and model
        self._load_data()
        self._load_or_train_model()
    
    def _load_data(self):
        """Load and flatten college data from JSON"""
        print("📚 Loading college data...")
        
        try:
            with open(self.data_path, 'r') as f:
                self.raw_colleges = json.load(f)
        except FileNotFoundError:
            print(f"❌ Data file not found: {self.data_path}")
            return
        
        # Flatten to college-course level
        rows = []
        for college in self.raw_colleges:
            base_info = {
                'college_id': college['CollegeId'],
                'college_name': college['Name'],
                'location': college['Location'].split(',')[0].strip(),  # First part of location
                'full_location': college['Location'],
                'college_type': college['Type'],
                'hostel': college.get('HostelAvailability', False),
                'contact': college.get('ContactNumber'),
                'email': college.get('Email'),
            }
            
            for dept in college.get('Departments', []):
                for course in dept.get('Courses', []):
                    row = base_info.copy()
                    row.update({
                        'department': dept['Name'],
                        'course_name': course['Name'],
                        'fee': course.get('Fee', 0),
                        'seats': course.get('TotalSeats', 0),
                        'rating': course.get('Rating', 0),
                        'cutoff': course.get('AverageCutoffRank', 10000),
                        'pass_rate': course.get('PassPercentage', 0),
                        'internship': course.get('InternshipOpportunities', False),
                        'scholarship': course.get('GeneralScholarship', 0),
                        'duration': course.get('DurationInYears', 4),
                        'admission': course.get('AdmissionProcess', 'ENTRANCE'),
                    })
                    rows.append(row)
        
        self.colleges_df = pd.DataFrame(rows)
        
        # Extract unique values for encoding
        self.all_locations = self.colleges_df['location'].unique().tolist()
        self.all_types = self.colleges_df['college_type'].unique().tolist()
        self.all_courses = self.colleges_df['department'].unique().tolist()
        
        print(f"✅ Loaded {len(self.colleges_df)} college-course combinations")
        print(f"   📍 {len(self.all_locations)} locations, {len(self.all_types)} types, {len(self.all_courses)} departments")
    
    def _load_or_train_model(self):
        """Load existing model or train new one"""
        model_file = os.path.join(self.model_path, 'xgboost_model.json')
        
        if os.path.exists(model_file):
            try:
                self._load_model()
                print("✅ XGBoost model loaded from disk")
                return
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
        
        # Train new model
        print("🏋️ Training XGBoost recommendation model...")
        self._train_model()
    
    def _prepare_encoders(self):
        """Prepare encoders for categorical features"""
        # MinMax scaler for numeric features
        numeric_cols = ['fee', 'seats', 'rating', 'cutoff', 'pass_rate', 'scholarship']
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.colleges_df[numeric_cols])
        
        # One-hot encoders for categorical (but we'll use simpler matching)
        # For XGBoost, we use match features instead of full OHE
    
    def _generate_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for XGBoost.
        
        Strategy: Create user-college pairs with heuristic relevance labels.
        A college is more relevant if it matches user preferences.
        """
        print("🔢 Generating training data...")
        
        X_list = []
        y_list = []
        
        # Generate diverse user preference scenarios
        scenarios = self._generate_user_scenarios()
        
        for scenario in scenarios:
            for _, college_row in self.colleges_df.iterrows():
                # Build feature vector
                features = self._build_feature_vector(scenario, college_row)
                X_list.append(features)
                
                # Calculate relevance label (heuristic)
                relevance = self._calculate_relevance(scenario, college_row)
                y_list.append(relevance)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"✅ Generated {len(X)} training samples")
        return X, y
    
    def _generate_user_scenarios(self) -> List[Dict]:
        """Generate diverse user preference scenarios for training"""
        scenarios = []
        
        # Scenario templates
        fee_ranges = [None, 100000, 200000, 500000, 800000]
        hostel_options = [None, True, False]
        scholarship_options = [None, True, False]
        rating_options = [None, 3.0, 4.0, 4.5]
        
        # Sample locations and courses
        sample_locations = self.all_locations[:10] + [None] * 5
        sample_courses = self.all_courses[:10] + [None] * 5
        sample_types = self.all_types + [None]
        
        import random
        random.seed(42)
        
        # Generate scenarios
        for _ in range(100):  # 100 diverse scenarios
            scenario = {
                'location': random.choice(sample_locations),
                'fee_range': random.choice(fee_ranges),
                'college_type': random.choice(sample_types),
                'course': random.choice(sample_courses),
                'hostel_required': random.choice(hostel_options),
                'scholarship_needed': random.choice(scholarship_options),
                'min_rating': random.choice(rating_options),
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _build_feature_vector(self, user_prefs: Dict, college_row: pd.Series) -> np.ndarray:
        """
        Build feature vector for a user-college pair.
        
        Features:
        1. User features (normalized preferences)
        2. College features (normalized attributes)
        3. Match features (compatibility indicators)
        """
        features = []
        
        # ===== USER FEATURES =====
        # Fee limit (normalized, 0-1)
        fee_limit = user_prefs.get('fee_range')
        if fee_limit:
            fee_norm = min(fee_limit / 1000000, 1.0)  # Normalize to 1M
        else:
            fee_norm = 0.5  # Default middle value
        features.append(fee_norm)
        
        # Hostel required (binary)
        features.append(1.0 if user_prefs.get('hostel_required') else 0.0)
        
        # Scholarship needed (binary)
        features.append(1.0 if user_prefs.get('scholarship_needed') else 0.0)
        
        # Min rating (normalized)
        min_rating = user_prefs.get('min_rating')
        features.append((min_rating / 5.0) if min_rating else 0.5)
        
        # ===== COLLEGE FEATURES =====
        max_fee = self.colleges_df['fee'].max() or 1
        max_seats = self.colleges_df['seats'].max() or 1
        max_cutoff = self.colleges_df['cutoff'].max() or 1
        
        features.append(college_row['fee'] / max_fee)  # Fee normalized
        features.append(college_row['seats'] / max_seats)  # Seats normalized
        features.append(college_row['rating'] / 5.0)  # Rating normalized to 5
        features.append(college_row['cutoff'] / max_cutoff)  # Cutoff normalized
        features.append(college_row['pass_rate'] / 100.0)  # Pass rate as percentage
        features.append(1.0 if college_row['internship'] else 0.0)  # Internship binary
        features.append(min(college_row['scholarship'] / 100.0, 1.0))  # Scholarship normalized
        features.append(1.0 if college_row['hostel'] else 0.0)  # Hostel binary
        
        # ===== MATCH FEATURES =====
        # Location match
        user_loc = user_prefs.get('location', '').lower() if user_prefs.get('location') else ''
        college_loc = str(college_row['location']).lower()
        features.append(1.0 if user_loc and user_loc in college_loc else 0.0)
        
        # Fee within range
        if user_prefs.get('fee_range'):
            features.append(1.0 if college_row['fee'] <= user_prefs['fee_range'] else 0.0)
        else:
            features.append(0.5)  # No preference
        
        # Hostel match
        if user_prefs.get('hostel_required'):
            features.append(1.0 if college_row['hostel'] else 0.0)
        else:
            features.append(0.5)  # No requirement
        
        # Course match
        user_course = user_prefs.get('course', '').lower() if user_prefs.get('course') else ''
        college_dept = str(college_row['department']).lower()
        course_match = 1.0 if user_course and user_course in college_dept else 0.0
        features.append(course_match)
        
        # College type match
        user_type = user_prefs.get('college_type', '').lower() if user_prefs.get('college_type') else ''
        college_type = str(college_row['college_type']).lower()
        features.append(1.0 if user_type and user_type == college_type else 0.5)
        
        # Scholarship match
        if user_prefs.get('scholarship_needed'):
            features.append(1.0 if college_row['scholarship'] > 0 else 0.0)
        else:
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_relevance(self, user_prefs: Dict, college_row: pd.Series) -> float:
        """
        Calculate heuristic relevance score for training.
        
        Score components:
        - Base quality score (from rating, pass rate)
        - Match bonuses (for matching preferences)
        - Penalties (for mismatches on hard requirements)
        """
        score = 0.0
        
        # Base quality (0.0 - 0.4)
        rating_score = (college_row['rating'] / 5.0) * 0.2
        pass_score = (college_row['pass_rate'] / 100.0) * 0.1
        internship_score = 0.05 if college_row['internship'] else 0.0
        scholarship_score = min(college_row['scholarship'] / 100.0, 0.05)
        
        score += rating_score + pass_score + internship_score + scholarship_score
        
        # Match bonuses (0.0 - 0.4)
        # Location match
        if user_prefs.get('location'):
            user_loc = user_prefs['location'].lower()
            college_loc = str(college_row['location']).lower()
            if user_loc in college_loc:
                score += 0.15
        
        # Course match (important)
        if user_prefs.get('course'):
            user_course = user_prefs['course'].lower()
            college_dept = str(college_row['department']).lower()
            if user_course in college_dept:
                score += 0.2
        
        # College type match
        if user_prefs.get('college_type'):
            if user_prefs['college_type'].lower() == str(college_row['college_type']).lower():
                score += 0.05
        
        # Penalty: Fee exceeds budget
        if user_prefs.get('fee_range'):
            if college_row['fee'] > user_prefs['fee_range']:
                score -= 0.2
            else:
                score += 0.1  # Bonus for being within budget
        
        # Penalty: Hostel required but not available
        if user_prefs.get('hostel_required') and not college_row['hostel']:
            score -= 0.15
        
        # Rating threshold
        if user_prefs.get('min_rating'):
            if college_row['rating'] >= user_prefs['min_rating']:
                score += 0.05
            else:
                score -= 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def _train_model(self):
        """Train XGBoost regressor"""
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available")
            return
        
        self._prepare_encoders()
        
        # Generate training data
        X, y = self._generate_training_data()
        
        # Train XGBoost - use n_jobs=1 to prevent segfaults with async frameworks
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1  # Single thread to prevent segfaults with FastAPI/async
        )
        
        self.model.fit(X, y)
        print("✅ XGBoost model trained")
        
        # Save model
        self._save_model()
    
    def _save_model(self):
        """Save model and encoders to disk"""
        os.makedirs(self.model_path, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(os.path.join(self.model_path, 'xgboost_model.json'))
        
        # Save scaler
        with open(os.path.join(self.model_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"💾 Model saved to {self.model_path}")
    
    def _load_model(self):
        """Load model and encoders from disk"""
        self._prepare_encoders()
        
        # Load XGBoost model
        self.model = xgb.XGBRegressor()
        self.model.load_model(os.path.join(self.model_path, 'xgboost_model.json'))
        
        # Load scaler
        scaler_path = os.path.join(self.model_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def recommend(self, 
                  query: str,
                  conversation_state: ConversationState,
                  entities: Dict[str, Any] = None,
                  k: int = 5,
                  exclude_previous: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend top-K colleges based on conversation state.
        
        Args:
            query: User's query text
            conversation_state: Current conversation state with preferences
            entities: Extracted entities from current query
            k: Number of recommendations
            exclude_previous: Whether to exclude previously recommended colleges
            
        Returns:
            List of recommended colleges with scores
        """
        if self.model is None or self.colleges_df is None:
            return []
        
        # Update state with new entities
        if entities:
            conversation_state.update_from_entities(entities, 'recommendation')
        
        prefs = conversation_state.preferences
        
        # ===== HARD FILTERS =====
        # Apply strict filters for budget and location before ranking
        max_fee = prefs.get('fee_range')
        user_location = prefs.get('location', '').lower() if prefs.get('location') else None
        user_type = prefs.get('college_type', '').lower() if prefs.get('college_type') else None
        
        print(f"🔍 Filters: type={user_type}, location={user_location}, max_fee={max_fee}")
        
        # Build feature vectors for all colleges that pass hard filters
        feature_vectors = []
        valid_indices = []
        
        for idx, row in self.colleges_df.iterrows():
            # Skip previously recommended colleges
            if exclude_previous and row['college_name'] in conversation_state.previous_recommendations:
                continue
            
            # Hard filter: Fee must be within budget (if specified)
            if max_fee and row['fee'] > max_fee:
                continue
            
            # Hard filter: College type must match (if specified)
            if user_type:
                college_type = str(row.get('college_type', '')).lower()
                if college_type != user_type:
                    continue
            
            # Hard filter: Location must match (if specified)
            if user_location:
                college_loc = str(row.get('location', '')).lower()
                full_loc = str(row.get('full_location', '')).lower()
                if user_location not in college_loc and user_location not in full_loc:
                    continue
            
            features = self._build_feature_vector(prefs, row)
            feature_vectors.append(features)
            valid_indices.append(idx)
        
        if not feature_vectors:
            # No colleges match the hard filters, try with relaxed filters
            print("⚠️ No colleges found with strict filters, relaxing fee/location...")
            for idx, row in self.colleges_df.iterrows():
                if exclude_previous and row['college_name'] in conversation_state.previous_recommendations:
                    continue
                # Still apply college type filter even in fallback
                if user_type:
                    college_type = str(row.get('college_type', '')).lower()
                    if college_type != user_type:
                        continue
                features = self._build_feature_vector(prefs, row)
                feature_vectors.append(features)
                valid_indices.append(idx)
        
        if not feature_vectors:
            # All colleges already recommended, reset
            conversation_state.clear_recommendations()
            return self.recommend(query, conversation_state, entities, k, exclude_previous=False)
        
        # Predict relevance scores
        X = np.array(feature_vectors)
        scores = self.model.predict(X)
        
        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]  # Descending
        
        # Collect top-K results
        results = []
        seen_colleges = set()
        
        for rank_idx in ranked_indices:
            if len(results) >= k:
                break
            
            df_idx = valid_indices[rank_idx]
            row = self.colleges_df.iloc[df_idx]
            college_name = row['college_name']
            
            # Avoid duplicates (same college different courses)
            if college_name in seen_colleges:
                continue
            seen_colleges.add(college_name)
            
            # Add to results
            result = {
                'college_name': college_name,
                'course': row['course_name'],
                'department': row['department'],
                'location': row['full_location'],
                'type': row['college_type'],
                'fee': row['fee'],
                'rating': row['rating'],
                'seats': row['seats'],
                'hostel': row['hostel'],
                'scholarship': row['scholarship'],
                'relevance_score': float(scores[rank_idx]),
            }
            results.append(result)
            
            # Track recommendation
            conversation_state.add_recommendation(college_name)
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}
        
        feature_names = (
            self.USER_FEATURES + 
            self.COLLEGE_FEATURES + 
            self.MATCH_FEATURES
        )
        
        importance = self.model.feature_importances_
        
        return dict(zip(feature_names, importance))


# ===================== FACTORY FUNCTION =====================

def create_xgboost_recommender(data_path: str = 'full_data.json') -> XGBoostRecommender:
    """Create and return XGBoost recommender instance"""
    return XGBoostRecommender(data_path=data_path)


# ===================== STATE MANAGER INTEGRATION =====================

class ConversationStateManager:
    """
    Manages conversation states for multiple sessions.
    Integrates with XGBoost recommender.
    
    Compatible with the old state manager API for seamless integration.
    """
    
    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, ConversationState] = {}
        self.max_history = max_history
    
    def get_state(self, session_id: str) -> ConversationState:
        """Get or create conversation state for session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState()
        return self.sessions[session_id]
    
    def update_state(self, session_id: str, entities: Dict, intent: str):
        """Update session state with new entities"""
        state = self.get_state(session_id)
        state.update_from_entities(entities, intent)
    
    def add_turn(self, session_id: str, user_message: str = None, bot_response: str = None,
                 intent: str = None, sub_intent: str = None, entities: Dict = None,
                 user_msg: str = None):
        """
        Add conversation turn to history.
        Compatible with both old and new API signatures.
        """
        state = self.get_state(session_id)
        
        # Handle both API styles
        user = user_message or user_msg or ""
        bot = bot_response or ""
        
        state.add_turn(user, bot)
        
        # Update state with entities if provided
        if entities:
            state.update_from_entities(entities, intent or 'unknown')
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get current conversation context for processing.
        Compatible with old state manager API.
        """
        state = self.get_state(session_id)
        
        # Build slots-like dict from preferences
        slots = {}
        prefs = state.preferences
        if prefs.get('location'):
            slots['location'] = prefs['location']
        if prefs.get('course'):
            slots['course_name'] = prefs['course']
        if prefs.get('college_type'):
            slots['college_type'] = prefs['college_type']
        if prefs.get('fee_range'):
            slots['max_fee'] = prefs['fee_range']
        if prefs.get('hostel_required'):
            slots['hostel_required'] = True
        
        # Get last mentioned colleges from previous recommendations
        last_colleges = state.previous_recommendations[-5:] if state.previous_recommendations else []
        
        return {
            'slots': slots,
            'last_colleges': last_colleges,
            'last_intent': state.last_intent,
            'turn_count': len(state.conversation_history),
            'history_summary': state.conversation_history[-3:] if state.conversation_history else []
        }
    
    def resolve_references(self, session_id: str, prompt: str, entities: Dict) -> Dict:
        """
        Resolve references like 'it', 'that college' to actual entities.
        Compatible with old state manager API.
        """
        state = self.get_state(session_id)
        resolved = entities.copy()
        
        prompt_lower = prompt.lower()
        
        # Check for references that need resolution
        reference_words = ['it', 'that', 'this', 'the college', 'that college', 'this college', 
                          'same', 'there', 'at the same', 'from there', 'in that', 'about that']
        needs_resolution = any(ref in prompt_lower for ref in reference_words)
        
        # If no college mentioned but reference words found
        if needs_resolution and not resolved.get('college_name'):
            # Try to get from last entities
            if state.last_entities.get('college_name'):
                resolved['college_name'] = state.last_entities['college_name']
                resolved['_resolved_from_context'] = True
            # Or from previous recommendations
            elif state.previous_recommendations:
                resolved['college_name'] = state.previous_recommendations[-1]
                resolved['_resolved_from_context'] = True
        
        # Resolve course/department from context
        if not resolved.get('department_name') and not resolved.get('course_name'):
            if state.last_entities.get('department_name'):
                resolved['department_name'] = state.last_entities['department_name']
            elif state.last_entities.get('course_name'):
                resolved['course_name'] = state.last_entities['course_name']
            elif state.preferences.get('course'):
                resolved['course_name'] = state.preferences['course']
        
        return resolved
    
    def clear_session(self, session_id: str):
        """Clear session state"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_preferences_summary(self, session_id: str) -> str:
        """Get human-readable summary of current preferences"""
        state = self.get_state(session_id)
        prefs = state.preferences
        
        parts = []
        if prefs.get('location'):
            parts.append(f"Location: {prefs['location']}")
        if prefs.get('fee_range'):
            parts.append(f"Max Fee: Rs. {prefs['fee_range']:,}")
        if prefs.get('college_type'):
            parts.append(f"Type: {prefs['college_type']}")
        if prefs.get('course'):
            parts.append(f"Course: {prefs['course']}")
        if prefs.get('hostel_required'):
            parts.append("Hostel: Required")
        if prefs.get('scholarship_needed'):
            parts.append("Scholarship: Required")
        
        return " | ".join(parts) if parts else "No specific preferences"


# Alias for compatibility with unified pipeline
XGBStateManager = ConversationStateManager