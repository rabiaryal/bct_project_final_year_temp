#!/usr/bin/env python3
"""
College Recommendation Chatbot - Standalone Script
Run with: python chatbot.py
"""

import json
import pandas as pd
import re
import warnings
from typing import Dict, List

# Optional imports for ML functionality
try:
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    ML_AVAILABLE = True
except ImportError:
    print("⚠️  ML libraries not available. Using basic functionality only.")
    ML_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# FAISS Vector Database for Question Answering
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import pickle
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION
# ========================================

CONFIG = {
    'data_path': 'full_data.json',
    'query_encoder_path': 'query_encoder.h5',
    'item_encoder_path': 'item_encoder.h5',
    'embedding_dim': 64,
    'epochs': 5,
    'batch_size': 32,
    'num_negatives': 5,
    'top_k_results': 5
}

# ========================================
# DATA LOADING
# ========================================

def load_and_preprocess_data(json_path):
    """Load and preprocess college data"""
    print("📊 Loading data...")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    rows = []
    for college in data:
        for dept in college.get('Departments', []):
            for course in dept.get('Courses', []):
                row = {
                    "CollegeId": college.get("CollegeId"),
                    "CollegeName": college.get("Name"),
                    "Location": college.get("Location"),
                    "Type": college.get("Type"),
                    "ContactNumber": college.get("ContactNumber"),
                    "Email": college.get("Email"),
                    "HostelAvailability": college.get("HostelAvailability", False),
                    "DepartmentName": dept.get("Name"),
                    "CourseId": course.get("CourseId"),
                    "CourseName": course.get("Name"),
                    "Fee": course.get("Fee", 0),
                    "TotalSeats": course.get("TotalSeats", 0),
                    "PassPercentage": course.get("PassPercentage", 0),
                    "InternshipOpportunities": course.get("InternshipOpportunities", False),
                    "GeneralScholarship": course.get("GeneralScholarship", False),
                    "SemesterScholarship": course.get("SemesterScholarship", False),
                    "DurationInYears": course.get("DurationInYears"),
                    "AdmissionProcess": course.get("AdmissionProcess", ""),
                    "Rating": course.get("Rating", 0)
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"✓ Loaded {len(df)} college-course combinations")
    
    # Keep original values for display
    df['OriginalFee'] = df['Fee'].copy()
    df['OriginalSeats'] = df['TotalSeats'].copy()
    df['OriginalRating'] = df['Rating'].copy()
    df['OriginalPassPercentage'] = df['PassPercentage'].copy()
    
    # Only do ML preprocessing if libraries are available
    if ML_AVAILABLE:
        try:
            NUM_COLS = ['Fee', 'TotalSeats', 'PassPercentage', 'Rating']
            scaler = MinMaxScaler()
            df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
            
            CAT_COLS = ['Location', 'Type', 'CourseName']
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cat_encoded = encoder.fit_transform(df[CAT_COLS])
            cat_feature_names = encoder.get_feature_names_out(CAT_COLS)
            df_cat_encoded = pd.DataFrame(cat_encoded, columns=cat_feature_names)
            
            df = pd.concat([df.reset_index(drop=True), df_cat_encoded], axis=1)
            
            item_features = [col for col in df.columns if col in NUM_COLS or col in cat_feature_names]
            query_input_cols = ['query_fee_norm'] + [col + '_query' for col in cat_feature_names]
            
            print("✓ ML preprocessing complete")
            
            return df, scaler, encoder, item_features, query_input_cols, cat_feature_names
        except Exception as e:
            print(f"⚠️  ML preprocessing failed: {e}. Using basic functionality.")
    
    print("✓ Basic preprocessing complete")
    
    return df, None, None, [], [], []

# ========================================
# TRAINING DATA GENERATION
# ========================================

def generate_training_data(df, cat_feature_names, num_negatives=5):
    """Generate training pairs"""
    print("🎯 Generating training data...")
    
    training_data = []
    
    for idx, item in df.iterrows():
        positive_pair = {
            'query_fee_norm': item['Fee'],
            'item_index': idx,
            'label': 1.0
        }
        
        for col in cat_feature_names:
            positive_pair[f'{col}_query'] = item[col]
        
        training_data.append(positive_pair)
        
        negative_indices = random.sample(range(len(df)), min(num_negatives, len(df)))
        for neg_idx in negative_indices:
            if neg_idx != idx:
                negative_pair = {
                    'query_fee_norm': item['Fee'],
                    'item_index': neg_idx,
                    'label': 0.0
                }
                for col in cat_feature_names:
                    negative_pair[f'{col}_query'] = item[col]
                training_data.append(negative_pair)
    
    df_train = pd.DataFrame(training_data)
    print(f"✓ Generated {len(df_train)} training pairs")
    
    return df_train

def prepare_training_arrays(df_train, df, item_features, query_input_cols):
    """Prepare numpy arrays"""
    item_indices = df_train['item_index'].values
    item_data = df.loc[item_indices, item_features].values
    
    query_cols = [col for col in df_train.columns if col in query_input_cols]
    query_data = df_train[query_cols].values
    labels = df_train['label'].values
    
    return query_data, item_data, labels

# ========================================
# MODEL BUILDING
# ========================================

def build_dual_encoder(query_dim, item_dim, embedding_dim=64):
    """Build dual-encoder model"""
    print("🏗️ Building model...")
    
    query_input = keras.Input(shape=(query_dim,), name="query_features")
    x = layers.Dense(128, activation="relu")(query_input)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    query_embedding = layers.Dense(embedding_dim, name="query_embedding")(x)
    query_encoder = keras.Model(query_input, query_embedding)
    
    item_input = keras.Input(shape=(item_dim,), name="item_features")
    y = layers.Dense(128, activation="relu")(item_input)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(64, activation="relu")(y)
    item_embedding = layers.Dense(embedding_dim, name="item_embedding")(y)
    item_encoder = keras.Model(item_input, item_embedding)
    
    similarity = layers.Dot(axes=1, normalize=True)([query_embedding, item_embedding])
    training_model = keras.Model([query_input, item_input], similarity)
    
    training_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    print("✓ Model built")
    return query_encoder, item_encoder, training_model

# ========================================
# QUERY PARSER
# ========================================

class QueryParser:
    def __init__(self, df):
        self.valid_locations = set(df['Location'].unique())
        self.valid_types = set(df['Type'].unique())
        self.valid_courses = set(df['CourseName'].unique())
        self.valid_colleges = set(df['CollegeName'].unique())
        self.fee_keywords = {
            'low': (0, 300000), 'cheap': (0, 300000),
            'affordable': (200000, 600000), 'moderate': (400000, 800000),
            'high': (600000, 1500000), 'expensive': (800000, 2000000)
        }
    
    def parse(self, query):
        query_lower = query.lower()
        
        # Check if this is a direct question about a specific college
        specific_college = self._find_specific_college(query_lower)
        if specific_college:
            return {
                'type': 'direct_question',
                'college_name': specific_college,
                'question_type': self._detect_question_type(query_lower),
                'original_query': query  # Store for neural fallback
            }
        
        # Otherwise, treat as a recommendation query
        return {
            'type': 'recommendation',
            'location': self._find_match(query_lower, self.valid_locations),
            'college_type': self._find_match(query_lower, self.valid_types),
            'course': self._find_course(query_lower),
            'fee_range': self._find_fee(query_lower),
            'facilities': self._find_facilities(query_lower)
        }
    
    def _find_match(self, query, valid_set):
        for item in valid_set:
            if item.lower() in query:
                return item
        return None
    
    def _find_course(self, query):
        for course in self.valid_courses:
            if course.lower() in query:
                return course
        course_map = {'computer': 'Computer', 'civil': 'Civil', 'bba': 'BBA'}
        for keyword, part in course_map.items():
            if keyword in query:
                for course in self.valid_courses:
                    if part.lower() in course.lower():
                        return course
        return None
    
    def _find_fee(self, query):
        for keyword, range_val in self.fee_keywords.items():
            if keyword in query:
                return range_val
        return (0, 2000000)
    
    def _find_facilities(self, query):
        facilities = {}
        if 'hostel' in query:
            facilities['hostel'] = True
        if 'scholarship' in query:
            facilities['scholarship'] = True
        return facilities
    
    def _find_specific_college(self, query):
        """Find if the query mentions a specific college name"""
        # Clean the query for better matching
        query_clean = re.sub(r'[^\w\s]', '', query).strip()
        
        # First try fuzzy matching for better accuracy
        from difflib import get_close_matches
        college_names = list(self.valid_colleges)
        
        # Try fuzzy matching on full names first
        fuzzy_matches = get_close_matches(query_clean, 
                                        [name.lower() for name in college_names], 
                                        n=1, cutoff=0.6)
        if fuzzy_matches:
            # Find the original college name
            for college in college_names:
                if college.lower() == fuzzy_matches[0]:
                    return college
        
        # Fallback to original scoring system with improvements
        best_match = None
        best_score = 0
        
        # Look for exact or partial college name matches
        for college in self.valid_colleges:
            college_clean = re.sub(r'[^\w\s]', '', college.lower()).strip()
            college_words = college_clean.split()
            
            # Check for exact match first
            if college_clean in query_clean:
                return college
            
            # Score-based matching for partial matches
            if len(college_words) >= 2:
                # Get meaningful words (longer than 3 characters)
                key_words = [word for word in college_words if len(word) > 3]
                
                if len(key_words) >= 1:
                    # Count exact word matches
                    exact_matches = sum(1 for word in key_words if word in query_clean)
                    
                    # Calculate match score (exact matches / total key words)
                    match_score = exact_matches / len(key_words)
                    
                    # Boost score for unique/distinctive words with fuzzy matching
                    distinctive_words = {
                        'sagarmatha': ['sagarmathe', 'sagarmatha'], 
                        'pulchowk': ['pulchowk', 'pulchwok'], 
                        'thapathali': ['thapathali', 'thapatali'],
                        'kathmandu': ['kathmandu', 'ktm'],
                        'kantipur': ['kantipur', 'kantipure']
                    }
                    
                    for base_word, variations in distinctive_words.items():
                        if any(word in college_words for word in variations):
                            if any(var in query_clean for var in variations):
                                match_score += 0.7  # Higher bonus for distinctive words
                                break
                    
                    # Update best match if this is better
                    if match_score > best_score and match_score >= 0.6:  # Threshold of 60%
                        best_score = match_score
                        best_match = college
        
        return best_match
    
    def _detect_question_type(self, query):
        """Detect what type of information is being asked"""
        if any(word in query for word in ['location', 'located', 'where', 'address']):
            return 'location'
        elif any(word in query for word in ['contact', 'phone', 'number', 'call']):
            return 'contact'
        elif any(word in query for word in ['email', 'mail']):
            return 'email'
        elif any(word in query for word in ['course', 'program', 'department', 'study']):
            return 'courses'
        elif any(word in query for word in ['fee', 'cost', 'price', 'tuition']):
            return 'fees'
        elif any(word in query for word in ['hostel', 'accommodation', 'dormitory']):
            return 'hostel'
        elif any(word in query for word in ['student', 'seats', 'capacity', 'enrollment']):
            return 'seats'
        elif any(word in query for word in ['rating', 'rank', 'reputation', 'quality']):
            return 'rating'
        elif any(word in query for word in ['scholarship', 'financial', 'aid', 'merit', 'provide']):
            return 'scholarship'
        elif any(word in query for word in ['internship', 'placement', 'job', 'opportunities']):
            return 'internship'
        elif any(word in query for word in ['pass', 'success', 'graduation']):
            return 'pass_rate'
        elif any(word in query for word in ['admission', 'entrance', 'apply']):
            return 'admission'
        elif any(word in query for word in ['type', 'public', 'private']):
            return 'type'
        elif any(word in query for word in ['duration', 'years', 'length']):
            return 'duration'
        else:
            return 'general'

# ========================================
# FAISS QA RETRIEVER  
# ========================================

class FAISSQARetriever:
    """FAISS Vector Database Question Answering component"""
    
    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2
        
    def load_model(self, model_name='all-MiniLM-L6-v2'):
        """Load sentence transformer model and build index"""
        if not FAISS_AVAILABLE:
            return False
            
        try:
            print("📥 Loading sentence transformer model...")
            self.model = SentenceTransformer(model_name)
            
            # Build knowledge base from college data
            success = self.build_knowledge_base('full_data.json')
            if success:
                print("✓ FAISS QA model loaded successfully")
                return True
            else:
                return False
        except Exception as e:
            print(f"⚠️ Failed to load FAISS model: {e}")
            return False
    
    def build_knowledge_base(self, json_path):
        """Build FAISS index from college data"""
        try:
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            documents = []
            
            for college in data:
                college_name = college['Name']
                
                # Create comprehensive information documents
                documents.extend([
                    {
                        'text': f"{college_name} is located at {college['Location']}",
                        'metadata': {'type': 'location', 'college': college_name},
                        'answer': f"{college_name} is located at {college['Location']}"
                    },
                    {
                        'text': f"{college_name} contact number phone {college['ContactNumber']}",
                        'metadata': {'type': 'contact', 'college': college_name},
                        'answer': f"{college_name} contact number: {college['ContactNumber']}"
                    },
                    {
                        'text': f"{college_name} email address {college.get('Email', 'N/A')}",
                        'metadata': {'type': 'email', 'college': college_name},
                        'answer': f"{college_name} email: {college.get('Email', 'Not available')}"
                    },
                    {
                        'text': f"{college_name} is a {college['Type']} college institution",
                        'metadata': {'type': 'type', 'college': college_name},
                        'answer': f"{college_name} is a {college['Type'].lower()} college"
                    },
                    {
                        'text': f"{college_name} hostel accommodation {'available' if college['HostelAvailability'] else 'not available'}",
                        'metadata': {'type': 'hostel', 'college': college_name},
                        'answer': f"Hostel is {'available' if college['HostelAvailability'] else 'not available'} at {college_name}"
                    }
                ])
                
                # Add course and fee information
                for dept in college.get('Departments', []):
                    for course in dept.get('Courses', []):
                        documents.extend([
                            {
                                'text': f"{college_name} offers {course['Name']} course program with {course.get('TotalSeats', 'N/A')} seats",
                                'metadata': {'type': 'courses', 'college': college_name},
                                'answer': f"{college_name} offers {course['Name']} with {course.get('TotalSeats', 'N/A')} total seats"
                            },
                            {
                                'text': f"{course['Name']} fee cost at {college_name} is {course['Fee']} rupees",
                                'metadata': {'type': 'fee', 'college': college_name, 'course': course['Name']},
                                'answer': f"Fee for {course['Name']} at {college_name} is Rs. {course['Fee']:,}"
                            },
                            {
                                'text': f"{course['Name']} at {college_name} has {course.get('PassPercentage', 'N/A')}% pass rate success",
                                'metadata': {'type': 'pass_rate', 'college': college_name, 'course': course['Name']},
                                'answer': f"{course['Name']} at {college_name} has {course.get('PassPercentage', 'N/A')}% pass rate"
                            },
                            {
                                'text': f"{course['Name']} rating at {college_name} is {course.get('Rating', 'N/A')} stars quality",
                                'metadata': {'type': 'rating', 'college': college_name, 'course': course['Name']},
                                'answer': f"{course['Name']} at {college_name} has {course.get('Rating', 'N/A')}/5.0 rating"
                            },
                            {
                                'text': f"{college_name} {course['Name']} admission process entrance {course.get('AdmissionProcess', 'N/A')}",
                                'metadata': {'type': 'admission', 'college': college_name, 'course': course['Name']},
                                'answer': f"Admission for {course['Name']} at {college_name}: {course.get('AdmissionProcess', 'Not specified')}"
                            },
                            {
                                'text': f"{course['Name']} at {college_name} offers {'internship opportunities' if course.get('InternshipOpportunities', False) else 'no internship program'}",
                                'metadata': {'type': 'internship', 'college': college_name, 'course': course['Name']},
                                'answer': f"Internship opportunities {'are available' if course.get('InternshipOpportunities', False) else 'are not available'} for {course['Name']} at {college_name}"
                            },
                            {
                                'text': f"{college_name} {course['Name']} scholarship {course.get('GeneralScholarship', course.get('GereralScholarship', 'N/A'))}% merit based",
                                'metadata': {'type': 'scholarship', 'college': college_name, 'course': course['Name']},
                                'answer': f"Scholarship for {course['Name']} at {college_name}: {course.get('GeneralScholarship', course.get('GereralScholarship', 'Information not available'))}"
                            }
                        ])
            
            self.documents = documents
            
            # Create embeddings
            print(f"🔢 Creating embeddings for {len(documents)} documents...")
            texts = [doc['text'] for doc in documents]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Build FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
            
            print(f"✓ Built FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            print(f"Failed to build knowledge base: {e}")
            return False
    
    def predict(self, question, k=3):
        """Search for most relevant answer using FAISS"""
        if not self.model or not self.index:
            return None
            
        try:
            # Encode the question
            question_embedding = self.model.encode([question], convert_to_numpy=True)
            
            # Search for similar documents
            distances, indices = self.index.search(question_embedding.astype('float32'), k)
            
            # Get the best match
            best_idx = indices[0][0]
            best_distance = distances[0][0]
            
            # Convert distance to confidence (lower distance = higher confidence)
            confidence = max(0.0, 1.0 - (best_distance / 2.0))  # Normalize distance
            
            # Only return if confidence is reasonable
            if confidence < 0.5:
                return None
            
            best_doc = self.documents[best_idx]
            
            return {
                'answer': best_doc['answer'],
                'confidence': confidence,
                'source': 'faiss',
                'metadata': best_doc['metadata']
            }
        except Exception as e:
            print(f"FAISS prediction error: {e}")
            return None

# ========================================
# RETRIEVER
# ========================================

class Retriever:
    def __init__(self, df):
        self.df = df
        
        # Initialize FAISS retriever
        self.faiss_qa = None
        if FAISS_AVAILABLE:
            self.faiss_qa = FAISSQARetriever()
            try:
                # Attempt to load the FAISS model
                model_loaded = self.faiss_qa.load_model()
                if not model_loaded:
                    self.faiss_qa = None
            except Exception as e:
                print(f"⚠️ FAISS model initialization failed: {e}")
                self.faiss_qa = None
    
    def retrieve(self, parsed_query, k=5):
        # Handle direct questions about specific colleges
        if parsed_query.get('type') == 'direct_question':
            result = self._handle_direct_question(parsed_query)
            
            # If rule-based fails and FAISS is available, try FAISS approach
            if result is None and self.faiss_qa:
                faiss_result = self.faiss_qa.predict(parsed_query.get('original_query', ''))
                if faiss_result:
                    return {
                        'answer': faiss_result['answer'],
                        'confidence': faiss_result['confidence'],
                        'source': 'faiss'
                    }
            
            return result
        
        # Handle recommendation queries
        return self._handle_recommendation(parsed_query, k)
    
    def _handle_direct_question(self, parsed_query):
        college_name = parsed_query['college_name']
        question_type = parsed_query['question_type']
        
        # Find the college in the data
        college_data = self.df[self.df['CollegeName'] == college_name]
        
        if college_data.empty:
            return []
        
        # Get the first row (all rows for same college should have same basic info)
        college_info = college_data.iloc[0]
        
        # Format response based on question type
        response_info = {
            'type': 'direct_answer',
            'college_name': college_name,
            'question_type': question_type
        }
        
        if question_type == 'location':
            response_info['answer'] = f"{college_name} is located at {college_info['Location']}"
        elif question_type == 'contact':
            response_info['answer'] = f"{college_name} contact number: {college_info['ContactNumber']}"
        elif question_type == 'email':
            response_info['answer'] = f"{college_name} email: {college_info['Email']}"
        elif question_type == 'hostel':
            hostel_status = "available" if college_info['HostelAvailability'] else "not available"
            response_info['answer'] = f"Hostel facility is {hostel_status} at {college_name}"
        elif question_type == 'courses':
            courses = college_data['CourseName'].unique()
            courses_list = ', '.join(courses[:5])  # Show first 5 courses
            if len(courses) > 5:
                courses_list += f" and {len(courses)-5} more"
            response_info['answer'] = f"{college_name} offers: {courses_list}"
        elif question_type == 'fees':
            fees = college_data[['CourseName', 'Fee']].drop_duplicates()
            if len(fees) == 1:
                fee_value = fees.iloc[0].get('OriginalFee', fees.iloc[0]['Fee'])
                response_info['answer'] = f"{college_name} fee: Rs. {fee_value:,.0f} for {fees.iloc[0]['CourseName']}"
            else:
                response_info['answer'] = f"{college_name} fees vary by course. Here are some examples:\n"
                for _, row in fees.head(3).iterrows():
                    fee_value = row.get('OriginalFee', row['Fee'])
                    response_info['answer'] += f"• {row['CourseName']}: Rs. {fee_value:,.0f}\n"
        elif question_type == 'seats':
            # Use original seat values
            seat_data = []
            for _, row in college_data.iterrows():
                seats = row.get('OriginalSeats', row['TotalSeats'])
                seat_data.append({'course': row['CourseName'], 'seats': seats})
            
            if len(college_data) == 1:
                response_info['answer'] = f"{college_name} has {seat_data[0]['seats']} seats for {seat_data[0]['course']}"
            else:
                total_seats = sum(row['seats'] for row in seat_data)
                response_info['answer'] = f"{college_name} has {total_seats} total seats across all courses:\n"
                for seat_info in seat_data:
                    response_info['answer'] += f"• {seat_info['course']}: {seat_info['seats']} seats\n"
        elif question_type == 'rating':
            # Use original rating
            original_rating = college_data.iloc[0].get('OriginalRating', college_data.iloc[0]['Rating'])
            response_info['answer'] = f"{college_name} has an average rating of {original_rating:.1f}/5.0"
        elif question_type == 'scholarship':
            scholarships = []
            if college_info['GeneralScholarship']:
                scholarships.append(f"General scholarship: {college_info['GeneralScholarship']}%")
            if college_info['SemesterScholarship']:
                scholarships.append(f"Semester scholarship: {college_info['SemesterScholarship']}")
            if scholarships:
                response_info['answer'] = f"{college_name} offers:\n• " + "\n• ".join(scholarships)
            else:
                response_info['answer'] = f"No specific scholarship information available for {college_name}"
        elif question_type == 'internship':
            internship_courses = college_data[college_data['InternshipOpportunities'] == True]['CourseName'].unique()
            if len(internship_courses) > 0:
                response_info['answer'] = f"{college_name} offers internship opportunities for: {', '.join(internship_courses)}"
            else:
                response_info['answer'] = f"No internship opportunities listed for {college_name}"
        elif question_type == 'pass_rate':
            avg_pass = college_data.get('OriginalPassPercentage', college_data['PassPercentage']).mean()
            response_info['answer'] = f"{college_name} has an average pass rate of {avg_pass:.1f}%"
        elif question_type == 'admission':
            admission_processes = college_data['AdmissionProcess'].unique()
            if len(admission_processes) == 1:
                response_info['answer'] = f"{college_name} admission process: {admission_processes[0]}"
            else:
                response_info['answer'] = f"{college_name} admission processes: {', '.join(admission_processes)}"
        elif question_type == 'type':
            response_info['answer'] = f"{college_name} is a {college_info['Type'].lower()} college"
        elif question_type == 'duration':
            durations = college_data['DurationInYears'].unique()
            if len(durations) == 1:
                response_info['answer'] = f"{college_name} programs are {durations[0]} years long"
            else:
                response_info['answer'] = f"{college_name} program durations: {', '.join([f'{d} years' for d in durations if d])}"
        else:  # general
            response_info['answer'] = f"""{college_name} is a {college_info['Type'].lower()} college located at {college_info['Location']}.
Contact: {college_info['ContactNumber']} | Email: {college_info['Email']}
Hostel: {'Available' if college_info['HostelAvailability'] else 'Not Available'}"""
        
        return [response_info]
    
    def _handle_recommendation(self, parsed_query, k):
        filtered = self.df.copy()
        
        if parsed_query.get('location'):
            filtered = filtered[filtered['Location'] == parsed_query['location']]
        if parsed_query.get('college_type'):
            filtered = filtered[filtered['Type'] == parsed_query['college_type']]
        if parsed_query.get('course'):
            filtered = filtered[filtered['CourseName'] == parsed_query['course']]
        
        if 'fee_range' in parsed_query:
            min_fee, max_fee = parsed_query['fee_range']
            # Use original fee values for filtering
            fee_column = 'OriginalFee' if 'OriginalFee' in filtered.columns else 'Fee'
            filtered = filtered[(filtered[fee_column] >= min_fee) & (filtered[fee_column] <= max_fee)]
        
        facilities = parsed_query.get('facilities', {})
        if facilities.get('hostel'):
            filtered = filtered[filtered['HostelAvailability'] == True]
        if facilities.get('scholarship'):
            filtered = filtered[
                (filtered['GeneralScholarship'] == True) | 
                (filtered['SemesterScholarship'] == True)
            ]
        
        # Sort by rating and take top results
        filtered = filtered.sort_values('Rating', ascending=False)
        
        results = []
        for _, row in filtered.head(k).iterrows():
            results.append({
                'type': 'recommendation',
                'college_name': row['CollegeName'],
                'course_name': row['CourseName'],
                'location': row['Location'],
                'college_type': row['Type'],
                'fee': row.get('OriginalFee', row['Fee']),
                'rating': row['Rating'],
                'hostel': row['HostelAvailability'],
                'scholarship': bool(row['GeneralScholarship']) or bool(row['SemesterScholarship'])
            })
        
        return results

# ========================================
# CHATBOT
# ========================================

class Chatbot:
    def __init__(self, retriever, parser):
        self.retriever = retriever
        self.parser = parser
        self.current_results = []
    
    def chat(self, message):
        intent = self._detect_intent(message)
        
        if intent == 'greeting':
            return self._greeting()
        elif intent == 'details':
            return self._details(message)
        else:
            return self._search(message)
    
    def _detect_intent(self, message):
        msg_lower = message.lower()
        # Be more specific about greetings - only detect standalone greeting words
        greeting_words = ['hi', 'hello', 'hey', 'namaste']
        if any(f' {g} ' in f' {msg_lower} ' or msg_lower.startswith(g) for g in greeting_words):
            return 'greeting'
        # Only detect details intent if explicitly asking for details about a numbered option
        if any(k in msg_lower for k in ['tell me more', 'details']) and any(c in msg_lower for c in ['1', '2', '3', '4', '5']):
            return 'details'
        return 'search'
    
    def _greeting(self):
        return """Namaste! 🙏 I'm your college recommendation assistant.

I can help you find colleges based on:
• Location (Kathmandu, Pokhara, etc.)
• Course (Engineering, BBA, etc.)
• Fee range (low, moderate, high)
• Facilities (hostel, scholarship)

What are you looking for?"""
    
    def _search(self, message):
        parsed = self.parser.parse(message)
        results = self.retriever.retrieve(parsed, k=5)
        self.current_results = results
        
        if not results:
            return "No colleges match your criteria. Try different filters!"
        
        # Handle direct questions
        if results and results[0].get('type') == 'direct_answer':
            return results[0]['answer']
        
        # Handle recommendations
        response = f"Found {len(results)} colleges:\n\n"
        for i, r in enumerate(results, 1):
            response += f"{i}. {r['college_name']}\n"
            response += f"   📚 {r['course_name']}\n"
            response += f"   📍 {r['location']} | {r['college_type']}\n"
            response += f"   💰 Rs. {r['fee']:,.0f}\n"
            response += f"   ⭐ Rating: {r['rating']}\n\n"
        
        return response + "Type a number (1-5) for details!"
    
    def _details(self, message):
        match = re.search(r'\b([1-5])\b', message)
        if match and self.current_results:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(self.current_results):
                c = self.current_results[idx]
                
                # Skip details for direct answers
                if c.get('type') == 'direct_answer':
                    return "This was a direct answer. Ask another question!"
                
                return f"""**{c['college_name']}**
📚 {c['course_name']}
📍 {c['location']}
💰 Fee: Rs. {c['fee']:,.0f}
⭐ Rating: {c['rating']}
🏠 Hostel: {'Yes' if c['hostel'] else 'No'}
💵 Scholarship: {'Yes' if c['scholarship'] else 'No'}"""
        return "Please specify which college (1-5)."

# ========================================
# MAIN
# ========================================

def main():
    print("\n" + "="*60)
    print("🎓 COLLEGE RECOMMENDATION CHATBOT - ADVANCED AI")
    print("="*60 + "\n")
    
    try:
        # Load data
        result = load_and_preprocess_data(CONFIG['data_path'])
        if len(result) == 6:
            df, scaler, encoder, item_features, query_input_cols, cat_feature_names = result
        else:
            df = result[0]
            print("Running with advanced AI features")
        
        # Try to use production chatbot system with message type detection
        try:
            from app.production_chatbot import create_production_chatbot
            chatbot = create_production_chatbot(df)
            print("✅ Production AI Pipeline Loaded!")
            print("   🧠 Message Type Detection")
            print("   🔄 Pipeline Routing")
            print("   🎯 Context-Aware Responses\n")
        except ImportError as e:
            print(f"⚠️  Production features not available ({e})")
            print("   Falling back to standard chatbot...\n")
            # Initialize fallback chatbot (works with or without ML)
        parser = QueryParser(df)
        retriever = Retriever(df)
        chatbot = Chatbot(retriever, parser)
        
        print("\n✅ Chatbot ready!\n")
        print("Examples:")
        print("• Where is Sagarmatha Engineering College located?")
        print("• Engineering colleges in Kathmandu")
        print("• Contact of Pulchowk Engineering Campus")
        print("\nType 'quit' to exit\n" + "-"*60 + "\n")
        
        # Chat loop
        while True:
            user_input = input("💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            response = chatbot.chat(user_input)
            print(f"\n🤖 Bot: {response}\n")
    
    except FileNotFoundError:
        print("❌ Error: full_data.json not found!")
        print("Please make sure the data file is in the same directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()