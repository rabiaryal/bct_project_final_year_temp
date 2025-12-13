"""
FAISS-based Direct Answer System
Uses FAISS for ultra-fast similarity search over college facts
"""

import json
import numpy as np
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu")


class FAISSDirectAnswerSystem:
    """
    FAISS-based retrieval system for direct question answering.
    
    Benefits over basic similarity search:
    - O(log n) retrieval vs O(n) for brute force
    - GPU acceleration support
    - Memory-efficient indexing
    - Persistent index storage
    """
    
    def __init__(self, data_path: str = 'full_data.json', 
                 index_path: str = './models/faiss_index',
                 model_name: str = 'all-MiniLM-L6-v2'):
        
        self.data_path = data_path
        self.index_path = index_path
        self.model_name = model_name
        
        # Knowledge base
        self.facts: List[Dict] = []
        self.fact_texts: List[str] = []
        
        # FAISS index
        self.index = None
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Sentence transformer for encoding
        self._encoder = None
        
        # Load or build index
        if os.path.exists(os.path.join(index_path, 'faiss.index')):
            self._load_index()
        else:
            self._build_index()
    
    @property
    def encoder(self) -> SentenceTransformer:
        """Lazy load encoder"""
        if self._encoder is None:
            print(f"📦 Loading encoder: {self.model_name}")
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder
    
    def _build_index(self):
        """Build FAISS index from college data"""
        print("🔧 Building FAISS index...")
        
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available. Using fallback.")
            self._load_facts()
            return
        
        # Load facts
        self._load_facts()
        
        if not self.facts:
            print("❌ No facts to index")
            return
        
        # Encode all facts
        print(f"🔢 Encoding {len(self.fact_texts)} facts...")
        embeddings = self.encoder.encode(
            self.fact_texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        # Using IndexFlatIP (Inner Product) for cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        
        # Save index
        self._save_index()
    
    def _load_facts(self):
        """Load and structure facts from college data"""
        try:
            with open(self.data_path, 'r') as f:
                colleges = json.load(f)
        except FileNotFoundError:
            print(f"❌ Could not find {self.data_path}")
            return
        
        # Extract facts from each college
        for college in colleges:
            college_facts = self._extract_facts(college)
            self.facts.extend(college_facts)
        
        # Create text list for encoding
        self.fact_texts = [f['fact_text'] for f in self.facts]
        
        print(f"📚 Loaded {len(self.facts)} facts")
    
    def _extract_facts(self, college: Dict) -> List[Dict]:
        """Extract structured facts from a single college"""
        facts = []
        name = college['Name']
        
        # Location fact
        facts.append({
            'type': 'location',
            'college': name,
            'fact_text': f"{name} is located in {college['Location']}.",
            'data': {
                'location': college['Location'],
                'latitude': college.get('Latitude'),
                'longitude': college.get('Longitude')
            }
        })
        
        # Contact fact
        facts.append({
            'type': 'contact',
            'college': name,
            'fact_text': f"Contact {name}: Phone {college['ContactNumber']}, Email {college.get('Email', 'N/A')}.",
            'data': {
                'phone': college['ContactNumber'],
                'email': college.get('Email')
            }
        })
        
        # Type fact
        facts.append({
            'type': 'college_type',
            'college': name,
            'fact_text': f"{name} is a {college['Type']} college.",
            'data': {'type': college['Type']}
        })
        
        # Hostel fact
        hostel_status = 'available' if college['HostelAvailability'] else 'not available'
        facts.append({
            'type': 'hostel',
            'college': name,
            'fact_text': f"Hostel is {hostel_status} at {name}.",
            'data': {'hostel': college['HostelAvailability']}
        })
        
        # Course-level facts
        for dept in college.get('Departments', []):
            dept_name = dept['Name']
            
            for course in dept.get('Courses', []):
                course_name = course['Name']
                
                # Fee fact (include department for better matching)
                fee = course.get('Fee', 0)
                facts.append({
                    'type': 'fee',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"The fee for {course_name} ({dept_name} department) at {name} is Rs. {fee:,} per year.",
                    'data': {
                        'fee': fee,
                        'course': course_name,
                        'department': dept_name,
                        'fee_category': self._fee_category(fee)
                    }
                })
                
                # Duration fact
                duration = course.get('DurationInYears', 4)
                facts.append({
                    'type': 'duration',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"{course_name} ({dept_name}) at {name} is a {duration}-year program.",
                    'data': {'duration': duration, 'course': course_name, 'department': dept_name}
                })
                
                # Seats fact
                seats = course.get('TotalSeats', 'N/A')
                facts.append({
                    'type': 'seats',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"{name} has {seats} seats for {course_name} ({dept_name} department).",
                    'data': {'seats': seats, 'course': course_name, 'department': dept_name}
                })
                
                # Rating fact
                rating = course.get('Rating', 'N/A')
                facts.append({
                    'type': 'rating',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"{course_name} ({dept_name}) at {name} has a rating of {rating}.",
                    'data': {'rating': rating, 'course': course_name, 'department': dept_name}
                })
                
                # Admission fact
                admission = course.get('AdmissionProcess', 'entrance exam based')
                facts.append({
                    'type': 'admission',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"Admission to {course_name} ({dept_name}) at {name} is {admission}.",
                    'data': {'admission_process': admission, 'course': course_name, 'department': dept_name}
                })
                
                # Scholarship fact
                scholarship = course.get('GeneralScholarship', course.get('GereralScholarship', 'merit-based'))
                facts.append({
                    'type': 'scholarship',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"{name} offers {scholarship} scholarship for {course_name} ({dept_name}).",
                    'data': {'scholarship': scholarship, 'course': course_name, 'department': dept_name}
                })
                
                # Internship fact
                internship = course.get('InternshipOpportunities', False)
                internship_text = 'provides internship opportunities' if internship else 'has limited internship programs'
                facts.append({
                    'type': 'internship',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"{name} {internship_text} for {course_name} ({dept_name}) students.",
                    'data': {'internship': internship, 'course': course_name, 'department': dept_name}
                })
                
                # Cutoff fact
                cutoff = course.get('CutOffRank', 'varies')
                facts.append({
                    'type': 'cutoff',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"The cutoff rank for {course_name} ({dept_name}) at {name} is {cutoff}.",
                    'data': {'cutoff': cutoff, 'course': course_name, 'department': dept_name}
                })
                
                # Pass percentage fact
                pass_rate = course.get('PassPercentage', 'N/A')
                facts.append({
                    'type': 'placement',
                    'college': name,
                    'department': dept_name,
                    'course': course_name,
                    'fact_text': f"{course_name} ({dept_name}) at {name} has a pass rate of {pass_rate}%.",
                    'data': {'pass_rate': pass_rate, 'course': course_name, 'department': dept_name}
                })
        
        return facts
    
    def _fee_category(self, fee: int) -> str:
        """Categorize fee"""
        if fee < 200000:
            return 'affordable'
        elif fee < 500000:
            return 'moderate'
        elif fee < 800000:
            return 'expensive'
        else:
            return 'premium'
    
    def _save_index(self):
        """Save FAISS index and facts to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_path, 'faiss.index'))
        
        # Save facts
        with open(os.path.join(self.index_path, 'facts.pkl'), 'wb') as f:
            pickle.dump({
                'facts': self.facts,
                'fact_texts': self.fact_texts
            }, f)
        
        print(f"💾 Index saved to {self.index_path}")
    
    def _load_index(self):
        """Load FAISS index and facts from disk"""
        print(f"📦 Loading FAISS index from {self.index_path}...")
        
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available")
            self._load_facts()
            return
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(self.index_path, 'faiss.index'))
            
            # Load facts
            with open(os.path.join(self.index_path, 'facts.pkl'), 'rb') as f:
                data = pickle.load(f)
                self.facts = data['facts']
                self.fact_texts = data['fact_texts']
            
            print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            self._build_index()
    
    def search(self, query: str, k: int = 5, 
               intent_filter: Optional[str] = None,
               entity_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for relevant facts using FAISS
        
        Args:
            query: User's question
            k: Number of results to return
            intent_filter: Optional intent to filter by (fee, location, hostel, etc.)
            entity_filter: Optional entities to filter by (college_name, course_name, etc.)
            
        Returns:
            List of relevant facts with scores
        """
        if not FAISS_AVAILABLE or self.index is None:
            return self._fallback_search(query, k, intent_filter, entity_filter)
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search with extra results for filtering
        search_k = min(k * 3, len(self.facts))  # Get more for filtering
        scores, indices = self.index.search(query_embedding, search_k)
        
        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            
            fact = self.facts[idx].copy()
            fact['similarity_score'] = float(score)
            
            # Apply filters
            if intent_filter and not self._matches_intent(fact, intent_filter):
                fact['similarity_score'] *= 0.5  # Reduce score for non-matching intent
            
            if entity_filter:
                entity_boost = self._entity_match_boost(fact, entity_filter)
                fact['similarity_score'] += entity_boost
            
            results.append(fact)
        
        # Re-sort after boosting
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:k]
    
    def _matches_intent(self, fact: Dict, intent: str) -> bool:
        """Check if fact matches the intent"""
        intent_lower = intent.lower()
        fact_type = fact.get('type', '').lower()
        
        # Direct type match
        if fact_type == intent_lower:
            return True
        
        # Intent to type mapping
        intent_mapping = {
            'fee': ['fee', 'fees'],
            'location': ['location'],
            'hostel': ['hostel', 'facilities'],
            'admission': ['admission'],
            'rating': ['rating', 'academics'],
            'courses': ['course', 'duration'],
            'facilities': ['hostel', 'facilities'],
            'cutoff': ['cutoff'],
            'scholarship': ['scholarship'],
            'internship': ['internship', 'opportunities'],
            'placement': ['placement', 'pass_rate'],
            'contact': ['contact'],
            'duration': ['duration'],
            'seats': ['seats', 'availability'],
        }
        
        for key, types in intent_mapping.items():
            if key in intent_lower and fact_type in types:
                return True
        
        return False
    
    def _normalize_course_department(self, text: str) -> List[str]:
        """
        Normalize course/department names for better matching.
        Handles variations like 'civil', 'civil engineering', 'BE CIVIL', etc.
        Returns list of normalized terms for matching.
        """
        text_lower = text.lower().strip()
        
        # Mapping of user terms to database terms
        term_mappings = {
            'civil': ['civil', 'civil engineering'],
            'computer': ['computer', 'computer engineering', 'electronics and computer', 'computer science'],
            'electrical': ['electrical', 'electrical engineering'],
            'mechanical': ['mechanical', 'mechanical engineering'],
            'electronics': ['electronics', 'electronics engineering', 'electronics and computer'],
            'architecture': ['architecture'],
            'it': ['information technology', 'it'],
            'software': ['software', 'software engineering'],
            'bce': ['computer engineering'],
            'bct': ['computer engineering', 'electronics and computer'],
            'bex': ['electronics and computer'],
        }
        
        terms = []
        for key, values in term_mappings.items():
            if key in text_lower:
                terms.extend(values)
        
        if not terms:
            terms.append(text_lower)
        
        return terms
    
    def _entity_match_boost(self, fact: Dict, entities: Dict) -> float:
        """
        Calculate boost based on entity matches.
        Improved to handle department matching and course/department normalization.
        """
        boost = 0.0
        match_found = False
        
        # College name match (highest priority)
        if entities.get('college_name'):
            college_entity = entities['college_name'].lower()
            fact_college = fact.get('college', '').lower()
            
            # More strict matching - avoid partial matches like 'nce' matching 'THAPATHALI ENGINEERING CAMPUS'
            college_words = college_entity.replace(' ', '').lower()
            fact_college_words = fact_college.replace(' ', '').lower()
            
            if college_words in fact_college_words or fact_college_words in college_words:
                boost += 0.5
                match_found = True
            elif college_entity in fact_college:
                boost += 0.5
                match_found = True
        
        # Department name match (HIGH priority for course-level facts)
        if entities.get('department_name'):
            dept_entity = entities['department_name'].lower()
            fact_dept = fact.get('department', '').lower()
            fact_course = fact.get('course', '').lower()
            fact_data_dept = fact.get('data', {}).get('department', '').lower()
            
            # Get normalized terms for matching
            normalized_terms = self._normalize_course_department(dept_entity)
            
            dept_matched = False
            for term in normalized_terms:
                if term in fact_dept or term in fact_course or term in fact_data_dept:
                    dept_matched = True
                    break
            
            if dept_matched:
                boost += 0.4  # High boost for department match
                match_found = True
            else:
                # Penalize if department is specified but doesn't match
                boost -= 0.3
        
        # Course name match
        if entities.get('course_name'):
            course_entity = entities['course_name'].lower()
            fact_course = fact.get('course', '').lower()
            fact_dept = fact.get('department', '').lower()
            
            # Get normalized terms for matching
            normalized_terms = self._normalize_course_department(course_entity)
            
            course_matched = False
            for term in normalized_terms:
                if term in fact_course or term in fact_dept:
                    course_matched = True
                    break
            
            if course_matched:
                boost += 0.3
                match_found = True
            else:
                # Penalize if course is specified but doesn't match
                boost -= 0.2
        
        # Location match
        if entities.get('location'):
            location_entity = entities['location'].lower()
            fact_location = fact.get('data', {}).get('location', '').lower()
            
            if location_entity in fact_location:
                boost += 0.2
        
        return boost
    
    def _fallback_search(self, query: str, k: int, 
                        intent_filter: Optional[str],
                        entity_filter: Optional[Dict]) -> List[Dict]:
        """Fallback search without FAISS"""
        print("⚠️ Using fallback search (no FAISS)")
        
        # Simple keyword matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for fact in self.facts:
            fact_text_lower = fact['fact_text'].lower()
            
            # Calculate simple overlap score
            fact_words = set(fact_text_lower.split())
            overlap = len(query_words & fact_words)
            score = overlap / max(len(query_words), 1)
            
            # Apply filters
            if intent_filter and self._matches_intent(fact, intent_filter):
                score += 0.3
            
            if entity_filter:
                score += self._entity_match_boost(fact, entity_filter)
            
            fact_copy = fact.copy()
            fact_copy['similarity_score'] = score
            results.append(fact_copy)
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:k]
    
    def get_answer(self, query: str, intent: str = None, 
                   entities: Dict = None, k: int = 5) -> Dict[str, Any]:
        """
        Get a direct answer to a question
        
        Args:
            query: User's question
            intent: Detected sub-intent (fee, location, etc.)
            entities: Extracted entities
            k: Number of facts to retrieve
            
        Returns:
            Dict with answer, facts, and confidence
        """
        # Extract sub-intent from full intent (e.g., "direct_question.fee" -> "fee")
        if intent and '.' in intent:
            intent = intent.split('.')[-1]
        
        # Search for relevant facts
        relevant_facts = self.search(query, k=k, intent_filter=intent, entity_filter=entities)
        
        if not relevant_facts:
            return {
                'answer': "I couldn't find information about that. Could you rephrase your question?",
                'facts': [],
                'confidence': 0.0
            }
        
        # Generate answer from facts
        answer = self._generate_answer(query, intent, relevant_facts, entities)
        
        # Calculate confidence
        avg_score = sum(f.get('similarity_score', 0) for f in relevant_facts) / len(relevant_facts)
        
        return {
            'answer': answer,
            'facts': relevant_facts,
            'confidence': min(avg_score, 1.0),
            'top_fact': relevant_facts[0] if relevant_facts else None
        }
    
    def _generate_answer(self, query: str, intent: str, 
                        facts: List[Dict], entities: Dict) -> str:
        """Generate natural answer from retrieved facts"""
        if not facts:
            return "I couldn't find relevant information."
        
        top_fact = facts[0]
        college = top_fact.get('college', 'the college')
        
        # Intent-specific answer templates
        if intent == 'fee':
            fee_facts = [f for f in facts if f.get('type') == 'fee']
            if fee_facts:
                fact = fee_facts[0]
                data = fact.get('data', {})
                fee = data.get('fee', 'N/A')
                course = data.get('course', 'the program')
                return f"The fee for {course} at {fact.get('college')} is **Rs. {fee:,}** per year. This is considered {data.get('fee_category', 'standard')} in the market."
        
        elif intent == 'location':
            loc_facts = [f for f in facts if f.get('type') == 'location']
            if loc_facts:
                fact = loc_facts[0]
                data = fact.get('data', {})
                return f"{fact.get('college')} is located in **{data.get('location')}**."
        
        elif intent == 'hostel':
            hostel_facts = [f for f in facts if f.get('type') == 'hostel']
            if hostel_facts:
                fact = hostel_facts[0]
                data = fact.get('data', {})
                has_hostel = data.get('hostel', False)
                status = "**Yes**, hostel is available" if has_hostel else "**No**, hostel is not available"
                return f"{status} at {fact.get('college')}."
        
        elif intent == 'admission':
            adm_facts = [f for f in facts if f.get('type') == 'admission']
            if adm_facts:
                fact = adm_facts[0]
                data = fact.get('data', {})
                return f"Admission to {data.get('course', 'programs')} at {fact.get('college')} is through **{data.get('admission_process', 'entrance exam')}**."
        
        elif intent == 'rating':
            rating_facts = [f for f in facts if f.get('type') == 'rating']
            if rating_facts:
                fact = rating_facts[0]
                data = fact.get('data', {})
                return f"{data.get('course', 'The program')} at {fact.get('college')} has a rating of **{data.get('rating')}**."
        
        elif intent == 'scholarship':
            sch_facts = [f for f in facts if f.get('type') == 'scholarship']
            if sch_facts:
                fact = sch_facts[0]
                data = fact.get('data', {})
                return f"{fact.get('college')} offers **{data.get('scholarship')}** scholarship for {data.get('course', 'students')}."
        
        elif intent == 'cutoff':
            cut_facts = [f for f in facts if f.get('type') == 'cutoff']
            if cut_facts:
                fact = cut_facts[0]
                data = fact.get('data', {})
                return f"The cutoff rank for {data.get('course', 'admission')} at {fact.get('college')} is **{data.get('cutoff')}**."
        
        elif intent == 'duration':
            dur_facts = [f for f in facts if f.get('type') == 'duration']
            if dur_facts:
                fact = dur_facts[0]
                data = fact.get('data', {})
                return f"{data.get('course', 'The program')} at {fact.get('college')} is a **{data.get('duration')}-year** program."
        
        elif intent == 'seats':
            seat_facts = [f for f in facts if f.get('type') == 'seats']
            if seat_facts:
                fact = seat_facts[0]
                data = fact.get('data', {})
                return f"{fact.get('college')} has **{data.get('seats')} seats** for {data.get('course', 'the program')}."
        
        elif intent == 'internship':
            int_facts = [f for f in facts if f.get('type') == 'internship']
            if int_facts:
                fact = int_facts[0]
                data = fact.get('data', {})
                has_int = data.get('internship', False)
                status = "**provides internship opportunities**" if has_int else "has **limited internship programs**"
                return f"{fact.get('college')} {status} for {data.get('course', 'students')}."
        
        elif intent == 'contact':
            cont_facts = [f for f in facts if f.get('type') == 'contact']
            if cont_facts:
                fact = cont_facts[0]
                data = fact.get('data', {})
                return f"Contact {fact.get('college')}:\n📞 Phone: **{data.get('phone')}**\n📧 Email: **{data.get('email', 'N/A')}**"
        
        # Default: return fact text
        answer_parts = []
        seen_colleges = set()
        
        for fact in facts[:3]:
            college = fact.get('college')
            if college not in seen_colleges:
                answer_parts.append(f"• {fact['fact_text']}")
                seen_colleges.add(college)
        
        return "Here's what I found:\n\n" + "\n".join(answer_parts)
    
    def get_multi_intent_answer(self, query: str, sub_intents: List[Tuple[str, float]], 
                                 entities: Dict = None, k_per_intent: int = 3) -> Dict[str, Any]:
        """
        Get answers for MULTIPLE sub-intents in a single query.
        
        For queries like "What is the fee and location of NCIT?" this retrieves
        and generates answers for both fee and location.
        
        Args:
            query: User's question
            sub_intents: List of (sub_intent, confidence) tuples
            entities: Extracted entities
            k_per_intent: Number of facts to retrieve per intent
            
        Returns:
            Dict with combined answer, per-intent facts, and confidence
        """
        if not sub_intents:
            return self.get_answer(query, intent='general_info', entities=entities)
        
        # Collect facts for each intent
        intent_results = {}
        all_facts = []
        
        for sub_intent, confidence in sub_intents:
            facts = self.search(
                query, 
                k=k_per_intent, 
                intent_filter=sub_intent, 
                entity_filter=entities
            )
            
            if facts:
                intent_results[sub_intent] = {
                    'facts': facts,
                    'confidence': confidence,
                    'answer_part': self._generate_intent_section(sub_intent, facts, entities)
                }
                all_facts.extend(facts)
        
        if not intent_results:
            return {
                'answer': "I couldn't find information about that. Could you rephrase your question?",
                'facts': [],
                'confidence': 0.0,
                'is_multi_intent': False
            }
        
        # Generate combined answer
        combined_answer = self._generate_multi_intent_answer(query, intent_results, entities)
        
        # Calculate overall confidence
        avg_confidence = sum(r['confidence'] for r in intent_results.values()) / len(intent_results)
        
        return {
            'answer': combined_answer,
            'facts': all_facts,
            'intent_results': intent_results,
            'intents_answered': list(intent_results.keys()),
            'confidence': avg_confidence,
            'is_multi_intent': len(intent_results) > 1
        }
    
    def _generate_intent_section(self, intent: str, facts: List[Dict], entities: Dict) -> str:
        """Generate answer section for a single intent"""
        if not facts:
            return f"No information found for {intent}."
        
        top_fact = facts[0]
        data = top_fact.get('data', {})
        college = top_fact.get('college', 'the college')
        
        # Intent-specific formatting
        if intent == 'fee':
            fee = data.get('fee', 'N/A')
            course = data.get('course', 'the program')
            if isinstance(fee, (int, float)):
                return f"**Fee**: Rs. {fee:,} per year for {course}"
            return f"**Fee**: {fee} for {course}"
        
        elif intent == 'location':
            location = data.get('location', 'N/A')
            return f"**Location**: {location}"
        
        elif intent == 'hostel':
            has_hostel = data.get('hostel', False)
            status = "Available ✅" if has_hostel else "Not available ❌"
            return f"**Hostel**: {status}"
        
        elif intent == 'admission':
            process = data.get('admission_process', 'entrance exam based')
            return f"**Admission**: Through {process}"
        
        elif intent == 'rating':
            rating = data.get('rating', 'N/A')
            return f"**Rating**: {rating}"
        
        elif intent == 'duration':
            duration = data.get('duration', 4)
            return f"**Duration**: {duration} years"
        
        elif intent == 'seats':
            seats = data.get('seats', 'N/A')
            return f"**Seats**: {seats}"
        
        elif intent == 'scholarship':
            scholarship = data.get('scholarship', 'merit-based')
            return f"**Scholarship**: {scholarship}"
        
        elif intent == 'internship':
            has_intern = data.get('internship', False)
            status = "Available ✅" if has_intern else "Limited ❌"
            return f"**Internship**: {status}"
        
        elif intent == 'cutoff':
            cutoff = data.get('cutoff', 'varies')
            return f"**Cutoff Rank**: {cutoff}"
        
        elif intent == 'contact':
            phone = data.get('phone', 'N/A')
            email = data.get('email', 'N/A')
            return f"**Contact**: 📞 {phone} | 📧 {email}"
        
        elif intent == 'placement':
            pass_rate = data.get('pass_rate', 'N/A')
            return f"**Pass Rate**: {pass_rate}%"
        
        else:
            return f"**{intent.title()}**: {top_fact.get('fact_text', 'Information available')}"
    
    def _generate_multi_intent_answer(self, query: str, intent_results: Dict, entities: Dict) -> str:
        """Generate combined answer for multiple intents"""
        # Get college name from entities or facts
        college_name = None
        if entities and entities.get('college_name'):
            college_name = entities['college_name']
        else:
            # Try to get from first fact
            for intent_data in intent_results.values():
                if intent_data['facts']:
                    college_name = intent_data['facts'][0].get('college')
                    break
        
        # Build response
        parts = []
        
        if college_name:
            parts.append(f"## 📚 Information about {college_name.upper()}\n")
        else:
            parts.append("## 📚 Here's what I found:\n")
        
        # Add each intent's answer
        for intent, data in intent_results.items():
            if data.get('answer_part'):
                parts.append(data['answer_part'])
        
        # Add helpful footer
        intents_list = list(intent_results.keys())
        if len(intents_list) > 1:
            parts.append(f"\n---\n💡 *Answered {len(intents_list)} questions: {', '.join(intents_list)}*")
        
        return "\n".join(parts)
    
    def rebuild_index(self):
        """Force rebuild the FAISS index"""
        # Clear existing
        self.facts = []
        self.fact_texts = []
        self.index = None
        
        # Rebuild
        self._build_index()


# Convenience function
def create_faiss_system(data_path: str = 'full_data.json') -> FAISSDirectAnswerSystem:
    """Create FAISS direct answer system"""
    return FAISSDirectAnswerSystem(data_path=data_path)
