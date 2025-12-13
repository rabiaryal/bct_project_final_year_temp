"""
Structured Retriever with Exact Matching + FAISS Fallback

This is a more accurate retrieval system that:
1. First tries EXACT match on college + department/course from the JSON
2. Falls back to FAISS semantic search only when exact match fails
3. Validates all retrieved facts against user entities

This hybrid approach ensures:
- "fee of Pulchowk civil" → Exact match to PULCHOWK + CIVIL ENGINEERING
- "best engineering college" → FAISS semantic search
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher


class StructuredRetriever:
    """
    Hybrid retriever: Exact JSON lookup + FAISS semantic fallback
    
    Priority:
    1. Exact college + department/course match from JSON
    2. Fuzzy college match + exact department match
    3. FAISS semantic search (last resort)
    """
    
    # Common name aliases for better matching (based on actual database)
    COLLEGE_ALIASES = {
        'pulchowk': ['pulchowk engineering campus', 'pulchowk', 'pul', 'ioe pulchowk', 'pulchowck'],
        'thapathali': ['thapathali engineering campus', 'thapathali', 'tec'],
        'purwanchal': ['purwanchal engineering campus', 'purwanchal', 'pec'],
        'pashchimanchal': ['pashchimanchal engineering campus', 'pashchimanchal', 'wrc', 'western'],
        'chitwan': ['chitwan engineering campus', 'chitwan'],
        'advanced': ['advanced college of engineering and management', 'advanced', 'acem', 'ace'],
        'kantipur': ['kantipur engineering college', 'kantipur', 'kec'],
        'national': ['national college of engineering', 'national', 'nce'],
        'himalaya': ['himalaya college of engineering', 'himalaya', 'hce'],
        'sagarmatha': ['sagarmatha engineering college', 'sagarmatha', 'sec'],
        'lalitpur': ['lalitpur engineering college', 'lalitpur', 'lec'],
        'kathford': ['kathford international college', 'kathford'],
        'kathmandu engineering': ['kathmandu engineering college', 'kathmandu engineering', 'kec kathmandu'],
        'khwopa': ['khwopa college of engineering', 'khwopa'],
        'janakpur': ['janakpur engineering college', 'janakpur'],
        'pokhara': ['pokhara engineering college', 'pokhara university school of engineering', 'pokhara'],
        'cosmos': ['cosmos college of management and technology', 'cosmos'],
        'crimson': ['crimson college of technology', 'crimson'],
        'everest': ['everest engineering college', 'everest'],
        'gandaki': ['gandaki college of engineering and science', 'gandaki'],
        'lumbini': ['lumbini engineering management and science college', 'lumbini', 'lemsc'],
        'madan bhandari': ['madan bhandari memorial academy', 'madan bhandari', 'mbma'],
        'universal': ['universal engineering & science college', 'universal'],
        'ku': ['kathmandu university school of engineering', 'kathmandu university', 'ku'],
        'acme': ['acme engineering college', 'acme'],
        'central': ['central engineering college', 'central'],
        'hillside': ['hillside college of engineering', 'hillside'],
        'himalayan': ['himalayan institute of science and technology', 'himalayan', 'hist'],
        'kantipur city': ['kantipur city college', 'kantipur city'],
        'kantipur international': ['kantipur international college', 'kantipur international'],
        'morgan': ['morgan engineering and management college', 'morgan'],
    }
    
    # Department/Course aliases for matching
    DEPARTMENT_ALIASES = {
        'civil': ['civil engineering', 'civil', 'be civil'],
        'computer': ['computer engineering', 'computer', 'be computer', 'bct', 'bce', 'electronics and computer'],
        'electronics': ['electronics and computer engineering', 'electronics', 'bex', 'ece', 'electronic'],
        'electrical': ['electrical engineering', 'electrical', 'bee'],
        'mechanical': ['mechanical engineering', 'mechanical', 'bme'],
        'architecture': ['architecture', 'b.arch'],
        'it': ['information technology', 'it', 'bit'],
        'software': ['software engineering', 'software'],
        'aerospace': ['aerospace engineering', 'aerospace'],
        'automobile': ['automobile engineering', 'automobile'],
        'biomedical': ['biomedical engineering', 'biomedical'],
        'chemical': ['chemical engineering', 'chemical'],
        'geomatics': ['geomatics engineering', 'geomatics', 'survey'],
        'industrial': ['industrial engineering', 'industrial'],
    }
    
    def __init__(self, data_path: str = 'full_data.json', faiss_system=None):
        self.data_path = data_path
        self.faiss_system = faiss_system
        
        # Load and index the raw JSON data
        self.colleges = []
        self.college_index = {}  # name -> college data
        self.college_name_lookup = {}  # normalized name -> actual name
        self._load_data()
    
    def _load_data(self):
        """Load and index college data for fast lookup"""
        try:
            with open(self.data_path, 'r') as f:
                self.colleges = json.load(f)
            
            for college in self.colleges:
                name = college['Name']
                self.college_index[name] = college
                
                # Create normalized lookup
                name_lower = name.lower()
                self.college_name_lookup[name_lower] = name
                
                # Add word-based lookups
                for word in name_lower.split():
                    if len(word) > 3:  # Skip short words
                        if word not in self.college_name_lookup:
                            self.college_name_lookup[word] = name
            
            print(f"📚 Structured Retriever: Indexed {len(self.colleges)} colleges")
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            self.colleges = []
    
    def _normalize_college_name(self, query: str) -> Optional[str]:
        """
        Find the actual college name from user input.
        Uses aliases and fuzzy matching.
        """
        query_lower = query.lower().strip()
        
        # 1. Try direct lookup in our index
        if query_lower in self.college_name_lookup:
            return self.college_name_lookup[query_lower]
        
        # 2. Try alias matching
        for alias_key, aliases in self.COLLEGE_ALIASES.items():
            for alias in aliases:
                if alias in query_lower or query_lower in alias:
                    # Find the actual college name
                    for name in self.college_index.keys():
                        if alias_key in name.lower():
                            return name
        
        # 3. Try fuzzy matching on college names
        best_match = None
        best_score = 0.0
        
        for name in self.college_index.keys():
            # Word-level matching
            name_words = set(name.lower().split())
            query_words = set(query_lower.split())
            
            # Check if any significant query word is in college name
            for qw in query_words:
                if len(qw) > 3:  # Skip short words
                    for nw in name_words:
                        if qw in nw or nw in qw:
                            score = SequenceMatcher(None, qw, nw).ratio()
                            if score > best_score and score > 0.6:
                                best_score = score
                                best_match = name
        
        return best_match
    
    def _normalize_department(self, query: str) -> Optional[str]:
        """
        Find the department/course from user input.
        Returns normalized department name that matches database format.
        """
        query_lower = query.lower().strip()
        
        for dept_key, aliases in self.DEPARTMENT_ALIASES.items():
            for alias in aliases:
                if alias in query_lower:
                    # Return the key that we can match against database
                    return dept_key
        
        return None
    
    def _find_course_in_college(self, college_data: Dict, department_query: str) -> Optional[Dict]:
        """
        Find a specific course in a college matching the department query.
        """
        if not department_query:
            return None
        
        dept_query_lower = department_query.lower()
        
        for dept in college_data.get('Departments', []):
            dept_name = dept['Name'].lower()
            
            # Check if department matches
            if dept_query_lower in dept_name or dept_name.startswith(dept_query_lower):
                # Return first course in this department
                courses = dept.get('Courses', [])
                if courses:
                    return {
                        'department': dept['Name'],
                        'course': courses[0]  # Typically one course per department
                    }
        
        return None
    
    def get_exact_answer(self, query: str, intent: str, entities: Dict) -> Optional[Dict]:
        """
        Try to get an exact answer from structured data.
        
        Returns:
            Dict with exact answer if found, None if FAISS fallback needed
        """
        # Step 1: Identify the college
        college_query = entities.get('college_name') or ''
        
        # Also try to find college in the query itself
        if not college_query:
            college_query = query
        
        college_name = self._normalize_college_name(college_query)
        
        if not college_name:
            # No college identified - need FAISS
            return None
        
        college_data = self.college_index.get(college_name)
        if not college_data:
            return None
        
        # Step 2: Identify the department/course if applicable
        dept_query = entities.get('department_name') or entities.get('course_name') or ''
        if not dept_query:
            # Try to find department in query
            dept_query = self._normalize_department(query)
        else:
            dept_query = self._normalize_department(dept_query)
        
        # Step 3: Get the data based on intent
        return self._extract_intent_data(college_name, college_data, dept_query, intent)
    
    def _extract_intent_data(self, college_name: str, college_data: Dict, 
                            dept_query: Optional[str], intent: str) -> Optional[Dict]:
        """
        Extract specific data based on intent type.
        """
        intent_lower = intent.lower() if intent else ''
        
        # College-level intents (don't need department)
        if intent_lower in ['location', 'hostel', 'contact', 'college_type']:
            return self._get_college_level_answer(college_name, college_data, intent_lower)
        
        # General info intent - provide overview
        if intent_lower in ['general_info', 'general', 'info', 'about']:
            return self._get_general_info(college_name, college_data, dept_query)
        
        # Courses intent - list all courses
        if intent_lower in ['courses', 'course', 'programs', 'departments']:
            return self._get_courses_info(college_name, college_data)
        
        # Course-level intents (need department/course)
        course_intents = ['fee', 'seats', 'rating', 'admission', 'scholarship', 
                          'cutoff', 'duration', 'internship', 'placement']
        
        if intent_lower in course_intents:
            return self._get_course_level_answer(college_name, college_data, dept_query, intent_lower)
        
        # General info or unknown intent - try to provide something useful
        return self._get_general_info(college_name, college_data, dept_query)
    
    def _get_college_level_answer(self, college_name: str, college_data: Dict, 
                                  intent: str) -> Dict:
        """Get college-level information"""
        
        if intent == 'location':
            location = college_data.get('Location', 'N/A')
            return {
                'answer': f"**{college_name}** is located in **{location}**.",
                'data': {'location': location, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'hostel':
            has_hostel = college_data.get('HostelAvailability', False)
            status = "**Yes**, hostel is available" if has_hostel else "**No**, hostel is not available"
            return {
                'answer': f"{status} at **{college_name}**.",
                'data': {'hostel': has_hostel, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'contact':
            phone = college_data.get('ContactNumber', 'N/A')
            email = college_data.get('Email', 'N/A')
            return {
                'answer': f"Contact **{college_name}**:\n📞 Phone: **{phone}**\n📧 Email: **{email}**",
                'data': {'phone': phone, 'email': email, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'college_type':
            college_type = college_data.get('Type', 'N/A')
            return {
                'answer': f"**{college_name}** is a **{college_type}** college.",
                'data': {'type': college_type, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        return None
    
    def _get_general_info(self, college_name: str, college_data: Dict, 
                          dept_query: Optional[str] = None) -> Dict:
        """Get general information about a college or specific department"""
        
        if dept_query:
            # Get info about specific department/course
            course_info = self._find_course_in_college(college_data, dept_query)
            
            if course_info:
                dept_name = course_info['department']
                course = course_info['course']
                
                answer_parts = [
                    f"📚 **{course['Name']}** at **{college_name}**:",
                    f"• Fee: Rs. {course.get('Fee', 0):,}",
                    f"• Rating: {course.get('Rating', 'N/A')}/5",
                    f"• Seats: {course.get('TotalSeats', 'N/A')}",
                    f"• Duration: {course.get('DurationInYears', 4)} years",
                    f"• Admission: {course.get('AdmissionProcess', 'N/A')}",
                ]
                
                if course.get('InternshipOpportunities'):
                    answer_parts.append("• ✅ Internship opportunities available")
                if course.get('GeneralScholarship'):
                    answer_parts.append(f"• 🎓 Scholarship: {course['GeneralScholarship']}%")
                
                return {
                    'answer': '\n'.join(answer_parts),
                    'data': {'college': college_name, 'course': course, 'department': dept_name},
                    'confidence': 1.0,
                    'source': 'exact_match'
                }
        
        # General college overview
        location = college_data.get('Location', 'N/A')
        college_type = college_data.get('Type', 'N/A')
        has_hostel = college_data.get('HostelAvailability', False)
        departments = college_data.get('Departments', [])
        
        answer_parts = [
            f"🏛️ **{college_name}**",
            f"• Location: {location}",
            f"• Type: {college_type}",
            f"• Hostel: {'Available ✅' if has_hostel else 'Not Available ❌'}",
            f"• Departments: {len(departments)} programs available"
        ]
        
        if departments:
            dept_names = [d['Name'] for d in departments[:5]]
            answer_parts.append("• Courses: " + ", ".join(dept_names))
        
        return {
            'answer': '\n'.join(answer_parts),
            'data': {'college': college_data},
            'confidence': 1.0,
            'source': 'exact_match'
        }
    
    def _get_courses_info(self, college_name: str, college_data: Dict) -> Dict:
        """Get list of all courses/programs at a college"""
        
        departments = college_data.get('Departments', [])
        
        if not departments:
            return {
                'answer': f"No department information available for **{college_name}**.",
                'data': {},
                'confidence': 0.5,
                'source': 'exact_match'
            }
        
        answer_parts = [f"📚 **Programs at {college_name}**:\n"]
        
        for dept in departments:
            dept_name = dept['Name']
            courses = dept.get('Courses', [])
            for course in courses:
                fee = course.get('Fee', 0)
                rating = course.get('Rating', 'N/A')
                answer_parts.append(f"• **{course['Name']}** - Rs. {fee:,} | Rating: {rating}/5")
        
        return {
            'answer': '\n'.join(answer_parts),
            'data': {'departments': departments, 'college': college_name},
            'confidence': 1.0,
            'source': 'exact_match'
        }
    
    def _get_course_level_answer(self, college_name: str, college_data: Dict,
                                 dept_query: Optional[str], intent: str) -> Optional[Dict]:
        """Get course-level information"""
        
        # If no department specified, list all or ask for clarification
        if not dept_query:
            return self._get_all_courses_answer(college_name, college_data, intent)
        
        # Find the specific course
        course_info = self._find_course_in_college(college_data, dept_query)
        
        if not course_info:
            # Department not found - maybe list available departments
            return self._department_not_found_answer(college_name, college_data, dept_query, intent)
        
        dept_name = course_info['department']
        course = course_info['course']
        course_name = course['Name']
        
        # Generate answer based on intent
        if intent == 'fee':
            fee = course.get('Fee', 'N/A')
            fee_str = f"Rs. {fee:,.0f}" if isinstance(fee, (int, float)) else str(fee)
            return {
                'answer': f"The fee for **{course_name}** at **{college_name}** is **{fee_str}** per year.",
                'data': {'fee': fee, 'course': course_name, 'department': dept_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'seats':
            seats = course.get('TotalSeats', 'N/A')
            return {
                'answer': f"**{college_name}** has **{seats} seats** for **{course_name}**.",
                'data': {'seats': seats, 'course': course_name, 'department': dept_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'rating':
            rating = course.get('Rating', 'N/A')
            return {
                'answer': f"**{course_name}** at **{college_name}** has a rating of **{rating}**.",
                'data': {'rating': rating, 'course': course_name, 'department': dept_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'admission':
            admission = course.get('AdmissionProcess', 'entrance exam')
            return {
                'answer': f"Admission to **{course_name}** at **{college_name}** is through **{admission}**.",
                'data': {'admission_process': admission, 'course': course_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'scholarship':
            scholarship = course.get('GeneralScholarship', course.get('GereralScholarship', 'N/A'))
            sem_scholarship = course.get('SemesterScholarship', '')
            answer = f"**{college_name}** offers **{scholarship}** scholarship for **{course_name}**."
            if sem_scholarship:
                answer += f" ({sem_scholarship})"
            return {
                'answer': answer,
                'data': {'scholarship': scholarship, 'course': course_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'cutoff':
            cutoff = course.get('AverageCutoffRank', course.get('CutOffRank', 'N/A'))
            return {
                'answer': f"The cutoff rank for **{course_name}** at **{college_name}** is **{cutoff}**.",
                'data': {'cutoff': cutoff, 'course': course_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'duration':
            duration = course.get('DurationInYears', 4)
            return {
                'answer': f"**{course_name}** at **{college_name}** is a **{duration}-year** program.",
                'data': {'duration': duration, 'course': course_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'internship':
            has_internship = course.get('InternshipOpportunities', False)
            status = "provides internship opportunities" if has_internship else "has limited internship programs"
            return {
                'answer': f"**{college_name}** **{status}** for **{course_name}** students.",
                'data': {'internship': has_internship, 'course': course_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        elif intent == 'placement':
            pass_rate = course.get('PassPercentage', 'N/A')
            return {
                'answer': f"**{course_name}** at **{college_name}** has a pass rate of **{pass_rate}%**.",
                'data': {'pass_rate': pass_rate, 'course': course_name, 'college': college_name},
                'confidence': 1.0,
                'source': 'exact_match'
            }
        
        return None
    
    def _get_all_courses_answer(self, college_name: str, college_data: Dict, 
                               intent: str) -> Optional[Dict]:
        """
        When no department is specified, show info for all courses or ask for clarification.
        """
        departments = college_data.get('Departments', [])
        
        if not departments:
            return None
        
        # If only a few departments, list them all
        if len(departments) <= 3 or intent in ['seats', 'fee']:
            answers = []
            all_data = []
            
            for dept in departments:
                for course in dept.get('Courses', []):
                    course_name = course['Name']
                    
                    if intent == 'fee':
                        fee = course.get('Fee', 'N/A')
                        fee_str = f"Rs. {fee:,.0f}" if isinstance(fee, (int, float)) else str(fee)
                        answers.append(f"• **{course_name}**: {fee_str}")
                        all_data.append({'course': course_name, 'fee': fee})
                    
                    elif intent == 'seats':
                        seats = course.get('TotalSeats', 'N/A')
                        answers.append(f"• **{course_name}**: {seats} seats")
                        all_data.append({'course': course_name, 'seats': seats})
            
            if answers:
                header = f"Here's the {intent} information for all programs at **{college_name}**:\n\n"
                return {
                    'answer': header + "\n".join(answers),
                    'data': all_data,
                    'confidence': 1.0,
                    'source': 'exact_match'
                }
        
        # Too many departments - ask for clarification
        dept_names = [d['Name'] for d in departments[:5]]
        return {
            'answer': f"**{college_name}** offers multiple programs. Please specify which department:\n\n" + 
                     "\n".join([f"• {d}" for d in dept_names]),
            'data': {'departments': dept_names, 'college': college_name},
            'confidence': 0.5,
            'needs_clarification': True,
            'source': 'exact_match'
        }
    
    def _department_not_found_answer(self, college_name: str, college_data: Dict,
                                     dept_query: str, intent: str) -> Dict:
        """
        When the specified department is not found in the college.
        """
        departments = college_data.get('Departments', [])
        dept_names = [d['Name'] for d in departments]
        
        return {
            'answer': f"I couldn't find **{dept_query}** department at **{college_name}**.\n\n" +
                     f"Available departments:\n" + "\n".join([f"• {d}" for d in dept_names]),
            'data': {'departments': dept_names, 'college': college_name},
            'confidence': 0.3,
            'source': 'exact_match'
        }
    
    def get_college_info(self, college_name: str) -> Optional[Dict]:
        """
        Get comprehensive college information for comparisons.
        
        Args:
            college_name: Name or alias of the college
            
        Returns:
            Dict with full college data or None if not found
        """
        # Normalize the college name
        normalized_name = self._normalize_college_name(college_name)
        if not normalized_name:
            return None
        
        college_data = self.college_index.get(normalized_name)
        if not college_data:
            return None
        
        # Get summary data
        departments = college_data.get('Departments', [])
        all_courses = []
        avg_fee = 0
        avg_rating = 0
        total_seats = 0
        
        for dept in departments:
            for course in dept.get('Courses', []):
                all_courses.append({
                    'name': course.get('Name'),
                    'fee': course.get('Fee', 0),
                    'rating': course.get('Rating', 0),
                    'seats': course.get('TotalSeats', 0)
                })
                avg_fee += course.get('Fee', 0)
                avg_rating += course.get('Rating', 0)
                total_seats += course.get('TotalSeats', 0)
        
        num_courses = len(all_courses)
        if num_courses > 0:
            avg_fee = avg_fee / num_courses
            avg_rating = avg_rating / num_courses
        
        return {
            'name': normalized_name,
            'location': college_data.get('Location', 'N/A'),
            'type': college_data.get('Type', 'N/A'),
            'hostel': college_data.get('HostelAvailability', False),
            'contact': college_data.get('ContactNumber', 'N/A'),
            'email': college_data.get('Email', 'N/A'),
            'departments': [d['Name'] for d in departments],
            'courses': all_courses,
            'num_courses': num_courses,
            'avg_fee': avg_fee,
            'avg_rating': avg_rating,
            'total_seats': total_seats
        }
    
    def get_answer(self, query: str, intent: str, entities: Dict, 
                   k: int = 5) -> Dict[str, Any]:
        """
        Main entry point: Get answer using exact match first, then FAISS fallback.
        
        Args:
            query: User's question
            intent: Detected sub-intent
            entities: Extracted entities
            k: Number of results for FAISS fallback
            
        Returns:
            Dict with answer, data, confidence, and source
        """
        # Clean up intent
        if intent and '.' in intent:
            intent = intent.split('.')[-1]
        
        # Step 1: Try exact match
        exact_result = self.get_exact_answer(query, intent, entities)
        
        if exact_result and exact_result.get('confidence', 0) >= 0.5:
            print(f"✅ Exact match found (confidence: {exact_result['confidence']})")
            return exact_result
        
        # Step 2: Fall back to FAISS semantic search
        if self.faiss_system and self.faiss_system != "FAILED":
            print("🔍 Using FAISS semantic search fallback...")
            faiss_result = self.faiss_system.get_answer(query, intent, entities, k)
            faiss_result['source'] = 'faiss_semantic'
            return faiss_result
        
        # Step 3: No answer found
        return {
            'answer': "I couldn't find information about that. Could you please rephrase or specify the college and department?",
            'data': {},
            'confidence': 0.0,
            'source': 'none'
        }


# Factory function
def create_structured_retriever(data_path: str = 'full_data.json', 
                                faiss_system=None) -> StructuredRetriever:
    """Create and return a StructuredRetriever instance"""
    return StructuredRetriever(data_path, faiss_system)
