"""
Safe Intent Detector - Fallback system without transformers
Uses keyword-based classification when transformers fail
"""

from typing import Dict, Tuple
import re

class SafeIntentDetector:
    """
    Safe intent detector that doesn't rely on heavy ML libraries
    Perfect fallback when transformer loading fails
    """
    
    def __init__(self):
        self.model_ready = True  # Always ready
        
        # Intent keyword mapping
        self.intent_patterns = {
            'location': [
                'where', 'located', 'address', 'location', 'place', 'situated', 'find'
            ],
            'fees': [
                'fee', 'cost', 'price', 'expensive', 'cheap', 'tuition', 'money', 'budget'
            ],
            'contact': [
                'contact', 'phone', 'email', 'call', 'reach', 'number', 'telephone'
            ],
            'recommendation': [
                'suggest', 'recommend', 'best', 'good', 'which', 'choose', 'help'
            ],
            'courses': [
                'course', 'program', 'degree', 'study', 'subject', 'engineering', 'department'
            ],
            'facilities': [
                'hostel', 'facility', 'accommodation', 'lab', 'library', 'canteen', 'wifi'
            ],
            'admission': [
                'admission', 'entrance', 'apply', 'join', 'enroll', 'eligibility', 'cutoff'
            ],
            'rating': [
                'rating', 'rank', 'quality', 'reputation', 'review', 'performance'
            ],
            'scholarship': [
                'scholarship', 'financial aid', 'funding', 'grant', 'stipend'
            ],
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'
            ]
        }
        
        # Entity patterns
        self.college_keywords = [
            'sagarmatha', 'sec', 'pulchowk', 'ioe', 'ku', 'kathmandu university', 
            'ace', 'advanced college', 'kantipur', 'himalaya', 'thapathali'
        ]
        
        self.location_keywords = [
            'kathmandu', 'lalitpur', 'bhaktapur', 'pokhara', 'chitwan', 
            'butwal', 'biratnagar', 'janakpur'
        ]
        
        self.course_keywords = [
            'computer engineering', 'civil engineering', 'electrical',
            'mechanical', 'electronics', 'bba', 'mba'
        ]
    
    def detect_intent(self, prompt: str) -> Tuple[str, float, Dict]:
        """
        Detect intent using keyword matching
        
        Returns:
            (intent, confidence, details)
        """
        prompt_lower = prompt.lower().strip()
        
        # Calculate scores for each intent
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1
                    # Bonus for exact matches
                    if keyword == prompt_lower:
                        score += 2
            intent_scores[intent] = score
        
        # Find best intent
        best_intent = 'recommendation'  # default fallback
        max_score = 0
        
        for intent, score in intent_scores.items():
            if score > max_score:
                max_score = score
                best_intent = intent
        
        # Calculate confidence
        total_words = len(prompt_lower.split())
        confidence = min(max_score / max(total_words, 1), 1.0)
        
        # Ensure minimum confidence for clear cases
        if max_score > 0:
            confidence = max(confidence, 0.3)
        else:
            confidence = 0.1  # Very low confidence for unknown
        
        # Extract entities
        entities = self._extract_entities(prompt_lower)
        
        return best_intent, confidence, {
            'entities': entities,
            'keyword_scores': intent_scores,
            'method': 'keyword_matching',
            'total_matches': max_score
        }
    
    def _extract_entities(self, prompt: str) -> Dict:
        """Extract entities using keyword matching"""
        entities = {
            'college_mentioned': None,
            'location_mentioned': None,
            'course_mentioned': None
        }
        
        # Find college mentions
        for college in self.college_keywords:
            if college in prompt:
                entities['college_mentioned'] = college
                break
        
        # Find location mentions
        for location in self.location_keywords:
            if location in prompt:
                entities['location_mentioned'] = location
                break
        
        # Find course mentions
        for course in self.course_keywords:
            if course in prompt:
                entities['course_mentioned'] = course
                break
        
        return entities

# Test function
if __name__ == "__main__":
    detector = SafeIntentDetector()
    
    test_prompts = [
        "Where is Sagarmatha Engineering College located?",
        "How much does Computer Engineering cost?",
        "Recommend good engineering colleges",
        "Hello, good morning!",
        "What courses are available?",
        "Does SEC have hostel facilities?"
    ]
    
    print("🧪 Testing Safe Intent Detector:")
    for prompt in test_prompts:
        intent, confidence, details = detector.detect_intent(prompt)
        print(f"'{prompt}' → {intent} ({confidence:.3f})")
        if details['entities']:
            print(f"  Entities: {details['entities']}")