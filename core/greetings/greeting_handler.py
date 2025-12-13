"""
Greeting Handler
Handles greeting intents directly without passing through entity extraction or other processing.
Optimized for fast response times.
"""

import random
from typing import Dict, Any, Optional
from datetime import datetime


class GreetingHandler:
    """
    Fast greeting response handler
    
    Design principle: Greetings should be handled immediately without
    expensive NLP processing. If a greeting contains a task (e.g., "Hi, recommend me a college"),
    the task should be extracted and processed separately.
    """
    
    # Greeting responses - varied for natural feel
    GREETING_RESPONSES = [
        "Hello! 🙏 I'm your college recommendation assistant. How can I help you today?",
        "Hi there! 👋 I'm here to help you find the perfect college in Nepal. What are you looking for?",
        "Namaste! 🙏 I can help you with college recommendations, comparisons, and information. What would you like to know?",
        "Hello! I'm your engineering college guide for Nepal. Ask me anything about colleges, courses, or admissions!",
        "Hi! 👋 Ready to help you explore engineering colleges in Nepal. What would you like to know?"
    ]
    
    # Time-based greetings
    MORNING_GREETINGS = [
        "Good morning! ☀️ Ready to help you explore engineering colleges in Nepal!",
        "Good morning! 🌅 How can I assist you with college information today?"
    ]
    
    AFTERNOON_GREETINGS = [
        "Good afternoon! 🌤️ Looking for college recommendations?",
        "Good afternoon! How can I help with your college search today?"
    ]
    
    EVENING_GREETINGS = [
        "Good evening! 🌙 I'm here to help with your college queries!",
        "Good evening! Looking for college information? I'm here to help!"
    ]
    
    # Response templates for greetings with context
    GREETING_WITH_HELP = """Hello! 🙏 I'm your college recommendation assistant.

I can help you with:
• 🎓 **College recommendations** based on your preferences
• 📍 **Location, contact, and fee** information
• 📚 **Course details** and admission requirements  
• 🏛️ **Facilities**, scholarships, and hostel info
• ⚖️ **Compare colleges** side by side

What would you like to know about engineering colleges in Nepal?"""

    def __init__(self):
        """Initialize greeting handler"""
        pass
    
    def handle(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle a greeting intent
        
        Args:
            prompt: The user's message
            context: Optional conversation context
            
        Returns:
            Dict with response and metadata
        """
        # Check if this is a pure greeting or greeting + task
        task = self._extract_task_from_greeting(prompt)
        
        if task:
            # This is a greeting + task, return info to process the task
            return {
                'type': 'greeting_with_task',
                'greeting_handled': True,
                'task_extracted': task,
                'response': None,  # Task handler will generate response
                'needs_further_processing': True,
                'pipeline': 'greeting_with_task'
            }
        
        # Pure greeting - respond immediately
        response = self._generate_greeting_response(prompt)
        
        return {
            'type': 'greeting',
            'response': response,
            'pipeline': 'greeting_only',
            'processing_time': 'instant',
            'needs_further_processing': False
        }
    
    def _generate_greeting_response(self, prompt: str) -> str:
        """
        Generate appropriate greeting response based on prompt and time
        """
        prompt_lower = prompt.lower().strip()
        
        # Check for help requests in greeting
        if any(word in prompt_lower for word in ['help', 'assist', 'what can you do', 'how can you help']):
            return self.GREETING_WITH_HELP
        
        # Check for time-based greetings
        if 'good morning' in prompt_lower:
            return random.choice(self.MORNING_GREETINGS)
        elif 'good afternoon' in prompt_lower:
            return random.choice(self.AFTERNOON_GREETINGS)
        elif 'good evening' in prompt_lower or 'good night' in prompt_lower:
            return random.choice(self.EVENING_GREETINGS)
        
        # Auto-detect time for generic greetings
        hour = datetime.now().hour
        if 5 <= hour < 12:
            # 50% chance to use time-based greeting
            if random.random() < 0.5:
                return random.choice(self.MORNING_GREETINGS)
        elif 12 <= hour < 17:
            if random.random() < 0.5:
                return random.choice(self.AFTERNOON_GREETINGS)
        elif 17 <= hour < 22:
            if random.random() < 0.5:
                return random.choice(self.EVENING_GREETINGS)
        
        # Default: random greeting
        return random.choice(self.GREETING_RESPONSES)
    
    def _extract_task_from_greeting(self, prompt: str) -> Optional[str]:
        """
        Check if greeting contains a task/question
        
        Examples:
        - "Hi, recommend me a college" -> "recommend me a college"
        - "Hello, what is the fee of NCIT" -> "what is the fee of NCIT"
        - "Hello" -> None (pure greeting)
        """
        prompt_lower = prompt.lower().strip()
        
        # Common greeting starters to strip
        greeting_prefixes = [
            'hello,', 'hi,', 'hey,', 'namaste,',
            'good morning,', 'good afternoon,', 'good evening,',
            'hello!', 'hi!', 'hey!', 'namaste!',
            'good morning!', 'good afternoon!', 'good evening!',
            'hello', 'hi', 'hey', 'namaste',
            'good morning', 'good afternoon', 'good evening',
        ]
        
        # Sort by length (longest first) to match longest prefix
        greeting_prefixes.sort(key=len, reverse=True)
        
        remaining_text = prompt_lower
        for prefix in greeting_prefixes:
            if remaining_text.startswith(prefix):
                remaining_text = remaining_text[len(prefix):].strip()
                break
        
        # If there's substantial remaining text, it's likely a task
        if len(remaining_text) > 10:
            # Check if it contains task-related words
            task_indicators = [
                'recommend', 'suggest', 'find', 'search', 'show', 'tell', 'list',
                'what', 'which', 'where', 'how', 'when', 'who', 'why',
                'compare', 'difference', 'better',
                'fee', 'cost', 'admission', 'location', 'hostel', 'scholarship',
                'course', 'program', 'rating', 'rank', 'cutoff',
                'college', 'university', 'campus', 'ioe', 'ncit', 'pulchowk'
            ]
            
            if any(word in remaining_text for word in task_indicators):
                # Find where the remaining text starts in original prompt
                # by locating the task in the original string
                original_lower = prompt.lower()
                start_idx = original_lower.find(remaining_text[:20])  # Match first 20 chars
                if start_idx != -1:
                    return prompt[start_idx:].strip()
                else:
                    # Fallback: return remaining text as-is
                    return remaining_text
        
        return None
    
    def is_pure_greeting(self, prompt: str) -> bool:
        """
        Quick check if a prompt is a pure greeting (no task)
        
        Useful for quick routing decisions
        """
        task = self._extract_task_from_greeting(prompt)
        return task is None


# Create singleton instance for easy import
greeting_handler = GreetingHandler()
