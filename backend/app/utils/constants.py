"""Application Constants"""

# Intent constants (updated to match actual model predictions)
INTENT_TYPES = [
    # Actual intent names from the model
    "GET_COLLEGE_INFO",
    "GET_COURSE_INFO", 
    "GET_ADMISSION_INFO",
    "GET_FEE_INFO",
    "GET_SCHOLARSHIP_INFO",
    "GET_PLACEMENT_INFO",
    "GET_FACILITY_INFO",
    "Get_college_location",
    "Get_contact_info",
    "Greeting",
    "Goodbye",
    "THANK_YOU",
    "Unknown",
    # Legacy intent names for backward compatibility
    "college_info",
    "course_info", 
    "admission_info",
    "fee_info",
    "scholarship_info",
    "placement_info",
    "facility_info",
    "location_info",
    "contact_info",
    "greeting",
    "goodbye",
    "thank_you",
    "unknown"
]

# Entity types
ENTITY_TYPES = [
    "college_mentioned",
    "course_mentioned", 
    "location_mentioned",
    "fee_mentioned",
    "facility_mentioned"
]

# Action constants
ACTION_TYPES = [
    "action_search_college",
    "action_search_course",
    "action_get_admission_info",
    "action_get_fee_info", 
    "action_get_placement_info",
    "action_provide_contact",
    "action_greet",
    "action_goodbye",
    "action_fallback"
]

# Response templates
RESPONSE_TEMPLATES = {
    "greeting": [
        "Hello! I'm here to help you with college information. What would you like to know?",
        "Hi! How can I assist you with your college search today?",
        "Welcome! I can help you find information about colleges and courses."
    ],
    "goodbye": [
        "Thank you for using our college information system. Have a great day!",
        "Goodbye! Feel free to ask if you need more information about colleges.",
        "Have a wonderful day! Come back anytime for college information."
    ],
    "fallback": [
        "I'm sorry, I didn't understand that. Could you please rephrase your question?",
        "I'm not sure about that. Could you ask in a different way?",
        "Could you clarify what you're looking for? I can help with college information."
    ]
}

# Database constants
MONGODB_COLLECTIONS = {
    "colleges": "college data",
    "courses": "courses",
    "users": "users"
}

# Session constants
SESSION_TIMEOUT = 1800  # 30 minutes
MAX_DIALOGUE_TURNS = 20