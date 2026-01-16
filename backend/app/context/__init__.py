"""
Context Management Module

Provides comprehensive context management for multi-turn conversational AI interactions.
Handles intent tracking, entity management, slot filling, and conversation state.

Key Components:
- ConversationContext: Core context data structure
- ContextManager: Main context management interface  
- ContextRules: Business rules for context updates
- EntityInfo: Structured entity information with metadata

Usage:
    from app.context import context_manager, ConversationContext
    
    # Get or create context
    context = context_manager.get_context("user-123")
    
    # Update intent
    context = context_manager.update_intent("user-123", "search_college", 0.9)
    
    # Update entities
    entities = [Entity(type="LOCATION", value="Kathmandu", confidence=0.95)]
    context = context_manager.update_entities("user-123", entities)
"""

from .context_models import (
    ConversationContext,
    IntentInfo,
    EntityInfo, 
    EntitySource,
    SlotInfo,
    PolicyState,
    ContextFlags,
    ContextStatus,
    ContextUpdateRequest
)

from .context_rules import (
    ContextRules,
    ContextUpdateRules
)

from .context_manager import (
    ContextManager,
    InMemoryStorage,
    context_manager  # Global instance
)

# Keep backwards compatibility
from app.context.tracker import DialogueTracker, Turn

__all__ = [
    # Core models
    'ConversationContext',
    'IntentInfo', 
    'EntityInfo',
    'EntitySource',
    'SlotInfo',
    'PolicyState', 
    'ContextFlags',
    'ContextStatus',
    'ContextUpdateRequest',
    
    # Rules and logic
    'ContextRules',
    'ContextUpdateRules',
    
    # Manager and storage
    'ContextManager',
    'InMemoryStorage',
    'context_manager',
    
    # Backwards compatibility
    'DialogueTracker',
    'Turn'
]