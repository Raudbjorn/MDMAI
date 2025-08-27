"""Context Management System for TTRPG Assistant.

This module provides comprehensive context management capabilities including:
- Context persistence with PostgreSQL backend
- Real-time synchronization for collaborative sessions
- Provider-agnostic context translation
- Versioning and rollback support
- Efficient compression and storage optimization
"""

from .context_manager import ContextManager
from .models import (
    Context,
    ContextState,
    ContextType,
    ContextVersion,
    ConversationContext,
    SessionContext,
    CollaborativeContext,
    ContextEvent,
    ContextDiff,
    ProviderContext,
)
from .persistence import ContextPersistenceLayer
from .serialization import ContextSerializer, ContextCompressor
from .synchronization import ContextSyncManager, EventBus
from .translation import ContextTranslator, ProviderAdapter
from .validation import ContextValidator

__all__ = [
    "ContextManager",
    "Context",
    "ContextState",
    "ContextType", 
    "ContextVersion",
    "ConversationContext",
    "SessionContext",
    "CollaborativeContext",
    "ContextEvent",
    "ContextDiff",
    "ProviderContext",
    "ContextPersistenceLayer",
    "ContextSerializer",
    "ContextCompressor",
    "ContextSyncManager",
    "EventBus",
    "ContextTranslator",
    "ProviderAdapter",
    "ContextValidator",
]