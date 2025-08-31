"""Data models for the Context Management System."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ContextType(Enum):
    """Types of contexts that can be managed."""
    
    CONVERSATION = "conversation"
    SESSION = "session"
    CAMPAIGN = "campaign"
    CHARACTER = "character"
    COLLABORATIVE = "collaborative"
    PROVIDER_SPECIFIC = "provider_specific"


class ContextState(Enum):
    """Lifecycle states for contexts."""
    
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"
    SYNCING = "syncing"
    CONFLICT = "conflict"


class CompressionType(Enum):
    """Supported compression algorithms."""
    
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving context conflicts."""
    
    LATEST_WINS = "latest_wins"
    MANUAL_MERGE = "manual_merge"
    PROVIDER_PRIORITY = "provider_priority"
    USER_CHOICE = "user_choice"
    AUTOMATIC_MERGE = "automatic_merge"


@dataclass
class ContextVersion:
    """Represents a version of a context with metadata."""
    
    version_id: str = field(default_factory=lambda: str(uuid4()))
    context_id: str = ""
    version_number: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    parent_version: Optional[str] = None
    branch: str = "main"
    checksum: str = ""
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class Context(BaseModel):
    """Base context model with core functionality."""
    
    context_id: str = Field(default_factory=lambda: str(uuid4()))
    context_type: ContextType
    title: str = ""
    description: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Versioning
    current_version: int = 1
    version_history: List[str] = Field(default_factory=list)
    
    # Ownership and permissions
    owner_id: Optional[str] = None
    collaborators: List[str] = Field(default_factory=list)
    permissions: Dict[str, List[str]] = Field(default_factory=dict)
    
    # State management
    state: ContextState = ContextState.ACTIVE
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    
    # Synchronization
    sync_status: Optional[str] = None
    last_sync: Optional[datetime] = None
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS
    
    # Provider integration
    provider_contexts: Dict[str, "ProviderContext"] = Field(default_factory=dict)
    
    # Storage optimization
    compression_type: CompressionType = CompressionType.ZSTD
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    
    # Lifecycle management
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    auto_archive_days: Optional[int] = 90
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def add_collaborator(self, user_id: str, permissions: List[str] = None) -> None:
        """Add a collaborator with specified permissions."""
        if user_id not in self.collaborators:
            self.collaborators.append(user_id)
        
        if permissions:
            self.permissions[user_id] = permissions
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        if self.owner_id == user_id:
            return True
        return permission in self.permissions.get(user_id, [])
    
    def should_archive(self) -> bool:
        """Check if context should be automatically archived."""
        if not self.auto_archive_days:
            return False
        
        days_inactive = (datetime.now(timezone.utc) - self.last_accessed).days
        return days_inactive >= self.auto_archive_days


class ConversationContext(Context):
    """Context for conversation threads."""
    
    context_type: ContextType = Field(default=ContextType.CONVERSATION, const=True)
    
    # Conversation-specific data
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    participants: List[str] = Field(default_factory=list)
    current_turn: int = 0
    max_turns: Optional[int] = None
    
    # AI Provider integration
    primary_provider: Optional[str] = None
    provider_settings: Dict[str, Any] = Field(default_factory=dict)
    model_parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # State tracking
    conversation_state: str = "active"
    last_message_at: Optional[datetime] = None
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to the conversation."""
        message["timestamp"] = datetime.now(timezone.utc).isoformat()
        message["turn"] = self.current_turn
        self.messages.append(message)
        self.current_turn += 1
        self.last_message_at = datetime.now(timezone.utc)
        self.last_modified = datetime.now(timezone.utc)
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.messages[-count:] if len(self.messages) > count else self.messages


class SessionContext(Context):
    """Context for user sessions."""
    
    context_type: ContextType = Field(default=ContextType.SESSION, const=True)
    
    # Session-specific data
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None
    active_conversations: List[str] = Field(default_factory=list)
    session_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Preferences
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    ui_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Activity tracking
    total_interactions: int = 0
    session_duration_seconds: int = 0
    last_interaction: Optional[datetime] = None
    
    def add_conversation(self, conversation_id: str) -> None:
        """Add a conversation to this session."""
        if conversation_id not in self.active_conversations:
            self.active_conversations.append(conversation_id)
    
    def record_interaction(self) -> None:
        """Record user interaction."""
        self.total_interactions += 1
        self.last_interaction = datetime.now(timezone.utc)
        self.last_modified = datetime.now(timezone.utc)


class CollaborativeContext(Context):
    """Context for collaborative sessions."""
    
    context_type: ContextType = Field(default=ContextType.COLLABORATIVE, const=True)
    
    # Collaboration-specific data
    room_id: str = Field(default_factory=lambda: str(uuid4()))
    active_participants: List[str] = Field(default_factory=list)
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Real-time synchronization
    sync_events: List[Dict[str, Any]] = Field(default_factory=list)
    pending_changes: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    
    # Conflict management
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    resolution_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Locking mechanism
    locked_by: Optional[str] = None
    locked_at: Optional[datetime] = None
    lock_timeout_seconds: int = 300
    
    def add_participant(self, user_id: str) -> None:
        """Add a participant to the collaborative session."""
        if user_id not in self.active_participants:
            self.active_participants.append(user_id)
    
    def remove_participant(self, user_id: str) -> None:
        """Remove a participant from the collaborative session."""
        if user_id in self.active_participants:
            self.active_participants.remove(user_id)
        
        # Release lock if held by this user
        if self.locked_by == user_id:
            self.release_lock()
    
    def acquire_lock(self, user_id: str) -> bool:
        """Acquire exclusive lock for editing."""
        if self.locked_by is None or self._is_lock_expired():
            self.locked_by = user_id
            self.locked_at = datetime.now(timezone.utc)
            return True
        return False
    
    def release_lock(self, user_id: Optional[str] = None) -> bool:
        """Release exclusive lock."""
        if user_id is None or self.locked_by == user_id:
            self.locked_by = None
            self.locked_at = None
            return True
        return False
    
    def _is_lock_expired(self) -> bool:
        """Check if current lock has expired."""
        if not self.locked_at:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.locked_at).total_seconds()
        return elapsed > self.lock_timeout_seconds


class ProviderContext(BaseModel):
    """Provider-specific context data."""
    
    provider_type: str
    provider_context_id: str = Field(default_factory=lambda: str(uuid4()))
    
    # Provider-specific data
    context_data: Dict[str, Any] = Field(default_factory=dict)
    format_version: str = "1.0"
    
    # Synchronization tracking
    last_sync: Optional[datetime] = None
    sync_status: str = "synced"
    
    # Translation metadata
    translation_metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_translation: bool = False
    
    # Performance tracking
    size_bytes: int = 0
    compression_ratio: float = 1.0
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextEvent(BaseModel):
    """Event for context change tracking."""
    
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    context_id: str
    event_type: str  # create, update, delete, sync, conflict, etc.
    
    # Event data
    user_id: Optional[str] = None
    changes: Dict[str, Any] = Field(default_factory=dict)
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    
    # Event metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "system"  # system, user, sync, ai_provider
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Synchronization
    propagated: bool = False
    propagation_targets: List[str] = Field(default_factory=list)


class ContextDiff(BaseModel):
    """Represents differences between context versions."""
    
    diff_id: str = Field(default_factory=lambda: str(uuid4()))
    context_id: str
    from_version: int
    to_version: int
    
    # Diff data
    added: Dict[str, Any] = Field(default_factory=dict)
    modified: Dict[str, Any] = Field(default_factory=dict)
    removed: List[str] = Field(default_factory=list)
    
    # Metadata
    diff_size_bytes: int = 0
    compression_savings: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextQuery(BaseModel):
    """Query model for context retrieval."""
    
    # Filters
    context_ids: Optional[List[str]] = None
    context_types: Optional[List[ContextType]] = None
    owner_id: Optional[str] = None
    collaborators: Optional[List[str]] = None
    states: Optional[List[ContextState]] = None
    
    # Search
    search_text: Optional[str] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Time range
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    modified_after: Optional[datetime] = None
    modified_before: Optional[datetime] = None
    accessed_after: Optional[datetime] = None
    
    # Pagination
    limit: int = 50
    offset: int = 0
    
    # Sorting
    sort_by: str = "last_modified"
    sort_order: str = "desc"  # asc or desc
    
    # Options
    include_archived: bool = False
    include_deleted: bool = False