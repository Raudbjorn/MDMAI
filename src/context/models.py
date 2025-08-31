"""Data models for the Context Management System."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum, auto
from functools import cached_property
from typing import Annotated, Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self


class ContextType(StrEnum):
    """Types of contexts that can be managed."""
    
    CONVERSATION = auto()
    SESSION = auto()
    CAMPAIGN = auto()
    CHARACTER = auto()
    COLLABORATIVE = auto()
    PROVIDER_SPECIFIC = auto()


class ContextState(StrEnum):
    """Lifecycle states for contexts."""
    
    ACTIVE = auto()
    INACTIVE = auto()
    ARCHIVED = auto()
    DELETED = auto()
    SYNCING = auto()
    CONFLICT = auto()


class CompressionType(StrEnum):
    """Supported compression algorithms."""
    
    NONE = auto()
    GZIP = auto()
    LZ4 = auto()
    ZSTD = auto()
    BROTLI = auto()


class ConflictResolutionStrategy(StrEnum):
    """Strategies for resolving context conflicts."""
    
    LATEST_WINS = auto()
    MANUAL_MERGE = auto()
    PROVIDER_PRIORITY = auto()
    USER_CHOICE = auto()
    AUTOMATIC_MERGE = auto()


@dataclass(frozen=True)
class ContextVersion:
    """Immutable version of a context with metadata."""
    
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
    
    @cached_property
    def is_main_branch(self) -> bool:
        """Check if this version is on the main branch."""
        return self.branch == "main"
    
    @cached_property
    def has_parent(self) -> bool:
        """Check if this version has a parent."""
        return self.parent_version is not None


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
    
    def update_access(self) -> Self:
        """Update access tracking with fluent interface."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        return self
    
    def add_collaborator(self, user_id: str, permissions: Optional[List[str]] = None) -> Self:
        """Add a collaborator with specified permissions using fluent interface."""
        if user_id not in self.collaborators:
            self.collaborators.append(user_id)
        
        if permissions:
            self.permissions[user_id] = permissions
        return self
    
    def has_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has specific permission."""
        return self.owner_id == user_id or permission in self.permissions.get(user_id, [])
    
    @cached_property
    def days_since_last_access(self) -> int:
        """Calculate days since last access."""
        return (datetime.now(timezone.utc) - self.last_accessed).days
    
    def should_archive(self) -> bool:
        """Check if context should be automatically archived."""
        return bool(
            self.auto_archive_days 
            and self.days_since_last_access >= self.auto_archive_days
        )


class ConversationContext(Context):
    """Context for conversation threads."""
    
    context_type: ContextType = Field(default=ContextType.CONVERSATION)
    
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
    
    def add_message(self, message: Dict[str, Any]) -> Self:
        """Add a message to the conversation with fluent interface."""
        now = datetime.now(timezone.utc)
        message.update({
            "timestamp": now.isoformat(),
            "turn": self.current_turn
        })
        self.messages.append(message)
        self.current_turn += 1
        self.last_message_at = now
        self.last_modified = now
        return self
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent messages."""
        return self.messages[-count:] if len(self.messages) > count else self.messages
    
    @cached_property
    def is_at_turn_limit(self) -> bool:
        """Check if conversation has reached turn limit."""
        return self.max_turns is not None and self.current_turn >= self.max_turns


class SessionContext(Context):
    """Context for user sessions."""
    
    context_type: ContextType = Field(default=ContextType.SESSION)
    
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
    
    def add_conversation(self, conversation_id: str) -> Self:
        """Add a conversation to this session with fluent interface."""
        if conversation_id not in self.active_conversations:
            self.active_conversations.append(conversation_id)
        return self
    
    def record_interaction(self) -> Self:
        """Record user interaction with fluent interface."""
        now = datetime.now(timezone.utc)
        self.total_interactions += 1
        self.last_interaction = now
        self.last_modified = now
        return self
    
    @cached_property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        if not self.last_interaction:
            return False
        elapsed = (datetime.now(timezone.utc) - self.last_interaction).total_seconds()
        return elapsed < self.session_timeout if hasattr(self, 'session_timeout') else True


class CollaborativeContext(Context):
    """Context for collaborative sessions."""
    
    context_type: ContextType = Field(default=ContextType.COLLABORATIVE)
    
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
    
    def add_participant(self, user_id: str) -> Self:
        """Add a participant to the collaborative session with fluent interface."""
        if user_id not in self.active_participants:
            self.active_participants.append(user_id)
        return self
    
    def remove_participant(self, user_id: str) -> Self:
        """Remove a participant from the collaborative session with fluent interface."""
        if user_id in self.active_participants:
            self.active_participants.remove(user_id)
        
        # Release lock if held by this user
        if self.locked_by == user_id:
            self.release_lock()
        return self
    
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
    
    @cached_property
    def is_locked(self) -> bool:
        """Check if context is currently locked."""
        return self.locked_by is not None and not self._is_lock_expired()
    
    def _is_lock_expired(self) -> bool:
        """Check if current lock has expired."""
        if not self.locked_at:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.locked_at).total_seconds()
        return elapsed > self.lock_timeout_seconds


class ProviderContext(BaseModel):
    """Provider-specific context data with enhanced validation."""
    
    provider_type: Annotated[str, Field(min_length=1, max_length=50)]
    provider_context_id: str = Field(default_factory=lambda: str(uuid4()))
    
    # Provider-specific data
    context_data: Dict[str, Any] = Field(default_factory=dict)
    format_version: Annotated[str, Field(pattern=r"^\d+\.\d+$")] = "1.0"
    
    # Synchronization tracking
    last_sync: Optional[datetime] = None
    sync_status: str = Field(default="synced", pattern="^(synced|pending|failed|syncing)$")
    
    # Translation metadata
    translation_metadata: Dict[str, Any] = Field(default_factory=dict)
    requires_translation: bool = False
    
    # Performance tracking
    size_bytes: Annotated[int, Field(ge=0)] = 0
    compression_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @cached_property
    def is_synced(self) -> bool:
        """Check if provider context is synced."""
        return self.sync_status == "synced"
    
    @cached_property
    def compression_percentage(self) -> float:
        """Calculate compression percentage."""
        return (1 - self.compression_ratio) * 100


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
    """Query model for context retrieval with enhanced validation."""
    
    # Filters
    context_ids: Optional[List[str]] = None
    context_types: Optional[List[ContextType]] = None
    owner_id: Optional[str] = None
    collaborators: Optional[List[str]] = None
    states: Optional[List[ContextState]] = None
    
    # Search
    search_text: Optional[Annotated[str, Field(min_length=1, max_length=500)]] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Time range
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    modified_after: Optional[datetime] = None
    modified_before: Optional[datetime] = None
    accessed_after: Optional[datetime] = None
    
    # Pagination
    limit: Annotated[int, Field(ge=1, le=1000)] = 50
    offset: Annotated[int, Field(ge=0)] = 0
    
    # Sorting
    sort_by: str = Field(
        default="last_modified",
        pattern="^(last_modified|created_at|last_accessed|title|access_count|size_bytes)$"
    )
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    
    # Options
    include_archived: bool = False
    include_deleted: bool = False
    
    @field_validator("created_before", "modified_before")
    @classmethod
    def validate_before_dates(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure 'before' dates are not in the future."""
        if v and v > datetime.now(timezone.utc):
            raise ValueError("'before' dates cannot be in the future")
        return v
    
    @cached_property
    def has_time_filters(self) -> bool:
        """Check if query has time-based filters."""
        return any([
            self.created_after, self.created_before,
            self.modified_after, self.modified_before,
            self.accessed_after
        ])
    
    @cached_property
    def is_paginated(self) -> bool:
        """Check if query uses pagination."""
        return self.limit < 1000 or self.offset > 0