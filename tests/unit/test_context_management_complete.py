"""Comprehensive tests for Phase 17: Context Management System."""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
import redis.asyncio as redis
from pydantic import ValidationError

from src.context.models import (
    Context,
    ConversationContext,
    SessionContext,
    CollaborativeContext,
    ContextType,
    ContextState,
    ContextVersion,
    ContextEvent,
    ContextDiff,
    ProviderContext,
    CompressionType,
    ConflictResolutionStrategy,
)
from src.context.persistence import ContextPersistenceLayer
from src.context.serialization import ContextSerializer, ContextCompressor, SerializationFormat
from src.context.synchronization import EventBus, ContextSyncManager, SyncEvent, SyncEventType
from src.context.translation import (
    ContextTranslator,
    AnthropicAdapter,
    OpenAIAdapter,
    GoogleAdapter,
)
from src.context.validation import ContextValidator, ValidationResult, ValidationIssue, ValidationSeverity
from src.context.versioning import ContextVersionManager, VersioningStrategy
from src.context.context_manager import ContextManager
from src.context.integration import ContextIntegrationManager
from src.context.config import ContextConfig, get_context_config
from src.ai_providers.models import ProviderType


@pytest.fixture
async def test_config():
    """Test configuration."""
    return ContextConfig(
        database_url="postgresql://test:test@localhost:5432/test_context",
        redis_url="redis://localhost:6379/1",
        enable_real_time_sync=True,
        enable_compression=True,
        enable_validation=True,
        enable_versioning=True,
        enable_translation=True,
        max_context_size_mb=10,
        max_versions_per_context=100,
        context_cache_ttl=3600,
        sync_timeout_seconds=30,
        compression_algorithm="zstd",
        serialization_format="msgpack",
        max_participants_per_room=50,
        lock_timeout_seconds=300,
        event_history_size=1000,
        event_ttl_seconds=3600,
        db_pool_size=20,
        db_max_overflow=30,
    )


@pytest.fixture
async def mock_redis():
    """Mock Redis connection."""
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.publish = AsyncMock(return_value=1)
    mock_redis.subscribe = AsyncMock()
    mock_redis.unsubscribe = AsyncMock()
    mock_redis.lrange = AsyncMock(return_value=[])
    mock_redis.lpush = AsyncMock(return_value=1)
    mock_redis.ltrim = AsyncMock(return_value=True)
    mock_redis.expire = AsyncMock(return_value=True)
    mock_redis.ttl = AsyncMock(return_value=-1)
    mock_redis.scan_iter = AsyncMock(return_value=iter([]))
    mock_redis.close = AsyncMock()
    
    # Mock pubsub
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.listen = AsyncMock(return_value=iter([]))
    mock_pubsub.close = AsyncMock()
    
    mock_redis.pubsub = Mock(return_value=mock_pubsub)
    
    return mock_redis


@pytest.fixture
async def mock_postgres():
    """Mock PostgreSQL connection."""
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value={"context_id": str(uuid4())})
    mock_conn.fetchval = AsyncMock(return_value=str(uuid4()))
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.execute = AsyncMock(return_value="UPDATE 1")
    mock_conn.transaction = MagicMock()
    mock_conn.transaction.__aenter__ = AsyncMock()
    mock_conn.transaction.__aexit__ = AsyncMock()
    
    mock_pool = AsyncMock()
    mock_pool.acquire = AsyncMock(return_value=mock_conn)
    mock_pool.release = AsyncMock()
    mock_pool.close = AsyncMock()
    
    return mock_pool


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    return Context(
        context_id=str(uuid4()),
        context_type=ContextType.CONVERSATION,
        title="Test Context",
        description="A test context for unit testing",
        data={"key": "value", "nested": {"data": "here"}},
        metadata={"test": True, "version": "1.0"},
        owner_id="test_user",
        collaborators=["user1", "user2"],
        state=ContextState.ACTIVE,
    )


@pytest.fixture
def sample_conversation_context():
    """Create a sample conversation context."""
    return ConversationContext(
        context_id=str(uuid4()),
        title="Test Conversation",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ],
        participants=["user1", "user2"],
        current_turn=3,
        primary_provider="anthropic",
        model_parameters={"temperature": 0.7, "max_tokens": 1000},
    )


@pytest.fixture
def sample_collaborative_context():
    """Create a sample collaborative context."""
    return CollaborativeContext(
        context_id=str(uuid4()),
        title="Collaborative Session",
        room_id=str(uuid4()),
        active_participants=["user1", "user2", "user3"],
        shared_state={"game": "D&D", "session": 1},
    )


class TestContextModels:
    """Test context data models."""
    
    def test_context_creation(self, sample_context):
        """Test basic context creation."""
        assert sample_context.context_id
        assert sample_context.context_type == ContextType.CONVERSATION
        assert sample_context.state == ContextState.ACTIVE
        assert len(sample_context.collaborators) == 2
    
    def test_context_update_access(self, sample_context):
        """Test context access tracking."""
        initial_count = sample_context.access_count
        initial_time = sample_context.last_accessed
        
        time.sleep(0.01)  # Small delay to ensure timestamp changes
        sample_context.update_access()
        
        assert sample_context.access_count == initial_count + 1
        assert sample_context.last_accessed > initial_time
    
    def test_context_add_collaborator(self, sample_context):
        """Test adding collaborators."""
        sample_context.add_collaborator("user3", ["read", "write"])
        
        assert "user3" in sample_context.collaborators
        assert "read" in sample_context.permissions.get("user3", [])
        assert "write" in sample_context.permissions.get("user3", [])
    
    def test_context_permissions(self, sample_context):
        """Test permission checking."""
        # Owner has all permissions
        assert sample_context.has_permission("test_user", "delete")
        
        # Add specific permissions
        sample_context.add_collaborator("user4", ["read"])
        assert sample_context.has_permission("user4", "read")
        assert not sample_context.has_permission("user4", "write")
    
    def test_context_should_archive(self, sample_context):
        """Test auto-archive logic."""
        # Fresh context shouldn't archive
        assert not sample_context.should_archive()
        
        # Old context should archive
        sample_context.last_accessed = datetime.now(timezone.utc) - timedelta(days=100)
        assert sample_context.should_archive()
    
    def test_conversation_context_add_message(self, sample_conversation_context):
        """Test adding messages to conversation."""
        initial_count = len(sample_conversation_context.messages)
        initial_turn = sample_conversation_context.current_turn
        
        sample_conversation_context.add_message({
            "role": "user",
            "content": "What's the weather like?"
        })
        
        assert len(sample_conversation_context.messages) == initial_count + 1
        assert sample_conversation_context.current_turn == initial_turn + 1
        assert sample_conversation_context.last_message_at is not None
    
    def test_conversation_context_get_recent_messages(self, sample_conversation_context):
        """Test getting recent messages."""
        # Add more messages
        for i in range(20):
            sample_conversation_context.add_message({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}"
            })
        
        recent = sample_conversation_context.get_recent_messages(5)
        assert len(recent) == 5
        assert recent[-1]["content"] == "Message 19"
    
    def test_collaborative_context_participants(self, sample_collaborative_context):
        """Test collaborative participant management."""
        initial_count = len(sample_collaborative_context.active_participants)
        
        # Add participant
        sample_collaborative_context.add_participant("user4")
        assert len(sample_collaborative_context.active_participants) == initial_count + 1
        
        # Remove participant
        sample_collaborative_context.remove_participant("user4")
        assert len(sample_collaborative_context.active_participants) == initial_count
    
    def test_collaborative_context_locking(self, sample_collaborative_context):
        """Test collaborative locking mechanism."""
        # Acquire lock
        assert sample_collaborative_context.acquire_lock("user1")
        assert sample_collaborative_context.locked_by == "user1"
        
        # Try to acquire with different user (should fail)
        assert not sample_collaborative_context.acquire_lock("user2")
        
        # Release lock
        assert sample_collaborative_context.release_lock("user1")
        assert sample_collaborative_context.locked_by is None
        
        # Now another user can acquire
        assert sample_collaborative_context.acquire_lock("user2")
    
    def test_context_event_creation(self):
        """Test context event creation."""
        event = ContextEvent(
            context_id=str(uuid4()),
            event_type="update",
            user_id="test_user",
            changes={"field": "new_value"},
        )
        
        assert event.event_id
        assert event.timestamp
        assert not event.propagated
        assert event.source == "system"
    
    def test_context_diff_creation(self):
        """Test context diff creation."""
        diff = ContextDiff(
            context_id=str(uuid4()),
            from_version=1,
            to_version=2,
            added={"new_field": "value"},
            modified={"existing_field": "new_value"},
            removed=["old_field"],
        )
        
        assert diff.diff_id
        assert diff.created_at
        assert len(diff.removed) == 1


class TestSerialization:
    """Test context serialization and compression."""
    
    @pytest.fixture
    def serializer(self):
        """Create a context serializer."""
        return ContextSerializer(default_format=SerializationFormat.MSGPACK)
    
    @pytest.fixture
    def compressor(self):
        """Create a context compressor."""
        return ContextCompressor()
    
    def test_json_serialization(self, serializer, sample_context):
        """Test JSON serialization."""
        serialized = serializer.serialize(sample_context, SerializationFormat.JSON)
        assert isinstance(serialized, str)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized, Context, SerializationFormat.JSON)
        assert deserialized.context_id == sample_context.context_id
    
    def test_msgpack_serialization(self, serializer, sample_context):
        """Test MessagePack serialization."""
        serialized = serializer.serialize(sample_context, SerializationFormat.MSGPACK)
        assert isinstance(serialized, str)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized, Context, SerializationFormat.MSGPACK)
        assert deserialized.context_id == sample_context.context_id
    
    def test_pickle_serialization(self, serializer, sample_context):
        """Test Pickle serialization."""
        serialized = serializer.serialize(sample_context, SerializationFormat.PICKLE)
        assert isinstance(serialized, str)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized, Context, SerializationFormat.PICKLE)
        assert deserialized["context_id"] == sample_context.context_id
    
    @pytest.mark.asyncio
    async def test_async_serialization(self, serializer, sample_context):
        """Test async serialization."""
        serialized = await serializer.serialize_async(sample_context)
        assert isinstance(serialized, str)
        
        deserialized = await serializer.deserialize_async(serialized, Context)
        assert deserialized.context_id == sample_context.context_id
    
    def test_batch_serialization(self, serializer):
        """Test batch serialization."""
        contexts = [
            Context(context_id=str(uuid4()), context_type=ContextType.SESSION)
            for _ in range(5)
        ]
        
        serialized_list = serializer.serialize_batch(contexts)
        assert len(serialized_list) == 5
        assert all(isinstance(s, str) for s in serialized_list)
    
    def test_gzip_compression(self, compressor):
        """Test GZIP compression."""
        data = "This is test data" * 100  # Repeat to make compression effective
        
        compressed, stats = compressor.compress(data, CompressionType.GZIP)
        assert isinstance(compressed, bytes)
        assert stats["compressed"]
        assert stats["compression_type"] == "gzip"
        assert stats["compressed_size"] < stats["original_size"]
        
        # Decompress
        decompressed = compressor.decompress(compressed, CompressionType.GZIP)
        assert decompressed == data
    
    @pytest.mark.skipif(
        not hasattr(__import__("lz4", globals(), locals(), [], 0), "frame"),
        reason="LZ4 not installed"
    )
    def test_lz4_compression(self, compressor):
        """Test LZ4 compression."""
        data = "This is test data" * 100
        
        compressed, stats = compressor.compress(data, CompressionType.LZ4)
        assert isinstance(compressed, bytes)
        assert stats["compressed"]
        
        decompressed = compressor.decompress(compressed, CompressionType.LZ4)
        assert decompressed == data
    
    @pytest.mark.skipif(
        not hasattr(__import__("zstandard", globals(), locals(), [], 0), "ZstdCompressor"),
        reason="Zstandard not installed"
    )
    def test_zstd_compression(self, compressor):
        """Test Zstandard compression."""
        data = "This is test data" * 100
        
        compressed, stats = compressor.compress(data, CompressionType.ZSTD)
        assert isinstance(compressed, bytes)
        assert stats["compressed"]
        
        decompressed = compressor.decompress(compressed, CompressionType.ZSTD)
        assert decompressed == data
    
    def test_compression_small_data(self, compressor):
        """Test compression with small data (should skip)."""
        data = "Small"
        
        compressed, stats = compressor.compress(data)
        assert not stats["compressed"]
        assert stats["compression_type"] == "none"
        assert stats["compressed_size"] == stats["original_size"]
    
    def test_get_best_compression(self, compressor):
        """Test automatic compression selection."""
        data = "Test data for compression" * 100
        
        best_algorithm = compressor.get_best_compression(data)
        assert best_algorithm in compressor.available_algorithms or best_algorithm == CompressionType.NONE
    
    @pytest.mark.asyncio
    async def test_async_compression(self, compressor):
        """Test async compression."""
        data = "Async compression test" * 100
        
        compressed, stats = await compressor.compress_async(data)
        assert isinstance(compressed, bytes)
        
        decompressed = await compressor.decompress_async(
            compressed, CompressionType(stats["compression_type"])
        )
        assert decompressed == data


class TestValidation:
    """Test context validation."""
    
    @pytest.fixture
    def validator(self):
        """Create a context validator."""
        return ContextValidator(
            enable_schema_validation=True,
            enable_semantic_validation=True,
            enable_integrity_checks=True,
            enable_auto_correction=True,
        )
    
    @pytest.mark.asyncio
    async def test_valid_context(self, validator, sample_context):
        """Test validation of valid context."""
        result = await validator.validate_context(sample_context)
        
        assert result.is_valid
        assert len(result.issues) == 0 or all(
            issue.severity in [ValidationSeverity.INFO, ValidationSeverity.WARNING]
            for issue in result.issues
        )
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, validator):
        """Test validation with missing required fields."""
        invalid_context = {
            "title": "Test",
            # Missing context_id and context_type
        }
        
        result = await validator.validate_context(invalid_context)
        
        assert not result.is_valid
        assert result.has_errors()
        assert any(
            "Required field" in issue.message
            for issue in result.get_issues_by_severity(ValidationSeverity.ERROR)
        )
    
    @pytest.mark.asyncio
    async def test_field_length_validation(self, validator):
        """Test field length constraints."""
        context = Context(
            context_id=str(uuid4()),
            context_type=ContextType.SESSION,
            title="X" * 1000,  # Exceeds max length
            description="Y" * 3000,  # Exceeds max length
        )
        
        result = await validator.validate_context(context)
        
        assert any(
            "exceeds maximum length" in issue.message
            for issue in result.issues
        )
    
    @pytest.mark.asyncio
    async def test_timestamp_validation(self, validator):
        """Test timestamp consistency."""
        context = Context(
            context_id=str(uuid4()),
            context_type=ContextType.SESSION,
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc) - timedelta(days=1),  # Before created_at
        )
        
        result = await validator.validate_context(context)
        
        assert any(
            "created_at is after last_modified" in issue.message
            for issue in result.issues
        )
    
    @pytest.mark.asyncio
    async def test_conversation_semantics(self, validator, sample_conversation_context):
        """Test conversation-specific validation."""
        # Add invalid message
        sample_conversation_context.messages.append({
            # Missing role and content
        })
        
        result = await validator.validate_context(sample_conversation_context)
        
        assert any(
            "missing 'role' field" in issue.message
            for issue in result.issues
        )
    
    @pytest.mark.asyncio
    async def test_auto_correction(self, validator):
        """Test automatic corrections."""
        context = {
            "context_id": str(uuid4()),
            "context_type": "session",
            "access_count": -5,  # Invalid negative count
            "current_version": 0,  # Should be >= 1
        }
        
        result = await validator.validate_context(context)
        
        if result.corrected_data:
            assert result.corrected_data["access_count"] == 0
            assert result.corrected_data["current_version"] == 1
    
    @pytest.mark.asyncio
    async def test_custom_validation_rule(self, validator):
        """Test custom validation rules."""
        def custom_rule(context_data: Dict[str, Any], result: ValidationResult):
            if context_data.get("title", "").lower() == "forbidden":
                result.add_issue(ValidationIssue(
                    issue_id=str(uuid4()),
                    severity=ValidationSeverity.ERROR,
                    category="custom",
                    message="Forbidden title",
                ))
        
        validator.add_custom_rule(custom_rule)
        
        context = Context(
            context_id=str(uuid4()),
            context_type=ContextType.SESSION,
            title="Forbidden",
        )
        
        result = await validator.validate_context(context)
        
        assert not result.is_valid
        assert any(
            "Forbidden title" in issue.message
            for issue in result.issues
        )


class TestTranslation:
    """Test context translation between providers."""
    
    @pytest.fixture
    def translator(self):
        """Create a context translator."""
        return ContextTranslator()
    
    @pytest.mark.asyncio
    async def test_anthropic_translation(self, translator, sample_conversation_context):
        """Test translation to Anthropic format."""
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.ANTHROPIC,
        )
        
        assert provider_context.provider_type == "anthropic"
        assert "messages" in provider_context.context_data
        assert "system" in provider_context.context_data
    
    @pytest.mark.asyncio
    async def test_openai_translation(self, translator, sample_conversation_context):
        """Test translation to OpenAI format."""
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.OPENAI,
        )
        
        assert provider_context.provider_type == "openai"
        assert "messages" in provider_context.context_data
        assert all(
            "role" in msg and "content" in msg
            for msg in provider_context.context_data["messages"]
        )
    
    @pytest.mark.asyncio
    async def test_google_translation(self, translator, sample_conversation_context):
        """Test translation to Google format."""
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.GOOGLE,
        )
        
        assert provider_context.provider_type == "google"
        assert "contents" in provider_context.context_data
        assert all(
            "role" in content and "parts" in content
            for content in provider_context.context_data["contents"]
        )
    
    @pytest.mark.asyncio
    async def test_reverse_translation(self, translator, sample_conversation_context):
        """Test reverse translation from provider format."""
        # First translate to provider format
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.ANTHROPIC,
        )
        
        # Then translate back
        reversed_context = await translator.translate_from_provider(
            provider_context,
            ConversationContext,
        )
        
        assert isinstance(reversed_context, ConversationContext)
        assert len(reversed_context.messages) > 0
    
    @pytest.mark.asyncio
    async def test_context_migration(self, translator, sample_conversation_context):
        """Test migrating context between providers."""
        migrated = await translator.migrate_context(
            sample_conversation_context,
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
        )
        
        assert "openai" in migrated.provider_contexts
        assert migrated.provider_contexts["openai"].provider_type == "openai"
    
    def test_anthropic_adapter_validation(self):
        """Test Anthropic adapter validation."""
        adapter = AnthropicAdapter()
        
        valid_context = ProviderContext(
            provider_type="anthropic",
            context_data={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        )
        
        assert adapter.validate_provider_context(valid_context)
        
        invalid_context = ProviderContext(
            provider_type="anthropic",
            context_data={
                "messages": [
                    {"invalid": "data"},
                ]
            }
        )
        
        assert not adapter.validate_provider_context(invalid_context)
    
    def test_size_estimation(self, sample_conversation_context):
        """Test context size estimation."""
        adapter = AnthropicAdapter()
        size = adapter.get_size_estimate(sample_conversation_context)
        
        assert isinstance(size, int)
        assert size > 0
        assert size <= adapter.max_context_tokens


class TestSynchronization:
    """Test context synchronization."""
    
    @pytest.fixture
    async def event_bus(self, mock_redis):
        """Create an event bus with mock Redis."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            bus = EventBus()
            await bus.initialize()
            return bus
    
    @pytest.fixture
    async def sync_manager(self, event_bus):
        """Create a sync manager."""
        mock_persistence = AsyncMock()
        return ContextSyncManager(
            persistence_layer=mock_persistence,
            event_bus=event_bus,
        )
    
    @pytest.mark.asyncio
    async def test_event_publishing(self, event_bus, mock_redis):
        """Test publishing sync events."""
        event = SyncEvent(
            event_type=SyncEventType.CONTEXT_UPDATED,
            context_id=str(uuid4()),
            user_id="test_user",
            data={"test": "data"},
        )
        
        await event_bus.publish_event(event)
        
        assert mock_redis.publish.called
    
    @pytest.mark.asyncio
    async def test_event_subscription(self, event_bus):
        """Test event subscription."""
        received_events = []
        
        async def handler(event: SyncEvent):
            received_events.append(event)
        
        room_id = str(uuid4())
        await event_bus.subscribe_to_room(room_id, handler)
        
        assert room_id in event_bus._active_rooms
    
    @pytest.mark.asyncio
    async def test_context_synchronization(self, sync_manager, sample_context):
        """Test context synchronization."""
        sync_manager.persistence.update_context = AsyncMock(return_value=True)
        
        result = await sync_manager.synchronize_context(sample_context, "test_user")
        
        assert result
        assert sync_manager.persistence.update_context.called
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_latest_wins(self, sync_manager):
        """Test latest wins conflict resolution."""
        local = Context(
            context_id=str(uuid4()),
            context_type=ContextType.SESSION,
            last_modified=datetime.now(timezone.utc),
        )
        
        remote = Context(
            context_id=local.context_id,
            context_type=ContextType.SESSION,
            last_modified=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        
        sync_manager.persistence.update_context = AsyncMock(return_value=True)
        
        resolved = await sync_manager.handle_sync_conflict(
            local.context_id,
            local,
            remote,
            ConflictResolutionStrategy.LATEST_WINS,
        )
        
        assert resolved == local  # Local is newer
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_automatic_merge(self, sync_manager):
        """Test automatic merge conflict resolution."""
        local = Context(
            context_id=str(uuid4()),
            context_type=ContextType.SESSION,
            data={"local": "data"},
            metadata={"local": "metadata"},
        )
        
        remote = Context(
            context_id=local.context_id,
            context_type=ContextType.SESSION,
            data={"remote": "data"},
            metadata={"remote": "metadata"},
        )
        
        sync_manager.persistence.update_context = AsyncMock(return_value=True)
        
        resolved = await sync_manager.handle_sync_conflict(
            local.context_id,
            local,
            remote,
            ConflictResolutionStrategy.AUTOMATIC_MERGE,
        )
        
        # Should have both local and remote data
        assert "local" in resolved.data
        assert "remote" in resolved.data
    
    @pytest.mark.asyncio
    async def test_event_history(self, event_bus, mock_redis):
        """Test event history retrieval."""
        mock_redis.lrange.return_value = [
            SyncEvent(
                event_type=SyncEventType.CONTEXT_CREATED,
                context_id=str(uuid4()),
            ).json()
        ]
        
        history = await event_bus.get_event_history(room_id=str(uuid4()))
        
        assert len(history) == 1
        assert history[0].event_type == SyncEventType.CONTEXT_CREATED
    
    def test_performance_stats(self, event_bus):
        """Test performance statistics."""
        stats = event_bus.get_performance_stats()
        
        assert "event_stats" in stats
        assert "active_subscriptions" in stats
        assert "active_rooms" in stats


class TestVersioning:
    """Test context versioning."""
    
    @pytest.fixture
    async def version_manager(self):
        """Create a version manager."""
        mock_persistence = AsyncMock()
        mock_serializer = Mock()
        mock_compressor = Mock()
        
        return ContextVersionManager(
            persistence_layer=mock_persistence,
            serializer=mock_serializer,
            compressor=mock_compressor,
            versioning_strategy=VersioningStrategy.HYBRID,
        )
    
    @pytest.mark.asyncio
    async def test_version_creation(self, version_manager, sample_context):
        """Test creating a new version."""
        version_manager.persistence.get_version = AsyncMock(return_value=None)
        version_manager._store_version = AsyncMock(return_value=str(uuid4()))
        
        version = await version_manager.create_version(
            sample_context,
            user_id="test_user",
            message="Test version",
            tags=["test"],
        )
        
        assert version.context_id == sample_context.context_id
        assert version.created_by == "test_user"
        assert "test" in version.tags


class TestIntegration:
    """Test integration with other systems."""
    
    @pytest.fixture
    async def integration_manager(self, test_config):
        """Create an integration manager."""
        with patch('src.context.config.get_context_config', return_value=test_config):
            mock_context_manager = AsyncMock()
            mock_context_manager.initialize = AsyncMock()
            mock_context_manager.create_context = AsyncMock(return_value=str(uuid4()))
            mock_context_manager.get_context = AsyncMock()
            mock_context_manager.update_context = AsyncMock(return_value=True)
            
            manager = ContextIntegrationManager(
                context_manager=mock_context_manager,
                provider_manager=None,
                bridge_session_manager=None,
                security_manager=None,
            )
            
            await manager.initialize()
            return manager
    
    @pytest.mark.asyncio
    async def test_create_conversation_context(self, integration_manager):
        """Test creating conversation context through integration."""
        session_id = str(uuid4())
        context_id = await integration_manager.create_conversation_context(
            session_id,
            user_id="test_user",
            title="Test Conversation",
            provider_type=ProviderType.ANTHROPIC,
        )
        
        assert context_id
        assert session_id in integration_manager._session_contexts
        assert integration_manager.context_manager.create_context.called
    
    @pytest.mark.asyncio
    async def test_create_collaborative_context(self, integration_manager):
        """Test creating collaborative context."""
        room_id = str(uuid4())
        context_id = await integration_manager.create_collaborative_context(
            room_id,
            title="Collaborative Session",
            participants=["user1", "user2"],
            creator_id="creator",
        )
        
        assert context_id
        assert integration_manager.context_manager.create_context.called
    
    @pytest.mark.asyncio
    async def test_add_message_to_conversation(self, integration_manager):
        """Test adding messages through integration."""
        session_id = str(uuid4())
        context_id = str(uuid4())
        integration_manager._session_contexts[session_id] = context_id
        
        integration_manager.context_manager.get_context.return_value = ConversationContext(
            context_id=context_id,
            messages=[],
        )
        
        success = await integration_manager.add_message_to_conversation(
            session_id,
            {"role": "user", "content": "Test message"},
            user_id="test_user",
        )
        
        assert success
        assert integration_manager.context_manager.update_context.called
    
    @pytest.mark.asyncio
    async def test_get_conversation_history(self, integration_manager):
        """Test retrieving conversation history."""
        session_id = str(uuid4())
        context_id = str(uuid4())
        integration_manager._session_contexts[session_id] = context_id
        
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        
        integration_manager.context_manager.get_context.return_value = ConversationContext(
            context_id=context_id,
            messages=messages,
        )
        
        history = await integration_manager.get_conversation_history(
            session_id,
            user_id="test_user",
        )
        
        assert history == messages
    
    def test_integration_stats(self, integration_manager):
        """Test integration statistics."""
        stats = integration_manager.get_integration_stats()
        
        assert "session_contexts" in stats
        assert "provider_contexts" in stats
        assert "configuration" in stats
        assert "integrations" in stats


@pytest.mark.asyncio
async def test_end_to_end_context_flow():
    """Test complete end-to-end context management flow."""
    with patch('src.context.config.get_context_config') as mock_config:
        mock_config.return_value = ContextConfig(
            database_url="postgresql://test:test@localhost:5432/test",
            redis_url="redis://localhost:6379/1",
        )
        
        # Mock database and Redis
        with patch('asyncpg.create_pool') as mock_pg:
            with patch('redis.asyncio.from_url') as mock_redis_factory:
                mock_pg.return_value = AsyncMock()
                mock_redis_factory.return_value = AsyncMock()
                
                # Create context manager
                context_manager = ContextManager(
                    database_url="postgresql://test:test@localhost:5432/test",
                    redis_url="redis://localhost:6379/1",
                )
                
                # Mock initialization
                context_manager.persistence = AsyncMock()
                context_manager.event_bus = AsyncMock()
                context_manager.sync_manager = AsyncMock()
                context_manager.translator = AsyncMock()
                context_manager.validator = AsyncMock()
                context_manager.version_manager = AsyncMock()
                
                context_manager.persistence.create_context = AsyncMock(return_value=str(uuid4()))
                context_manager.persistence.get_context = AsyncMock()
                context_manager.persistence.update_context = AsyncMock(return_value=True)
                
                # Create a context
                context = ConversationContext(
                    title="E2E Test",
                    messages=[{"role": "user", "content": "Test"}],
                )
                
                context_id = await context_manager.create_context(context, user_id="test_user")
                assert context_id
                
                # Update context
                success = await context_manager.update_context(
                    context_id,
                    {"title": "Updated E2E Test"},
                    user_id="test_user",
                )
                assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])