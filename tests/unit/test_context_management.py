"""Comprehensive test suite for the Context Management System."""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.context import (
    ContextManager,
    Context,
    ConversationContext,
    SessionContext,
    CollaborativeContext,
    ContextType,
    ContextState,
    ContextQuery,
    ContextPersistenceLayer,
    ContextValidator,
    ContextTranslator,
    ContextVersionManager,
    EventBus,
    ContextSyncManager,
)
from src.context.models import CompressionType
from src.ai_providers.models import ProviderType


@pytest.fixture
async def mock_database_url():
    """Mock database URL for testing."""
    return "postgresql://test:test@localhost:5432/test_db"


@pytest.fixture
async def mock_redis_url():
    """Mock Redis URL for testing."""
    return "redis://localhost:6379/1"


@pytest.fixture
async def context_manager(mock_database_url, mock_redis_url):
    """Create a test context manager instance."""
    with patch('src.context.persistence.ContextPersistenceLayer'):
        with patch('src.context.synchronization.EventBus'):
            manager = ContextManager(
                database_url=mock_database_url,
                redis_url=mock_redis_url,
                enable_real_time_sync=True,
                enable_compression=True,
                enable_validation=True,
                enable_versioning=True,
            )
            await manager.initialize()
            yield manager
            await manager.cleanup()


@pytest.fixture
def sample_conversation_context():
    """Create a sample conversation context."""
    return ConversationContext(
        title="Test Conversation",
        description="A test conversation for unit testing",
        messages=[
            {
                "role": "user",
                "content": "Hello, I need help with my D&D character.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "role": "assistant", 
                "content": "I'd be happy to help! What kind of character are you creating?",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        participants=["user_123", "assistant"],
        primary_provider="anthropic",
        model_parameters={"temperature": 0.7, "max_tokens": 1000},
    )


@pytest.fixture
def sample_collaborative_context():
    """Create a sample collaborative context."""
    return CollaborativeContext(
        title="Campaign Planning Session",
        description="Collaborative session for planning our D&D campaign",
        room_id="room_456",
        active_participants=["dm_user", "player_1", "player_2"],
        shared_state={"current_scene": "tavern", "active_npcs": ["barkeeper", "mysterious_stranger"]},
        lock_timeout_seconds=300,
    )


class TestContextManager:
    """Test cases for the main ContextManager class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_database_url, mock_redis_url):
        """Test context manager initialization."""
        with patch('src.context.persistence.ContextPersistenceLayer') as mock_persistence:
            with patch('src.context.synchronization.EventBus') as mock_event_bus:
                manager = ContextManager(
                    database_url=mock_database_url,
                    redis_url=mock_redis_url,
                )
                
                await manager.initialize()
                
                assert manager._initialized is True
                assert manager.persistence is not None
                assert manager.event_bus is not None
                assert manager.translator is not None
                assert manager.validator is not None
                
                await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_create_context(self, context_manager, sample_conversation_context):
        """Test context creation."""
        # Mock the persistence layer
        context_manager.persistence.create_context = AsyncMock(return_value="context_123")
        
        context_id = await context_manager.create_context(
            sample_conversation_context,
            user_id="user_123",
        )
        
        assert context_id == "context_123"
        assert context_manager._performance_stats["contexts_created"] == 1
        context_manager.persistence.create_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_context(self, context_manager, sample_conversation_context):
        """Test context retrieval."""
        # Mock the persistence layer
        context_manager.persistence.get_context = AsyncMock(return_value=sample_conversation_context)
        
        retrieved_context = await context_manager.get_context(
            "context_123",
            user_id="user_123",
        )
        
        assert retrieved_context is not None
        assert retrieved_context.title == sample_conversation_context.title
        assert context_manager._performance_stats["contexts_retrieved"] == 1
    
    @pytest.mark.asyncio
    async def test_update_context(self, context_manager, sample_conversation_context):
        """Test context updates."""
        # Mock the persistence layer
        context_manager.persistence.get_context = AsyncMock(return_value=sample_conversation_context)
        context_manager.persistence.update_context = AsyncMock(return_value=True)
        
        success = await context_manager.update_context(
            "context_123",
            {"title": "Updated Title"},
            user_id="user_123",
        )
        
        assert success is True
        assert context_manager._performance_stats["contexts_updated"] == 1
    
    @pytest.mark.asyncio
    async def test_delete_context(self, context_manager, sample_conversation_context):
        """Test context deletion."""
        # Mock the persistence layer
        context_manager.persistence.get_context = AsyncMock(return_value=sample_conversation_context)
        context_manager.persistence.delete_context = AsyncMock(return_value=True)
        
        success = await context_manager.delete_context(
            "context_123",
            user_id="user_123",
            hard_delete=False,
        )
        
        assert success is True
        assert context_manager._performance_stats["contexts_deleted"] == 1
    
    @pytest.mark.asyncio
    async def test_query_contexts(self, context_manager, sample_conversation_context):
        """Test context querying."""
        # Mock the persistence layer
        context_manager.persistence.query_contexts = AsyncMock(return_value=[sample_conversation_context])
        
        query = ContextQuery(
            context_types=[ContextType.CONVERSATION],
            limit=10,
        )
        
        contexts = await context_manager.query_contexts(query, user_id="user_123")
        
        assert len(contexts) == 1
        assert contexts[0].title == sample_conversation_context.title
    
    @pytest.mark.asyncio
    async def test_translate_context(self, context_manager, sample_conversation_context):
        """Test context translation."""
        # Mock the persistence layer and translator
        context_manager.persistence.get_context = AsyncMock(return_value=sample_conversation_context)
        context_manager.translator.translate_to_provider = AsyncMock(return_value={"translated": "data"})
        
        result = await context_manager.translate_context(
            "context_123",
            ProviderType.ANTHROPIC,
            user_id="user_123",
        )
        
        assert result == {"translated": "data"}
        assert context_manager._performance_stats["translation_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_collaborative_session(self, context_manager, sample_collaborative_context):
        """Test collaborative session management."""
        # Mock the persistence layer
        context_manager.persistence.get_context = AsyncMock(return_value=sample_collaborative_context)
        context_manager.persistence.update_context = AsyncMock(return_value=True)
        
        # Test joining session
        success = await context_manager.join_collaborative_session(
            "context_123",
            "new_user",
        )
        
        assert success is True
        
        # Test leaving session
        success = await context_manager.leave_collaborative_session(
            "context_123", 
            "new_user",
        )
        
        assert success is True


class TestContextPersistence:
    """Test cases for context persistence layer."""
    
    @pytest.mark.asyncio
    async def test_context_serialization(self, sample_conversation_context):
        """Test context serialization and deserialization."""
        from src.context.serialization import ContextSerializer, ContextCompressor
        
        serializer = ContextSerializer()
        compressor = ContextCompressor()
        
        # Test serialization
        serialized = serializer.serialize(sample_conversation_context)
        assert isinstance(serialized, str)
        assert len(serialized) > 0
        
        # Test compression
        compressed_data, stats = compressor.compress(serialized)
        assert isinstance(compressed_data, bytes)
        assert stats["compressed"] is True
        assert stats["compression_ratio"] > 1.0
        
        # Test decompression
        decompressed = compressor.decompress(compressed_data, CompressionType(stats["compression_type"]))
        assert decompressed == serialized
        
        # Test deserialization
        deserialized = serializer.deserialize(decompressed, ConversationContext)
        assert deserialized.title == sample_conversation_context.title
        assert len(deserialized.messages) == len(sample_conversation_context.messages)
    
    @pytest.mark.asyncio
    async def test_database_schema_creation(self, mock_database_url):
        """Test database schema creation."""
        with patch('psycopg2.pool.ThreadedConnectionPool') as mock_pool:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_pool.return_value.getconn.return_value = mock_conn
            
            persistence = ContextPersistenceLayer(mock_database_url)
            
            # Verify schema creation was attempted
            mock_cursor.execute.assert_called()
            mock_conn.commit.assert_called()


class TestContextValidation:
    """Test cases for context validation."""
    
    @pytest.mark.asyncio
    async def test_valid_context_validation(self, sample_conversation_context):
        """Test validation of a valid context."""
        validator = ContextValidator()
        
        result = await validator.validate_context(sample_conversation_context)
        
        assert result.is_valid is True
        assert len(result.issues) == 0
    
    @pytest.mark.asyncio
    async def test_invalid_context_validation(self):
        """Test validation of an invalid context."""
        validator = ContextValidator()
        
        # Create invalid context data
        invalid_context_data = {
            "context_type": "invalid_type",  # Invalid enum value
            "messages": [
                {"role": "invalid_role", "content": "test"}  # Invalid role
            ],
            "access_count": -1,  # Negative value
        }
        
        result = await validator.validate_context(invalid_context_data)
        
        assert result.is_valid is False
        assert len(result.issues) > 0
        
        # Check specific issue types
        issue_categories = [issue.category for issue in result.issues]
        assert "schema_validation" in issue_categories or "data_consistency" in issue_categories
    
    @pytest.mark.asyncio
    async def test_auto_correction(self):
        """Test automatic correction of fixable issues."""
        validator = ContextValidator(enable_auto_correction=True)
        
        # Create context with correctable issues
        context_data = {
            "context_id": "test_123",
            "context_type": "conversation",
            "access_count": -5,  # Will be corrected to 0
            "current_version": 0,  # Will be corrected to 1
        }
        
        result = await validator.validate_context(context_data)
        
        assert result.corrected_data is not None
        assert result.corrected_data["access_count"] == 0
        assert result.corrected_data["current_version"] == 1


class TestContextTranslation:
    """Test cases for context translation between providers."""
    
    @pytest.mark.asyncio
    async def test_anthropic_translation(self, sample_conversation_context):
        """Test translation to Anthropic format."""
        translator = ContextTranslator()
        
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.ANTHROPIC,
        )
        
        assert provider_context.provider_type == ProviderType.ANTHROPIC.value
        assert "messages" in provider_context.context_data
        assert provider_context.requires_translation is False
    
    @pytest.mark.asyncio
    async def test_openai_translation(self, sample_conversation_context):
        """Test translation to OpenAI format."""
        translator = ContextTranslator()
        
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.OPENAI,
        )
        
        assert provider_context.provider_type == ProviderType.OPENAI.value
        assert "messages" in provider_context.context_data
    
    @pytest.mark.asyncio
    async def test_google_translation(self, sample_conversation_context):
        """Test translation to Google format."""
        translator = ContextTranslator()
        
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.GOOGLE,
        )
        
        assert provider_context.provider_type == ProviderType.GOOGLE.value
        assert "contents" in provider_context.context_data
    
    @pytest.mark.asyncio
    async def test_round_trip_translation(self, sample_conversation_context):
        """Test round-trip translation (to provider format and back)."""
        translator = ContextTranslator()
        
        # Translate to Anthropic format
        provider_context = await translator.translate_to_provider(
            sample_conversation_context,
            ProviderType.ANTHROPIC,
        )
        
        # Translate back to internal format
        internal_context = await translator.translate_from_provider(
            provider_context,
            ConversationContext,
        )
        
        assert internal_context.title == sample_conversation_context.title
        assert len(internal_context.messages) == len(sample_conversation_context.messages)
    
    @pytest.mark.asyncio
    async def test_context_migration(self, sample_conversation_context):
        """Test context migration between providers."""
        translator = ContextTranslator()
        
        migrated_context = await translator.migrate_context(
            sample_conversation_context,
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
        )
        
        assert ProviderType.OPENAI.value in migrated_context.provider_contexts
        provider_context = migrated_context.provider_contexts[ProviderType.OPENAI.value]
        assert provider_context.provider_type == ProviderType.OPENAI.value


class TestContextVersioning:
    """Test cases for context versioning system."""
    
    @pytest.fixture
    async def mock_persistence(self):
        """Mock persistence layer for version tests."""
        persistence = AsyncMock()
        persistence._get_async_connection = AsyncMock()
        return persistence
    
    @pytest.mark.asyncio
    async def test_version_creation(self, mock_persistence, sample_conversation_context):
        """Test version creation."""
        with patch('src.context.versioning.ContextVersionManager._store_full_version', new_callable=AsyncMock):
            version_manager = ContextVersionManager(mock_persistence)
            
            version = await version_manager.create_version(
                sample_conversation_context,
                user_id="user_123",
                message="Initial version",
            )
            
            assert version.version_number == sample_conversation_context.current_version
            assert version.created_by == "user_123"
            assert version.metadata["message"] == "Initial version"
    
    @pytest.mark.asyncio
    async def test_version_retrieval(self, mock_persistence, sample_conversation_context):
        """Test version retrieval."""
        with patch('src.context.versioning.ContextVersionManager._reconstruct_context_from_version') as mock_reconstruct:
            mock_reconstruct.return_value = sample_conversation_context
            
            version_manager = ContextVersionManager(mock_persistence)
            version_manager.persistence._get_async_connection.return_value.__aenter__.return_value.fetchrow.return_value = {
                "context_id": "test_123",
                "version_number": 1,
                "context_type": "conversation",
                "data_compressed": b"test_data",
                "compression_type": "none",
            }
            
            retrieved_context = await version_manager.get_version("test_123", 1)
            
            assert retrieved_context is not None
            assert retrieved_context.title == sample_conversation_context.title


class TestContextSynchronization:
    """Test cases for context synchronization."""
    
    @pytest.fixture
    async def mock_event_bus(self):
        """Mock event bus for sync tests."""
        event_bus = AsyncMock()
        event_bus.publish_event = AsyncMock()
        event_bus.subscribe_to_room = AsyncMock()
        return event_bus
    
    @pytest.mark.asyncio
    async def test_sync_manager_initialization(self, mock_event_bus):
        """Test sync manager initialization."""
        mock_persistence = AsyncMock()
        
        sync_manager = ContextSyncManager(mock_persistence, mock_event_bus)
        
        assert sync_manager.persistence is mock_persistence
        assert sync_manager.event_bus is mock_event_bus
    
    @pytest.mark.asyncio
    async def test_context_synchronization(self, mock_event_bus, sample_conversation_context):
        """Test context synchronization."""
        mock_persistence = AsyncMock()
        mock_persistence.update_context = AsyncMock(return_value=True)
        
        sync_manager = ContextSyncManager(mock_persistence, mock_event_bus)
        
        success = await sync_manager.synchronize_context(
            sample_conversation_context,
            user_id="user_123",
        )
        
        assert success is True
        mock_event_bus.publish_event.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_conflict_resolution(self, mock_event_bus, sample_conversation_context):
        """Test conflict resolution."""
        mock_persistence = AsyncMock()
        sync_manager = ContextSyncManager(mock_persistence, mock_event_bus)
        
        # Create two different versions of the same context
        local_version = sample_conversation_context
        remote_version = sample_conversation_context.copy(deep=True)
        remote_version.title = "Remote Modified Title"
        
        resolved_context = await sync_manager.handle_sync_conflict(
            "context_123",
            local_version,
            remote_version,
        )
        
        assert resolved_context is not None
        # Should resolve to the latest modified version
        assert resolved_context.title in ["Test Conversation", "Remote Modified Title"]


class TestEventBus:
    """Test cases for the event bus system."""
    
    @pytest.mark.asyncio
    async def test_event_bus_initialization(self):
        """Test event bus initialization."""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock()
            
            event_bus = EventBus()
            await event_bus.initialize()
            
            assert event_bus._redis is not None
            mock_redis_instance.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_publishing(self):
        """Test event publishing."""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock()
            mock_redis_instance.publish = AsyncMock()
            mock_redis_instance.lpush = AsyncMock()
            mock_redis_instance.ltrim = AsyncMock()
            mock_redis_instance.expire = AsyncMock()
            
            event_bus = EventBus()
            await event_bus.initialize()
            
            from src.context.synchronization import SyncEvent
            
            event = SyncEvent(
                event_type="test_event",
                context_id="context_123",
                user_id="user_123",
            )
            
            await event_bus.publish_event(event, room_id="room_456")
            
            # Verify event was published
            mock_redis_instance.publish.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_event_subscription(self):
        """Test event subscription."""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock()
            
            mock_pubsub = AsyncMock()
            mock_redis_instance.pubsub.return_value = mock_pubsub
            mock_pubsub.subscribe = AsyncMock()
            
            event_bus = EventBus()
            await event_bus.initialize()
            
            # Test subscription
            async def mock_handler(event):
                pass
            
            await event_bus.subscribe_to_room("room_123", mock_handler)
            
            mock_pubsub.subscribe.assert_called_once()


class TestPerformanceAndScaling:
    """Test cases for performance and scaling aspects."""
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, context_manager):
        """Test batch context operations."""
        # Mock batch creation
        contexts = []
        for i in range(10):
            context = ConversationContext(
                title=f"Test Conversation {i}",
                messages=[{"role": "user", "content": f"Message {i}"}],
            )
            contexts.append(context)
        
        # Mock the persistence layer for batch operations
        context_manager.persistence.create_context = AsyncMock(side_effect=lambda x: f"context_{x.title.split()[-1]}")
        
        # Create contexts in batch
        created_ids = []
        for context in contexts:
            context_id = await context_manager.create_context(context, user_id="user_123")
            created_ids.append(context_id)
        
        assert len(created_ids) == 10
        assert context_manager._performance_stats["contexts_created"] == 10
    
    @pytest.mark.asyncio
    async def test_large_context_handling(self, context_manager):
        """Test handling of large contexts."""
        # Create a large context with many messages
        large_context = ConversationContext(
            title="Large Conversation",
            messages=[
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 100}
                for i in range(1000)
            ],
        )
        
        # Mock compression for large contexts
        if context_manager.compressor:
            test_data = "large data" * 1000
            compressed_data, stats = context_manager.compressor.compress(test_data)
            
            assert stats["compressed"] is True
            assert stats["compression_ratio"] > 1.0
            assert len(compressed_data) < len(test_data.encode('utf-8'))
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, context_manager, sample_conversation_context):
        """Test concurrent context operations."""
        # Mock persistence layer
        context_manager.persistence.get_context = AsyncMock(return_value=sample_conversation_context)
        context_manager.persistence.update_context = AsyncMock(return_value=True)
        
        # Perform concurrent updates
        async def update_context(i):
            return await context_manager.update_context(
                "context_123",
                {"title": f"Updated Title {i}"},
                user_id="user_123",
            )
        
        tasks = [update_context(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(results)
        assert context_manager._performance_stats["contexts_updated"] == 5
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, context_manager):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many contexts
        contexts = []
        for i in range(100):
            context = ConversationContext(
                title=f"Memory Test {i}",
                messages=[{"role": "user", "content": f"Test message {i}"}],
            )
            contexts.append(context)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 100 contexts)
        assert memory_increase < 100 * 1024 * 1024


class TestErrorHandling:
    """Test cases for error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self, mock_database_url, mock_redis_url):
        """Test handling of database connection errors."""
        with patch('src.context.persistence.ThreadedConnectionPool') as mock_pool:
            mock_pool.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception):
                manager = ContextManager(mock_database_url, mock_redis_url)
                await manager.initialize()
    
    @pytest.mark.asyncio
    async def test_redis_connection_error(self, mock_database_url, mock_redis_url):
        """Test handling of Redis connection errors."""
        with patch('src.context.persistence.ContextPersistenceLayer'):
            with patch('redis.asyncio.from_url') as mock_redis:
                mock_redis.side_effect = Exception("Redis connection failed")
                
                manager = ContextManager(
                    mock_database_url, 
                    mock_redis_url,
                    enable_real_time_sync=True,
                )
                
                with pytest.raises(Exception):
                    await manager.initialize()
    
    @pytest.mark.asyncio
    async def test_invalid_context_data(self, context_manager):
        """Test handling of invalid context data."""
        invalid_data = {"invalid": "data"}
        
        with pytest.raises((ValueError, TypeError)):
            await context_manager.create_context(invalid_data, user_id="user_123")
    
    @pytest.mark.asyncio
    async def test_permission_denied(self, context_manager, sample_conversation_context):
        """Test permission denied scenarios."""
        # Mock security manager to deny permissions
        if context_manager.security_manager:
            context_manager._check_read_permission = AsyncMock(return_value=False)
            
            context_manager.persistence.get_context = AsyncMock(return_value=sample_conversation_context)
            
            with pytest.raises(PermissionError):
                await context_manager.get_context("context_123", user_id="unauthorized_user")


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete end-to-end workflow."""
    with patch('src.context.persistence.ContextPersistenceLayer'):
        with patch('src.context.synchronization.EventBus'):
            # Initialize context manager
            manager = ContextManager(
                database_url="postgresql://test:test@localhost:5432/test_db",
                redis_url="redis://localhost:6379/1",
            )
            await manager.initialize()
            
            try:
                # Mock all the async calls
                manager.persistence.create_context = AsyncMock(return_value="context_123")
                manager.persistence.get_context = AsyncMock()
                manager.persistence.update_context = AsyncMock(return_value=True)
                manager.persistence.query_contexts = AsyncMock(return_value=[])
                
                # 1. Create a conversation context
                conversation = ConversationContext(
                    title="End-to-End Test",
                    messages=[{"role": "user", "content": "Hello world!"}],
                )
                
                context_id = await manager.create_context(conversation, user_id="test_user")
                assert context_id == "context_123"
                
                # 2. Update the context
                manager.persistence.get_context.return_value = conversation
                success = await manager.update_context(
                    context_id,
                    {"title": "Updated Title"},
                    user_id="test_user",
                )
                assert success is True
                
                # 3. Query contexts
                contexts = await manager.query_contexts(
                    ContextQuery(limit=10),
                    user_id="test_user",
                )
                assert isinstance(contexts, list)
                
                # 4. Translate context
                manager.translator.translate_to_provider = AsyncMock(return_value={"translated": True})
                translated = await manager.translate_context(
                    context_id,
                    ProviderType.ANTHROPIC,
                    user_id="test_user",
                )
                assert translated["translated"] is True
                
            finally:
                await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])