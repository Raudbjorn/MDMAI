"""Main context manager integrating all context management components."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, Union

from ..ai_providers.models import ProviderType
from ..security.enhanced_security_manager import EnhancedSecurityManager
from ..bridge.models import BridgeMessage
from .models import (
    Context,
    ConversationContext,
    SessionContext,
    CollaborativeContext,
    ContextQuery,
    ContextEvent,
    ContextType,
)
from .persistence import ContextPersistenceLayer
from .serialization import ContextSerializer, ContextCompressor
from .synchronization import EventBus, ContextSyncManager
from .translation import ContextTranslator
from .validation import ContextValidator
from .versioning import ContextVersionManager

logger = logging.getLogger(__name__)


class ContextManager:
    """Comprehensive context management system integrating all components."""
    
    def __init__(
        self,
        database_url: str,
        redis_url: str = "redis://localhost:6379/0",
        security_manager: Optional[EnhancedSecurityManager] = None,
        enable_real_time_sync: bool = True,
        enable_compression: bool = True,
        enable_validation: bool = True,
        enable_versioning: bool = True,
    ):
        self.database_url = database_url
        self.redis_url = redis_url
        self.security_manager = security_manager
        self.enable_real_time_sync = enable_real_time_sync
        self.enable_compression = enable_compression
        self.enable_validation = enable_validation
        self.enable_versioning = enable_versioning
        
        # Core components
        self.persistence: Optional[ContextPersistenceLayer] = None
        self.event_bus: Optional[EventBus] = None
        self.sync_manager: Optional[ContextSyncManager] = None
        self.translator: Optional[ContextTranslator] = None
        self.validator: Optional[ContextValidator] = None
        self.version_manager: Optional[ContextVersionManager] = None
        self.serializer: Optional[ContextSerializer] = None
        self.compressor: Optional[ContextCompressor] = None
        
        # State
        self._initialized = False
        
        # Performance tracking
        self._performance_stats = {
            "contexts_created": 0,
            "contexts_retrieved": 0,
            "contexts_updated": 0,
            "contexts_deleted": 0,
            "sync_operations": 0,
            "validation_checks": 0,
            "translation_operations": 0,
        }
        
        logger.info(
            "Context manager initialized",
            real_time_sync=enable_real_time_sync,
            compression=enable_compression,
            validation=enable_validation,
            versioning=enable_versioning,
        )
    
    async def initialize(self) -> None:
        """Initialize all context management components."""
        if self._initialized:
            return
        
        try:
            # Initialize serialization components
            self.serializer = ContextSerializer(enable_async=True)
            
            if self.enable_compression:
                self.compressor = ContextCompressor(enable_async=True)
            
            # Initialize persistence layer
            self.persistence = ContextPersistenceLayer(
                database_url=self.database_url,
                enable_compression=self.enable_compression,
                compressor=self.compressor,
            )
            
            # Initialize event bus for real-time sync
            if self.enable_real_time_sync:
                self.event_bus = EventBus(redis_url=self.redis_url)
                await self.event_bus.initialize()
                
                # Initialize sync manager
                self.sync_manager = ContextSyncManager(
                    persistence_layer=self.persistence,
                    event_bus=self.event_bus,
                )
            
            # Initialize translation system
            self.translator = ContextTranslator()
            
            # Initialize validation system
            if self.enable_validation:
                self.validator = ContextValidator(enable_auto_correction=True)
            
            # Initialize versioning system
            if self.enable_versioning:
                self.version_manager = ContextVersionManager(
                    persistence_layer=self.persistence,
                    serializer=self.serializer,
                    compressor=self.compressor,
                )
            
            self._initialized = True
            
            logger.info("Context manager fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize context manager: {e}")
            raise
    
    async def create_context(
        self,
        context: Union[Context, Dict[str, Any]],
        user_id: Optional[str] = None,
        validate: bool = True,
        create_version: bool = True,
    ) -> str:
        """Create a new context with validation and versioning."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert dict to Context object if needed
            if isinstance(context, dict):
                context_type = context.get("context_type", "conversation")
                if context_type == "conversation":
                    context = ConversationContext(**context)
                elif context_type == "session":
                    context = SessionContext(**context)
                elif context_type == "collaborative":
                    context = CollaborativeContext(**context)
                else:
                    context = Context(**context)
            
            # Security check
            if self.security_manager and user_id:
                if not await self._check_create_permission(user_id, context):
                    raise PermissionError(f"User {user_id} does not have permission to create this context")
            
            # Validate context
            if validate and self.validator:
                validation_result = await self.validator.validate_context(context)
                if not validation_result.is_valid:
                    raise ValueError(f"Context validation failed: {validation_result.issues}")
                
                # Apply auto-corrections if available
                if validation_result.corrected_data:
                    context = type(context)(**validation_result.corrected_data)
                
                self._performance_stats["validation_checks"] += 1
            
            # Set ownership
            if user_id:
                context.owner_id = user_id
            
            # Create context in persistence layer
            context_id = await self.persistence.create_context(context)
            
            # Create initial version if enabled
            if create_version and self.version_manager:
                await self.version_manager.create_version(context, user_id=user_id)
            
            # Publish creation event
            if self.event_bus:
                event = ContextEvent(
                    context_id=context_id,
                    event_type="create",
                    user_id=user_id,
                    new_state=context.dict(),
                )
                await self._publish_context_event(event, context)
            
            self._performance_stats["contexts_created"] += 1
            
            logger.info(
                "Context created",
                context_id=context_id,
                context_type=context.context_type.value,
                user_id=user_id,
            )
            
            return context_id
            
        except Exception as e:
            logger.error(f"Failed to create context: {e}")
            raise
    
    async def get_context(
        self,
        context_id: str,
        user_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[Context]:
        """Retrieve a context with access control."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get specific version if requested
            if version and self.version_manager:
                context = await self.version_manager.get_version(context_id, version, user_id=user_id)
            else:
                context = await self.persistence.get_context(context_id, user_id=user_id)
            
            if not context:
                return None
            
            # Security check
            if self.security_manager and user_id:
                if not await self._check_read_permission(user_id, context):
                    raise PermissionError(f"User {user_id} does not have permission to read this context")
            
            self._performance_stats["contexts_retrieved"] += 1
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context {context_id}: {e}")
            raise
    
    async def update_context(
        self,
        context_id: str,
        updates: Dict[str, Any],
        user_id: Optional[str] = None,
        validate: bool = True,
        create_version: bool = True,
        sync: bool = True,
    ) -> bool:
        """Update a context with validation, versioning, and synchronization."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get current context for permissions check
            current_context = await self.persistence.get_context(context_id, user_id=user_id)
            if not current_context:
                return False
            
            # Security check
            if self.security_manager and user_id:
                if not await self._check_update_permission(user_id, current_context):
                    raise PermissionError(f"User {user_id} does not have permission to update this context")
            
            # Validate updates if requested
            if validate and self.validator:
                # Create updated context for validation
                updated_data = current_context.dict()
                updated_data.update(updates)
                
                validation_result = await self.validator.validate_context(updated_data, type(current_context))
                if not validation_result.is_valid:
                    raise ValueError(f"Update validation failed: {validation_result.issues}")
                
                # Use corrected data if available
                if validation_result.corrected_data:
                    updates = {
                        key: validation_result.corrected_data[key] 
                        for key in updates.keys() 
                        if key in validation_result.corrected_data
                    }
                
                self._performance_stats["validation_checks"] += 1
            
            # Store previous state for events
            previous_state = current_context.dict()
            
            # Update context
            success = await self.persistence.update_context(
                context_id, updates, user_id=user_id, create_version=create_version
            )
            
            if not success:
                return False
            
            # Create version if enabled and not already created
            if not create_version and self.version_manager:
                updated_context = await self.persistence.get_context(context_id, user_id=user_id)
                if updated_context:
                    await self.version_manager.create_version(updated_context, user_id=user_id)
            
            # Synchronize if enabled
            if sync and self.sync_manager:
                updated_context = await self.persistence.get_context(context_id, user_id=user_id)
                if updated_context:
                    await self.sync_manager.synchronize_context(updated_context, user_id=user_id)
                    self._performance_stats["sync_operations"] += 1
            
            # Publish update event
            if self.event_bus:
                updated_context = await self.persistence.get_context(context_id, user_id=user_id)
                if updated_context:
                    event = ContextEvent(
                        context_id=context_id,
                        event_type="update",
                        user_id=user_id,
                        changes=updates,
                        previous_state=previous_state,
                        new_state=updated_context.dict(),
                    )
                    await self._publish_context_event(event, updated_context)
            
            self._performance_stats["contexts_updated"] += 1
            
            logger.info(
                "Context updated",
                context_id=context_id,
                user_id=user_id,
                changes=list(updates.keys()),
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update context {context_id}: {e}")
            raise
    
    async def delete_context(
        self,
        context_id: str,
        user_id: Optional[str] = None,
        hard_delete: bool = False,
    ) -> bool:
        """Delete a context with access control."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get context for permissions check
            context = await self.persistence.get_context(context_id, user_id=user_id)
            if not context:
                return False
            
            # Security check
            if self.security_manager and user_id:
                if not await self._check_delete_permission(user_id, context):
                    raise PermissionError(f"User {user_id} does not have permission to delete this context")
            
            # Delete context
            success = await self.persistence.delete_context(
                context_id, user_id=user_id, hard_delete=hard_delete
            )
            
            if success:
                # Publish deletion event
                if self.event_bus:
                    event = ContextEvent(
                        context_id=context_id,
                        event_type="delete" if hard_delete else "soft_delete",
                        user_id=user_id,
                        previous_state=context.dict(),
                        metadata={"hard_delete": hard_delete},
                    )
                    await self._publish_context_event(event, context)
                
                self._performance_stats["contexts_deleted"] += 1
                
                logger.info(
                    "Context deleted",
                    context_id=context_id,
                    user_id=user_id,
                    hard_delete=hard_delete,
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete context {context_id}: {e}")
            raise
    
    async def query_contexts(
        self,
        query: Union[ContextQuery, Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[Context]:
        """Query contexts with access control."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Convert dict to ContextQuery if needed
            if isinstance(query, dict):
                query = ContextQuery(**query)
            
            # Execute query
            contexts = await self.persistence.query_contexts(query, user_id=user_id)
            
            # Apply additional security filtering if needed
            if self.security_manager and user_id:
                filtered_contexts = []
                for context in contexts:
                    if await self._check_read_permission(user_id, context):
                        filtered_contexts.append(context)
                contexts = filtered_contexts
            
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to query contexts: {e}")
            raise
    
    async def translate_context(
        self,
        context_id: str,
        target_provider: ProviderType,
        user_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Translate context to provider-specific format."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get context
            context = await self.get_context(context_id, user_id=user_id)
            if not context:
                raise ValueError(f"Context {context_id} not found")
            
            # Translate
            provider_context = await self.translator.translate_to_provider(
                context, target_provider, options
            )
            
            self._performance_stats["translation_operations"] += 1
            
            return provider_context
            
        except Exception as e:
            logger.error(f"Failed to translate context {context_id}: {e}")
            raise
    
    async def migrate_context(
        self,
        context_id: str,
        source_provider: ProviderType,
        target_provider: ProviderType,
        user_id: Optional[str] = None,
        migration_options: Optional[Dict[str, Any]] = None,
    ) -> Context:
        """Migrate context between providers."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get context
            context = await self.get_context(context_id, user_id=user_id)
            if not context:
                raise ValueError(f"Context {context_id} not found")
            
            # Migrate
            migrated_context = await self.translator.migrate_context(
                context, source_provider, target_provider, migration_options
            )
            
            # Update context with migrated data
            await self.update_context(
                context_id,
                {"provider_contexts": migrated_context.provider_contexts},
                user_id=user_id,
                sync=True,
            )
            
            self._performance_stats["translation_operations"] += 1
            
            logger.info(
                "Context migrated",
                context_id=context_id,
                source_provider=source_provider.value,
                target_provider=target_provider.value,
            )
            
            return migrated_context
            
        except Exception as e:
            logger.error(f"Failed to migrate context {context_id}: {e}")
            raise
    
    async def join_collaborative_session(
        self,
        context_id: str,
        user_id: str,
        room_id: Optional[str] = None,
    ) -> bool:
        """Join a collaborative context session."""
        if not self._initialized:
            await self.initialize()
        
        try:
            context = await self.get_context(context_id, user_id=user_id)
            if not context or not isinstance(context, CollaborativeContext):
                return False
            
            # Add user to participants
            if user_id not in context.active_participants:
                context.add_participant(user_id)
                
                # Update context
                await self.update_context(
                    context_id,
                    {"active_participants": context.active_participants},
                    user_id=user_id,
                    sync=True,
                )
            
            # Subscribe to real-time events
            if self.event_bus:
                await self.event_bus.subscribe_to_room(
                    room_id or context.room_id,
                    self._handle_collaborative_event,
                    event_types=["context_updated", "participant_joined", "participant_left"],
                )
                
                # Publish join event
                event = ContextEvent(
                    context_id=context_id,
                    event_type="participant_joined",
                    user_id=user_id,
                    data={"room_id": room_id or context.room_id},
                )
                await self._publish_context_event(event, context, room_id or context.room_id)
            
            logger.info(
                "User joined collaborative session",
                context_id=context_id,
                user_id=user_id,
                room_id=room_id or context.room_id,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to join collaborative session: {e}")
            raise
    
    async def leave_collaborative_session(
        self,
        context_id: str,
        user_id: str,
        room_id: Optional[str] = None,
    ) -> bool:
        """Leave a collaborative context session."""
        if not self._initialized:
            await self.initialize()
        
        try:
            context = await self.get_context(context_id, user_id=user_id)
            if not context or not isinstance(context, CollaborativeContext):
                return False
            
            # Remove user from participants
            if user_id in context.active_participants:
                context.remove_participant(user_id)
                
                # Update context
                await self.update_context(
                    context_id,
                    {
                        "active_participants": context.active_participants,
                        "locked_by": context.locked_by,
                        "locked_at": context.locked_at,
                    },
                    user_id=user_id,
                    sync=True,
                )
            
            # Unsubscribe from events
            if self.event_bus:
                await self.event_bus.unsubscribe_from_room(room_id or context.room_id)
                
                # Publish leave event
                event = ContextEvent(
                    context_id=context_id,
                    event_type="participant_left",
                    user_id=user_id,
                    data={"room_id": room_id or context.room_id},
                )
                await self._publish_context_event(event, context, room_id or context.room_id)
            
            logger.info(
                "User left collaborative session",
                context_id=context_id,
                user_id=user_id,
                room_id=room_id or context.room_id,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to leave collaborative session: {e}")
            raise
    
    async def _publish_context_event(
        self, event: ContextEvent, context: Context, room_id: Optional[str] = None
    ) -> None:
        """Publish a context event to the event bus."""
        if not self.event_bus:
            return
        
        try:
            from .synchronization import SyncEvent
            
            sync_event = SyncEvent(
                event_type=event.event_type,
                context_id=event.context_id,
                user_id=event.user_id,
                data=event.new_state or {},
                changes=event.changes or {},
                version=context.current_version,
            )
            
            if isinstance(context, CollaborativeContext):
                await self.event_bus.publish_event(sync_event, room_id=room_id or context.room_id)
            else:
                await self.event_bus.publish_event(sync_event)
                
        except Exception as e:
            logger.warning(f"Failed to publish context event: {e}")
    
    async def _handle_collaborative_event(self, event) -> None:
        """Handle collaborative context events."""
        try:
            logger.debug(
                "Received collaborative event",
                event_type=event.event_type,
                context_id=event.context_id,
                user_id=event.user_id,
            )
            
            # Handle different event types
            if event.event_type == "context_updated":
                # Update local cache or notify UI
                pass
            elif event.event_type in ["participant_joined", "participant_left"]:
                # Update participant list
                pass
                
        except Exception as e:
            logger.error(f"Failed to handle collaborative event: {e}")
    
    async def _check_create_permission(self, user_id: str, context: Context) -> bool:
        """Check if user has permission to create context."""
        if not self.security_manager:
            return True
        
        # Implementation depends on security manager interface
        return True  # Placeholder
    
    async def _check_read_permission(self, user_id: str, context: Context) -> bool:
        """Check if user has permission to read context."""
        if not self.security_manager:
            return True
        
        # Check ownership and collaboration
        if context.owner_id == user_id:
            return True
        
        if user_id in context.collaborators:
            return True
        
        # Check with security manager for additional rules
        return True  # Placeholder
    
    async def _check_update_permission(self, user_id: str, context: Context) -> bool:
        """Check if user has permission to update context."""
        return await self._check_read_permission(user_id, context)
    
    async def _check_delete_permission(self, user_id: str, context: Context) -> bool:
        """Check if user has permission to delete context."""
        if not self.security_manager:
            return True
        
        # Only owner can delete by default
        return context.owner_id == user_id
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "context_manager": self._performance_stats,
        }
        
        # Gather stats from all components asynchronously
        tasks = []
        components = []
        
        if self.persistence:
            tasks.append(self.persistence.get_performance_stats())
            components.append("persistence")
        
        if self.event_bus:
            # If event_bus has async method, await it
            if hasattr(self.event_bus.get_performance_stats, '__call__'):
                if asyncio.iscoroutinefunction(self.event_bus.get_performance_stats):
                    tasks.append(self.event_bus.get_performance_stats())
                    components.append("event_bus")
                else:
                    stats["event_bus"] = self.event_bus.get_performance_stats()
        
        if self.sync_manager:
            if hasattr(self.sync_manager, 'get_performance_stats'):
                if asyncio.iscoroutinefunction(self.sync_manager.get_performance_stats):
                    tasks.append(self.sync_manager.get_performance_stats())
                    components.append("sync_manager")
                else:
                    stats["sync_manager"] = self.sync_manager.get_performance_stats()
        
        if self.translator:
            if hasattr(self.translator, 'get_performance_stats'):
                if asyncio.iscoroutinefunction(self.translator.get_performance_stats):
                    tasks.append(self.translator.get_performance_stats())
                    components.append("translator")
                else:
                    stats["translator"] = self.translator.get_performance_stats()
        
        if self.validator:
            if hasattr(self.validator, 'get_performance_stats'):
                if asyncio.iscoroutinefunction(self.validator.get_performance_stats):
                    tasks.append(self.validator.get_performance_stats())
                    components.append("validator")
                else:
                    stats["validator"] = self.validator.get_performance_stats()
        
        if self.version_manager:
            if hasattr(self.version_manager, 'get_performance_stats'):
                if asyncio.iscoroutinefunction(self.version_manager.get_performance_stats):
                    tasks.append(self.version_manager.get_performance_stats())
                    components.append("version_manager")
                else:
                    stats["version_manager"] = self.version_manager.get_performance_stats()
        
        # Await all async stats gathering
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for component, result in zip(components, results):
                if not isinstance(result, Exception):
                    stats[component] = result
                else:
                    logger.warning(f"Failed to get stats from {component}: {result}")
                    stats[component] = {"error": str(result)}
        
        return stats
    
    # Keep the old method for backwards compatibility but mark as deprecated
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics (deprecated - use get_stats() instead)."""
        import warnings
        warnings.warn("get_performance_stats() is deprecated, use get_stats() instead", DeprecationWarning)
        
        # Return synchronous stats only
        return {
            "context_manager": self._performance_stats,
            "warning": "This method is deprecated and returns partial stats. Use get_stats() for complete async stats."
        }
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            cleanup_tasks = []
            
            if self.persistence:
                cleanup_tasks.append(self.persistence.cleanup())
            
            if self.event_bus:
                cleanup_tasks.append(self.event_bus.cleanup())
            
            if self.sync_manager:
                cleanup_tasks.append(self.sync_manager.cleanup())
            
            if self.version_manager:
                cleanup_tasks.append(self.version_manager.cleanup())
            
            if self.translator:
                self.translator.cleanup()
            
            if self.validator:
                self.validator.cleanup()
            
            if self.serializer:
                self.serializer.cleanup()
            
            if self.compressor:
                self.compressor.cleanup()
            
            # Wait for all cleanup tasks
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self._initialized = False
            
            logger.info("Context manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
        return False