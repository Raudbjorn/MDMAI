"""Advanced context synchronization and real-time collaboration system."""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

import redis.asyncio as redis
from pydantic import BaseModel

from .models import (
    Context,
    ConflictResolutionStrategy,
    CollaborativeContext,
)

logger = logging.getLogger(__name__)


class SyncEventType:
    """Types of synchronization events."""
    
    CONTEXT_UPDATED = "context_updated"
    CONTEXT_CREATED = "context_created"
    CONTEXT_DELETED = "context_deleted"
    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    CONFLICT_DETECTED = "conflict_detected"
    CONFLICT_RESOLVED = "conflict_resolved"


class SyncEvent(BaseModel):
    """Synchronization event for distributed state updates."""
    
    event_id: str = ""
    event_type: str = ""
    context_id: str = ""
    user_id: Optional[str] = None
    timestamp: datetime = None
    
    # Event payload
    data: Dict[str, Any] = {}
    changes: Dict[str, Any] = {}
    version: int = 0
    
    # Routing and delivery
    target_participants: List[str] = []
    target_rooms: List[str] = []
    propagation_level: str = "room"  # room, global, targeted
    
    # Conflict resolution
    conflict_resolution: Optional[str] = None
    merged_changes: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        if "event_id" not in data:
            data["event_id"] = str(uuid4())
        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


class EventBus:
    """High-performance distributed event bus with Redis backend."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        channel_prefix: str = "ttrpg_context",
        enable_clustering: bool = True,
        max_event_history: int = 1000,
        event_ttl_seconds: int = 3600,
    ):
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.enable_clustering = enable_clustering
        self.max_event_history = max_event_history
        self.event_ttl_seconds = event_ttl_seconds
        
        # Redis connections
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        
        # Event handlers
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._room_handlers: Dict[str, Dict[str, List[Callable]]] = defaultdict(lambda: defaultdict(list))
        
        # Subscription management
        self._subscribed_channels: Set[str] = set()
        self._active_rooms: Set[str] = set()
        
        # Performance tracking
        self._event_stats = {
            "events_published": 0,
            "events_received": 0,
            "events_processed": 0,
            "events_failed": 0,
            "avg_processing_time": 0.0,
        }
        
        # Background tasks
        self._listen_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Event bus initialized",
            redis_url=redis_url,
            channel_prefix=channel_prefix,
            clustering_enabled=enable_clustering,
        )
    
    async def initialize(self) -> None:
        """Initialize Redis connections and start background tasks."""
        try:
            # Connect to Redis
            self._redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
            )
            
            # Test connection
            await self._redis.ping()
            
            # Setup pub/sub
            self._pubsub = self._redis.pubsub()
            
            # Start background tasks
            self._listen_task = asyncio.create_task(self._listen_for_events())
            self._cleanup_task = asyncio.create_task(self._cleanup_old_events())
            
            logger.info("Event bus initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize event bus: {e}")
            raise
    
    async def publish_event(
        self,
        event: SyncEvent,
        room_id: Optional[str] = None,
        global_broadcast: bool = False,
    ) -> None:
        """Publish an event to the distributed system."""
        start_time = time.time()
        
        try:
            if not self._redis:
                await self.initialize()
            
            # Serialize event
            event_data = event.json()
            
            # Determine channels to publish to
            channels = []
            
            if global_broadcast or event.propagation_level == "global":
                channels.append(f"{self.channel_prefix}:global")
            elif room_id or event.propagation_level == "room":
                room_channel = room_id or event.context_id
                channels.append(f"{self.channel_prefix}:room:{room_channel}")
            elif event.propagation_level == "targeted" and event.target_participants:
                for participant in event.target_participants:
                    channels.append(f"{self.channel_prefix}:user:{participant}")
            else:
                # Default to context-specific channel
                channels.append(f"{self.channel_prefix}:context:{event.context_id}")
            
            # Publish to all channels
            for channel in channels:
                await self._redis.publish(channel, event_data)
            
            # Store event history
            await self._store_event_history(event, room_id)
            
            # Update statistics
            execution_time = time.time() - start_time
            self._event_stats["events_published"] += 1
            
            logger.debug(
                "Event published",
                event_id=event.event_id,
                event_type=event.event_type,
                channels=channels,
                execution_time=execution_time,
            )
            
        except Exception as e:
            self._event_stats["events_failed"] += 1
            logger.error(f"Failed to publish event: {e}")
            raise
    
    async def subscribe_to_room(
        self,
        room_id: str,
        handler: Callable[[SyncEvent], None],
        event_types: Optional[List[str]] = None,
    ) -> None:
        """Subscribe to events for a specific room."""
        try:
            if not self._pubsub:
                await self.initialize()
            
            # Register handler
            if event_types:
                for event_type in event_types:
                    self._room_handlers[room_id][event_type].append(handler)
            else:
                self._room_handlers[room_id]["*"].append(handler)
            
            # Subscribe to room channel
            room_channel = f"{self.channel_prefix}:room:{room_id}"
            if room_channel not in self._subscribed_channels:
                await self._pubsub.subscribe(room_channel)
                self._subscribed_channels.add(room_channel)
                self._active_rooms.add(room_id)
            
            logger.debug(
                "Subscribed to room",
                room_id=room_id,
                event_types=event_types or ["all"],
            )
            
        except Exception as e:
            logger.error(f"Failed to subscribe to room {room_id}: {e}")
            raise
    
    async def subscribe_to_context(
        self,
        context_id: str,
        handler: Callable[[SyncEvent], None],
        event_types: Optional[List[str]] = None,
    ) -> None:
        """Subscribe to events for a specific context."""
        try:
            if not self._pubsub:
                await self.initialize()
            
            # Register handler
            if event_types:
                for event_type in event_types:
                    self._handlers[f"context:{context_id}:{event_type}"].append(handler)
            else:
                self._handlers[f"context:{context_id}:*"].append(handler)
            
            # Subscribe to context channel
            context_channel = f"{self.channel_prefix}:context:{context_id}"
            if context_channel not in self._subscribed_channels:
                await self._pubsub.subscribe(context_channel)
                self._subscribed_channels.add(context_channel)
            
            logger.debug(
                "Subscribed to context",
                context_id=context_id,
                event_types=event_types or ["all"],
            )
            
        except Exception as e:
            logger.error(f"Failed to subscribe to context {context_id}: {e}")
            raise
    
    async def subscribe_global(
        self,
        handler: Callable[[SyncEvent], None],
        event_types: Optional[List[str]] = None,
    ) -> None:
        """Subscribe to global events."""
        try:
            if not self._pubsub:
                await self.initialize()
            
            # Register handler
            if event_types:
                for event_type in event_types:
                    self._handlers[f"global:{event_type}"].append(handler)
            else:
                self._handlers["global:*"].append(handler)
            
            # Subscribe to global channel
            global_channel = f"{self.channel_prefix}:global"
            if global_channel not in self._subscribed_channels:
                await self._pubsub.subscribe(global_channel)
                self._subscribed_channels.add(global_channel)
            
            logger.debug(
                "Subscribed to global events",
                event_types=event_types or ["all"],
            )
            
        except Exception as e:
            logger.error(f"Failed to subscribe to global events: {e}")
            raise
    
    async def unsubscribe_from_room(
        self,
        room_id: str,
        handler: Optional[Callable[[SyncEvent], None]] = None,
    ) -> None:
        """Unsubscribe from room events."""
        try:
            if handler:
                # Remove specific handler
                for event_type, handlers in self._room_handlers[room_id].items():
                    if handler in handlers:
                        handlers.remove(handler)
            else:
                # Remove all handlers for room
                if room_id in self._room_handlers:
                    del self._room_handlers[room_id]
            
            # Unsubscribe from channel if no more handlers
            if (room_id not in self._room_handlers or 
                not any(self._room_handlers[room_id].values())):
                room_channel = f"{self.channel_prefix}:room:{room_id}"
                if room_channel in self._subscribed_channels:
                    await self._pubsub.unsubscribe(room_channel)
                    self._subscribed_channels.remove(room_channel)
                    self._active_rooms.discard(room_id)
            
            logger.debug("Unsubscribed from room", room_id=room_id)
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from room {room_id}: {e}")
            raise
    
    async def get_event_history(
        self,
        room_id: Optional[str] = None,
        context_id: Optional[str] = None,
        limit: int = 50,
        event_types: Optional[List[str]] = None,
    ) -> List[SyncEvent]:
        """Get event history for a room or context."""
        try:
            if not self._redis:
                await self.initialize()
            
            # Determine history key
            if room_id:
                history_key = f"{self.channel_prefix}:history:room:{room_id}"
            elif context_id:
                history_key = f"{self.channel_prefix}:history:context:{context_id}"
            else:
                history_key = f"{self.channel_prefix}:history:global"
            
            # Get events from Redis list
            raw_events = await self._redis.lrange(history_key, 0, limit - 1)
            
            events = []
            for raw_event in raw_events:
                try:
                    event = SyncEvent.parse_raw(raw_event)
                    
                    # Filter by event types if specified
                    if event_types and event.event_type not in event_types:
                        continue
                    
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event from history: {e}")
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            return []
    
    async def _listen_for_events(self) -> None:
        """Background task to listen for and process events."""
        try:
            while True:
                try:
                    if not self._pubsub:
                        await asyncio.sleep(1)
                        continue
                    
                    async for message in self._pubsub.listen():
                        if message["type"] == "message":
                            await self._process_event_message(message)
                            
                except redis.ConnectionError:
                    logger.warning("Redis connection lost, reconnecting...")
                    await self._reconnect()
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("Event listener task cancelled")
        except Exception as e:
            logger.error(f"Event listener task failed: {e}")
    
    async def _process_event_message(self, message: Dict[str, Any]) -> None:
        """Process an incoming event message."""
        start_time = time.time()
        
        try:
            # Parse event
            event = SyncEvent.parse_raw(message["data"])
            channel = message["channel"]
            
            # Update statistics
            self._event_stats["events_received"] += 1
            
            # Determine handlers to call
            handlers_to_call = []
            
            # Channel-specific handlers
            if channel.startswith(f"{self.channel_prefix}:room:"):
                room_id = channel.split(":")[-1]
                if room_id in self._room_handlers:
                    # Event-specific handlers
                    handlers_to_call.extend(
                        self._room_handlers[room_id].get(event.event_type, [])
                    )
                    # Wildcard handlers
                    handlers_to_call.extend(
                        self._room_handlers[room_id].get("*", [])
                    )
            
            elif channel.startswith(f"{self.channel_prefix}:context:"):
                context_id = channel.split(":")[-1]
                handlers_to_call.extend(
                    self._handlers.get(f"context:{context_id}:{event.event_type}", [])
                )
                handlers_to_call.extend(
                    self._handlers.get(f"context:{context_id}:*", [])
                )
            
            elif channel == f"{self.channel_prefix}:global":
                handlers_to_call.extend(
                    self._handlers.get(f"global:{event.event_type}", [])
                )
                handlers_to_call.extend(
                    self._handlers.get("global:*", [])
                )
            
            # Call handlers
            for handler in handlers_to_call:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
            
            # Update statistics
            execution_time = time.time() - start_time
            self._event_stats["events_processed"] += 1
            self._update_avg_processing_time(execution_time)
            
            logger.debug(
                "Event processed",
                event_id=event.event_id,
                event_type=event.event_type,
                handlers_called=len(handlers_to_call),
                execution_time=execution_time,
            )
            
        except Exception as e:
            self._event_stats["events_failed"] += 1
            logger.error(f"Failed to process event: {e}")
    
    async def _store_event_history(
        self, event: SyncEvent, room_id: Optional[str] = None
    ) -> None:
        """Store event in history for replay."""
        try:
            if not self._redis:
                return
            
            event_data = event.json()
            
            # Determine history keys
            history_keys = []
            
            if room_id:
                history_keys.append(f"{self.channel_prefix}:history:room:{room_id}")
            if event.context_id:
                history_keys.append(f"{self.channel_prefix}:history:context:{event.context_id}")
            
            # Always store in global history
            history_keys.append(f"{self.channel_prefix}:history:global")
            
            # Store in all relevant history lists
            for history_key in history_keys:
                await self._redis.lpush(history_key, event_data)
                await self._redis.ltrim(history_key, 0, self.max_event_history - 1)
                await self._redis.expire(history_key, self.event_ttl_seconds)
                
        except Exception as e:
            logger.warning(f"Failed to store event history: {e}")
    
    async def _cleanup_old_events(self) -> None:
        """Background task to clean up old events."""
        try:
            while True:
                try:
                    if not self._redis:
                        await asyncio.sleep(60)
                        continue
                    
                    # Clean up expired history entries
                    pattern = f"{self.channel_prefix}:history:*"
                    async for key in self._redis.scan_iter(pattern):
                        # Check if key has TTL set
                        ttl = await self._redis.ttl(key)
                        if ttl == -1:  # No TTL set
                            await self._redis.expire(key, self.event_ttl_seconds)
                    
                    # Sleep for cleanup interval
                    await asyncio.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
    
    async def _reconnect(self) -> None:
        """Reconnect to Redis after connection loss."""
        try:
            if self._redis:
                await self._redis.close()
            if self._pubsub:
                await self._pubsub.close()
            
            # Reinitialize
            await self.initialize()
            
            # Resubscribe to channels
            for channel in list(self._subscribed_channels):
                await self._pubsub.subscribe(channel)
            
        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")
    
    def _update_avg_processing_time(self, new_time: float) -> None:
        """Update average processing time statistic."""
        count = self._event_stats["events_processed"]
        current_avg = self._event_stats["avg_processing_time"]
        
        if count > 1:
            self._event_stats["avg_processing_time"] = (
                (current_avg * (count - 1) + new_time) / count
            )
        else:
            self._event_stats["avg_processing_time"] = new_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get event bus performance statistics."""
        return {
            "event_stats": self._event_stats,
            "active_subscriptions": len(self._subscribed_channels),
            "active_rooms": len(self._active_rooms),
            "registered_handlers": sum(
                len(handlers) for handler_list in self._handlers.values() 
                for handlers in (handler_list if isinstance(handler_list, list) else [handler_list])
            ),
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Cancel background tasks
            if self._listen_task:
                self._listen_task.cancel()
                try:
                    await self._listen_task
                except asyncio.CancelledError:
                    pass
            
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Close connections
            if self._pubsub:
                await self._pubsub.close()
            if self._redis:
                await self._redis.close()
            
            logger.info("Event bus cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class ContextSyncManager:
    """Advanced context synchronization manager with conflict resolution."""
    
    def __init__(
        self,
        persistence_layer,
        event_bus: EventBus,
        default_conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.LATEST_WINS,
        sync_timeout_seconds: int = 30,
        max_sync_retries: int = 3,
    ):
        self.persistence = persistence_layer
        self.event_bus = event_bus
        self.default_conflict_resolution = default_conflict_resolution
        self.sync_timeout_seconds = sync_timeout_seconds
        self.max_sync_retries = max_sync_retries
        
        # Active synchronization locks
        self._sync_locks: Dict[str, asyncio.Lock] = {}
        self._pending_syncs: Dict[str, Dict[str, Any]] = {}
        
        # Conflict tracking
        self._active_conflicts: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self._sync_stats = {
            "syncs_initiated": 0,
            "syncs_completed": 0,
            "syncs_failed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "avg_sync_time": 0.0,
        }
        
        # Event handlers
        self._setup_event_handlers()
        
        logger.info(
            "Context sync manager initialized",
            default_resolution=default_conflict_resolution.value,
            sync_timeout=sync_timeout_seconds,
            max_retries=max_sync_retries,
        )
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for synchronization."""
        # We'll register these when contexts are accessed
        pass
    
    async def synchronize_context(
        self,
        context: Context,
        user_id: Optional[str] = None,
        force_sync: bool = False,
    ) -> bool:
        """Synchronize a context across all participants."""
        start_time = time.time()
        context_id = context.context_id
        
        try:
            # Get or create sync lock
            if context_id not in self._sync_locks:
                self._sync_locks[context_id] = asyncio.Lock()
            
            async with self._sync_locks[context_id]:
                # Check if sync is already pending
                if context_id in self._pending_syncs and not force_sync:
                    logger.debug(f"Sync already pending for context {context_id}")
                    return False
                
                # Mark sync as pending
                self._pending_syncs[context_id] = {
                    "initiated_by": user_id,
                    "start_time": time.time(),
                    "retry_count": 0,
                }
                
                try:
                    # Update sync status
                    await self.persistence.update_context(
                        context_id,
                        {"sync_status": "syncing"},
                        user_id=user_id,
                        create_version=False,
                    )
                    
                    # Create sync event
                    sync_event = SyncEvent(
                        event_type=SyncEventType.CONTEXT_UPDATED,
                        context_id=context_id,
                        user_id=user_id,
                        data=context.dict(),
                        version=context.current_version,
                        propagation_level="room",
                    )
                    
                    # Publish to collaborators
                    if isinstance(context, CollaborativeContext):
                        await self.event_bus.publish_event(
                            sync_event, room_id=context.room_id
                        )
                    else:
                        await self.event_bus.publish_event(sync_event)
                    
                    # Update sync status
                    await self.persistence.update_context(
                        context_id,
                        {
                            "sync_status": "synced",
                            "last_sync": datetime.now(timezone.utc),
                        },
                        user_id=user_id,
                        create_version=False,
                    )
                    
                    # Update statistics
                    execution_time = time.time() - start_time
                    self._sync_stats["syncs_completed"] += 1
                    self._update_avg_sync_time(execution_time)
                    
                    logger.info(
                        "Context synchronized",
                        context_id=context_id,
                        user_id=user_id,
                        execution_time=execution_time,
                    )
                    
                    return True
                    
                finally:
                    # Remove from pending syncs
                    if context_id in self._pending_syncs:
                        del self._pending_syncs[context_id]
            
            self._sync_stats["syncs_initiated"] += 1
            
        except Exception as e:
            self._sync_stats["syncs_failed"] += 1
            logger.error(f"Failed to synchronize context {context_id}: {e}")
            
            # Update sync status to error
            try:
                await self.persistence.update_context(
                    context_id,
                    {"sync_status": f"error: {str(e)}"},
                    user_id=user_id,
                    create_version=False,
                )
            except Exception:
                pass
            
            raise
    
    async def handle_sync_conflict(
        self,
        context_id: str,
        local_version: Context,
        remote_version: Context,
        resolution_strategy: Optional[ConflictResolutionStrategy] = None,
    ) -> Context:
        """Handle synchronization conflict between versions."""
        strategy = resolution_strategy or self.default_conflict_resolution
        
        try:
            self._sync_stats["conflicts_detected"] += 1
            
            # Track conflict
            conflict_id = str(uuid4())
            self._active_conflicts[conflict_id] = {
                "context_id": context_id,
                "local_version": local_version.current_version,
                "remote_version": remote_version.current_version,
                "strategy": strategy.value,
                "start_time": time.time(),
            }
            
            # Create conflict event
            conflict_event = SyncEvent(
                event_type=SyncEventType.CONFLICT_DETECTED,
                context_id=context_id,
                data={
                    "conflict_id": conflict_id,
                    "local_version": local_version.current_version,
                    "remote_version": remote_version.current_version,
                    "strategy": strategy.value,
                },
            )
            
            await self.event_bus.publish_event(conflict_event)
            
            # Resolve conflict based on strategy
            resolved_context = None
            
            if strategy == ConflictResolutionStrategy.LATEST_WINS:
                resolved_context = await self._resolve_latest_wins(local_version, remote_version)
            elif strategy == ConflictResolutionStrategy.MANUAL_MERGE:
                resolved_context = await self._resolve_manual_merge(local_version, remote_version)
            elif strategy == ConflictResolutionStrategy.AUTOMATIC_MERGE:
                resolved_context = await self._resolve_automatic_merge(local_version, remote_version)
            elif strategy == ConflictResolutionStrategy.USER_CHOICE:
                resolved_context = await self._resolve_user_choice(local_version, remote_version)
            else:
                # Default to latest wins
                resolved_context = await self._resolve_latest_wins(local_version, remote_version)
            
            # Update resolved context
            if resolved_context:
                await self.persistence.update_context(
                    context_id,
                    {
                        "data": resolved_context.data,
                        "metadata": resolved_context.metadata,
                        "sync_status": "resolved",
                    },
                    create_version=True,
                )
                
                # Create resolution event
                resolution_event = SyncEvent(
                    event_type=SyncEventType.CONFLICT_RESOLVED,
                    context_id=context_id,
                    data={
                        "conflict_id": conflict_id,
                        "resolution_strategy": strategy.value,
                        "resolved_version": resolved_context.current_version,
                    },
                    merged_changes=resolved_context.data,
                )
                
                await self.event_bus.publish_event(resolution_event)
                
                # Clean up conflict tracking
                if conflict_id in self._active_conflicts:
                    del self._active_conflicts[conflict_id]
                
                self._sync_stats["conflicts_resolved"] += 1
                
                logger.info(
                    "Sync conflict resolved",
                    context_id=context_id,
                    conflict_id=conflict_id,
                    strategy=strategy.value,
                )
                
                return resolved_context
            else:
                raise RuntimeError(f"Failed to resolve conflict with strategy {strategy.value}")
            
        except Exception as e:
            logger.error(f"Failed to resolve sync conflict: {e}")
            raise
    
    async def _resolve_latest_wins(
        self, local_version: Context, remote_version: Context
    ) -> Context:
        """Resolve conflict by choosing the latest version."""
        if local_version.last_modified >= remote_version.last_modified:
            return local_version
        else:
            return remote_version
    
    async def _resolve_manual_merge(
        self, local_version: Context, remote_version: Context
    ) -> Context:
        """Resolve conflict by manual merge (requires user intervention)."""
        # For now, create a context that includes both versions' data
        # In a real implementation, this would trigger a UI for manual resolution
        merged = local_version.copy(deep=True)
        merged.metadata["conflict_resolution"] = "manual_merge_required"
        merged.metadata["conflicting_versions"] = {
            "local": local_version.dict(),
            "remote": remote_version.dict(),
        }
        return merged
    
    async def _resolve_automatic_merge(
        self, local_version: Context, remote_version: Context
    ) -> Context:
        """Resolve conflict by automatic merge of non-conflicting changes."""
        # Simple implementation: merge data dictionaries
        merged = local_version.copy(deep=True)
        
        # Merge data - remote changes win for conflicts
        merged_data = local_version.data.copy()
        merged_data.update(remote_version.data)
        merged.data = merged_data
        
        # Merge metadata
        merged_metadata = local_version.metadata.copy()
        merged_metadata.update(remote_version.metadata)
        merged.metadata = merged_metadata
        merged.metadata["conflict_resolution"] = "automatic_merge"
        
        return merged
    
    async def _resolve_user_choice(
        self, local_version: Context, remote_version: Context
    ) -> Context:
        """Resolve conflict by user choice (placeholder implementation)."""
        # For now, default to local version but mark as requiring user choice
        merged = local_version.copy(deep=True)
        merged.metadata["conflict_resolution"] = "user_choice_required"
        merged.metadata["alternate_version"] = remote_version.dict()
        return merged
    
    def _update_avg_sync_time(self, new_time: float) -> None:
        """Update average sync time statistic."""
        count = self._sync_stats["syncs_completed"]
        current_avg = self._sync_stats["avg_sync_time"]
        
        if count > 1:
            self._sync_stats["avg_sync_time"] = (
                (current_avg * (count - 1) + new_time) / count
            )
        else:
            self._sync_stats["avg_sync_time"] = new_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get synchronization performance statistics."""
        return {
            "sync_stats": self._sync_stats,
            "active_syncs": len(self._pending_syncs),
            "active_conflicts": len(self._active_conflicts),
            "sync_locks": len(self._sync_locks),
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clear locks and pending operations
        self._sync_locks.clear()
        self._pending_syncs.clear()
        self._active_conflicts.clear()
        
        logger.info("Context sync manager cleaned up")