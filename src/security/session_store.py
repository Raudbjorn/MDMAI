"""Enhanced session management with Redis support."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from src.security.models import WebSession, SessionStatus

logger = get_logger(__name__)


class SessionStoreConfig(BaseModel):
    """Session store configuration."""

    redis_url: str = "redis://localhost:6379/0"
    session_prefix: str = "session:"
    user_session_prefix: str = "user_sessions:"
    session_ttl_seconds: int = 86400  # 24 hours
    max_sessions_per_user: int = 10
    enable_clustering: bool = False
    cluster_nodes: List[str] = Field(default_factory=list)
    connection_pool_size: int = 50
    socket_timeout: int = 5
    retry_on_timeout: bool = True
    max_retries: int = 3


class SessionStore:
    """Redis-backed session store with clustering support."""

    def __init__(self, config: Optional[SessionStoreConfig] = None):
        """Initialize session store."""
        self.config = config or SessionStoreConfig()
        self._redis: Optional[redis.Redis] = None
        self._local_cache: Dict[str, Tuple[WebSession, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "SessionStore":
        """Enter async context."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        await self.disconnect()

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._redis:
            return
        
        try:
            if self.config.enable_clustering:
                # Redis cluster mode
                from redis.asyncio.cluster import RedisCluster
                self._redis = await RedisCluster.from_url(
                    self.config.redis_url,
                    max_connections=self.config.connection_pool_size,
                    socket_timeout=self.config.socket_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    max_retries=self.config.max_retries,
                )
            else:
                # Single Redis instance
                self._redis = await redis.from_url(
                    self.config.redis_url,
                    max_connections=self.config.connection_pool_size,
                    socket_timeout=self.config.socket_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    max_retries=self.config.max_retries,
                )
            
            # Test connection
            await self._redis.ping()
            logger.info("Connected to Redis session store")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fall back to in-memory storage
            self._redis = None

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def create_session(self, session: WebSession) -> bool:
        """Create new session."""
        try:
            # Check user session limit
            if not await self._check_session_limit(session.user_id):
                await self._evict_oldest_session(session.user_id)
            
            # Store session
            session_key = f"{self.config.session_prefix}{session.session_id}"
            session_data = session.model_dump_json()
            
            if self._redis:
                # Store in Redis with TTL
                await self._redis.setex(
                    session_key,
                    self.config.session_ttl_seconds,
                    session_data,
                )
                
                # Add to user's session set
                user_sessions_key = f"{self.config.user_session_prefix}{session.user_id}"
                await self._redis.sadd(user_sessions_key, session.session_id)
                await self._redis.expire(user_sessions_key, self.config.session_ttl_seconds)
            else:
                # Fallback to local cache
                async with self._lock:
                    self._local_cache[session.session_id] = (session, datetime.utcnow())
            
            logger.debug(f"Created session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[WebSession]:
        """Get session by ID."""
        try:
            # Check local cache first
            if session_id in self._local_cache:
                session, cached_at = self._local_cache[session_id]
                if datetime.utcnow() - cached_at < self._cache_ttl:
                    return session
            
            session_key = f"{self.config.session_prefix}{session_id}"
            
            if self._redis:
                # Get from Redis
                session_data = await self._redis.get(session_key)
                if session_data:
                    session = WebSession.model_validate_json(session_data)
                    
                    # Update local cache
                    async with self._lock:
                        self._local_cache[session_id] = (session, datetime.utcnow())
                    
                    return session
            else:
                # Get from local cache
                if session_id in self._local_cache:
                    session, _ = self._local_cache[session_id]
                    if not session.is_expired():
                        return session
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    async def update_session(self, session: WebSession) -> bool:
        """Update existing session."""
        try:
            session_key = f"{self.config.session_prefix}{session.session_id}"
            session_data = session.model_dump_json()
            
            if self._redis:
                # Update in Redis with TTL reset
                await self._redis.setex(
                    session_key,
                    self.config.session_ttl_seconds,
                    session_data,
                )
            else:
                # Update local cache
                async with self._lock:
                    self._local_cache[session.session_id] = (session, datetime.utcnow())
            
            logger.debug(f"Updated session: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """Delete session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session_key = f"{self.config.session_prefix}{session_id}"
            
            if self._redis:
                # Delete from Redis
                await self._redis.delete(session_key)
                
                # Remove from user's session set
                if session:
                    user_sessions_key = f"{self.config.user_session_prefix}{session.user_id}"
                    await self._redis.srem(user_sessions_key, session_id)
            else:
                # Delete from local cache
                async with self._lock:
                    self._local_cache.pop(session_id, None)
            
            logger.debug(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False

    async def get_user_sessions(self, user_id: str) -> List[WebSession]:
        """Get all sessions for a user."""
        try:
            sessions = []
            
            if self._redis:
                # Get session IDs from set
                user_sessions_key = f"{self.config.user_session_prefix}{user_id}"
                session_ids = await self._redis.smembers(user_sessions_key)
                
                # Get each session
                for session_id in session_ids:
                    session = await self.get_session(session_id.decode() if isinstance(session_id, bytes) else session_id)
                    if session and not session.is_expired():
                        sessions.append(session)
            else:
                # Get from local cache
                async with self._lock:
                    for session_id, (session, _) in self._local_cache.items():
                        if session.user_id == user_id and not session.is_expired():
                            sessions.append(session)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []

    async def delete_user_sessions(self, user_id: str) -> int:
        """Delete all sessions for a user."""
        try:
            sessions = await self.get_user_sessions(user_id)
            deleted = 0
            
            for session in sessions:
                if await self.delete_session(session.session_id):
                    deleted += 1
            
            logger.info(f"Deleted {deleted} sessions for user: {user_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete user sessions: {e}")
            return 0

    async def update_activity(self, session_id: str) -> bool:
        """Update session activity timestamp."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.update_activity()
            return await self.update_session(session)
            
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")
            return False

    async def cleanup_expired(self) -> int:
        """Clean up expired sessions."""
        try:
            cleaned = 0
            
            if self._redis:
                # Redis handles expiration automatically with TTL
                # But we can scan for expired sessions in cache
                cursor = 0
                pattern = f"{self.config.session_prefix}*"
                
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor,
                        match=pattern,
                        count=100,
                    )
                    
                    for key in keys:
                        session_data = await self._redis.get(key)
                        if session_data:
                            try:
                                session = WebSession.model_validate_json(session_data)
                                if session.is_expired():
                                    await self._redis.delete(key)
                                    cleaned += 1
                            except Exception:
                                pass
                    
                    if cursor == 0:
                        break
            else:
                # Clean local cache
                expired = []
                async with self._lock:
                    for session_id, (session, _) in self._local_cache.items():
                        if session.is_expired():
                            expired.append(session_id)
                    
                    for session_id in expired:
                        del self._local_cache[session_id]
                        cleaned += 1
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} expired sessions")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0

    async def _check_session_limit(self, user_id: str) -> bool:
        """Check if user has reached session limit."""
        sessions = await self.get_user_sessions(user_id)
        return len(sessions) < self.config.max_sessions_per_user

    async def _evict_oldest_session(self, user_id: str) -> bool:
        """Evict oldest session for user."""
        sessions = await self.get_user_sessions(user_id)
        if not sessions:
            return False
        
        # Sort by creation time
        sessions.sort(key=lambda s: s.created_at)
        
        # Delete oldest
        return await self.delete_session(sessions[0].session_id)

    async def get_session_count(self) -> int:
        """Get total session count."""
        try:
            if self._redis:
                pattern = f"{self.config.session_prefix}*"
                cursor = 0
                count = 0
                
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor,
                        match=pattern,
                        count=100,
                    )
                    count += len(keys)
                    
                    if cursor == 0:
                        break
                
                return count
            else:
                return len(self._local_cache)
            
        except Exception as e:
            logger.error(f"Failed to get session count: {e}")
            return 0

    async def get_active_users(self) -> Set[str]:
        """Get set of users with active sessions."""
        try:
            users = set()
            
            if self._redis:
                pattern = f"{self.config.user_session_prefix}*"
                cursor = 0
                
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor=cursor,
                        match=pattern,
                        count=100,
                    )
                    
                    for key in keys:
                        # Extract user ID from key
                        user_id = key.decode().replace(self.config.user_session_prefix, "")
                        users.add(user_id)
                    
                    if cursor == 0:
                        break
            else:
                async with self._lock:
                    for _, (session, _) in self._local_cache.items():
                        if not session.is_expired():
                            users.add(session.user_id)
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to get active users: {e}")
            return set()

    async def extend_session(self, session_id: str, minutes: int = 30) -> bool:
        """Extend session expiration."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            # Extend expiration
            session.expires_at += timedelta(minutes=minutes)
            session.update_activity()
            
            return await self.update_session(session)
            
        except Exception as e:
            logger.error(f"Failed to extend session: {e}")
            return False

    async def lock_session(self, session_id: str, reason: str = "") -> bool:
        """Lock a session."""
        try:
            session = await self.get_session(session_id)
            if not session:
                return False
            
            session.status = SessionStatus.LOCKED
            session.metadata["lock_reason"] = reason
            session.metadata["locked_at"] = datetime.utcnow().isoformat()
            
            return await self.update_session(session)
            
        except Exception as e:
            logger.error(f"Failed to lock session: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if self._redis:
                # Check Redis connection
                await self._redis.ping()
                info = await self._redis.info()
                
                return {
                    "status": "healthy",
                    "backend": "redis",
                    "connected": True,
                    "memory_used": info.get("used_memory_human", "N/A"),
                    "connected_clients": info.get("connected_clients", 0),
                    "session_count": await self.get_session_count(),
                    "active_users": len(await self.get_active_users()),
                }
            else:
                return {
                    "status": "degraded",
                    "backend": "memory",
                    "connected": False,
                    "session_count": len(self._local_cache),
                    "message": "Running in fallback mode (in-memory storage)",
                }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }