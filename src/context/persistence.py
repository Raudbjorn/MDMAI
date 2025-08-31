"""PostgreSQL-based context persistence layer with advanced optimizations."""

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from psycopg2.pool import ThreadedConnectionPool

from .models import (
    Context,
    ContextQuery,
    ContextState,
    ContextType,
    CompressionType,
)
from .serialization import ContextSerializer, ContextCompressor

logger = logging.getLogger(__name__)


class ContextPersistenceLayer:
    """High-performance PostgreSQL-based context persistence with advanced optimizations."""
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 30,
        enable_compression: bool = True,
        default_compression: CompressionType = CompressionType.ZSTD,
        enable_partitioning: bool = True,
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.enable_compression = enable_compression
        self.default_compression = default_compression
        self.enable_partitioning = enable_partitioning
        
        # Connection pools
        self._sync_pool: Optional[ThreadedConnectionPool] = None
        self._async_pool: Optional[asyncpg.Pool] = None
        
        # Components
        self.serializer = ContextSerializer()
        self.compressor = ContextCompressor() if enable_compression else None
        
        # Performance tracking
        self._query_stats: Dict[str, Dict[str, float]] = {}
        self._cache_stats: Dict[str, int] = {"hits": 0, "misses": 0}
        
        # Schema versioning
        self.schema_version = "1.0.0"
        
        # Initialize
        self._initialize_sync()
    
    def _initialize_sync(self) -> None:
        """Initialize synchronous connection pool and schema."""
        try:
            # Parse connection parameters
            import urllib.parse
            parsed = urllib.parse.urlparse(self.database_url)
            
            self._sync_pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=self.pool_size,
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else "postgres",
                user=parsed.username,
                password=parsed.password,
            )
            
            # Initialize schema
            self._create_schema()
            
            logger.info(
                "Context persistence layer initialized",
                pool_size=self.pool_size,
                compression_enabled=self.enable_compression,
                partitioning_enabled=self.enable_partitioning,
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize persistence layer: {e}")
            raise
    
    async def initialize_async(self) -> None:
        """Initialize asynchronous connection pool."""
        try:
            self._async_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=30,
            )
            
            logger.info("Async connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize async pool: {e}")
            raise
    
    def _create_schema(self) -> None:
        """Create database schema with optimizations."""
        schema_sql = """
        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";
        CREATE EXTENSION IF NOT EXISTS "btree_gin";
        
        -- Contexts table with partitioning support
        CREATE TABLE IF NOT EXISTS contexts (
            context_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            context_type VARCHAR(50) NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            
            -- JSON data with GIN index for fast queries
            data JSONB NOT NULL DEFAULT '{}',
            metadata JSONB NOT NULL DEFAULT '{}',
            
            -- Versioning
            current_version INTEGER NOT NULL DEFAULT 1,
            version_history UUID[] DEFAULT '{}',
            
            -- Ownership and permissions
            owner_id TEXT,
            collaborators TEXT[] DEFAULT '{}',
            permissions JSONB DEFAULT '{}',
            
            -- State management
            state VARCHAR(20) NOT NULL DEFAULT 'active',
            last_modified TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            access_count INTEGER NOT NULL DEFAULT 0,
            
            -- Synchronization
            sync_status TEXT,
            last_sync TIMESTAMPTZ,
            conflict_resolution VARCHAR(50) DEFAULT 'latest_wins',
            
            -- Provider integration
            provider_contexts JSONB DEFAULT '{}',
            
            -- Storage optimization
            compression_type VARCHAR(20) DEFAULT 'zstd',
            size_bytes INTEGER DEFAULT 0,
            compressed_size_bytes INTEGER DEFAULT 0,
            
            -- Lifecycle
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ,
            auto_archive_days INTEGER DEFAULT 90,
            
            -- Constraints
            CONSTRAINT valid_context_type CHECK (context_type IN ('conversation', 'session', 'campaign', 'character', 'collaborative', 'provider_specific')),
            CONSTRAINT valid_state CHECK (state IN ('active', 'inactive', 'archived', 'deleted', 'syncing', 'conflict')),
            CONSTRAINT valid_compression CHECK (compression_type IN ('none', 'gzip', 'lz4', 'zstd', 'brotli')),
            CONSTRAINT positive_version CHECK (current_version > 0),
            CONSTRAINT positive_access_count CHECK (access_count >= 0)
        );
        
        -- Partition by context_type if enabled
        """ + ("""
        -- Create partitions for each context type
        CREATE TABLE IF NOT EXISTS contexts_conversation PARTITION OF contexts
            FOR VALUES IN ('conversation');
        CREATE TABLE IF NOT EXISTS contexts_session PARTITION OF contexts
            FOR VALUES IN ('session');
        CREATE TABLE IF NOT EXISTS contexts_campaign PARTITION OF contexts
            FOR VALUES IN ('campaign');
        CREATE TABLE IF NOT EXISTS contexts_character PARTITION OF contexts
            FOR VALUES IN ('character');
        CREATE TABLE IF NOT EXISTS contexts_collaborative PARTITION OF contexts
            FOR VALUES IN ('collaborative');
        CREATE TABLE IF NOT EXISTS contexts_provider_specific PARTITION OF contexts
            FOR VALUES IN ('provider_specific');
        """ if self.enable_partitioning else "") + """
        
        -- Context versions table
        CREATE TABLE IF NOT EXISTS context_versions (
            version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            context_id UUID NOT NULL,
            version_number INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_by TEXT,
            parent_version UUID,
            branch VARCHAR(100) NOT NULL DEFAULT 'main',
            
            -- Version data (compressed)
            data_compressed BYTEA,
            checksum VARCHAR(64) NOT NULL,
            compressed BOOLEAN DEFAULT false,
            compression_type VARCHAR(20) DEFAULT 'none',
            size_bytes INTEGER DEFAULT 0,
            
            -- Metadata
            metadata JSONB DEFAULT '{}',
            tags TEXT[] DEFAULT '{}',
            
            FOREIGN KEY (context_id) REFERENCES contexts(context_id) ON DELETE CASCADE,
            UNIQUE(context_id, version_number, branch)
        );
        
        -- Context events table
        CREATE TABLE IF NOT EXISTS context_events (
            event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            context_id UUID NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            
            -- Event data
            user_id TEXT,
            changes JSONB DEFAULT '{}',
            previous_state JSONB,
            new_state JSONB,
            
            -- Event metadata
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            source VARCHAR(50) NOT NULL DEFAULT 'system',
            metadata JSONB DEFAULT '{}',
            
            -- Synchronization
            propagated BOOLEAN DEFAULT false,
            propagation_targets TEXT[] DEFAULT '{}',
            
            FOREIGN KEY (context_id) REFERENCES contexts(context_id) ON DELETE CASCADE
        );
        
        -- Performance optimization indexes
        CREATE INDEX IF NOT EXISTS idx_contexts_type_state ON contexts(context_type, state);
        CREATE INDEX IF NOT EXISTS idx_contexts_owner ON contexts(owner_id) WHERE owner_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_contexts_collaborators ON contexts USING GIN(collaborators) WHERE array_length(collaborators, 1) > 0;
        CREATE INDEX IF NOT EXISTS idx_contexts_last_modified ON contexts(last_modified DESC);
        CREATE INDEX IF NOT EXISTS idx_contexts_last_accessed ON contexts(last_accessed DESC);
        CREATE INDEX IF NOT EXISTS idx_contexts_created_at ON contexts(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_contexts_metadata ON contexts USING GIN(metadata);
        CREATE INDEX IF NOT EXISTS idx_contexts_data ON contexts USING GIN(data);
        CREATE INDEX IF NOT EXISTS idx_contexts_text_search ON contexts USING GIN(to_tsvector('english', title || ' ' || description));
        
        -- Version indexes
        CREATE INDEX IF NOT EXISTS idx_context_versions_context_version ON context_versions(context_id, version_number DESC);
        CREATE INDEX IF NOT EXISTS idx_context_versions_created_at ON context_versions(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_context_versions_branch ON context_versions(context_id, branch);
        
        -- Event indexes
        CREATE INDEX IF NOT EXISTS idx_context_events_context_timestamp ON context_events(context_id, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_context_events_type ON context_events(event_type);
        CREATE INDEX IF NOT EXISTS idx_context_events_user ON context_events(user_id) WHERE user_id IS NOT NULL;
        CREATE INDEX IF NOT EXISTS idx_context_events_unpropagated ON context_events(propagated) WHERE propagated = false;
        
        -- Materialized view for context statistics
        CREATE MATERIALIZED VIEW IF NOT EXISTS context_stats AS
        SELECT 
            context_type,
            state,
            COUNT(*) as count,
            AVG(size_bytes) as avg_size_bytes,
            AVG(access_count) as avg_access_count,
            MAX(last_modified) as most_recent_modification
        FROM contexts 
        GROUP BY context_type, state;
        
        CREATE UNIQUE INDEX IF NOT EXISTS idx_context_stats_type_state ON context_stats(context_type, state);
        """
        
        with self._get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
        
        logger.info("Database schema created successfully")
    
    @asynccontextmanager
    async def _get_async_connection(self):
        """Get async database connection from pool."""
        if not self._async_pool:
            await self.initialize_async()
        
        conn = await self._async_pool.acquire()
        try:
            yield conn
        finally:
            await self._async_pool.release(conn)
    
    @asynccontextmanager
    def _get_sync_connection(self):
        """Get synchronous database connection from pool."""
        if not self._sync_pool:
            raise RuntimeError("Sync pool not initialized")
        
        conn = self._sync_pool.getconn()
        try:
            yield conn
        finally:
            self._sync_pool.putconn(conn)
    
    def _serialize_and_compress(self, context: Context) -> Tuple[bytes, Dict[str, Any]]:
        """Serialize and optionally compress context data."""
        # Serialize to JSON
        serialized = self.serializer.serialize(context)
        
        # Compress if enabled
        if self.compressor and self.enable_compression:
            compressed_data, compression_stats = self.compressor.compress(
                serialized, self.default_compression
            )
            return compressed_data, compression_stats
        
        return serialized.encode('utf-8'), {
            "compressed": False,
            "compression_type": "none",
            "original_size": len(serialized),
            "compressed_size": len(serialized),
            "compression_ratio": 1.0,
        }
    
    def _decompress_and_deserialize(
        self, data: bytes, compression_type: CompressionType, target_type: type = Context
    ) -> Context:
        """Decompress and deserialize context data."""
        # Decompress if needed
        if compression_type != CompressionType.NONE and self.compressor:
            decompressed = self.compressor.decompress(data, compression_type)
        else:
            decompressed = data.decode('utf-8')
        
        # Deserialize
        return self.serializer.deserialize(decompressed, target_type)
    
    async def create_context(self, context: Context) -> str:
        """Create a new context with optimized storage."""
        start_time = time.time()
        
        try:
            # Serialize and compress
            data_bytes, compression_stats = self._serialize_and_compress(context)
            
            # Store in database
            async with self._get_async_connection() as conn:
                # Prepare metadata without duplicating fields stored separately
                metadata_to_store = context.metadata.copy() if context.metadata else {}
                
                # Remove fields that are stored in dedicated columns to avoid duplication
                fields_to_exclude = {
                    'title', 'description', 'owner_id', 'collaborators', 
                    'permissions', 'state', 'expires_at', 'auto_archive_days',
                    'context_type', 'current_version', 'version_history'
                }
                
                for field in fields_to_exclude:
                    metadata_to_store.pop(field, None)
                
                result = await conn.fetchrow("""
                    INSERT INTO contexts (
                        context_id, context_type, title, description,
                        data, metadata, current_version, version_history,
                        owner_id, collaborators, permissions,
                        state, compression_type, size_bytes, compressed_size_bytes,
                        expires_at, auto_archive_days
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
                    ) RETURNING context_id
                """, 
                    context.context_id,
                    context.context_type.value,
                    context.title,
                    context.description,
                    context.data,
                    metadata_to_store,  # Use filtered metadata
                    context.current_version,
                    context.version_history,
                    context.owner_id,
                    context.collaborators,
                    context.permissions,
                    context.state.value,
                    compression_stats["compression_type"],
                    compression_stats["original_size"],
                    compression_stats["compressed_size"],
                    context.expires_at,
                    context.auto_archive_days,
                )
                
                # Create initial version
                await self._create_version(conn, context, data_bytes, compression_stats)
                
                # Log creation event
                await self._log_event(
                    conn, context.context_id, "create", 
                    user_id=context.owner_id,
                    new_state=context.dict(),
                )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("create_context", execution_time)
            
            logger.info(
                "Context created",
                context_id=context.context_id,
                context_type=context.context_type.value,
                size_bytes=compression_stats["original_size"],
                compressed_size=compression_stats["compressed_size"],
                compression_ratio=compression_stats["compression_ratio"],
                execution_time=execution_time,
            )
            
            return result["context_id"]
            
        except Exception as e:
            logger.error(f"Failed to create context: {e}", exc_info=True)
            raise
    
    async def get_context(self, context_id: str, user_id: Optional[str] = None) -> Optional[Context]:
        """Retrieve a context with access control and caching."""
        start_time = time.time()
        
        try:
            async with self._get_async_connection() as conn:
                # Get context with access check
                row = await conn.fetchrow("""
                    SELECT * FROM contexts 
                    WHERE context_id = $1 
                    AND state != 'deleted'
                    AND (owner_id = $2 OR $2 = ANY(collaborators) OR $2 IS NULL)
                """, context_id, user_id)
                
                if not row:
                    self._cache_stats["misses"] += 1
                    return None
                
                # Update access tracking
                await conn.execute("""
                    UPDATE contexts 
                    SET last_accessed = NOW(), access_count = access_count + 1
                    WHERE context_id = $1
                """, context_id)
                
                # Convert to context object
                context = self._row_to_context(row)
                
                # Track performance
                execution_time = time.time() - start_time
                self._track_query_performance("get_context", execution_time)
                self._cache_stats["hits"] += 1
                
                logger.debug(
                    "Context retrieved",
                    context_id=context_id,
                    context_type=context.context_type.value,
                    execution_time=execution_time,
                )
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get context {context_id}: {e}")
            raise
    
    async def update_context(
        self, 
        context_id: str, 
        updates: Dict[str, Any], 
        user_id: Optional[str] = None,
        create_version: bool = True,
    ) -> bool:
        """Update context with optimistic locking and versioning."""
        start_time = time.time()
        
        try:
            async with self._get_async_connection() as conn:
                async with conn.transaction():
                    # Get current context for optimistic locking
                    current = await conn.fetchrow("""
                        SELECT * FROM contexts 
                        WHERE context_id = $1 
                        AND state != 'deleted'
                        FOR UPDATE
                    """, context_id)
                    
                    if not current:
                        return False
                    
                    # Check permissions
                    if user_id and current["owner_id"] != user_id and user_id not in (current["collaborators"] or []):
                        raise PermissionError(f"User {user_id} does not have permission to update context {context_id}")
                    
                    # Build update query dynamically
                    set_clauses = []
                    params = [context_id]
                    param_idx = 2
                    
                    allowed_fields = {
                        "title", "description", "data", "metadata", "state", 
                        "collaborators", "permissions", "expires_at", "auto_archive_days"
                    }
                    
                    for field, value in updates.items():
                        if field in allowed_fields:
                            set_clauses.append(f"{field} = ${param_idx}")
                            params.append(value)
                            param_idx += 1
                    
                    if not set_clauses:
                        return True  # No valid updates
                    
                    # Always update modification time and version
                    set_clauses.extend([
                        "last_modified = NOW()",
                        "current_version = current_version + 1",
                    ])
                    
                    update_query = f"""
                        UPDATE contexts 
                        SET {', '.join(set_clauses)}
                        WHERE context_id = $1
                        RETURNING *
                    """
                    
                    updated_row = await conn.fetchrow(update_query, *params)
                    
                    # Create version if requested
                    if create_version:
                        updated_context = self._row_to_context(updated_row)
                        data_bytes, compression_stats = self._serialize_and_compress(updated_context)
                        await self._create_version(conn, updated_context, data_bytes, compression_stats)
                    
                    # Log update event
                    await self._log_event(
                        conn, context_id, "update",
                        user_id=user_id,
                        changes=updates,
                        previous_state=dict(current),
                        new_state=dict(updated_row),
                    )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("update_context", execution_time)
            
            logger.info(
                "Context updated",
                context_id=context_id,
                updates=list(updates.keys()),
                user_id=user_id,
                execution_time=execution_time,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update context {context_id}: {e}")
            raise
    
    async def delete_context(self, context_id: str, user_id: Optional[str] = None, hard_delete: bool = False) -> bool:
        """Delete context (soft delete by default)."""
        start_time = time.time()
        
        try:
            async with self._get_async_connection() as conn:
                async with conn.transaction():
                    # Check permissions and current state
                    current = await conn.fetchrow("""
                        SELECT owner_id, state FROM contexts 
                        WHERE context_id = $1 AND state != 'deleted'
                    """, context_id)
                    
                    if not current:
                        return False
                    
                    if user_id and current["owner_id"] != user_id:
                        raise PermissionError(f"User {user_id} does not have permission to delete context {context_id}")
                    
                    if hard_delete:
                        # Hard delete - remove from database
                        result = await conn.execute("DELETE FROM contexts WHERE context_id = $1", context_id)
                    else:
                        # Soft delete - mark as deleted
                        result = await conn.execute("""
                            UPDATE contexts 
                            SET state = 'deleted', last_modified = NOW()
                            WHERE context_id = $1
                        """, context_id)
                    
                    # Log deletion event
                    await self._log_event(
                        conn, context_id, "delete" if hard_delete else "soft_delete",
                        user_id=user_id,
                        metadata={"hard_delete": hard_delete},
                    )
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("delete_context", execution_time)
            
            logger.info(
                "Context deleted",
                context_id=context_id,
                user_id=user_id,
                hard_delete=hard_delete,
                execution_time=execution_time,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete context {context_id}: {e}")
            raise
    
    async def query_contexts(self, query: ContextQuery, user_id: Optional[str] = None) -> List[Context]:
        """Query contexts with advanced filtering and optimization."""
        start_time = time.time()
        
        try:
            # Build query dynamically
            where_clauses = ["state != 'deleted'"]
            params = []
            param_idx = 1
            
            # Access control
            if user_id:
                where_clauses.append(f"(owner_id = ${param_idx} OR ${param_idx} = ANY(collaborators))")
                params.append(user_id)
                param_idx += 1
            
            # Context filters
            if query.context_ids:
                where_clauses.append(f"context_id = ANY(${param_idx})")
                params.append(query.context_ids)
                param_idx += 1
            
            if query.context_types:
                where_clauses.append(f"context_type = ANY(${param_idx})")
                params.append([ct.value for ct in query.context_types])
                param_idx += 1
            
            if query.owner_id:
                where_clauses.append(f"owner_id = ${param_idx}")
                params.append(query.owner_id)
                param_idx += 1
            
            if query.states:
                where_clauses.append(f"state = ANY(${param_idx})")
                params.append([s.value for s in query.states])
                param_idx += 1
            
            # Time range filters
            if query.created_after:
                where_clauses.append(f"created_at >= ${param_idx}")
                params.append(query.created_after)
                param_idx += 1
            
            if query.created_before:
                where_clauses.append(f"created_at <= ${param_idx}")
                params.append(query.created_before)
                param_idx += 1
            
            if query.modified_after:
                where_clauses.append(f"last_modified >= ${param_idx}")
                params.append(query.modified_after)
                param_idx += 1
            
            if query.modified_before:
                where_clauses.append(f"last_modified <= ${param_idx}")
                params.append(query.modified_before)
                param_idx += 1
            
            # Text search
            if query.search_text:
                where_clauses.append(f"to_tsvector('english', title || ' ' || description) @@ plainto_tsquery('english', ${param_idx})")
                params.append(query.search_text)
                param_idx += 1
            
            # Build final query
            sql = f"""
                SELECT * FROM contexts 
                WHERE {' AND '.join(where_clauses)}
                ORDER BY {query.sort_by} {query.sort_order.upper()}
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([query.limit, query.offset])
            
            async with self._get_async_connection() as conn:
                rows = await conn.fetch(sql, *params)
                contexts = [self._row_to_context(row) for row in rows]
            
            # Track performance
            execution_time = time.time() - start_time
            self._track_query_performance("query_contexts", execution_time)
            
            logger.debug(
                "Contexts queried",
                result_count=len(contexts),
                filters=len(where_clauses),
                execution_time=execution_time,
            )
            
            return contexts
            
        except Exception as e:
            logger.error(f"Failed to query contexts: {e}")
            raise
    
    async def _create_version(
        self, 
        conn, 
        context: Context, 
        data_bytes: bytes, 
        compression_stats: Dict[str, Any],
    ) -> str:
        """Create a new version entry."""
        import hashlib
        
        # Calculate checksum
        checksum = hashlib.sha256(data_bytes).hexdigest()
        
        version_id = await conn.fetchval("""
            INSERT INTO context_versions (
                context_id, version_number, data_compressed, 
                checksum, compressed, compression_type, size_bytes
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING version_id
        """,
            context.context_id,
            context.current_version,
            data_bytes,
            checksum,
            compression_stats["compressed"],
            compression_stats["compression_type"],
            compression_stats["original_size"],
        )
        
        return str(version_id)
    
    async def _log_event(
        self, 
        conn, 
        context_id: str, 
        event_type: str,
        user_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a context event."""
        await conn.execute("""
            INSERT INTO context_events (
                context_id, event_type, user_id, changes, 
                previous_state, new_state, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            context_id, event_type, user_id,
            json.dumps(changes) if changes else None,
            json.dumps(previous_state) if previous_state else None,
            json.dumps(new_state) if new_state else None,
            json.dumps(metadata) if metadata else None,
        )
    
    def _row_to_context(self, row) -> Context:
        """Convert database row to Context object."""
        from .models import ConversationContext, SessionContext, CollaborativeContext
        
        # Determine context type and create appropriate instance
        context_type_map = {
            "conversation": ConversationContext,
            "session": SessionContext,
            "collaborative": CollaborativeContext,
            # Add other types as needed
        }
        
        context_class = context_type_map.get(row["context_type"], Context)
        
        return context_class(
            context_id=str(row["context_id"]),
            context_type=ContextType(row["context_type"]),
            title=row["title"],
            description=row["description"],
            data=row["data"] or {},
            metadata=row["metadata"] or {},
            current_version=row["current_version"],
            version_history=[str(v) for v in (row["version_history"] or [])],
            owner_id=row["owner_id"],
            collaborators=row["collaborators"] or [],
            permissions=row["permissions"] or {},
            state=ContextState(row["state"]),
            last_modified=row["last_modified"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            sync_status=row["sync_status"],
            last_sync=row["last_sync"],
            compression_type=CompressionType(row["compression_type"]),
            size_bytes=row["size_bytes"],
            compressed_size_bytes=row["compressed_size_bytes"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            auto_archive_days=row["auto_archive_days"],
        )
    
    def _track_query_performance(self, query_type: str, execution_time: float) -> None:
        """Track query performance metrics."""
        if query_type not in self._query_stats:
            self._query_stats[query_type] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
            }
        
        stats = self._query_stats[query_type]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], execution_time)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "query_stats": self._query_stats,
            "cache_stats": self._cache_stats,
            "pool_stats": {
                "sync_pool_size": self._sync_pool.maxconn if self._sync_pool else 0,
                "async_pool_size": self._async_pool.get_size() if self._async_pool else 0,
            },
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._async_pool:
            await self._async_pool.close()
        
        if self._sync_pool:
            self._sync_pool.closeall()
        
        logger.info("Context persistence layer cleaned up")