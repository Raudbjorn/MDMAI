"""Connection pooling for database operations."""

import asyncio
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionPoolManager:
    """Manages connection pooling for ChromaDB."""
    
    def __init__(self, max_connections: int = 10):
        """Initialize connection pool."""
        self.max_connections = max_connections
        self.connections = []
        self.available = asyncio.Queue(maxsize=max_connections)
        self.in_use = set()
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self, db_factory) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            # Create initial connections
            for _ in range(self.max_connections):
                conn = await self._create_connection(db_factory)
                self.connections.append(conn)
                await self.available.put(conn)
            
            self._initialized = True
            logger.info(f"Connection pool initialized with {self.max_connections} connections")
    
    async def _create_connection(self, db_factory):
        """Create a new database connection."""
        # This would create a new ChromaDB client instance
        # For now, return a placeholder
        return db_factory()
    
    async def acquire(self):
        """Acquire a connection from the pool."""
        conn = await self.available.get()
        self.in_use.add(conn)
        return conn
    
    async def release(self, conn) -> None:
        """Release a connection back to the pool."""
        if conn in self.in_use:
            self.in_use.remove(conn)
            await self.available.put(conn)
    
    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            # Close all available connections
            while not self.available.empty():
                try:
                    conn = self.available.get_nowait()
                    # Close connection (implementation depends on ChromaDB)
                    # If the connection has a close method, use it
                    if hasattr(conn, 'close'):
                        await conn.close()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            # Wait for in-use connections with timeout
            timeout = 5.0
            start_time = asyncio.get_event_loop().time()
            while self.in_use and (asyncio.get_event_loop().time() - start_time < timeout):
                await asyncio.sleep(0.1)
            
            if self.in_use:
                logger.warning(f"Forcefully closing {len(self.in_use)} in-use connections")
                self.in_use.clear()
            
            self.connections.clear()
            self._initialized = False
            logger.info("All connections closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total": self.max_connections,
            "available": self.available.qsize(),
            "in_use": len(self.in_use),
        }