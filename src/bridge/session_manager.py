"""Session management for the MCP Bridge Service."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from structlog import get_logger

from .mcp_process_manager import MCPProcessManager
from .models import (
    BridgeConfig,
    MCPSession,
    PendingRequest,
    SessionState,
    TransportType,
)

logger = get_logger(__name__)


class BridgeSessionManager:
    """Manages client sessions and their associated MCP processes."""
    
    def __init__(self, config: BridgeConfig, process_manager: MCPProcessManager):
        self.config = config
        self.process_manager = process_manager
        self.sessions: Dict[str, MCPSession] = {}
        self.client_sessions: Dict[str, Set[str]] = {}  # client_id -> session_ids
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._started = False
    
    async def start(self) -> None:
        """Start the session manager."""
        if self._started:
            return
        
        self._started = True
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Session manager started")
    
    async def stop(self) -> None:
        """Stop the session manager and cleanup all sessions."""
        self._started = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup all sessions
        async with self._lock:
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                await self._cleanup_session(session_id)
            
            self.sessions.clear()
            self.client_sessions.clear()
        
        logger.info("Session manager stopped")
    
    async def create_session(
        self,
        client_id: Optional[str] = None,
        transport: TransportType = TransportType.WEBSOCKET,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MCPSession:
        """Create a new session for a client."""
        async with self._lock:
            # Generate client ID if not provided
            if not client_id:
                client_id = str(uuid4())
            
            # Check session limit per client
            client_session_ids = self.client_sessions.get(client_id, set())
            if len(client_session_ids) >= self.config.max_sessions_per_client:
                # Try to cleanup inactive sessions
                await self._cleanup_client_sessions(client_id)
                
                client_session_ids = self.client_sessions.get(client_id, set())
                if len(client_session_ids) >= self.config.max_sessions_per_client:
                    raise RuntimeError(
                        f"Maximum sessions ({self.config.max_sessions_per_client}) "
                        f"reached for client {client_id}"
                    )
            
            # Create new session
            session = MCPSession(
                client_id=client_id,
                transport=transport,
                metadata=metadata or {},
            )
            
            try:
                # Create associated MCP process
                process = await self.process_manager.create_process(
                    session.session_id,
                    env={"CLIENT_ID": client_id, "SESSION_ID": session.session_id},
                )
                
                session.process_id = process.process.pid if process.process else None
                session.capabilities = process.capabilities
                session.state = SessionState.READY
                
            except Exception as e:
                logger.error(
                    "Failed to create MCP process for session",
                    session_id=session.session_id,
                    error=str(e),
                )
                session.state = SessionState.ERROR
                raise
            
            # Store session
            self.sessions[session.session_id] = session
            
            # Track client sessions
            if client_id not in self.client_sessions:
                self.client_sessions[client_id] = set()
            self.client_sessions[client_id].add(session.session_id)
            
            logger.info(
                "Created session",
                session_id=session.session_id,
                client_id=client_id,
                transport=transport.value,
            )
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[MCPSession]:
        """Get an existing session by ID."""
        session = self.sessions.get(session_id)
        
        if session:
            # Update activity timestamp
            session.update_activity()
        
        return session
    
    async def get_client_sessions(self, client_id: str) -> List[MCPSession]:
        """Get all sessions for a client."""
        session_ids = self.client_sessions.get(client_id, set())
        sessions = []
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session:
                sessions.append(session)
        
        return sessions
    
    async def remove_session(self, session_id: str) -> None:
        """Remove and cleanup a session."""
        async with self._lock:
            await self._cleanup_session(session_id)
    
    async def send_request(
        self,
        session_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a request to a session's MCP process."""
        # Get session
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.state != SessionState.READY:
            raise RuntimeError(f"Session {session_id} not ready (state: {session.state})")
        
        # Get associated process
        process = await self.process_manager.get_process(session_id)
        if not process:
            raise RuntimeError(f"No MCP process for session {session_id}")
        
        # Update session state
        session.state = SessionState.BUSY
        session.update_activity()
        
        try:
            # Send request to MCP process
            result = await process.send_request(method, params, timeout)
            
            # Update session state
            session.state = SessionState.READY
            session.update_activity()
            
            return result
            
        except Exception as e:
            # Update session state on error
            session.state = SessionState.ERROR
            raise
    
    async def send_notification(
        self,
        session_id: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send a notification to a session's MCP process."""
        # Get session
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.state not in (SessionState.READY, SessionState.BUSY):
            raise RuntimeError(f"Session {session_id} not ready (state: {session.state})")
        
        # Get associated process
        process = await self.process_manager.get_process(session_id)
        if not process:
            raise RuntimeError(f"No MCP process for session {session_id}")
        
        # Send notification to MCP process
        await process.send_notification(method, params)
        
        # Update activity
        session.update_activity()
    
    async def update_session_state(
        self,
        session_id: str,
        state: SessionState,
    ) -> None:
        """Update the state of a session."""
        session = self.sessions.get(session_id)
        if session:
            session.state = state
            session.update_activity()
            
            logger.debug(
                "Updated session state",
                session_id=session_id,
                state=state.value,
            )
    
    async def _cleanup_session(self, session_id: str) -> None:
        """Cleanup a session and its resources."""
        session = self.sessions.pop(session_id, None)
        
        if session:
            # Remove from client sessions
            if session.client_id:
                client_sessions = self.client_sessions.get(session.client_id, set())
                client_sessions.discard(session_id)
                
                if not client_sessions:
                    self.client_sessions.pop(session.client_id, None)
            
            # Update state
            session.state = SessionState.TERMINATED
            
            # Remove associated MCP process
            await self.process_manager.remove_process(session_id)
            
            logger.info(
                "Cleaned up session",
                session_id=session_id,
                client_id=session.client_id,
            )
    
    async def _cleanup_client_sessions(self, client_id: str) -> None:
        """Cleanup inactive sessions for a client."""
        session_ids = self.client_sessions.get(client_id, set()).copy()
        
        for session_id in session_ids:
            session = self.sessions.get(session_id)
            if session and not session.is_active(self.config.session_timeout):
                await self._cleanup_session(session_id)
    
    async def _cleanup_loop(self) -> None:
        """Periodically cleanup inactive sessions."""
        try:
            while self._started:
                await asyncio.sleep(self.config.session_cleanup_interval)
                
                async with self._lock:
                    # Find inactive sessions
                    to_remove = []
                    
                    for session_id, session in self.sessions.items():
                        if not session.is_active(self.config.session_timeout):
                            to_remove.append(session_id)
                    
                    # Remove inactive sessions
                    for session_id in to_remove:
                        await self._cleanup_session(session_id)
                    
                    if to_remove:
                        logger.info(
                            "Cleaned up inactive sessions",
                            count=len(to_remove),
                            remaining=len(self.sessions),
                        )
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in cleanup loop", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        active_sessions = 0
        error_sessions = 0
        
        for session in self.sessions.values():
            if session.state == SessionState.ERROR:
                error_sessions += 1
            elif session.is_active():
                active_sessions += 1
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "error_sessions": error_sessions,
            "total_clients": len(self.client_sessions),
            "sessions_by_transport": self._count_sessions_by_transport(),
            "sessions_by_state": self._count_sessions_by_state(),
        }
    
    def _count_sessions_by_transport(self) -> Dict[str, int]:
        """Count sessions by transport type."""
        counts = {}
        for session in self.sessions.values():
            transport = session.transport.value if session.transport else "unknown"
            counts[transport] = counts.get(transport, 0) + 1
        return counts
    
    def _count_sessions_by_state(self) -> Dict[str, int]:
        """Count sessions by state."""
        counts = {}
        for session in self.sessions.values():
            state = session.state.value
            counts[state] = counts.get(state, 0) + 1
        return counts