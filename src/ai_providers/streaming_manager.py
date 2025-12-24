"""Streaming response manager for AI providers."""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from structlog import get_logger

from .models import AIResponse, ProviderType, StreamingChunk

logger = get_logger(__name__)


class StreamingSession(BaseModel):
    """Represents a streaming session."""
    
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    provider_type: ProviderType
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    chunks_sent: int = 0
    total_content: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamingResponse:
    """Wrapper for streaming responses with buffering and aggregation."""
    
    def __init__(
        self,
        request_id: str,
        provider_type: ProviderType,
        chunk_generator: AsyncGenerator[StreamingChunk, None],
    ):
        self.request_id = request_id
        self.provider_type = provider_type
        self.chunk_generator = chunk_generator
        self.chunks: List[StreamingChunk] = []
        self.aggregated_content = ""
        self.aggregated_tool_calls: List[Dict[str, Any]] = []
        self.is_complete = False
        self.finish_reason: Optional[str] = None
        self._buffer: List[StreamingChunk] = []
        self._buffer_size = 5  # Buffer chunks before yielding
    
    async def stream(self) -> AsyncGenerator[StreamingChunk, None]:
        """Stream chunks with optional buffering."""
        try:
            async for chunk in self.chunk_generator:
                self.chunks.append(chunk)
                
                # Aggregate content
                if chunk.content:
                    self.aggregated_content += chunk.content
                
                # Aggregate tool calls
                if chunk.tool_calls:
                    self.aggregated_tool_calls.extend(chunk.tool_calls)
                
                # Update completion status
                if chunk.is_complete:
                    self.is_complete = True
                    self.finish_reason = chunk.finish_reason
                
                # Add to buffer
                self._buffer.append(chunk)
                
                # Yield buffered chunks if buffer is full or stream is complete
                if len(self._buffer) >= self._buffer_size or chunk.is_complete:
                    for buffered_chunk in self._buffer:
                        yield buffered_chunk
                    self._buffer.clear()
                
        except Exception as e:
            logger.error(
                "Error in streaming response",
                request_id=self.request_id,
                error=str(e),
            )
            # Send error chunk
            error_chunk = StreamingChunk(
                request_id=self.request_id,
                content=f"\n[Error: {str(e)}]",
                is_complete=True,
                metadata={"error": str(e)},
            )
            yield error_chunk
    
    async def stream_unbuffered(self) -> AsyncGenerator[StreamingChunk, None]:
        """Stream chunks without buffering."""
        try:
            async for chunk in self.chunk_generator:
                self.chunks.append(chunk)
                
                # Aggregate content
                if chunk.content:
                    self.aggregated_content += chunk.content
                
                # Aggregate tool calls
                if chunk.tool_calls:
                    self.aggregated_tool_calls.extend(chunk.tool_calls)
                
                # Update completion status
                if chunk.is_complete:
                    self.is_complete = True
                    self.finish_reason = chunk.finish_reason
                
                yield chunk
                
        except Exception as e:
            logger.error(
                "Error in unbuffered streaming",
                request_id=self.request_id,
                error=str(e),
            )
            # Send error chunk
            error_chunk = StreamingChunk(
                request_id=self.request_id,
                content=f"\n[Error: {str(e)}]",
                is_complete=True,
                metadata={"error": str(e)},
            )
            yield error_chunk
    
    def to_response(self) -> AIResponse:
        """Convert aggregated streaming data to AIResponse."""
        return AIResponse(
            request_id=self.request_id,
            provider_type=self.provider_type,
            model="",  # Should be set by caller
            content=self.aggregated_content,
            tool_calls=self.aggregated_tool_calls if self.aggregated_tool_calls else None,
            finish_reason=self.finish_reason,
            cached=False,
            metadata={
                "streamed": True,
                "chunks_count": len(self.chunks),
            },
        )


class StreamingManager:
    """Manages streaming responses across providers."""
    
    def __init__(self):
        self._sessions: Dict[str, StreamingSession] = {}
        self._session_lock = asyncio.Lock()
        self._max_sessions = 100
        self._session_timeout = 3600  # 1 hour
    
    async def create_session(
        self,
        request_id: str,
        provider_type: ProviderType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamingSession:
        """Create a new streaming session.
        
        Args:
            request_id: Request ID
            provider_type: Provider type handling the stream
            metadata: Optional session metadata
            
        Returns:
            Created streaming session
        """
        async with self._session_lock:
            # Clean up old sessions if at capacity
            if len(self._sessions) >= self._max_sessions:
                await self._cleanup_old_sessions()
            
            session = StreamingSession(
                request_id=request_id,
                provider_type=provider_type,
                metadata=metadata or {},
            )
            
            self._sessions[session.session_id] = session
            
            logger.info(
                "Created streaming session",
                session_id=session.session_id,
                request_id=request_id,
                provider=provider_type.value,
            )
            
            return session
    
    async def update_session(
        self,
        session_id: str,
        chunk: StreamingChunk,
    ) -> None:
        """Update session with new chunk data.
        
        Args:
            session_id: Session ID to update
            chunk: Streaming chunk received
        """
        async with self._session_lock:
            if session_id not in self._sessions:
                logger.warning("Session not found", session_id=session_id)
                return
            
            session = self._sessions[session_id]
            session.chunks_sent += 1
            
            if chunk.content:
                session.total_content += chunk.content
            
            if chunk.tool_calls:
                session.tool_calls.extend(chunk.tool_calls)
            
            if chunk.is_complete:
                session.is_active = False
                session.ended_at = datetime.now()
    
    async def end_session(self, session_id: str) -> Optional[StreamingSession]:
        """End a streaming session.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            Ended session or None if not found
        """
        async with self._session_lock:
            if session_id not in self._sessions:
                logger.warning("Session not found", session_id=session_id)
                return None
            
            session = self._sessions[session_id]
            session.is_active = False
            session.ended_at = datetime.now()
            
            logger.info(
                "Ended streaming session",
                session_id=session_id,
                chunks_sent=session.chunks_sent,
                duration=(session.ended_at - session.started_at).total_seconds(),
            )
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get a streaming session.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            Session or None if not found
        """
        return self._sessions.get(session_id)
    
    async def create_streaming_response(
        self,
        request_id: str,
        provider_type: ProviderType,
        chunk_generator: AsyncGenerator[StreamingChunk, None],
        create_session: bool = True,
    ) -> StreamingResponse:
        """Create a managed streaming response.
        
        Args:
            request_id: Request ID
            provider_type: Provider type
            chunk_generator: Async generator of chunks
            create_session: Whether to create a session for tracking
            
        Returns:
            StreamingResponse wrapper
        """
        # Create session if requested
        session = None
        if create_session:
            session = await self.create_session(request_id, provider_type)
        
        # Create response wrapper
        response = StreamingResponse(request_id, provider_type, chunk_generator)
        
        # If we have a session, wrap the generator to track chunks
        if session:
            original_generator = response.chunk_generator
            
            async def tracked_generator():
                try:
                    async for chunk in original_generator:
                        await self.update_session(session.session_id, chunk)
                        yield chunk
                finally:
                    await self.end_session(session.session_id)
            
            response.chunk_generator = tracked_generator()
        
        return response
    
    async def stream_to_sse(
        self,
        streaming_response: StreamingResponse,
        include_metadata: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Convert streaming response to Server-Sent Events format.
        
        Args:
            streaming_response: Streaming response to convert
            include_metadata: Whether to include metadata in events
            
        Yields:
            SSE formatted strings
        """
        try:
            async for chunk in streaming_response.stream():
                # Create SSE event data
                event_data = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                }
                
                if chunk.tool_calls:
                    event_data["tool_calls"] = chunk.tool_calls
                
                if chunk.is_complete:
                    event_data["finish_reason"] = chunk.finish_reason
                    event_data["is_complete"] = True
                
                if include_metadata and chunk.metadata:
                    event_data["metadata"] = chunk.metadata
                
                # Format as SSE
                event_json = json.dumps(event_data)
                yield f"data: {event_json}\n\n"
            
            # Send completion event
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(
                "Error converting to SSE",
                request_id=streaming_response.request_id,
                error=str(e),
            )
            # Send error event
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    async def stream_to_json_lines(
        self,
        streaming_response: StreamingResponse,
    ) -> AsyncGenerator[str, None]:
        """Convert streaming response to JSON Lines format.
        
        Args:
            streaming_response: Streaming response to convert
            
        Yields:
            JSON Lines formatted strings
        """
        try:
            async for chunk in streaming_response.stream():
                chunk_dict = chunk.dict(exclude_none=True)
                yield json.dumps(chunk_dict) + "\n"
            
        except Exception as e:
            logger.error(
                "Error converting to JSON Lines",
                request_id=streaming_response.request_id,
                error=str(e),
            )
            # Send error line
            error_data = {"error": str(e), "is_complete": True}
            yield json.dumps(error_data) + "\n"
    
    async def aggregate_stream(
        self,
        streaming_response: StreamingResponse,
    ) -> AIResponse:
        """Aggregate a streaming response into a single AIResponse.
        
        Args:
            streaming_response: Streaming response to aggregate
            
        Returns:
            Aggregated AIResponse
        """
        # Consume the entire stream
        async for _ in streaming_response.stream():
            pass  # Chunks are automatically aggregated in StreamingResponse
        
        # Convert to response
        return streaming_response.to_response()
    
    async def _cleanup_old_sessions(self) -> None:
        """Clean up old or inactive sessions."""
        now = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self._sessions.items():
            # Check if session is old
            age = (now - session.started_at).total_seconds()
            if age > self._session_timeout:
                sessions_to_remove.append(session_id)
            # Check if session is inactive and old enough
            elif not session.is_active and session.ended_at:
                inactive_time = (now - session.ended_at).total_seconds()
                if inactive_time > 300:  # 5 minutes after ending
                    sessions_to_remove.append(session_id)
        
        # Remove old sessions
        for session_id in sessions_to_remove:
            del self._sessions[session_id]
        
        if sessions_to_remove:
            logger.info(
                "Cleaned up old streaming sessions",
                count=len(sessions_to_remove),
            )
    
    def get_active_sessions(self) -> List[StreamingSession]:
        """Get list of active streaming sessions.
        
        Returns:
            List of active sessions
        """
        return [
            session for session in self._sessions.values()
            if session.is_active
        ]
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get streaming session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        active_sessions = self.get_active_sessions()
        total_chunks = sum(s.chunks_sent for s in self._sessions.values())
        
        provider_stats = {}
        for session in self._sessions.values():
            provider = session.provider_type.value
            if provider not in provider_stats:
                provider_stats[provider] = {
                    "sessions": 0,
                    "active": 0,
                    "chunks": 0,
                }
            provider_stats[provider]["sessions"] += 1
            if session.is_active:
                provider_stats[provider]["active"] += 1
            provider_stats[provider]["chunks"] += session.chunks_sent
        
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(active_sessions),
            "total_chunks_sent": total_chunks,
            "providers": provider_stats,
            "max_sessions": self._max_sessions,
            "session_timeout": self._session_timeout,
        }