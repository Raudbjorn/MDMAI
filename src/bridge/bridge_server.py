"""Main MCP Bridge Server with WebSocket and SSE support."""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from structlog import get_logger

from .mcp_process_manager import MCPProcessManager
from .models import (
    BridgeConfig,
    BridgeMessage,
    BridgeStats,
    ClientMessage,
    MCPErrorCode,
    SessionState,
    TransportType,
)
from .protocol_translator import MCPProtocolTranslator
from .session_manager import BridgeSessionManager

logger = get_logger(__name__)


class ToolCallRequest(BaseModel):
    """Request to call an MCP tool."""
    
    session_id: Optional[str] = None
    tool: str
    params: Optional[Dict[str, Any]] = None


class ToolDiscoveryRequest(BaseModel):
    """Request to discover available tools."""
    
    session_id: Optional[str] = None


class SessionCreateRequest(BaseModel):
    """Request to create a new session."""
    
    client_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BridgeServer:
    """Main MCP Bridge Server application."""
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self.process_manager = MCPProcessManager(self.config)
        self.session_manager = BridgeSessionManager(self.config, self.process_manager)
        self.protocol_translator = MCPProtocolTranslator()
        self.security = HTTPBearer(auto_error=False) if self.config.require_auth else None
        self.rate_limits: Dict[str, List[datetime]] = {}  # Simple rate limiting
        self.app = self._create_app()
        self._started = False
        self._start_time = datetime.now()
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Manage application lifecycle."""
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()
        
        app = FastAPI(
            title="MCP Bridge Server",
            description="Bridge service for MCP stdio servers",
            version="0.1.0",
            lifespan=lifespan,
        )
        
        # Add CORS middleware
        from src.bridge.config import settings as bridge_settings
        app.add_middleware(
            CORSMiddleware,
            allow_origins=bridge_settings.cors_origins,
            allow_credentials=bridge_settings.cors_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files if enabled
        if bridge_settings.enable_static_files:
            from pathlib import Path
            static_path = Path(bridge_settings.static_dir)
            if static_path.exists():
                app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    async def verify_auth(self, credentials: Optional[HTTPAuthorizationCredentials] = None) -> bool:
        """Verify API key authentication."""
        if not self.config.require_auth:
            return True
        
        if not credentials or not credentials.credentials:
            return False
        
        return credentials.credentials in self.config.api_keys
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        from src.bridge.config import settings as bridge_settings
        
        if not bridge_settings.enable_rate_limiting:
            return True
        
        now = datetime.now()
        
        # Clean old entries
        if client_id in self.rate_limits:
            self.rate_limits[client_id] = [
                t for t in self.rate_limits[client_id]
                if (now - t).total_seconds() < bridge_settings.rate_limit_period
            ]
        
        # Check limit
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        if len(self.rate_limits[client_id]) >= bridge_settings.rate_limit_requests:
            return False
        
        # Add current request
        self.rate_limits[client_id].append(now)
        return True
    
    def _register_routes(self, app: FastAPI) -> None:
        """Register API routes."""
        
        # Health check
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "uptime": (datetime.now() - self._start_time).total_seconds(),
            }
        
        # Session management
        @app.post("/sessions")
        async def create_session(request: SessionCreateRequest):
            """Create a new MCP session."""
            try:
                session = await self.session_manager.create_session(
                    client_id=request.client_id,
                    transport=TransportType.HTTP,
                    metadata=request.metadata,
                )
                
                return {
                    "session_id": session.session_id,
                    "client_id": session.client_id,
                    "state": session.state.value,
                    "capabilities": session.capabilities,
                }
                
            except Exception as e:
                logger.error("Failed to create session", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )
        
        @app.get("/sessions/{session_id}")
        async def get_session(session_id: str):
            """Get session information."""
            session = await self.session_manager.get_session(session_id)
            
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {session_id} not found",
                )
            
            return {
                "session_id": session.session_id,
                "client_id": session.client_id,
                "state": session.state.value,
                "capabilities": session.capabilities,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
            }
        
        @app.delete("/sessions/{session_id}")
        async def delete_session(session_id: str):
            """Delete a session."""
            await self.session_manager.remove_session(session_id)
            return {"status": "deleted"}
        
        # Tool discovery
        @app.post("/tools/discover")
        async def discover_tools(request: ToolDiscoveryRequest):
            """Discover available MCP tools."""
            try:
                # Create or get session
                if not request.session_id:
                    session = await self.session_manager.create_session(
                        transport=TransportType.HTTP
                    )
                    session_id = session.session_id
                else:
                    session_id = request.session_id
                
                # Send tool discovery request
                result = await self.session_manager.send_request(
                    session_id,
                    "tools/list",
                    {},
                )
                
                # Translate tool format if needed
                tools = result.get("tools", [])
                
                return {
                    "session_id": session_id,
                    "tools": tools,
                }
                
            except Exception as e:
                logger.error("Failed to discover tools", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )
        
        # Tool invocation
        @app.post("/tools/call")
        async def call_tool(request: ToolCallRequest):
            """Call an MCP tool."""
            try:
                # Create or get session
                if not request.session_id:
                    session = await self.session_manager.create_session(
                        transport=TransportType.HTTP
                    )
                    session_id = session.session_id
                else:
                    session_id = request.session_id
                
                # Send tool request
                result = await self.session_manager.send_request(
                    session_id,
                    f"tools/{request.tool}",
                    request.params,
                )
                
                return {
                    "session_id": session_id,
                    "tool": request.tool,
                    "result": result,
                }
                
            except Exception as e:
                logger.error("Failed to call tool", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e),
                )
        
        # WebSocket endpoint
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication."""
            await self._handle_websocket(websocket)
        
        # SSE endpoint
        @app.get("/events/{session_id}")
        async def sse_endpoint(session_id: str, request: Request):
            """Server-Sent Events endpoint for streaming."""
            return StreamingResponse(
                self._handle_sse(session_id, request),
                media_type="text/event-stream",
            )
        
        # Statistics
        @app.get("/stats")
        async def get_stats():
            """Get bridge statistics."""
            session_stats = self.session_manager.get_stats()
            process_stats = self.process_manager.get_stats()
            
            return BridgeStats(
                active_sessions=session_stats["active_sessions"],
                active_processes=len(process_stats),
                uptime_seconds=(datetime.now() - self._start_time).total_seconds(),
                process_stats=process_stats,
            ).dict()
    
    async def _handle_websocket(self, websocket: WebSocket) -> None:
        """Handle WebSocket connections."""
        await websocket.accept()
        
        session = None
        session_id = None
        
        try:
            # Wait for initial session creation or identification
            data = await websocket.receive_json()
            
            if data.get("type") == "create_session":
                # Create new session
                session = await self.session_manager.create_session(
                    client_id=data.get("client_id"),
                    transport=TransportType.WEBSOCKET,
                    metadata=data.get("metadata", {}),
                )
                session_id = session.session_id
                
                # Send session info
                await websocket.send_json({
                    "type": "session_created",
                    "session_id": session_id,
                    "capabilities": session.capabilities,
                })
                
            elif data.get("type") == "attach_session":
                # Attach to existing session
                session_id = data.get("session_id")
                session = await self.session_manager.get_session(session_id)
                
                if not session:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Session {session_id} not found",
                    })
                    await websocket.close()
                    return
                
                # Send session info
                await websocket.send_json({
                    "type": "session_attached",
                    "session_id": session_id,
                    "capabilities": session.capabilities,
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": "Must create or attach to a session first",
                })
                await websocket.close()
                return
            
            # Main message loop
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Parse client message
                try:
                    message = self.protocol_translator.parse_client_message(data)
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                    })
                    continue
                
                # Handle request
                if hasattr(message, "method"):
                    try:
                        # Send request to MCP process
                        result = await self.session_manager.send_request(
                            session_id,
                            message.method,
                            message.params,
                        )
                        
                        # Send response to client
                        response = self.protocol_translator.format_response(
                            {
                                "id": message.id,
                                "result": result,
                            },
                            format_type=data.get("format", "json-rpc"),
                        )
                        
                        await websocket.send_json({
                            "type": "response",
                            "data": response,
                        })
                        
                    except Exception as e:
                        # Send error response
                        error_response = self.protocol_translator.create_error_response(
                            message.id,
                            MCPErrorCode.INTERNAL_ERROR,
                            str(e),
                        )
                        
                        await websocket.send_json({
                            "type": "error",
                            "data": error_response.dict(),
                        })
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", session_id=session_id)
        except Exception as e:
            logger.error("WebSocket error", session_id=session_id, error=str(e))
        finally:
            # Update session state
            if session_id:
                await self.session_manager.update_session_state(
                    session_id,
                    SessionState.DISCONNECTED,
                )
    
    async def _handle_sse(
        self,
        session_id: str,
        request: Request,
    ) -> AsyncGenerator[str, None]:
        """Handle Server-Sent Events streaming."""
        # Get or create session
        session = await self.session_manager.get_session(session_id)
        
        if not session:
            # Create new session
            session = await self.session_manager.create_session(
                transport=TransportType.SSE
            )
            session_id = session.session_id
        
        # Send initial connection event
        yield f"event: connected\ndata: {json.dumps({'session_id': session_id, 'capabilities': session.capabilities})}\n\n"
        
        try:
            # Keep connection alive with heartbeat
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                # Send heartbeat
                yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"
                
                # Wait before next heartbeat
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            pass
        finally:
            # Update session state
            await self.session_manager.update_session_state(
                session_id,
                SessionState.DISCONNECTED,
            )
    
    async def startup(self) -> None:
        """Start the bridge server components."""
        if self._started:
            return
        
        logger.info("Starting MCP Bridge Server")
        
        # Start process manager
        await self.process_manager.start()
        
        # Start session manager
        await self.session_manager.start()
        
        self._started = True
        
        logger.info("MCP Bridge Server started")
    
    async def shutdown(self) -> None:
        """Shutdown the bridge server components."""
        if not self._started:
            return
        
        logger.info("Shutting down MCP Bridge Server")
        
        # Stop session manager
        await self.session_manager.stop()
        
        # Stop process manager
        await self.process_manager.stop()
        
        self._started = False
        
        logger.info("MCP Bridge Server stopped")


def create_bridge_app(config: Optional[BridgeConfig] = None) -> FastAPI:
    """Create a configured bridge server application."""
    server = BridgeServer(config)
    return server.app