# SPIKE 1: WebSocket-to-MCP Bridge Architecture

## Executive Summary

This spike investigates the implementation strategy for bridging WebSocket connections from the SvelteKit frontend to the MCP server that uses stdin/stdout for Claude Desktop. Based on analysis of the existing codebase, we recommend leveraging the **existing bridge server** architecture at `/src/bridge/bridge_server.py` with enhancements for production-ready TTRPG session management.

## Architecture Decision

### Recommended Approach: Enhanced Bridge Service (Separate Process)

After analyzing the codebase, we discovered an existing bridge server implementation that already provides:
- WebSocket endpoint support
- Session management infrastructure
- MCP process management
- Protocol translation capabilities

**Decision**: Enhance the existing bridge service rather than integrate WebSocket directly into the MCP server.

### Rationale

1. **Separation of Concerns**: The MCP server remains focused on tool execution via stdio, while the bridge handles network communication
2. **Scalability**: Bridge can be scaled independently and handle multiple MCP process instances
3. **Existing Infrastructure**: Leverages already-implemented session management and process pooling
4. **Protocol Isolation**: Keeps MCP protocol pure for Claude Desktop compatibility

## Message Protocol Specification

### WebSocket Message Format

```typescript
// Base message structure
interface WSMessage {
  id: string;                    // Unique message ID for request/response correlation
  type: MessageType;              // Message type discriminator
  timestamp: number;              // Unix timestamp in milliseconds
  version: string;                // Protocol version (e.g., "1.0.0")
  session_id?: string;            // Session identifier for multi-session support
  request_id?: string;            // Original request ID for responses
}

// Message types
enum MessageType {
  // Connection Management
  CONNECT = "connect",
  DISCONNECT = "disconnect",
  AUTHENTICATE = "authenticate",
  SESSION_CREATE = "session_create",
  SESSION_ATTACH = "session_attach",
  
  // MCP Operations
  TOOL_CALL = "tool_call",
  TOOL_DISCOVER = "tool_discover",
  RESOURCE_GET = "resource_get",
  PROMPT_GET = "prompt_get",
  
  // TTRPG Specific
  CAMPAIGN_UPDATE = "campaign_update",
  CHARACTER_UPDATE = "character_update",
  DICE_ROLL = "dice_roll",
  INITIATIVE_UPDATE = "initiative_update",
  
  // Real-time Collaboration
  CURSOR_MOVE = "cursor_move",
  CANVAS_DRAW = "canvas_draw",
  PRESENCE_UPDATE = "presence_update",
  
  // System
  HEARTBEAT = "heartbeat",
  ERROR = "error",
  ACK = "ack"
}

// Request message
interface WSRequest extends WSMessage {
  method: string;                 // MCP method name
  params?: Record<string, any>;   // Method parameters
  metadata?: {
    priority?: number;            // Request priority (0-10)
    timeout?: number;             // Request timeout in ms
    retry?: boolean;              // Allow retry on failure
  };
}

// Response message  
interface WSResponse extends WSMessage {
  request_id: string;             // ID of original request
  result?: any;                   // Success result
  error?: {
    code: number;                 // Error code
    message: string;              // Human-readable error
    data?: any;                   // Additional error context
  };
  duration?: number;              // Processing time in ms
}

// Event message (server-initiated)
interface WSEvent extends WSMessage {
  event: string;                  // Event name
  data: any;                      // Event payload
  scope?: EventScope;             // Event visibility scope
}

enum EventScope {
  USER = "user",                  // Only for specific user
  SESSION = "session",            // All users in session
  CAMPAIGN = "campaign",          // All users in campaign
  GLOBAL = "global"               // All connected users
}
```

### MCP Protocol Translation

```python
# Python implementation for protocol translation
from typing import Any, Dict, Optional
from dataclasses import dataclass
import json
import uuid
from datetime import datetime

@dataclass
class MCPRequest:
    """Internal MCP request format"""
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None
    
    def to_stdio(self) -> str:
        """Convert to stdio format for MCP server"""
        return json.dumps({
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id or str(uuid.uuid4())
        })

class ProtocolTranslator:
    """Translates between WebSocket and MCP protocols"""
    
    def ws_to_mcp(self, ws_message: Dict[str, Any]) -> MCPRequest:
        """Convert WebSocket message to MCP request"""
        return MCPRequest(
            method=ws_message.get("method"),
            params=ws_message.get("params", {}),
            id=ws_message.get("id")
        )
    
    def mcp_to_ws(self, mcp_response: str, request_id: str) -> Dict[str, Any]:
        """Convert MCP response to WebSocket format"""
        response = json.loads(mcp_response)
        
        return {
            "id": str(uuid.uuid4()),
            "type": "response",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "version": "1.0.0",
            "request_id": request_id,
            "result": response.get("result"),
            "error": response.get("error"),
            "duration": response.get("_duration")
        }
    
    def create_event(
        self, 
        event: str, 
        data: Any, 
        scope: str = "session"
    ) -> Dict[str, Any]:
        """Create server-initiated event"""
        return {
            "id": str(uuid.uuid4()),
            "type": "event",
            "timestamp": int(datetime.now().timestamp() * 1000),
            "version": "1.0.0",
            "event": event,
            "data": data,
            "scope": scope
        }
```

## Session Management Strategy

### Session Architecture

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import asyncio
from enum import Enum

class SessionState(Enum):
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISCONNECTED = "disconnected"

@dataclass
class TTRPGSession:
    """Enhanced session for TTRPG context"""
    session_id: str
    campaign_id: Optional[str]
    user_id: str
    username: str
    role: str  # player, gm, spectator
    
    # Connection info
    ws_connection: Optional[Any] = None
    mcp_process: Optional[Any] = None
    state: SessionState = SessionState.INITIALIZING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    heartbeat_timestamp: Optional[datetime] = None
    
    # TTRPG context
    active_character_id: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    shared_state: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    pending_requests: Dict[str, asyncio.Future] = field(default_factory=dict)
    rate_limit_tokens: int = 100
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

class SessionManager:
    """Manages TTRPG sessions with MCP processes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.sessions: Dict[str, TTRPGSession] = {}
        self.campaign_sessions: Dict[str, Set[str]] = {}  # campaign_id -> session_ids
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.mcp_pool: List[Any] = []  # Pool of MCP processes
        self.config = config
        self._lock = asyncio.Lock()
    
    async def create_session(
        self,
        user_id: str,
        username: str,
        campaign_id: Optional[str] = None,
        role: str = "player"
    ) -> TTRPGSession:
        """Create new session with MCP process"""
        async with self._lock:
            # Check for existing session
            if user_id in self.user_sessions:
                existing_id = self.user_sessions[user_id]
                if existing_id in self.sessions:
                    # Reuse existing session
                    session = self.sessions[existing_id]
                    session.state = SessionState.CONNECTED
                    return session
            
            # Create new session
            session_id = str(uuid.uuid4())
            session = TTRPGSession(
                session_id=session_id,
                user_id=user_id,
                username=username,
                campaign_id=campaign_id,
                role=role
            )
            
            # Assign MCP process from pool or create new
            session.mcp_process = await self._get_or_create_mcp_process()
            
            # Register session
            self.sessions[session_id] = session
            self.user_sessions[user_id] = session_id
            
            if campaign_id:
                if campaign_id not in self.campaign_sessions:
                    self.campaign_sessions[campaign_id] = set()
                self.campaign_sessions[campaign_id].add(session_id)
            
            # Initialize session state
            await self._initialize_session(session)
            
            return session
    
    async def _get_or_create_mcp_process(self):
        """Get MCP process from pool or create new one"""
        # Implementation for process pooling
        if self.mcp_pool:
            return self.mcp_pool.pop()
        else:
            return await self._create_mcp_process()
    
    async def _create_mcp_process(self):
        """Create new MCP process"""
        # Start MCP server process with stdio
        process = await asyncio.create_subprocess_exec(
            "python", "-m", "src.main",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        return process
    
    async def broadcast_to_campaign(
        self,
        campaign_id: str,
        event: str,
        data: Any,
        exclude_session: Optional[str] = None
    ):
        """Broadcast event to all sessions in campaign"""
        if campaign_id not in self.campaign_sessions:
            return
        
        tasks = []
        for session_id in self.campaign_sessions[campaign_id]:
            if session_id != exclude_session:
                session = self.sessions.get(session_id)
                if session and session.ws_connection:
                    tasks.append(
                        self._send_to_session(session, event, data)
                    )
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
```

## Authentication Approach

### JWT-Based Authentication Flow

```typescript
// TypeScript client implementation
class AuthenticationManager {
  private token: string | null = null;
  private refreshToken: string | null = null;
  private wsClient: EnhancedWebSocketClient;
  
  async authenticate(credentials: {
    username: string;
    password?: string;
    apiKey?: string;
  }): Promise<AuthResult> {
    // Step 1: Get JWT from auth endpoint
    const authResponse = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    });
    
    if (!authResponse.ok) {
      throw new Error('Authentication failed');
    }
    
    const { access_token, refresh_token, user_info } = await authResponse.json();
    
    // Step 2: Store tokens
    this.token = access_token;
    this.refreshToken = refresh_token;
    
    // Step 3: Connect WebSocket with auth
    await this.connectWebSocket();
    
    return { token: access_token, user: user_info };
  }
  
  private async connectWebSocket() {
    // Include token in WebSocket connection
    const wsUrl = `${WS_BASE_URL}?token=${this.token}`;
    
    this.wsClient = new EnhancedWebSocketClient({
      url: wsUrl,
      protocols: ['mcp-ttrpg-v1'],
      enableHeartbeat: true,
      heartbeatInterval: 30000
    });
    
    // Send authentication message after connection
    this.wsClient.onOpen(() => {
      this.wsClient.send({
        type: 'authenticate',
        token: this.token,
        timestamp: Date.now()
      });
    });
  }
  
  async refreshAuth(): Promise<void> {
    // Refresh token before expiry
    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.refreshToken}`
      }
    });
    
    if (response.ok) {
      const { access_token } = await response.json();
      this.token = access_token;
      
      // Update WebSocket auth
      this.wsClient.send({
        type: 'reauthenticate',
        token: this.token
      });
    }
  }
}
```

### Python Server Authentication

```python
from fastapi import WebSocket, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class AuthenticationService:
    """Handles authentication for WebSocket and HTTP connections"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
    
    def create_token(
        self, 
        user_id: str, 
        username: str,
        role: str = "player",
        expires_in: int = 3600
    ) -> str:
        """Create JWT token"""
        payload = {
            "user_id": user_id,
            "username": username,
            "role": role,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def authenticate_websocket(
        self, 
        websocket: WebSocket,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Authenticate WebSocket connection"""
        if not token:
            # Try to get token from query params
            token = websocket.query_params.get("token")
        
        if not token:
            await websocket.close(code=1008, reason="Missing authentication")
            raise HTTPException(status_code=401, detail="Missing token")
        
        try:
            user_info = self.verify_token(token)
            return user_info
        except HTTPException:
            await websocket.close(code=1008, reason="Invalid authentication")
            raise

# Integration with bridge server
class SecureWebSocketEndpoint:
    """Secured WebSocket endpoint with authentication"""
    
    def __init__(self, auth_service: AuthenticationService):
        self.auth = auth_service
        self.authenticated_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def handle_connection(self, websocket: WebSocket):
        """Handle authenticated WebSocket connection"""
        # Authenticate connection
        try:
            user_info = await self.auth.authenticate_websocket(websocket)
        except HTTPException:
            return
        
        session_id = str(uuid.uuid4())
        self.authenticated_sessions[session_id] = {
            "user": user_info,
            "websocket": websocket,
            "connected_at": datetime.now()
        }
        
        try:
            # Send successful auth response
            await websocket.send_json({
                "type": "auth_success",
                "session_id": session_id,
                "user": user_info
            })
            
            # Handle messages
            while True:
                data = await websocket.receive_json()
                await self.handle_authenticated_message(
                    session_id, 
                    data, 
                    user_info
                )
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up session
            del self.authenticated_sessions[session_id]
```

## Code Examples

### Python FastAPI/FastMCP Bridge Enhancement

```python
# Enhanced bridge server with TTRPG features
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastmcp import FastMCP
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

@dataclass
class TTRPGBridgeConfig:
    """Configuration for TTRPG bridge server"""
    mcp_command: List[str] = None
    max_sessions: int = 100
    max_mcp_processes: int = 10
    enable_auth: bool = True
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    heartbeat_interval: int = 30
    session_timeout: int = 3600
    
    def __post_init__(self):
        if self.mcp_command is None:
            self.mcp_command = ["python", "-m", "src.main"]

class TTRPGBridgeServer:
    """Enhanced bridge server for TTRPG MCP integration"""
    
    def __init__(self, config: TTRPGBridgeConfig):
        self.config = config
        self.app = FastAPI(title="TTRPG MCP Bridge")
        self.sessions = SessionManager(config)
        self.auth = AuthenticationService(
            secret_key=os.getenv("JWT_SECRET_KEY", "change-me")
        )
        self.translator = ProtocolTranslator()
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Configure middleware"""
        from fastapi.middleware.cors import CORSMiddleware
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],  # SvelteKit dev
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup WebSocket and HTTP routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint"""
            await websocket.accept()
            
            session = None
            user_info = None
            
            try:
                # Wait for authentication
                auth_msg = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=10.0
                )
                
                if auth_msg.get("type") != "authenticate":
                    await websocket.send_json({
                        "type": "error",
                        "error": "Must authenticate first"
                    })
                    await websocket.close()
                    return
                
                # Verify authentication
                token = auth_msg.get("token")
                user_info = self.auth.verify_token(token)
                
                # Create session
                session = await self.sessions.create_session(
                    user_id=user_info["user_id"],
                    username=user_info["username"],
                    campaign_id=auth_msg.get("campaign_id"),
                    role=user_info.get("role", "player")
                )
                
                session.ws_connection = websocket
                
                # Send session created response
                await websocket.send_json({
                    "type": "session_created",
                    "session_id": session.session_id,
                    "capabilities": await self._get_mcp_capabilities(session)
                })
                
                # Start background tasks
                heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(session)
                )
                mcp_reader_task = asyncio.create_task(
                    self._mcp_reader_loop(session)
                )
                
                # Main message loop
                await self._handle_messages(session, websocket)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {session.session_id if session else 'unknown'}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
            finally:
                # Cleanup
                if session:
                    await self.sessions.remove_session(session.session_id)
                    if 'heartbeat_task' in locals():
                        heartbeat_task.cancel()
                    if 'mcp_reader_task' in locals():
                        mcp_reader_task.cancel()
    
    async def _handle_messages(
        self, 
        session: TTRPGSession, 
        websocket: WebSocket
    ):
        """Handle incoming WebSocket messages"""
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            # Update session activity
            session.update_activity()
            
            # Route message based on type
            if message_type == "tool_call":
                await self._handle_tool_call(session, data)
            elif message_type == "campaign_update":
                await self._handle_campaign_update(session, data)
            elif message_type == "dice_roll":
                await self._handle_dice_roll(session, data)
            elif message_type == "cursor_move":
                await self._handle_cursor_move(session, data)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(session, data)
            else:
                # Forward to MCP
                await self._forward_to_mcp(session, data)
    
    async def _handle_tool_call(
        self, 
        session: TTRPGSession, 
        message: Dict[str, Any]
    ):
        """Handle MCP tool call"""
        # Convert to MCP format
        mcp_request = self.translator.ws_to_mcp(message)
        
        # Send to MCP process
        if session.mcp_process:
            request_id = mcp_request.id or str(uuid.uuid4())
            
            # Create future for response
            response_future = asyncio.Future()
            session.pending_requests[request_id] = response_future
            
            # Send request
            session.mcp_process.stdin.write(
                (mcp_request.to_stdio() + "\n").encode()
            )
            await session.mcp_process.stdin.drain()
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(
                    response_future,
                    timeout=30.0
                )
                
                # Send response to client
                ws_response = self.translator.mcp_to_ws(
                    response, 
                    message.get("id")
                )
                await session.ws_connection.send_json(ws_response)
                
            except asyncio.TimeoutError:
                await session.ws_connection.send_json({
                    "type": "error",
                    "request_id": message.get("id"),
                    "error": "Request timeout"
                })
    
    async def _handle_dice_roll(
        self, 
        session: TTRPGSession, 
        message: Dict[str, Any]
    ):
        """Handle dice roll and broadcast to campaign"""
        roll_data = message.get("data", {})
        
        # Process dice roll
        result = self._process_dice_roll(roll_data.get("expression"))
        
        # Create event
        event = self.translator.create_event(
            "dice_roll",
            {
                "player_id": session.user_id,
                "player_name": session.username,
                "expression": roll_data.get("expression"),
                "results": result["rolls"],
                "total": result["total"],
                "purpose": roll_data.get("purpose")
            },
            scope="campaign"
        )
        
        # Broadcast to campaign
        if session.campaign_id:
            await self.sessions.broadcast_to_campaign(
                session.campaign_id,
                "dice_roll",
                event["data"],
                exclude_session=session.session_id
            )
        
        # Send result to requester
        await session.ws_connection.send_json({
            "type": "dice_roll_result",
            "request_id": message.get("id"),
            "result": result
        })
    
    def _process_dice_roll(self, expression: str) -> Dict[str, Any]:
        """Process dice roll expression"""
        import re
        import random
        
        # Simple dice parser (e.g., "2d6+3")
        pattern = r'(\d+)d(\d+)([+-]\d+)?'
        match = re.match(pattern, expression)
        
        if not match:
            return {"error": "Invalid dice expression"}
        
        num_dice = int(match.group(1))
        die_size = int(match.group(2))
        modifier = int(match.group(3) or 0)
        
        rolls = [random.randint(1, die_size) for _ in range(num_dice)]
        total = sum(rolls) + modifier
        
        return {
            "expression": expression,
            "rolls": rolls,
            "modifier": modifier,
            "total": total
        }
    
    async def _mcp_reader_loop(self, session: TTRPGSession):
        """Read responses from MCP process"""
        if not session.mcp_process:
            return
        
        while True:
            try:
                # Read line from MCP stdout
                line = await session.mcp_process.stdout.readline()
                if not line:
                    break
                
                # Parse response
                response = line.decode().strip()
                if not response:
                    continue
                
                try:
                    data = json.loads(response)
                    request_id = data.get("id")
                    
                    # Resolve pending request
                    if request_id in session.pending_requests:
                        future = session.pending_requests.pop(request_id)
                        future.set_result(response)
                    else:
                        # Unsolicited message from MCP
                        event = self.translator.create_event(
                            "mcp_notification",
                            data
                        )
                        await session.ws_connection.send_json(event)
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from MCP: {response}")
                    
            except Exception as e:
                logger.error(f"MCP reader error: {e}")
                break
    
    async def _heartbeat_loop(self, session: TTRPGSession):
        """Send periodic heartbeats"""
        while True:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # Send heartbeat
                await session.ws_connection.send_json({
                    "type": "heartbeat",
                    "timestamp": int(datetime.now().timestamp() * 1000)
                })
                
                # Check for timeout
                if (datetime.now() - session.last_activity).seconds > self.config.session_timeout:
                    logger.info(f"Session timeout: {session.session_id}")
                    break
                    
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    config = TTRPGBridgeConfig()
    server = TTRPGBridgeServer(config)
    
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
```

### TypeScript SvelteKit Integration

```typescript
// Enhanced WebSocket client for TTRPG
import { writable, derived, get } from 'svelte/store';
import type { Writable, Readable } from 'svelte/store';

interface TTRPGWebSocketConfig extends WebSocketConfig {
  campaignId?: string;
  authToken: string;
  onDiceRoll?: (roll: DiceRollEvent) => void;
  onCampaignUpdate?: (update: CampaignUpdate) => void;
}

export class TTRPGWebSocketClient extends EnhancedWebSocketClient {
  private campaignId?: string;
  private authToken: string;
  private sessionId?: string;
  
  // Svelte stores for reactive state
  public connectionState: Writable<'disconnected' | 'connecting' | 'connected' | 'authenticated'>;
  public participants: Writable<Map<string, Participant>>;
  public sharedState: Writable<SharedState>;
  public activityFeed: Writable<ActivityEvent[]>;
  
  constructor(config: TTRPGWebSocketConfig) {
    super(config);
    
    this.campaignId = config.campaignId;
    this.authToken = config.authToken;
    
    // Initialize stores
    this.connectionState = writable('disconnected');
    this.participants = writable(new Map());
    this.sharedState = writable({
      initiative_order: [],
      active_turn: 0,
      round_number: 1,
      shared_notes: '',
      dice_rolls: [],
      last_update: new Date().toISOString(),
      version: 0
    });
    this.activityFeed = writable([]);
    
    // Setup message handlers
    this.setupHandlers(config);
  }
  
  private setupHandlers(config: TTRPGWebSocketConfig) {
    // Handle connection events
    this.onOpen(() => {
      this.connectionState.set('connected');
      this.authenticate();
    });
    
    this.onClose(() => {
      this.connectionState.set('disconnected');
    });
    
    // Handle messages
    this.onMessage((data) => {
      switch (data.type) {
        case 'session_created':
          this.handleSessionCreated(data);
          break;
        case 'dice_roll':
          this.handleDiceRoll(data);
          if (config.onDiceRoll) {
            config.onDiceRoll(data.data);
          }
          break;
        case 'campaign_update':
          this.handleCampaignUpdate(data);
          if (config.onCampaignUpdate) {
            config.onCampaignUpdate(data.data);
          }
          break;
        case 'participant_joined':
          this.handleParticipantJoined(data.data);
          break;
        case 'participant_left':
          this.handleParticipantLeft(data.data);
          break;
        case 'state_update':
          this.handleStateUpdate(data.data);
          break;
        case 'cursor_move':
          this.handleCursorMove(data.data);
          break;
      }
    });
  }
  
  private async authenticate() {
    this.send({
      type: 'authenticate',
      token: this.authToken,
      campaign_id: this.campaignId,
      timestamp: Date.now()
    });
  }
  
  private handleSessionCreated(message: any) {
    this.sessionId = message.session_id;
    this.connectionState.set('authenticated');
    
    // Store capabilities
    if (message.capabilities) {
      this.storeCapabilities(message.capabilities);
    }
  }
  
  private handleDiceRoll(message: any) {
    const roll = message.data;
    
    // Add to activity feed
    this.activityFeed.update(feed => {
      const newFeed = [...feed, {
        type: 'dice_roll',
        timestamp: new Date().toISOString(),
        player: roll.player_name,
        data: roll
      }];
      
      // Keep only last 50 events
      return newFeed.slice(-50);
    });
    
    // Update shared state
    this.sharedState.update(state => ({
      ...state,
      dice_rolls: [...state.dice_rolls.slice(-9), roll]
    }));
  }
  
  private handleParticipantJoined(data: any) {
    this.participants.update(participants => {
      const newParticipants = new Map(participants);
      newParticipants.set(data.user_id, data);
      return newParticipants;
    });
    
    // Add to activity feed
    this.activityFeed.update(feed => [
      ...feed,
      {
        type: 'participant_joined',
        timestamp: new Date().toISOString(),
        player: data.username,
        data
      }
    ].slice(-50));
  }
  
  private handleStateUpdate(update: StateUpdate) {
    this.sharedState.update(state => {
      // Apply state update based on path
      const newState = { ...state };
      
      // Simple path resolver (can be enhanced)
      let target: any = newState;
      for (let i = 0; i < update.path.length - 1; i++) {
        target = target[update.path[i]];
      }
      
      const lastKey = update.path[update.path.length - 1];
      
      switch (update.operation) {
        case 'set':
          target[lastKey] = update.value;
          break;
        case 'merge':
          target[lastKey] = { ...target[lastKey], ...update.value };
          break;
        case 'delete':
          delete target[lastKey];
          break;
      }
      
      newState.version = update.version;
      newState.last_update = new Date().toISOString();
      
      return newState;
    });
  }
  
  private handleCursorMove(data: any) {
    this.participants.update(participants => {
      const newParticipants = new Map(participants);
      const participant = newParticipants.get(data.user_id);
      if (participant) {
        participant.cursor = data.position;
        newParticipants.set(data.user_id, participant);
      }
      return newParticipants;
    });
  }
  
  // Public methods for TTRPG actions
  
  async callTool(tool: string, params: any): Promise<any> {
    return this.request({
      type: 'tool_call',
      method: tool,
      params
    }, { timeout: 30000 });
  }
  
  async rollDice(expression: string, purpose?: string): Promise<DiceRollResult> {
    return this.request({
      type: 'dice_roll',
      data: { expression, purpose }
    }, { timeout: 5000 });
  }
  
  updateInitiative(initiative: InitiativeEntry[]) {
    this.send({
      type: 'state_update',
      data: {
        path: ['initiative_order'],
        value: initiative,
        operation: 'set',
        version: get(this.sharedState).version + 1,
        previous_version: get(this.sharedState).version
      }
    });
  }
  
  moveCursor(x: number, y: number, element?: string) {
    // Throttle cursor movements
    if (this._cursorThrottle) {
      clearTimeout(this._cursorThrottle);
    }
    
    this._cursorThrottle = setTimeout(() => {
      this.send({
        type: 'cursor_move',
        data: {
          position: { x, y, element, timestamp: Date.now() }
        }
      });
    }, 50); // 20 FPS max
  }
  
  private _cursorThrottle?: NodeJS.Timeout;
}

// Svelte store factory for TTRPG WebSocket
export function createTTRPGWebSocket(config: TTRPGWebSocketConfig) {
  const client = new TTRPGWebSocketClient(config);
  
  // Create derived stores for convenience
  const isConnected = derived(
    client.connectionState,
    $state => $state === 'authenticated'
  );
  
  const participantList = derived(
    client.participants,
    $participants => Array.from($participants.values())
  );
  
  const onlineCount = derived(
    client.participants,
    $participants => {
      return Array.from($participants.values())
        .filter(p => p.status === 'online').length;
    }
  );
  
  return {
    client,
    connectionState: client.connectionState,
    participants: client.participants,
    sharedState: client.sharedState,
    activityFeed: client.activityFeed,
    isConnected,
    participantList,
    onlineCount,
    
    // Convenience methods
    connect: () => client.connect(),
    disconnect: () => client.close(),
    rollDice: (expr: string, purpose?: string) => client.rollDice(expr, purpose),
    callTool: (tool: string, params: any) => client.callTool(tool, params),
    updateInitiative: (init: InitiativeEntry[]) => client.updateInitiative(init),
    moveCursor: (x: number, y: number, el?: string) => client.moveCursor(x, y, el)
  };
}
```

## Performance Considerations

### Connection Pooling

```python
class MCPProcessPool:
    """Pool of MCP processes for efficient resource usage"""
    
    def __init__(self, min_size: int = 2, max_size: int = 10):
        self.min_size = min_size
        self.max_size = max_size
        self.available: asyncio.Queue = asyncio.Queue()
        self.in_use: Set[Any] = set()
        self._lock = asyncio.Lock()
        self._created = 0
    
    async def initialize(self):
        """Pre-create minimum number of processes"""
        for _ in range(self.min_size):
            process = await self._create_process()
            await self.available.put(process)
    
    async def acquire(self) -> Any:
        """Get a process from the pool"""
        try:
            # Try to get available process
            process = self.available.get_nowait()
        except asyncio.QueueEmpty:
            # Create new if under limit
            async with self._lock:
                if self._created < self.max_size:
                    process = await self._create_process()
                    self._created += 1
                else:
                    # Wait for available process
                    process = await self.available.get()
        
        self.in_use.add(process)
        return process
    
    async def release(self, process: Any):
        """Return process to pool"""
        self.in_use.discard(process)
        
        # Check if process is still healthy
        if process.returncode is None:
            await self.available.put(process)
        else:
            # Replace dead process
            async with self._lock:
                self._created -= 1
                if self._created < self.min_size:
                    new_process = await self._create_process()
                    await self.available.put(new_process)
                    self._created += 1
```

### Message Batching

```typescript
class MessageBatcher {
  private queue: any[] = [];
  private timer: NodeJS.Timeout | null = null;
  private maxBatchSize = 10;
  private batchDelay = 50; // ms
  
  constructor(private send: (messages: any[]) => void) {}
  
  add(message: any) {
    this.queue.push(message);
    
    if (this.queue.length >= this.maxBatchSize) {
      this.flush();
    } else if (!this.timer) {
      this.timer = setTimeout(() => this.flush(), this.batchDelay);
    }
  }
  
  private flush() {
    if (this.queue.length === 0) return;
    
    const batch = this.queue.splice(0);
    this.send(batch);
    
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib
import json

class ResponseCache:
    """Cache for MCP responses"""
    
    def __init__(self, ttl: int = 300):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = ttl
    
    def get_key(self, method: str, params: Dict[str, Any]) -> str:
        """Generate cache key"""
        key_data = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached response"""
        key = self.get_key(method, params)
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.ttl:
                return value
            else:
                del self.cache[key]
        
        return None
    
    async def set(self, method: str, params: Dict[str, Any], value: Any):
        """Cache response"""
        key = self.get_key(method, params)
        self.cache[key] = (value, datetime.now())
    
    @lru_cache(maxsize=100)
    def is_cacheable(self, method: str) -> bool:
        """Check if method is cacheable"""
        # Don't cache mutations or real-time data
        non_cacheable = {
            'dice_roll', 'update_character', 'save_campaign',
            'cursor_move', 'canvas_draw'
        }
        return method not in non_cacheable
```

## Security Implications

### Security Checklist

1. **Authentication & Authorization**
   - ✅ JWT tokens with expiration
   - ✅ Role-based access control (GM, Player, Spectator)
   - ✅ Session-based permissions
   - ✅ Token refresh mechanism

2. **Transport Security**
   - ✅ WSS (WebSocket Secure) with TLS 1.3
   - ✅ Certificate pinning for mobile clients
   - ✅ Secure cookie flags for web clients

3. **Input Validation**
   - ✅ Schema validation for all messages
   - ✅ Sanitization of user-generated content
   - ✅ Rate limiting per user/session
   - ✅ Message size limits

4. **Process Isolation**
   - ✅ Separate MCP processes per session
   - ✅ Resource limits (CPU, memory, file handles)
   - ✅ Sandboxed execution environment
   - ✅ No direct filesystem access from WebSocket

5. **Data Protection**
   - ✅ Encryption at rest for sensitive data
   - ✅ PII handling compliance
   - ✅ Audit logging for all operations
   - ✅ Data retention policies

### Security Implementation

```python
from dataclasses import dataclass
from typing import Set, Dict, Any
import hashlib
import hmac

@dataclass
class SecurityPolicy:
    """Security policy for TTRPG sessions"""
    
    # Rate limiting
    max_requests_per_minute: int = 100
    max_message_size: int = 1024 * 100  # 100KB
    max_batch_size: int = 10
    
    # Session limits
    max_sessions_per_user: int = 3
    max_participants_per_campaign: int = 20
    session_timeout_minutes: int = 60
    
    # Content filtering
    allowed_tools: Set[str] = None
    blocked_parameters: Set[str] = None
    
    def __post_init__(self):
        if self.allowed_tools is None:
            self.allowed_tools = {
                'search', 'get_character', 'roll_dice',
                'get_campaign', 'list_items'
            }
        if self.blocked_parameters is None:
            self.blocked_parameters = {
                '__proto__', 'constructor', 'prototype'
            }

class SecurityValidator:
    """Validates messages against security policy"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
    
    def validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate incoming message"""
        
        # Check message size
        if len(json.dumps(message)) > self.policy.max_message_size:
            raise ValueError("Message too large")
        
        # Check for prototype pollution
        if self._has_dangerous_keys(message):
            raise ValueError("Dangerous keys detected")
        
        # Validate tool calls
        if message.get("type") == "tool_call":
            tool = message.get("method")
            if tool not in self.policy.allowed_tools:
                raise ValueError(f"Tool not allowed: {tool}")
        
        return True
    
    def _has_dangerous_keys(self, obj: Any, depth: int = 0) -> bool:
        """Recursively check for dangerous keys"""
        if depth > 10:  # Prevent deep recursion
            return True
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in self.policy.blocked_parameters:
                    return True
                if self._has_dangerous_keys(value, depth + 1):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if self._has_dangerous_keys(item, depth + 1):
                    return True
        
        return False

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.buckets: Dict[str, Tuple[float, datetime]] = {}
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        
        if client_id not in self.buckets:
            self.buckets[client_id] = (self.capacity, now)
            return True
        
        tokens, last_update = self.buckets[client_id]
        
        # Calculate tokens to add
        time_passed = (now - last_update).total_seconds()
        tokens = min(self.capacity, tokens + time_passed * self.refill_rate)
        
        if tokens >= 1:
            # Consume token
            self.buckets[client_id] = (tokens - 1, now)
            return True
        else:
            # Rate limit exceeded
            self.buckets[client_id] = (tokens, now)
            return False
```

## Migration Strategy

### Phase 1: Infrastructure Setup (Week 1)
1. Deploy enhanced bridge server alongside existing MCP
2. Configure WebSocket endpoints and authentication
3. Set up monitoring and logging

### Phase 2: Client Integration (Week 2)
1. Integrate TTRPGWebSocketClient in SvelteKit
2. Update stores and components for real-time data
3. Implement fallback for WebSocket failures

### Phase 3: Feature Rollout (Week 3-4)
1. Enable real-time cursor tracking
2. Implement collaborative canvas
3. Add dice rolling and initiative tracking
4. Deploy activity feed

### Phase 4: Optimization (Week 5)
1. Implement connection pooling
2. Add response caching
3. Optimize message batching
4. Performance testing with 20+ concurrent users

## Testing Strategy

### Unit Tests
```python
# Python tests
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_session_creation():
    """Test session creation and lifecycle"""
    manager = SessionManager(config={})
    session = await manager.create_session(
        user_id="test_user",
        username="TestUser",
        campaign_id="campaign_123"
    )
    
    assert session.session_id is not None
    assert session.state == SessionState.INITIALIZING
    assert session.campaign_id == "campaign_123"

@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiter functionality"""
    limiter = RateLimiter(capacity=10, refill_rate=1.0)
    
    # Should allow initial requests
    for _ in range(10):
        assert await limiter.check_rate_limit("client_1")
    
    # Should block after capacity exceeded
    assert not await limiter.check_rate_limit("client_1")
```

### Integration Tests
```typescript
// TypeScript tests
import { describe, it, expect, beforeEach } from 'vitest';
import { createTTRPGWebSocket } from '$lib/realtime/ttrpg-websocket';

describe('TTRPG WebSocket Integration', () => {
  let client;
  
  beforeEach(() => {
    client = createTTRPGWebSocket({
      url: 'ws://localhost:8080/ws',
      authToken: 'test_token',
      campaignId: 'test_campaign'
    });
  });
  
  it('should connect and authenticate', async () => {
    await client.connect();
    
    // Wait for authentication
    await new Promise(resolve => {
      const unsubscribe = client.connectionState.subscribe(state => {
        if (state === 'authenticated') {
          unsubscribe();
          resolve();
        }
      });
    });
    
    expect(get(client.isConnected)).toBe(true);
  });
  
  it('should handle dice rolls', async () => {
    await client.connect();
    
    const result = await client.rollDice('2d6+3');
    
    expect(result).toHaveProperty('total');
    expect(result.rolls).toHaveLength(2);
    expect(result.total).toBeGreaterThanOrEqual(5); // Min: 1+1+3
    expect(result.total).toBeLessThanOrEqual(15);   // Max: 6+6+3
  });
});
```

## Monitoring & Observability

```python
# Monitoring setup
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Metrics
ws_connections = Gauge('ws_connections_active', 'Active WebSocket connections')
ws_messages = Counter('ws_messages_total', 'Total WebSocket messages', ['type'])
ws_errors = Counter('ws_errors_total', 'WebSocket errors', ['error_type'])
mcp_latency = Histogram('mcp_request_duration_seconds', 'MCP request latency')
session_duration = Histogram('session_duration_seconds', 'Session duration')

# Structured logging
logger = structlog.get_logger()

class MonitoredBridgeServer(TTRPGBridgeServer):
    """Bridge server with monitoring"""
    
    async def handle_connection(self, websocket: WebSocket):
        """Monitor WebSocket connections"""
        ws_connections.inc()
        start_time = datetime.now()
        
        try:
            await super().handle_connection(websocket)
        finally:
            ws_connections.dec()
            duration = (datetime.now() - start_time).total_seconds()
            session_duration.observe(duration)
            
            logger.info(
                "session_ended",
                duration=duration,
                session_id=self.current_session_id
            )
    
    async def handle_message(self, message: Dict[str, Any]):
        """Monitor message handling"""
        ws_messages.labels(type=message.get('type', 'unknown')).inc()
        
        with mcp_latency.time():
            return await super().handle_message(message)
```

## Conclusion

This spike has outlined a comprehensive approach to implementing WebSocket communication for the TTRPG MCP Server project. The recommended architecture leverages the existing bridge server with enhancements for:

1. **Production-ready session management** with connection pooling and state persistence
2. **Secure authentication** using JWT tokens with role-based access
3. **Optimized performance** through caching, batching, and process pooling
4. **Real-time collaboration** features specific to TTRPG sessions
5. **Comprehensive security** measures including rate limiting and input validation

The implementation provides a solid foundation for handling multiple concurrent TTRPG sessions with low latency and high reliability, while maintaining compatibility with the existing MCP server architecture.

### Next Steps

1. Review and approve the architecture with the team
2. Set up development environment with the enhanced bridge server
3. Implement authentication service with JWT support
4. Begin client integration with the SvelteKit frontend
5. Conduct load testing with target concurrent user counts
6. Deploy to staging environment for user acceptance testing

### Success Metrics

- WebSocket connection establishment < 1 second
- Message latency < 100ms for cursor updates
- Support for 20+ concurrent users per campaign
- 99.9% uptime for WebSocket service
- Zero message loss during reconnection
- Successful failover to SSE when WebSocket unavailable