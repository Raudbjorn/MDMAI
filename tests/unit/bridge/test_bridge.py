"""Tests for the MCP Bridge Service."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from src.bridge.bridge_server import BridgeServer, create_bridge_app
from src.bridge.mcp_process_manager import MCPProcess, MCPProcessManager
from src.bridge.models import (
    BridgeConfig,
    MCPRequest,
    MCPResponse,
    MCPSession,
    SessionState,
    TransportType,
)
from src.bridge.protocol_translator import MCPProtocolTranslator
from src.bridge.session_manager import BridgeSessionManager


@pytest.fixture
def bridge_config():
    """Create test bridge configuration."""
    return BridgeConfig(
        mcp_server_path="src.main",
        max_processes=5,
        process_timeout=30,
        session_timeout=300,
        max_sessions_per_client=2,
        enable_websocket=True,
        enable_sse=True,
        enable_http=True,
        log_requests=True,
        log_responses=True,
    )


@pytest.fixture
def process_manager(bridge_config):
    """Create test process manager."""
    return MCPProcessManager(bridge_config)


@pytest.fixture
def session_manager(bridge_config, process_manager):
    """Create test session manager."""
    return BridgeSessionManager(bridge_config, process_manager)


@pytest.fixture
def protocol_translator():
    """Create test protocol translator."""
    return MCPProtocolTranslator()


@pytest.fixture
def bridge_server(bridge_config):
    """Create test bridge server."""
    return BridgeServer(bridge_config)


@pytest.fixture
def test_client(bridge_server):
    """Create test client."""
    return TestClient(bridge_server.app)


class TestMCPProcess:
    """Test MCP process management."""
    
    @pytest.mark.asyncio
    async def test_process_start_stop(self, bridge_config):
        """Test starting and stopping an MCP process."""
        process = MCPProcess("test-session", bridge_config)
        
        # Mock subprocess
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Mock stdout readline to return initialization response
            mock_process.stdout.readline = AsyncMock(
                side_effect=[
                    json.dumps({
                        "jsonrpc": "2.0",
                        "id": process.pending_requests.get("1", MagicMock()).request_id if process.pending_requests else "1",
                        "result": {"capabilities": {"tools": True}},
                    }).encode() + b"\n",
                    b"",  # EOF
                ]
            )
            
            # Start process
            assert await process.start()
            assert process._running
            assert process.process == mock_process
            
            # Stop process
            await process.stop()
            assert not process._running
            assert process.process is None
    
    @pytest.mark.asyncio
    async def test_send_request(self, bridge_config):
        """Test sending a request to an MCP process."""
        process = MCPProcess("test-session", bridge_config)
        process._running = True
        process._initialized = True
        
        # Mock process
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        process.process = mock_process
        
        # Create a future for the response
        response_future = asyncio.Future()
        response_future.set_result({"status": "success"})
        
        # Mock pending request
        process.pending_requests["test-id"] = MagicMock(callback=response_future)
        
        with patch.object(process, "_write_message", new_callable=AsyncMock):
            # Send request (using existing pending request)
            with patch.object(MCPRequest, "__init__", return_value=None):
                with patch.object(MCPRequest, "id", new="test-id"):
                    result = await process.send_request("test_method", {"param": "value"})
        
        assert result == {"status": "success"}


class TestMCPProcessManager:
    """Test MCP process manager."""
    
    @pytest.mark.asyncio
    async def test_create_process(self, process_manager):
        """Test creating a new process."""
        # Mock process creation
        mock_process = AsyncMock()
        mock_process.start = AsyncMock(return_value=True)
        
        with patch.object(MCPProcess, "__init__", return_value=None):
            with patch.object(MCPProcess, "start", new_callable=AsyncMock, return_value=True):
                process_manager.processes["test-session"] = mock_process
                
                # Get process
                process = await process_manager.get_process("test-session")
                assert process == mock_process
    
    @pytest.mark.asyncio
    async def test_max_processes_limit(self, bridge_config):
        """Test maximum processes limit enforcement."""
        process_manager = MCPProcessManager(bridge_config)
        process_manager.config.max_processes = 2
        
        # Mock process creation
        mock_process = AsyncMock()
        mock_process.start = AsyncMock(return_value=True)
        mock_process._running = True
        mock_process.last_activity = None
        
        with patch.object(MCPProcess, "__init__", return_value=None):
            with patch.object(MCPProcess, "start", new_callable=AsyncMock, return_value=True):
                # Create max processes
                for i in range(2):
                    session_id = f"session-{i}"
                    process_manager.processes[session_id] = mock_process
                
                # Try to create one more
                with pytest.raises(RuntimeError, match="Maximum number of processes"):
                    await process_manager.create_process("session-3")


class TestBridgeSessionManager:
    """Test bridge session manager."""
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a new session."""
        # Mock process creation
        mock_process = AsyncMock()
        mock_process.capabilities = {"tools": True}
        mock_process.process = MagicMock(pid=12345)
        
        with patch.object(session_manager.process_manager, "create_process", return_value=mock_process):
            session = await session_manager.create_session(
                client_id="test-client",
                transport=TransportType.WEBSOCKET,
            )
        
        assert session.client_id == "test-client"
        assert session.transport == TransportType.WEBSOCKET
        assert session.state == SessionState.READY
        assert session.capabilities == {"tools": True}
    
    @pytest.mark.asyncio
    async def test_session_limit_per_client(self, session_manager):
        """Test session limit per client enforcement."""
        session_manager.config.max_sessions_per_client = 2
        
        # Mock process creation
        mock_process = AsyncMock()
        mock_process.capabilities = {}
        mock_process.process = MagicMock(pid=12345)
        
        with patch.object(session_manager.process_manager, "create_process", return_value=mock_process):
            # Create max sessions
            for i in range(2):
                session = await session_manager.create_session(
                    client_id="test-client",
                    transport=TransportType.HTTP,
                )
                assert session is not None
            
            # Try to create one more
            with pytest.raises(RuntimeError, match="Maximum sessions"):
                await session_manager.create_session(
                    client_id="test-client",
                    transport=TransportType.HTTP,
                )
    
    @pytest.mark.asyncio
    async def test_send_request(self, session_manager):
        """Test sending a request through session manager."""
        # Create mock session
        session = MCPSession(
            session_id="test-session",
            client_id="test-client",
            state=SessionState.READY,
        )
        session_manager.sessions["test-session"] = session
        
        # Mock process
        mock_process = AsyncMock()
        mock_process.send_request = AsyncMock(return_value={"result": "success"})
        
        with patch.object(session_manager.process_manager, "get_process", return_value=mock_process):
            result = await session_manager.send_request(
                "test-session",
                "test_method",
                {"param": "value"},
            )
        
        assert result == {"result": "success"}
        assert session.state == SessionState.READY


class TestProtocolTranslator:
    """Test protocol translator."""
    
    def test_parse_mcp_request(self, protocol_translator):
        """Test parsing MCP request."""
        message = {
            "jsonrpc": "2.0",
            "id": "123",
            "method": "test_method",
            "params": {"param": "value"},
        }
        
        request = protocol_translator.parse_client_message(message)
        
        assert isinstance(request, MCPRequest)
        assert request.id == "123"
        assert request.method == "test_method"
        assert request.params == {"param": "value"}
    
    def test_translate_tool_call(self, protocol_translator):
        """Test translating tool call."""
        message = {
            "tool": "search",
            "params": {"query": "test"},
            "id": "456",
        }
        
        request = protocol_translator.parse_client_message(message)
        
        assert isinstance(request, MCPRequest)
        assert request.method == "tools/search"
        assert request.params == {"query": "test"}
    
    def test_format_response(self, protocol_translator):
        """Test formatting response."""
        response = MCPResponse(
            id="789",
            result={"status": "success"},
        )
        
        # JSON-RPC format
        formatted = protocol_translator.format_response(response, "json-rpc")
        assert formatted["id"] == "789"
        assert formatted["result"] == {"status": "success"}
        
        # OpenAI format
        formatted = protocol_translator.format_response(response, "openai")
        assert formatted["role"] == "function"
        assert formatted["name"] == "789"
        assert json.loads(formatted["content"]) == {"status": "success"}


class TestBridgeServer:
    """Test bridge server endpoints."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
    
    @pytest.mark.asyncio
    async def test_create_session_endpoint(self, test_client, bridge_server):
        """Test session creation endpoint."""
        # Mock session creation
        mock_session = MCPSession(
            session_id="test-session",
            client_id="test-client",
            state=SessionState.READY,
            capabilities={"tools": True},
        )
        
        with patch.object(bridge_server.session_manager, "create_session", return_value=mock_session):
            response = test_client.post(
                "/sessions",
                json={"client_id": "test-client"},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["client_id"] == "test-client"
        assert data["state"] == "ready"
    
    @pytest.mark.asyncio
    async def test_tool_discovery(self, test_client, bridge_server):
        """Test tool discovery endpoint."""
        # Mock session and request
        mock_session = MCPSession(session_id="test-session", state=SessionState.READY)
        
        with patch.object(bridge_server.session_manager, "create_session", return_value=mock_session):
            with patch.object(
                bridge_server.session_manager,
                "send_request",
                return_value={"tools": [{"name": "search", "description": "Search tool"}]},
            ):
                response = test_client.post("/tools/discover", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "search"
    
    @pytest.mark.asyncio
    async def test_tool_call(self, test_client, bridge_server):
        """Test tool call endpoint."""
        # Mock session and request
        mock_session = MCPSession(session_id="test-session", state=SessionState.READY)
        
        with patch.object(bridge_server.session_manager, "create_session", return_value=mock_session):
            with patch.object(
                bridge_server.session_manager,
                "send_request",
                return_value={"result": "search results"},
            ):
                response = test_client.post(
                    "/tools/call",
                    json={
                        "tool": "search",
                        "params": {"query": "test"},
                    },
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["tool"] == "search"
        assert data["result"] == {"result": "search results"}


class TestWebSocket:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_session_creation(self, bridge_server):
        """Test WebSocket session creation."""
        # Create mock WebSocket
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.receive_json = AsyncMock(
            side_effect=[
                {
                    "type": "create_session",
                    "client_id": "test-client",
                },
                asyncio.CancelledError(),  # Stop the loop
            ]
        )
        mock_ws.send_json = AsyncMock()
        
        # Mock session creation
        mock_session = MCPSession(
            session_id="test-session",
            state=SessionState.READY,
            capabilities={"tools": True},
        )
        
        with patch.object(bridge_server.session_manager, "create_session", return_value=mock_session):
            try:
                await bridge_server._handle_websocket(mock_ws)
            except asyncio.CancelledError:
                pass
        
        # Verify WebSocket interactions
        mock_ws.accept.assert_called_once()
        mock_ws.send_json.assert_called()
        
        # Check session created message
        call_args = mock_ws.send_json.call_args_list[0][0][0]
        assert call_args["type"] == "session_created"
        assert call_args["session_id"] == "test-session"