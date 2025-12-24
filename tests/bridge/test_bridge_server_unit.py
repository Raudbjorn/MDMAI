"""Comprehensive unit tests for the MCP Bridge Server."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from httpx import AsyncClient
from pydantic import ValidationError

from src.bridge.bridge_server import (
    BridgeServer,
    SessionCreateRequest,
    ToolCallRequest,
    ToolDiscoveryRequest,
    create_bridge_app,
)
from src.bridge.models import (
    BridgeConfig,
    BridgeMessage,
    BridgeStats,
    MCPError,
    MCPErrorCode,
    MCPNotification,
    MCPRequest,
    MCPResponse,
    MCPSession,
    PendingRequest,
    SessionState,
    TransportType,
)


@pytest.fixture
def mock_config() -> BridgeConfig:
    """Create a mock bridge configuration for testing."""
    return BridgeConfig(
        mcp_server_path="test.mcp.server",
        max_processes=10,
        process_timeout=60,
        session_timeout=600,
        max_sessions_per_client=5,
        enable_websocket=True,
        enable_sse=True,
        enable_http=True,
        require_auth=False,
        api_keys=[],
        log_requests=True,
        log_responses=True,
    )


@pytest.fixture
def mock_config_with_auth() -> BridgeConfig:
    """Create a mock bridge configuration with authentication enabled."""
    return BridgeConfig(
        mcp_server_path="test.mcp.server",
        require_auth=True,
        api_keys=["test-api-key-123", "test-api-key-456"],
    )


@pytest.fixture
def bridge_server(mock_config: BridgeConfig) -> BridgeServer:
    """Create a bridge server instance for testing."""
    return BridgeServer(mock_config)


@pytest.fixture
def test_client(bridge_server: BridgeServer) -> TestClient:
    """Create a test client for the bridge server."""
    return TestClient(bridge_server.app)


class TestBridgeServerInitialization:
    """Test BridgeServer initialization and configuration."""
    
    def test_initialization_with_default_config(self):
        """Test server initialization with default configuration."""
        server = BridgeServer()
        
        assert server.config is not None
        assert server.process_manager is not None
        assert server.session_manager is not None
        assert server.protocol_translator is not None
        assert server.app is not None
        assert not server._started
        assert server.security is None  # No auth by default
    
    def test_initialization_with_custom_config(self, mock_config: BridgeConfig):
        """Test server initialization with custom configuration."""
        server = BridgeServer(mock_config)
        
        assert server.config == mock_config
        assert server.config.mcp_server_path == "test.mcp.server"
        assert server.config.max_processes == 10
        assert server.security is None  # Auth disabled in mock_config
    
    def test_initialization_with_auth_enabled(self, mock_config_with_auth: BridgeConfig):
        """Test server initialization with authentication enabled."""
        server = BridgeServer(mock_config_with_auth)
        
        assert server.config.require_auth is True
        assert server.security is not None
        assert len(server.config.api_keys) == 2
    
    def test_app_creation(self, bridge_server: BridgeServer):
        """Test FastAPI app creation and configuration."""
        app = bridge_server.app
        
        assert app.title == "MCP Bridge Server"
        assert app.version == "0.1.0"
        
        # Check routes are registered
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/sessions" in routes
        assert "/tools/discover" in routes
        assert "/tools/call" in routes
        assert "/ws" in routes
        assert "/stats" in routes


class TestAuthenticationAndRateLimiting:
    """Test authentication and rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_verify_auth_disabled(self, bridge_server: BridgeServer):
        """Test authentication verification when disabled."""
        result = await bridge_server.verify_auth(None)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_auth_enabled_valid_key(self, mock_config_with_auth: BridgeConfig):
        """Test authentication with valid API key."""
        server = BridgeServer(mock_config_with_auth)
        
        credentials = MagicMock()
        credentials.credentials = "test-api-key-123"
        
        result = await server.verify_auth(credentials)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_verify_auth_enabled_invalid_key(self, mock_config_with_auth: BridgeConfig):
        """Test authentication with invalid API key."""
        server = BridgeServer(mock_config_with_auth)
        
        credentials = MagicMock()
        credentials.credentials = "invalid-key"
        
        result = await server.verify_auth(credentials)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_verify_auth_enabled_no_credentials(self, mock_config_with_auth: BridgeConfig):
        """Test authentication with no credentials provided."""
        server = BridgeServer(mock_config_with_auth)
        
        result = await server.verify_auth(None)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self, bridge_server: BridgeServer):
        """Test rate limiting when disabled."""
        with patch("src.bridge.config.settings.enable_rate_limiting", False):
            result = await bridge_server.check_rate_limit("test-client")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiting_within_limit(self, bridge_server: BridgeServer):
        """Test rate limiting within allowed limits."""
        with patch("src.bridge.config.settings.enable_rate_limiting", True):
            with patch("src.bridge.config.settings.rate_limit_requests", 5):
                with patch("src.bridge.config.settings.rate_limit_period", 60):
                    # First few requests should pass
                    for _ in range(5):
                        result = await bridge_server.check_rate_limit("test-client")
                        assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiting_exceeded(self, bridge_server: BridgeServer):
        """Test rate limiting when limit is exceeded."""
        with patch("src.bridge.config.settings.enable_rate_limiting", True):
            with patch("src.bridge.config.settings.rate_limit_requests", 3):
                with patch("src.bridge.config.settings.rate_limit_period", 60):
                    # Fill up the limit
                    for _ in range(3):
                        await bridge_server.check_rate_limit("test-client")
                    
                    # Next request should fail
                    result = await bridge_server.check_rate_limit("test-client")
                    assert result is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_cleanup(self, bridge_server: BridgeServer):
        """Test rate limiting cleanup of old entries."""
        with patch("src.bridge.config.settings.enable_rate_limiting", True):
            with patch("src.bridge.config.settings.rate_limit_requests", 3):
                with patch("src.bridge.config.settings.rate_limit_period", 1):  # 1 second period
                    # Add some entries
                    client_id = "test-client"
                    now = datetime.now()
                    old_time = now - timedelta(seconds=2)
                    
                    bridge_server.rate_limits[client_id] = [old_time, old_time, now]
                    
                    # Check rate limit should clean old entries
                    result = await bridge_server.check_rate_limit(client_id)
                    assert result is True
                    assert len(bridge_server.rate_limits[client_id]) == 2  # Old entries removed, new one added


class TestHTTPEndpoints:
    """Test HTTP REST endpoints."""
    
    def test_health_check(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert isinstance(data["uptime"], (int, float))
    
    @pytest.mark.asyncio
    async def test_create_session_success(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test successful session creation."""
        mock_session = MCPSession(
            session_id="test-session-123",
            client_id="test-client",
            state=SessionState.READY,
            capabilities={"tools": ["search", "analyze"]},
        )
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            response = test_client.post(
                "/sessions",
                json={
                    "client_id": "test-client",
                    "metadata": {"user": "test"},
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["client_id"] == "test-client"
        assert data["state"] == "ready"
        assert data["capabilities"] == {"tools": ["search", "analyze"]}
    
    @pytest.mark.asyncio
    async def test_create_session_failure(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test session creation failure."""
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            side_effect=RuntimeError("Process creation failed")
        ):
            response = test_client.post(
                "/sessions",
                json={"client_id": "test-client"}
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "Process creation failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_session_exists(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test getting existing session information."""
        mock_session = MCPSession(
            session_id="test-session-123",
            client_id="test-client",
            state=SessionState.READY,
            capabilities={"tools": ["search"]},
        )
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=mock_session
        ):
            response = test_client.get("/sessions/test-session-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["state"] == "ready"
    
    @pytest.mark.asyncio
    async def test_get_session_not_found(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test getting non-existent session."""
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=None
        ):
            response = test_client.get("/sessions/non-existent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_delete_session(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test deleting a session."""
        with patch.object(
            bridge_server.session_manager,
            "remove_session",
            return_value=None
        ):
            response = test_client.delete("/sessions/test-session-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
    
    @pytest.mark.asyncio
    async def test_discover_tools_new_session(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test tool discovery with new session creation."""
        mock_session = MCPSession(session_id="new-session", state=SessionState.READY)
        mock_tools = {
            "tools": [
                {"name": "search", "description": "Search for information"},
                {"name": "analyze", "description": "Analyze data"},
            ]
        }
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "send_request",
                return_value=mock_tools
            ):
                response = test_client.post("/tools/discover", json={})
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new-session"
        assert len(data["tools"]) == 2
        assert data["tools"][0]["name"] == "search"
    
    @pytest.mark.asyncio
    async def test_discover_tools_existing_session(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test tool discovery with existing session."""
        mock_tools = {
            "tools": [
                {"name": "calculate", "description": "Perform calculations"},
            ]
        }
        
        with patch.object(
            bridge_server.session_manager,
            "send_request",
            return_value=mock_tools
        ):
            response = test_client.post(
                "/tools/discover",
                json={"session_id": "existing-session"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session"
        assert len(data["tools"]) == 1
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test successful tool invocation."""
        mock_session = MCPSession(session_id="test-session", state=SessionState.READY)
        mock_result = {"result": "Search completed", "items": ["item1", "item2"]}
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "send_request",
                return_value=mock_result
            ):
                response = test_client.post(
                    "/tools/call",
                    json={
                        "tool": "search",
                        "params": {"query": "test query"},
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["tool"] == "search"
        assert data["result"]["result"] == "Search completed"
        assert len(data["result"]["items"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_stats(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test statistics endpoint."""
        mock_session_stats = {"active_sessions": 3}
        mock_process_stats = [
            {"pid": 1234, "memory_mb": 100},
            {"pid": 5678, "memory_mb": 150},
        ]
        
        with patch.object(
            bridge_server.session_manager,
            "get_stats",
            return_value=mock_session_stats
        ):
            with patch.object(
                bridge_server.process_manager,
                "get_stats",
                return_value=mock_process_stats
            ):
                response = test_client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["active_sessions"] == 3
        assert data["active_processes"] == 2
        assert "uptime_seconds" in data
        assert len(data["process_stats"]) == 2


class TestWebSocketHandling:
    """Test WebSocket connection handling."""
    
    @pytest.mark.asyncio
    async def test_websocket_create_session(self, bridge_server: BridgeServer):
        """Test WebSocket session creation flow."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()
        
        mock_session = MCPSession(
            session_id="ws-session-123",
            state=SessionState.READY,
            capabilities={"tools": ["search"]},
        )
        
        # Simulate create_session message then disconnect
        mock_ws.receive_json = AsyncMock(side_effect=[
            {
                "type": "create_session",
                "client_id": "ws-client",
                "metadata": {"source": "websocket"},
            },
            WebSocketDisconnect(),
        ])
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "update_session_state"
            ) as mock_update:
                await bridge_server._handle_websocket(mock_ws)
        
        # Verify WebSocket interactions
        mock_ws.accept.assert_called_once()
        
        # Verify session created response
        expected_response = {
            "type": "session_created",
            "session_id": "ws-session-123",
            "capabilities": {"tools": ["search"]},
        }
        mock_ws.send_json.assert_called_with(expected_response)
        
        # Verify session state updated on disconnect
        mock_update.assert_called_with("ws-session-123", SessionState.DISCONNECTED)
    
    @pytest.mark.asyncio
    async def test_websocket_attach_existing_session(self, bridge_server: BridgeServer):
        """Test attaching to existing session via WebSocket."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        mock_session = MCPSession(
            session_id="existing-session",
            state=SessionState.READY,
            capabilities={"tools": ["analyze"]},
        )
        
        mock_ws.receive_json = AsyncMock(side_effect=[
            {
                "type": "attach_session",
                "session_id": "existing-session",
            },
            WebSocketDisconnect(),
        ])
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "update_session_state"
            ):
                await bridge_server._handle_websocket(mock_ws)
        
        # Verify session attached response
        expected_response = {
            "type": "session_attached",
            "session_id": "existing-session",
            "capabilities": {"tools": ["analyze"]},
        }
        mock_ws.send_json.assert_called_with(expected_response)
    
    @pytest.mark.asyncio
    async def test_websocket_attach_nonexistent_session(self, bridge_server: BridgeServer):
        """Test attaching to non-existent session via WebSocket."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()
        
        mock_ws.receive_json = AsyncMock(return_value={
            "type": "attach_session",
            "session_id": "non-existent",
        })
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=None
        ):
            await bridge_server._handle_websocket(mock_ws)
        
        # Verify error response and connection closure
        mock_ws.send_json.assert_called_with({
            "type": "error",
            "error": "Session non-existent not found",
        })
        mock_ws.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_invalid_initial_message(self, bridge_server: BridgeServer):
        """Test WebSocket with invalid initial message."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()
        
        mock_ws.receive_json = AsyncMock(return_value={
            "type": "invalid_type",
        })
        
        await bridge_server._handle_websocket(mock_ws)
        
        # Verify error response
        mock_ws.send_json.assert_called_with({
            "type": "error",
            "error": "Must create or attach to a session first",
        })
        mock_ws.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, bridge_server: BridgeServer):
        """Test WebSocket message processing after session creation."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        mock_session = MCPSession(
            session_id="ws-session",
            state=SessionState.READY,
        )
        
        mock_request = MCPRequest(
            id="req-123",
            method="tools/search",
            params={"query": "test"},
        )
        
        mock_result = {"results": ["result1", "result2"]}
        
        # Simulate session creation, then request, then disconnect
        mock_ws.receive_json = AsyncMock(side_effect=[
            {"type": "create_session"},
            {
                "jsonrpc": "2.0",
                "id": "req-123",
                "method": "tools/search",
                "params": {"query": "test"},
                "format": "json-rpc",
            },
            WebSocketDisconnect(),
        ])
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.protocol_translator,
                "parse_client_message",
                return_value=mock_request
            ):
                with patch.object(
                    bridge_server.session_manager,
                    "send_request",
                    return_value=mock_result
                ):
                    with patch.object(
                        bridge_server.protocol_translator,
                        "format_response",
                        return_value={"id": "req-123", "result": mock_result}
                    ):
                        with patch.object(
                            bridge_server.session_manager,
                            "update_session_state"
                        ):
                            await bridge_server._handle_websocket(mock_ws)
        
        # Verify response was sent
        calls = mock_ws.send_json.call_args_list
        assert len(calls) >= 2  # Session created + response
        
        # Check the response message
        response_call = calls[1][0][0]
        assert response_call["type"] == "response"
        assert response_call["data"]["id"] == "req-123"
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, bridge_server: BridgeServer):
        """Test WebSocket error response handling."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        mock_session = MCPSession(session_id="ws-session", state=SessionState.READY)
        mock_request = MCPRequest(id="req-456", method="tools/invalid", params={})
        
        mock_ws.receive_json = AsyncMock(side_effect=[
            {"type": "create_session"},
            {
                "jsonrpc": "2.0",
                "id": "req-456",
                "method": "tools/invalid",
                "params": {},
            },
            WebSocketDisconnect(),
        ])
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.protocol_translator,
                "parse_client_message",
                return_value=mock_request
            ):
                with patch.object(
                    bridge_server.session_manager,
                    "send_request",
                    side_effect=RuntimeError("Tool not found")
                ):
                    with patch.object(
                        bridge_server.protocol_translator,
                        "create_error_response",
                        return_value=MCPResponse(
                            id="req-456",
                            error={"code": MCPErrorCode.METHOD_NOT_FOUND, "message": "Tool not found"}
                        )
                    ):
                        with patch.object(
                            bridge_server.session_manager,
                            "update_session_state"
                        ):
                            await bridge_server._handle_websocket(mock_ws)
        
        # Verify error response was sent
        calls = mock_ws.send_json.call_args_list
        error_response = None
        for call_args in calls:
            if call_args[0][0].get("type") == "error":
                error_response = call_args[0][0]
                break
        
        assert error_response is not None
        assert error_response["type"] == "error"


class TestSSEHandling:
    """Test Server-Sent Events handling."""
    
    @pytest.mark.asyncio
    async def test_sse_new_session(self, bridge_server: BridgeServer):
        """Test SSE with new session creation."""
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(side_effect=[False, True])
        
        mock_session = MCPSession(
            session_id="sse-session",
            capabilities={"tools": ["search"]},
        )
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=None
        ):
            with patch.object(
                bridge_server.session_manager,
                "create_session",
                return_value=mock_session
            ):
                with patch.object(
                    bridge_server.session_manager,
                    "update_session_state"
                ):
                    events = []
                    async for event in bridge_server._handle_sse("new-session", mock_request):
                        events.append(event)
                        if len(events) >= 2:  # Collect connected + heartbeat
                            break
        
        # Verify connected event
        assert "event: connected" in events[0]
        assert "sse-session" in events[0]
        
        # Verify heartbeat event
        if len(events) > 1:
            assert "event: heartbeat" in events[1]
    
    @pytest.mark.asyncio
    async def test_sse_existing_session(self, bridge_server: BridgeServer):
        """Test SSE with existing session."""
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(side_effect=[False, True])
        
        mock_session = MCPSession(
            session_id="existing-sse",
            capabilities={"tools": ["analyze"]},
        )
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "update_session_state"
            ):
                events = []
                async for event in bridge_server._handle_sse("existing-sse", mock_request):
                    events.append(event)
                    if len(events) >= 1:
                        break
        
        # Verify connected event with existing session
        assert "event: connected" in events[0]
        assert "existing-sse" in events[0]
    
    @pytest.mark.asyncio
    async def test_sse_heartbeat(self, bridge_server: BridgeServer):
        """Test SSE heartbeat mechanism."""
        mock_request = AsyncMock()
        disconnect_count = 0
        
        async def mock_is_disconnected():
            nonlocal disconnect_count
            disconnect_count += 1
            return disconnect_count > 2  # Disconnect after 2 heartbeats
        
        mock_request.is_disconnected = mock_is_disconnected
        
        mock_session = MCPSession(session_id="heartbeat-session")
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "update_session_state"
            ):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    events = []
                    async for event in bridge_server._handle_sse("heartbeat-session", mock_request):
                        events.append(event)
                        if len(events) >= 3:  # connected + 2 heartbeats
                            break
        
        # Count heartbeat events
        heartbeat_count = sum(1 for e in events if "event: heartbeat" in e)
        assert heartbeat_count >= 1


class TestLifecycleManagement:
    """Test server lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_startup(self, bridge_server: BridgeServer):
        """Test server startup sequence."""
        assert not bridge_server._started
        
        with patch.object(
            bridge_server.process_manager,
            "start",
            new_callable=AsyncMock
        ) as mock_pm_start:
            with patch.object(
                bridge_server.session_manager,
                "start",
                new_callable=AsyncMock
            ) as mock_sm_start:
                await bridge_server.startup()
        
        assert bridge_server._started
        mock_pm_start.assert_called_once()
        mock_sm_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_startup_idempotent(self, bridge_server: BridgeServer):
        """Test that startup is idempotent."""
        bridge_server._started = True
        
        with patch.object(
            bridge_server.process_manager,
            "start",
            new_callable=AsyncMock
        ) as mock_pm_start:
            await bridge_server.startup()
        
        # Should not call start again
        mock_pm_start.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, bridge_server: BridgeServer):
        """Test server shutdown sequence."""
        bridge_server._started = True
        
        with patch.object(
            bridge_server.session_manager,
            "stop",
            new_callable=AsyncMock
        ) as mock_sm_stop:
            with patch.object(
                bridge_server.process_manager,
                "stop",
                new_callable=AsyncMock
            ) as mock_pm_stop:
                await bridge_server.shutdown()
        
        assert not bridge_server._started
        mock_sm_stop.assert_called_once()
        mock_pm_stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, bridge_server: BridgeServer):
        """Test that shutdown is idempotent."""
        bridge_server._started = False
        
        with patch.object(
            bridge_server.session_manager,
            "stop",
            new_callable=AsyncMock
        ) as mock_sm_stop:
            await bridge_server.shutdown()
        
        # Should not call stop if not started
        mock_sm_stop.assert_not_called()


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_websocket_parse_error(self, bridge_server: BridgeServer):
        """Test WebSocket handling of message parse errors."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_json = AsyncMock()
        
        mock_session = MCPSession(session_id="error-session")
        
        mock_ws.receive_json = AsyncMock(side_effect=[
            {"type": "create_session"},
            {"invalid": "message"},  # Invalid format
            WebSocketDisconnect(),
        ])
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.protocol_translator,
                "parse_client_message",
                side_effect=ValueError("Invalid message format")
            ):
                with patch.object(
                    bridge_server.session_manager,
                    "update_session_state"
                ):
                    await bridge_server._handle_websocket(mock_ws)
        
        # Check error was sent
        error_sent = False
        for call_args in mock_ws.send_json.call_args_list:
            if call_args[0][0].get("type") == "error":
                error_sent = True
                assert "Invalid message format" in call_args[0][0]["error"]
                break
        
        assert error_sent
    
    @pytest.mark.asyncio
    async def test_websocket_unexpected_exception(self, bridge_server: BridgeServer):
        """Test WebSocket handling of unexpected exceptions."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.receive_json = AsyncMock(side_effect=Exception("Unexpected error"))
        
        with patch.object(
            bridge_server.session_manager,
            "update_session_state"
        ):
            # Should handle exception gracefully
            await bridge_server._handle_websocket(mock_ws)
        
        mock_ws.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_sse_cancellation(self, bridge_server: BridgeServer):
        """Test SSE handling of cancellation."""
        mock_request = AsyncMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)
        
        mock_session = MCPSession(session_id="cancel-session")
        
        with patch.object(
            bridge_server.session_manager,
            "get_session",
            return_value=mock_session
        ):
            with patch.object(
                bridge_server.session_manager,
                "update_session_state"
            ) as mock_update:
                with patch(
                    "asyncio.sleep",
                    side_effect=asyncio.CancelledError()
                ):
                    events = []
                    async for event in bridge_server._handle_sse("cancel-session", mock_request):
                        events.append(event)
                        if len(events) >= 1:  # Just get connected event
                            break
                    
                    # Should update session state on cancellation
                    mock_update.assert_called_with("cancel-session", SessionState.DISCONNECTED)


class TestIntegrationScenarios:
    """Test integration scenarios with multiple components."""
    
    @pytest.mark.asyncio
    async def test_full_request_response_cycle(self, test_client: TestClient, bridge_server: BridgeServer):
        """Test complete request-response cycle through HTTP."""
        # Setup mocks
        mock_session = MCPSession(
            session_id="full-cycle-session",
            state=SessionState.READY,
            capabilities={"tools": ["search", "analyze"]},
        )
        
        tool_discovery_result = {
            "tools": [
                {
                    "name": "search",
                    "description": "Search for information",
                    "parameters": {"query": "string"},
                },
            ]
        }
        
        search_result = {
            "results": [
                {"title": "Result 1", "content": "Content 1"},
                {"title": "Result 2", "content": "Content 2"},
            ]
        }
        
        with patch.object(
            bridge_server.session_manager,
            "create_session",
            return_value=mock_session
        ):
            # Create session
            response = test_client.post("/sessions", json={"client_id": "test-client"})
            assert response.status_code == 200
            session_data = response.json()
            session_id = session_data["session_id"]
            
            with patch.object(
                bridge_server.session_manager,
                "send_request"
            ) as mock_send:
                # Discover tools
                mock_send.return_value = tool_discovery_result
                response = test_client.post(
                    "/tools/discover",
                    json={"session_id": session_id}
                )
                assert response.status_code == 200
                tools_data = response.json()
                assert len(tools_data["tools"]) == 1
                
                # Call tool
                mock_send.return_value = search_result
                response = test_client.post(
                    "/tools/call",
                    json={
                        "session_id": session_id,
                        "tool": "search",
                        "params": {"query": "test query"},
                    }
                )
                assert response.status_code == 200
                result_data = response.json()
                assert len(result_data["result"]["results"]) == 2
            
            # Get session info
            with patch.object(
                bridge_server.session_manager,
                "get_session",
                return_value=mock_session
            ):
                response = test_client.get(f"/sessions/{session_id}")
                assert response.status_code == 200
            
            # Delete session
            with patch.object(
                bridge_server.session_manager,
                "remove_session"
            ):
                response = test_client.delete(f"/sessions/{session_id}")
                assert response.status_code == 200


class TestFactoryFunction:
    """Test the create_bridge_app factory function."""
    
    def test_create_bridge_app_default(self):
        """Test creating bridge app with default config."""
        app = create_bridge_app()
        
        assert app is not None
        assert app.title == "MCP Bridge Server"
        
        # Check routes are registered
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/sessions" in routes
    
    def test_create_bridge_app_custom_config(self, mock_config: BridgeConfig):
        """Test creating bridge app with custom config."""
        app = create_bridge_app(mock_config)
        
        assert app is not None
        
        # Verify config was used
        test_client = TestClient(app)
        response = test_client.get("/health")
        assert response.status_code == 200